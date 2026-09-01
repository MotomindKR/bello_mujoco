"""Deterministic held-out evaluation for strict Bello SONIC transfer."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from bello_mujoco.envs.sonic import BelloSonicEnv
from bello_mujoco.evaluation.render import (
    CameraConfig,
    open_rollout_video_resources,
    update_camera,
)
from bello_mujoco.motion.sonic_dataset import (
    SonicReferenceBank,
    build_sonic_sampling_bins,
    load_sonic_dataset_index,
    load_sonic_reference_bank,
    materialize_sonic_reference_clip,
    select_sonic_reference_clip_in_bank,
)
from bello_mujoco.policy import (
    apply_actor,
    init_actor_critic,
    sonic_frozen_parameter_hashes,
)
from bello_mujoco.sonic_policy import (
    SONIC_OFFICIAL_CHECKPOINT_SHA256,
    build_bello_decoder_initialization,
    load_sonic_release_artifact,
)
from bello_mujoco.training.checkpoint import (
    SonicCheckpointMetadata,
    load_checkpoint_metadata,
    restore_evaluation_checkpoint,
)
from bello_mujoco.training.ppo import normalize_observation
from bello_mujoco.training.sonic import (
    _bind_reference_bank,
    _device_reference_bank_arrays,
)


_REFERENCE_FRAME_ARRAY_FIELDS = (
    "qpos",
    "qvel",
    "body_pos",
    "body_quat_wxyz",
    "body_lin_vel",
    "body_ang_vel",
    "foot_contacts",
    "smpl_joint_pos",
    "smpl_root_quat_wxyz",
    "smpl_root_translation",
    "g1_joint_pos",
)

_REWARD_TERM_NAMES = (
    "reward/tracking_anchor_position",
    "reward/tracking_anchor_orientation",
    "reward/tracking_relative_body_position",
    "reward/tracking_relative_body_orientation",
    "reward/tracking_body_linear_velocity",
    "reward/tracking_body_angular_velocity",
    "reward/tracking_joint_position",
    "reward/tracking_foot_contact",
    "reward/action_rate",
    "reward/joint_limit",
    "reward/target_joint_limit",
    "reward/undesired_contacts",
    "reward/anti_shake_angular_velocity",
    "reward/tracking_five_point_local",
    "reward/feet_acceleration",
)

_TERMINATION_NAMES = (
    "done/root_height",
    "done/orientation",
    "done/end_effector_height",
    "done/foot_position",
    "done/nonfinite",
    "done/timeout",
    "done/sonic_timeout",
    "done/terminal",
    "done/any",
    "done/height",
    "done/pose",
    "done/root_velocity",
    "done/torso_contact",
)


@dataclass(frozen=True)
class SonicAcceptanceThresholds:
    minimum_completion_rate: float = 0.90
    maximum_root_position_rms_m: float = 0.20
    maximum_root_orientation_rms_deg: float = 15.0
    maximum_joint_position_rms_rad: float = 0.30
    maximum_foot_slip_mean_mps: float = 0.20
    minimum_static_completion_fraction: float = 0.95
    maximum_static_root_xy_drift_m: float = 0.30

    def __post_init__(self) -> None:
        for name in (
            "minimum_completion_rate",
            "minimum_static_completion_fraction",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"SONIC {name} must be in [0, 1]")
        for name in (
            "maximum_root_position_rms_m",
            "maximum_root_orientation_rms_deg",
            "maximum_joint_position_rms_rad",
            "maximum_foot_slip_mean_mps",
            "maximum_static_root_xy_drift_m",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"SONIC {name} must be positive and finite")


@dataclass(frozen=True)
class SonicEvaluationConfig:
    output_path: Path = Path("outputs/evaluation/sonic.json")
    video_path: Path | None = None
    release_artifact_directory: Path = Path("artifacts/sonic/default_release")
    runtime_clip_directory: Path = Path("outputs/sonic_runtime_clips/evaluation")
    video_width: int = 640
    video_height: int = 480
    video_fps: int = 50
    video_max_steps: int = 500
    camera: CameraConfig = CameraConfig()
    thresholds: SonicAcceptanceThresholds = SonicAcceptanceThresholds()

    def __post_init__(self) -> None:
        for name in ("video_width", "video_height", "video_fps", "video_max_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"SONIC {name} must be a positive integer")


def evaluate_sonic_checkpoint(
    checkpoint_path: Path,
    config: SonicEvaluationConfig = SonicEvaluationConfig(),
    *,
    allow_git_commit_mismatch: bool = False,
) -> dict[str, Any]:
    metadata = load_checkpoint_metadata(
        checkpoint_path,
        allow_git_commit_mismatch=allow_git_commit_mismatch,
    )
    if not isinstance(metadata, SonicCheckpointMetadata):
        raise ValueError("SONIC evaluation requires a typed sonic checkpoint")
    index_path = metadata.dataset.index_path.expanduser().resolve()
    index = load_sonic_dataset_index(index_path)
    _validate_dataset_contract(index, metadata)
    release = load_sonic_release_artifact(config.release_artifact_directory)
    if release.checkpoint_sha256 != SONIC_OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError("SONIC evaluation requires NVIDIA's pinned release checkpoint")
    if (
        _prefixed_sha256(release.checkpoint_sha256)
        != (metadata.release_checkpoint_sha256)
        or _prefixed_sha256(release.artifact_sha256) != metadata.release_artifact_sha256
    ):
        raise ValueError("SONIC release artifact differs from the checkpoint contract")

    heldout = sorted(
        (clip for clip in index["clips"] if clip["split"] == "evaluation"),
        key=lambda clip: int(clip["evaluation_order"]),
    )
    if not heldout:
        raise ValueError("SONIC evaluation dataset contains no held-out clips")
    current_bank = load_sonic_reference_bank(
        index_path,
        split="evaluation",
        sampling_bin_size=metadata.sonic_env.sampling.bin_size,
    )
    if current_bank.calibration != metadata.dataset.calibration:
        raise ValueError("SONIC evaluation calibration differs from the checkpoint")

    first_env = _load_clip_env(
        metadata,
        config,
        current_bank,
        heldout[0],
        _bank_clip_index(current_bank, heldout[0]),
    )
    current_bank = _reference_bank_with_frame_arrays(
        current_bank, first_env.reference_bank
    )
    policy_template = _policy_template(metadata, first_env, release)
    expected_frozen_hashes = sonic_frozen_parameter_hashes(policy_template)
    train_state, restored_metadata, normalization = restore_evaluation_checkpoint(
        checkpoint_path,
        allow_git_commit_mismatch=allow_git_commit_mismatch,
        policy=policy_template,
    )
    if restored_metadata != metadata:
        raise RuntimeError("SONIC checkpoint metadata changed between integrity reads")
    actual_frozen_hashes = sonic_frozen_parameter_hashes(train_state.policy)
    if actual_frozen_hashes != expected_frozen_hashes:
        changed = sorted(
            name
            for name in set(actual_frozen_hashes) | set(expected_frozen_hashes)
            if actual_frozen_hashes.get(name) != expected_frozen_hashes.get(name)
        )
        raise ValueError(
            "checkpoint changed frozen SONIC parameters: " + ", ".join(changed)
        )
    del policy_template, release

    evaluate_clip = _build_compiled_clip_evaluator(
        first_env,
        train_state.policy,
        normalization.observation,
        metadata,
    )
    evaluate_reference_oracle = _build_compiled_clip_evaluator(
        first_env,
        train_state.policy,
        normalization.observation,
        metadata,
        reference_action_oracle=True,
    )
    static_bank, static_source = _stationary_static_evaluation_bank(
        current_bank,
        heldout,
        frames=metadata.sonic_env.max_episode_steps + 1,
    )
    static_env = replace(
        first_env,
        reference_bank=_device_reference_bank(static_bank),
    )
    evaluate_static = _build_compiled_clip_evaluator(
        static_env,
        train_state.policy,
        normalization.observation,
        metadata,
    )

    clip_results = []
    oracle_clip_results = []
    video_env = first_env if config.video_path is not None else None
    for clip_index, clip in enumerate(heldout):
        selected_bank = select_sonic_reference_clip_in_bank(
            current_bank,
            _bank_clip_index(current_bank, clip),
        )
        arrays = _device_reference_bank_arrays(selected_bank)
        accumulators = evaluate_clip(arrays)
        oracle_accumulators = evaluate_reference_oracle(arrays)
        clip_results.append(
            {
                **clip,
                **_compiled_clip_result(accumulators),
            }
        )
        oracle_clip_results.append(
            {
                **clip,
                **_compiled_clip_result(oracle_accumulators),
            }
        )
        completed = clip_index + 1
        if completed % 100 == 0 or completed == len(heldout):
            print(
                f"SONIC held-out evaluation: {completed}/{len(heldout)} clips",
                file=sys.stderr,
                flush=True,
            )

    clip_results.sort(key=lambda clip: int(clip["evaluation_order"]))
    oracle_clip_results.sort(key=lambda clip: int(clip["evaluation_order"]))
    static_result = {
        **static_source,
        **_compiled_clip_result(
            evaluate_static(_device_reference_bank_arrays(static_bank))
        ),
    }

    video = None
    if video_env is not None and config.video_path is not None:
        video_path = config.video_path.expanduser().resolve()
        steps = min(
            config.video_max_steps,
            video_env.config.max_episode_steps,
            int(
                video_env.reference_bank.lengths[
                    _bank_clip_index(current_bank, heldout[0])
                ]
            )
            - 1,
        )
        _render_rollout(
            video_env,
            train_state.policy,
            normalization.observation,
            metadata,
            video_path,
            steps=max(1, steps),
            config=config,
        )
        video = {"path": str(video_path), "steps": max(1, steps)}
    report = _aggregate_results(metadata, clip_results, config.thresholds, video)
    static_checks = {
        "completion_fraction": (
            static_result["completion_fraction"]
            >= config.thresholds.minimum_static_completion_fraction
        ),
        "root_xy_drift_m": (
            static_result["root_xy_drift_m"]
            <= config.thresholds.maximum_static_root_xy_drift_m
        ),
    }
    report["static_reference"] = {
        "result": static_result,
        "acceptance": {
            "passed": all(static_checks.values()),
            "checks": static_checks,
            "thresholds": {
                "minimum_completion_fraction": (
                    config.thresholds.minimum_static_completion_fraction
                ),
                "maximum_root_xy_drift_m": (
                    config.thresholds.maximum_static_root_xy_drift_m
                ),
            },
        },
    }
    report["acceptance"]["checks"]["static_reference"] = all(
        static_checks.values()
    )
    report["acceptance"]["passed"] = all(
        report["acceptance"]["checks"].values()
    )
    oracle_report = _aggregate_results(
        metadata, oracle_clip_results, config.thresholds, None
    )
    report["reference_action_oracle"] = {
        name: oracle_report[name]
        for name in (
            "global",
            "frame_weighted",
            "acceptance",
            "per_source",
            "per_semantic_label",
            "per_clip",
        )
    }
    output_path = config.output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _load_clip_env(
    metadata: SonicCheckpointMetadata,
    config: SonicEvaluationConfig,
    bank: SonicReferenceBank,
    clip: dict[str, Any],
    clip_index: int,
) -> BelloSonicEnv:
    runtime_path = (
        config.runtime_clip_directory
        / metadata.dataset.dataset_fingerprint.removeprefix("sha256:")
        / f"{clip['source_id']}__{clip['sequence_id']}.npz"
    )
    materialize_sonic_reference_clip(
        bank,
        runtime_path,
        clip_index=clip_index,
        source_sha256=clip["source_sha256"],
    )
    evaluation_env = replace(
        metadata.sonic_env,
        observation_noise=replace(metadata.sonic_env.observation_noise, enabled=False),
        domain_randomization=replace(
            metadata.sonic_env.domain_randomization, enabled=False
        ),
        sampling=replace(metadata.sonic_env.sampling, enabled=False),
        freeze_frame_probability=0.0,
    )
    return BelloSonicEnv.load(
        metadata.asset,
        metadata.simulation,
        motion_path=runtime_path,
        reference_bank=_device_reference_bank(
            select_sonic_reference_clip_in_bank(bank, clip_index)
        ),
        config=evaluation_env,
        warp=metadata.warp,
        batch_size=1,
    )


def _device_reference_bank(bank: SonicReferenceBank) -> SonicReferenceBank:
    return _reference_bank_with_frame_arrays(bank, bank, device_put=True)


def _stationary_static_evaluation_bank(
    bank: SonicReferenceBank,
    clips: list[dict[str, Any]],
    *,
    frames: int,
) -> tuple[SonicReferenceBank, dict[str, Any]]:
    ankle_indices = np.asarray(
        [
            bank.body_names.index("left_ankle_roll_link"),
            bank.body_names.index("right_ankle_roll_link"),
        ],
        dtype=np.int32,
    )
    candidates = []
    for clip in clips:
        labels = set(clip.get("semantic_labels", ()))
        if not {"stationary_root", "non_gait_motion"}.issubset(labels):
            continue
        clip_index = _bank_clip_index(bank, clip)
        offset = int(bank.offsets[clip_index])
        length = int(bank.lengths[clip_index])
        clip_slice = slice(offset, offset + length)
        ankle_height = np.asarray(bank.body_pos)[clip_slice][:, ankle_indices, 2]
        near_support = np.all(
            ankle_height - np.min(ankle_height, axis=0, keepdims=True) <= 0.03,
            axis=-1,
        )
        if not np.any(near_support):
            continue
        velocity = np.asarray(bank.qvel)[clip_slice]
        velocity_score = np.linalg.norm(velocity, axis=-1)
        frame = int(np.argmin(np.where(near_support, velocity_score, np.inf)))
        candidates.append((float(velocity_score[frame]), clip_index, frame, clip))
    if not candidates:
        raise ValueError(
            "SONIC evaluation dataset has no kinematically supported stationary "
            "reference"
        )
    _, clip_index, frame, clip = min(
        candidates,
        key=lambda item: (item[0], item[3]["source_id"], item[3]["sequence_id"]),
    )
    absolute = int(bank.offsets[clip_index]) + frame

    def repeated(name: str) -> np.ndarray:
        value = np.asarray(getattr(bank, name))[absolute : absolute + 1]
        return np.repeat(value, frames, axis=0)

    lengths = np.asarray((frames,), dtype=np.int32)
    bins = build_sonic_sampling_bins(lengths, bin_size=bank.sampling_bin_size)
    static = replace(
        bank,
        qpos=repeated("qpos"),
        qvel=np.zeros_like(repeated("qvel")),
        body_pos=repeated("body_pos"),
        body_quat_wxyz=repeated("body_quat_wxyz"),
        body_lin_vel=np.zeros_like(repeated("body_lin_vel")),
        body_ang_vel=np.zeros_like(repeated("body_ang_vel")),
        foot_contacts=np.ones((frames, 2), dtype=np.float32),
        smpl_joint_pos=repeated("smpl_joint_pos"),
        smpl_root_quat_wxyz=repeated("smpl_root_quat_wxyz"),
        smpl_root_translation=repeated("smpl_root_translation"),
        g1_joint_pos=repeated("g1_joint_pos"),
        offsets=np.asarray((0,), dtype=np.int32),
        lengths=lengths,
        sampling_probabilities=np.asarray((1.0,), dtype=np.float32),
        sampling_bin_clip_indices=bins[0],
        sampling_bin_starts=bins[1],
        sampling_bin_widths=bins[2],
        sampling_bin_valid=bins[3],
        sampling_clip_bin_offsets=bins[4],
        valid_clip_count=1,
        source_ids=(clip["source_id"],),
        sequence_ids=(clip["sequence_id"],),
    )
    return static, {
        "source_id": clip["source_id"],
        "sequence_id": clip["sequence_id"],
        "source_frame": frame,
        "frames": frames,
    }


def _reference_bank_with_frame_arrays(
    bank: SonicReferenceBank,
    source: SonicReferenceBank,
    *,
    device_put: bool = False,
) -> SonicReferenceBank:
    return replace(
        bank,
        **{
            name: (
                jax.device_put(getattr(source, name))
                if device_put
                else getattr(source, name)
            )
            for name in _REFERENCE_FRAME_ARRAY_FIELDS
        },
    )


def _policy_template(metadata, env, release):
    initialization = build_bello_decoder_initialization(
        release,
        metadata.dataset.calibration,
        bello_joint_names=env.action_layout.names,
        bello_action_scale=env.action_scale,
        bello_default_joint_position=np.asarray(env.model.default_joint_target)[
            np.asarray(env.model.policy_ctrl_indices)
        ],
    )
    _, policy_key = jax.random.split(jax.random.PRNGKey(metadata.ppo.seed))
    return init_actor_critic(
        policy_key,
        metadata.obs_size,
        metadata.action_size,
        metadata.ppo.policy,
        critic_obs_size=metadata.critic_obs_size,
        sonic_release=release,
        sonic_initialization=initialization,
    )


def _bank_clip_index(bank: SonicReferenceBank, clip: dict[str, Any]) -> int:
    identity = (clip["source_id"], clip["sequence_id"])
    matches = [
        index
        for index, candidate in enumerate(
            zip(bank.source_ids, bank.sequence_ids, strict=True)
        )
        if candidate == identity
    ]
    if len(matches) != 1:
        raise ValueError(
            f"SONIC reference bank does not contain exactly one {identity}"
        )
    return matches[0]


def _build_compiled_clip_evaluator(
    env: BelloSonicEnv,
    policy,
    observation_norm_state,
    metadata: SonicCheckpointMetadata,
    *,
    reference_action_oracle: bool = False,
):
    reference_bank = env.reference_bank
    if reference_bank is None:
        raise ValueError("SONIC compiled evaluation requires a reference bank")
    static_env = env._unchecked_runtime_copy(reference_bank=None)
    bank_metadata = (
        reference_bank.source_ids,
        reference_bank.sequence_ids,
        reference_bank.joint_names,
        reference_bank.qpos_joint_names,
        reference_bank.body_names,
        reference_bank.calibration,
        reference_bank.sampling_bin_size,
    )
    maximum_steps = env.config.max_episode_steps

    def evaluate(arrays):
        current_env = _bind_reference_bank(static_env, arrays, bank_metadata)
        state, observation = current_env.reset()
        clip_index = getattr(state, "clip_index", jnp.asarray(0, dtype=jnp.int32))
        expected_steps = jnp.minimum(
            jnp.asarray(maximum_steps, dtype=jnp.int32),
            jnp.asarray(current_env.reference_bank.lengths)[clip_index] - 1,
        )
        totals = {
            "steps": jnp.asarray(0, dtype=jnp.int32),
            "episode_return": jnp.asarray(0.0, dtype=jnp.float32),
            "root_position_square_sum": jnp.asarray(0.0, dtype=jnp.float32),
            "root_orientation_square_sum": jnp.asarray(0.0, dtype=jnp.float32),
            "joint_position_square_sum": jnp.asarray(0.0, dtype=jnp.float32),
            "foot_slip_sum": jnp.asarray(0.0, dtype=jnp.float32),
            "root_xy_drift_m": jnp.asarray(0.0, dtype=jnp.float32),
            "terminal": jnp.asarray(False),
            "timeout": jnp.asarray(False),
            "termination_counts": {
                name: jnp.asarray(0, dtype=jnp.int32) for name in _TERMINATION_NAMES
            },
            "reward_sums": {
                name: jnp.asarray(0.0, dtype=jnp.float32) for name in _REWARD_TERM_NAMES
            },
        }

        def loop_body(_, carry):
            current_state, current_observation, active, current_totals = carry
            should_step = active & (current_totals["steps"] < expected_steps)

            def take_step(operand):
                step_state, step_observation, step_totals = operand
                if reference_action_oracle:
                    action = current_env.reference_action(step_state)
                else:
                    policy_observation = normalize_observation(
                        step_observation,
                        observation_norm_state,
                        metadata.ppo,
                    )
                    action = apply_actor(policy, policy_observation)
                next_state, next_observation, reward, done, info = current_env.step(
                    step_state, action
                )
                updated = {
                    **step_totals,
                    "steps": step_totals["steps"] + 1,
                    "episode_return": step_totals["episode_return"] + reward,
                    "root_position_square_sum": (
                        step_totals["root_position_square_sum"]
                        + info["tracking/root_position_error_m"] ** 2
                    ),
                    "root_orientation_square_sum": (
                        step_totals["root_orientation_square_sum"]
                        + info["tracking/root_orientation_error_rad"] ** 2
                    ),
                    "joint_position_square_sum": (
                        step_totals["joint_position_square_sum"]
                        + info["tracking/joint_position_rms_rad"] ** 2
                    ),
                    "foot_slip_sum": (
                        step_totals["foot_slip_sum"]
                        + info["tracking/foot_slip_mean_mps"]
                    ),
                    "root_xy_drift_m": jnp.maximum(
                        step_totals["root_xy_drift_m"],
                        info["tracking/root_xy_error_m"],
                    ),
                    "terminal": (
                        step_totals["terminal"] | info["done/terminal"].astype(bool)
                    ),
                    "timeout": (
                        step_totals["timeout"] | info["done/timeout"].astype(bool)
                    ),
                    "termination_counts": {
                        name: step_totals["termination_counts"][name]
                        + info[name].astype(jnp.int32)
                        for name in _TERMINATION_NAMES
                    },
                    "reward_sums": {
                        name: step_totals["reward_sums"][name] + info[name]
                        for name in _REWARD_TERM_NAMES
                    },
                }
                return next_state, next_observation, ~done.astype(bool), updated

            def stop(operand):
                stopped_state, stopped_observation, stopped_totals = operand
                return (
                    stopped_state,
                    stopped_observation,
                    jnp.asarray(False),
                    stopped_totals,
                )

            return jax.lax.cond(
                should_step,
                take_step,
                stop,
                (current_state, current_observation, current_totals),
            )

        _, _, _, totals = jax.lax.fori_loop(
            0,
            maximum_steps,
            loop_body,
            (state, observation, jnp.asarray(True), totals),
        )
        return {**totals, "expected_steps": expected_steps}

    return jax.jit(evaluate)


def _compiled_clip_result(accumulators) -> dict[str, Any]:
    values = jax.device_get(accumulators)
    steps = int(values["steps"])
    expected_steps = int(values["expected_steps"])
    terminal = bool(values["terminal"])
    denominator = max(steps, 1)

    def rms(name: str) -> float:
        return float(np.sqrt(float(values[name]) / denominator))

    termination_counts = {
        name: int(value)
        for name, value in values["termination_counts"].items()
        if int(value)
    }
    return {
        "steps": steps,
        "expected_steps": expected_steps,
        "completion_fraction": (
            float(steps / expected_steps) if expected_steps else 1.0
        ),
        "completed": steps >= expected_steps and not terminal,
        "episode_return": float(values["episode_return"]),
        "root_position_rms_m": rms("root_position_square_sum"),
        "root_orientation_rms_deg": float(
            np.degrees(rms("root_orientation_square_sum"))
        ),
        "joint_position_rms_rad": rms("joint_position_square_sum"),
        "foot_slip_mean_mps": float(values["foot_slip_sum"]) / denominator,
        "root_xy_drift_m": float(values["root_xy_drift_m"]),
        "terminal": terminal,
        "timeout": bool(values["timeout"]),
        "termination_counts": termination_counts,
        "reward_terms": {
            name: float(value) / denominator
            for name, value in values["reward_sums"].items()
        },
    }


def _render_rollout(
    env,
    policy,
    observation_norm_state,
    metadata,
    output_path: Path,
    *,
    steps: int,
    config: SonicEvaluationConfig,
) -> None:
    state, observation = env.reset()
    mj_model = env.model.mj_model
    mj_data = mujoco.MjData(mj_model)
    resources, renderer, camera, writer = open_rollout_video_resources(
        mj_model,
        output_path,
        width=config.video_width,
        height=config.video_height,
        fps=config.video_fps,
        camera_config=config.camera,
    )
    try:
        for _ in range(steps):
            mj_data.qpos[:] = np.asarray(
                jax.device_get(state.motion_state.data.qpos), dtype=np.float64
            )
            mj_data.qvel[:] = np.asarray(
                jax.device_get(state.motion_state.data.qvel), dtype=np.float64
            )
            mujoco.mj_forward(mj_model, mj_data)
            update_camera(camera, mj_model, mj_data, config.camera)
            renderer.update_scene(mj_data, camera=camera)
            writer.append_data(renderer.render())
            policy_observation = normalize_observation(
                observation, observation_norm_state, metadata.ppo
            )
            action = apply_actor(policy, policy_observation)
            state, observation, _, done, _ = env.step(state, action)
            if bool(done):
                break
    finally:
        resources.close()


def _aggregate_results(metadata, clips, thresholds, video) -> dict[str, Any]:
    metric_names = (
        "episode_return",
        "root_position_rms_m",
        "root_orientation_rms_deg",
        "joint_position_rms_rad",
        "foot_slip_mean_mps",
    )
    global_metrics = {
        name: float(np.mean([clip[name] for clip in clips])) for name in metric_names
    }
    global_metrics["completion_rate"] = float(
        np.mean([clip["completed"] for clip in clips])
    )
    frame_weighted = _frame_weighted_metrics(clips)
    checks = {
        "completion_rate": (
            global_metrics["completion_rate"] >= thresholds.minimum_completion_rate
        ),
        "root_position_rms_m": (
            global_metrics["root_position_rms_m"]
            <= thresholds.maximum_root_position_rms_m
        ),
        "root_orientation_rms_deg": (
            global_metrics["root_orientation_rms_deg"]
            <= thresholds.maximum_root_orientation_rms_deg
        ),
        "joint_position_rms_rad": (
            global_metrics["joint_position_rms_rad"]
            <= thresholds.maximum_joint_position_rms_rad
        ),
        "foot_slip_mean_mps": (
            global_metrics["foot_slip_mean_mps"]
            <= thresholds.maximum_foot_slip_mean_mps
        ),
    }
    sources = {}
    for source_id in sorted({clip["source_id"] for clip in clips}):
        selected = [clip for clip in clips if clip["source_id"] == source_id]
        sources[source_id] = {
            "clips": len(selected),
            "completion_rate": float(np.mean([clip["completed"] for clip in selected])),
            **{
                name: float(np.mean([clip[name] for clip in selected]))
                for name in metric_names
            },
        }
    semantic_labels = sorted(
        {label for clip in clips for label in clip.get("semantic_labels", ())}
    )
    semantic_strata = {}
    for label in semantic_labels:
        selected = [clip for clip in clips if label in clip["semantic_labels"]]
        semantic_strata[label] = {
            "clips": len(selected),
            "clip_weighted": {
                "completion_rate": float(
                    np.mean([clip["completed"] for clip in selected])
                ),
                **{
                    name: float(np.mean([clip[name] for clip in selected]))
                    for name in metric_names
                },
            },
            "frame_weighted": _frame_weighted_metrics(selected),
        }
    return {
        "task": "sonic",
        "dataset_fingerprint": metadata.dataset.dataset_fingerprint,
        "robot_fingerprint": metadata.dataset.robot_fingerprint,
        "release_checkpoint_sha256": metadata.release_checkpoint_sha256,
        "release_artifact_sha256": metadata.release_artifact_sha256,
        "global": global_metrics,
        "frame_weighted": frame_weighted,
        "acceptance": {
            "passed": all(checks.values()),
            "checks": checks,
            "thresholds": {
                "minimum_completion_rate": thresholds.minimum_completion_rate,
                "maximum_root_position_rms_m": (thresholds.maximum_root_position_rms_m),
                "maximum_root_orientation_rms_deg": (
                    thresholds.maximum_root_orientation_rms_deg
                ),
                "maximum_joint_position_rms_rad": (
                    thresholds.maximum_joint_position_rms_rad
                ),
                "maximum_foot_slip_mean_mps": (thresholds.maximum_foot_slip_mean_mps),
            },
        },
        "per_source": sources,
        "per_semantic_label": semantic_strata,
        "per_clip": clips,
        "video": video,
    }


def _frame_weighted_metrics(clips: list[dict[str, Any]]) -> dict[str, float]:
    steps = np.asarray([clip["steps"] for clip in clips], dtype=np.float64)
    expected = np.asarray([clip["expected_steps"] for clip in clips], dtype=np.float64)
    total_steps = max(float(np.sum(steps)), 1.0)

    def weighted_rms(name: str) -> float:
        values = np.asarray([clip[name] for clip in clips], dtype=np.float64)
        return float(np.sqrt(np.sum(values * values * steps) / total_steps))

    foot_slip = np.asarray(
        [clip["foot_slip_mean_mps"] for clip in clips], dtype=np.float64
    )
    returns = np.asarray([clip["episode_return"] for clip in clips], dtype=np.float64)
    return {
        "completion_fraction": float(np.sum(steps) / max(float(np.sum(expected)), 1.0)),
        "episode_return_per_step": float(np.sum(returns) / total_steps),
        "root_position_rms_m": weighted_rms("root_position_rms_m"),
        "root_orientation_rms_deg": weighted_rms("root_orientation_rms_deg"),
        "joint_position_rms_rad": weighted_rms("joint_position_rms_rad"),
        "foot_slip_mean_mps": float(np.sum(foot_slip * steps) / total_steps),
    }


def _validate_dataset_contract(index: dict[str, Any], metadata) -> None:
    expected = {
        "schema_version": metadata.dataset.schema_version,
        "dataset_fingerprint": metadata.dataset.dataset_fingerprint,
        "robot_fingerprint": metadata.dataset.robot_fingerprint,
        "gmr_revision": metadata.dataset.bello_gmr_revision,
        "bello_gmr_config_sha256": metadata.dataset.bello_gmr_config_sha256,
        "g1_gmr_revision": metadata.dataset.g1_gmr_revision,
        "g1_gmr_config_sha256": metadata.dataset.g1_config_sha256,
        "g1_robot_sha256": metadata.dataset.g1_robot_sha256,
        "joint_names": list(metadata.dataset.joint_names),
        "body_names": list(metadata.dataset.body_names),
        "joint_calibration": json.loads(
            json.dumps(metadata.dataset.calibration.to_document())
        ),
    }
    mismatches = sorted(
        name
        for name, value in expected.items()
        if _normalized_contract_value(name, index.get(name)) != value
    )
    if mismatches:
        raise ValueError(
            "SONIC evaluation dataset differs from checkpoint fields: "
            + ", ".join(mismatches)
        )


def _normalized_contract_value(name: str, value: Any) -> Any:
    if name in {
        "robot_fingerprint",
        "bello_gmr_config_sha256",
        "g1_gmr_config_sha256",
        "g1_robot_sha256",
    }:
        return _prefixed_sha256(value) if isinstance(value, str) else value
    return value


def _prefixed_sha256(value: str) -> str:
    return "sha256:" + value.removeprefix("sha256:")
