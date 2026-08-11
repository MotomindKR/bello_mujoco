"""Deterministic held-out evaluation for strict Bello SONIC transfer."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import jax
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
    load_sonic_dataset_index,
    load_sonic_reference_bank,
    materialize_sonic_reference_clip,
    select_sonic_reference_clip,
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


@dataclass(frozen=True)
class SonicAcceptanceThresholds:
    minimum_completion_rate: float = 0.90
    maximum_root_position_rms_m: float = 0.20
    maximum_root_orientation_rms_deg: float = 15.0
    maximum_joint_position_rms_rad: float = 0.30
    maximum_foot_slip_mean_mps: float = 0.20

    def __post_init__(self) -> None:
        if not np.isfinite(self.minimum_completion_rate) or not (
            0.0 <= self.minimum_completion_rate <= 1.0
        ):
            raise ValueError("SONIC minimum completion rate must be in [0, 1]")
        for name in (
            "maximum_root_position_rms_m",
            "maximum_root_orientation_rms_deg",
            "maximum_joint_position_rms_rad",
            "maximum_foot_slip_mean_mps",
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
        key=lambda clip: (clip["shard_index"], clip["offset"]),
    )
    if not heldout:
        raise ValueError("SONIC evaluation dataset contains no held-out clips")
    current_shard_index = int(heldout[0]["shard_index"])
    current_bank = load_sonic_reference_bank(
        index_path,
        split="evaluation",
        shard_index=current_shard_index,
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

    clip_results = []
    video_env = None
    for clip_index, clip in enumerate(heldout):
        shard_index = int(clip["shard_index"])
        if shard_index != current_shard_index:
            current_bank = load_sonic_reference_bank(
                index_path,
                split="evaluation",
                shard_index=shard_index,
                sampling_bin_size=metadata.sonic_env.sampling.bin_size,
            )
            current_shard_index = shard_index
        env = (
            first_env
            if clip_index == 0
            else first_env._unchecked_runtime_copy(
                reference_bank=select_sonic_reference_clip(
                    current_bank,
                    _bank_clip_index(current_bank, clip),
                ),
                sampling_bin_probabilities=None,
            )
        )
        result = _evaluate_clip(
            env,
            train_state.policy,
            normalization.observation,
            metadata,
        )
        clip_results.append({**clip, **result})
        if video_env is None and config.video_path is not None:
            video_env = env

    clip_results.sort(key=lambda clip: int(clip["evaluation_order"]))

    video = None
    if video_env is not None and config.video_path is not None:
        video_path = config.video_path.expanduser().resolve()
        steps = min(
            config.video_max_steps,
            video_env.config.max_episode_steps,
            int(video_env.reference_bank.lengths[0]) - 1,
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
    )
    return BelloSonicEnv.load(
        metadata.asset,
        metadata.simulation,
        motion_path=runtime_path,
        reference_bank=select_sonic_reference_clip(bank, clip_index),
        config=evaluation_env,
        warp=metadata.warp,
        batch_size=1,
    )


def _policy_template(metadata, env, release):
    initialization = build_bello_decoder_initialization(
        release,
        metadata.dataset.calibration,
        bello_joint_names=env.action_layout.names,
        minimum_action_std=float(np.exp(metadata.ppo.policy.init_log_std)),
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


def _evaluate_clip(env, policy, observation_norm_state, metadata) -> dict[str, Any]:
    state, observation = env.reset()
    root_errors = []
    orientation_errors = []
    joint_errors = []
    foot_slips = []
    rewards = []
    reward_terms: dict[str, list[float]] = {}
    termination_counts: dict[str, int] = {}
    terminal = False
    timeout = False
    expected_steps = min(
        env.config.max_episode_steps,
        int(env.reference_bank.lengths[0]) - 1,
    )
    for _ in range(expected_steps):
        policy_observation = normalize_observation(
            observation, observation_norm_state, metadata.ppo
        )
        action = apply_actor(policy, policy_observation)
        state, observation, reward, done, info = env.step(state, action)
        rewards.append(float(reward))
        root_errors.append(float(info["tracking/root_position_error_m"]))
        orientation_errors.append(float(info["tracking/root_orientation_error_rad"]))
        joint_errors.append(float(info["tracking/joint_position_rms_rad"]))
        foot_slips.append(float(info["tracking/foot_slip_mean_mps"]))
        for name, value in info.items():
            if name.startswith("reward/"):
                reward_terms.setdefault(name, []).append(float(value))
            if name.startswith("done/") and bool(value):
                termination_counts[name] = termination_counts.get(name, 0) + 1
        if bool(done):
            terminal = bool(info["done/terminal"])
            timeout = bool(info["done/timeout"])
            break
    completed = len(rewards) >= expected_steps and not terminal
    return {
        "steps": len(rewards),
        "expected_steps": expected_steps,
        "completion_fraction": (
            float(len(rewards) / expected_steps) if expected_steps else 1.0
        ),
        "completed": completed,
        "episode_return": float(sum(rewards)),
        "root_position_rms_m": _rms(root_errors),
        "root_orientation_rms_deg": float(np.degrees(_rms(orientation_errors))),
        "joint_position_rms_rad": _rms(joint_errors),
        "foot_slip_mean_mps": float(np.mean(foot_slips)) if foot_slips else 0.0,
        "terminal": terminal,
        "timeout": timeout,
        "termination_counts": termination_counts,
        "reward_terms": {
            name: float(np.mean(values)) for name, values in reward_terms.items()
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
    return {
        "task": "sonic",
        "dataset_fingerprint": metadata.dataset.dataset_fingerprint,
        "robot_fingerprint": metadata.dataset.robot_fingerprint,
        "release_checkpoint_sha256": metadata.release_checkpoint_sha256,
        "release_artifact_sha256": metadata.release_artifact_sha256,
        "global": global_metrics,
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
        "per_clip": clips,
        "video": video,
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
        "shard_frame_capacity": metadata.dataset.shard_frame_capacity,
        "shard_clip_capacity": metadata.dataset.shard_clip_capacity,
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


def _rms(values) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array)))) if array.size else 0.0
