"""SONIC transfer learning on synchronized Bello AMASS data."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import jax
import numpy as np

from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.envs.sonic import (
    BelloSonicEnv,
    SonicEnvConfig,
    default_sonic_asset_config,
)
from bello_mujoco.envs.twist import TwistSimulationConfig
from bello_mujoco.motion.sonic_dataset import (
    SonicReferenceBank,
    load_sonic_dataset_index,
    load_sonic_reference_bank,
    materialize_sonic_reference_clip,
)
from bello_mujoco.policy import (
    init_actor_critic,
    sonic_frozen_parameter_hashes,
)
from bello_mujoco.repository import (
    capture_repository_commit,
    require_checkpoint_output_directory,
)
from bello_mujoco.sim import WarpConfig
from bello_mujoco.sonic_policy import (
    SONIC_OFFICIAL_CHECKPOINT_SHA256,
    build_bello_decoder_initialization,
    load_sonic_release_artifact,
)
from bello_mujoco.training.checkpoint import (
    CheckpointConfig,
    SonicCheckpointMetadata,
    SonicDatasetContract,
    checkpoint_path,
    restore_runner_checkpoint,
    save_training_checkpoint,
    sonic_metadata_with_runner_state,
    validate_checkpoint_options,
)
from bello_mujoco.training.ppo import (
    PPOConfig,
    init_runner_state,
    runner_train_state,
    update_runner_state,
    validate_action_limit_contract,
    validate_ppo_config,
)
from bello_mujoco.training.sonic_pretraining import (
    SonicOfflineBatchBuilder,
    SonicPretrainingConfig,
    pretrain_sonic_adapters,
)
from bello_mujoco.training.tracking import (
    WandbConfig,
    close_run,
    create_wandb_run,
    log_config,
    track_checkpoint_artifact,
    track_metric_snapshot,
)

_SONIC_BANK_ARRAY_FIELDS = (
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
    "offsets",
    "lengths",
    "sampling_probabilities",
    "sampling_bin_clip_indices",
    "sampling_bin_starts",
    "sampling_bin_widths",
    "sampling_bin_valid",
    "sampling_clip_bin_offsets",
    "valid_clip_count",
)


@dataclass(frozen=True)
class SonicTrainingRunConfig:
    dataset_index: Path = Path("datasets/sonic/index.json")
    release_artifact_directory: Path = Path("artifacts/sonic/default_release")
    config: PPOConfig = PPOConfig()
    asset_config: AssetPrepConfig = field(default_factory=default_sonic_asset_config)
    simulation: TwistSimulationConfig = TwistSimulationConfig()
    sonic_env: SonicEnvConfig = SonicEnvConfig()
    pretraining: SonicPretrainingConfig = SonicPretrainingConfig()
    checkpoint_config: CheckpointConfig = CheckpointConfig()
    wandb_config: WandbConfig = WandbConfig()
    warp: WarpConfig = WarpConfig()
    runtime_clip_directory: Path = Path("outputs/sonic_runtime_clips")


def run_training(run_config: SonicTrainingRunConfig) -> None:
    validate_sonic_training_config(run_config)
    index_path = run_config.dataset_index.expanduser().resolve()
    index = load_sonic_dataset_index(index_path)
    reference_bank = load_sonic_reference_bank(
        index_path,
        split="train",
        sampling_bin_size=run_config.sonic_env.sampling.bin_size,
    )
    runtime_clip = (
        run_config.runtime_clip_directory
        / index["dataset_fingerprint"].removeprefix("sha256:")
        / "train.npz"
    )
    train_clips = [clip for clip in index["clips"] if clip["split"] == "train"]
    if not train_clips:
        raise ValueError("SONIC packed dataset has no training clips")
    materialize_sonic_reference_clip(
        reference_bank,
        runtime_clip,
        clip_index=0,
        source_sha256=train_clips[0]["source_sha256"],
        policy_fps=index["policy_fps"],
    )
    env = BelloSonicEnv.load(
        run_config.asset_config,
        run_config.simulation,
        motion_path=runtime_clip,
        reference_bank=reference_bank,
        config=run_config.sonic_env,
        warp=run_config.warp,
        batch_size=run_config.config.num_envs,
    )
    validate_action_limit_contract(run_config.config, env.action_clip)
    release = load_sonic_release_artifact(run_config.release_artifact_directory)
    if release.checkpoint_sha256 != SONIC_OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError("SONIC training requires NVIDIA's pinned release checkpoint")
    initialization = build_bello_decoder_initialization(
        release,
        reference_bank.calibration,
        bello_joint_names=env.action_layout.names,
        bello_action_scale=env.action_scale,
        bello_default_joint_position=np.asarray(env.model.default_joint_target)[
            np.asarray(env.model.policy_ctrl_indices)
        ],
        initial_action_std=float(np.exp(run_config.config.policy.init_log_std)),
    )
    rng = jax.random.PRNGKey(run_config.config.seed)
    rng, policy_key = jax.random.split(rng)
    policy_template = init_actor_critic(
        policy_key,
        env.observation_size,
        env.action_size,
        run_config.config.policy,
        critic_obs_size=env.critic_observation_size,
        sonic_release=release,
        sonic_initialization=initialization,
    )
    pretraining_records: tuple[tuple[int, dict[str, float]], ...] = ()
    if (
        run_config.pretraining.enabled
        and run_config.checkpoint_config.resume_path is None
    ):
        policy_indices = np.asarray(env.model.policy_ctrl_indices)
        default = np.asarray(env.model.default_joint_target)[policy_indices]
        scale = np.asarray(env.action_scale, dtype=np.float32)
        builder = SonicOfflineBatchBuilder(
            reference_bank,
            bello_default_joint_position=default,
            bello_action_scale=scale,
            validation_clip_fraction=(run_config.pretraining.validation_clip_fraction),
            seed=run_config.pretraining.seed,
        )

        def report_pretraining(update: int, metrics: dict[str, float]) -> None:
            print({"pretraining_update": update, **metrics})

        policy_template, pretraining_records = pretrain_sonic_adapters(
            policy_template,
            release,
            builder,
            run_config.pretraining,
            action_low=(
                np.asarray(env.model.joint_target_low)[policy_indices] - default
            )
            / scale,
            action_high=(
                np.asarray(env.model.joint_target_high)[policy_indices] - default
            )
            / scale,
            progress=report_pretraining,
        )
    frozen_hashes = sonic_frozen_parameter_hashes(policy_template)
    git_commit = capture_repository_commit()
    metadata = _checkpoint_metadata(run_config, env, index, release, git_commit)
    checkpoint_config = replace(
        run_config.checkpoint_config,
        directory=require_checkpoint_output_directory(
            run_config.checkpoint_config.directory
        ),
    )
    root = (
        checkpoint_path(checkpoint_config, run_config.config)
        if checkpoint_config.enabled
        else None
    )
    if checkpoint_config.resume_path is None:
        runner, static_policy, optimizer = init_runner_state(
            rng, env, run_config.config, policy=policy_template
        )
    else:
        runner, static_policy, optimizer, _ = restore_runner_checkpoint(
            checkpoint_config.resume_path,
            env,
            metadata,
            allow_git_commit_mismatch=checkpoint_config.allow_git_commit_mismatch,
            policy=policy_template,
        )
    _require_frozen_hashes(runner, static_policy, frozen_hashes)
    first_update = int(jax.device_get(runner.update))
    static_env = env._unchecked_runtime_copy(reference_bank=None)
    bank_arrays = _device_reference_bank_arrays(reference_bank)
    bank_metadata = (
        reference_bank.source_ids,
        reference_bank.sequence_ids,
        reference_bank.joint_names,
        reference_bank.qpos_joint_names,
        reference_bank.body_names,
        reference_bank.calibration,
        reference_bank.sampling_bin_size,
    )
    del env, reference_bank, policy_template
    jax.clear_caches()
    run = None
    succeeded = False
    try:
        run = create_wandb_run(run_config.wandb_config)
        log_config(
            run,
            {
                "task": "sonic",
                "ppo": run_config.config,
                "dataset": _tracking_dataset_summary(index),
                "simulation": run_config.simulation,
                "sonic_env": run_config.sonic_env,
                "pretraining": run_config.pretraining,
                "reference_bank_frames": int(bank_arrays[0].shape[0]),
                "reference_bank_clips": int(bank_arrays[11].shape[0]),
                "release_checkpoint_sha256": release.checkpoint_sha256,
                "release_artifact_sha256": release.artifact_sha256,
                "frozen_parameter_sha256": frozen_hashes,
                "checkpoint": checkpoint_config,
                "git_commit": git_commit,
            },
        )
        for pretraining_update, metrics in pretraining_records:
            track_metric_snapshot(
                run,
                metrics,
                env_steps=pretraining_update,
                update=pretraining_update,
                context={"phase": "pretrain", "stage": "sonic_adapter"},
            )
        last_metrics = {}

        def update_fn(state, arrays):
            current_env = _bind_reference_bank(static_env, arrays, bank_metadata)
            return update_runner_state(
                state,
                static_policy,
                optimizer,
                current_env,
                run_config.config,
            )

        update_fn = jax.jit(update_fn, donate_argnums=(0,))
        for update in range(first_update, run_config.config.num_updates):
            runner, last_metrics = update_fn(runner, bank_arrays)
            last_metrics = jax.tree_util.tree_map(
                lambda value: value.block_until_ready(), last_metrics
            )
            track_metric_snapshot(
                run,
                last_metrics,
                env_steps=(update + 1)
                * run_config.config.num_envs
                * run_config.config.num_steps,
                update=update + 1,
                context={"phase": "train", "stage": "sonic_edge_transfer"},
            )
            if (
                root is not None
                and checkpoint_config.every_updates is not None
                and (update + 1) % checkpoint_config.every_updates == 0
            ):
                _require_frozen_hashes(runner, static_policy, frozen_hashes)
                path_at_update = root / f"update_{update + 1:06d}"
                save_sonic_checkpoint(path_at_update, runner, metadata, git_commit)
                upload_sonic_checkpoint(run, path_at_update, root, checkpoint_config)
        _require_frozen_hashes(runner, static_policy, frozen_hashes)
        final_checkpoint = None
        if root is not None:
            final_checkpoint = root / "final"
            save_sonic_checkpoint(final_checkpoint, runner, metadata, git_commit)
            upload_sonic_checkpoint(run, final_checkpoint, root, checkpoint_config)
        print(last_metrics)
        if final_checkpoint is not None:
            print({"checkpoint_path": str(final_checkpoint)})
        succeeded = True
    finally:
        close_run(run, exit_code=0 if succeeded else 1)


def validate_sonic_training_config(config: SonicTrainingRunConfig) -> None:
    validate_ppo_config(config.config)
    validate_checkpoint_options(config.checkpoint_config)
    ppo = config.config
    if ppo.policy.architecture != "sonic":
        raise ValueError("SONIC requires policy.architecture=sonic")
    if ppo.policy.action_distribution != "gaussian":
        raise ValueError("SONIC uses an unsquashed Gaussian action distribution")
    if ppo.policy.obs_stack_size != 1:
        raise ValueError("SONIC owns its ten-frame observation history")
    if ppo.normalize_observations:
        raise ValueError("released SONIC actor inputs must remain raw")
    if not ppo.normalize_critic_observations:
        raise ValueError("SONIC requires the release-style critic normalizer")
    if ppo.critic_learning_rate is None:
        raise ValueError("SONIC requires an explicit independent critic learning rate")


def _tracking_dataset_summary(index: dict) -> dict:
    """Return the immutable dataset contract without multi-megabyte clip metadata."""

    split_summary = {}
    for split in ("train", "evaluation"):
        clips = [clip for clip in index["clips"] if clip["split"] == split]
        descriptor = index["split_files"][split]
        split_summary[split] = {
            "file": descriptor["file"],
            "clips": len(clips),
            "frames": sum(int(clip["frames"]) for clip in clips),
            "sampling_mass": float(descriptor["sampling_mass"]),
        }
    return {
        "schema_version": index["schema_version"],
        "dataset_fingerprint": index["dataset_fingerprint"],
        "portable_manifest_fingerprint": index["portable_manifest_fingerprint"],
        "policy_fps": index["policy_fps"],
        "train_sampling": index["train_sampling"],
        "split_files": index["split_files"],
        "joint_names": index["joint_names"],
        "qpos_joint_names": index["qpos_joint_names"],
        "body_names": index["body_names"],
        "g1_joint_names": index["g1_joint_names"],
        "robot_fingerprint": index["robot_fingerprint"],
        "gmr_revision": index["gmr_revision"],
        "bello_gmr_config_sha256": index["bello_gmr_config_sha256"],
        "g1_gmr_revision": index["g1_gmr_revision"],
        "g1_gmr_config_sha256": index["g1_gmr_config_sha256"],
        "g1_robot_sha256": index["g1_robot_sha256"],
        "joint_calibration": index["joint_calibration"],
        "heldout_calibration": index["heldout_calibration"],
        "sanitation_config": index["sanitation_config"],
        "splits": split_summary,
    }


def _checkpoint_metadata(
    run_config: SonicTrainingRunConfig,
    env: BelloSonicEnv,
    index: dict,
    release,
    source_revision: str,
) -> SonicCheckpointMetadata:
    calibration = env.reference_bank.calibration
    return SonicCheckpointMetadata(
        ppo=run_config.config,
        asset=replace(run_config.asset_config, output_root=None),
        warp=run_config.warp,
        obs_size=env.observation_size,
        critic_obs_size=env.critic_observation_size,
        action_size=env.action_size,
        actor_observation_layout=env.observation_layout,
        critic_observation_layout=env.critic_observation_layout,
        action_layout=env.action_layout,
        simulation=run_config.simulation,
        sonic_env=run_config.sonic_env,
        observation_contract=env.contract,
        dataset=SonicDatasetContract(
            index_path=run_config.dataset_index.expanduser().resolve(),
            schema_version=index["schema_version"],
            dataset_fingerprint=index["dataset_fingerprint"],
            robot_fingerprint=_prefixed_sha256(index["robot_fingerprint"]),
            bello_gmr_revision=index["gmr_revision"],
            bello_gmr_config_sha256=_prefixed_sha256(index["bello_gmr_config_sha256"]),
            g1_gmr_revision=index["g1_gmr_revision"],
            g1_config_sha256=_prefixed_sha256(index["g1_gmr_config_sha256"]),
            g1_robot_sha256=_prefixed_sha256(index["g1_robot_sha256"]),
            joint_names=tuple(index["joint_names"]),
            qpos_joint_names=tuple(index["qpos_joint_names"]),
            body_names=tuple(index["body_names"]),
            calibration=calibration,
        ),
        pretraining=run_config.pretraining,
        release_checkpoint_sha256=_prefixed_sha256(release.checkpoint_sha256),
        release_artifact_sha256=_prefixed_sha256(release.artifact_sha256),
        source_revision=source_revision,
    )


def _device_reference_bank_arrays(bank: SonicReferenceBank) -> tuple[jax.Array, ...]:
    arrays = tuple(
        jax.device_put(getattr(bank, name)) for name in _SONIC_BANK_ARRAY_FIELDS
    )
    return jax.block_until_ready(arrays)


def _bind_reference_bank(
    env: BelloSonicEnv,
    arrays: tuple[jax.Array, ...],
    metadata: tuple,
) -> BelloSonicEnv:
    if len(arrays) != len(_SONIC_BANK_ARRAY_FIELDS):
        raise ValueError("SONIC reference-bank array contract has drifted")
    (
        source_ids,
        sequence_ids,
        joint_names,
        qpos_joint_names,
        body_names,
        calibration,
        sampling_bin_size,
    ) = metadata
    bank = SonicReferenceBank(
        **dict(zip(_SONIC_BANK_ARRAY_FIELDS, arrays, strict=True)),
        source_ids=source_ids,
        sequence_ids=sequence_ids,
        joint_names=joint_names,
        qpos_joint_names=qpos_joint_names,
        body_names=body_names,
        calibration=calibration,
        sampling_bin_size=sampling_bin_size,
    )
    return env._unchecked_runtime_copy(reference_bank=bank)


def _require_frozen_hashes(runner, static_policy, expected: dict[str, str]) -> None:
    policy = runner_train_state(runner, static_policy).policy
    actual = sonic_frozen_parameter_hashes(policy)
    if actual != expected:
        changed = sorted(
            name
            for name in set(actual) | set(expected)
            if actual.get(name) != expected.get(name)
        )
        raise RuntimeError("frozen SONIC parameters changed: " + ", ".join(changed))


def _prefixed_sha256(value: str) -> str:
    return "sha256:" + value.removeprefix("sha256:")


def save_sonic_checkpoint(path: Path, runner, metadata, git_commit: str) -> None:
    save_training_checkpoint(
        path,
        runner,
        sonic_metadata_with_runner_state(metadata, runner),
        git_commit,
    )


def upload_sonic_checkpoint(
    run, path: Path, checkpoint_root: Path, config: CheckpointConfig
) -> None:
    if not config.upload_wandb_artifact or run is None:
        return
    track_checkpoint_artifact(
        run,
        path,
        name=config.wandb_artifact_name or checkpoint_root.name,
        artifact_type=config.wandb_artifact_type,
    )
