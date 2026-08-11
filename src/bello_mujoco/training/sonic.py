"""Strict edge-only SONIC transfer learning on synchronized Bello AMASS data."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.envs.sonic import (
    BelloSonicEnv,
    SonicEnvConfig,
    SonicSamplingState,
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
    reset_runner_environment,
    runner_train_state,
    update_runner_state,
    validate_action_limit_contract,
    validate_ppo_config,
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
    checkpoint_config: CheckpointConfig = CheckpointConfig()
    wandb_config: WandbConfig = WandbConfig()
    warp: WarpConfig = WarpConfig()
    runtime_clip_directory: Path = Path("outputs/sonic_runtime_clips")
    reference_shard_updates: int = 32


def run_training(run_config: SonicTrainingRunConfig) -> None:
    validate_sonic_training_config(run_config)
    index_path = run_config.dataset_index.expanduser().resolve()
    index = load_sonic_dataset_index(index_path)
    initial_shard_index = sonic_reference_shard_for_update(
        index,
        update=0,
        seed=run_config.config.seed,
        full_shard_updates=run_config.reference_shard_updates,
    )
    reference_bank = load_sonic_reference_bank(
        index_path,
        split="train",
        shard_index=initial_shard_index,
        sampling_bin_size=run_config.sonic_env.sampling.bin_size,
    )
    runtime_clip = (
        run_config.runtime_clip_directory
        / index["dataset_fingerprint"].removeprefix("sha256:")
        / "train.npz"
    )
    train_clips = [
        clip
        for clip in index["clips"]
        if clip["split"] == "train" and clip["shard_index"] == initial_shard_index
    ]
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
        minimum_action_std=float(np.exp(run_config.config.policy.init_log_std)),
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
    current_shard_index = (
        initial_shard_index
        if first_update == 0
        else sonic_reference_shard_for_update(
            index,
            update=first_update - 1,
            seed=run_config.config.seed,
            full_shard_updates=run_config.reference_shard_updates,
        )
    )
    if current_shard_index != initial_shard_index:
        reference_bank = load_sonic_reference_bank(
            index_path,
            split="train",
            shard_index=current_shard_index,
            sampling_bin_size=run_config.sonic_env.sampling.bin_size,
        )
    static_env = env._unchecked_runtime_copy(reference_bank=None)
    bank_arrays = _device_reference_bank_arrays(reference_bank)
    bank_metadata = (
        reference_bank.source_ids,
        reference_bank.sequence_ids,
        reference_bank.joint_names,
        reference_bank.body_names,
        reference_bank.calibration,
        reference_bank.sampling_bin_size,
    )
    sampling_states = {
        current_shard_index: jax.device_get(runner.task_state),
    }
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
                "dataset": index,
                "simulation": run_config.simulation,
                "sonic_env": run_config.sonic_env,
                "reference_shard_updates": run_config.reference_shard_updates,
                "release_checkpoint_sha256": release.checkpoint_sha256,
                "release_artifact_sha256": release.artifact_sha256,
                "frozen_parameter_sha256": frozen_hashes,
                "checkpoint": checkpoint_config,
                "git_commit": git_commit,
            },
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

        def reset_fn(state, arrays, task_state):
            current_env = _bind_reference_bank(
                static_env, arrays, bank_metadata
            ).with_task_state(task_state)
            reset = reset_runner_environment(state, current_env, run_config.config)
            return reset._replace(task_state=task_state)

        update_fn = jax.jit(update_fn, donate_argnums=(0,))
        reset_fn = jax.jit(reset_fn, donate_argnums=(0,))
        for update in range(first_update, run_config.config.num_updates):
            desired_shard_index = sonic_reference_shard_for_update(
                index,
                update=update,
                seed=run_config.config.seed,
                full_shard_updates=run_config.reference_shard_updates,
            )
            if desired_shard_index != current_shard_index:
                sampling_states[current_shard_index] = jax.device_get(runner.task_state)
                del bank_arrays
                reference_bank = load_sonic_reference_bank(
                    index_path,
                    split="train",
                    shard_index=desired_shard_index,
                    sampling_bin_size=run_config.sonic_env.sampling.bin_size,
                )
                task_state = sampling_states.get(desired_shard_index)
                if task_state is None:
                    task_state = _initial_sampling_state(reference_bank, run_config)
                    sampling_states[desired_shard_index] = task_state
                bank_arrays = _device_reference_bank_arrays(reference_bank)
                del reference_bank
                runner = reset_fn(runner, bank_arrays, task_state)
                current_shard_index = desired_shard_index
            runner, last_metrics = update_fn(runner, bank_arrays)
            last_metrics = {
                **last_metrics,
                "dataset/reference_shard": jnp.asarray(
                    current_shard_index, dtype=jnp.float32
                ),
                "dataset/reference_shard_count": jnp.asarray(
                    len(index["shards"]["train"]), dtype=jnp.float32
                ),
            }
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
    if (
        isinstance(config.reference_shard_updates, bool)
        or config.reference_shard_updates <= 0
    ):
        raise ValueError("SONIC reference_shard_updates must be a positive integer")


def sonic_reference_shard_for_update(
    index: dict,
    *,
    update: int,
    seed: int,
    full_shard_updates: int,
) -> int:
    """Return a deterministic, sampling-mass-weighted shard sweep position."""

    if update < 0:
        raise ValueError("SONIC shard schedule update must be nonnegative")
    if isinstance(full_shard_updates, bool) or full_shard_updates <= 0:
        raise ValueError("SONIC full-shard update count must be positive")
    descriptors = index.get("shards", {}).get("train")
    if not isinstance(descriptors, list) or not descriptors:
        raise ValueError("SONIC index has no training shards")
    masses = np.asarray(
        [float(descriptor["sampling_mass"]) for descriptor in descriptors],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("SONIC training shard masses are invalid")
    block_updates = np.maximum(
        1,
        np.floor(full_shard_updates * masses / np.max(masses) + 0.5).astype(np.int64),
    )
    order = np.roll(
        np.arange(len(descriptors), dtype=np.int32), -(seed % len(descriptors))
    )
    cycle_updates = int(np.sum(block_updates, dtype=np.int64))
    position = update % cycle_updates
    for shard_index in order.tolist():
        count = int(block_updates[shard_index])
        if position < count:
            return shard_index
        position -= count
    raise RuntimeError("SONIC shard schedule failed to select a shard")


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
            body_names=tuple(index["body_names"]),
            calibration=calibration,
            shard_frame_capacity=index["shard_frame_capacity"],
            shard_clip_capacity=index["shard_clip_capacity"],
        ),
        release_checkpoint_sha256=_prefixed_sha256(release.checkpoint_sha256),
        release_artifact_sha256=_prefixed_sha256(release.artifact_sha256),
        source_revision=source_revision,
        reference_shard_updates=run_config.reference_shard_updates,
    )


def _device_reference_bank_arrays(bank: SonicReferenceBank) -> tuple[jax.Array, ...]:
    return tuple(jnp.asarray(getattr(bank, name)) for name in _SONIC_BANK_ARRAY_FIELDS)


def _initial_sampling_state(
    bank: SonicReferenceBank, run_config: SonicTrainingRunConfig
) -> SonicSamplingState:
    bins = len(bank.sampling_bin_clip_indices)
    failures = np.full(
        (bins,),
        run_config.sonic_env.sampling.initial_failures,
        dtype=np.float32,
    )
    return SonicSamplingState(failures=failures, attempts=np.ones_like(failures))


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
        body_names,
        calibration,
        sampling_bin_size,
    ) = metadata
    bank = SonicReferenceBank(
        **dict(zip(_SONIC_BANK_ARRAY_FIELDS, arrays, strict=True)),
        source_ids=source_ids,
        sequence_ids=sequence_ids,
        joint_names=joint_names,
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
