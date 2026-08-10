"""Single-stage PPO training for the isolated TWIST2 Bello task."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from pathlib import Path

import jax

from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.envs.twist import TwistSamplingConfig, TwistSimulationConfig
from bello_mujoco.envs.twist2 import (
    BelloTwist2Env,
    Twist2EnvConfig,
    Twist2ParityContract,
    default_twist2_asset_config,
)
from bello_mujoco.motion.twist2_dataset import (
    Twist2DatasetAudit,
    audit_twist2_dataset,
    require_released_twist2_dataset,
)
from bello_mujoco.motion.twist_dataset import (
    PackedReferenceBank,
    load_packed_reference_bank,
    materialize_packed_clip,
)
from bello_mujoco.repository import (
    capture_repository_commit,
    require_checkpoint_output_directory,
)
from bello_mujoco.sim import WarpConfig
from bello_mujoco.training.checkpoint import (
    CheckpointConfig,
    Twist2CheckpointMetadata,
    TwistDatasetContract,
    checkpoint_path,
    restore_runner_checkpoint,
    save_training_checkpoint,
    twist_metadata_with_runner_state,
    validate_checkpoint_options,
)
from bello_mujoco.training.ppo import (
    PPOConfig,
    init_runner_state,
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
from bello_mujoco.training.twist import (
    _REFERENCE_BANK_ARRAY_FIELDS,
    _device_reference_bank_arrays,
    _load_dataset_contract,
    _source_sampling_mass,
)


@dataclass(frozen=True)
class Twist2TrainingRunConfig:
    dataset_index: Path = Path("datasets/twist/index.json")
    require_released_dataset: bool = False
    config: PPOConfig = PPOConfig()
    asset_config: AssetPrepConfig = field(default_factory=default_twist2_asset_config)
    simulation: TwistSimulationConfig = TwistSimulationConfig()
    twist2_env: Twist2EnvConfig = Twist2EnvConfig()
    sampling: TwistSamplingConfig = TwistSamplingConfig(
        base_distribution="manifest_weight",
        curriculum_max_difficulty=10.0,
    )
    parity_contract: Twist2ParityContract = Twist2ParityContract()
    checkpoint_config: CheckpointConfig = CheckpointConfig()
    wandb_config: WandbConfig = WandbConfig()
    warp: WarpConfig = WarpConfig()
    runtime_clip_directory: Path = Path("outputs/twist2_runtime_clips")


def run_training(run_config: Twist2TrainingRunConfig) -> None:
    validate_twist2_training_config(run_config)
    dataset_index, dataset = _load_dataset_contract(run_config.dataset_index)
    dataset_audit = (
        require_released_twist2_dataset(dataset_index)
        if run_config.require_released_dataset
        else audit_twist2_dataset(dataset_index)
    )
    runtime_clip = (
        run_config.runtime_clip_directory
        / dataset.dataset_fingerprint.removeprefix("sha256:")
        / "train.npz"
    )
    materialize_packed_clip(dataset_index, runtime_clip, split="train")
    reference_bank = load_packed_reference_bank(
        dataset_index,
        split="train",
        sampling_mode=run_config.sampling.base_distribution,
    )
    source_sampling_mass = _source_sampling_mass(reference_bank)
    env = BelloTwist2Env.load(
        run_config.asset_config,
        run_config.simulation,
        motion_path=runtime_clip,
        config=run_config.twist2_env,
        warp=run_config.warp,
        batch_size=run_config.config.num_envs,
        reference_bank=reference_bank,
        sampling=run_config.sampling,
    )
    if env.action_layout.names != dataset.joint_names:
        raise ValueError("TWIST2 dataset joint order differs from the Bello model")
    missing_bodies = set(env.contract.key_body_names) - set(dataset.body_names)
    if missing_bodies:
        raise ValueError(
            f"TWIST2 dataset is missing configured key bodies: {sorted(missing_bodies)}"
        )
    validate_action_limit_contract(run_config.config, env.action_clip)
    git_commit = capture_repository_commit()
    metadata = _checkpoint_metadata(run_config, env, dataset, dataset_audit, git_commit)
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
    rng = jax.random.PRNGKey(run_config.config.seed)
    if checkpoint_config.resume_path is None:
        runner, static_policy, optimizer = init_runner_state(
            rng, env, run_config.config
        )
    else:
        runner, static_policy, optimizer, _ = restore_runner_checkpoint(
            checkpoint_config.resume_path,
            env,
            metadata,
            allow_git_commit_mismatch=checkpoint_config.allow_git_commit_mismatch,
        )
    static_env = env._unchecked_runtime_copy(reference_bank=None)
    reference_bank_arrays = _device_reference_bank_arrays(reference_bank)
    del env, reference_bank
    jax.clear_caches()
    run = None
    succeeded = False
    try:
        run = create_wandb_run(run_config.wandb_config)
        log_config(
            run,
            {
                "task": "twist2",
                "ppo": run_config.config,
                "dataset": dataset,
                "dataset_audit": dataset_audit,
                "simulation": run_config.simulation,
                "twist2_env": run_config.twist2_env,
                "sampling": run_config.sampling,
                "parity_contract": run_config.parity_contract,
                "source_sampling_mass": source_sampling_mass,
                "reference_bank_frames": int(reference_bank_arrays[0].shape[0]),
                "reference_bank_clips": int(reference_bank_arrays[-1].shape[0]),
                "checkpoint": checkpoint_config,
                "git_commit": git_commit,
            },
        )
        first_update = int(jax.device_get(runner.update))
        last_metrics = {}

        def update_fn(state, bank_arrays):
            current_env = _bind_reference_bank(static_env, bank_arrays)
            return update_runner_state(
                state,
                static_policy,
                optimizer,
                current_env,
                run_config.config,
            )

        update_fn = jax.jit(update_fn, donate_argnums=(0,))
        for update in range(first_update, run_config.config.num_updates):
            runner, last_metrics = update_fn(runner, reference_bank_arrays)
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
                context={"phase": "train", "stage": "twist2"},
            )
            if (
                root is not None
                and checkpoint_config.every_updates is not None
                and (update + 1) % checkpoint_config.every_updates == 0
            ):
                path_at_update = root / f"update_{update + 1:06d}"
                save_twist2_checkpoint(path_at_update, runner, metadata, git_commit)
                upload_twist2_checkpoint(run, path_at_update, root, checkpoint_config)
        final_checkpoint = None
        if root is not None:
            final_checkpoint = root / "final"
            save_twist2_checkpoint(final_checkpoint, runner, metadata, git_commit)
            upload_twist2_checkpoint(run, final_checkpoint, root, checkpoint_config)
        print(last_metrics)
        if final_checkpoint is not None:
            print({"checkpoint_path": str(final_checkpoint)})
        succeeded = True
    finally:
        close_run(run, exit_code=0 if succeeded else 1)


def validate_twist2_training_config(config: Twist2TrainingRunConfig) -> None:
    validate_ppo_config(config.config)
    validate_checkpoint_options(config.checkpoint_config)
    policy = config.config.policy
    if policy.architecture != "twist2":
        raise ValueError("TWIST2 requires policy.architecture=twist2")
    if policy.action_distribution != "gaussian":
        raise ValueError("released TWIST2 uses an unsquashed Gaussian policy")
    if not config.config.normalize_observations:
        raise ValueError("released TWIST2 uses running observation normalization")
    if config.config.normalize_critic_observations:
        raise ValueError(
            "TWIST2 explicitly preserves the release's effectively raw critic input"
        )
    if config.config.observation_norm_update_mode != "rollout":
        raise ValueError("TWIST2 observation moments update between rollouts")
    if config.config.observation_norm_update_updates != 3_000:
        raise ValueError("TWIST2 observation moments must freeze after 3,000 updates")
    if config.config.observation_norm_epsilon_mode != "std_offset" or not math.isclose(
        config.config.observation_norm_eps, 1.0e-4
    ):
        raise ValueError("TWIST2 observation normalization requires std + 1e-4")
    if config.config.observation_norm_clip < 1.0e8:
        raise ValueError(
            "TWIST2 observation normalization must be effectively unclipped"
        )
    if config.parity_contract != Twist2ParityContract():
        raise ValueError("TWIST2 parity contract differs from the pinned release")
    sampling = config.sampling
    if sampling.base_distribution != "manifest_weight":
        raise ValueError("released TWIST2 samples clips by unit manifest weight")
    if (
        not sampling.curriculum_enabled
        or not math.isclose(sampling.curriculum_gamma, 0.01)
        or not math.isclose(sampling.curriculum_min_difficulty, 1.0)
        or not math.isclose(sampling.curriculum_max_difficulty, 10.0)
    ):
        raise ValueError("TWIST2 motion curriculum must use gamma=.01 and [1, 10]")


def _checkpoint_metadata(
    run_config: Twist2TrainingRunConfig,
    env: BelloTwist2Env,
    dataset: TwistDatasetContract,
    dataset_audit: Twist2DatasetAudit,
    source_revision: str,
) -> Twist2CheckpointMetadata:
    return Twist2CheckpointMetadata(
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
        twist2_env=run_config.twist2_env,
        sampling=run_config.sampling,
        observation_contract=env.contract,
        parity_contract=run_config.parity_contract,
        dataset=dataset,
        dataset_audit=dataset_audit,
        robot_fingerprint=dataset.robot_fingerprint,
        source_revision=source_revision,
    )


def _bind_reference_bank(
    env: BelloTwist2Env, arrays: tuple[jax.Array, ...]
) -> BelloTwist2Env:
    if len(arrays) != len(_REFERENCE_BANK_ARRAY_FIELDS):
        raise ValueError("TWIST2 reference bank array contract has drifted")
    clip_count = int(arrays[-1].shape[0])
    reference_bank = PackedReferenceBank(
        **dict(zip(_REFERENCE_BANK_ARRAY_FIELDS, arrays, strict=True)),
        source_ids=("",) * clip_count,
        sequence_ids=("",) * clip_count,
    )
    return env._unchecked_runtime_copy(reference_bank=reference_bank)


def save_twist2_checkpoint(
    path: Path,
    runner,
    metadata: Twist2CheckpointMetadata,
    git_commit: str,
) -> None:
    save_training_checkpoint(
        path,
        runner,
        twist_metadata_with_runner_state(metadata, runner),
        git_commit,
    )


def upload_twist2_checkpoint(
    run,
    path: Path,
    checkpoint_root: Path,
    config: CheckpointConfig,
) -> None:
    if not config.upload_wandb_artifact or run is None:
        return
    track_checkpoint_artifact(
        run,
        path,
        name=config.wandb_artifact_name or checkpoint_root.name,
        artifact_type=config.wandb_artifact_type,
    )
