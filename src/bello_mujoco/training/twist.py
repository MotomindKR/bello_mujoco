"""Typed teacher PPO and frozen-teacher student distillation for TWIST."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import jax
import numpy as np

from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.envs.twist import (
    BelloTwistEnv,
    TwistEnvConfig,
    TwistSimulationConfig,
    TwistStage,
    default_twist_asset_config,
)
from bello_mujoco.motion.twist_dataset import (
    PackedReferenceBank,
    TWIST_DATASET_SCHEMA_VERSION,
    TWIST_PACKING_ALGORITHM,
    TWIST_TRAIN_SAMPLING,
    load_packed_reference_bank,
    materialize_packed_clip,
    packed_reference_bank_shards,
)
from bello_mujoco.repository import (
    capture_repository_commit,
    require_checkpoint_output_directory,
)
from bello_mujoco.sim import WarpConfig
from bello_mujoco.training.checkpoint import (
    CheckpointConfig,
    TwistDatasetContract,
    TwistKLSchedule,
    TwistStudentCheckpointMetadata,
    TwistTeacherCheckpointMetadata,
    checkpoint_metadata_fingerprint,
    checkpoint_path,
    load_checkpoint_metadata,
    load_checkpoint_update,
    restore_evaluation_checkpoint,
    restore_runner_checkpoint,
    save_training_checkpoint,
    twist_metadata_with_runner_state,
    validate_checkpoint_options,
)
from bello_mujoco.training.ppo import (
    DistillationContext,
    PPOConfig,
    init_runner_state,
    update_runner_state,
    validate_action_limit_contract,
    validate_ppo_config,
)
from bello_mujoco.policy import init_observation_stack
from bello_mujoco.training.tracking import (
    WandbConfig,
    close_run,
    create_wandb_run,
    log_config,
    track_metric_snapshot,
)


@dataclass(frozen=True)
class TwistTrainingRunConfig:
    stage: TwistStage
    dataset_index: Path = Path("datasets/twist/index.json")
    teacher_checkpoint_path: Path | None = None
    config: PPOConfig = PPOConfig()
    asset_config: AssetPrepConfig = field(default_factory=default_twist_asset_config)
    simulation: TwistSimulationConfig = TwistSimulationConfig()
    twist_env: TwistEnvConfig = TwistEnvConfig()
    checkpoint_config: CheckpointConfig = CheckpointConfig()
    wandb_config: WandbConfig = WandbConfig()
    warp: WarpConfig = WarpConfig()
    runtime_clip_directory: Path = Path("outputs/twist_runtime_clips")
    reference_bank_refresh_updates: int = 500


def run_training(run_config: TwistTrainingRunConfig) -> None:
    validate_twist_training_config(run_config)
    dataset_index, dataset = _load_dataset_contract(run_config.dataset_index)
    runtime_clip = (
        run_config.runtime_clip_directory
        / dataset.dataset_fingerprint.removeprefix("sha256:")
        / "train.npz"
    )
    materialize_packed_clip(dataset_index, runtime_clip, split="train")
    shard_names, shard_probabilities, pad_frames, pad_clips = (
        packed_reference_bank_shards(dataset_index, split="train")
    )
    resume_update = 0
    if run_config.checkpoint_config.resume_path is not None:
        resume_update = load_checkpoint_update(
            run_config.checkpoint_config.resume_path,
            allow_git_commit_mismatch=(
                run_config.checkpoint_config.allow_git_commit_mismatch
            ),
        )

    def load_bank(update: int) -> tuple[str, PackedReferenceBank]:
        shard_index = _reference_bank_shard_index(
            shard_probabilities,
            seed=run_config.config.seed,
            update=update,
            refresh_updates=run_config.reference_bank_refresh_updates,
        )
        shard_name = shard_names[shard_index]
        return shard_name, load_packed_reference_bank(
            dataset_index,
            split="train",
            shard_name=shard_name,
            pad_frames=pad_frames,
            pad_clips=pad_clips,
        )

    active_shard, reference_bank = load_bank(resume_update)
    simulation = run_config.simulation
    env = BelloTwistEnv.load(
        run_config.stage,
        run_config.asset_config,
        simulation,
        motion_path=runtime_clip,
        config=run_config.twist_env,
        warp=run_config.warp,
        batch_size=run_config.config.num_envs,
        reference_bank=reference_bank,
    )
    if env.action_layout.names != dataset.joint_names:
        raise ValueError("TWIST dataset joint order differs from the Bello model")
    missing_bodies = set(env.contract.key_body_names) - set(dataset.body_names)
    if missing_bodies:
        raise ValueError(
            f"TWIST dataset is missing configured key bodies: {sorted(missing_bodies)}"
        )
    validate_action_limit_contract(run_config.config, env.action_clip)
    git_commit = capture_repository_commit()
    metadata = _checkpoint_metadata(run_config, env, simulation, dataset, git_commit)
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
    distillation = _distillation_context(run_config, metadata)
    rng = jax.random.PRNGKey(run_config.config.seed)
    reset_resumed_environment = False
    if checkpoint_config.resume_path is None:
        runner, static_policy, optimizer = init_runner_state(
            rng, env, run_config.config
        )
    else:
        runner, static_policy, optimizer, _ = restore_runner_checkpoint(
            checkpoint_config.resume_path,
            env,
            metadata,
            allow_git_commit_mismatch=(checkpoint_config.allow_git_commit_mismatch),
        )
        reset_resumed_environment = (
            resume_update > 0
            and resume_update % run_config.reference_bank_refresh_updates == 0
        )
    static_env = _replace_reference_bank_unchecked(env, None)
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
                "task": f"twist_{run_config.stage}",
                "ppo": run_config.config,
                "dataset": dataset,
                "simulation": simulation,
                "twist_env": run_config.twist_env,
                "reference_bank_shards": len(shard_names),
                "reference_bank_refresh_updates": (
                    run_config.reference_bank_refresh_updates
                ),
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
                distillation=distillation,
            )

        def reset_fn(state, bank_arrays):
            current_env = _bind_reference_bank(static_env, bank_arrays)
            return _reset_runner_environment(state, current_env, run_config.config)

        update_fn = jax.jit(update_fn, donate_argnums=(0,))
        reset_fn = jax.jit(reset_fn, donate_argnums=(0,))
        if reset_resumed_environment:
            runner = jax.block_until_ready(
                reset_fn(runner, reference_bank_arrays)
            )
        for update in range(first_update, run_config.config.num_updates):
            if (
                update > first_update
                and update % run_config.reference_bank_refresh_updates == 0
            ):
                active_shard, next_reference_bank = load_bank(update)
                _ = replace(static_env, reference_bank=next_reference_bank)
                _delete_device_arrays(reference_bank_arrays)
                reference_bank_arrays = _device_reference_bank_arrays(
                    next_reference_bank
                )
                del next_reference_bank
                runner = jax.block_until_ready(
                    reset_fn(runner, reference_bank_arrays)
                )
            runner, last_metrics = update_fn(runner, reference_bank_arrays)
            last_metrics = jax.tree_util.tree_map(
                lambda value: value.block_until_ready(), last_metrics
            )
            last_metrics["dataset/reference_bank_shard"] = jax.numpy.asarray(
                shard_names.index(active_shard), dtype=jax.numpy.float32
            )
            track_metric_snapshot(
                run,
                last_metrics,
                env_steps=(update + 1)
                * run_config.config.num_envs
                * run_config.config.num_steps,
                update=update + 1,
                context={"phase": "train", "stage": run_config.stage},
            )
            if (
                root is not None
                and checkpoint_config.every_updates is not None
                and (update + 1) % checkpoint_config.every_updates == 0
            ):
                save_twist_checkpoint(
                    root / f"update_{update + 1:06d}",
                    runner,
                    metadata,
                    git_commit,
                )
        final_checkpoint = None
        if root is not None:
            final_checkpoint = root / "final"
            save_twist_checkpoint(final_checkpoint, runner, metadata, git_commit)
        print(last_metrics)
        if final_checkpoint is not None:
            print({"checkpoint_path": str(final_checkpoint)})
        succeeded = True
    finally:
        close_run(run, exit_code=0 if succeeded else 1)


def validate_twist_training_config(config: TwistTrainingRunConfig) -> None:
    validate_ppo_config(config.config)
    validate_checkpoint_options(config.checkpoint_config)
    if config.reference_bank_refresh_updates <= 0:
        raise ValueError("reference_bank_refresh_updates must be positive")
    expected_architecture = f"twist_{config.stage}"
    if config.config.policy.architecture != expected_architecture:
        raise ValueError(
            f"stage={config.stage} requires policy.architecture={expected_architecture}"
        )
    if not config.config.normalize_observations:
        raise ValueError(
            "released TWIST training requires running observation normalization"
        )
    if config.stage == "teacher" and config.teacher_checkpoint_path is not None:
        raise ValueError("TWIST teacher does not accept teacher_checkpoint_path")
    if config.stage == "student" and config.teacher_checkpoint_path is None:
        raise ValueError("TWIST student requires teacher_checkpoint_path")


def _reference_bank_shard_index(
    probabilities: np.ndarray,
    *,
    seed: int,
    update: int,
    refresh_updates: int,
) -> int:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("TWIST shard probabilities must be a nonempty vector")
    if np.any(~np.isfinite(probabilities)) or np.any(probabilities <= 0.0):
        raise ValueError("TWIST shard probabilities must be positive and finite")
    refresh_epoch = update // refresh_updates
    cycle, cycle_index = divmod(refresh_epoch, len(probabilities))
    rng = np.random.default_rng(np.random.SeedSequence((seed, cycle)))
    permutation = rng.permutation(len(probabilities))
    return int(permutation[cycle_index])


def _reset_runner_environment(runner, env: BelloTwistEnv, config: PPOConfig):
    rng, reset_key = jax.random.split(runner.rng)
    env_state, observation = env.reset_batch(config.num_envs, reset_key)
    critic_observation = env.critic_observation_batch(env_state)
    return runner._replace(
        env_state=env_state,
        obs_stack=init_observation_stack(observation, config.policy.obs_stack_size),
        critic_obs_stack=init_observation_stack(
            critic_observation, config.policy.obs_stack_size
        ),
        rng=rng,
    )


_REFERENCE_BANK_ARRAY_FIELDS = (
    "qpos",
    "qvel",
    "body_pos",
    "body_quat_wxyz",
    "body_lin_vel",
    "body_ang_vel",
    "foot_contacts",
    "offsets",
    "lengths",
    "sampling_probabilities",
)


def _device_reference_bank_arrays(
    reference_bank: PackedReferenceBank,
) -> tuple[jax.Array, ...]:
    arrays = tuple(
        jax.device_put(getattr(reference_bank, name))
        for name in _REFERENCE_BANK_ARRAY_FIELDS
    )
    return jax.block_until_ready(arrays)


def _delete_device_arrays(arrays: tuple[jax.Array, ...]) -> None:
    jax.block_until_ready(arrays)
    for value in arrays:
        value.delete()


def _bind_reference_bank(
    env: BelloTwistEnv, arrays: tuple[jax.Array, ...]
) -> BelloTwistEnv:
    if len(arrays) != len(_REFERENCE_BANK_ARRAY_FIELDS):
        raise ValueError("TWIST reference bank array contract has drifted")
    clip_count = int(arrays[-1].shape[0])
    reference_bank = PackedReferenceBank(
        **dict(zip(_REFERENCE_BANK_ARRAY_FIELDS, arrays, strict=True)),
        source_ids=("",) * clip_count,
        sequence_ids=("",) * clip_count,
    )
    return _replace_reference_bank_unchecked(env, reference_bank)


def _replace_reference_bank_unchecked(
    env: BelloTwistEnv, reference_bank: PackedReferenceBank | None
) -> BelloTwistEnv:
    bound = object.__new__(BelloTwistEnv)
    object.__setattr__(bound, "model", env.model)
    object.__setattr__(bound, "motion", env.motion)
    object.__setattr__(bound, "stage", env.stage)
    object.__setattr__(bound, "config", env.config)
    object.__setattr__(bound, "reference_bank", reference_bank)
    return bound


def save_twist_checkpoint(
    path: Path,
    runner,
    metadata: TwistTeacherCheckpointMetadata | TwistStudentCheckpointMetadata,
    git_commit: str,
) -> None:
    save_training_checkpoint(
        path,
        runner,
        twist_metadata_with_runner_state(metadata, runner),
        git_commit,
    )


def _checkpoint_metadata(
    run_config: TwistTrainingRunConfig,
    env: BelloTwistEnv,
    simulation: TwistSimulationConfig,
    dataset: TwistDatasetContract,
    source_revision: str,
):
    common = {
        "ppo": run_config.config,
        "asset": replace(run_config.asset_config, output_root=None),
        "warp": run_config.warp,
        "obs_size": env.observation_size,
        "critic_obs_size": env.critic_observation_size,
        "action_size": env.action_size,
        "actor_observation_layout": env.observation_layout,
        "critic_observation_layout": env.critic_observation_layout,
        "action_layout": env.action_layout,
        "simulation": simulation,
        "twist_env": run_config.twist_env,
        "observation_contract": env.contract,
        "dataset": dataset,
        "robot_fingerprint": dataset.robot_fingerprint,
        "source_revision": source_revision,
    }
    if run_config.stage == "teacher":
        return TwistTeacherCheckpointMetadata(**common)
    teacher_path = run_config.teacher_checkpoint_path
    if teacher_path is None:
        raise ValueError("teacher checkpoint path is required")
    teacher_metadata = load_checkpoint_metadata(teacher_path)
    if not isinstance(teacher_metadata, TwistTeacherCheckpointMetadata):
        raise ValueError("TWIST student teacher_checkpoint_path must be a teacher")
    _validate_teacher_compatibility(teacher_metadata, dataset, env)
    return TwistStudentCheckpointMetadata(
        **common,
        teacher_fingerprint=checkpoint_metadata_fingerprint(teacher_metadata),
        kl_schedule=TwistKLSchedule(),
    )


def _distillation_context(run_config: TwistTrainingRunConfig, student_metadata):
    if run_config.stage == "teacher":
        return None
    teacher_path = run_config.teacher_checkpoint_path
    if teacher_path is None:
        raise ValueError("teacher checkpoint path is required")
    teacher_state, teacher_metadata, teacher_normalization = (
        restore_evaluation_checkpoint(
            teacher_path,
            allow_git_commit_mismatch=(
                run_config.checkpoint_config.allow_git_commit_mismatch
            ),
        )
    )
    if not isinstance(teacher_metadata, TwistTeacherCheckpointMetadata):
        raise ValueError("student distillation requires a typed TWIST teacher")
    if (
        checkpoint_metadata_fingerprint(teacher_metadata)
        != student_metadata.teacher_fingerprint
    ):
        raise ValueError("student metadata teacher fingerprint mismatch")
    return DistillationContext(
        teacher_policy=teacher_state.policy,
        teacher_observation_size=teacher_metadata.obs_size,
        teacher_observation_norm_state=teacher_normalization.observation,
        teacher_config=teacher_metadata.ppo,
    )


def _validate_teacher_compatibility(
    teacher: TwistTeacherCheckpointMetadata,
    dataset: TwistDatasetContract,
    env: BelloTwistEnv,
) -> None:
    mismatches = []
    if teacher.dataset.dataset_fingerprint != dataset.dataset_fingerprint:
        mismatches.append("dataset fingerprint")
    if teacher.robot_fingerprint != dataset.robot_fingerprint:
        mismatches.append("robot fingerprint")
    if teacher.action_layout != env.action_layout:
        mismatches.append("joint ordering")
    if teacher.observation_contract != env.contract:
        mismatches.append("observation contract")
    if mismatches:
        raise ValueError(
            "TWIST teacher is incompatible with the student run: "
            + ", ".join(mismatches)
        )


def _load_dataset_contract(
    index_path: Path,
) -> tuple[Path, TwistDatasetContract]:
    index_path = Path(index_path).expanduser().resolve()
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"TWIST packed index not found: {index_path}; build operator-supplied "
            "AMASS/OMOMO data first"
        ) from error
    if index.get("schema_version") != TWIST_DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported TWIST dataset index schema")
    packing = index.get("packing")
    if (
        not isinstance(packing, dict)
        or packing.get("algorithm") != TWIST_PACKING_ALGORITHM
    ):
        raise ValueError("unsupported TWIST dataset packing algorithm")
    if packing.get("train_sampling") != TWIST_TRAIN_SAMPLING:
        raise ValueError("unsupported TWIST dataset sampling contract")
    return index_path, TwistDatasetContract(
        index_path=index_path,
        schema_version=index["schema_version"],
        dataset_fingerprint=index["dataset_fingerprint"],
        robot_fingerprint=index["robot_fingerprint"],
        gmr_revision=index["gmr_revision"],
        joint_names=tuple(index["joint_names"]),
        body_names=tuple(index["body_names"]),
    )
