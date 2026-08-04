"""Typed teacher PPO and frozen-teacher student distillation for TWIST."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import jax

from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.envs.motion_tracking import MotionTrackingEnvConfig
from bello_mujoco.envs.twist import (
    BelloTwistEnv,
    TwistEnvConfig,
    TwistStage,
    default_twist_asset_config,
)
from bello_mujoco.motion.twist_dataset import (
    TWIST_DATASET_SCHEMA_VERSION,
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
    TwistDatasetContract,
    TwistKLSchedule,
    TwistStudentCheckpointMetadata,
    TwistTeacherCheckpointMetadata,
    checkpoint_metadata_fingerprint,
    checkpoint_path,
    load_checkpoint_metadata,
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
    motion_env: MotionTrackingEnvConfig = MotionTrackingEnvConfig()
    twist_env: TwistEnvConfig = TwistEnvConfig()
    checkpoint_config: CheckpointConfig = CheckpointConfig()
    wandb_config: WandbConfig = WandbConfig()
    warp: WarpConfig = WarpConfig()
    runtime_clip_directory: Path = Path("outputs/twist_runtime_clips")


def run_training(run_config: TwistTrainingRunConfig) -> None:
    validate_twist_training_config(run_config)
    dataset_index, dataset = _load_dataset_contract(run_config.dataset_index)
    runtime_clip = (
        run_config.runtime_clip_directory
        / dataset.dataset_fingerprint.removeprefix("sha256:")
        / "train.npz"
    )
    materialize_packed_clip(dataset_index, runtime_clip, split="train")
    reference_bank = load_packed_reference_bank(dataset_index, split="train")
    motion_env = replace(run_config.motion_env, motion_path=runtime_clip)
    env = BelloTwistEnv.load(
        run_config.stage,
        run_config.asset_config,
        motion_env,
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
    metadata = _checkpoint_metadata(run_config, env, motion_env, dataset, git_commit)
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
                "motion_env": motion_env,
                "twist_env": run_config.twist_env,
                "checkpoint": checkpoint_config,
                "git_commit": git_commit,
            },
        )
        first_update = int(jax.device_get(runner.update))
        last_metrics = {}
        for update in range(first_update, run_config.config.num_updates):
            runner, last_metrics = update_runner_state(
                runner,
                static_policy,
                optimizer,
                env,
                run_config.config,
                distillation=distillation,
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
    expected_architecture = f"twist_{config.stage}"
    if config.config.policy.architecture != expected_architecture:
        raise ValueError(
            f"stage={config.stage} requires policy.architecture={expected_architecture}"
        )
    if config.config.normalize_observations:
        raise ValueError(
            "TWIST uses its explicit sensor scaling contract; generic PPO "
            "observation normalization must remain disabled so student critic "
            "references match frozen teacher actor inputs"
        )
    if config.stage == "teacher" and config.teacher_checkpoint_path is not None:
        raise ValueError("TWIST teacher does not accept teacher_checkpoint_path")
    if config.stage == "student" and config.teacher_checkpoint_path is None:
        raise ValueError("TWIST student requires teacher_checkpoint_path")


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
    motion_env: MotionTrackingEnvConfig,
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
        "motion_env": motion_env,
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
    teacher_state, teacher_metadata, _ = restore_evaluation_checkpoint(
        teacher_path,
        allow_git_commit_mismatch=(
            run_config.checkpoint_config.allow_git_commit_mismatch
        ),
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
    return index_path, TwistDatasetContract(
        index_path=index_path,
        schema_version=index["schema_version"],
        dataset_fingerprint=index["dataset_fingerprint"],
        robot_fingerprint=index["robot_fingerprint"],
        gmr_revision=index["gmr_revision"],
        joint_names=tuple(index["joint_names"]),
        body_names=tuple(index["body_names"]),
    )
