"""Train PPO with BeyondMimic motion tracking."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import jax

from bello_mujoco.models.compiler import ModelConfig
from bello_mujoco.training.checkpoint import (
    CheckpointConfig,
    MotionCheckpointMetadata,
    checkpoint_path,
    MOTION_FILE,
    restore_runner_checkpoint,
    save_training_checkpoint,
    validate_checkpoint_options,
)
from bello_mujoco.envs.motion_tracking import (
    BelloMotionTrackingEnv,
    MotionTrackingEnvConfig,
)
from bello_mujoco.evaluation.software import require_linux_software_rendering
from bello_mujoco.evaluation.motion import (
    MotionEvaluationRequest,
    evaluate_motion_video,
)
from bello_mujoco.motion.clip import MotionClip, motion_tracking_asset_config
from bello_mujoco.policy import policy_observation_size
from bello_mujoco.sim import WarpConfig
from bello_mujoco.training.runtime import (
    RunnerInitialization,
    should_render_training_video,
    training_updates,
)
from bello_mujoco.training.tracking import (
    WandbConfig,
    close_run,
    create_wandb_run,
    log_config,
    track_checkpoint_artifact,
    track_metric_snapshot,
    track_video,
)
from bello_mujoco.training.ppo_config import (
    PPOConfig,
    validate_action_limit_contract,
    validate_ppo_config,
)
from bello_mujoco.training.ppo_types import RunnerState
from bello_mujoco.training.ppo import init_runner_state
from bello_mujoco.repository import (
    capture_repository_commit,
    require_checkpoint_output_directory,
)


@dataclass(frozen=True)
class MotionSelectionConfig:
    directory: Path = Path("motions/asap_fixed_head")
    sequence: str = "video_single_foot_balance_level1_filter_amass"


@dataclass(frozen=True)
class MotionEvalVideoConfig:
    enabled: bool = True
    every_updates: int = 100
    render_initial: bool = False
    render_final: bool = True
    output_dir: Path = Path("outputs/motion_tracking/eval_videos")
    filename_prefix: str = "eval"
    steps: int = 300
    width: int = 640
    height: int = 480
    fps: int = 50
    log_wandb: bool = True
    wandb_name: str = "eval/motion_policy"


@dataclass(frozen=True)
class MotionTrainingRunConfig:
    asset_config: ModelConfig
    config: PPOConfig = PPOConfig()
    env_config: MotionTrackingEnvConfig = MotionTrackingEnvConfig()
    motion: MotionSelectionConfig = MotionSelectionConfig()
    checkpoint_config: CheckpointConfig = CheckpointConfig()
    wandb_config: WandbConfig = WandbConfig()
    video_config: MotionEvalVideoConfig = MotionEvalVideoConfig()
    warp: WarpConfig = WarpConfig()


def run_training(run_config: MotionTrainingRunConfig) -> None:
    run_config = resolve_motion_selection(run_config)
    validate_ppo_config(run_config.config)
    validate_action_limit_contract(run_config.config, run_config.env_config.action_clip)
    _validate_motion_asset(run_config.asset_config)
    validate_motion_video_config(run_config.video_config)
    if run_config.video_config.enabled:
        require_linux_software_rendering()
    validate_motion_checkpoint_config(run_config.checkpoint_config)
    if run_config.video_config.enabled and not run_config.checkpoint_config.enabled:
        raise ValueError("motion evaluation requires checkpointing to be enabled")
    git_commit = None
    if run_config.checkpoint_config.enabled:
        run_config = replace(
            run_config,
            checkpoint_config=replace(
                run_config.checkpoint_config,
                directory=require_checkpoint_output_directory(
                    run_config.checkpoint_config.directory
                ),
            ),
        )
        git_commit = capture_repository_commit()
    run = None
    succeeded = False
    try:
        run = create_wandb_run(run_config.wandb_config)
        env = BelloMotionTrackingEnv.load(
            run_config.asset_config,
            run_config.env_config,
            warp=run_config.warp,
            batch_size=run_config.config.num_envs,
        )
        policy_obs_size = policy_observation_size(
            env.observation_size, run_config.config.policy
        )
        critic_obs_size = env.critic_observation_size
        critic_policy_obs_size = policy_observation_size(
            critic_obs_size, run_config.config.policy
        )
        metadata = motion_checkpoint_metadata(
            run_config,
            env,
            policy_obs_size,
            critic_policy_obs_size,
        )
        rng = jax.random.PRNGKey(run_config.config.seed)
        log_config(
            run,
            {
                "task": "motion_tracking",
                "ppo": run_config.config,
                "asset": run_config.asset_config,
                "env": run_config.env_config,
                "checkpoint": run_config.checkpoint_config,
                "git_commit": git_commit,
                "video": run_config.video_config,
                "warp": run_config.warp,
            },
        )
        checkpoint_root = (
            checkpoint_path(run_config.checkpoint_config, run_config.config)
            if run_config.checkpoint_config.enabled
            else None
        )
        initial_runner = load_motion_resume(
            run_config.checkpoint_config.resume_path,
            env,
            metadata,
        )
        out = motion_train_update_loop(
            env,
            run_config.config,
            rng,
            run_config.video_config,
            run,
            checkpoint_root=checkpoint_root,
            checkpoint_context=(
                (metadata, git_commit, run_config.checkpoint_config)
                if git_commit is not None
                else None
            ),
            initial_runner=initial_runner,
        )
        final_checkpoint = None
        if checkpoint_root is not None:
            if git_commit is None:
                raise RuntimeError("checkpoint Git commit was not captured")
            final_checkpoint = checkpoint_root / "final"
            runner_state = out["runner_state"]
            save_motion_checkpoint(
                final_checkpoint,
                runner_state,
                metadata,
                git_commit,
                env.motion,
            )
            upload_motion_checkpoint(
                run,
                final_checkpoint,
                checkpoint_root,
                run_config.checkpoint_config,
            )
        print(out["metrics"])
        if final_checkpoint is not None:
            print({"checkpoint_path": str(final_checkpoint)})
        succeeded = True
    finally:
        close_run(run, exit_code=0 if succeeded else 1)


def resolve_motion_selection(
    run_config: MotionTrainingRunConfig,
) -> MotionTrainingRunConfig:
    if run_config.checkpoint_config.resume_path is not None:
        return replace(
            run_config,
            env_config=replace(
                run_config.env_config,
                motion_path=run_config.checkpoint_config.resume_path.resolve()
                / MOTION_FILE,
            ),
        )
    sequence = run_config.motion.sequence
    if not sequence or Path(sequence).name != sequence or Path(sequence).suffix:
        raise ValueError(
            "motion.sequence must be a non-empty sequence name without a path or suffix"
        )
    motion_path = run_config.motion.directory / f"{sequence}.npz"
    if not motion_path.is_file():
        available = sorted(
            path.stem for path in run_config.motion.directory.glob("*.npz")
        )
        preview = ", ".join(available[:10])
        if len(available) > 10:
            preview += ", ..."
        raise FileNotFoundError(
            f"motion sequence {sequence!r} was not found at {motion_path}; "
            f"available sequences: {preview or '(none)'}"
        )
    return replace(
        run_config,
        env_config=replace(run_config.env_config, motion_path=motion_path),
    )


def motion_train_update_loop(
    env: BelloMotionTrackingEnv,
    config: PPOConfig,
    rng: jax.Array,
    video_config: MotionEvalVideoConfig,
    run,
    *,
    checkpoint_root: Path | None = None,
    checkpoint_context: tuple[
        MotionCheckpointMetadata,
        str,
        CheckpointConfig,
    ]
    | None = None,
    initial_runner: RunnerInitialization | None = None,
) -> dict:
    if initial_runner is None:
        runner_state, static_policy, optimizer = init_runner_state(rng, env, config)
    else:
        runner_state, static_policy, optimizer = initial_runner

    if video_config.enabled and video_config.render_initial:
        if checkpoint_root is None or checkpoint_context is None:
            raise ValueError("initial evaluation requires checkpoint context")
        initial_update = int(jax.device_get(runner_state.update))
        initial_checkpoint = checkpoint_root / f"update_{initial_update:06d}"
        metadata, git_commit, _ = checkpoint_context
        save_motion_checkpoint(
            initial_checkpoint,
            runner_state,
            metadata,
            git_commit,
            env.motion,
        )
        render_motion_checkpoint_video(
            initial_checkpoint,
            initial_update,
            video_config,
            run,
            env_steps=0,
        )
    last_metrics = None
    updates = training_updates(
        env, config, (runner_state, static_policy, optimizer), run
    )
    for runner_state, metrics, env_steps in updates:
        update = int(jax.device_get(runner_state.update))
        last_metrics = metrics
        scheduled_checkpoint = None
        if (
            checkpoint_root is not None
            and checkpoint_context is not None
            and should_save_motion_checkpoint(update, checkpoint_context[2])
        ):
            metadata, git_commit, checkpoint_config = checkpoint_context
            scheduled_checkpoint = checkpoint_root / f"update_{update:06d}"
            save_motion_checkpoint(
                scheduled_checkpoint,
                runner_state,
                metadata,
                git_commit,
                env.motion,
            )
            upload_motion_checkpoint(
                run,
                scheduled_checkpoint,
                checkpoint_root,
                checkpoint_config,
            )
        if should_render_training_video(update, config.num_updates, video_config):
            if checkpoint_root is None or checkpoint_context is None:
                raise ValueError("motion evaluation requires checkpoint context")
            evaluation_checkpoint = (
                scheduled_checkpoint
                if scheduled_checkpoint is not None
                else checkpoint_root / f"eval_{update:06d}"
            )
            if scheduled_checkpoint is None:
                metadata, git_commit, _ = checkpoint_context
                save_motion_checkpoint(
                    evaluation_checkpoint,
                    runner_state,
                    metadata,
                    git_commit,
                    env.motion,
                )
            render_motion_checkpoint_video(
                evaluation_checkpoint,
                update,
                video_config,
                run,
                env_steps=env_steps,
            )
    if last_metrics is None:
        raise ValueError("PPO config produced zero updates")
    return {
        "runner_state": runner_state,
        "metrics": last_metrics,
    }


def should_save_motion_checkpoint(update: int, config: CheckpointConfig) -> bool:
    return (
        config.enabled
        and config.every_updates is not None
        and update > 0
        and update % config.every_updates == 0
    )


def upload_motion_checkpoint(
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


def save_motion_checkpoint(
    path: Path,
    runner_state: RunnerState,
    metadata: MotionCheckpointMetadata,
    git_commit: str,
    motion: MotionClip,
) -> None:
    save_training_checkpoint(
        path,
        runner_state,
        metadata,
        git_commit,
        motion=motion,
    )


def motion_checkpoint_metadata(
    run_config: MotionTrainingRunConfig,
    env: BelloMotionTrackingEnv,
    policy_obs_size: int,
    critic_policy_obs_size: int,
) -> MotionCheckpointMetadata:
    return MotionCheckpointMetadata(
        ppo=run_config.config,
        asset=run_config.asset_config,
        warp=run_config.warp,
        obs_size=policy_obs_size,
        critic_obs_size=critic_policy_obs_size,
        action_size=env.action_size,
        actor_observation_layout=env.observation_layout,
        critic_observation_layout=env.critic_observation_layout,
        action_layout=env.action_layout,
        motion_env=replace(
            run_config.env_config,
            motion_path=Path(MOTION_FILE),
        ),
    )


def load_motion_resume(
    path: Path | None,
    env: BelloMotionTrackingEnv,
    expected_metadata: MotionCheckpointMetadata,
) -> RunnerInitialization | None:
    if path is None:
        return None
    runner_state, static_policy, optimizer, _ = restore_runner_checkpoint(
        path, env, expected_metadata
    )
    return runner_state, static_policy, optimizer


def render_motion_checkpoint_video(
    checkpoint: Path,
    update: int,
    config: MotionEvalVideoConfig,
    run,
    *,
    env_steps: int,
) -> Path:
    output_path = config.output_dir / f"{config.filename_prefix}_{update:06d}.mp4"
    stats = evaluate_motion_video(
        MotionEvaluationRequest(
            checkpoint=checkpoint.resolve(),
            output_path=output_path.resolve(),
            steps=config.steps,
            width=config.width,
            height=config.height,
            fps=config.fps,
        )
    )
    log_motion_evaluation(run, output_path, stats, config, update, env_steps)
    return output_path


def log_motion_evaluation(
    run,
    output_path: Path,
    stats,
    config: MotionEvalVideoConfig,
    update: int,
    env_steps: int,
) -> None:
    if not config.log_wandb:
        return
    track_metric_snapshot(
        run,
        {
            "completion_fraction": stats["completion_fraction"],
            "episode_return": stats["episode_return"],
            "root_position_rms_m": stats["root_position_rms_m"],
            "root_orientation_rms_deg": stats["root_orientation_rms_deg"],
            "joint_position_rms_rad": stats["joint_position_rms_rad"],
            "terminal": float(stats["terminal"]),
            "timeout": float(stats["timeout"]),
        },
        env_steps=env_steps,
        update=update,
        context={"phase": "eval"},
    )
    track_video(
        run,
        output_path,
        name=config.wandb_name,
        step=env_steps,
        caption=(
            f"update={update}, rendered_steps={stats['rendered_steps']}, "
            f"episode_return={stats['episode_return']:.3f}, "
            f"completion={stats['completion_fraction']:.3f}, "
            f"root_rms={stats['root_position_rms_m']:.3f}m, "
            f"root_orientation_rms={stats['root_orientation_rms_deg']:.1f}deg, "
            f"joint_rms={stats['joint_position_rms_rad']:.3f}rad, "
            f"terminal={int(stats['terminal'])}, timeout={int(stats['timeout'])}"
        ),
    )


def validate_motion_video_config(config: MotionEvalVideoConfig) -> None:
    if not config.enabled:
        return
    if config.every_updates <= 0:
        raise ValueError("video_config.every_updates must be positive")
    if config.steps <= 0:
        raise ValueError("video_config.steps must be positive")
    if config.width <= 0 or config.height <= 0:
        raise ValueError("video_config width and height must be positive")
    if config.fps <= 0:
        raise ValueError("video_config.fps must be positive")
    if not config.filename_prefix:
        raise ValueError("video_config.filename_prefix must not be empty")
    if not config.wandb_name:
        raise ValueError("video_config.wandb_name must not be empty")


def validate_motion_checkpoint_config(config: CheckpointConfig) -> None:
    validate_checkpoint_options(config)
    if config.every_updates is not None and config.every_updates <= 0:
        raise ValueError("checkpoint_config.every_updates must be positive or null")
    if config.resume_path is not None and not Path(config.resume_path).exists():
        raise FileNotFoundError(config.resume_path)


def _validate_motion_asset(config: ModelConfig) -> None:
    expected = motion_tracking_asset_config(
        asset=config.asset,
        end_effector_payload_mass_kg=config.end_effector_payload_mass_kg,
    )
    if config != expected:
        raise ValueError(
            "motion tracking asset_config must match the selected robot's "
            "canonical all-policy motion asset"
        )
