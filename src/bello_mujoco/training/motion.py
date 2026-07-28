"""Train PPO on the fixed-waist, fixed-head Bello tracking environment."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import warnings

import jax

from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.training.checkpoint import (
    CheckpointConfig,
    CheckpointMetadata,
    MotionCheckpointMetadata,
    checkpoint_path,
    load_checkpoint_metadata,
    motion_clip_contract,
    motion_metadata_with_runner_state,
    restore_runner_checkpoint,
    save_training_checkpoint,
    validate_checkpoint_options,
    validate_motion_clip_contract,
)
from bello_mujoco.envs.motion_tracking import (
    BelloMotionTrackingEnv,
    MotionTrackingEnvConfig,
)
from bello_mujoco.evaluation.motion import (
    MotionEvaluationRequest,
    MotionEvaluationWorker,
)
from bello_mujoco.motion.clip import motion_tracking_asset_config
from bello_mujoco.policy import policy_observation_size
from bello_mujoco.sim import WarpConfig
from bello_mujoco.training.runtime import (
    RunnerInitialization,
    raise_on_invalid_training_metrics,
    should_render_training_video,
    training_update_range,
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
from bello_mujoco.training.ppo import (
    PPOConfig,
    RunnerState,
    critic_observation_size,
    init_runner_state,
    update_runner_state,
    validate_action_limit_contract,
    validate_ppo_config,
)
from bello_mujoco.repository import (
    capture_repository_commit,
    require_checkpoint_output_directory,
    require_tracked_repository_path,
    resolve_repository_path,
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
    config: PPOConfig = PPOConfig()
    asset_config: AssetPrepConfig = field(default_factory=motion_tracking_asset_config)
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
    _validate_fixed_tracking_asset(run_config.asset_config)
    validate_motion_video_config(run_config.video_config)
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
        tracked_motion = require_tracked_repository_path(
            run_config.env_config.motion_path
        )
        run_config = replace(
            run_config,
            env_config=replace(
                run_config.env_config,
                motion_path=resolve_repository_path(tracked_motion),
            ),
        )
    evaluation_worker = None
    run = None
    succeeded = False
    try:
        evaluation_worker = (
            MotionEvaluationWorker() if run_config.video_config.enabled else None
        )
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
        critic_obs_size = critic_observation_size(env)
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
            run_config,
            env,
            metadata,
            policy_obs_size,
            critic_policy_obs_size,
        )
        out = motion_train_update_loop(
            env,
            evaluation_worker,
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
        if evaluation_worker is not None:
            evaluation_worker.close()
        close_run(run, exit_code=0 if succeeded else 1)


def resolve_motion_selection(
    run_config: MotionTrainingRunConfig,
) -> MotionTrainingRunConfig:
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
    evaluation_worker: MotionEvaluationWorker | None,
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

    def update_fn(state):
        return update_runner_state(state, static_policy, optimizer, env, config)

    update_fn = jax.jit(update_fn)
    if video_config.render_initial:
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
        )
        try_render_motion_checkpoint_video(
            evaluation_worker,
            initial_checkpoint,
            initial_update,
            video_config,
            run,
            env_steps=0,
        )
    last_metrics = None
    start_update = int(jax.device_get(runner_state.update))
    if not 0 <= start_update < config.num_updates:
        raise ValueError(
            "runner update must be within the unfinished training range, got "
            f"update={start_update}, num_updates={config.num_updates}"
        )
    for _ in training_update_range(start_update, config.num_updates):
        runner_state, metrics = update_fn(runner_state)
        metrics = jax.tree_util.tree_map(
            lambda value: value.block_until_ready(), metrics
        )
        update = int(jax.device_get(runner_state.update))
        raise_on_invalid_training_metrics(metrics, update)
        last_metrics = metrics
        env_steps = update * config.num_envs * config.num_steps
        track_metric_snapshot(
            run,
            metrics,
            env_steps=env_steps,
            update=update,
            context={"phase": "train"},
        )
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
                )
            try_render_motion_checkpoint_video(
                evaluation_worker,
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
) -> None:
    save_training_checkpoint(
        path,
        runner_state,
        motion_metadata_with_runner_state(metadata, runner_state),
        git_commit,
    )


def motion_checkpoint_metadata(
    run_config: MotionTrainingRunConfig,
    env: BelloMotionTrackingEnv,
    policy_obs_size: int,
    critic_policy_obs_size: int,
) -> MotionCheckpointMetadata:
    if run_config.asset_config.output_root is not None:
        raise ValueError(
            "training requires asset_config.output_root=null so the exact source "
            "revision owns asset generation"
        )
    motion_path = run_config.env_config.motion_path
    if run_config.checkpoint_config.enabled:
        motion_path = require_tracked_repository_path(motion_path)
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
            motion_path=motion_path,
        ),
        motion=replace(
            motion_clip_contract(
                run_config.env_config.motion_path,
                env.motion,
            ),
            path=motion_path,
        ),
    )


def load_motion_resume(
    path: Path | None,
    run_config: MotionTrainingRunConfig,
    env: BelloMotionTrackingEnv,
    expected_metadata: MotionCheckpointMetadata,
    policy_obs_size: int,
    critic_policy_obs_size: int,
) -> RunnerInitialization | None:
    if path is None:
        return None
    metadata = load_checkpoint_metadata(
        path,
        allow_git_commit_mismatch=(
            run_config.checkpoint_config.allow_git_commit_mismatch
        ),
    )
    validate_motion_resume(
        metadata,
        run_config,
        env,
        expected_metadata,
        policy_obs_size,
        critic_policy_obs_size,
    )
    runner_state, static_policy, optimizer, restored_metadata = (
        restore_runner_checkpoint(
            path,
            env,
            expected_metadata,
            allow_git_commit_mismatch=(
                run_config.checkpoint_config.allow_git_commit_mismatch
            ),
        )
    )
    if restored_metadata != expected_metadata:
        raise RuntimeError("resume did not resolve to the requested run")
    return runner_state, static_policy, optimizer


def validate_motion_resume(
    metadata: CheckpointMetadata,
    run_config: MotionTrainingRunConfig,
    env: BelloMotionTrackingEnv,
    expected_metadata: MotionCheckpointMetadata,
    policy_obs_size: int,
    critic_policy_obs_size: int,
) -> None:
    if not isinstance(metadata, MotionCheckpointMetadata):
        raise ValueError("resume checkpoint is not a motion-tracking checkpoint")
    if metadata.asset != expected_metadata.asset:
        raise ValueError("resume checkpoint asset config does not match this run")
    if metadata.warp != run_config.warp:
        raise ValueError("resume checkpoint Warp config does not match this run")
    expected_motion_env = replace(
        run_config.env_config,
        motion_path=metadata.motion_env.motion_path,
    )
    if metadata.motion_env != expected_motion_env:
        raise ValueError("resume checkpoint motion environment does not match this run")
    expected_ppo = replace(
        run_config.config, total_timesteps=metadata.ppo.total_timesteps
    )
    if metadata.ppo != expected_ppo:
        raise ValueError("resume checkpoint PPO config differs beyond total_timesteps")
    if (
        metadata.obs_size != policy_obs_size
        or metadata.critic_obs_size != critic_policy_obs_size
        or metadata.action_size != env.action_size
        or metadata.actor_observation_layout != env.observation_layout
        or metadata.critic_observation_layout != env.critic_observation_layout
        or metadata.action_layout != env.action_layout
    ):
        raise ValueError("resume checkpoint policy contract does not match this run")
    validate_motion_clip_contract(
        run_config.env_config.motion_path,
        env.motion,
        metadata.motion,
    )


def render_motion_checkpoint_video(
    worker: MotionEvaluationWorker,
    checkpoint: Path,
    update: int,
    config: MotionEvalVideoConfig,
    run,
    *,
    env_steps: int,
) -> Path:
    output_path = config.output_dir / f"{config.filename_prefix}_{update:06d}.mp4"
    stats = worker.evaluate(
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


def try_render_motion_checkpoint_video(*args, **kwargs) -> Path | None:
    update = int(args[2])
    if args[0] is None:
        warnings.warn(
            f"motion evaluation worker is unavailable at update {update}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    try:
        return render_motion_checkpoint_video(*args, **kwargs)
    except Exception as error:  # noqa: BLE001 - evaluation is noncritical
        warnings.warn(
            f"isolated motion evaluation failed at update {update}: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


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


def _validate_fixed_tracking_asset(config: AssetPrepConfig) -> None:
    expected = motion_tracking_asset_config(output_root=config.output_root)
    if config.collision_mode != "box":
        raise ValueError("motion tracking requires collision_mode='box'")
    if config.joint_availability != expected.joint_availability:
        raise ValueError(
            "motion tracking requires fixed waist roll/pitch and neck/head joints"
        )
    if config.end_effector_sphere_mode != "both":
        raise ValueError("motion tracking requires end_effector_sphere_mode='both'")
    if config.actuator_config != expected.actuator_config:
        raise ValueError(
            "motion tracking requires the canonical all-policy actuator contract"
        )
    if config.ankle_differential_config != expected.ankle_differential_config:
        raise ValueError(
            "motion tracking requires the canonical differential-ankle contract"
        )
