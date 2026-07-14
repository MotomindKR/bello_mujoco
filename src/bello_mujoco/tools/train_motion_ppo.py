"""Train PPO on the fixed-waist Bello whole-body tracking environment."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import warnings

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.checkpoint import (
    CheckpointConfig,
    CheckpointMetadata,
    checkpoint_path,
    observation_norm_metadata,
    restore_training_checkpoint,
    save_training_checkpoint,
    value_norm_metadata,
)
from bello_mujoco.envs.locomotion import BelloEnvConfig, DoneConfig
from bello_mujoco.envs.motion_tracking import (
    BelloMotionTrackingEnv,
    MotionTrackingEnvConfig,
)
from bello_mujoco.envs.rewards import CommandConfig, RewardConfig
from bello_mujoco.hydra_config import config_path, structured_config
from bello_mujoco.motion_evaluation import (
    MotionEvaluationRequest,
    MotionEvaluationWorker,
)
from bello_mujoco.policy import policy_observation_size
from bello_mujoco.sim import WarpConfig
from bello_mujoco.tools.train_ppo import (
    raise_on_nonfinite_training_metrics,
    should_render_training_video,
    training_update_range,
)
from bello_mujoco.tools.eval_motion_policy import render_motion_policy
from bello_mujoco.tracking import (
    WandbConfig,
    close_run,
    create_wandb_run,
    log_config,
    track_checkpoint_artifact,
    track_metric_snapshot,
    track_video,
)
from bello_mujoco.train.ppo import (
    ObservationNormState,
    PPOConfig,
    RunnerState,
    ValueNormState,
    critic_observation_size,
    effective_learning_rate_schedule,
    init_runner_state,
    init_train_state,
    runner_train_state,
    update_runner_state,
    validate_ppo_config,
)


FIXED_WAIST_JOINTS = ("waist_roll_joint", "waist_pitch_joint")


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
    asset_config: AssetPrepConfig = AssetPrepConfig(
        asset="full_body",
        collision_mode="box",
        end_effector_sphere_mode="both",
        fixed_joint_names=FIXED_WAIST_JOINTS,
    )
    env_config: MotionTrackingEnvConfig = MotionTrackingEnvConfig()
    checkpoint_config: CheckpointConfig = CheckpointConfig()
    wandb_config: WandbConfig = WandbConfig()
    video_config: MotionEvalVideoConfig = MotionEvalVideoConfig()
    warp: WarpConfig = WarpConfig()
    dry_run: bool = True
    jit: bool = True


def run_training(run_config: MotionTrainingRunConfig) -> None:
    validate_ppo_config(run_config.config)
    _validate_fixed_waist_asset(run_config.asset_config)
    validate_motion_video_config(run_config.video_config)
    validate_motion_checkpoint_config(run_config.checkpoint_config)
    if (
        not run_config.dry_run
        and run_config.video_config.enabled
        and not run_config.checkpoint_config.enabled
    ):
        raise ValueError("motion evaluation requires checkpointing to be enabled")
    evaluation_worker = None
    run = None
    succeeded = False
    try:
        evaluation_worker = (
            MotionEvaluationWorker()
            if not run_config.dry_run and run_config.video_config.enabled
            else None
        )
        run = (
            create_wandb_run(run_config.wandb_config)
            if not run_config.dry_run
            else None
        )
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
        rng = jax.random.PRNGKey(run_config.config.seed)
        if run_config.dry_run:
            train_state, _ = init_train_state(
                rng,
                policy_obs_size,
                env.action_size,
                run_config.config,
                critic_obs_size=critic_policy_obs_size,
            )
            state, observation = env.reset_batch(
                run_config.config.num_envs, jax.random.PRNGKey(run_config.config.seed)
            )
            print(
                {
                    "task": "motion_tracking",
                    "backend": jax.default_backend(),
                    "motion_path": str(run_config.env_config.motion_path),
                    "motion_frames": env.motion.num_frames,
                    "motion_duration": env.motion.duration,
                    "sim_timestep": env.model.mj_model.opt.timestep,
                    "control_timestep": env.control_timestep,
                    "obs_size": env.observation_size,
                    "policy_obs_size": policy_obs_size,
                    "critic_obs_size": critic_obs_size,
                    "critic_policy_obs_size": critic_policy_obs_size,
                    "action_size": env.action_size,
                    "num_envs": run_config.config.num_envs,
                    "num_steps": run_config.config.num_steps,
                    "num_updates": run_config.config.num_updates,
                    "batch_obs_shape": tuple(observation.shape),
                    "batch_action_shape": tuple(state.last_action.shape),
                    "fixed_joint_names": run_config.asset_config.fixed_joint_names,
                    "sphere_mode": run_config.asset_config.end_effector_sphere_mode,
                    "learning_rate_schedule": effective_learning_rate_schedule(
                        run_config.config
                    ),
                    "eval_video_enabled": run_config.video_config.enabled,
                    "eval_video_wandb_name": run_config.video_config.wandb_name,
                    "params_leaves": len(jax.tree.leaves(train_state.policy)),
                }
            )
            succeeded = True
            return

        log_config(
            run,
            {
                "task": "motion_tracking",
                "ppo": run_config.config,
                "asset": run_config.asset_config,
                "env": run_config.env_config,
                "checkpoint": run_config.checkpoint_config,
                "video": run_config.video_config,
                "warp": run_config.warp,
            },
        )
        checkpoint_root = (
            checkpoint_path(run_config.checkpoint_config, run_config.config)
            if run_config.checkpoint_config.enabled
            else None
        )
        resume = load_motion_resume(
            run_config.checkpoint_config.resume_path,
            run_config,
            env,
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
            run_config.jit,
            checkpoint_root=checkpoint_root,
            checkpoint_context=(
                run_config,
                policy_obs_size,
                critic_policy_obs_size,
            ),
            resume=resume,
        )
        final_checkpoint = None
        if checkpoint_root is not None:
            final_checkpoint = checkpoint_root / "final"
            runner_state = out["runner_state"]
            save_motion_checkpoint(
                final_checkpoint,
                runner_state,
                out["static_policy"],
                run_config,
                env,
                policy_obs_size,
                critic_policy_obs_size,
            )
            if run_config.checkpoint_config.upload_wandb_artifact and run is not None:
                track_checkpoint_artifact(
                    run,
                    final_checkpoint,
                    name=(
                        run_config.checkpoint_config.wandb_artifact_name
                        or checkpoint_root.name
                    ),
                    artifact_type=run_config.checkpoint_config.wandb_artifact_type,
                )
        print(out["metrics"])
        if final_checkpoint is not None:
            print({"checkpoint_path": str(final_checkpoint)})
        succeeded = True
    finally:
        if evaluation_worker is not None:
            evaluation_worker.close()
        close_run(run, exit_code=0 if succeeded else 1)


def motion_train_update_loop(
    env: BelloMotionTrackingEnv,
    evaluation_worker: MotionEvaluationWorker | None,
    config: PPOConfig,
    rng: jax.Array,
    video_config: MotionEvalVideoConfig,
    run,
    jit: bool,
    *,
    checkpoint_root: Path | None = None,
    checkpoint_context: tuple[MotionTrainingRunConfig, int, int] | None = None,
    resume: tuple[object, CheckpointMetadata] | None = None,
) -> dict:
    runner_state, static_policy, optimizer = init_runner_state(rng, env, config)
    if resume is not None:
        runner_state = resume_runner_state(runner_state, resume[0], resume[1])

    def update_fn(state):
        return update_runner_state(state, static_policy, optimizer, env, config)

    update_fn = jax.jit(update_fn) if jit else update_fn
    if video_config.render_initial:
        if checkpoint_root is None or checkpoint_context is None:
            raise ValueError("initial evaluation requires checkpoint context")
        initial_update = int(jax.device_get(runner_state.update))
        initial_checkpoint = checkpoint_root / f"update_{initial_update:06d}"
        run_config, policy_obs_size, critic_policy_obs_size = checkpoint_context
        save_motion_checkpoint(
            initial_checkpoint,
            runner_state,
            static_policy,
            run_config,
            env,
            policy_obs_size,
            critic_policy_obs_size,
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
    remaining_updates = config.num_updates - start_update
    if remaining_updates <= 0:
        raise ValueError(
            f"checkpoint update {start_update} has reached configured "
            f"num_updates={config.num_updates}"
        )
    for _ in training_update_range(remaining_updates):
        runner_state, metrics = update_fn(runner_state)
        metrics = jax.tree_util.tree_map(
            lambda value: value.block_until_ready(), metrics
        )
        update = int(jax.device_get(runner_state.update))
        raise_on_nonfinite_training_metrics(metrics, update)
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
            and should_save_motion_checkpoint(
                update, checkpoint_context[0].checkpoint_config
            )
        ):
            run_config, policy_obs_size, critic_policy_obs_size = checkpoint_context
            scheduled_checkpoint = checkpoint_root / f"update_{update:06d}"
            save_motion_checkpoint(
                scheduled_checkpoint,
                runner_state,
                static_policy,
                run_config,
                env,
                policy_obs_size,
                critic_policy_obs_size,
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
                run_config, policy_obs_size, critic_policy_obs_size = checkpoint_context
                save_motion_checkpoint(
                    evaluation_checkpoint,
                    runner_state,
                    static_policy,
                    run_config,
                    env,
                    policy_obs_size,
                    critic_policy_obs_size,
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
        "static_policy": static_policy,
        "metrics": last_metrics,
    }


def should_save_motion_checkpoint(update: int, config: CheckpointConfig) -> bool:
    return (
        config.enabled
        and config.every_updates is not None
        and update > 0
        and update % config.every_updates == 0
    )


def save_motion_checkpoint(
    path: Path,
    runner_state: RunnerState,
    static_policy,
    run_config: MotionTrainingRunConfig,
    env: BelloMotionTrackingEnv,
    policy_obs_size: int,
    critic_policy_obs_size: int,
) -> None:
    save_training_checkpoint(
        path,
        runner_train_state(runner_state, static_policy),
        motion_checkpoint_metadata(
            runner_state,
            run_config,
            env,
            policy_obs_size,
            critic_policy_obs_size,
        ),
    )


def motion_checkpoint_metadata(
    runner_state: RunnerState,
    run_config: MotionTrainingRunConfig,
    env: BelloMotionTrackingEnv,
    policy_obs_size: int,
    critic_policy_obs_size: int,
) -> CheckpointMetadata:
    return CheckpointMetadata(
        ppo=run_config.config,
        asset=run_config.asset_config,
        reward=RewardConfig(),
        command=CommandConfig(enabled=False),
        done=DoneConfig(),
        env=BelloEnvConfig(),
        warp=run_config.warp,
        obs_size=policy_obs_size,
        critic_obs_size=critic_policy_obs_size,
        action_size=env.action_size,
        value_norm=value_norm_metadata(runner_state.value_norm_state),
        observation_norm=observation_norm_metadata(runner_state.observation_norm_state),
        critic_observation_norm=observation_norm_metadata(
            runner_state.critic_observation_norm_state
        ),
        task="motion_tracking",
        motion_env=run_config.env_config,
        motion_source_sha256=env.motion.source_sha256,
        rng=tuple(int(value) for value in jax.device_get(runner_state.rng)),
        learning_rate=float(jax.device_get(runner_state.learning_rate)),
    )


def load_motion_resume(
    path: Path | None,
    run_config: MotionTrainingRunConfig,
    env: BelloMotionTrackingEnv,
    policy_obs_size: int,
    critic_policy_obs_size: int,
):
    if path is None:
        return None
    train_state, metadata = restore_training_checkpoint(path)
    validate_motion_resume(
        metadata,
        run_config,
        env,
        policy_obs_size,
        critic_policy_obs_size,
    )
    return train_state, metadata


def validate_motion_resume(
    metadata: CheckpointMetadata,
    run_config: MotionTrainingRunConfig,
    env: BelloMotionTrackingEnv,
    policy_obs_size: int,
    critic_policy_obs_size: int,
) -> None:
    if metadata.task != "motion_tracking" or metadata.motion_env is None:
        raise ValueError("resume checkpoint is not a motion-tracking checkpoint")
    if metadata.asset != run_config.asset_config:
        raise ValueError("resume checkpoint asset config does not match this run")
    if metadata.motion_env != run_config.env_config:
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
    ):
        raise ValueError("resume checkpoint policy dimensions do not match this run")
    if metadata.motion_source_sha256 != env.motion.source_sha256:
        raise ValueError("resume checkpoint was trained on a different motion export")


def resume_runner_state(
    runner_state: RunnerState,
    train_state,
    metadata: CheckpointMetadata,
) -> RunnerState:
    params, _ = eqx.partition(train_state.policy, eqx.is_array)
    value_norm = runner_state.value_norm_state
    if metadata.value_norm is not None:
        value_norm = ValueNormState(
            mean=jnp.asarray(metadata.value_norm.mean, dtype=jnp.float32),
            var=jnp.asarray(metadata.value_norm.var, dtype=jnp.float32),
            count=jnp.asarray(metadata.value_norm.count, dtype=jnp.float32),
        )
    observation_norm = restore_observation_norm_state(
        runner_state.observation_norm_state, metadata.observation_norm
    )
    critic_observation_norm = restore_observation_norm_state(
        runner_state.critic_observation_norm_state,
        metadata.critic_observation_norm,
    )
    rng = (
        jnp.asarray(metadata.rng, dtype=jnp.uint32)
        if metadata.rng is not None
        else runner_state.rng
    )
    learning_rate = (
        jnp.asarray(metadata.learning_rate, dtype=jnp.float32)
        if metadata.learning_rate is not None
        else runner_state.learning_rate
    )
    return runner_state._replace(
        params=params,
        opt_state=train_state.opt_state,
        update=train_state.update,
        rng=rng,
        learning_rate=learning_rate,
        value_norm_state=value_norm,
        observation_norm_state=observation_norm,
        critic_observation_norm_state=critic_observation_norm,
    )


def restore_observation_norm_state(state, metadata):
    if metadata is None:
        return state
    mean = jnp.asarray(metadata.mean, dtype=jnp.float32)
    var = jnp.asarray(metadata.var, dtype=jnp.float32)
    if mean.shape != state.mean.shape or var.shape != state.var.shape:
        raise ValueError("checkpoint observation normalization shape does not match")
    return ObservationNormState(
        mean=mean,
        var=var,
        count=jnp.asarray(metadata.count, dtype=jnp.float32),
    )


def render_motion_training_video(
    env: BelloMotionTrackingEnv,
    runner_state,
    static_policy,
    config: MotionEvalVideoConfig,
    run,
    *,
    env_steps: int,
    obs_stack_size: int,
) -> Path:
    update = int(jax.device_get(runner_state.update))
    output_path = config.output_dir / f"{config.filename_prefix}_{update:06d}.mp4"
    train_state = runner_train_state(runner_state, static_policy)
    stats = render_motion_policy(
        env,
        train_state.policy,
        output_path,
        steps=config.steps,
        width=config.width,
        height=config.height,
        fps=config.fps,
        obs_stack_size=obs_stack_size,
    )
    log_motion_evaluation(run, output_path, stats, config, update, env_steps)
    return output_path


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
        fps=config.fps,
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


def try_render_motion_training_video(*args, **kwargs) -> Path | None:
    update = int(jax.device_get(args[1].update))
    try:
        return render_motion_training_video(*args, **kwargs)
    except Exception as error:  # noqa: BLE001 - evaluation is noncritical
        warnings.warn(
            f"motion evaluation failed at update {update}: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


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
    if config.every_updates is not None and config.every_updates <= 0:
        raise ValueError("checkpoint_config.every_updates must be positive or null")
    if config.resume_path is not None and not Path(config.resume_path).exists():
        raise FileNotFoundError(config.resume_path)


def _validate_fixed_waist_asset(config: AssetPrepConfig) -> None:
    if tuple(config.fixed_joint_names) != FIXED_WAIST_JOINTS:
        raise ValueError(
            f"motion tracking requires fixed_joint_names={FIXED_WAIST_JOINTS}"
        )
    if config.end_effector_sphere_mode != "both":
        raise ValueError("motion tracking requires end_effector_sphere_mode='both'")


def entrypoint() -> None:
    hydra_entrypoint()


@hydra.main(
    version_base=None,
    config_path=config_path("train_motion"),
    config_name="default",
)
def hydra_entrypoint(config: DictConfig) -> None:
    run_training(structured_config(MotionTrainingRunConfig, config))


if __name__ == "__main__":
    entrypoint()
