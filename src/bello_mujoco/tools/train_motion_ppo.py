"""Train PPO on the fixed-waist Bello whole-body tracking environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import hydra
import jax
from omegaconf import DictConfig

from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.checkpoint import (
    CheckpointConfig,
    CheckpointMetadata,
    checkpoint_path,
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
from bello_mujoco.policy import policy_observation_size
from bello_mujoco.sim import WarpConfig
from bello_mujoco.tools.train_ppo import (
    final_train_state,
    final_value_norm_state,
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
    PPOConfig,
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
        collision_mode="primitive",
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
    run = create_wandb_run(run_config.wandb_config) if not run_config.dry_run else None
    succeeded = False
    try:
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
        render_env = (
            BelloMotionTrackingEnv.load(
                run_config.asset_config,
                run_config.env_config,
                warp=run_config.warp,
                batch_size=1,
                use_viewer_model=True,
            )
            if run_config.video_config.enabled
            else env
        )
        out = motion_train_update_loop(
            env,
            render_env,
            run_config.config,
            rng,
            run_config.video_config,
            run,
            run_config.jit,
        )
        checkpoint_dir = None
        if run_config.checkpoint_config.enabled:
            checkpoint_dir = checkpoint_path(
                run_config.checkpoint_config, run_config.config
            )
            final_state = final_train_state(out, env, run_config.config)
            save_training_checkpoint(
                checkpoint_dir,
                final_state,
                CheckpointMetadata(
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
                    value_norm=value_norm_metadata(final_value_norm_state(out)),
                    task="motion_tracking",
                    motion_env=run_config.env_config,
                ),
            )
            if run_config.checkpoint_config.upload_wandb_artifact and run is not None:
                track_checkpoint_artifact(
                    run,
                    checkpoint_dir,
                    name=(
                        run_config.checkpoint_config.wandb_artifact_name
                        or checkpoint_dir.name
                    ),
                    artifact_type=run_config.checkpoint_config.wandb_artifact_type,
                )
        print(out["metrics"])
        if checkpoint_dir is not None:
            print({"checkpoint_path": str(checkpoint_dir)})
        succeeded = True
    finally:
        close_run(run, exit_code=0 if succeeded else 1)


def motion_train_update_loop(
    env: BelloMotionTrackingEnv,
    render_env: BelloMotionTrackingEnv,
    config: PPOConfig,
    rng: jax.Array,
    video_config: MotionEvalVideoConfig,
    run,
    jit: bool,
) -> dict:
    runner_state, static_policy, optimizer = init_runner_state(rng, env, config)

    def update_fn(state):
        return update_runner_state(state, static_policy, optimizer, env, config)

    update_fn = jax.jit(update_fn) if jit else update_fn
    if video_config.render_initial:
        render_motion_training_video(
            render_env,
            runner_state,
            static_policy,
            video_config,
            run,
            env_steps=0,
            obs_stack_size=config.policy.obs_stack_size,
        )
    last_metrics = None
    for _ in training_update_range(config.num_updates):
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
        if should_render_training_video(update, config.num_updates, video_config):
            render_motion_training_video(
                render_env,
                runner_state,
                static_policy,
                video_config,
                run,
                env_steps=env_steps,
                obs_stack_size=config.policy.obs_stack_size,
            )
    if last_metrics is None:
        raise ValueError("PPO config produced zero updates")
    return {"runner_state": runner_state, "metrics": last_metrics}


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
    if config.log_wandb:
        track_video(
            run,
            output_path,
            name=config.wandb_name,
            step=env_steps,
            fps=config.fps,
            caption=(
                f"update={update}, rendered_steps={stats['rendered_steps']}, "
                f"episode_return={stats['episode_return']:.3f}, "
                f"terminal={int(stats['terminal'])}, timeout={int(stats['timeout'])}"
            ),
        )
    return output_path


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
