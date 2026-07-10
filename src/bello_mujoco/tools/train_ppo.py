from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
    save_training_checkpoint,
    value_norm_metadata,
)
from bello_mujoco.envs.locomotion import BelloEnvConfig, BelloLocomotionEnv, DoneConfig
from bello_mujoco.envs.rewards import CommandConfig, RewardConfig
from bello_mujoco.hydra_config import config_path, structured_config
from bello_mujoco.policy import policy_observation_size
from bello_mujoco.render import CameraConfig, render_policy_rollout
from bello_mujoco.sim import (
    WarpConfig,
)
from bello_mujoco.tracking import (
    WandbConfig,
    close_run,
    create_wandb_run,
    log_config,
    track_metric_history,
    track_metric_snapshot,
    track_checkpoint_artifact,
    track_video,
)
from bello_mujoco.train.ppo import (
    PPOConfig,
    RunnerState,
    TrainState,
    critic_observation_size,
    effective_learning_rate_schedule,
    init_runner_state,
    init_train_state,
    make_train,
    runner_train_state,
    update_runner_state,
    validate_ppo_config,
)


@dataclass(frozen=True)
class TrainingVideoConfig:
    enabled: bool = False
    every_updates: int = 100
    render_initial: bool = False
    render_final: bool = True
    simulation_backend: Literal["mjx", "mujoco"] = "mujoco"
    output_dir: Path = Path("outputs/train_videos")
    filename_prefix: str = "rollout"
    command: tuple[float, float, float] = (0.0, 0.0, 0.0)
    steps: int = 250
    width: int = 640
    height: int = 480
    fps: int = 50
    stop_on_done: bool = True
    stochastic_actions: bool = False
    action_seed: int = 0
    log_wandb: bool = True
    wandb_name: str = "train/rollout"
    camera_config: CameraConfig = CameraConfig()


@dataclass(frozen=True)
class TrainingMetricConfig:
    stream_updates: bool = True


@dataclass(frozen=True)
class TrainingRunConfig:
    config: PPOConfig = PPOConfig()
    asset_config: AssetPrepConfig = AssetPrepConfig()
    reward_config: RewardConfig = RewardConfig()
    command_config: CommandConfig = CommandConfig()
    done_config: DoneConfig = DoneConfig()
    env_config: BelloEnvConfig = BelloEnvConfig()
    checkpoint_config: CheckpointConfig = CheckpointConfig()
    wandb_config: WandbConfig = WandbConfig()
    metric_config: TrainingMetricConfig = TrainingMetricConfig()
    video_config: TrainingVideoConfig = TrainingVideoConfig()
    warp: WarpConfig = WarpConfig()
    dry_run: bool = True
    jit: bool = True


def main(
    config: PPOConfig = PPOConfig(),
    asset_config: AssetPrepConfig = AssetPrepConfig(),
    reward_config: RewardConfig = RewardConfig(),
    command_config: CommandConfig = CommandConfig(),
    done_config: DoneConfig = DoneConfig(),
    env_config: BelloEnvConfig = BelloEnvConfig(),
    checkpoint_config: CheckpointConfig = CheckpointConfig(),
    wandb_config: WandbConfig = WandbConfig(),
    metric_config: TrainingMetricConfig = TrainingMetricConfig(),
    video_config: TrainingVideoConfig = TrainingVideoConfig(),
    warp: WarpConfig = WarpConfig(),
    dry_run: bool = True,
    jit: bool = True,
) -> None:
    run_training(
        TrainingRunConfig(
            config=config,
            asset_config=asset_config,
            reward_config=reward_config,
            command_config=command_config,
            done_config=done_config,
            env_config=env_config,
            checkpoint_config=checkpoint_config,
            wandb_config=wandb_config,
            metric_config=metric_config,
            video_config=video_config,
            warp=warp,
            dry_run=dry_run,
            jit=jit,
        )
    )


def run_training(run_config: TrainingRunConfig) -> None:
    config = run_config.config
    asset_config = run_config.asset_config
    reward_config = run_config.reward_config
    command_config = run_config.command_config
    done_config = run_config.done_config
    env_config = run_config.env_config
    checkpoint_config = run_config.checkpoint_config
    wandb_config = run_config.wandb_config
    metric_config = run_config.metric_config
    video_config = run_config.video_config
    warp = run_config.warp
    dry_run = run_config.dry_run
    jit = run_config.jit

    validate_ppo_config(config)
    validate_video_config(video_config)
    env = BelloLocomotionEnv.load(
        asset_config,
        warp=warp,
        batch_size=config.num_envs,
        env_config=env_config,
        reward_config=reward_config,
        command_config=command_config,
        done_config=done_config,
    )
    render_env = (
        BelloLocomotionEnv.load(
            asset_config,
            warp=warp,
            batch_size=1,
            env_config=env_config,
            reward_config=reward_config,
            command_config=command_config,
            done_config=done_config,
            use_viewer_model=True,
        )
        if video_config.enabled
        else env
    )
    rng = jax.random.PRNGKey(config.seed)
    policy_obs_size = policy_observation_size(env.observation_size, config.policy)
    critic_obs_size = critic_observation_size(env)
    critic_policy_obs_size = policy_observation_size(critic_obs_size, config.policy)
    if dry_run:
        state, _ = init_train_state(
            rng,
            policy_obs_size,
            env.action_size,
            config,
            critic_obs_size=critic_policy_obs_size,
        )
        batch_state, batch_obs = env.reset_batch(config.num_envs)
        print(
            {
                "backend": jax.default_backend(),
                "mjx_impl": "warp",
                "terrain": env_config.terrain.mode,
                "sim_timestep": env.model.mj_model.opt.timestep,
                "decimation": env_config.decimation,
                "control_timestep": env.control_timestep,
                "max_episode_steps": env_config.max_episode_steps,
                "warp_naconmax": env.model.warp_naconmax,
                "warp_njmax": env.model.warp_njmax,
                "obs_size": env.observation_size,
                "policy_obs_size": policy_obs_size,
                "critic_obs_size": critic_obs_size,
                "critic_policy_obs_size": critic_policy_obs_size,
                "obs_stack_size": config.policy.obs_stack_size,
                "action_size": env.action_size,
                "num_envs": config.num_envs,
                "num_steps": config.num_steps,
                "num_updates": config.num_updates,
                "policy_hidden_dims": config.policy.hidden_dims,
                "policy_activation": config.policy.activation,
                "command_ranges": tuple(
                    tuple(row) for row in command_config.range_array().tolist()
                ),
                "command_curriculum_enabled": command_config.curriculum.enabled,
                "random_push_enabled": env_config.random_push.enabled,
                "observation_noise_enabled": env_config.observation.noise.enabled,
                "learning_rate": config.learning_rate,
                "learning_rate_schedule": effective_learning_rate_schedule(config),
                "desired_kl": config.desired_kl,
                "target_kl": config.target_kl,
                "ent_coef": config.ent_coef,
                "normalize_value": config.normalize_value,
                "value_norm_eps": config.value_norm_eps,
                "batch_obs_shape": tuple(batch_obs.shape),
                "batch_action_shape": tuple(batch_state.last_action.shape),
                "checkpoint_enabled": checkpoint_config.enabled,
                "wandb_enabled": wandb_config.enabled,
                "video_enabled": video_config.enabled,
                "params_leaves": len(jax.tree.leaves(state.policy)),
            }
        )
        return

    run = create_wandb_run(wandb_config)
    checkpoint_dir = (
        checkpoint_path(checkpoint_config, config)
        if checkpoint_config.enabled
        else None
    )
    try:
        log_config(
            run,
            {
                "ppo": config,
                "asset": asset_config,
                "reward": reward_config,
                "command": command_config,
                "done": done_config,
                "env_config": env_config,
                "checkpoint": checkpoint_config,
                "metric": metric_config,
                "video": video_config,
                "warp": warp,
                "env": {
                    "obs_size": env.observation_size,
                    "policy_obs_size": policy_obs_size,
                    "critic_obs_size": critic_obs_size,
                    "critic_policy_obs_size": critic_policy_obs_size,
                    "obs_stack_size": config.policy.obs_stack_size,
                    "action_size": env.action_size,
                    "mjx_impl": "warp",
                    "xml_path": env.model.xml_path,
                    "checkpoint_path": checkpoint_dir,
                },
            },
        )
        stream_metrics = metric_config.stream_updates
        if video_config.enabled or stream_metrics:
            out = train_update_loop(
                env,
                render_env,
                config,
                rng,
                video_config,
                run,
                jit,
                stream_metrics=stream_metrics,
            )
        else:
            train = make_train(config, env)
            train = jax.jit(train) if jit else train
            out = train(rng)

        metrics = jax.tree_util.tree_map(
            lambda x: x[-1] if getattr(x, "ndim", 0) > 0 else x, out["metrics"]
        )
        if checkpoint_config.enabled:
            final_state = final_train_state(out, env, config)
            save_training_checkpoint(
                checkpoint_dir,
                final_state,
                CheckpointMetadata(
                    ppo=config,
                    asset=asset_config,
                    reward=reward_config,
                    command=command_config,
                    done=done_config,
                    env=env_config,
                    warp=warp,
                    obs_size=policy_obs_size,
                    critic_obs_size=critic_policy_obs_size,
                    action_size=env.action_size,
                    value_norm=value_norm_metadata(final_value_norm_state(out)),
                ),
                overwrite=checkpoint_config.overwrite,
            )
            if checkpoint_config.upload_wandb_artifact and run is not None:
                track_checkpoint_artifact(
                    run,
                    checkpoint_dir,
                    name=checkpoint_config.wandb_artifact_name or checkpoint_dir.name,
                    artifact_type=checkpoint_config.wandb_artifact_type,
                )
        if not stream_metrics and run is not None:
            track_metric_history(
                run,
                out["metrics"],
                samples_per_update=config.num_envs * config.num_steps,
                context={"phase": "train"},
            )
        print(metrics)
        if checkpoint_dir is not None:
            print({"checkpoint_path": str(checkpoint_dir)})
    finally:
        close_run(run)


def train_update_loop(
    env: BelloLocomotionEnv,
    render_env: BelloLocomotionEnv,
    config: PPOConfig,
    rng: jax.Array,
    video_config: TrainingVideoConfig,
    run,
    jit: bool,
    *,
    stream_metrics: bool,
) -> dict:
    runner_state, static_policy, optimizer = init_runner_state(rng, env, config)

    def update_fn(state):
        return update_runner_state(state, static_policy, optimizer, env, config)

    update_fn = jax.jit(update_fn) if jit else update_fn
    metrics_history = []
    if video_config.render_initial:
        render_training_video(
            render_env,
            runner_state,
            static_policy,
            video_config,
            run,
            env_steps=0,
            obs_stack_size=config.policy.obs_stack_size,
        )
    for _ in range(config.num_updates):
        runner_state, metrics = update_fn(runner_state)
        metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
        update = int(jax.device_get(runner_state.update))
        raise_on_nonfinite_training_metrics(metrics, update)
        metrics_history.append(metrics)
        env_steps = update * config.num_envs * config.num_steps
        if stream_metrics and run is not None:
            track_metric_snapshot(
                run,
                metrics,
                env_steps=env_steps,
                update=update,
                context={"phase": "train"},
            )
        if should_render_training_video(update, config.num_updates, video_config):
            render_training_video(
                render_env,
                runner_state,
                static_policy,
                video_config,
                run,
                env_steps=env_steps,
                obs_stack_size=config.policy.obs_stack_size,
            )
    return {
        "runner_state": runner_state,
        "metrics": stack_metric_history(metrics_history),
    }


def should_render_training_video(
    update: int, num_updates: int, config: TrainingVideoConfig
) -> bool:
    if not config.enabled:
        return False
    if update <= 0:
        return False
    if config.render_final and update == num_updates:
        return True
    return update % config.every_updates == 0


def render_training_video(
    env: BelloLocomotionEnv,
    runner_state: RunnerState,
    static_policy,
    config: TrainingVideoConfig,
    run,
    *,
    env_steps: int,
    obs_stack_size: int = 1,
) -> Path:
    update = int(jax.device_get(runner_state.update))
    output_path = config.output_dir / f"{config.filename_prefix}_{update:06d}.mp4"
    train_state = runner_train_state(runner_state, static_policy)
    rollout_stats = render_policy_rollout(
        env,
        train_state.policy,
        output_path=output_path,
        command=config.command,
        steps=config.steps,
        width=config.width,
        height=config.height,
        fps=config.fps,
        stop_on_done=config.stop_on_done,
        camera_config=config.camera_config,
        simulation_backend=config.simulation_backend,
        obs_stack_size=obs_stack_size,
        stochastic_actions=config.stochastic_actions,
        action_seed=config.action_seed,
    )
    if config.log_wandb:
        track_video(
            run,
            output_path,
            name=config.wandb_name,
            step=env_steps,
            fps=config.fps,
            caption=video_rollout_caption(update, rollout_stats),
        )
    return output_path


def video_rollout_caption(update: int, stats: dict[str, float | int]) -> str:
    first_done_step = int(stats.get("first_done_step", -1))
    completed = int(stats.get("completed_episodes", 0))
    if first_done_step < 0:
        return f"update={update}, no_done, completed_episodes={completed}"
    first_done_seconds = float(stats.get("first_done_seconds", -1.0))
    terminal = int(stats.get("done/terminal", 0))
    timeout = int(stats.get("done/timeout", 0))
    return (
        f"update={update}, first_done_step={first_done_step}, "
        f"first_done_seconds={first_done_seconds:.2f}, "
        f"completed_episodes={completed}, terminal={terminal}, timeout={timeout}"
    )


def stack_metric_history(metrics_history: list[dict[str, jax.Array]]) -> dict:
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *metrics_history)


def raise_on_nonfinite_training_metrics(metrics: dict, update: int) -> None:
    nonfinite_update = metrics.get("nonfinite_update")
    if nonfinite_update is not None and float(jax.device_get(nonfinite_update)) > 0.0:
        raise RuntimeError(
            "PPO produced a nonfinite update at update "
            f"{update}; skipped the poisoned optimizer step. "
            "Lower learning_rate/init_log_std or inspect rewards/observations."
        )
    checked_keys = (
        "loss",
        "actor_loss",
        "value_loss",
        "value_loss_raw",
        "entropy",
        "approx_kl",
        "action_std_mean",
        "action_std_min",
        "action_std_max",
    )
    for name in checked_keys:
        value = metrics.get(name)
        if value is None:
            continue
        if not bool(jnp.all(jnp.isfinite(jax.device_get(value)))):
            raise RuntimeError(
                f"PPO metric {name!r} became nonfinite at update {update}; "
                "stopping training instead of continuing with invalid policy state."
            )


def validate_video_config(config: TrainingVideoConfig) -> None:
    if not config.enabled:
        return
    if config.every_updates <= 0:
        raise ValueError(
            f"video_config.every_updates must be positive, got {config.every_updates}"
        )
    if config.steps <= 0:
        raise ValueError(f"video_config.steps must be positive, got {config.steps}")
    if config.width <= 0 or config.height <= 0:
        raise ValueError(
            "video_config.width and video_config.height must be positive, got "
            f"{config.width}x{config.height}"
        )
    if config.fps <= 0:
        raise ValueError(f"video_config.fps must be positive, got {config.fps}")
    if config.action_seed < 0:
        raise ValueError(
            f"video_config.action_seed must be nonnegative, got {config.action_seed}"
        )
    if config.simulation_backend not in {"mjx", "mujoco"}:
        raise ValueError(
            "video_config.simulation_backend must be 'mjx' or 'mujoco', got "
            f"{config.simulation_backend!r}"
        )


def final_train_state(
    out: dict, env: BelloLocomotionEnv, config: PPOConfig
) -> TrainState:
    runner_state = out["runner_state"]
    if isinstance(runner_state, RunnerState):
        params = runner_state.params
        opt_state = runner_state.opt_state
        update = runner_state.update
    else:
        params, opt_state, update, *_ = runner_state
    template_state, _ = init_train_state(
        jax.random.PRNGKey(config.seed),
        policy_observation_size(env.observation_size, config.policy),
        env.action_size,
        config,
        critic_obs_size=policy_observation_size(
            critic_observation_size(env), config.policy
        ),
    )
    _, static_policy = eqx.partition(template_state.policy, eqx.is_array)
    return TrainState(
        policy=eqx.combine(params, static_policy),
        opt_state=opt_state,
        update=update,
    )


def final_value_norm_state(out: dict):
    runner_state = out["runner_state"]
    if isinstance(runner_state, RunnerState):
        return runner_state.value_norm_state
    if len(runner_state) >= 9:
        return runner_state[-1]
    return None


def entrypoint() -> None:
    hydra_entrypoint()


@hydra.main(
    version_base=None,
    config_path=config_path("train"),
    config_name="default",
)
def hydra_entrypoint(config: DictConfig) -> None:
    run_training(structured_config(TrainingRunConfig, config))


if __name__ == "__main__":
    entrypoint()
