from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from tqdm.auto import tqdm

from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.training.checkpoint import (
    CheckpointConfig,
    CheckpointMetadata,
    DeploymentContract,
    capture_asset_contract,
    capture_deployment_contract,
    checkpoint_path,
    normalized_asset_config,
    restore_runner_checkpoint,
    save_training_checkpoint,
)
from bello_mujoco.envs.locomotion import BelloEnvConfig, BelloLocomotionEnv, DoneConfig
from bello_mujoco.envs.rewards import CommandConfig, RewardConfig
from bello_mujoco.policy import policy_observation_size
from bello_mujoco.evaluation.render import CameraConfig, render_policy_rollout
from bello_mujoco.sim import (
    WarpConfig,
)
from bello_mujoco.training.tracking import (
    WandbConfig,
    close_run,
    create_wandb_run,
    log_config,
    track_metric_snapshot,
    track_checkpoint_artifact,
    track_video,
)
from bello_mujoco.training.ppo import (
    PPOConfig,
    RunnerState,
    critic_observation_size,
    init_runner_state,
    runner_train_state,
    update_runner_state,
    validate_ppo_config,
    validate_action_limit_contract,
)

RunnerInitialization = tuple[
    RunnerState,
    eqx.Module,
    optax.GradientTransformation,
]


@dataclass(frozen=True)
class TrainingVideoConfig:
    enabled: bool = False
    every_updates: int = 100
    render_initial: bool = False
    render_final: bool = True
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
class TrainingRunConfig:
    config: PPOConfig = PPOConfig()
    asset_config: AssetPrepConfig = AssetPrepConfig()
    reward_config: RewardConfig = RewardConfig()
    command_config: CommandConfig = CommandConfig()
    done_config: DoneConfig = DoneConfig()
    env_config: BelloEnvConfig = BelloEnvConfig()
    checkpoint_config: CheckpointConfig = CheckpointConfig()
    wandb_config: WandbConfig = WandbConfig()
    video_config: TrainingVideoConfig = TrainingVideoConfig()
    warp: WarpConfig = WarpConfig()


def run_training(run_config: TrainingRunConfig) -> None:
    config = run_config.config
    asset_config = run_config.asset_config
    reward_config = run_config.reward_config
    command_config = run_config.command_config
    done_config = run_config.done_config
    env_config = run_config.env_config
    checkpoint_config = run_config.checkpoint_config
    wandb_config = run_config.wandb_config
    video_config = run_config.video_config
    warp = run_config.warp

    validate_ppo_config(config)
    validate_video_config(video_config)
    validate_checkpoint_config(checkpoint_config)
    validate_action_limit_contract(config, env_config.action_clip)
    run = create_wandb_run(wandb_config)
    succeeded = False
    try:
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
                load_mjx=False,
            )
            if video_config.enabled
            else env
        )
        rng = jax.random.PRNGKey(config.seed)
        policy_obs_size = policy_observation_size(env.observation_size, config.policy)
        critic_obs_size = critic_observation_size(env)
        critic_policy_obs_size = policy_observation_size(critic_obs_size, config.policy)
        metadata = checkpoint_metadata(
            run_config,
            env,
            policy_obs_size=policy_obs_size,
            critic_policy_obs_size=critic_policy_obs_size,
        )
        deployment_contract = capture_deployment_contract(metadata)
        checkpoint_dir = (
            checkpoint_path(checkpoint_config, config)
            if checkpoint_config.enabled
            else None
        )
        initial_runner: RunnerInitialization | None = None
        if checkpoint_config.resume_path is not None:
            resume_path = checkpoint_config.resume_path.expanduser()
            (
                resumed_state,
                static_policy,
                optimizer,
                resumed_metadata,
            ) = restore_runner_checkpoint(resume_path, env, metadata)
            if resumed_metadata != metadata:
                raise RuntimeError("resume did not resolve to the requested run")
            initial_runner = resumed_state, static_policy, optimizer
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
                "video": video_config,
                "warp": warp,
                "env": {
                    "obs_size": env.observation_size,
                    "actor_observation_layout": env.observation_layout,
                    "policy_obs_size": policy_obs_size,
                    "critic_obs_size": critic_obs_size,
                    "critic_observation_layout": env.critic_observation_layout,
                    "critic_policy_obs_size": critic_policy_obs_size,
                    "obs_stack_size": config.policy.obs_stack_size,
                    "action_size": env.action_size,
                    "action_layout": env.action_layout,
                    "mjx_impl": "warp",
                    "xml_path": env.model.xml_path,
                    "checkpoint_path": checkpoint_dir,
                },
            },
        )
        out = train_update_loop(
            env,
            render_env,
            config,
            rng,
            video_config,
            run,
            initial_runner=initial_runner,
            checkpoint_every_updates=checkpoint_config.every_updates,
            checkpoint_callback=periodic_checkpoint_callback(
                checkpoint_dir,
                metadata,
                deployment_contract,
                checkpoint_config,
                run,
            ),
        )
        metrics = out["metrics"]
        if checkpoint_config.enabled:
            save_training_checkpoint(
                checkpoint_dir,
                out["runner_state"],
                metadata,
                deployment_contract,
            )
            if checkpoint_config.upload_wandb_artifact and run is not None:
                track_checkpoint_artifact(
                    run,
                    checkpoint_dir,
                    name=checkpoint_config.wandb_artifact_name or checkpoint_dir.name,
                    artifact_type=checkpoint_config.wandb_artifact_type,
                )
        print(metrics)
        if checkpoint_dir is not None:
            print({"checkpoint_path": str(checkpoint_dir)})
        succeeded = True
    finally:
        close_run(run, exit_code=0 if succeeded else 1)


def checkpoint_metadata(
    run_config: TrainingRunConfig,
    env: BelloLocomotionEnv,
    *,
    policy_obs_size: int,
    critic_policy_obs_size: int,
) -> CheckpointMetadata:
    normalized_asset = normalized_asset_config(run_config.asset_config)
    if run_config.asset_config.output_root is not None:
        actual_runtime = capture_asset_contract(
            run_config.asset_config, run_config.env_config.terrain
        )
        reproducible_runtime = capture_asset_contract(
            normalized_asset, run_config.env_config.terrain
        )
        if actual_runtime != reproducible_runtime:
            raise ValueError(
                "asset_config.output_root resolved to prepared assets that differ "
                "from the repository-derived cache; refusing to create a "
                "checkpoint that cannot be reproduced"
            )
    return CheckpointMetadata(
        ppo=run_config.config,
        asset=normalized_asset,
        reward=run_config.reward_config,
        command=run_config.command_config,
        done=run_config.done_config,
        env=run_config.env_config,
        warp=run_config.warp,
        obs_size=policy_obs_size,
        critic_obs_size=critic_policy_obs_size,
        action_size=env.action_size,
        actor_observation_layout=env.observation_layout,
        critic_observation_layout=env.critic_observation_layout,
        action_layout=env.action_layout,
    )


def periodic_checkpoint_callback(
    final_path: Path | None,
    metadata: CheckpointMetadata,
    deployment_contract: DeploymentContract,
    config: CheckpointConfig,
    run,
) -> Callable[[RunnerState], None] | None:
    if final_path is None or config.every_updates is None:
        return None

    def save(runner_state: RunnerState) -> None:
        update = int(jax.device_get(runner_state.update))
        path = periodic_checkpoint_path(final_path, update)
        save_training_checkpoint(path, runner_state, metadata, deployment_contract)
        if config.upload_wandb_artifact and run is not None:
            base_name = config.wandb_artifact_name or final_path.name
            track_checkpoint_artifact(
                run,
                path,
                name=f"{base_name}-update-{update:06d}",
                artifact_type=config.wandb_artifact_type,
            )

    return save


def periodic_checkpoint_path(final_path: Path, update: int) -> Path:
    return final_path.with_name(f"{final_path.name}_update_{update:06d}")


def train_update_loop(
    env: BelloLocomotionEnv,
    render_env: BelloLocomotionEnv,
    config: PPOConfig,
    rng: jax.Array,
    video_config: TrainingVideoConfig,
    run,
    *,
    initial_runner: RunnerInitialization | None = None,
    checkpoint_every_updates: int | None = None,
    checkpoint_callback: Callable[[RunnerState], None] | None = None,
) -> dict:
    if initial_runner is None:
        runner_state, static_policy, optimizer = init_runner_state(rng, env, config)
    else:
        runner_state, static_policy, optimizer = initial_runner
    start_update = int(jax.device_get(runner_state.update))
    if not 0 <= start_update < config.num_updates:
        raise ValueError(
            "runner update must be within the unfinished training range, got "
            f"update={start_update}, num_updates={config.num_updates}"
        )

    def update_fn(state):
        return update_runner_state(state, static_policy, optimizer, env, config)

    update_fn = jax.jit(update_fn)
    last_metrics = None
    if video_config.render_initial and start_update == 0:
        render_training_video(
            render_env,
            runner_state,
            static_policy,
            video_config,
            run,
            env_steps=0,
            obs_stack_size=config.policy.obs_stack_size,
        )
    for _ in training_update_range(start_update, config.num_updates):
        runner_state, metrics = update_fn(runner_state)
        metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
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
        if (
            checkpoint_callback is not None
            and checkpoint_every_updates is not None
            and should_save_periodic_checkpoint(
                update, config.num_updates, checkpoint_every_updates
            )
        ):
            checkpoint_callback(runner_state)
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
    if last_metrics is None:
        raise ValueError("PPO config produced zero updates")
    return {
        "runner_state": runner_state,
        "metrics": last_metrics,
    }


def training_update_range(start_update: int, num_updates: int):
    return tqdm(
        range(start_update, num_updates),
        desc="PPO updates",
        unit="update",
        dynamic_ncols=True,
        initial=start_update,
        total=num_updates,
    )


def should_save_periodic_checkpoint(
    update: int,
    num_updates: int,
    every_updates: int,
) -> bool:
    return 0 < update < num_updates and update % every_updates == 0


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


def raise_on_invalid_training_metrics(metrics: dict, update: int) -> None:
    contaminated: list[tuple[str, str]] = []
    for name, value in metrics.items():
        prefix = next(
            (
                prefix
                for prefix in ("nonfinite/", "invalid/")
                if name.startswith(prefix)
            ),
            None,
        )
        if prefix is None or value is None:
            continue
        if float(jax.device_get(value)) > 0.0:
            contaminated.append((prefix.removesuffix("/"), name.removeprefix(prefix)))
    if contaminated:
        kind, source = contaminated[0]
        raise RuntimeError(
            f"PPO encountered {kind} {source} at update {update}; "
            "the poisoned optimizer step was skipped."
        )
    invalid_update = metrics.get("invalid_update")
    if invalid_update is not None and float(jax.device_get(invalid_update)) > 0.0:
        raise RuntimeError(
            f"PPO rejected an implausible update at update {update}; "
            "the poisoned optimizer step was skipped."
        )
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


def validate_checkpoint_config(config: CheckpointConfig) -> None:
    if config.every_updates is not None and config.every_updates <= 0:
        raise ValueError(
            "checkpoint_config.every_updates must be null or positive, got "
            f"{config.every_updates}"
        )
    if config.every_updates is not None and not config.enabled:
        raise ValueError(
            "checkpoint_config.every_updates requires checkpoint_config.enabled=true"
        )
