from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
import math
import hashlib

import jax

from bello_mujoco.models.compiler import ModelConfig
from bello_mujoco.training.checkpoint import (
    CheckpointConfig,
    LocomotionCheckpointMetadata,
    checkpoint_path,
    restore_runner_checkpoint,
    save_training_checkpoint,
    validate_checkpoint_options,
)
from bello_mujoco.envs.locomotion_config import BelloEnvConfig, DoneConfig
from bello_mujoco.envs.locomotion import BelloLocomotionEnv
from bello_mujoco.envs.rewards import CommandConfig, RewardConfig
from bello_mujoco.policy import policy_observation_size
from bello_mujoco.evaluation.video import CameraConfig
from bello_mujoco.evaluation.render import render_policy_rollout_isolated
from bello_mujoco.evaluation.software import require_linux_software_rendering
from bello_mujoco.sim import WarpConfig
from bello_mujoco.training.tracking import (
    WandbConfig,
    close_run,
    create_wandb_run,
    log_config,
    track_checkpoint_artifact,
    track_video,
    track_metric_snapshot,
)
from bello_mujoco.training.ppo_config import (
    PPOConfig,
    validate_ppo_config,
    validate_action_limit_contract,
)
from bello_mujoco.training.ppo_types import RunnerState
from bello_mujoco.training.ppo import init_runner_state, runner_train_state
from bello_mujoco.training.runtime import (
    RunnerInitialization,
    should_render_training_video,
    training_updates,
)
from bello_mujoco.repository import (
    capture_repository_commit,
    require_checkpoint_output_directory,
)


@dataclass(frozen=True)
class TrainingVideoConfig:
    enabled: bool = False
    every_updates: int = 100
    render_initial: bool = False
    render_final: bool = True
    output_dir: Path = Path("outputs/train_videos")
    filename_prefix: str = "rollout"
    command: tuple[float, float, float] = (0.0, 0.0, 0.0)
    torso_command: tuple[float, float, float] = (0.0, 0.0, 0.0)
    steps: int = 250
    width: int = 640
    height: int = 480
    fps: int = 50
    stop_on_done: bool = True
    stochastic_actions: bool = False
    action_seed: int = 0
    standing_push_enabled: bool = False
    standing_push_at_seconds: float = 5.0
    standing_push_speed: float = 0.5
    standing_suite_enabled: bool = False
    log_wandb: bool = True
    wandb_name: str = "train/rollout"
    camera_config: CameraConfig = CameraConfig()


@dataclass(frozen=True)
class TrainingRunConfig:
    config: PPOConfig = PPOConfig()
    asset_config: ModelConfig = ModelConfig()
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

    if env_config.arm_pose_path is not None:
        path = (
            checkpoint_config.resume_path.resolve() / "standing.npz"
            if checkpoint_config.resume_path is not None
            else env_config.arm_pose_path.resolve()
        )
        env_config = replace(
            env_config,
            arm_pose_path=path,
            arm_pose_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    validate_ppo_config(config)
    validate_video_config(video_config)
    command_config.torso.validate_target(video_config.torso_command)
    if (
        video_config.enabled
        and video_config.standing_push_enabled
        and not video_config.standing_suite_enabled
    ):
        dt = env_config.decimation * env_config.simulation_physics.timestep
        if (
            not 0
            < round(video_config.standing_push_at_seconds / dt)
            < video_config.steps
        ):
            raise ValueError("standing push must occur inside the evaluation video")
    if video_config.enabled:
        require_linux_software_rendering()
    validate_checkpoint_config(checkpoint_config)
    validate_action_limit_contract(config, env_config.action_clip)
    git_commit = None
    if checkpoint_config.enabled:
        checkpoint_config = replace(
            checkpoint_config,
            directory=require_checkpoint_output_directory(checkpoint_config.directory),
        )
        git_commit = capture_repository_commit()
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
                env_config=(
                    replace(
                        env_config,
                        model_randomization=replace(
                            env_config.model_randomization,
                            enabled=False,
                            payload_distribution="fixed",
                        ),
                    )
                    if asset_config.end_effector_payload_mass_kg > 0
                    else env_config
                ),
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
        critic_obs_size = env.critic_observation_size
        critic_policy_obs_size = policy_observation_size(critic_obs_size, config.policy)
        metadata = checkpoint_metadata(
            run_config,
            env,
            policy_obs_size=policy_obs_size,
            critic_policy_obs_size=critic_policy_obs_size,
        )
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
            ) = restore_runner_checkpoint(
                resume_path,
                env,
                metadata,
            )
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
                "git_commit": git_commit,
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
                git_commit,
                checkpoint_config,
                run,
            ),
        )
        metrics = out["metrics"]
        if checkpoint_config.enabled:
            if git_commit is None:
                raise RuntimeError("checkpoint Git commit was not captured")
            save_training_checkpoint(
                checkpoint_dir,
                out["runner_state"],
                metadata,
                git_commit,
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
) -> LocomotionCheckpointMetadata:
    return LocomotionCheckpointMetadata(
        ppo=run_config.config,
        asset=run_config.asset_config,
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
    metadata: LocomotionCheckpointMetadata,
    git_commit: str | None,
    config: CheckpointConfig,
    run,
) -> Callable[[RunnerState], None] | None:
    if final_path is None or config.every_updates is None:
        return None
    if git_commit is None:
        raise RuntimeError("checkpoint Git commit was not captured")

    def save(runner_state: RunnerState) -> None:
        update = int(jax.device_get(runner_state.update))
        path = periodic_checkpoint_path(final_path, update)
        save_training_checkpoint(path, runner_state, metadata, git_commit)
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

    last_metrics = None
    if video_config.enabled and video_config.render_initial and start_update == 0:
        render_training_video(
            render_env,
            runner_state,
            static_policy,
            video_config,
            run,
            env_steps=0,
            ppo_config=config,
            obs_stack_size=config.policy.obs_stack_size,
        )
    updates = training_updates(
        env, config, (runner_state, static_policy, optimizer), run
    )
    for runner_state, metrics, env_steps in updates:
        update = int(jax.device_get(runner_state.update))
        last_metrics = metrics
        if (
            checkpoint_callback is not None
            and checkpoint_every_updates is not None
            and should_save_periodic_checkpoint(
                update, config.num_updates, checkpoint_every_updates
            )
        ):
            checkpoint_callback(runner_state)
        if video_config.enabled and should_render_training_video(
            update, config.num_updates, video_config
        ):
            render_training_video(
                render_env,
                runner_state,
                static_policy,
                video_config,
                run,
                env_steps=env_steps,
                ppo_config=config,
                obs_stack_size=config.policy.obs_stack_size,
            )
    if last_metrics is None:
        raise ValueError("PPO config produced zero updates")
    return {
        "runner_state": runner_state,
        "metrics": last_metrics,
    }


def should_save_periodic_checkpoint(
    update: int,
    num_updates: int,
    every_updates: int,
) -> bool:
    return 0 < update < num_updates and update % every_updates == 0


def render_training_video(
    env: BelloLocomotionEnv,
    runner_state: RunnerState,
    static_policy,
    config: TrainingVideoConfig,
    run,
    *,
    env_steps: int,
    ppo_config: PPOConfig,
    obs_stack_size: int = 1,
) -> Path:
    update = int(jax.device_get(runner_state.update))
    output_path = config.output_dir / f"{config.filename_prefix}_{update:06d}.mp4"
    train_state = runner_train_state(runner_state, static_policy)

    def render_case(path, name, *, diagnostic=False, **overrides):
        options = dict(
            command=config.command,
            torso_command=config.torso_command,
            steps=config.steps,
            stop_on_done=config.stop_on_done,
            camera_config=config.camera_config,
            stochastic_actions=config.stochastic_actions,
        )
        options.update(overrides)
        stats = render_policy_rollout_isolated(
            env,
            train_state.policy,
            output_path=path,
            width=config.width,
            height=config.height,
            fps=config.fps,
            ppo_config=ppo_config,
            observation_norm_state=runner_state.observation_norm_state,
            obs_stack_size=obs_stack_size,
            action_seed=config.action_seed,
            **options,
        )
        if config.log_wandb:
            track_video(
                run,
                path,
                name=name,
                step=env_steps,
                caption=video_rollout_caption(update, stats),
            )
            if diagnostic:
                track_metric_snapshot(
                    run,
                    stats,
                    env_steps=env_steps,
                    update=update,
                    context={"phase": name.removesuffix("/video")},
                )
                if options.get("record_standing") and run is not None:
                    run.save(
                        str(path.with_suffix(".jsonl")),
                        base_path=str(config.output_dir),
                        policy="now",
                    )

    render_case(output_path, config.wandb_name)
    if config.standing_suite_enabled or config.standing_push_enabled:
        speed = config.standing_push_speed
        directions = (
            ("world_pos_x", (speed, 0.0)),
            ("world_neg_x", (-speed, 0.0)),
            ("world_pos_y", (0.0, speed)),
            ("world_neg_y", (0.0, -speed)),
        )
        push_step = round(config.standing_push_at_seconds / env.control_timestep)
        if config.standing_suite_enabled:
            cases = [("quiet", None, (0.0, 0.0))] + [
                (name, push_step, velocity) for name, velocity in directions
            ]
        else:
            direction = max(0, update // config.every_updates - 1) % len(directions)
            cases = [("push", push_step, directions[direction][1])]
        for name, push_at, velocity in cases:
            duration = (
                30.0 if push_at is None else config.standing_push_at_seconds + 15.0
            )
            suite = config.standing_suite_enabled
            render_case(
                output_path.with_stem(output_path.stem + "_standing_" + name),
                f"eval/standing_{name}/video",
                diagnostic=True,
                command=(0.0, 0.0, 0.0),
                steps=round(duration / env.control_timestep) if suite else config.steps,
                stop_on_done=True,
                camera_config=replace(config.camera_config, mode="fixed")
                if suite
                else config.camera_config,
                stochastic_actions=False,
                push_at_step=push_at,
                push_velocity=velocity,
                record_standing=suite,
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


def validate_video_config(config: TrainingVideoConfig) -> None:
    if (
        not math.isfinite(config.standing_push_at_seconds)
        or config.standing_push_at_seconds <= 0
    ):
        raise ValueError(
            "video_config.standing_push_at_seconds must be positive and finite"
        )
    if not math.isfinite(config.standing_push_speed) or config.standing_push_speed <= 0:
        raise ValueError("video_config.standing_push_speed must be positive and finite")
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
    validate_checkpoint_options(config)
    if config.every_updates is not None and config.every_updates <= 0:
        raise ValueError(
            "checkpoint_config.every_updates must be null or positive, got "
            f"{config.every_updates}"
        )
    if config.every_updates is not None and not config.enabled:
        raise ValueError(
            "checkpoint_config.every_updates requires checkpoint_config.enabled=true"
        )
