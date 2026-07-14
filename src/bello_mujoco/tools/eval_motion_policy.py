"""Render a deterministic rollout from a motion-tracking PPO checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TypedDict

import imageio.v2 as imageio
import jax
import mujoco

from bello_mujoco.checkpoint import restore_training_checkpoint
from bello_mujoco.envs.motion_tracking import BelloMotionTrackingEnv
from bello_mujoco.policy import (
    apply_actor,
    init_observation_stack,
    stacked_observation,
    update_observation_stack,
)
from bello_mujoco.render import (
    CameraConfig,
    make_camera,
    sync_mj_data,
    update_camera,
)


class MotionRolloutStats(TypedDict):
    rendered_steps: int
    episode_return: float
    terminal: bool
    timeout: bool


def render_checkpoint(
    checkpoint: Path,
    output_path: Path,
    *,
    steps: int | None = None,
    width: int = 960,
    height: int = 720,
    fps: int = 50,
) -> Path:
    train_state, metadata = restore_training_checkpoint(checkpoint)
    if metadata.task != "motion_tracking" or metadata.motion_env is None:
        raise ValueError(f"checkpoint task is {metadata.task!r}, not motion_tracking")
    env = BelloMotionTrackingEnv.load(
        metadata.asset,
        metadata.motion_env,
        warp=metadata.warp,
        batch_size=1,
        use_viewer_model=True,
    )
    render_motion_policy(
        env,
        train_state.policy,
        output_path,
        steps=steps or env.max_episode_steps,
        width=width,
        height=height,
        fps=fps,
        obs_stack_size=metadata.ppo.policy.obs_stack_size,
    )
    return output_path


def render_motion_policy(
    env: BelloMotionTrackingEnv,
    policy,
    output_path: Path,
    *,
    steps: int,
    width: int,
    height: int,
    fps: int,
    obs_stack_size: int,
) -> MotionRolloutStats:
    state, observation = env.reset(reference_time=jax.numpy.asarray(0.0))
    observation_stack = init_observation_stack(observation, obs_stack_size)
    mj_model = env.model.mj_model
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, width=width, height=height)
    camera_config = CameraConfig(
        mode="follow", distance=3.0, azimuth=135.0, elevation=-16.0
    )
    camera = make_camera(mj_model, camera_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps)
    episode_return = 0.0
    rendered_steps = 0
    terminal = False
    timeout = False
    try:
        for _ in range(steps):
            sync_mj_data(mj_model, mj_data, state)
            update_camera(camera, mj_model, mj_data, camera_config)
            renderer.update_scene(mj_data, camera=camera)
            writer.append_data(renderer.render())
            action = apply_actor(policy, stacked_observation(observation_stack))
            state, observation, reward, done, info = env.step(state, action)
            episode_return += float(jax.device_get(reward))
            rendered_steps += 1
            observation_stack = update_observation_stack(
                observation_stack, observation, done
            )
            if bool(jax.device_get(done)):
                terminal = bool(jax.device_get(info["done/terminal"]))
                timeout = bool(jax.device_get(info["done/timeout"]))
                break
    finally:
        writer.close()
        renderer.close()
    return MotionRolloutStats(
        rendered_steps=rendered_steps,
        episode_return=episode_return,
        terminal=terminal,
        timeout=timeout,
    )


def entrypoint() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=50)
    args = parser.parse_args()
    print(
        render_checkpoint(
            args.checkpoint,
            args.output,
            steps=args.steps,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
    )


if __name__ == "__main__":
    entrypoint()
