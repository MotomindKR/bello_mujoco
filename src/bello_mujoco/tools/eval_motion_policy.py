"""Render a deterministic rollout from a motion-tracking PPO checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

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
    env_config = replace(metadata.motion_env)
    env = BelloMotionTrackingEnv.load(
        metadata.asset,
        env_config,
        warp=metadata.warp,
        batch_size=1,
        use_viewer_model=True,
    )
    state, observation = env.reset(reference_time=jax.numpy.asarray(0.0))
    observation_stack = init_observation_stack(
        observation, metadata.ppo.policy.obs_stack_size
    )
    mj_model = env.model.mj_model
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, width=width, height=height)
    camera_config = CameraConfig(
        mode="follow", distance=3.0, azimuth=135.0, elevation=-16.0
    )
    camera = make_camera(mj_model, camera_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps)
    rollout_steps = steps or env.max_episode_steps
    try:
        for _ in range(rollout_steps):
            sync_mj_data(mj_model, mj_data, state)
            update_camera(camera, mj_model, mj_data, camera_config)
            renderer.update_scene(mj_data, camera=camera)
            writer.append_data(renderer.render())
            action = apply_actor(
                train_state.policy, stacked_observation(observation_stack)
            )
            state, observation, _, done, _ = env.step(state, action)
            observation_stack = update_observation_stack(
                observation_stack, observation, done
            )
            if bool(jax.device_get(done)):
                break
    finally:
        writer.close()
        renderer.close()
    return output_path


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
