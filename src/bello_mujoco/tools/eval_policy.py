from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import tyro

from bello_mujoco.checkpoint import restore_training_checkpoint
from bello_mujoco.envs.locomotion import BelloLocomotionEnv
from bello_mujoco.tracking import (
    WandbConfig,
    close_run,
    create_wandb_run,
    log_config,
    track_video,
)
from bello_mujoco.policy import apply_policy


@dataclass(frozen=True)
class CameraConfig:
    mode: Literal["follow", "fixed"] = "follow"
    distance: float = 3.0
    azimuth: float = 135.0
    elevation: float = -18.0
    lookat_offset: tuple[float, float, float] = (0.0, 0.0, 0.35)


def main(
    checkpoint_path: Path,
    output_path: Path = Path("outputs/eval_rollout.mp4"),
    command: tuple[float, float, float] = (0.0, 0.0, 0.0),
    steps: int = 250,
    width: int = 640,
    height: int = 480,
    fps: int = 50,
    stop_on_done: bool = True,
    camera_config: CameraConfig = CameraConfig(),
    log_wandb: bool = False,
    wandb_config: WandbConfig = WandbConfig(),
) -> None:
    train_state, metadata = restore_training_checkpoint(checkpoint_path)
    env = BelloLocomotionEnv.load(
        metadata.asset,
        warp=metadata.warp,
        batch_size=1,
        env_config=metadata.env,
        reward_config=metadata.reward,
        command_config=metadata.command,
        done_config=metadata.done,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_policy_rollout(
        env,
        train_state.policy,
        output_path=output_path,
        command=command,
        steps=steps,
        width=width,
        height=height,
        fps=fps,
        stop_on_done=stop_on_done,
        camera_config=camera_config,
    )

    run = create_wandb_run(
        replace(wandb_config, enabled=wandb_config.enabled or log_wandb)
    )
    try:
        log_config(
            run,
            {
                "checkpoint_path": checkpoint_path,
                "eval": {
                    "command": command,
                    "steps": steps,
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "stop_on_done": stop_on_done,
                    "camera": camera_config,
                },
            },
        )
        track_video(
            run,
            output_path,
            name="eval/rollout",
            step=int(jax.device_get(train_state.update)),
            fps=fps,
            caption=f"checkpoint={checkpoint_path}",
        )
    finally:
        close_run(run)
    print({"video_path": str(output_path)})


def render_policy_rollout(
    env: BelloLocomotionEnv,
    policy,
    *,
    output_path: Path,
    command: tuple[float, float, float],
    steps: int,
    width: int,
    height: int,
    fps: int,
    stop_on_done: bool,
    camera_config: CameraConfig,
) -> None:
    state, obs = env.reset(jnp.asarray(command, dtype=jnp.float32))
    mj_model = env.model.mj_model
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    camera = make_camera(mj_model, camera_config)
    writer = imageio.get_writer(output_path, fps=fps)
    try:
        for _ in range(steps):
            sync_mj_data(mj_model, mj_data, state)
            update_camera(camera, mj_model, mj_data, camera_config)
            renderer.update_scene(mj_data, camera=camera)
            writer.append_data(renderer.render())
            mean, _, _ = apply_policy(policy, obs)
            state, obs, _, done, _ = env.step(state, mean)
            if stop_on_done and bool(jax.device_get(done)):
                sync_mj_data(mj_model, mj_data, state)
                update_camera(camera, mj_model, mj_data, camera_config)
                renderer.update_scene(mj_data, camera=camera)
                writer.append_data(renderer.render())
                break
    finally:
        writer.close()
        renderer.close()


def sync_mj_data(mj_model: mujoco.MjModel, mj_data: mujoco.MjData, state) -> None:
    mj_data.qpos[:] = np.asarray(jax.device_get(state.data.qpos))
    mj_data.qvel[:] = np.asarray(jax.device_get(state.data.qvel))
    mj_data.ctrl[:] = np.asarray(jax.device_get(state.data.ctrl))
    mujoco.mj_forward(mj_model, mj_data)


def make_camera(mj_model: mujoco.MjModel, config: CameraConfig) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(mj_model, camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = config.distance
    camera.azimuth = config.azimuth
    camera.elevation = config.elevation
    return camera


def update_camera(
    camera: mujoco.MjvCamera,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    config: CameraConfig,
) -> None:
    if config.mode != "follow":
        return
    root_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "bello_root")
    if root_body_id < 0:
        raise ValueError("follow camera requires body 'bello_root'")
    camera.lookat[:] = mj_data.xpos[root_body_id] + np.asarray(config.lookat_offset)


def entrypoint() -> None:
    tyro.cli(main)


if __name__ == "__main__":
    entrypoint()
