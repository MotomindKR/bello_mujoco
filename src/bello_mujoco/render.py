from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np

from bello_mujoco.envs.locomotion import BelloLocomotionEnv
from bello_mujoco.policy import apply_policy


@dataclass(frozen=True)
class CameraConfig:
    mode: Literal["follow", "fixed"] = "follow"
    distance: float = 3.0
    azimuth: float = 135.0
    elevation: float = -18.0
    lookat_offset: tuple[float, float, float] = (0.0, 0.0, 0.35)


@dataclass(frozen=True)
class RewardOverlayConfig:
    enabled: bool = True
    position: Literal["top_left", "top_right", "bottom_left", "bottom_right"] = (
        "top_right"
    )
    font_scale: Literal["small", "normal", "large"] = "small"
    precision: int = 3
    max_reward_terms: int = 12
    show_components: bool = True


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
    simulation_backend: Literal["mjx", "mujoco"] = "mjx",
) -> None:
    if simulation_backend == "mujoco":
        render_policy_rollout_mujoco(
            env,
            policy,
            output_path=output_path,
            command=command,
            steps=steps,
            width=width,
            height=height,
            fps=fps,
            stop_on_done=stop_on_done,
            camera_config=camera_config,
        )
        return
    if simulation_backend != "mjx":
        raise ValueError(
            f"simulation_backend must be 'mjx' or 'mujoco', got {simulation_backend!r}"
        )
    state, obs = env.reset(jnp.asarray(command, dtype=jnp.float32))
    mj_model = env.model.mj_model
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    camera = make_camera(mj_model, camera_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
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


def render_policy_rollout_mujoco(
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
    mj_model = env.model.mj_model
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = env.model.default_qpos
    mj_data.qvel[:] = env.model.default_qvel
    mj_data.ctrl[:] = env.model.default_ctrl
    mujoco.mj_forward(mj_model, mj_data)

    command_array = np.asarray(command, dtype=np.float32)
    last_action = np.zeros(env.action_size, dtype=np.float32)
    step_count = 0
    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    camera = make_camera(mj_model, camera_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps)
    try:
        for _ in range(steps):
            update_camera(camera, mj_model, mj_data, camera_config)
            renderer.update_scene(mj_data, camera=camera)
            writer.append_data(renderer.render())
            obs = mujoco_policy_observation(
                env,
                mj_data,
                command=command_array,
                last_action=last_action,
                step_count=step_count,
            )
            mean, _, _ = apply_policy(policy, obs)
            action = np.asarray(jax.device_get(mean), dtype=np.float32)
            action = np.clip(action, -env.action_clip, env.action_clip)
            mj_data.ctrl[:] = mujoco_action_to_ctrl(env, action)
            for _ in range(env.decimation):
                mujoco.mj_step(mj_model, mj_data)
            last_action = action
            step_count += 1
            if stop_on_done and mujoco_done(env, mj_data, step_count):
                update_camera(camera, mj_model, mj_data, camera_config)
                renderer.update_scene(mj_data, camera=camera)
                writer.append_data(renderer.render())
                break
    finally:
        writer.close()
        renderer.close()


def mujoco_policy_observation(
    env: BelloLocomotionEnv,
    mj_data: mujoco.MjData,
    *,
    command: np.ndarray,
    last_action: np.ndarray,
    step_count: int,
) -> jax.Array:
    observation = env.config.observation
    parts: list[np.ndarray] = []
    root_qpos = int(env.model.root_qpos_adr)
    root_qvel = int(env.model.root_qvel_adr)
    quat = np.asarray(mj_data.qpos[root_qpos + 3 : root_qpos + 7], dtype=np.float32)
    if observation.base_ang_vel:
        parts.append(
            np.asarray(mj_data.qvel[root_qvel + 3 : root_qvel + 6], dtype=np.float32)
        )
    if observation.projected_gravity:
        parts.append(
            quat_rotate_inverse_np(
                quat,
                np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
            )
        )
    if observation.gait_phase:
        phase = (step_count * env.control_timestep / env.config.gait.cycle_time) % 1.0
        angle = 2.0 * np.pi * phase
        parts.append(np.asarray([np.sin(angle), np.cos(angle)], dtype=np.float32))
    if observation.command and env.command_config.enabled:
        parts.append(command)
    if observation.joint_pos:
        qpos_idx = env._observed_qpos_indices()
        parts.append(
            np.asarray(
                mj_data.qpos[qpos_idx] - env.model.default_qpos[qpos_idx],
                dtype=np.float32,
            )
        )
    if observation.joint_vel:
        parts.append(
            np.asarray(mj_data.qvel[env._observed_qvel_indices()], dtype=np.float32)
        )
    if observation.actuator_force:
        parts.append(
            np.asarray(
                mj_data.actuator_force[env._observed_actuator_indices()],
                dtype=np.float32,
            )
        )
    if observation.last_action:
        parts.append(last_action)
    if not parts:
        return jnp.zeros((0,), dtype=jnp.float32)
    return jnp.asarray(np.concatenate(parts).astype(np.float32))


def mujoco_action_to_ctrl(env: BelloLocomotionEnv, action: np.ndarray) -> np.ndarray:
    ctrl = np.asarray(env.model.default_ctrl, dtype=np.float32).copy()
    ctrl[env.model.policy_ctrl_indices] = (
        ctrl[env.model.policy_ctrl_indices] + env.model.policy_action_scale * action
    )
    ctrlrange = env.model.mj_model.actuator_ctrlrange
    return np.clip(ctrl, ctrlrange[:, 0], ctrlrange[:, 1]).astype(np.float32)


def mujoco_done(
    env: BelloLocomotionEnv, mj_data: mujoco.MjData, step_count: int
) -> bool:
    config = env.done_config
    bad_height = False
    if config.base_height.enabled:
        bad_height = (
            mujoco_link_height(env, mj_data, config.base_height.link)
            < config.base_height.min_height
        )
    bad_orientation = False
    if config.roll_pitch.enabled:
        quat = mujoco_link_quat(env, mj_data, config.roll_pitch.link)
        roll, pitch = roll_pitch_from_quat_np(quat)
        bad_orientation = (
            abs(roll) > config.roll_pitch.max_abs_roll
            or abs(pitch) > config.roll_pitch.max_abs_pitch
        )
    timeout = config.timeout.enabled and step_count >= env.max_episode_steps
    nonfinite = config.nonfinite.enabled and not (
        np.all(np.isfinite(mj_data.qpos)) and np.all(np.isfinite(mj_data.qvel))
    )
    return bool(bad_height or bad_orientation or timeout or nonfinite)


def mujoco_link_quat(
    env: BelloLocomotionEnv, mj_data: mujoco.MjData, link_name: str
) -> np.ndarray:
    if link_name == "bello_root":
        root_qpos = int(env.model.root_qpos_adr)
        return np.asarray(mj_data.qpos[root_qpos + 3 : root_qpos + 7], dtype=np.float32)
    body_id = mujoco.mj_name2id(env.model.mj_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
    if body_id < 0:
        raise ValueError(f"configured link/body not found: {link_name!r}")
    return np.asarray(mj_data.xquat[body_id], dtype=np.float32)


def mujoco_link_height(
    env: BelloLocomotionEnv, mj_data: mujoco.MjData, link_name: str
) -> float:
    if link_name == "bello_root":
        return float(mj_data.qpos[int(env.model.root_qpos_adr) + 2])
    body_id = mujoco.mj_name2id(env.model.mj_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
    if body_id < 0:
        raise ValueError(f"configured link/body not found: {link_name!r}")
    return float(mj_data.xpos[body_id, 2])


def quat_rotate_inverse_np(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz
    qvec = np.asarray([x, y, z], dtype=np.float32)
    uv = np.cross(qvec, vec)
    uuv = np.cross(qvec, uv)
    return (vec + 2.0 * (-w * uv + uuv)).astype(np.float32)


def roll_pitch_from_quat_np(quat_wxyz: np.ndarray) -> tuple[float, float]:
    w, x, y, z = quat_wxyz
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    return float(roll), float(pitch)


def launch_policy_viewer(
    env: BelloLocomotionEnv,
    policy,
    *,
    command: tuple[float, float, float],
    steps: int | None,
    realtime: bool,
    stop_on_done: bool,
    camera_config: CameraConfig,
    reward_overlay: RewardOverlayConfig = RewardOverlayConfig(),
) -> None:
    state, obs = env.reset(jnp.asarray(command, dtype=jnp.float32))
    mj_model = env.model.mj_model
    mj_data = mujoco.MjData(mj_model)
    camera_dt = env.control_timestep
    reward = jnp.asarray(0.0, dtype=jnp.float32)
    done = jnp.asarray(False)
    info: dict[str, jax.Array] = {}
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        configure_viewer_camera(viewer, mj_model, camera_config)
        step_count = 0
        while viewer.is_running() and (steps is None or step_count < steps):
            start = time.perf_counter()
            with viewer.lock():
                sync_mj_data(mj_model, mj_data, state)
                update_camera(viewer.cam, mj_model, mj_data, camera_config)
                sync_reward_overlay(
                    viewer,
                    reward_overlay,
                    step_count=step_count,
                    reward=reward,
                    done=done,
                    state=state,
                    info=info,
                )
            viewer.sync()
            mean, _, _ = apply_policy(policy, obs)
            state, obs, reward, done, info = env.step(state, mean)
            step_count += 1
            if stop_on_done and bool(jax.device_get(done)):
                break
            if realtime:
                elapsed = time.perf_counter() - start
                time.sleep(max(0.0, camera_dt - elapsed))


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


def configure_viewer_camera(viewer, mj_model: mujoco.MjModel, config: CameraConfig):
    camera = make_camera(mj_model, config)
    viewer.cam.type = camera.type
    viewer.cam.distance = camera.distance
    viewer.cam.azimuth = camera.azimuth
    viewer.cam.elevation = camera.elevation


def sync_reward_overlay(
    viewer,
    config: RewardOverlayConfig,
    *,
    step_count: int,
    reward: jax.Array,
    done: jax.Array,
    state,
    info: dict[str, jax.Array],
) -> None:
    if not config.enabled:
        viewer.clear_texts()
        return
    viewer.set_texts(
        (
            viewer_overlay_font(config),
            viewer_overlay_position(config),
            *format_reward_overlay(
                config,
                step_count=step_count,
                reward=reward,
                done=done,
                state=state,
                info=info,
            ),
        )
    )


def format_reward_overlay(
    config: RewardOverlayConfig,
    *,
    step_count: int,
    reward: jax.Array,
    done: jax.Array,
    state,
    info: dict[str, jax.Array],
) -> tuple[str, str]:
    labels = ["step", "reward", "done", "episode_return", "episode_length"]
    values = [
        str(step_count),
        format_scalar(reward, config.precision),
        str(bool(jax.device_get(done))),
        format_scalar(state.episode_return, config.precision),
        str(int(jax.device_get(state.episode_length))),
    ]
    if "base_height" in info:
        labels.append("base_height")
        values.append(format_scalar(info["base_height"], config.precision))
    if config.show_components:
        reward_terms = 0
        for name in sorted(info):
            if not name.startswith("reward/"):
                continue
            if reward_terms >= config.max_reward_terms:
                break
            labels.append(name.removeprefix("reward/"))
            values.append(format_scalar(info[name], config.precision))
            reward_terms += 1
    return "\n".join(labels), "\n".join(values)


def format_scalar(value: jax.Array, precision: int) -> str:
    return f"{float(jax.device_get(value)):.{precision}f}"


def viewer_overlay_position(config: RewardOverlayConfig):
    positions = {
        "top_left": mujoco.mjtGridPos.mjGRID_TOPLEFT,
        "top_right": mujoco.mjtGridPos.mjGRID_TOPRIGHT,
        "bottom_left": mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
        "bottom_right": mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
    }
    return positions[config.position]


def viewer_overlay_font(config: RewardOverlayConfig):
    fonts = {
        "small": mujoco.mjtFontScale.mjFONTSCALE_100,
        "normal": mujoco.mjtFontScale.mjFONTSCALE_150,
        "large": mujoco.mjtFontScale.mjFONTSCALE_200,
    }
    return fonts[config.font_scale]


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
