from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable
from typing import Literal
from typing import TypeAlias

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from bello_mujoco.envs.locomotion import BelloLocomotionEnv
from bello_mujoco.policy import (
    apply_actor,
    init_observation_stack,
    policy_log_std,
    stacked_observation,
    update_observation_stack,
)

CONTACT_FOOT_RGBA = np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
RolloutStats: TypeAlias = dict[str, float | int]
DONE_FLAG_KEYS = (
    "done/any",
    "done/terminal",
    "done/base_height",
    "done/bad_orientation",
    "done/timeout",
    "done/nonfinite",
)


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


@dataclass(frozen=True)
class ActionOverlayConfig:
    enabled: bool = True
    max_actions: int = 13
    bar_width: int = 12


@dataclass(frozen=True)
class JoystickConfig:
    enabled: bool = True
    lin_vel_x_step: float = 0.05
    lin_vel_y_step: float = 0.05
    yaw_vel_step: float = 0.05
    lin_vel_x_limit: tuple[float, float] = (-1.0, 1.0)
    lin_vel_y_limit: tuple[float, float] = (-1.0, 1.0)
    yaw_vel_limit: tuple[float, float] = (-1.0, 1.0)


@dataclass
class ViewerRuntimeState:
    realtime: bool
    stochastic_actions: bool
    paused: bool = False
    step_once: bool = False
    reset_requested: bool = False

    def should_step(self) -> bool:
        return not self.paused or self.step_once

    def consume_step_once(self) -> None:
        self.step_once = False


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
    obs_stack_size: int = 1,
    stochastic_actions: bool = False,
    action_seed: int = 0,
) -> RolloutStats:
    if simulation_backend == "mujoco":
        return render_policy_rollout_mujoco(
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
            obs_stack_size=obs_stack_size,
            stochastic_actions=stochastic_actions,
            action_seed=action_seed,
        )
        return
    if simulation_backend != "mjx":
        raise ValueError(
            f"simulation_backend must be 'mjx' or 'mujoco', got {simulation_backend!r}"
        )
    command_array = jnp.asarray(command, dtype=jnp.float32)
    state, obs = env.reset(command_array)
    obs_stack = init_observation_stack(obs, obs_stack_size)
    mj_model = env.model.mj_model
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    camera = make_camera(mj_model, camera_config)
    foot_geom_ids_by_foot = foot_geom_ids_for_bodies(mj_model, env.model.foot_body_ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps)
    action_key = jax.random.PRNGKey(action_seed)
    stats = init_rollout_stats(
        steps, stochastic_actions=stochastic_actions, action_seed=action_seed
    )
    episode_length = 0
    try:
        for step_index in range(steps):
            sync_mj_data(mj_model, mj_data, state)
            update_camera(camera, mj_model, mj_data, camera_config)
            renderer.update_scene(mj_data, camera=camera)
            sync_renderer_foot_contact_markers(
                mj_model,
                mj_data,
                foot_geom_ids_by_foot,
                viewer_foot_contact(env, state),
                renderer.scene,
            )
            writer.append_data(renderer.render())
            action_rng = None
            if stochastic_actions:
                action_key, action_rng = jax.random.split(action_key)
            action = rollout_policy_action(
                policy,
                obs_stack,
                action_rng=action_rng,
                action_clip=env.action_clip,
            )
            state, obs, _, done, info = env.step(state, action)
            episode_length += 1
            done_bool = bool(jax.device_get(done))
            if done_bool:
                record_rollout_done(
                    stats,
                    step_index=step_index + 1,
                    episode_length=episode_length,
                    done_flags=jax.device_get(info),
                    control_timestep=env.control_timestep,
                )
            if done_bool and stop_on_done:
                sync_mj_data(mj_model, mj_data, state)
                update_camera(camera, mj_model, mj_data, camera_config)
                renderer.update_scene(mj_data, camera=camera)
                sync_renderer_foot_contact_markers(
                    mj_model,
                    mj_data,
                    foot_geom_ids_by_foot,
                    viewer_foot_contact(env, state),
                    renderer.scene,
                )
                writer.append_data(renderer.render())
                break
            if done_bool:
                state, obs = env.reset(command_array)
                obs_stack = init_observation_stack(obs, obs_stack_size)
                episode_length = 0
            else:
                obs_stack = update_observation_stack(obs_stack, obs, done)
    finally:
        writer.close()
        renderer.close()
    return finalize_rollout_stats(stats, current_episode_length=episode_length)


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
    obs_stack_size: int = 1,
    stochastic_actions: bool = False,
    action_seed: int = 0,
) -> RolloutStats:
    mj_model = env.model.mj_model
    mj_data = mujoco.MjData(mj_model)
    reset_mujoco_data(env, mj_data)
    foot_geom_ids_by_foot = foot_geom_ids_for_bodies(mj_model, env.model.foot_body_ids)

    command_array = np.asarray(command, dtype=np.float32)
    last_action = np.zeros(env.action_size, dtype=np.float32)
    step_count = 0
    obs = mujoco_policy_observation(
        env,
        mj_data,
        command=command_array,
        last_action=last_action,
        step_count=step_count,
    )
    obs_stack = init_observation_stack(obs, obs_stack_size)
    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    camera = make_camera(mj_model, camera_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps)
    action_key = jax.random.PRNGKey(action_seed)
    stats = init_rollout_stats(
        steps, stochastic_actions=stochastic_actions, action_seed=action_seed
    )
    try:
        for step_index in range(steps):
            update_camera(camera, mj_model, mj_data, camera_config)
            renderer.update_scene(mj_data, camera=camera)
            sync_renderer_foot_contact_markers(
                mj_model,
                mj_data,
                foot_geom_ids_by_foot,
                mujoco_foot_contact(env, mj_data),
                renderer.scene,
            )
            writer.append_data(renderer.render())
            action_rng = None
            if stochastic_actions:
                action_key, action_rng = jax.random.split(action_key)
            action = rollout_policy_action(
                policy,
                obs_stack,
                action_rng=action_rng,
                action_clip=env.action_clip,
            )
            action = np.asarray(jax.device_get(action), dtype=np.float32)
            mj_data.ctrl[:] = mujoco_action_to_ctrl(env, action)
            for _ in range(env.decimation):
                mujoco.mj_step(mj_model, mj_data)
            last_action = action
            step_count += 1
            done_flags = mujoco_done_flags(env, mj_data, step_count)
            done = bool(done_flags["done/any"])
            if done:
                record_rollout_done(
                    stats,
                    step_index=step_index + 1,
                    episode_length=step_count,
                    done_flags=done_flags,
                    control_timestep=env.control_timestep,
                )
            if done and stop_on_done:
                obs = mujoco_policy_observation(
                    env,
                    mj_data,
                    command=command_array,
                    last_action=last_action,
                    step_count=step_count,
                )
                obs_stack = update_observation_stack(obs_stack, obs, jnp.asarray(True))
                update_camera(camera, mj_model, mj_data, camera_config)
                renderer.update_scene(mj_data, camera=camera)
                sync_renderer_foot_contact_markers(
                    mj_model,
                    mj_data,
                    foot_geom_ids_by_foot,
                    mujoco_foot_contact(env, mj_data),
                    renderer.scene,
                )
                writer.append_data(renderer.render())
                break
            if done:
                reset_mujoco_data(env, mj_data)
                last_action = np.zeros(env.action_size, dtype=np.float32)
                step_count = 0
                obs = mujoco_policy_observation(
                    env,
                    mj_data,
                    command=command_array,
                    last_action=last_action,
                    step_count=step_count,
                )
                obs_stack = init_observation_stack(obs, obs_stack_size)
                continue
            obs = mujoco_policy_observation(
                env,
                mj_data,
                command=command_array,
                last_action=last_action,
                step_count=step_count,
            )
            obs_stack = update_observation_stack(obs_stack, obs, jnp.asarray(False))
    finally:
        writer.close()
        renderer.close()
    return finalize_rollout_stats(stats, current_episode_length=step_count)


def rollout_policy_action(
    policy,
    obs_stack: jax.Array,
    *,
    action_rng: jax.Array | None = None,
    action_clip: float | None = None,
) -> jax.Array:
    obs = stacked_observation(obs_stack)
    mean = apply_actor(policy, obs)
    action = mean
    if action_rng is not None:
        log_std = policy_log_std(policy)
        noise = jax.random.normal(action_rng, mean.shape)
        action = mean + jnp.exp(log_std) * noise
    if action_clip is not None:
        action = jnp.clip(action, -action_clip, action_clip)
    return action


def init_rollout_stats(
    steps: int, *, stochastic_actions: bool, action_seed: int
) -> RolloutStats:
    stats: RolloutStats = {
        "steps": int(steps),
        "stochastic_actions": int(stochastic_actions),
        "action_seed": int(action_seed),
        "completed_episodes": 0,
        "first_done_step": -1,
        "first_done_seconds": -1.0,
        "last_done_step": -1,
        "mean_completed_episode_length": 0.0,
        "max_completed_episode_length": 0,
        "current_episode_length": 0,
        "_completed_episode_length_sum": 0.0,
    }
    for key in DONE_FLAG_KEYS:
        stats[key] = 0
    return stats


def record_rollout_done(
    stats: RolloutStats,
    *,
    step_index: int,
    episode_length: int,
    done_flags,
    control_timestep: float,
) -> None:
    stats["completed_episodes"] = int(stats["completed_episodes"]) + 1
    stats["last_done_step"] = int(step_index)
    if int(stats["first_done_step"]) < 0:
        stats["first_done_step"] = int(step_index)
        stats["first_done_seconds"] = float(step_index * control_timestep)
    stats["_completed_episode_length_sum"] = (
        float(stats["_completed_episode_length_sum"]) + float(episode_length)
    )
    stats["max_completed_episode_length"] = max(
        int(stats["max_completed_episode_length"]), int(episode_length)
    )
    for key in DONE_FLAG_KEYS:
        value = done_flags.get(key, False)
        stats[key] = int(stats[key]) + int(bool(np.asarray(value)))


def finalize_rollout_stats(
    stats: RolloutStats, *, current_episode_length: int
) -> RolloutStats:
    completed = int(stats["completed_episodes"])
    length_sum = float(stats.pop("_completed_episode_length_sum"))
    stats["mean_completed_episode_length"] = (
        length_sum / completed if completed else 0.0
    )
    stats["current_episode_length"] = int(current_episode_length)
    return stats


def mujoco_policy_observation(
    env: BelloLocomotionEnv,
    mj_data: mujoco.MjData,
    *,
    command: np.ndarray,
    last_action: np.ndarray,
    step_count: int,
) -> jax.Array:
    observation = env.config.observation
    scale = observation.scale
    parts: list[np.ndarray] = []
    root_link = env.config.root_link
    if observation.base_lin_vel:
        parts.append(
            mujoco_scaled(
                mujoco_link_lin_vel(env, mj_data, root_link), scale.base_lin_vel
            )
        )
    if observation.base_ang_vel:
        parts.append(
            mujoco_scaled(
                mujoco_link_ang_vel(env, mj_data, root_link), scale.base_ang_vel
            )
        )
    if observation.projected_gravity:
        parts.append(
            mujoco_scaled(
                quat_rotate_inverse_np(
                    mujoco_link_quat(env, mj_data, root_link),
                    np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
                ),
                scale.projected_gravity,
            )
        )
    if observation.base_height:
        parts.append(
            mujoco_scaled(
                np.asarray(
                    [mujoco_link_height(env, mj_data, root_link)], dtype=np.float32
                ),
                scale.base_height,
            )
        )
    if observation.gait_phase:
        phase = mujoco_gait_phase(env, step_count, command)
        angle = 2.0 * np.pi * phase
        parts.append(
            mujoco_scaled(
                np.asarray([np.sin(angle), np.cos(angle)], dtype=np.float32),
                scale.gait_phase,
            )
        )
    if observation.command and env.command_config.enabled:
        parts.append(mujoco_scaled(command, scale.command))
    if observation.joint_pos:
        qpos_idx = env._observed_qpos_indices(observation)
        parts.append(
            mujoco_scaled(
                np.asarray(
                    mj_data.qpos[qpos_idx] - env.model.default_qpos[qpos_idx],
                    dtype=np.float32,
                ),
                scale.joint_pos,
            )
        )
    if observation.joint_vel:
        parts.append(
            mujoco_scaled(
                np.asarray(
                    mj_data.qvel[env._observed_qvel_indices(observation)],
                    dtype=np.float32,
                ),
                scale.joint_vel,
            )
        )
    if observation.actuator_force:
        parts.append(
            mujoco_scaled(
                np.asarray(
                    mj_data.actuator_force[
                        env._observed_actuator_indices(observation)
                    ],
                    dtype=np.float32,
                ),
                scale.actuator_force,
            )
        )
    if observation.last_action:
        parts.append(mujoco_scaled(last_action, scale.last_action))
    if not parts:
        return jnp.zeros((0,), dtype=jnp.float32)
    return jnp.asarray(np.concatenate(parts).astype(np.float32))


def reset_mujoco_data(env: BelloLocomotionEnv, mj_data: mujoco.MjData) -> None:
    mj_data.qpos[:] = env.model.default_qpos
    mj_data.qvel[:] = env.model.default_qvel
    mj_data.ctrl[:] = env.model.default_ctrl
    mujoco.mj_forward(env.model.mj_model, mj_data)


def mujoco_scaled(value: np.ndarray, scale: float) -> np.ndarray:
    return (np.asarray(value, dtype=np.float32) * np.float32(scale)).astype(np.float32)


def mujoco_action_to_ctrl(env: BelloLocomotionEnv, action: np.ndarray) -> np.ndarray:
    ctrl = np.asarray(env.model.default_ctrl, dtype=np.float32).copy()
    ctrl[env.model.policy_ctrl_indices] = (
        ctrl[env.model.policy_ctrl_indices] + env.model.policy_action_scale * action
    )
    return ctrl.astype(np.float32)


def mujoco_done(
    env: BelloLocomotionEnv, mj_data: mujoco.MjData, step_count: int
) -> bool:
    return bool(mujoco_done_flags(env, mj_data, step_count)["done/any"])


def mujoco_done_flags(
    env: BelloLocomotionEnv, mj_data: mujoco.MjData, step_count: int
) -> dict[str, bool]:
    config = env.done_config
    bad_height = False
    if config.base_height.enabled:
        bad_height = (
            mujoco_link_height(env, mj_data, env._done_link(config.base_height))
            < config.base_height.min_height
        )
    bad_orientation = False
    if config.bad_orientation.enabled:
        projected_gravity = mujoco_link_projected_gravity(
            env, mj_data, env._done_link(config.bad_orientation)
        )
        bad_orientation = (
            projected_gravity_tilt_angle_np(projected_gravity)
            > config.bad_orientation.limit_angle
        )
    timeout = config.timeout.enabled and step_count >= env.max_episode_steps
    nonfinite = config.nonfinite.enabled and not (
        np.all(np.isfinite(mj_data.qpos)) and np.all(np.isfinite(mj_data.qvel))
        and np.all(np.isfinite(mj_data.xpos))
        and np.all(np.isfinite(mj_data.xquat))
        and np.all(np.isfinite(mj_data.cvel))
        and np.all(np.isfinite(mj_data.actuator_force))
    )
    terminal = bool(bad_height or bad_orientation or nonfinite)
    done = bool(terminal or timeout)
    return {
        "done/any": done,
        "done/terminal": terminal,
        "done/base_height": bool(bad_height),
        "done/bad_orientation": bool(bad_orientation),
        "done/timeout": bool(timeout),
        "done/nonfinite": bool(nonfinite),
    }


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


def mujoco_link_projected_gravity(
    env: BelloLocomotionEnv, mj_data: mujoco.MjData, link_name: str
) -> np.ndarray:
    return quat_rotate_inverse_np(
        mujoco_link_quat(env, mj_data, link_name),
        np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
    )


def mujoco_link_height(
    env: BelloLocomotionEnv, mj_data: mujoco.MjData, link_name: str
) -> float:
    if link_name == "bello_root":
        return float(mj_data.qpos[int(env.model.root_qpos_adr) + 2])
    body_id = mujoco.mj_name2id(env.model.mj_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
    if body_id < 0:
        raise ValueError(f"configured link/body not found: {link_name!r}")
    return float(mj_data.xpos[body_id, 2])


def mujoco_link_lin_vel(
    env: BelloLocomotionEnv, mj_data: mujoco.MjData, link_name: str
) -> np.ndarray:
    if link_name == "bello_root":
        root_qpos = int(env.model.root_qpos_adr)
        root_qvel = int(env.model.root_qvel_adr)
        quat = np.asarray(mj_data.qpos[root_qpos + 3 : root_qpos + 7], dtype=np.float32)
        return quat_rotate_inverse_np(
            quat,
            np.asarray(mj_data.qvel[root_qvel : root_qvel + 3], dtype=np.float32),
        )
    body_id = mujoco.mj_name2id(env.model.mj_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
    if body_id < 0:
        raise ValueError(f"configured link/body not found: {link_name!r}")
    return np.asarray(mj_data.cvel[body_id, 3:6], dtype=np.float32)


def mujoco_link_ang_vel(
    env: BelloLocomotionEnv, mj_data: mujoco.MjData, link_name: str
) -> np.ndarray:
    if link_name == "bello_root":
        root_qpos = int(env.model.root_qpos_adr)
        root_qvel = int(env.model.root_qvel_adr)
        quat = np.asarray(mj_data.qpos[root_qpos + 3 : root_qpos + 7], dtype=np.float32)
        return quat_rotate_inverse_np(
            quat,
            np.asarray(mj_data.qvel[root_qvel + 3 : root_qvel + 6], dtype=np.float32),
        )
    body_id = mujoco.mj_name2id(env.model.mj_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
    if body_id < 0:
        raise ValueError(f"configured link/body not found: {link_name!r}")
    return np.asarray(mj_data.cvel[body_id, :3], dtype=np.float32)


def quat_rotate_inverse_np(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz
    qvec = np.asarray([x, y, z], dtype=np.float32)
    uv = np.cross(qvec, vec)
    uuv = np.cross(qvec, uv)
    return (vec + 2.0 * (-w * uv + uuv)).astype(np.float32)


def quat_rotate_np(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz
    qvec = np.asarray([x, y, z], dtype=np.float32)
    uv = np.cross(qvec, vec)
    uuv = np.cross(qvec, uv)
    return (vec + 2.0 * (w * uv + uuv)).astype(np.float32)


def projected_gravity_tilt_angle_np(projected_gravity: np.ndarray) -> float:
    return float(np.arccos(np.clip(-projected_gravity[2], -1.0, 1.0)))


class CommandJoystick:
    KEY_SPACE = 32
    KEY_LEFT = 263
    KEY_RIGHT = 262
    KEY_DOWN = 264
    KEY_UP = 265

    def __init__(
        self, command: np.ndarray, config: JoystickConfig, *, enabled: bool
    ) -> None:
        self.command = np.asarray(command, dtype=np.float32).copy()
        self.config = config
        self.enabled = enabled

    def key_callback(self, key: int) -> None:
        if not self.enabled:
            return
        key = self._normalize_letter_key(key)
        if key in {ord("W"), self.KEY_UP}:
            self.command[1] += self.config.lin_vel_y_step
        elif key in {ord("S"), self.KEY_DOWN}:
            self.command[1] -= self.config.lin_vel_y_step
        elif key in {ord("D"), self.KEY_RIGHT}:
            self.command[0] += self.config.lin_vel_x_step
        elif key in {ord("A"), self.KEY_LEFT}:
            self.command[0] -= self.config.lin_vel_x_step
        elif key == ord("Q"):
            self.command[2] += self.config.yaw_vel_step
        elif key == ord("E"):
            self.command[2] -= self.config.yaw_vel_step
        elif key in {self.KEY_SPACE, ord("0")}:
            self.command[:] = 0.0
        else:
            return
        self._clip()

    def _clip(self) -> None:
        self.command[0] = np.clip(self.command[0], *self.config.lin_vel_x_limit)
        self.command[1] = np.clip(self.command[1], *self.config.lin_vel_y_limit)
        self.command[2] = np.clip(self.command[2], *self.config.yaw_vel_limit)

    @staticmethod
    def _normalize_letter_key(key: int) -> int:
        if ord("a") <= key <= ord("z"):
            return key - ord("a") + ord("A")
        return key


def load_imgui_glfw():
    try:
        import glfw
        import imgui
        from imgui.integrations.glfw import GlfwRenderer
    except ImportError as exc:
        raise RuntimeError(
            "live evaluation requires the imgui GLFW renderer; install the "
            "project dependencies with uv sync"
        ) from exc
    return glfw, imgui, GlfwRenderer


class ImguiMujocoViewer:
    def __init__(
        self,
        mj_model: mujoco.MjModel,
        camera_config: CameraConfig,
        *,
        key_callback: Callable[[int], None] | None = None,
        title: str = "Bello MuJoCo Policy Viewer",
        width: int = 1280,
        height: int = 900,
        maxgeom: int = 10000,
    ) -> None:
        self.mj_model = mj_model
        self.glfw, self.imgui, glfw_renderer = load_imgui_glfw()
        self.key_callback = key_callback
        self.window = None
        self.impl = None
        self._imgui_context_created = False
        self._last_cursor: tuple[float, float] | None = None

        if not self.glfw.init():
            raise RuntimeError("failed to initialize GLFW")
        self.glfw.window_hint(self.glfw.VISIBLE, self.glfw.TRUE)
        self.window = self.glfw.create_window(width, height, title, None, None)
        if self.window is None:
            self.glfw.terminate()
            raise RuntimeError("failed to create GLFW window")
        try:
            self.glfw.make_context_current(self.window)
            self.glfw.swap_interval(1)

            self.imgui.create_context()
            self._imgui_context_created = True
            self.impl = glfw_renderer(self.window, attach_callbacks=False)
            self.glfw.set_key_callback(self.window, self._on_key)
            self.glfw.set_cursor_pos_callback(self.window, self._on_cursor_pos)
            self.glfw.set_window_size_callback(self.window, self._on_window_size)
            self.glfw.set_char_callback(self.window, self._on_char)
            self.glfw.set_scroll_callback(self.window, self._on_scroll)

            self.camera = make_camera(mj_model, camera_config)
            self.option = mujoco.MjvOption()
            self.scene = mujoco.MjvScene(mj_model, maxgeom=maxgeom)
            self.context = mujoco.MjrContext(
                mj_model, mujoco.mjtFontScale.mjFONTSCALE_150
            )
        except Exception:
            self.close()
            raise

    def is_running(self) -> bool:
        return bool(
            self.window is not None and not self.glfw.window_should_close(self.window)
        )

    def close(self) -> None:
        if self.impl is not None:
            self.impl.shutdown()
            self.impl = None
        if self._imgui_context_created and hasattr(self.imgui, "destroy_context"):
            self.imgui.destroy_context()
            self._imgui_context_created = False
        if getattr(self, "context", None) is not None:
            self.context.free()
            self.context = None
        if self.window is not None:
            self.glfw.destroy_window(self.window)
            self.window = None
        self.glfw.terminate()

    def render(
        self,
        mj_data: mujoco.MjData,
        *,
        foot_geom_ids_by_foot: tuple[np.ndarray, ...],
        foot_contact: np.ndarray,
        draw_ui: Callable[[], None],
    ) -> None:
        self.glfw.poll_events()
        self.impl.process_inputs()
        self.imgui.new_frame()
        draw_ui()

        framebuffer_width, framebuffer_height = self.glfw.get_framebuffer_size(
            self.window
        )
        viewport = mujoco.MjrRect(0, 0, framebuffer_width, framebuffer_height)
        mujoco.mjv_updateScene(
            self.mj_model,
            mj_data,
            self.option,
            None,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )
        append_foot_contact_markers(
            self.mj_model,
            mj_data,
            foot_geom_ids_by_foot,
            foot_contact,
            self.scene,
        )
        mujoco.mjr_render(viewport, self.scene, self.context)
        self.imgui.render()
        self.impl.render(self.imgui.get_draw_data())
        self.glfw.swap_buffers(self.window)

    def _on_key(self, window, key: int, scancode: int, action: int, mods: int) -> None:
        self.impl.keyboard_callback(window, key, scancode, action, mods)
        if action not in {self.glfw.PRESS, self.glfw.REPEAT}:
            return
        if self.imgui.get_io().want_capture_keyboard:
            return
        if self.key_callback is not None:
            self.key_callback(key)

    def _on_cursor_pos(self, *args) -> None:
        self.impl.mouse_callback(*args)
        _, x_pos, y_pos = args
        cursor = (float(x_pos), float(y_pos))
        if self._last_cursor is None:
            self._last_cursor = cursor
            return
        dx = cursor[0] - self._last_cursor[0]
        dy = cursor[1] - self._last_cursor[1]
        self._last_cursor = cursor
        if self.imgui.get_io().want_capture_mouse:
            return
        action = self._camera_mouse_action()
        if action is None:
            return
        _, height = self.glfw.get_window_size(self.window)
        height = max(float(height), 1.0)
        mujoco.mjv_moveCamera(
            self.mj_model,
            action,
            dx / height,
            dy / height,
            self.scene,
            self.camera,
        )

    def _on_scroll(self, window, x_offset: float, y_offset: float) -> None:
        self.impl.scroll_callback(window, x_offset, y_offset)
        if self.imgui.get_io().want_capture_mouse:
            return
        mujoco.mjv_moveCamera(
            self.mj_model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0.0,
            -0.05 * float(y_offset),
            self.scene,
            self.camera,
        )

    def _on_window_size(self, window, width: int, height: int) -> None:
        self.impl.resize_callback(window, width, height)

    def _on_char(self, window, char: int) -> None:
        self.impl.char_callback(window, char)

    def _camera_mouse_action(self):
        if self.glfw.get_mouse_button(self.window, self.glfw.MOUSE_BUTTON_MIDDLE):
            return mujoco.mjtMouse.mjMOUSE_MOVE_H
        shift = (
            self.glfw.get_key(self.window, self.glfw.KEY_LEFT_SHIFT)
            or self.glfw.get_key(self.window, self.glfw.KEY_RIGHT_SHIFT)
        )
        if self.glfw.get_mouse_button(self.window, self.glfw.MOUSE_BUTTON_RIGHT):
            return (
                mujoco.mjtMouse.mjMOUSE_MOVE_H
                if shift
                else mujoco.mjtMouse.mjMOUSE_MOVE_V
            )
        if self.glfw.get_mouse_button(self.window, self.glfw.MOUSE_BUTTON_LEFT):
            return (
                mujoco.mjtMouse.mjMOUSE_ROTATE_H
                if shift
                else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            )
        return None


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
    action_overlay: ActionOverlayConfig = ActionOverlayConfig(),
    joystick_config: JoystickConfig = JoystickConfig(),
    obs_stack_size: int = 1,
    stochastic_actions: bool = False,
    action_seed: int = 0,
) -> None:
    mj_model = env.model.mj_model
    mj_data = mujoco.MjData(mj_model)
    initial_command = (
        np.asarray(command, dtype=np.float32)
        if env.command_config.enabled
        else np.zeros(3, dtype=np.float32)
    )
    joystick = CommandJoystick(
        initial_command,
        joystick_config,
        enabled=joystick_config.enabled and env.command_config.enabled,
    )
    command_array = joystick.command
    viewer_step_count = 0
    camera_dt = env.control_timestep
    base_geom_rgba = np.asarray(mj_model.geom_rgba, dtype=np.float32).copy()
    foot_geom_ids_by_foot = foot_geom_ids_for_bodies(mj_model, env.model.foot_body_ids)
    runtime_state = ViewerRuntimeState(
        realtime=realtime,
        stochastic_actions=stochastic_actions,
    )
    action_key = jax.random.PRNGKey(action_seed)

    last_action = np.zeros(env.action_size, dtype=np.float32)
    foot_air_time = jnp.zeros(env.model.foot_body_ids.size, dtype=jnp.float32)
    foot_contact_debounce_time = jnp.zeros(
        env.model.foot_body_ids.size, dtype=jnp.float32
    )
    last_foot_contact = jnp.zeros(env.model.foot_body_ids.size, dtype=bool)
    episode_return = jnp.asarray(0.0, dtype=jnp.float32)
    episode_track_lin_vel_xy = jnp.asarray(0.0, dtype=jnp.float32)
    episode_length = jnp.asarray(0, dtype=jnp.int32)
    episode_step_count = 0
    obs_stack = init_observation_stack(
        jnp.zeros(env.observation_size, dtype=jnp.float32), obs_stack_size
    )
    reward = jnp.asarray(0.0, dtype=jnp.float32)
    done = jnp.asarray(False)
    info: dict[str, jax.Array] = {}

    def reset_episode_state() -> None:
        nonlocal last_action
        nonlocal foot_air_time
        nonlocal foot_contact_debounce_time
        nonlocal last_foot_contact
        nonlocal episode_return
        nonlocal episode_track_lin_vel_xy
        nonlocal episode_length
        nonlocal episode_step_count
        nonlocal obs_stack
        nonlocal reward
        nonlocal done
        nonlocal info

        reset_mujoco_data(env, mj_data)
        last_action = np.zeros(env.action_size, dtype=np.float32)
        foot_air_time = jnp.zeros(env.model.foot_body_ids.size, dtype=jnp.float32)
        foot_contact_debounce_time = jnp.zeros(
            env.model.foot_body_ids.size, dtype=jnp.float32
        )
        last_foot_contact = jnp.zeros(env.model.foot_body_ids.size, dtype=bool)
        episode_return = jnp.asarray(0.0, dtype=jnp.float32)
        episode_track_lin_vel_xy = jnp.asarray(0.0, dtype=jnp.float32)
        episode_length = jnp.asarray(0, dtype=jnp.int32)
        episode_step_count = 0
        obs = mujoco_policy_observation(
            env,
            mj_data,
            command=command_array,
            last_action=last_action,
            step_count=episode_step_count,
        )
        obs_stack = init_observation_stack(obs, obs_stack_size)
        reward = jnp.asarray(0.0, dtype=jnp.float32)
        done = jnp.asarray(False)
        info = mujoco_viewer_info(env, mj_data, command=command_array)

    def key_callback(key: int) -> None:
        normalized_key = CommandJoystick._normalize_letter_key(key)
        if joystick.enabled:
            joystick.key_callback(key)
        if normalized_key == ord("P"):
            runtime_state.paused = not runtime_state.paused
        elif normalized_key == ord("N"):
            runtime_state.paused = True
            runtime_state.step_once = True
        elif normalized_key == ord("R"):
            runtime_state.reset_requested = True

    reset_episode_state()
    viewer = ImguiMujocoViewer(
        mj_model,
        camera_config,
        key_callback=key_callback,
    )
    try:
        while viewer.is_running() and (steps is None or viewer_step_count < steps):
            start = time.perf_counter()
            action_mean = apply_actor(policy, stacked_observation(obs_stack))
            action = action_mean
            if runtime_state.stochastic_actions:
                action_key, action_rng = jax.random.split(action_key)
                log_std = policy_log_std(policy)
                noise = jax.random.normal(action_rng, action_mean.shape)
                action = action_mean + jnp.exp(log_std) * noise
            if env.action_clip is not None:
                action = jnp.clip(action, -env.action_clip, env.action_clip)
            action_mean_np = np.asarray(jax.device_get(action_mean), dtype=np.float32)
            action = np.asarray(jax.device_get(action), dtype=np.float32)
            ctrl = mujoco_action_to_ctrl(env, action)
            overlay_state = SimpleNamespace(
                episode_return=episode_return,
                episode_length=episode_length,
            )
            foot_contact_visual = mujoco_foot_contact(env, mj_data)
            sync_foot_contact_colors(
                mj_model,
                base_geom_rgba,
                foot_geom_ids_by_foot,
                foot_contact_visual,
            )
            update_camera(viewer.camera, mj_model, mj_data, camera_config)

            def draw_ui() -> None:
                draw_policy_viewer_ui(
                    viewer.imgui,
                    runtime_state=runtime_state,
                    env=env,
                    command_array=command_array,
                    joystick_config=joystick_config,
                    reward_config=reward_overlay,
                    action_config=action_overlay,
                    step_count=viewer_step_count,
                    reward=reward,
                    done=done,
                    state=overlay_state,
                    info=info,
                    action_mean=action_mean_np,
                    action=action,
                    ctrl=ctrl,
                )

            viewer.render(
                mj_data,
                foot_geom_ids_by_foot=foot_geom_ids_by_foot,
                foot_contact=foot_contact_visual,
                draw_ui=draw_ui,
            )
            if runtime_state.reset_requested:
                runtime_state.reset_requested = False
                reset_episode_state()
                continue
            if not runtime_state.should_step():
                if runtime_state.realtime:
                    elapsed = time.perf_counter() - start
                    time.sleep(max(0.0, (1.0 / 60.0) - elapsed))
                continue

            state = mujoco_viewer_state(
                env,
                mj_data,
                command=command_array,
                last_action=last_action,
                foot_air_time=foot_air_time,
                foot_contact_debounce_time=foot_contact_debounce_time,
                last_foot_contact=last_foot_contact,
                step_count=episode_step_count,
                episode_return=episode_return,
                episode_track_lin_vel_xy=episode_track_lin_vel_xy,
                episode_length=episode_length,
            )
            mj_data.ctrl[:] = ctrl
            for _ in range(env.decimation):
                mujoco.mj_step(mj_model, mj_data)
            last_action = action
            episode_step_count += 1
            viewer_step_count += 1
            runtime_state.consume_step_once()
            done_flags = mujoco_done_flags(env, mj_data, episode_step_count)
            done = jnp.asarray(done_flags["done/any"])
            done_bool = bool(done_flags["done/any"])
            next_state = mujoco_viewer_state(
                env,
                mj_data,
                command=command_array,
                last_action=last_action,
                foot_air_time=foot_air_time,
                foot_contact_debounce_time=foot_contact_debounce_time,
                last_foot_contact=last_foot_contact,
                step_count=episode_step_count,
                episode_return=episode_return,
                episode_track_lin_vel_xy=episode_track_lin_vel_xy,
                episode_length=episode_length,
            )
            (
                raw_foot_contact,
                foot_height,
                foot_lateral_speed_sq,
                foot_angular_speed,
                undesired_contact,
                self_collision,
            ) = mujoco_contact_terms(env, mj_data)
            foot_contact, foot_contact_debounce_time = env._debounce_foot_contact(
                next_state, raw_foot_contact, done
            )
            reward, reward_info = env._reward(
                state,
                next_state,
                jnp.asarray(action, dtype=jnp.float32),
                jnp.asarray(done_flags["done/terminal"]),
                foot_contact,
                foot_height,
                foot_lateral_speed_sq,
                foot_angular_speed,
                undesired_contact,
                self_collision,
            )
            done_info = {key: jnp.asarray(value) for key, value in done_flags.items()}
            reward, reward_info = env._sanitize_nonfinite_reward(
                reward, reward_info, done_info
            )
            episode_return = episode_return + reward
            episode_track_lin_vel_xy = (
                episode_track_lin_vel_xy + reward_info["reward/track_lin_vel_xy"]
            )
            episode_length = episode_length + 1
            foot_air_time, last_foot_contact = env._next_foot_memory(
                next_state, foot_contact, done
            )
            info = {
                **mujoco_viewer_info(env, mj_data, command=command_array),
                "episode_return": jnp.where(done, episode_return, 0.0),
                "episode_reward/track_lin_vel_xy": jnp.where(
                    done, episode_track_lin_vel_xy, 0.0
                ),
                "episode_length": jnp.where(done, episode_length, 0),
                "foot_contact/raw": raw_foot_contact,
                "foot_contact/filtered": foot_contact,
                "foot_contact/debounce_time": foot_contact_debounce_time,
                "returned_episode": done,
                **done_info,
                **reward_info,
            }
            if done_bool and not stop_on_done:
                reset_episode_state()
            else:
                obs = mujoco_policy_observation(
                    env,
                    mj_data,
                    command=command_array,
                    last_action=last_action,
                    step_count=episode_step_count,
                )
                obs_stack = update_observation_stack(obs_stack, obs, done)
            if stop_on_done and done_bool:
                break
            if runtime_state.realtime:
                elapsed = time.perf_counter() - start
                time.sleep(max(0.0, camera_dt - elapsed))
    finally:
        mj_model.geom_rgba[:] = base_geom_rgba
        viewer.close()


def mujoco_viewer_state(
    env: BelloLocomotionEnv,
    mj_data: mujoco.MjData,
    *,
    command: np.ndarray,
    last_action: np.ndarray,
    foot_air_time: jax.Array,
    foot_contact_debounce_time: jax.Array,
    last_foot_contact: jax.Array,
    step_count: int,
    episode_return: jax.Array,
    episode_track_lin_vel_xy: jax.Array,
    episode_length: jax.Array,
):
    return SimpleNamespace(
        data=SimpleNamespace(
            qpos=jnp.asarray(np.asarray(mj_data.qpos, dtype=np.float32).copy()),
            qvel=jnp.asarray(np.asarray(mj_data.qvel, dtype=np.float32).copy()),
            ctrl=jnp.asarray(np.asarray(mj_data.ctrl, dtype=np.float32).copy()),
            xpos=jnp.asarray(np.asarray(mj_data.xpos, dtype=np.float32).copy()),
            xquat=jnp.asarray(np.asarray(mj_data.xquat, dtype=np.float32).copy()),
            cvel=jnp.asarray(np.asarray(mj_data.cvel, dtype=np.float32).copy()),
            actuator_force=jnp.asarray(
                np.asarray(mj_data.actuator_force, dtype=np.float32).copy()
            ),
        ),
        last_action=jnp.asarray(last_action, dtype=jnp.float32),
        foot_air_time=foot_air_time,
        foot_contact_debounce_time=foot_contact_debounce_time,
        last_foot_contact=last_foot_contact,
        command=jnp.asarray(command, dtype=jnp.float32),
        step_count=jnp.asarray(step_count, dtype=jnp.int32),
        episode_return=episode_return,
        episode_track_lin_vel_xy=episode_track_lin_vel_xy,
        episode_length=episode_length,
        last_episode_return=jnp.asarray(0.0, dtype=jnp.float32),
        last_episode_track_lin_vel_xy=jnp.asarray(0.0, dtype=jnp.float32),
        last_episode_length=jnp.asarray(0, dtype=jnp.int32),
    )


def mujoco_viewer_info(
    env: BelloLocomotionEnv,
    mj_data: mujoco.MjData,
    *,
    command: np.ndarray | None = None,
) -> dict[str, jax.Array]:
    info = {
        "base_height": jnp.asarray(
            mujoco_link_height(
                env, mj_data, env._reward_link(env.reward_config.base_height)
            ),
            dtype=jnp.float32,
        )
    }
    if command is not None:
        info["command"] = jnp.asarray(command, dtype=jnp.float32)
    return info


def mujoco_contact_terms(
    env: BelloLocomotionEnv, mj_data: mujoco.MjData
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    return (
        jnp.asarray(mujoco_foot_contact(env, mj_data), dtype=jnp.float32),
        jnp.asarray(mujoco_foot_height(env, mj_data), dtype=jnp.float32),
        jnp.asarray(mujoco_foot_lateral_speed_sq(env, mj_data), dtype=jnp.float32),
        jnp.asarray(mujoco_foot_angular_speed(env, mj_data), dtype=jnp.float32),
        jnp.asarray(mujoco_undesired_contact(env, mj_data), dtype=jnp.float32),
        jnp.asarray(mujoco_self_collision(env, mj_data), dtype=jnp.float32),
    )


def mujoco_gait_phase(
    env: BelloLocomotionEnv, step_count: int, command: np.ndarray
) -> float:
    base_frequency = 1.0 / float(env.config.gait.cycle_time)
    command_speed = float(np.linalg.norm(np.asarray(command, dtype=np.float32)[:2]))
    frequency = np.clip(
        base_frequency + env.config.gait.command_speed_frequency_gain * command_speed,
        env.config.gait.min_frequency,
        env.config.gait.max_frequency,
    )
    return float((step_count * env.control_timestep * frequency) % 1.0)


def viewer_foot_contact(env: BelloLocomotionEnv, state) -> np.ndarray:
    foot_contact, *_ = env._contact_terms(state, jnp.asarray(0, dtype=jnp.int32))
    return np.asarray(jax.device_get(foot_contact), dtype=np.float32)


def mujoco_foot_contact(env: BelloLocomotionEnv, mj_data: mujoco.MjData) -> np.ndarray:
    foot_contact = np.zeros(env.model.foot_body_ids.size, dtype=np.float32)
    if env.model.terrain_geom_ids.size == 0:
        return foot_contact
    force_threshold = float(env.config.contact.foot_contact_force_threshold)
    terrain_geom_ids = set(int(geom_id) for geom_id in env.model.terrain_geom_ids)
    foot_body_ids = np.asarray(env.model.foot_body_ids, dtype=np.int32)
    foot_body_index = {
        int(body_id): foot_index for foot_index, body_id in enumerate(foot_body_ids)
    }
    geom_body_ids = np.asarray(env.model.geom_body_ids, dtype=np.int32)
    for contact_index in range(int(mj_data.ncon)):
        contact = mj_data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if geom1 in terrain_geom_ids:
            other_geom = geom2
        elif geom2 in terrain_geom_ids:
            other_geom = geom1
        else:
            continue
        if other_geom < 0 or other_geom >= geom_body_ids.shape[0]:
            continue
        normal_force = mujoco_contact_normal_force(
            env.model.mj_model, mj_data, contact_index
        )
        if normal_force <= force_threshold:
            continue
        foot_index = foot_body_index.get(int(geom_body_ids[other_geom]))
        if foot_index is not None:
            foot_contact[foot_index] = 1.0
    return foot_contact


def mujoco_foot_height(env: BelloLocomotionEnv, mj_data: mujoco.MjData) -> np.ndarray:
    if env.model.foot_body_ids.size == 0:
        return np.zeros((0,), dtype=np.float32)
    terrain_z = np.float32(env.terrain_config.pos[2])
    return (
        np.asarray(mj_data.xpos[env.model.foot_body_ids, 2], dtype=np.float32)
        - terrain_z
    )


def mujoco_foot_lateral_speed_sq(
    env: BelloLocomotionEnv, mj_data: mujoco.MjData
) -> np.ndarray:
    if env.model.foot_body_ids.size == 0:
        return np.zeros((0,), dtype=np.float32)
    foot_body_ids = np.asarray(env.model.foot_body_ids, dtype=np.int32)
    foot_cvel = np.asarray(mj_data.cvel[foot_body_ids, 3:6], dtype=np.float32)
    foot_quat = np.asarray(mj_data.xquat[foot_body_ids], dtype=np.float32)
    foot_world_vel = np.stack(
        [quat_rotate_np(quat, vel) for quat, vel in zip(foot_quat, foot_cvel)],
        axis=0,
    )
    return np.sum(foot_world_vel[:, :2] ** 2, axis=1, dtype=np.float32)


def mujoco_foot_angular_speed(
    env: BelloLocomotionEnv, mj_data: mujoco.MjData
) -> np.ndarray:
    if env.model.foot_body_ids.size == 0:
        return np.zeros((0,), dtype=np.float32)
    foot_body_ids = np.asarray(env.model.foot_body_ids, dtype=np.int32)
    foot_ang_vel = np.asarray(mj_data.cvel[foot_body_ids, :3], dtype=np.float32)
    return np.linalg.norm(foot_ang_vel, axis=1).astype(np.float32)


def mujoco_undesired_contact(
    env: BelloLocomotionEnv, mj_data: mujoco.MjData
) -> np.float32:
    if env.model.terrain_geom_ids.size == 0:
        return np.float32(0.0)
    terrain_geom_ids = set(int(geom_id) for geom_id in env.model.terrain_geom_ids)
    nonfoot_geom_ids = set(int(geom_id) for geom_id in env.model.nonfoot_robot_geom_ids)
    nonfoot_body_ids = set(int(body_id) for body_id in env.model.nonfoot_robot_body_ids)
    geom_body_ids = np.asarray(env.model.geom_body_ids, dtype=np.int32)
    force_threshold = float(env.config.contact.undesired_contact_force_threshold)
    contacted_body_ids: set[int] = set()
    for contact_index in range(int(mj_data.ncon)):
        contact = mj_data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if geom1 in terrain_geom_ids:
            other_geom = geom2
        elif geom2 in terrain_geom_ids:
            other_geom = geom1
        else:
            continue
        if other_geom not in nonfoot_geom_ids:
            continue
        normal_force = mujoco_contact_normal_force(
            env.model.mj_model, mj_data, contact_index
        )
        if normal_force <= force_threshold:
            continue
        body_id = int(geom_body_ids[other_geom])
        if body_id in nonfoot_body_ids:
            contacted_body_ids.add(body_id)
    return np.float32(len(contacted_body_ids))


def mujoco_self_collision(env: BelloLocomotionEnv, mj_data: mujoco.MjData) -> np.float32:
    robot_geom_ids = set(int(geom_id) for geom_id in env.model.robot_geom_ids)
    if not robot_geom_ids:
        return np.float32(0.0)
    force_threshold = float(env.config.contact.self_collision_force_threshold)
    count = 0
    for contact_index in range(int(mj_data.ncon)):
        contact = mj_data.contact[contact_index]
        if int(contact.geom1) not in robot_geom_ids:
            continue
        if int(contact.geom2) not in robot_geom_ids:
            continue
        normal_force = mujoco_contact_normal_force(
            env.model.mj_model, mj_data, contact_index
        )
        if normal_force > force_threshold:
            count += 1
    return np.float32(count)


def mujoco_contact_normal_force(
    mj_model: mujoco.MjModel, mj_data: mujoco.MjData, contact_index: int
) -> float:
    contact_force = np.zeros(6, dtype=np.float64)
    mujoco.mj_contactForce(mj_model, mj_data, contact_index, contact_force)
    return abs(float(contact_force[0]))


def foot_geom_ids_for_bodies(
    mj_model: mujoco.MjModel, foot_body_ids: np.ndarray
) -> tuple[np.ndarray, ...]:
    return tuple(
        np.asarray(
            np.nonzero(np.asarray(mj_model.geom_bodyid) == int(body_id))[0],
            dtype=np.int32,
        )
        for body_id in np.asarray(foot_body_ids, dtype=np.int32)
    )


def sync_foot_contact_colors(
    mj_model: mujoco.MjModel,
    base_geom_rgba: np.ndarray,
    foot_geom_ids_by_foot: tuple[np.ndarray, ...],
    foot_contact: np.ndarray,
    *,
    contact_rgba: np.ndarray = CONTACT_FOOT_RGBA,
) -> None:
    for foot_index, geom_ids in enumerate(foot_geom_ids_by_foot):
        if geom_ids.size == 0:
            continue
        contacted = (
            foot_index < foot_contact.shape[0]
            and float(foot_contact[foot_index]) > 0.5
        )
        mj_model.geom_rgba[geom_ids] = (
            np.asarray(contact_rgba, dtype=np.float32)
            if contacted
            else base_geom_rgba[geom_ids]
        )


def sync_foot_contact_markers(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    foot_geom_ids_by_foot: tuple[np.ndarray, ...],
    foot_contact: np.ndarray,
    user_scn: mujoco.MjvScene | None,
    *,
    contact_rgba: np.ndarray = CONTACT_FOOT_RGBA,
) -> None:
    if user_scn is None:
        return
    user_scn.ngeom = 0
    append_foot_contact_markers(
        mj_model,
        mj_data,
        foot_geom_ids_by_foot,
        foot_contact,
        user_scn,
        contact_rgba=contact_rgba,
    )


def sync_renderer_foot_contact_markers(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    foot_geom_ids_by_foot: tuple[np.ndarray, ...],
    foot_contact: np.ndarray,
    scene: mujoco.MjvScene,
    *,
    contact_rgba: np.ndarray = CONTACT_FOOT_RGBA,
) -> None:
    append_foot_contact_markers(
        mj_model,
        mj_data,
        foot_geom_ids_by_foot,
        foot_contact,
        scene,
        contact_rgba=contact_rgba,
    )


def append_foot_contact_markers(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    foot_geom_ids_by_foot: tuple[np.ndarray, ...],
    foot_contact: np.ndarray,
    scene: mujoco.MjvScene,
    *,
    contact_rgba: np.ndarray = CONTACT_FOOT_RGBA,
) -> None:
    for foot_index, geom_ids in enumerate(foot_geom_ids_by_foot):
        contacted = (
            foot_index < foot_contact.shape[0]
            and float(foot_contact[foot_index]) > 0.5
        )
        if not contacted:
            continue
        for geom_id in geom_ids:
            if scene.ngeom >= scene.maxgeom:
                return
            add_foot_contact_marker(
                mj_model,
                mj_data,
                scene,
                int(geom_id),
                contact_rgba=contact_rgba,
            )


def add_foot_contact_marker(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    user_scn: mujoco.MjvScene,
    geom_id: int,
    *,
    contact_rgba: np.ndarray = CONTACT_FOOT_RGBA,
) -> None:
    size = np.asarray(mj_model.geom_size[geom_id], dtype=np.float64).copy()
    if not np.all(np.isfinite(size)) or np.all(size <= 0.0):
        radius = max(float(mj_model.geom_rbound[geom_id]), 0.03)
        size = np.asarray([radius, 0.6 * radius, 0.3 * radius], dtype=np.float64)
    size = np.maximum(size, np.asarray([0.03, 0.03, 0.015], dtype=np.float64))
    size = size * 1.25
    pos = np.asarray(mj_data.geom_xpos[geom_id], dtype=np.float64)
    mat = np.asarray(mj_data.geom_xmat[geom_id], dtype=np.float64).reshape(9)
    rgba = np.asarray(contact_rgba, dtype=np.float32)
    mujoco.mjv_initGeom(
        user_scn.geoms[user_scn.ngeom],
        mujoco.mjtGeom.mjGEOM_BOX,
        size,
        pos,
        mat,
        rgba,
    )
    user_scn.ngeom += 1


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


def draw_policy_viewer_ui(
    imgui,
    *,
    runtime_state: ViewerRuntimeState,
    env: BelloLocomotionEnv,
    command_array: np.ndarray,
    joystick_config: JoystickConfig,
    reward_config: RewardOverlayConfig,
    action_config: ActionOverlayConfig,
    step_count: int,
    reward: jax.Array,
    done: jax.Array,
    state,
    info: dict[str, jax.Array],
    action_mean: np.ndarray,
    action: np.ndarray,
    ctrl: np.ndarray,
) -> None:
    if reward_config.enabled:
        draw_rows_panel(
            imgui,
            "Status",
            reward_status_rows(
                reward_config,
                step_count=step_count,
                reward=reward,
                done=done,
                state=state,
                info=info,
            ),
            position=imgui_overlay_position(imgui, reward_config, panel_index=0),
            font_scale=imgui_overlay_font_scale(reward_config),
        )
        if reward_config.show_components:
            draw_rows_panel(
                imgui,
                "Rewards",
                reward_component_rows(reward_config, info),
                position=imgui_overlay_position(imgui, reward_config, panel_index=1),
                font_scale=imgui_overlay_font_scale(reward_config),
            )
    draw_command_panel(
        imgui,
        env=env,
        command_array=command_array,
        joystick_config=joystick_config,
    )
    draw_controls_panel(imgui, runtime_state)
    if action_config.enabled:
        draw_rows_panel(
            imgui,
            "Actions",
            action_overlay_rows(
                action_config,
                env=env,
                action_mean=action_mean,
                action=action,
                ctrl=ctrl,
            ),
        )


def draw_rows_panel(
    imgui,
    title: str,
    rows: list[tuple[str, str]],
    *,
    position: tuple[float, float] | None = None,
    font_scale: float = 1.0,
) -> None:
    if not rows:
        return
    if position is not None:
        imgui.set_next_window_position(*position, condition=imgui.ONCE)
        imgui.set_next_window_size(330, 0, condition=imgui.ONCE)
    imgui.begin(title)
    imgui.set_window_font_scale(font_scale)
    for label, value in rows:
        imgui.text(label)
        imgui.same_line(180)
        imgui.text(value)
    imgui.end()


def imgui_overlay_position(
    imgui,
    config: RewardOverlayConfig,
    *,
    panel_index: int,
) -> tuple[float, float]:
    display_width, display_height = imgui.get_io().display_size
    margin = 10.0
    panel_width = 330.0
    panel_height = 220.0
    if config.position.endswith("right"):
        x = max(margin, float(display_width) - panel_width - margin)
    else:
        x = margin
    if config.position.startswith("bottom"):
        y = max(margin, float(display_height) - ((panel_index + 1) * panel_height))
    else:
        y = margin + (panel_index * panel_height)
    return x, y


def imgui_overlay_font_scale(config: RewardOverlayConfig) -> float:
    scales = {
        "small": 0.85,
        "normal": 1.0,
        "large": 1.25,
    }
    return scales[config.font_scale]


def draw_command_panel(
    imgui,
    *,
    env: BelloLocomotionEnv,
    command_array: np.ndarray,
    joystick_config: JoystickConfig,
) -> None:
    imgui.begin("Command")
    if not env.command_config.enabled:
        imgui.text("disabled")
        imgui.end()
        return
    changed, command_x = imgui.slider_float(
        "cmd_x",
        float(command_array[0]),
        float(joystick_config.lin_vel_x_limit[0]),
        float(joystick_config.lin_vel_x_limit[1]),
        "%.2f",
    )
    if changed:
        command_array[0] = command_x
    changed, command_y = imgui.slider_float(
        "cmd_y",
        float(command_array[1]),
        float(joystick_config.lin_vel_y_limit[0]),
        float(joystick_config.lin_vel_y_limit[1]),
        "%.2f",
    )
    if changed:
        command_array[1] = command_y
    changed, command_yaw = imgui.slider_float(
        "cmd_yaw",
        float(command_array[2]),
        float(joystick_config.yaw_vel_limit[0]),
        float(joystick_config.yaw_vel_limit[1]),
        "%.2f",
    )
    if changed:
        command_array[2] = command_yaw
    if imgui.button("Zero"):
        command_array[:] = 0.0
    imgui.end()


def draw_controls_panel(imgui, runtime_state: ViewerRuntimeState) -> None:
    imgui.begin("Controls")
    if imgui.button("Resume" if runtime_state.paused else "Pause"):
        runtime_state.paused = not runtime_state.paused
    imgui.same_line()
    if imgui.button("Step"):
        runtime_state.paused = True
        runtime_state.step_once = True
    imgui.same_line()
    if imgui.button("Reset"):
        runtime_state.reset_requested = True
    _, runtime_state.realtime = imgui.checkbox("Realtime", runtime_state.realtime)
    _, runtime_state.stochastic_actions = imgui.checkbox(
        "Stochastic", runtime_state.stochastic_actions
    )
    imgui.end()


def action_overlay_rows(
    config: ActionOverlayConfig,
    *,
    env: BelloLocomotionEnv,
    action_mean: np.ndarray,
    action: np.ndarray,
    ctrl: np.ndarray,
) -> list[tuple[str, str]]:
    action_mean = np.asarray(action_mean, dtype=np.float32)
    action = np.asarray(action, dtype=np.float32)
    rows = [
        ("action_mean_abs_max", f"{float(np.max(np.abs(action_mean))):.2f}"),
        ("action_abs_max", f"{float(np.max(np.abs(action))):.2f}"),
        ("ctrl_saturation", f"{ctrl_saturation_fraction(env, ctrl):.3f}"),
    ]
    action_limit = float(getattr(env, "action_clip", 1.0) or 1.0)
    max_actions = min(max(config.max_actions, 0), int(action.size))
    names = policy_action_names(env)
    for action_id in range(max_actions):
        name = names[action_id] if action_id < len(names) else f"action_{action_id}"
        mean_value = float(action_mean[action_id])
        action_value = float(action[action_id])
        rows.append(
            (
                compact_action_name(name),
                f"m {mean_value:+.2f} "
                f"{signed_bar(mean_value, action_limit, config.bar_width)} "
                f"a {action_value:+.2f} "
                f"{signed_bar(action_value, action_limit, config.bar_width)}",
            )
        )
    hidden = int(action.size) - max_actions
    if hidden > 0:
        rows.append(("actions_hidden", str(hidden)))
    return rows


def join_overlay_rows(rows: list[tuple[str, str]]) -> tuple[str, str]:
    labels = "\n".join(label for label, _ in rows)
    values = "\n".join(value for _, value in rows)
    return labels, values


def format_action_overlay(
    config: ActionOverlayConfig,
    *,
    env: BelloLocomotionEnv,
    action_mean: np.ndarray,
    action: np.ndarray,
    ctrl: np.ndarray,
) -> tuple[str, str]:
    return join_overlay_rows(
        action_overlay_rows(
            config, env=env, action_mean=action_mean, action=action, ctrl=ctrl
        )
    )


def ctrl_saturation_fraction(env: BelloLocomotionEnv, ctrl: np.ndarray) -> float:
    policy_indices = np.asarray(env.model.policy_ctrl_indices, dtype=np.int32)
    if policy_indices.size == 0:
        return 0.0
    ctrl = np.asarray(ctrl, dtype=np.float32)
    ctrlrange = np.asarray(env.model.mj_model.actuator_ctrlrange, dtype=np.float32)
    policy_ctrl = ctrl[policy_indices]
    policy_range = ctrlrange[policy_indices]
    saturated = (policy_ctrl <= policy_range[:, 0] + 1.0e-6) | (
        policy_ctrl >= policy_range[:, 1] - 1.0e-6
    )
    return float(np.mean(saturated.astype(np.float32)))


def policy_action_names(env: BelloLocomotionEnv) -> list[str]:
    names = getattr(env.model, "actuator_joint_names", ())
    policy_indices = np.asarray(env.model.policy_ctrl_indices, dtype=np.int32)
    return [names[int(index)] for index in policy_indices if int(index) < len(names)]


def compact_action_name(name: str) -> str:
    return (
        name.removeprefix("left_")
        .removeprefix("right_")
        .removesuffix("_joint")
        .replace("hip_", "h_")
        .replace("ankle_", "a_")
    )


def signed_bar(value: float, limit: float, width: int) -> str:
    width = max(int(width), 3)
    limit = max(float(limit), 1.0e-6)
    center = width // 2
    chars = ["-"] * width
    chars[center] = "|"
    normalized = float(np.clip(value / limit, -1.0, 1.0))
    if normalized > 0.0:
        cells = int(round(normalized * (width - center - 1)))
        for idx in range(center + 1, min(width, center + 1 + cells)):
            chars[idx] = "#"
    elif normalized < 0.0:
        cells = int(round(abs(normalized) * center))
        for idx in range(max(0, center - cells), center):
            chars[idx] = "#"
    return "[" + "".join(chars) + "]"


def format_reward_overlay(
    config: RewardOverlayConfig,
    *,
    step_count: int,
    reward: jax.Array,
    done: jax.Array,
    state,
    info: dict[str, jax.Array],
) -> tuple[str, str]:
    return join_overlay_rows(
        reward_status_rows(
            config,
            step_count=step_count,
            reward=reward,
            done=done,
            state=state,
            info=info,
        )
        + reward_component_rows(config, info)
    )


def reward_status_rows(
    config: RewardOverlayConfig,
    *,
    step_count: int,
    reward: jax.Array,
    done: jax.Array,
    state,
    info: dict[str, jax.Array],
) -> list[tuple[str, str]]:
    rows = [
        ("step", str(step_count)),
        ("reward", format_scalar(reward, config.precision)),
        ("done", str(bool(jax.device_get(done)))),
        ("episode_return", format_scalar(state.episode_return, config.precision)),
        ("episode_length", str(int(jax.device_get(state.episode_length)))),
    ]
    if "base_height" in info:
        rows.append(("base_height", format_scalar(info["base_height"], config.precision)))
    if "command" in info:
        command = np.asarray(jax.device_get(info["command"]), dtype=np.float32)
        rows.extend(
            [
                ("cmd_x", f"{float(command[0]):.{config.precision}f}"),
                ("cmd_y", f"{float(command[1]):.{config.precision}f}"),
                ("cmd_yaw", f"{float(command[2]):.{config.precision}f}"),
            ]
        )
    return rows


def reward_component_rows(
    config: RewardOverlayConfig,
    info: dict[str, jax.Array],
) -> list[tuple[str, str]]:
    if not config.show_components:
        return []
    rows = []
    for name in sorted(info):
        if not name.startswith("reward/"):
            continue
        if len(rows) >= config.max_reward_terms:
            break
        rows.append(
            (
                name.removeprefix("reward/"),
                format_scalar(info[name], config.precision),
            )
        )
    return rows


def format_scalar(value: jax.Array, precision: int) -> str:
    return f"{float(jax.device_get(value)):.{precision}f}"


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
