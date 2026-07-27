"""Shared native MuJoCo rendering and evaluation math."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal
from typing import TypeAlias

import imageio_ffmpeg
import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from bello_mujoco.envs.locomotion import BelloLocomotionEnv, BelloState
from bello_mujoco.policy import (
    apply_actor,
    init_observation_stack,
    sample_actor_action,
    stacked_observation,
    update_observation_stack,
)
from bello_mujoco.terrain import terrain_surface_height

RolloutStats: TypeAlias = dict[str, float | int]
DONE_FLAG_KEYS = (
    "done/any",
    "done/terminal",
    "done/base_height",
    "done/bad_orientation",
    "done/timeout",
    "done/nonfinite",
    "done/unstable",
)


@dataclass(frozen=True)
class CameraConfig:
    mode: Literal["follow", "fixed"] = "follow"
    distance: float = 3.0
    azimuth: float = 135.0
    elevation: float = -18.0
    lookat_offset: tuple[float, float, float] = (0.0, 0.0, 0.35)


@dataclass(frozen=True)
class NativeDataImplementation:
    cacc: jax.Array
    actuator_velocity: jax.Array
    efc__force: jax.Array


@dataclass(frozen=True)
class NativeDataView:
    """JAX-array view of native MuJoCo data used by shared task functions."""

    qpos: jax.Array
    qvel: jax.Array
    qacc: jax.Array
    ctrl: jax.Array
    xpos: jax.Array
    xquat: jax.Array
    cvel: jax.Array
    subtree_com: jax.Array
    geom_xpos: jax.Array
    geom_xmat: jax.Array
    actuator_force: jax.Array
    qfrc_constraint: jax.Array
    _impl: NativeDataImplementation


class VideoWriter:
    """Raw-RGB FFmpeg writer started with ``posix_spawn``.

    Python's usual subprocess path may call ``fork``. Forking after JAX has
    created worker threads can deadlock, so rollout video uses the OS's atomic
    spawn-and-exec primitive directly.
    """

    def __init__(self, path: Path, *, width: int, height: int, fps: int) -> None:
        if not hasattr(os, "posix_spawn"):
            raise RuntimeError("rollout video requires os.posix_spawn support")
        path.parent.mkdir(parents=True, exist_ok=True)
        executable = imageio_ffmpeg.get_ffmpeg_exe()
        read_fd, write_fd = os.pipe()
        argv = [
            executable,
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
        file_actions = [
            (os.POSIX_SPAWN_DUP2, read_fd, 0),
            (os.POSIX_SPAWN_CLOSE, read_fd),
            (os.POSIX_SPAWN_CLOSE, write_fd),
        ]
        try:
            self._pid = os.posix_spawn(
                executable,
                argv,
                os.environ.copy(),
                file_actions=file_actions,
            )
        except BaseException:
            os.close(read_fd)
            os.close(write_fd)
            raise
        os.close(read_fd)
        try:
            self._stream = os.fdopen(write_fd, "wb")
        except BaseException:
            try:
                os.close(write_fd)
            except OSError:
                pass
            os.waitpid(self._pid, 0)
            raise
        self._shape = (height, width, 3)
        self._closed = False

    def append_data(self, frame: np.ndarray) -> None:
        image = np.asarray(frame, dtype=np.uint8)
        if image.shape != self._shape:
            raise ValueError(
                f"video frame has shape {image.shape}, expected {self._shape}"
            )
        self._stream.write(image.tobytes(order="C"))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        stream_error: BaseException | None = None
        try:
            self._stream.close()
        except BaseException as error:
            stream_error = error
        _, status = os.waitpid(self._pid, 0)
        if stream_error is not None:
            raise stream_error
        if status != 0:
            exit_code = os.waitstatus_to_exitcode(status)
            raise RuntimeError(f"FFmpeg video encoder exited with status {exit_code}")


def open_rollout_video_resources(
    mj_model: mujoco.MjModel,
    output_path: Path,
    *,
    width: int,
    height: int,
    fps: int,
    camera_config: CameraConfig,
):
    """Acquire rollout rendering resources with failure-safe cleanup."""

    resources = ExitStack()
    try:
        renderer = mujoco.Renderer(mj_model, height=height, width=width)
        resources.callback(renderer.close)
        camera = make_camera(mj_model, camera_config)
        writer = VideoWriter(output_path, width=width, height=height, fps=fps)
        resources.callback(writer.close)
    except BaseException:
        resources.close()
        raise
    return resources, renderer, camera, writer


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
    obs_stack_size: int = 1,
    stochastic_actions: bool = False,
    action_seed: int = 0,
) -> RolloutStats:
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
    action_key = jax.random.PRNGKey(action_seed)
    stats = init_rollout_stats(
        steps, stochastic_actions=stochastic_actions, action_seed=action_seed
    )
    resources, renderer, camera, writer = open_rollout_video_resources(
        mj_model,
        output_path,
        width=width,
        height=height,
        fps=fps,
        camera_config=camera_config,
    )
    try:
        for step_index in range(steps):
            update_camera(camera, mj_model, mj_data, camera_config)
            renderer.update_scene(mj_data, camera=camera)
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
        resources.close()
    return finalize_rollout_stats(stats, current_episode_length=step_count)


def rollout_policy_action(
    policy,
    obs_stack: jax.Array,
    *,
    action_rng: jax.Array | None = None,
    action_clip: float,
) -> jax.Array:
    obs = stacked_observation(obs_stack)
    mean = apply_actor(policy, obs)
    action = mean
    if action_rng is not None:
        action = sample_actor_action(policy, obs, action_rng)
    return jnp.clip(action, -action_clip, action_clip)


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
    stats["_completed_episode_length_sum"] = float(
        stats["_completed_episode_length_sum"]
    ) + float(episode_length)
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
    state = mujoco_eval_state(
        env,
        mj_data,
        command=command,
        last_action=last_action,
        foot_air_time=jnp.zeros(env.model.foot_body_ids.size, dtype=jnp.float32),
        foot_contact_time=jnp.zeros(env.model.foot_body_ids.size, dtype=jnp.float32),
        foot_contact_debounce_time=jnp.zeros(
            env.model.foot_body_ids.size, dtype=jnp.float32
        ),
        last_foot_contact=jnp.zeros(env.model.foot_body_ids.size, dtype=jnp.float32),
        step_count=step_count,
        episode_return=jnp.asarray(0.0, dtype=jnp.float32),
        episode_track_lin_vel_xy=jnp.asarray(0.0, dtype=jnp.float32),
        episode_length=jnp.asarray(0, dtype=jnp.int32),
    )
    return env.observation(state)


def reset_mujoco_data(env: BelloLocomotionEnv, mj_data: mujoco.MjData) -> None:
    mujoco.mj_resetData(env.model.mj_model, mj_data)
    mj_data.qpos[:] = env.model.default_qpos
    mj_data.qvel[:] = env.model.default_qvel
    mj_data.ctrl[:] = mujoco_action_to_ctrl(
        env, np.zeros(env.action_size, dtype=np.float32)
    )
    mujoco.mj_forward(env.model.mj_model, mj_data)
    if env.model.foot_body_ids.size:
        clearance_adjustment = env.config.spawn.clearance - float(
            np.min(mujoco_foot_height(env, mj_data))
        )
        mj_data.qpos[env.model.root_qpos_adr + 2] += clearance_adjustment
        mujoco.mj_forward(env.model.mj_model, mj_data)


def mujoco_action_to_ctrl(env: BelloLocomotionEnv, action: np.ndarray) -> np.ndarray:
    return np.asarray(
        jax.device_get(env.action_to_ctrl(jnp.asarray(action, dtype=jnp.float32))),
        dtype=np.float32,
    )


def mujoco_done_flags(
    env: BelloLocomotionEnv, mj_data: mujoco.MjData, step_count: int
) -> dict[str, bool]:
    zeros = jnp.zeros(env.model.foot_body_ids.size, dtype=jnp.float32)
    state = mujoco_eval_state(
        env,
        mj_data,
        command=np.zeros(3, dtype=np.float32),
        last_action=np.zeros(env.action_size, dtype=np.float32),
        foot_air_time=zeros,
        foot_contact_time=zeros,
        foot_contact_debounce_time=zeros,
        last_foot_contact=zeros,
        step_count=step_count,
        episode_return=jnp.asarray(0.0, dtype=jnp.float32),
        episode_track_lin_vel_xy=jnp.asarray(0.0, dtype=jnp.float32),
        episode_length=jnp.asarray(step_count, dtype=jnp.int32),
    )
    return {
        name: bool(jax.device_get(value))
        for name, value in env.done_flags(state).items()
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


def mujoco_link_clearance(
    env: BelloLocomotionEnv, mj_data: mujoco.MjData, link_name: str
) -> float:
    if link_name == "bello_root":
        root_qpos = int(env.model.root_qpos_adr)
        position = np.asarray(mj_data.qpos[root_qpos : root_qpos + 3])
    else:
        body_id = mujoco.mj_name2id(
            env.model.mj_model, mujoco.mjtObj.mjOBJ_BODY, link_name
        )
        if body_id < 0:
            raise ValueError(f"configured link/body not found: {link_name!r}")
        position = np.asarray(mj_data.xpos[body_id])

    if env.terrain_config.mode == "none":
        terrain_height = env.terrain_config.pos[2]
    else:
        terrain_height = terrain_surface_height(
            env.model.mj_model,
            int(env.model.terrain_geom_ids[0]),
            position[None, :2],
        )[0]
    return float(position[2] - terrain_height)


def mujoco_link_lin_vel(
    env: BelloLocomotionEnv, mj_data: mujoco.MjData, link_name: str
) -> np.ndarray:
    if link_name == "bello_root":
        root_qpos = int(env.model.root_qpos_adr)
        root_qvel = int(env.model.root_qvel_adr)
        return quat_rotate_inverse_np(
            np.asarray(mj_data.qpos[root_qpos + 3 : root_qpos + 7], dtype=np.float32),
            np.asarray(mj_data.qvel[root_qvel : root_qvel + 3], dtype=np.float32),
        )
    body_id = mujoco.mj_name2id(env.model.mj_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
    if body_id < 0:
        raise ValueError(f"configured link/body not found: {link_name!r}")
    velocity = np.empty(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        env.model.mj_model,
        mj_data,
        mujoco.mjtObj.mjOBJ_XBODY,
        body_id,
        velocity,
        1,
    )
    return np.asarray(velocity[3:6], dtype=np.float32)


def mujoco_link_ang_vel(
    env: BelloLocomotionEnv, mj_data: mujoco.MjData, link_name: str
) -> np.ndarray:
    if link_name == "bello_root":
        root_qvel = int(env.model.root_qvel_adr)
        return np.asarray(mj_data.qvel[root_qvel + 3 : root_qvel + 6], dtype=np.float32)
    body_id = mujoco.mj_name2id(env.model.mj_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
    if body_id < 0:
        raise ValueError(f"configured link/body not found: {link_name!r}")
    velocity = np.empty(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        env.model.mj_model,
        mj_data,
        mujoco.mjtObj.mjOBJ_XBODY,
        body_id,
        velocity,
        1,
    )
    return np.asarray(velocity[:3], dtype=np.float32)


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


def mujoco_eval_state(
    env: BelloLocomotionEnv,
    mj_data: mujoco.MjData,
    *,
    command: np.ndarray,
    last_action: np.ndarray,
    foot_air_time: jax.Array,
    foot_contact_time: jax.Array,
    foot_contact_debounce_time: jax.Array,
    last_foot_contact: jax.Array,
    step_count: int,
    episode_return: jax.Array,
    episode_track_lin_vel_xy: jax.Array,
    episode_length: jax.Array,
) -> BelloState:
    data = NativeDataView(
        qpos=jnp.asarray(np.asarray(mj_data.qpos, dtype=np.float32).copy()),
        qvel=jnp.asarray(np.asarray(mj_data.qvel, dtype=np.float32).copy()),
        qacc=jnp.asarray(np.asarray(mj_data.qacc, dtype=np.float32).copy()),
        ctrl=jnp.asarray(np.asarray(mj_data.ctrl, dtype=np.float32).copy()),
        xpos=jnp.asarray(np.asarray(mj_data.xpos, dtype=np.float32).copy()),
        xquat=jnp.asarray(np.asarray(mj_data.xquat, dtype=np.float32).copy()),
        cvel=jnp.asarray(np.asarray(mj_data.cvel, dtype=np.float32).copy()),
        subtree_com=jnp.asarray(
            np.asarray(mj_data.subtree_com, dtype=np.float32).copy()
        ),
        geom_xpos=jnp.asarray(np.asarray(mj_data.geom_xpos, dtype=np.float32).copy()),
        geom_xmat=jnp.asarray(
            np.asarray(mj_data.geom_xmat, dtype=np.float32).reshape((-1, 3, 3)).copy()
        ),
        actuator_force=jnp.asarray(
            np.asarray(mj_data.actuator_force, dtype=np.float32).copy()
        ),
        qfrc_constraint=jnp.asarray(
            np.asarray(mj_data.qfrc_constraint, dtype=np.float32).copy()
        ),
        _impl=NativeDataImplementation(
            cacc=jnp.asarray(np.asarray(mj_data.cacc, dtype=np.float32).copy()),
            actuator_velocity=jnp.asarray(
                np.asarray(mj_data.actuator_velocity, dtype=np.float32).copy()
            ),
            efc__force=jnp.asarray(
                np.asarray(mj_data.efc_force, dtype=np.float32).copy()
            ),
        ),
    )
    return BelloState(
        data=data,
        last_action=jnp.asarray(last_action, dtype=jnp.float32),
        foot_air_time=foot_air_time,
        foot_contact_time=foot_contact_time,
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


def mujoco_eval_info(
    env: BelloLocomotionEnv,
    mj_data: mujoco.MjData,
    *,
    command: np.ndarray | None = None,
) -> dict[str, jax.Array]:
    info = {
        "base_height": jnp.asarray(
            mujoco_link_clearance(env, mj_data, env.config.root_link),
            dtype=jnp.float32,
        )
    }
    if command is not None:
        info["command"] = jnp.asarray(command, dtype=jnp.float32)
    return info


def mujoco_contact_terms(
    env: BelloLocomotionEnv, mj_data: mujoco.MjData
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    return (
        jnp.asarray(mujoco_foot_contact(env, mj_data), dtype=jnp.float32),
        jnp.asarray(mujoco_foot_linear_velocity(env, mj_data), dtype=jnp.float32),
        jnp.asarray(mujoco_foot_angular_speed(env, mj_data), dtype=jnp.float32),
        jnp.asarray(mujoco_undesired_contact(env, mj_data), dtype=jnp.float32),
        jnp.asarray(mujoco_self_collision(env, mj_data), dtype=jnp.float32),
    )


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
    foot_position = mujoco_foot_sole_positions(env, mj_data)
    if env.terrain_config.mode == "none":
        terrain_height = np.full(
            foot_position.shape[0], env.terrain_config.pos[2], dtype=np.float32
        )
    else:
        terrain_height = terrain_surface_height(
            env.model.mj_model,
            int(env.model.terrain_geom_ids[0]),
            foot_position[:, :2],
        )
    return foot_position[:, 2] - terrain_height


def mujoco_foot_sole_positions(
    env: BelloLocomotionEnv, mj_data: mujoco.MjData
) -> np.ndarray:
    """Mirror the MJX terrain-relative collision support calculation."""

    geom_ids = np.asarray(env.model.foot_geom_ids, dtype=np.int32)
    geom_rotation = np.asarray(mj_data.geom_xmat[geom_ids], dtype=np.float32).reshape(
        (-1, 3, 3)
    )
    geom_position = np.asarray(mj_data.geom_xpos[geom_ids], dtype=np.float32)
    terrain_up = mujoco_terrain_surface_normal(env, geom_position[:, :2])
    local_down = np.einsum("gji,gj->gi", geom_rotation, -terrain_up)
    geom_size = np.asarray(env.model.mj_model.geom_size[geom_ids], dtype=np.float32)
    geom_type = np.asarray(env.model.mj_model.geom_type[geom_ids], dtype=np.int32)

    radial_direction = local_down[:, :2]
    radial_norm = np.linalg.norm(radial_direction, axis=1, keepdims=True)
    unit_radial = np.divide(
        radial_direction,
        np.maximum(radial_norm, 1.0e-8),
        out=np.zeros_like(radial_direction),
        where=radial_norm > 1.0e-8,
    )
    sphere_support = geom_size[:, :1] * local_down
    capsule_support = sphere_support.copy()
    capsule_support[:, 2] += np.sign(local_down[:, 2]) * geom_size[:, 1]
    ellipsoid_denominator = np.sqrt(
        np.sum((geom_size * local_down) ** 2, axis=1, keepdims=True)
    )
    ellipsoid_support = (
        geom_size**2
        * local_down
        / np.maximum(ellipsoid_denominator, np.float32(1.0e-8))
    )
    cylinder_support = np.concatenate(
        [
            geom_size[:, :1] * unit_radial,
            (np.sign(local_down[:, 2]) * geom_size[:, 1])[:, None],
        ],
        axis=1,
    )
    support = sphere_support
    for shape, candidate in (
        (mujoco.mjtGeom.mjGEOM_CAPSULE, capsule_support),
        (mujoco.mjtGeom.mjGEOM_ELLIPSOID, ellipsoid_support),
        (mujoco.mjtGeom.mjGEOM_CYLINDER, cylinder_support),
    ):
        support = np.where((geom_type == int(shape))[:, None], candidate, support)
    support_world = geom_position + np.einsum("gij,gj->gi", geom_rotation, support)

    box_corner_signs = np.asarray(
        [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
        dtype=np.float32,
    )
    box_corners = box_corner_signs[None, :, :] * geom_size[:, None, :]
    box_corners_world = geom_position[:, None, :] + np.einsum(
        "gij,gkj->gki", geom_rotation, box_corners
    )
    if env.terrain_config.mode == "none":
        box_corner_terrain_height = np.full(
            box_corners_world.shape[:2],
            env.terrain_config.pos[2],
            dtype=np.float32,
        )
    else:
        box_corner_terrain_height = terrain_surface_height(
            env.model.mj_model,
            int(env.model.terrain_geom_ids[0]),
            box_corners_world[:, :, :2],
        )
    box_corner_clearance = box_corners_world[:, :, 2] - box_corner_terrain_height
    lowest_box_corner = np.argmin(box_corner_clearance, axis=1)
    box_support_world = box_corners_world[
        np.arange(box_corners_world.shape[0]), lowest_box_corner
    ]
    support_world = np.where(
        (geom_type == int(mujoco.mjtGeom.mjGEOM_BOX))[:, None],
        box_support_world,
        support_world,
    )
    if env.terrain_config.mode == "none":
        terrain_height = np.full(
            support_world.shape[0], env.terrain_config.pos[2], dtype=np.float32
        )
    else:
        terrain_height = terrain_surface_height(
            env.model.mj_model,
            int(env.model.terrain_geom_ids[0]),
            support_world[:, :2],
        )
    support_clearance = support_world[:, 2] - terrain_height

    geom_body_ids = np.asarray(env.model.geom_body_ids[geom_ids], dtype=np.int32)
    foot_body_ids = np.asarray(env.model.foot_body_ids, dtype=np.int32)
    belongs_to_foot = foot_body_ids[:, None] == geom_body_ids[None, :]
    clearance_by_foot = np.where(belongs_to_foot, support_clearance[None, :], np.inf)
    lowest_geom = np.argmin(clearance_by_foot, axis=1)
    return support_world[lowest_geom]


def mujoco_terrain_surface_normal(
    env: BelloLocomotionEnv, world_xy: np.ndarray
) -> np.ndarray:
    world_xy = np.asarray(world_xy, dtype=np.float32)
    output_shape = world_xy.shape[:-1] + (3,)
    if env.terrain_config.mode == "none":
        return np.broadcast_to(
            np.asarray([0.0, 0.0, 1.0], dtype=np.float32), output_shape
        )
    if env.terrain_config.mode == "plane":
        terrain_geom_id = int(env.model.terrain_geom_ids[0])
        terrain_up = quat_rotate_np(
            np.asarray(env.model.mj_model.geom_quat[terrain_geom_id], dtype=np.float32),
            np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        )
        return np.broadcast_to(terrain_up, output_shape)

    terrain_geom_id = int(env.model.terrain_geom_ids[0])
    epsilon = np.float32(1.0e-3)
    x_offset = np.asarray([epsilon, 0.0], dtype=np.float32)
    y_offset = np.asarray([0.0, epsilon], dtype=np.float32)
    gradient_x = (
        terrain_surface_height(env.model.mj_model, terrain_geom_id, world_xy + x_offset)
        - terrain_surface_height(
            env.model.mj_model, terrain_geom_id, world_xy - x_offset
        )
    ) / (2.0 * epsilon)
    gradient_y = (
        terrain_surface_height(env.model.mj_model, terrain_geom_id, world_xy + y_offset)
        - terrain_surface_height(
            env.model.mj_model, terrain_geom_id, world_xy - y_offset
        )
    ) / (2.0 * epsilon)
    normal = np.stack([-gradient_x, -gradient_y, np.ones_like(gradient_x)], axis=-1)
    return normal / np.maximum(
        np.linalg.norm(normal, axis=-1, keepdims=True), np.float32(1.0e-8)
    )


def mujoco_foot_linear_velocity(
    env: BelloLocomotionEnv, mj_data: mujoco.MjData
) -> np.ndarray:
    if env.model.foot_body_ids.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    foot_world_vel = []
    for body_id in np.asarray(env.model.foot_body_ids, dtype=np.int32):
        velocity = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            env.model.mj_model,
            mj_data,
            mujoco.mjtObj.mjOBJ_XBODY,
            int(body_id),
            velocity,
            0,
        )
        foot_world_vel.append(velocity[3:6])
    return np.asarray(foot_world_vel, dtype=np.float32)


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


def mujoco_self_collision(
    env: BelloLocomotionEnv, mj_data: mujoco.MjData
) -> np.float32:
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
