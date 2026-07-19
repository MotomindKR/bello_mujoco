from __future__ import annotations

from contextlib import ExitStack
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from rerun.urdf import UrdfTree

from bello_mujoco.envs.locomotion import BelloLocomotionEnv
from bello_mujoco.policy import (
    apply_actor,
    init_observation_stack,
    sample_actor_action,
    stacked_observation,
    update_observation_stack,
)
from bello_mujoco.render import (
    mujoco_action_to_ctrl,
    mujoco_contact_terms,
    mujoco_done_flags,
    mujoco_eval_info,
    mujoco_eval_state,
    mujoco_link_ang_vel,
    mujoco_link_lin_vel,
    mujoco_link_quat,
    mujoco_policy_observation,
    quat_rotate_inverse_np,
    quat_rotate_np,
    reset_mujoco_data,
    sync_mj_data,
)

DONE_KEYS = (
    "done/any",
    "done/terminal",
    "done/base_height",
    "done/bad_orientation",
    "done/timeout",
    "done/nonfinite",
    "done/unstable",
)


@dataclass(frozen=True)
class GamepadConfig:
    enabled: bool = True
    controller_index: int = 0
    deadzone: float = 0.1


@dataclass(frozen=True)
class GamepadInput:
    command: np.ndarray
    connected: bool
    reset_requested: bool = False
    exit_requested: bool = False
    connection_changed: bool = False
    controller_name: str | None = None


@dataclass(frozen=True)
class EpisodeDisplayState:
    episode_return: jax.Array
    episode_length: jax.Array


def apply_deadzone(value: float, deadzone: float) -> float:
    value = float(np.clip(value, -1.0, 1.0))
    magnitude = abs(value)
    if magnitude <= deadzone:
        return 0.0
    return math.copysign((magnitude - deadzone) / (1.0 - deadzone), value)


def axis_to_range(value: float, bounds: tuple[float, float]) -> float:
    low, high = bounds
    return value * (high if value >= 0.0 else -low)


def ctrl_saturation_fraction(env: BelloLocomotionEnv, ctrl: np.ndarray) -> float:
    indices = env.model.policy_ctrl_indices
    if indices.size == 0:
        return 0.0
    ranges = np.asarray(env.model.mj_model.actuator_ctrlrange[indices])
    values = np.asarray(ctrl)[indices]
    return float(np.mean((values <= ranges[:, 0]) | (values >= ranges[:, 1])))


class GamepadCommandSource:
    def __init__(
        self,
        config: GamepadConfig,
        command_ranges: tuple[
            tuple[float, float], tuple[float, float], tuple[float, float]
        ],
        *,
        controller_module=None,
        event_module=None,
        constants_module=None,
        pygame_module=None,
    ) -> None:
        display_initialized = False
        if controller_module is None or event_module is None:
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            import pygame
            from pygame._sdl2 import controller

            if config.enabled:
                try:
                    pygame.display.init()
                    display_initialized = True
                except BaseException:
                    pygame.display.quit()
                    raise
            controller_module = controller
            event_module = pygame.event
            constants_module = pygame
            pygame_module = pygame
        elif constants_module is None:
            constants_module = controller_module
        self.config = config
        self.command_ranges = command_ranges
        self.controller_module = controller_module
        self.event_module = event_module
        self.constants = constants_module
        self.pygame = pygame_module
        self.controller = None
        self._was_connected = False
        self._reset_pressed = False
        self._exit_pressed = False
        self._closed = False
        if config.enabled:
            try:
                self.controller_module.init()
            except BaseException:
                try:
                    self.controller_module.quit()
                except BaseException:
                    pass
                if display_initialized:
                    try:
                        self.pygame.display.quit()
                    except BaseException:
                        pass
                raise

    def poll(self) -> GamepadInput:
        if not self.config.enabled:
            return GamepadInput(np.zeros(3, dtype=np.float32), connected=False)
        try:
            self.event_module.pump()
            self._refresh_controller()
            connected = self.controller is not None and self.controller.attached()
        except Exception:
            self._close_controller()
            connected = False
        connection_changed = connected != self._was_connected
        self._was_connected = connected
        if not connected:
            self._reset_pressed = False
            self._exit_pressed = False
            return GamepadInput(
                np.zeros(3, dtype=np.float32),
                connected=False,
                connection_changed=connection_changed,
            )

        try:
            module = self.constants
            left_x = self._axis(module.CONTROLLER_AXIS_LEFTX)
            left_y = self._axis(module.CONTROLLER_AXIS_LEFTY)
            right_x = self._axis(module.CONTROLLER_AXIS_RIGHTX)
            normalized = (
                apply_deadzone(-left_y, self.config.deadzone),
                apply_deadzone(left_x, self.config.deadzone),
                apply_deadzone(-right_x, self.config.deadzone),
            )
            command = np.asarray(
                [
                    axis_to_range(value, bounds)
                    for value, bounds in zip(
                        normalized, self.command_ranges, strict=True
                    )
                ],
                dtype=np.float32,
            )
            reset_pressed = bool(self.controller.get_button(module.CONTROLLER_BUTTON_A))
            exit_pressed = bool(
                self.controller.get_button(module.CONTROLLER_BUTTON_START)
            )
            controller_name = self.controller.as_joystick().get_name()
        except Exception:
            self._close_controller()
            self._was_connected = False
            self._reset_pressed = False
            self._exit_pressed = False
            return GamepadInput(
                np.zeros(3, dtype=np.float32),
                connected=False,
                connection_changed=True,
            )
        result = GamepadInput(
            command,
            connected=True,
            reset_requested=reset_pressed and not self._reset_pressed,
            exit_requested=exit_pressed and not self._exit_pressed,
            connection_changed=connection_changed,
            controller_name=controller_name,
        )
        self._reset_pressed = reset_pressed
        self._exit_pressed = exit_pressed
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._close_controller()
            if self.config.enabled:
                self.controller_module.quit()
        finally:
            if self.pygame is not None and self.config.enabled:
                self.pygame.display.quit()

    def _axis(self, axis: int) -> float:
        raw = int(self.controller.get_axis(axis))
        divisor = 32767.0 if raw >= 0 else 32768.0
        return float(np.clip(raw / divisor, -1.0, 1.0))

    def _refresh_controller(self) -> None:
        if self.controller is not None:
            try:
                if self.controller.attached():
                    return
            except Exception:
                pass
        self._close_controller()
        index = self.config.controller_index
        if (
            index < self.controller_module.get_count()
            and self.controller_module.is_controller(index)
        ):
            try:
                self.controller = self.controller_module.Controller(index)
            except Exception:
                self.controller = None

    def _close_controller(self) -> None:
        if self.controller is None:
            return
        try:
            self.controller.quit()
        except Exception:
            pass
        self.controller = None


class RerunEvaluationLogger:
    def __init__(
        self,
        env: BelloLocomotionEnv,
        urdf_path: Path,
        *,
        recording: rr.RecordingStream | None = None,
    ) -> None:
        self.env = env
        self.urdf = UrdfTree.from_file_path(
            urdf_path,
            entity_path_prefix="world",
            frame_prefix="bello/",
            static_transform_entity_path="/world/tf_static",
        )
        self.dynamic_joints = self._dynamic_joint_qpos_addresses()
        root_visual_paths = self.urdf.get_visual_geometry_paths(
            self.urdf.root_link().name
        )
        if not root_visual_paths:
            raise ValueError("Rerun camera tracking requires root-link visual geometry")
        self.root_tracking_path = root_visual_paths[0]
        self.action_names = tuple(
            env.model.actuator_joint_names[index]
            for index in env.model.policy_ctrl_indices
        )
        self.joint_names = env.model.actuator_joint_names
        self.actuator_names = env.model.actuator_names
        self.reward_names: tuple[str, ...] | None = None
        self.recording = recording or rr.RecordingStream("bello_policy_evaluation")
        self._closed = False

    def start(self) -> None:
        blueprint = self._blueprint()
        self.recording.spawn(default_blueprint=blueprint)
        self.recording.send_blueprint(blueprint, make_active=True, make_default=True)
        self.recording.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        self.urdf.log_urdf_to_recording(self.recording)
        self._log_terrain()
        self._style_series(
            "metrics/command",
            (
                "command/lin_vel_x",
                "actual/lin_vel_x",
                "command/lin_vel_y",
                "actual/lin_vel_y",
                "command/yaw_vel",
                "actual/yaw_vel",
            ),
        )
        self._style_series("metrics/action/mean", self.action_names)
        self._style_series("metrics/action/applied", self.action_names)
        self._style_series("metrics/joint/position", self.joint_names)
        self._style_series("metrics/joint/velocity", self.joint_names)
        self._style_series("metrics/actuator/force", self.actuator_names)
        for actuator_name in self.action_names:
            self._style_series(
                f"metrics/policy_actuators/{actuator_name}",
                ("joint_position", "joint_target", "action", "actuator_force"),
            )
        self._style_series(
            "metrics/episode", ("index", "return", "length", "base_height")
        )
        self._style_series(
            "metrics/done",
            (
                "any",
                "terminal",
                "base_height",
                "bad_orientation",
                "timeout",
                "nonfinite",
                "unstable",
            ),
        )
        self._style_series("metrics/control/saturation", ("fraction",))
        self._style_series("metrics/gamepad/connected", ("connected",))

    def log_step(
        self,
        *,
        global_step: int,
        episode: int,
        mj_data: mujoco.MjData,
        command: np.ndarray,
        connected: bool,
        action_mean: np.ndarray,
        action: np.ndarray,
        ctrl: np.ndarray,
        reward: jax.Array,
        done_flags: dict[str, bool],
        state,
        info: dict[str, jax.Array],
    ) -> None:
        self.recording.set_time("step", sequence=global_step)
        self.recording.set_time(
            "simulation_time", duration=global_step * self.env.control_timestep
        )
        self._log_robot_transforms(mj_data)
        actual_lin_vel = command_frame_linear_velocity(self.env, mj_data)
        actual_yaw_vel = mujoco_link_ang_vel(
            self.env,
            mj_data,
            self.env._reward_link(self.env.reward_config.track_ang_vel_z),
        )[2]
        self.recording.log(
            "metrics/command",
            rr.Scalars(
                [
                    command[0],
                    actual_lin_vel[0],
                    command[1],
                    actual_lin_vel[1],
                    command[2],
                    actual_yaw_vel,
                ]
            ),
        )
        self.recording.log("metrics/action/mean", rr.Scalars(action_mean))
        self.recording.log("metrics/action/applied", rr.Scalars(action))
        self.recording.log(
            "metrics/joint/position",
            rr.Scalars(mj_data.qpos[self.env.model.actuator_qpos_indices]),
        )
        self.recording.log(
            "metrics/joint/velocity",
            rr.Scalars(mj_data.qvel[self.env.model.actuator_qvel_indices]),
        )
        self.recording.log(
            "metrics/actuator/force", rr.Scalars(np.asarray(mj_data.actuator_force))
        )
        joint_target = np.asarray(self.env.action_to_joint_target(action))
        for action_index, actuator_name in enumerate(self.action_names):
            actuator_index = self.env.model.policy_ctrl_indices[action_index]
            qpos_index = self.env.model.actuator_qpos_indices[actuator_index]
            self.recording.log(
                f"metrics/policy_actuators/{actuator_name}",
                rr.Scalars(
                    [
                        mj_data.qpos[qpos_index],
                        joint_target[actuator_index],
                        action[action_index],
                        mj_data.actuator_force[actuator_index],
                    ]
                ),
            )
        self.recording.log(
            "metrics/control/saturation",
            rr.Scalars(ctrl_saturation_fraction(self.env, ctrl)),
        )
        self.recording.log("metrics/reward/total", rr.Scalars(float(reward)))
        reward_items = sorted(
            (key.removeprefix("reward/"), float(value))
            for key, value in info.items()
            if key.startswith("reward/")
        )
        if reward_items:
            names = tuple(name for name, _ in reward_items)
            if self.reward_names is None:
                self.reward_names = names
                self._style_series("metrics/reward/components", names)
            self.recording.log(
                "metrics/reward/components",
                rr.Scalars([value for _, value in reward_items]),
            )
        self.recording.log(
            "metrics/episode",
            rr.Scalars(
                [
                    episode,
                    float(state.episode_return),
                    int(state.episode_length),
                    float(info.get("base_height", 0.0)),
                ]
            ),
        )
        self.recording.log(
            "metrics/done",
            rr.Scalars([float(done_flags[key]) for key in DONE_KEYS]),
        )
        self.recording.log("metrics/gamepad/connected", rr.Scalars(float(connected)))

    def log_event(self, message: str, *, level: str = "INFO") -> None:
        self.recording.log("events", rr.TextLog(message, level=level))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.recording.flush()
        finally:
            self.recording.disconnect()

    def _blueprint(self) -> rrb.Blueprint:
        link_names = {self.urdf.root_link().name}
        for joint in self.urdf.joints():
            link_names.add(joint.parent_link)
            link_names.add(joint.child_link)
        collision_paths = [
            path
            for link_name in sorted(link_names)
            for path in self.urdf.get_collision_geometry_paths(link_name)
        ]
        scene_contents = [
            "+ /world/**",
            *[f"- /{path.lstrip('/')}" for path in collision_paths],
        ]
        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name="Bello",
                    origin="world",
                    contents=scene_contents,
                    spatial_information=rrb.SpatialInformation(target_frame="world"),
                    eye_controls=rrb.EyeControls3D(
                        kind=rrb.Eye3DKind.Orbital,
                        position=(2.5, -2.5, 1.6),
                        look_target=(0.0, 0.0, 0.8),
                        eye_up=(0.0, 0.0, 1.0),
                        tracking_entity=self.root_tracking_path,
                    ),
                ),
                rrb.Tabs(
                    rrb.TimeSeriesView(name="Reward", origin="metrics/reward"),
                    rrb.TimeSeriesView(name="Command", origin="metrics/command"),
                    rrb.Tabs(
                        *[
                            rrb.TimeSeriesView(
                                name=joint_name,
                                origin=f"metrics/policy_actuators/{joint_name}",
                            )
                            for joint_name in self.action_names
                        ],
                        name="Policy Actuators",
                    ),
                    rrb.TimeSeriesView(name="Status", origin="metrics"),
                ),
                column_shares=[3, 2],
            ),
            collapse_panels=True,
        )

    def _dynamic_joint_qpos_addresses(self) -> tuple[tuple[object, int], ...]:
        result = []
        model = self.env.model.mj_model
        for joint in self.urdf.joints():
            if joint.joint_type not in {"revolute", "continuous", "prismatic"}:
                continue
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint.name)
            if joint_id < 0:
                raise ValueError(f"URDF joint missing from MuJoCo model: {joint.name}")
            result.append((joint, int(model.jnt_qposadr[joint_id])))
        return tuple(result)

    def _log_robot_transforms(self, mj_data: mujoco.MjData) -> None:
        root_qpos = self.env.model.root_qpos_adr
        root_translation = np.asarray(mj_data.qpos[root_qpos : root_qpos + 3])
        root_quat = np.asarray(mj_data.qpos[root_qpos + 3 : root_qpos + 7])
        root_rotation = rr.Quaternion(xyzw=root_quat[[1, 2, 3, 0]])
        self.recording.log(
            "world/robot/transforms",
            rr.Transform3D(
                translation=root_translation,
                quaternion=root_rotation,
                parent_frame="world",
                child_frame=f"bello/{self.urdf.root_link().name}",
            ),
        )
        for joint, qpos_address in self.dynamic_joints:
            self.recording.log(
                "world/robot/transforms",
                joint.compute_transform(mj_data.qpos[qpos_address], clamp=False),
            )

    def _log_terrain(self) -> None:
        config = self.env.terrain_config
        color = np.asarray(config.rgba) * 255.0
        if config.mode == "none":
            return
        if config.mode == "plane":
            terrain_geom_id = int(self.env.model.terrain_geom_ids[0])
            quaternion = np.asarray(
                self.env.model.mj_model.geom_quat[terrain_geom_id], dtype=np.float32
            )
            center = np.asarray(config.pos, dtype=np.float32) + quat_rotate_np(
                quaternion,
                np.asarray([0.0, 0.0, -config.size[2]], dtype=np.float32),
            )
            self.recording.log(
                "world/terrain",
                rr.Boxes3D(
                    centers=[center],
                    half_sizes=[config.size],
                    quaternions=[rr.Quaternion(xyzw=quaternion[[1, 2, 3, 0]])],
                    colors=[color.astype(np.uint8)],
                ),
                static=True,
            )
            return
        vertices, triangles = rough_terrain_mesh(config)
        self.recording.log(
            "world/terrain",
            rr.Mesh3D(
                vertex_positions=vertices,
                triangle_indices=triangles,
                albedo_factor=color.astype(np.uint8),
            ),
            static=True,
        )

    def _style_series(self, path: str, names: tuple[str, ...]) -> None:
        self.recording.log(path, rr.SeriesLines(names=list(names)), static=True)


def rough_terrain_mesh(config) -> tuple[np.ndarray, np.ndarray]:
    from bello_mujoco.terrain import random_slope_quat, rough_height_data

    resolution = config.hfield_resolution
    normalized = np.fromstring(rough_height_data(config), sep=" ").reshape(
        resolution, resolution
    )
    normalized -= normalized.min()
    height_range = normalized.max()
    if height_range > 0.0:
        normalized /= height_range
    normalized = np.flipud(normalized)
    xs = np.linspace(-config.size[0], config.size[0], resolution)
    ys = np.linspace(-config.size[1], config.size[1], resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    local_vertices = np.column_stack(
        [
            grid_x.ravel(),
            grid_y.ravel(),
            normalized.ravel() * config.hfield_amplitude,
        ]
    ).astype(np.float32)
    vertices = quat_rotate_np(
        np.asarray(random_slope_quat(config), dtype=np.float32), local_vertices
    ) + np.asarray(config.pos, dtype=np.float32)
    triangles = []
    for row in range(resolution - 1):
        for col in range(resolution - 1):
            lower_left = row * resolution + col
            lower_right = lower_left + 1
            upper_left = lower_left + resolution
            upper_right = upper_left + 1
            triangles.extend(
                (
                    (lower_left, lower_right, upper_right),
                    (lower_left, upper_right, upper_left),
                )
            )
    return vertices, np.asarray(triangles, dtype=np.uint32)


def command_frame_linear_velocity(
    env: BelloLocomotionEnv, mj_data: mujoco.MjData
) -> np.ndarray:
    reward = env.reward_config.track_lin_vel_xy
    link_name = env._reward_link(reward)
    frame_link_name = env._reward_frame_link(reward)
    body_velocity = mujoco_link_lin_vel(env, mj_data, link_name)
    world_velocity = quat_rotate_np(
        mujoco_link_quat(env, mj_data, link_name), body_velocity
    )
    frame_quat = mujoco_link_quat(env, mj_data, frame_link_name)
    w, x, y, z = frame_quat
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    yaw_quat = np.asarray(
        [np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=np.float32
    )
    return quat_rotate_inverse_np(yaw_quat, world_velocity)


def run_rerun_evaluation(
    env: BelloLocomotionEnv,
    policy,
    *,
    urdf_path: Path,
    command: tuple[float, float, float],
    gamepad_config: GamepadConfig,
    steps: int | None,
    stop_on_done: bool,
    simulation_backend: Literal["mujoco", "mjx"] = "mujoco",
    obs_stack_size: int,
    stochastic_actions: bool,
    action_seed: int,
) -> None:
    if simulation_backend not in {"mujoco", "mjx"}:
        raise ValueError(
            f"simulation_backend must be 'mujoco' or 'mjx', got {simulation_backend!r}"
        )
    mj_model = env.model.mj_model
    mj_data = mujoco.MjData(mj_model)
    command_ranges = (
        env.command_config.lin_vel_x,
        env.command_config.lin_vel_y,
        env.command_config.ang_vel_yaw,
    )
    mjx_step = jax.jit(env.step) if simulation_backend == "mjx" else None
    action_key = jax.random.PRNGKey(action_seed)
    global_step = 0
    episode = 0
    fixed_command = np.asarray(command, dtype=np.float32)
    current_command = np.zeros(3, dtype=np.float32)
    mjx_state = None

    last_action = np.zeros(env.action_size, dtype=np.float32)
    foot_air_time = jnp.zeros(env.model.foot_body_ids.size, dtype=jnp.float32)
    foot_contact_time = jnp.zeros(env.model.foot_body_ids.size, dtype=jnp.float32)
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

    def reset_episode() -> None:
        nonlocal mjx_state
        nonlocal last_action, foot_air_time, foot_contact_time
        nonlocal foot_contact_debounce_time
        nonlocal last_foot_contact, episode_return, episode_track_lin_vel_xy
        nonlocal episode_length, episode_step_count, obs_stack
        if simulation_backend == "mjx":
            mjx_state, obs = env.reset(jnp.asarray(current_command))
            sync_mj_data(mj_model, mj_data, mjx_state)
            obs_stack = init_observation_stack(obs, obs_stack_size)
            return
        reset_mujoco_data(env, mj_data)
        last_action = np.zeros(env.action_size, dtype=np.float32)
        foot_air_time = jnp.zeros(env.model.foot_body_ids.size, dtype=jnp.float32)
        foot_contact_time = jnp.zeros(env.model.foot_body_ids.size, dtype=jnp.float32)
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
            command=current_command,
            last_action=last_action,
            step_count=episode_step_count,
        )
        obs_stack = init_observation_stack(obs, obs_stack_size)

    resources = ExitStack()
    try:
        gamepad = GamepadCommandSource(gamepad_config, command_ranges)
        resources.callback(gamepad.close)
        logger = RerunEvaluationLogger(env, urdf_path)
        resources.callback(logger.close)
    except BaseException:
        resources.close()
        raise

    try:
        logger.start()
        initial_input = gamepad.poll()
        current_command = (
            initial_input.command if gamepad_config.enabled else fixed_command.copy()
        )
        if not env.command_config.enabled:
            current_command[:] = 0.0
        reset_episode()
        pending_input = initial_input
        while steps is None or global_step < steps:
            started = time.perf_counter()
            gamepad_input = pending_input or gamepad.poll()
            pending_input = None
            if gamepad_config.enabled:
                current_command = gamepad_input.command
            else:
                current_command = fixed_command.copy()
            if not env.command_config.enabled:
                current_command[:] = 0.0
            if gamepad_input.connection_changed:
                state = "connected" if gamepad_input.connected else "disconnected"
                suffix = (
                    f": {gamepad_input.controller_name}"
                    if gamepad_input.controller_name
                    else ""
                )
                logger.log_event(f"gamepad {state}{suffix}")
            if gamepad_input.exit_requested:
                logger.log_event("gamepad exit requested")
                break
            if gamepad_input.reset_requested:
                episode += 1
                reset_episode()
                logger.log_event("episode reset requested by gamepad")

            action_mean = apply_actor(policy, stacked_observation(obs_stack))
            action = action_mean
            if stochastic_actions:
                action_key, action_rng = jax.random.split(action_key)
                action = sample_actor_action(
                    policy, stacked_observation(obs_stack), action_rng
                )
            requested_action = action
            action = jnp.clip(action, -env.action_clip, env.action_clip)
            action_mean_np = np.asarray(jax.device_get(action_mean), dtype=np.float32)
            requested_action_np = np.asarray(
                jax.device_get(requested_action), dtype=np.float32
            )
            action_np = np.asarray(jax.device_get(action), dtype=np.float32)
            if simulation_backend == "mjx":
                mjx_state = mjx_state.replace(command=jnp.asarray(current_command))
                mjx_state, obs, reward, done, info = mjx_step(
                    mjx_state, requested_action
                )
                sync_mj_data(mj_model, mj_data, mjx_state)
                ctrl = np.asarray(mj_data.ctrl, dtype=np.float32).copy()
                done_flags = {key: bool(jax.device_get(info[key])) for key in DONE_KEYS}
                display_state = mjx_state
            else:
                ctrl = mujoco_action_to_ctrl(env, action_np)
                state = mujoco_eval_state(
                    env,
                    mj_data,
                    command=current_command,
                    last_action=last_action,
                    foot_air_time=foot_air_time,
                    foot_contact_time=foot_contact_time,
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
                last_action = action_np
                episode_step_count += 1
                done_flags = mujoco_done_flags(env, mj_data, episode_step_count)
                done = jnp.asarray(done_flags["done/any"])
                next_state = mujoco_eval_state(
                    env,
                    mj_data,
                    command=current_command,
                    last_action=last_action,
                    foot_air_time=foot_air_time,
                    foot_contact_time=foot_contact_time,
                    foot_contact_debounce_time=foot_contact_debounce_time,
                    last_foot_contact=last_foot_contact,
                    step_count=episode_step_count,
                    episode_return=episode_return,
                    episode_track_lin_vel_xy=episode_track_lin_vel_xy,
                    episode_length=episode_length,
                )
                contact_terms = mujoco_contact_terms(env, mj_data)
                (
                    raw_foot_contact,
                    foot_height,
                    lateral_speed,
                    angular_speed,
                    undesired,
                    self_collision,
                ) = contact_terms
                foot_contact, foot_contact_debounce_time = env.debounce_foot_contact(
                    next_state, raw_foot_contact, done
                )
                reward, reward_info = env.reward(
                    state,
                    next_state,
                    jnp.asarray(action_np),
                    jnp.asarray(requested_action_np),
                    jnp.asarray(done_flags["done/terminal"]),
                    foot_contact,
                    foot_height,
                    lateral_speed,
                    angular_speed,
                    undesired,
                    self_collision,
                )
                done_info = {
                    key: jnp.asarray(value) for key, value in done_flags.items()
                }
                reward, reward_info = env.sanitize_nonfinite_reward(
                    reward, reward_info, done_info
                )
                episode_return = episode_return + reward
                episode_track_lin_vel_xy = (
                    episode_track_lin_vel_xy + reward_info["reward/track_lin_vel_xy"]
                )
                episode_length = episode_length + 1
                (
                    foot_air_time,
                    foot_contact_time,
                    last_foot_contact,
                ) = env.next_foot_memory(next_state, foot_contact, done)
                info = {
                    **mujoco_eval_info(env, mj_data, command=current_command),
                    **done_info,
                    **reward_info,
                }
                display_state = EpisodeDisplayState(
                    episode_return=episode_return,
                    episode_length=episode_length,
                )
            global_step += 1
            logger.log_step(
                global_step=global_step,
                episode=episode,
                mj_data=mj_data,
                command=current_command,
                connected=gamepad_input.connected,
                action_mean=action_mean_np,
                action=action_np,
                ctrl=ctrl,
                reward=reward,
                done_flags=done_flags,
                state=display_state,
                info=info,
            )
            if bool(done):
                logger.log_event(
                    "episode done: "
                    + ", ".join(key for key, value in done_flags.items() if value)
                )
                if stop_on_done:
                    break
                episode += 1
                reset_episode()
            else:
                if simulation_backend == "mujoco":
                    obs = mujoco_policy_observation(
                        env,
                        mj_data,
                        command=current_command,
                        last_action=last_action,
                        step_count=episode_step_count,
                    )
                obs_stack = update_observation_stack(obs_stack, obs, done)
            elapsed = time.perf_counter() - started
            time.sleep(max(0.0, env.control_timestep - elapsed))
    except KeyboardInterrupt:
        logger.log_event("evaluation interrupted")
    finally:
        resources.close()
