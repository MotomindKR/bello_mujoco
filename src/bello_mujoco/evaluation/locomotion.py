"""Native locomotion checkpoint evaluation and Rerun visualization."""

from __future__ import annotations

from contextlib import ExitStack
import math
import time
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from rerun.urdf import UrdfTree

from bello_mujoco.envs.locomotion import BelloLocomotionEnv, advance_gait_phase
from bello_mujoco.gamepad import GamepadCommandSource, GamepadConfig
from bello_mujoco.locomotion_runtime import load_locomotion_policy_runtime
from bello_mujoco.policy import (
    infer_policy_action,
    init_observation_stack,
    stacked_observation,
    update_observation_stack,
)
from bello_mujoco.evaluation.render import (
    mujoco_action_to_ctrl,
    mujoco_contact_terms,
    mujoco_done_flags,
    mujoco_eval_info,
    mujoco_eval_state,
    mujoco_link_ang_vel,
    mujoco_link_lin_vel,
    mujoco_link_quat,
    mujoco_policy_observation,
    mujoco_requested_actuator_force,
    mujoco_foot_linear_velocity,
    reset_mujoco_data,
)
from bello_mujoco.spatial import quat_rotate_inverse_wxyz, quat_rotate_wxyz
from bello_mujoco.training.ppo import (
    ObservationNormState,
    PPOConfig,
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
class EpisodeDisplayState:
    episode_return: jax.Array
    episode_length: jax.Array


def ctrl_saturation_fraction(env: BelloLocomotionEnv, ctrl: np.ndarray) -> float:
    indices = env.model.policy_ctrl_indices
    if indices.size == 0:
        return 0.0
    ranges = np.asarray(env.model.mj_model.actuator_ctrlrange[indices])
    values = np.asarray(ctrl)[indices]
    return float(np.mean((values <= ranges[:, 0]) | (values >= ranges[:, 1])))


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
            self.env._reward_link("track_ang_vel_z"),
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
            center = np.asarray(config.pos, dtype=np.float32) + quat_rotate_wxyz(
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
    vertices = quat_rotate_wxyz(
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
    link_name = env._reward_link("track_lin_vel_xy")
    frame_link_name = env._reward_frame_link("track_lin_vel_xy")
    body_velocity = mujoco_link_lin_vel(env, mj_data, link_name)
    world_velocity = quat_rotate_wxyz(
        mujoco_link_quat(env, mj_data, link_name), body_velocity
    )
    frame_quat = mujoco_link_quat(env, mj_data, frame_link_name)
    w, x, y, z = frame_quat
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    yaw_quat = np.asarray(
        [np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=np.float32
    )
    return quat_rotate_inverse_wxyz(yaw_quat, world_velocity)


def run_rerun_evaluation(
    env: BelloLocomotionEnv,
    policy,
    *,
    normalization: ObservationNormState,
    ppo_config: PPOConfig,
    urdf_path: Path,
    command: tuple[float, float, float],
    gamepad_config: GamepadConfig,
    steps: int | None,
    stop_on_done: bool,
    obs_stack_size: int,
    stochastic_actions: bool,
    action_seed: int,
) -> None:
    mj_model = env.model.mj_model
    mj_data = mujoco.MjData(mj_model)
    command_ranges = (
        env.command_config.lin_vel_x,
        env.command_config.lin_vel_y,
        env.command_config.ang_vel_yaw,
    )
    infer_action = eqx.filter_jit(
        lambda observation, rng: infer_policy_action(
            policy,
            observation,
            normalization,
            ppo_config,
            env.action_clip,
            rng=rng,
        )
    )
    action_key = jax.random.PRNGKey(action_seed)
    global_step = 0
    episode = 0
    fixed_command = np.asarray(command, dtype=np.float32)
    current_command = np.zeros(3, dtype=np.float32)

    last_action = np.zeros(env.action_size, dtype=np.float32)
    last_action_delta = np.zeros(env.action_size, dtype=np.float32)
    foot_contact_debounce_time = jnp.zeros(
        env.model.foot_body_ids.size, dtype=jnp.float32
    )
    last_foot_contact = jnp.zeros(env.model.foot_body_ids.size, dtype=bool)
    episode_return = jnp.asarray(0.0, dtype=jnp.float32)
    episode_track_lin_vel_xy = jnp.asarray(0.0, dtype=jnp.float32)
    episode_length = jnp.asarray(0, dtype=jnp.int32)
    episode_step_count = 0
    gait_phase = jnp.asarray(0.0, dtype=jnp.float32)
    obs_stack = init_observation_stack(
        jnp.zeros(env.observation_size, dtype=jnp.float32), obs_stack_size
    )

    def reset_episode() -> None:
        nonlocal last_action, last_action_delta, foot_contact_debounce_time
        nonlocal last_foot_contact, episode_return, episode_track_lin_vel_xy
        nonlocal episode_length, episode_step_count, gait_phase, obs_stack
        reset_mujoco_data(env, mj_data)
        last_action = np.zeros(env.action_size, dtype=np.float32)
        last_action_delta = np.zeros(env.action_size, dtype=np.float32)
        foot_contact_debounce_time = jnp.zeros(
            env.model.foot_body_ids.size, dtype=jnp.float32
        )
        last_foot_contact = jnp.zeros(env.model.foot_body_ids.size, dtype=bool)
        episode_return = jnp.asarray(0.0, dtype=jnp.float32)
        episode_track_lin_vel_xy = jnp.asarray(0.0, dtype=jnp.float32)
        episode_length = jnp.asarray(0, dtype=jnp.int32)
        episode_step_count = 0
        gait_phase = jnp.asarray(0.0, dtype=jnp.float32)
        obs = mujoco_policy_observation(
            env,
            mj_data,
            command=current_command,
            last_action=last_action,
            step_count=episode_step_count,
            gait_phase=gait_phase,
        )
        obs_stack = init_observation_stack(obs, obs_stack_size)

    resources = ExitStack()
    try:
        minimum_moving_speeds = (
            env.command_config.minimum_moving_linear_speed,
            env.command_config.minimum_moving_linear_speed,
            env.command_config.minimum_moving_yaw_speed,
        )
        gamepad = GamepadCommandSource(
            gamepad_config,
            command_ranges,
            minimum_moving_speeds,
        )
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

            current_observation = mujoco_policy_observation(
                env,
                mj_data,
                command=current_command,
                last_action=last_action,
                step_count=episode_step_count,
                gait_phase=gait_phase,
            )
            obs_stack = obs_stack.at[-1].set(current_observation)
            action_rng = None
            if stochastic_actions:
                action_key, action_rng = jax.random.split(action_key)
            policy_step = infer_action(stacked_observation(obs_stack), action_rng)
            requested_action = policy_step.requested_action
            action_mean_np = np.asarray(
                jax.device_get(policy_step.action_mean), dtype=np.float32
            )
            requested_action_np = np.asarray(
                jax.device_get(requested_action), dtype=np.float32
            )
            action_np = np.asarray(
                jax.device_get(policy_step.applied_action), dtype=np.float32
            )
            action_delta = action_np - last_action
            ctrl = mujoco_action_to_ctrl(env, action_np)
            state = mujoco_eval_state(
                env,
                mj_data,
                command=current_command,
                last_action=last_action,
                last_action_delta=last_action_delta,
                foot_contact_debounce_time=foot_contact_debounce_time,
                last_foot_contact=last_foot_contact,
                step_count=episode_step_count,
                episode_return=episode_return,
                episode_track_lin_vel_xy=episode_track_lin_vel_xy,
                episode_length=episode_length,
                gait_phase=gait_phase,
            )
            mj_data.ctrl[:] = ctrl
            touchdown_eligible = (
                (episode_step_count > 0)
                & ~np.asarray(last_foot_contact, dtype=bool)
                & (np.asarray(foot_contact_debounce_time) <= 0.0)
            )
            foot_touchdown_np = np.zeros(env.model.foot_body_ids.size, dtype=bool)
            foot_normal_impulse_np = np.zeros(
                env.model.foot_body_ids.size, dtype=np.float32
            )
            foot_peak_normal_force_np = np.zeros_like(foot_normal_impulse_np)
            touchdown_velocity_np = np.zeros(
                (env.model.foot_body_ids.size, 3), dtype=np.float32
            )
            requested_actuator_force_np = np.zeros(
                env.model.mj_model.nu, dtype=np.float32
            )
            for _ in range(env.decimation):
                preimpact_velocity = mujoco_foot_linear_velocity(env, mj_data)
                mujoco.mj_step(mj_model, mj_data)
                substep_contact, substep_force = mujoco_contact_terms(env, mj_data)[:2]
                substep_contact_np = np.asarray(substep_contact, dtype=bool)
                substep_force_np = np.asarray(substep_force, dtype=np.float32)
                new_touchdown = (
                    touchdown_eligible & ~foot_touchdown_np & substep_contact_np
                )
                touchdown_velocity_np[new_touchdown] = preimpact_velocity[new_touchdown]
                foot_touchdown_np |= new_touchdown
                foot_normal_impulse_np += substep_force_np * float(
                    mj_model.opt.timestep
                )
                foot_peak_normal_force_np = np.maximum(
                    foot_peak_normal_force_np, substep_force_np
                )
                requested_actuator_force_np = np.maximum(
                    requested_actuator_force_np,
                    np.abs(mujoco_requested_actuator_force(env, mj_data)),
                )
            last_action = action_np
            last_action_delta = action_delta
            episode_step_count += 1
            gait_phase = advance_gait_phase(
                gait_phase,
                jnp.asarray(current_command, dtype=jnp.float32),
                env.command_config,
                env.config.gait,
                env.control_timestep,
            )
            done_flags = mujoco_done_flags(env, mj_data, episode_step_count)
            done = jnp.asarray(done_flags["done/any"])
            next_state = mujoco_eval_state(
                env,
                mj_data,
                command=current_command,
                last_action=last_action,
                last_action_delta=last_action_delta,
                foot_contact_debounce_time=foot_contact_debounce_time,
                last_foot_contact=last_foot_contact,
                step_count=episode_step_count,
                episode_return=episode_return,
                episode_track_lin_vel_xy=episode_track_lin_vel_xy,
                episode_length=episode_length,
                gait_phase=gait_phase,
            )
            (
                raw_foot_contact,
                foot_normal_force,
                foot_linear_velocity,
                angular_speed,
                undesired,
                self_collision,
            ) = mujoco_contact_terms(env, mj_data)
            foot_touchdown = jnp.asarray(foot_touchdown_np)
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
                foot_touchdown,
                jnp.asarray(foot_normal_impulse_np),
                jnp.asarray(foot_peak_normal_force_np),
                jnp.asarray(touchdown_velocity_np),
                jnp.asarray(requested_actuator_force_np),
                foot_linear_velocity,
                angular_speed,
                undesired,
                self_collision,
            )
            done_info = {key: jnp.asarray(value) for key, value in done_flags.items()}
            reward, reward_info = env.sanitize_nonfinite_reward(
                reward, reward_info, done_info
            )
            episode_return = episode_return + reward
            episode_track_lin_vel_xy = (
                episode_track_lin_vel_xy + reward_info["reward/track_lin_vel_xy"]
            )
            episode_length = episode_length + 1
            last_foot_contact = foot_contact.astype(bool)
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
                obs = mujoco_policy_observation(
                    env,
                    mj_data,
                    command=current_command,
                    last_action=last_action,
                    step_count=episode_step_count,
                    gait_phase=gait_phase,
                )
                obs_stack = update_observation_stack(obs_stack, obs, done)
            elapsed = time.perf_counter() - started
            time.sleep(max(0.0, env.control_timestep - elapsed))
    except KeyboardInterrupt:
        logger.log_event("evaluation interrupted")
    finally:
        resources.close()


@dataclass(frozen=True)
class EvalPolicyConfig:
    command: tuple[float, float, float] = (0.0, 0.0, 0.0)
    steps: int | None = None
    stop_on_done: bool = False
    stochastic_actions: bool = False
    action_seed: int = 0
    gamepad: GamepadConfig = GamepadConfig()
    allow_git_commit_mismatch: bool = False


def evaluate_locomotion_checkpoint(
    checkpoint_path: Path,
    eval_config: EvalPolicyConfig = EvalPolicyConfig(),
) -> None:
    validate_eval_config(eval_config)
    runtime = load_locomotion_policy_runtime(
        checkpoint_path,
        use_viewer_model=True,
        allow_git_commit_mismatch=eval_config.allow_git_commit_mismatch,
    )
    run_rerun_evaluation(
        runtime.env,
        runtime.policy,
        normalization=runtime.normalization.observation,
        ppo_config=runtime.metadata.ppo,
        urdf_path=runtime.urdf_path,
        command=eval_config.command,
        gamepad_config=eval_config.gamepad,
        steps=eval_config.steps,
        stop_on_done=eval_config.stop_on_done,
        obs_stack_size=runtime.metadata.ppo.policy.obs_stack_size,
        stochastic_actions=eval_config.stochastic_actions,
        action_seed=eval_config.action_seed,
    )


def validate_eval_config(config: EvalPolicyConfig) -> None:
    if config.steps is not None and config.steps <= 0:
        raise ValueError(f"eval_config.steps must be positive, got {config.steps}")
    if len(config.command) != 3:
        raise ValueError(
            f"eval_config.command must have three values, got {config.command}"
        )
    if not all(math.isfinite(value) for value in config.command):
        raise ValueError(
            f"eval_config.command must contain finite values, got {config.command}"
        )
    if config.action_seed < 0:
        raise ValueError(
            f"eval_config.action_seed must be nonnegative, got {config.action_seed}"
        )
    if config.gamepad.controller_index < 0:
        raise ValueError(
            "eval_config.gamepad.controller_index must be nonnegative, got "
            f"{config.gamepad.controller_index}"
        )
    if not 0.0 <= config.gamepad.deadzone < 1.0:
        raise ValueError(
            "eval_config.gamepad.deadzone must be in [0, 1), got "
            f"{config.gamepad.deadzone}"
        )
