"""Bello MuJoCo/MJX locomotion environment scaffold."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from typing import Literal

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from bello_mujoco.actuators import ActuatorConfig
from bello_mujoco.models.compiler import ModelConfig, CompiledRobot
from bello_mujoco.models.ratings import motor_continuous_torque
from bello_mujoco.contracts import ActionLayout, ObservationLayout
from bello_mujoco.envs.observations import (
    build_observation,
    advance_feedback_history,
    sample_feedback_delay,
)
from bello_mujoco.envs.geometry import (
    quat_rotate_inverse,
    quat_rotate,
    yaw_quat,
    projected_gravity_tilt_angle,
)
from bello_mujoco.envs.contacts import contact_forces, debounce_contact
from bello_mujoco.envs.randomization import randomize_spawn, apply_push_batch
from bello_mujoco.envs.actions import (
    bounded_motor_target,
    motor_default,
    project_motor_target,
)
from bello_mujoco.envs.rewards import (
    CommandConfig,
    RewardConfig,
    RewardInputs,
    command_is_moving,
    command_is_standing,
    compute_locomotion_reward,
)
from bello_mujoco.envs.foot_geometry import (
    sole_geometry,
    sole_lateral_clearance,
    sole_support_positions,
    heightfield_surface_normal,
)
from bello_mujoco.sim import BelloModel, WarpConfig, load_model, load_compiled_model
from bello_mujoco.envs.feedback import delayed_ctrl
from bello_mujoco.terrain import TerrainConfig, terrain_surface_height_jax


from bello_mujoco.envs.gait import (
    gait_is_active,
    gait_phase_features,
    desired_gait_foot_contact,
    resolve_gait_command_transition,
)
from bello_mujoco.envs.locomotion_config import (
    DoneConfig,
    validate_locomotion_config,
    ObservationConfig,
    BelloEnvConfig,
)
from bello_mujoco.envs.locomotion_state import BelloState
from bello_mujoco.envs import locomotion_step


@dataclass(frozen=True)
class BelloLocomotionEnv:
    """Locomotion stepping with motor actions and joint-space kinematics.

    Actions select physical motor targets. Held actuators stay at their
    configured default targets; joint limits are enforced by physics and rewards.
    """

    model: BelloModel
    config: BelloEnvConfig = BelloEnvConfig()
    reward_config: RewardConfig = RewardConfig()
    command_config: CommandConfig = CommandConfig()
    done_config: DoneConfig = DoneConfig()
    random_push_scale: float | jax.Array = 1.0

    def __post_init__(self) -> None:
        validate_locomotion_config(
            self.config,
            self.reward_config,
            self.command_config,
            self.done_config,
            physics_dt=float(self.model.mj_model.opt.timestep),
        )
        self._validate_model_metadata()
        _ = self.model.target_limits
        if self.reward_config.enabled("stand_motor_continuous_torque_excess"):
            _ = self._motor_continuous_torque
        if self.reward_config.gait.phase_shaped_clearance or any(
            self.reward_config.enabled(name)
            for name in ("stand_foot_foreaft", "stand_foot_width", "foot_crossing")
        ):
            _ = self._sole_reward_geometry

    @property
    def task(self) -> Literal["locomotion"]:
        return "locomotion"

    @property
    def root_link_name(self) -> str:
        return self.config.root_link or self.model.robot.root_body_name

    @classmethod
    def load(
        cls,
        config: ModelConfig = ModelConfig(),
        *,
        warp: WarpConfig = WarpConfig(),
        batch_size: int = 1,
        env_config: BelloEnvConfig = BelloEnvConfig(),
        reward_config: RewardConfig = RewardConfig(),
        command_config: CommandConfig = CommandConfig(),
        done_config: DoneConfig = DoneConfig(),
        use_viewer_model: bool = False,
        load_mjx: bool = True,
    ) -> "BelloLocomotionEnv":
        return cls(
            load_model(
                config,
                warp=warp,
                batch_size=batch_size,
                terrain=env_config.terrain,
                simulation_physics=env_config.simulation_physics,
                model_randomization=env_config.model_randomization,
                spawn_clearance=env_config.spawn.clearance,
                use_viewer_model=use_viewer_model,
                load_mjx=load_mjx,
            ),
            config=env_config,
            reward_config=reward_config,
            command_config=command_config,
            done_config=done_config,
        )

    @classmethod
    def from_compiled(
        cls,
        prepared: CompiledRobot,
        *,
        warp: WarpConfig = WarpConfig(),
        batch_size: int = 1,
        env_config: BelloEnvConfig = BelloEnvConfig(),
        actuator_config: ActuatorConfig | None = None,
        imu_body_name: str | None = None,
        reward_config: RewardConfig = RewardConfig(),
        command_config: CommandConfig = CommandConfig(),
        done_config: DoneConfig = DoneConfig(),
        use_viewer_model: bool = False,
        load_mjx: bool = True,
    ) -> "BelloLocomotionEnv":
        return cls(
            load_compiled_model(
                prepared,
                warp=warp,
                batch_size=batch_size,
                terrain=env_config.terrain,
                simulation_physics=env_config.simulation_physics,
                model_randomization=env_config.model_randomization,
                actuator_config=actuator_config,
                imu_body_name=imu_body_name,
                spawn_clearance=env_config.spawn.clearance,
                use_viewer_model=use_viewer_model,
                load_mjx=load_mjx,
            ),
            config=env_config,
            reward_config=reward_config,
            command_config=command_config,
            done_config=done_config,
        )

    @property
    def decimation(self) -> int:
        return self.config.decimation

    @property
    def max_episode_steps(self) -> int:
        return self.config.max_episode_steps

    @property
    def terrain_config(self) -> TerrainConfig:
        return self.config.terrain

    @property
    def action_clip(self) -> float:
        return self.config.action_clip

    @property
    def ppo_stagger_initial_step_count(self) -> bool:
        return self.config.spawn.stagger_initial_step_count

    @property
    def action_size(self) -> int:
        return int(self.model.policy_ctrl_indices.size)

    @property
    def action_layout(self) -> ActionLayout:
        return ActionLayout(
            tuple(
                self.model.actuator_coordinate_names[index]
                for index in self.model.policy_ctrl_indices
            )
        )

    @property
    def observation_layout(self) -> ObservationLayout:
        return self._observation_layout(self.config.observation)

    @property
    def critic_observation_layout(self) -> ObservationLayout:
        return self._observation_layout(self.critic_observation_config)

    @property
    def observation_size(self) -> int:
        return self.observation_layout.size

    @property
    def critic_observation_config(self) -> ObservationConfig:
        return self.config.critic_observation

    @property
    def critic_observation_size(self) -> int:
        return self.critic_observation_layout.size

    @property
    def maximum_observation_delay_steps(self) -> int:
        if not self.config.observation_latency.enabled:
            return 0
        return self.config.observation_latency.delay_steps[1]

    @property
    def actor_feedback_size(self) -> int:
        return 9 + 2 * self.action_size

    def initial_task_state(self) -> jax.Array:
        if not self._curriculum_steps:
            return jnp.zeros((0,), dtype=jnp.float32)
        return jnp.zeros((1,), dtype=jnp.float32)

    @property
    def _curriculum_steps(self) -> int:
        return max(
            self.config.random_push.curriculum_steps,
            self.config.torso_com_curriculum.ramp_steps
            if self.config.torso_com_curriculum.enabled
            else 0,
        )

    def _push_curriculum_scale(self, task_state: jax.Array) -> jax.Array:
        config = self.config.random_push
        progress = jnp.clip(task_state[0] / config.curriculum_steps, 0.0, 1.0)
        return config.initial_scale + (1.0 - config.initial_scale) * progress

    def with_initial_step_counts(
        self, state: BelloState, step_count: jax.Array
    ) -> BelloState:
        return state.replace(step_count=step_count, episode_length=step_count)

    def with_task_state(self, task_state: jax.Array) -> "BelloLocomotionEnv":
        if self._curriculum_steps and task_state.shape != (1,):
            raise ValueError("locomotion curriculum task state must have shape (1,)")
        env = self
        if self.config.random_push.curriculum_steps:
            env = replace(
                self, random_push_scale=self._push_curriculum_scale(task_state)
            )
        if not self.config.torso_com_curriculum.enabled:
            return env
        if self.model.mjx_model is None:
            raise RuntimeError("torso COM curriculum requires an MJX model")
        progress = jnp.clip(
            task_state[0] / self.config.torso_com_curriculum.ramp_steps, 0.0, 1.0
        )
        torso_body_id = mujoco.mj_name2id(
            self.model.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
        )
        offsets = jnp.asarray(self.model.torso_com_offset)
        body_ipos = self.model.mjx_model.body_ipos.at[:, torso_body_id].add(
            (progress - 1.0) * offsets
        )
        model = replace(
            self.model,
            mjx_model=self.model.mjx_model.replace(body_ipos=body_ipos),
        )
        return replace(env, model=model)

    def update_task_state(
        self, task_state: jax.Array, info: dict[str, jax.Array]
    ) -> jax.Array:
        del info
        if not self._curriculum_steps:
            return task_state
        return jnp.minimum(
            task_state + 1.0,
            jnp.asarray([self._curriculum_steps], dtype=jnp.float32),
        )

    def task_state_metrics(self, task_state: jax.Array) -> dict[str, jax.Array]:
        metrics = {}
        if self.config.random_push.curriculum_steps:
            metrics["curriculum/push_scale"] = self._push_curriculum_scale(task_state)
        if not self.config.torso_com_curriculum.enabled:
            return metrics
        return {
            **metrics,
            "curriculum/torso_com_scale": jnp.clip(
                task_state[0] / self.config.torso_com_curriculum.ramp_steps,
                0.0,
                1.0,
            ),
        }

    def _observation_layout(self, observation: ObservationConfig) -> ObservationLayout:
        components: list[tuple[str, int]] = [
            ("imu_linear_acceleration", 3),
            ("imu_angular_velocity", 3),
            ("projected_gravity", 3),
        ]
        if self.command_config.enabled:
            components.append(("command", 3))
            components.append(("gait_phase", 3))
        components.extend(
            (
                ("joint_position", self.action_size),
                ("joint_velocity", self.action_size),
                ("last_action", self.action_size),
            )
        )
        if observation.privileged:
            components.extend(
                (
                    ("base_linear_velocity", 3),
                    ("base_height", 1),
                    ("actuator_force", self.action_size),
                )
            )
        return ObservationLayout(tuple(components))

    def _validate_model_metadata(self) -> None:
        model = self.model
        mj_model = model.mj_model
        require_index_range(
            "policy_ctrl_indices", model.policy_ctrl_indices, mj_model.nu
        )
        require_index_range("hold_ctrl_indices", model.hold_ctrl_indices, mj_model.nu)
        require_disjoint(
            "policy_ctrl_indices",
            model.policy_ctrl_indices,
            "hold_ctrl_indices",
            model.hold_ctrl_indices,
        )
        if model.policy_ctrl_indices.size + model.hold_ctrl_indices.size != mj_model.nu:
            raise ValueError(
                "policy_ctrl_indices and hold_ctrl_indices must cover every actuator "
                f"exactly once, got {model.policy_ctrl_indices.size} policy and "
                f"{model.hold_ctrl_indices.size} hold controls for nu={mj_model.nu}"
            )
        policy_peak_force = model.actuator_peak_force[model.policy_ctrl_indices]
        if not np.all(np.isfinite(policy_peak_force)) or np.any(
            policy_peak_force <= 0.0
        ):
            raise ValueError(
                "policy actuators require positive finite peak-force metadata"
            )
        if self.reward_config.enabled("torque_peak") and (
            np.any(mj_model.actuator_gaintype != mujoco.mjtGain.mjGAIN_FIXED)
            or np.any(mj_model.actuator_biastype != mujoco.mjtBias.mjBIAS_AFFINE)
            or np.any(mj_model.actuator_dyntype != mujoco.mjtDyn.mjDYN_NONE)
        ):
            raise ValueError(
                "torque_peak demand reconstruction requires stateless position "
                "actuators with fixed gain and affine bias"
            )
        require_index_range(
            "actuator_qpos_indices", model.actuator_qpos_indices, mj_model.nq
        )
        require_index_range(
            "actuator_qvel_indices", model.actuator_qvel_indices, mj_model.nv
        )
        require_index_range(
            "actuator_joint_ids", model.actuator_joint_ids, mj_model.njnt
        )
        require_index_range(
            "policy_qpos_indices", model.policy_qpos_indices, mj_model.nq
        )
        require_index_range(
            "policy_qvel_indices", model.policy_qvel_indices, mj_model.nv
        )
        require_index_range("foot_body_ids", model.foot_body_ids, mj_model.nbody)
        if self.reward_config.limits_pose.joint_pose_weights:
            self._joint_pose_weights()
        if self.command_config.enabled and model.foot_body_ids.size != 2:
            raise ValueError(
                "gait-conditioned locomotion requires exactly two foot bodies"
            )
        require_index_range("robot_geom_ids", model.robot_geom_ids, mj_model.ngeom)
        require_index_range("foot_geom_ids", model.foot_geom_ids, mj_model.ngeom)
        require_index_range("terrain_geom_ids", model.terrain_geom_ids, mj_model.ngeom)
        require_index_range(
            "nonfoot_robot_geom_ids", model.nonfoot_robot_geom_ids, mj_model.ngeom
        )
        require_index_range(
            "nonfoot_robot_body_ids", model.nonfoot_robot_body_ids, mj_model.nbody
        )
        if isinstance(mj_model, mujoco.MjModel):
            required_link_names = {
                self.root_link_name,
                model.imu_body_name,
                model.robot.locomotion_command_frame_body_name,
                *self.reward_config.enabled_link_names,
                *self.done_config.enabled_link_names,
            }
            if self.reward_config.enabled(
                "stand_foot_heading"
            ) or self.reward_config.enabled("foot_heading"):
                required_link_names.update(
                    name for name, _ in self.model.robot.foot_forward_axes
                )
            for link_name in required_link_names:
                if self._link_body_id(link_name) < 0:
                    raise ValueError(f"configured link/body not found: {link_name!r}")
        if model.actuator_coordinate_transform.shape != (mj_model.nu, mj_model.nu):
            raise ValueError(
                "actuator_coordinate_transform must have shape "
                f"({mj_model.nu}, {mj_model.nu}), got "
                f"{model.actuator_coordinate_transform.shape}"
            )
        if (
            model.joint_angle_offset.ndim != 2
            or model.joint_angle_offset.shape[1] != mj_model.nu
            or model.joint_angle_offset.shape[0] < 1
        ):
            raise ValueError(
                "joint_angle_offset must have shape (worlds, nu), got "
                f"{model.joint_angle_offset.shape} for nu={mj_model.nu}"
            )
        for name in ("default_joint_target", "joint_target_low", "joint_target_high"):
            value = getattr(model, name)
            if value.shape != (mj_model.nu,):
                raise ValueError(
                    f"{name} must have shape ({mj_model.nu},), got {value.shape}"
                )
        if np.any(model.joint_target_low >= model.joint_target_high):
            raise ValueError("joint target lower limits must be below upper limits")
        if model.actuator_qpos_indices.shape != (mj_model.nu,):
            raise ValueError(
                "actuator_qpos_indices must have one entry per actuator, got "
                f"{model.actuator_qpos_indices.shape} for nu={mj_model.nu}"
            )
        if model.actuator_qvel_indices.shape != (mj_model.nu,):
            raise ValueError(
                "actuator_qvel_indices must have one entry per actuator, got "
                f"{model.actuator_qvel_indices.shape} for nu={mj_model.nu}"
            )
        if self.reward_config.uses_foot_metadata and model.foot_body_ids.shape != (2,):
            raise ValueError(
                "enabled foot reward terms require exactly two foot bodies, got "
                f"{model.foot_body_ids.shape}. Disable foot-dependent rewards for "
                "assets without feet."
            )
        if (
            self.reward_config.uses_foot_contact_metadata
            and model.foot_geom_ids.size == 0
        ):
            raise ValueError(
                "enabled foot reward terms require foot collision geoms. Disable "
                "contact-dependent foot rewards for assets without feet."
            )
        if self.terrain_config.mode != "none" and model.terrain_geom_ids.size == 0:
            raise ValueError(
                f"terrain mode {self.terrain_config.mode!r} requires a terrain geom "
                f"named {self.terrain_config.name!r}"
            )
        if model.default_qpos.shape != (mj_model.nq,):
            raise ValueError(
                f"default_qpos shape {model.default_qpos.shape} does not match nq={mj_model.nq}"
            )
        if model.default_qvel.shape != (mj_model.nv,):
            raise ValueError(
                f"default_qvel shape {model.default_qvel.shape} does not match nv={mj_model.nv}"
            )
        if model.default_ctrl.shape != (mj_model.nu,):
            raise ValueError(
                f"default_ctrl shape {model.default_ctrl.shape} does not match nu={mj_model.nu}"
            )

    def reset(
        self,
        command: jax.Array | None = None,
        rng: jax.Array | None = None,
    ) -> tuple[BelloState, jax.Array]:
        return self._reset(command, rng, world_id=0)

    def _reset(
        self,
        command: jax.Array | None,
        rng: jax.Array | None,
        world_id: jax.Array | int,
        *,
        forward: bool = True,
        align_clearance: bool = True,
    ) -> tuple[BelloState, jax.Array]:
        command_rng = reset_rng = feedback_rng = obs_rng = latency_rng = gait_rng = rng
        if rng is not None:
            (
                command_rng,
                reset_rng,
                obs_rng,
                latency_rng,
                gait_rng,
            ) = jax.random.split(rng, 5)
            feedback_rng = jax.random.fold_in(obs_rng, 1)
            obs_rng = jax.random.fold_in(obs_rng, 2)
        command = self._resolve_command(command, command_rng)
        _, gait_phase, _ = resolve_gait_command_transition(
            jnp.zeros(3),
            command,
            jnp.asarray(0.0),
            jnp.asarray(0.0),
            jnp.asarray(0),
            self.command_config,
            self.config.gait,
            jnp.asarray(True),
            -self._locomotion_right_axis()[:2],
        )
        if gait_rng is not None:
            sampled_phase = jax.random.uniform(gait_rng, (), dtype=jnp.float32)
            gait_phase = jnp.where(
                command_is_moving(command, self.command_config), sampled_phase, 0.0
            )
        data = self.model.make_data()
        data = data.replace(
            qpos=jnp.asarray(self.model.default_qpos),
            qvel=jnp.asarray(self.model.default_qvel),
            ctrl=self.action_to_ctrl(jnp.zeros(self.action_size, dtype=jnp.float32)),
        )
        data = self._apply_reset_randomization(data, reset_rng, world_id)
        data = mjx.kinematics(self.model.mjx_model, data)
        state = BelloState(
            data=data,
            last_action=jnp.zeros(self.action_size, dtype=jnp.float32),
            last_action_delta=jnp.zeros(self.action_size, dtype=jnp.float32),
            tracking_ang_vel_z=jnp.asarray(0.0, dtype=jnp.float32),
            actor_feedback_history=jnp.zeros(
                (
                    self.maximum_observation_delay_steps + 1,
                    self.actor_feedback_size,
                ),
                dtype=jnp.float32,
            ),
            actor_feedback_delay_steps=self._sample_observation_delay(latency_rng),
            foot_contact_debounce_time=jnp.zeros(
                self.model.foot_body_ids.size, dtype=jnp.float32
            ),
            last_foot_contact=jnp.zeros(self.model.foot_body_ids.size, dtype=bool),
            command=command,
            command_target=command,
            gait_phase=gait_phase,
            standing_transition_pending=jnp.asarray(0, dtype=jnp.int32),
            next_start_left=~command_is_moving(command, self.command_config),
            standing_reward_blend=(
                ~command_is_moving(command, self.command_config)
            ).astype(jnp.float32),
            command_resample_step_count=jnp.asarray(0, dtype=jnp.int32),
            step_count=jnp.asarray(0, dtype=jnp.int32),
            push_steps_remaining=jnp.asarray(-1, dtype=jnp.int32),
            episode_return=jnp.asarray(0.0, dtype=jnp.float32),
            episode_track_lin_vel_xy=jnp.asarray(0.0, dtype=jnp.float32),
            episode_length=jnp.asarray(0, dtype=jnp.int32),
            last_episode_return=jnp.asarray(0.0, dtype=jnp.float32),
            last_episode_track_lin_vel_xy=jnp.asarray(0.0, dtype=jnp.float32),
            last_episode_length=jnp.asarray(0, dtype=jnp.int32),
        )
        if align_clearance:
            state = self._align_reset_foot_clearance(state, world_id)
            data = (
                mjx.forward(self.model.mjx_model, state.data)
                if forward
                else mjx.kinematics(self.model.mjx_model, state.data)
            )
            state = state.replace(data=data)
        actor_feedback = self._actor_feedback(state)
        if self.config.observation_low_pass_filter.noise_before_filter:
            actor_feedback = self._noisy_actor_feedback(actor_feedback, feedback_rng)
        state = state.replace(
            tracking_ang_vel_z=self._raw_tracking_ang_vel_z(state),
            actor_feedback_history=jnp.repeat(
                actor_feedback[None, :],
                self.maximum_observation_delay_steps + 1,
                axis=0,
            ),
        )
        return state, self.observation(state, obs_rng)

    def reset_batch(
        self,
        batch_size: int,
        command: jax.Array | None = None,
        rng: jax.Array | None = None,
    ) -> tuple[BelloState, jax.Array]:
        command_rng = reset_rng = feedback_rng = obs_rng = command_clock_rng = rng
        if rng is not None:
            command_rng, reset_rng, obs_rng = jax.random.split(rng, 3)
            command_clock_rng = jax.random.fold_in(rng, 3)
            feedback_rng = jax.random.fold_in(obs_rng, 1)
            obs_rng = jax.random.fold_in(obs_rng, 2)
        command = self._resolve_batch_command(batch_size, command, command_rng)
        world_ids = jnp.arange(batch_size, dtype=jnp.int32)
        if reset_rng is None:
            state, _ = jax.vmap(
                lambda item, world_id: self._reset(
                    item,
                    None,
                    world_id,
                    forward=False,
                    align_clearance=False,
                )
            )(command, world_ids)
        else:
            reset_keys = jax.random.split(reset_rng, batch_size)
            state, _ = jax.vmap(
                lambda item, key, world_id: self._reset(
                    item,
                    key,
                    world_id,
                    forward=False,
                    align_clearance=False,
                )
            )(command, reset_keys, world_ids)
        data = state.data.replace(_impl=state.data._impl.replace(nworld=batch_size))
        data = data.replace(
            ctrl=jax.vmap(self.action_to_ctrl)(
                jnp.zeros((batch_size, self.action_size), dtype=jnp.float32),
                self._joint_angle_offset_batch(batch_size),
            )
        )
        data = mjx.kinematics(self.model.mjx_model, data)
        state = state.replace(data=data)
        clearance_adjustment = jax.vmap(self._reset_foot_clearance_adjustment)(
            state, world_ids
        )
        root_qpos = int(self.model.root_qpos_adr)
        data = data.replace(
            qpos=data.qpos.at[:, root_qpos + 2].add(clearance_adjustment)
        )
        data = mjx.forward(self.model.mjx_model, data)
        state = state.replace(data=data)
        state = state.replace(
            tracking_ang_vel_z=jax.vmap(self._raw_tracking_ang_vel_z)(state)
        )
        actor_feedback = jax.vmap(self._actor_feedback)(state)
        if (
            feedback_rng is not None
            and self.config.observation_low_pass_filter.noise_before_filter
        ):
            feedback_keys = jax.random.split(feedback_rng, batch_size)
            actor_feedback = jax.vmap(self._noisy_actor_feedback)(
                actor_feedback, feedback_keys
            )
        state = state.replace(
            actor_feedback_history=jnp.repeat(
                actor_feedback[:, None, :],
                self.maximum_observation_delay_steps + 1,
                axis=1,
            ),
            command_resample_step_count=(
                self.initial_command_resample_step_counts(batch_size, command_clock_rng)
            ),
        )
        return state, self.observation_batch(state, obs_rng)

    def _reset_foot_clearance_adjustment(
        self, state: BelloState, world_id: jax.Array | int
    ) -> jax.Array:
        minimum_clearance = jnp.min(self._foot_height(state, world_id))
        return (
            jnp.asarray(self.config.spawn.clearance, dtype=jnp.float32)
            - minimum_clearance
        )

    def _align_reset_foot_clearance(
        self, state: BelloState, world_id: jax.Array | int
    ) -> BelloState:
        root_qpos = int(self.model.root_qpos_adr)
        adjustment = self._reset_foot_clearance_adjustment(state, world_id)
        return state.replace(
            data=state.data.replace(
                qpos=state.data.qpos.at[root_qpos + 2].add(adjustment)
            )
        )

    def initial_step_counts(
        self, batch_size: int, rng: jax.Array | None = None
    ) -> jax.Array:
        if rng is None or not self.config.spawn.stagger_initial_step_count:
            return jnp.zeros((batch_size,), dtype=jnp.int32)
        return jax.random.randint(
            rng,
            (batch_size,),
            minval=0,
            maxval=self.max_episode_steps,
            dtype=jnp.int32,
        )

    def initial_command_resample_step_counts(
        self, batch_size: int, rng: jax.Array | None = None
    ) -> jax.Array:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if rng is None or not self.command_config.enabled:
            return jnp.zeros((batch_size,), dtype=jnp.int32)

        resample_steps = self._command_resample_steps()
        permutation_key, shift_key = jax.random.split(rng)
        evenly_spaced = (
            jnp.arange(batch_size, dtype=jnp.int32) * resample_steps // batch_size
        )
        shift = jax.random.randint(
            shift_key, (), minval=0, maxval=resample_steps, dtype=jnp.int32
        )
        return jax.random.permutation(
            permutation_key, jnp.mod(evenly_spaced + shift, resample_steps)
        )

    def _apply_reset_randomization(self, data, rng, world_id=0):
        return randomize_spawn(
            self.model, self.config.spawn, self._terrain_height, data, rng, world_id
        )

    def delayed_ctrl(
        self,
        previous_ctrl: jax.Array | np.ndarray,
        requested_ctrl: jax.Array | np.ndarray,
        substep: jax.Array | int,
    ) -> jax.Array:
        return delayed_ctrl(
            previous_ctrl,
            requested_ctrl,
            substep,
            delay_seconds=self.config.command_delay_seconds,
            physics_dt=float(self.model.mj_model.opt.timestep),
        )

    def step(
        self, state: BelloState, action: jax.Array
    ) -> tuple[BelloState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        return locomotion_step.step(self, state, action)

    def step_batch(
        self, state: BelloState, action: jax.Array, rng: jax.Array | None = None
    ) -> tuple[BelloState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        return locomotion_step.step_batch(self, state, action, rng)

    def observation_batch(
        self, state: BelloState, rng: jax.Array | None = None
    ) -> jax.Array:
        world_ids = jnp.arange(int(state.command.shape[0]), dtype=jnp.int32)
        if rng is None:
            return jax.vmap(
                lambda item, world_id: self.observation(item, None, world_id)
            )(state, world_ids)
        keys = jax.random.split(rng, int(state.command.shape[0]))
        return jax.vmap(self.observation)(state, keys, world_ids)

    def critic_observation_batch(
        self, state: BelloState, rng: jax.Array | None = None
    ) -> jax.Array:
        world_ids = jnp.arange(int(state.command.shape[0]), dtype=jnp.int32)
        if rng is None:
            return jax.vmap(
                lambda item, world_id: self.critic_observation(item, None, world_id)
            )(state, world_ids)
        keys = jax.random.split(rng, int(state.command.shape[0]))
        return jax.vmap(self.critic_observation)(state, keys, world_ids)

    def observation(
        self,
        state: BelloState,
        rng: jax.Array | None = None,
        world_id: jax.Array | int = 0,
    ) -> jax.Array:
        return self._observation(
            state,
            self.config.observation,
            rng,
            world_id,
            _feedback=self._delayed_actor_feedback(state),
        )

    def critic_observation(
        self,
        state: BelloState,
        rng: jax.Array | None = None,
        world_id: jax.Array | int = 0,
    ) -> jax.Array:
        return self._observation(state, self.critic_observation_config, rng, world_id)

    def _observation(
        self,
        state: BelloState,
        observation: ObservationConfig,
        rng: jax.Array | None = None,
        world_id: jax.Array | int = 0,
        *,
        _feedback: jax.Array | None = None,
    ) -> jax.Array:
        gait_features = None
        if self.command_config.enabled:
            gait_features = gait_phase_features(
                state.gait_phase,
                state.command,
                self.command_config,
                state.standing_transition_pending,
                self.config.gait,
            )
        filtering = self.config.observation_low_pass_filter
        return build_observation(
            self._frame_terms(state, world_id),
            gait_features,
            observation,
            rng,
            _feedback,
            noise_before_filter=filtering.noise_before_filter,
            residual_noise_scale=filtering.residual_noise_scale,
        )

    def _actor_feedback(self, state: BelloState) -> jax.Array:
        policy_indices = jnp.asarray(self.model.policy_ctrl_indices)
        qpos_indices = jnp.asarray(self.model.actuator_qpos_indices)[policy_indices]
        qvel_indices = jnp.asarray(self.model.actuator_qvel_indices)[policy_indices]
        joint_position = (
            state.data.qpos[qpos_indices]
            - jnp.asarray(self.model.default_joint_target)[policy_indices]
        )
        return jnp.concatenate(
            (
                self._imu_linear_acceleration(state),
                self._imu_angular_velocity(state),
                self._link_projected_gravity(state, self.model.imu_body_name),
                joint_position,
                state.data.qvel[qvel_indices],
            )
        )

    def _delayed_actor_feedback(self, state: BelloState) -> jax.Array:
        if (
            not self.config.observation_low_pass_filter.enabled
            and not self.config.observation_latency.enabled
        ):
            return self._actor_feedback(state)
        return state.actor_feedback_history[state.actor_feedback_delay_steps]

    def _advance_actor_feedback_history(
        self,
        history: jax.Array,
        state: BelloState,
        rng: jax.Array | None = None,
    ) -> jax.Array:
        return advance_feedback_history(
            history,
            self._actor_feedback(state),
            rng,
            self.control_timestep,
            self.config.observation.noise,
            self.config.observation_low_pass_filter,
        )

    def _sample_observation_delay(self, rng: jax.Array | None) -> jax.Array:
        return sample_feedback_delay(self.config.observation_latency, rng)

    def action_to_ctrl(
        self,
        action: jax.Array,
        joint_angle_offset: jax.Array | None = None,
    ) -> jax.Array:
        ctrl, _ = self._action_target_details(action, joint_angle_offset)
        return ctrl

    def _locomotion_right_axis(self) -> jax.Array:
        forward_x, forward_y, _ = self.model.robot.locomotion_forward_axis
        return jnp.asarray((forward_y, -forward_x, 0.0), dtype=jnp.float32)

    def ppo_action_diagnostics(self, env_action: jax.Array) -> dict[str, jax.Array]:
        flat_action = env_action.reshape((-1, self.action_size))
        num_steps, num_envs = env_action.shape[:2]
        joint_angle_offset = self._joint_angle_offset_batch(num_envs)
        flat_joint_angle_offset = jnp.broadcast_to(
            joint_angle_offset[None, ...],
            (num_steps, *joint_angle_offset.shape),
        ).reshape((-1, joint_angle_offset.shape[-1]))
        ctrl = jax.vmap(self.action_to_ctrl)(flat_action, flat_joint_angle_offset)
        policy_ctrl_indices = jnp.asarray(self.model.policy_ctrl_indices)
        policy_ctrl = ctrl[:, policy_ctrl_indices]
        ctrlrange = jnp.asarray(self.model.mj_model.actuator_ctrlrange)[
            policy_ctrl_indices
        ]
        ctrl_low = ctrlrange[:, 0]
        ctrl_high = ctrlrange[:, 1]
        ctrl_margin = jnp.minimum(policy_ctrl - ctrl_low, ctrl_high - policy_ctrl)
        saturated = (policy_ctrl <= ctrl_low + 1.0e-6) | (
            policy_ctrl >= ctrl_high - 1.0e-6
        )
        return {
            "ctrl_saturation_fraction": saturated.astype(jnp.float32).mean(),
            "ctrl_margin_min": jnp.min(ctrl_margin),
        }

    def _action_target_details(
        self, action: jax.Array, joint_angle_offset: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array]:
        if joint_angle_offset is None:
            joint_angle_offset = jnp.asarray(self.model.joint_angle_offset[0])
        action = jnp.asarray(action, dtype=jnp.float32)
        ctrl = bounded_motor_target(
            action,
            motor_default(self.model, joint_angle_offset),
            jnp.asarray(self.model.mj_model.actuator_ctrlrange),
            jnp.asarray(self.model.policy_ctrl_indices),
            self.action_clip,
        )
        saturation = jnp.mean((jnp.abs(action) > self.action_clip).astype(jnp.float32))
        return project_motor_target(
            ctrl, jnp.asarray(self.model.default_ctrl), self.model.target_limits
        ), saturation

    def _joint_angle_offset_batch(self, batch_size: int) -> jax.Array:
        offset = jnp.asarray(self.model.joint_angle_offset, dtype=jnp.float32)
        if offset.shape[0] == 1:
            return jnp.broadcast_to(offset, (batch_size, offset.shape[1]))
        if offset.shape[0] != batch_size:
            raise ValueError(
                "model randomization batch does not match environment batch: "
                f"{offset.shape[0]} != {batch_size}"
            )
        return offset

    def _resolve_command(
        self,
        command: jax.Array | None,
        rng: jax.Array | None,
    ) -> jax.Array:
        if not self.command_config.enabled:
            return jnp.zeros(3, dtype=jnp.float32)
        if command is not None:
            command = jnp.asarray(command, dtype=jnp.float32)
            if command.shape != (3,):
                raise ValueError(f"command must have shape (3,), got {command.shape}")
            return command
        if rng is not None:
            return self.command_config.sample(rng)
        return jnp.zeros(3, dtype=jnp.float32)

    def _resolve_batch_command(
        self,
        batch_size: int,
        command: jax.Array | None,
        rng: jax.Array | None,
    ) -> jax.Array:
        if not self.command_config.enabled:
            return jnp.zeros((batch_size, 3), dtype=jnp.float32)
        if command is None:
            if rng is not None:
                return self.command_config.sample_stratified_batch(rng, batch_size)
            return jnp.zeros((batch_size, 3), dtype=jnp.float32)
        command = jnp.asarray(command, dtype=jnp.float32)
        if command.shape == (3,):
            return jnp.repeat(command[None, :], batch_size, axis=0)
        if command.shape != (batch_size, 3):
            raise ValueError(
                "batch command must have shape (3,) or "
                f"({batch_size}, 3), got {command.shape}"
            )
        return command

    def _maybe_resample_batch_command_and_phase(
        self,
        state: BelloState,
        rng: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        resample_steps = self._command_resample_steps()
        if rng is None or not self.command_config.enabled:
            return (
                state.command,
                state.gait_phase,
                state.standing_transition_pending,
                jnp.minimum(state.command_resample_step_count, resample_steps),
                jnp.zeros_like(state.command_resample_step_count, dtype=bool),
                state.command_target,
            )
        resample = state.command_resample_step_count >= resample_steps
        sampled = self._resample_batch_command(
            state.command, int(state.command.shape[0]), rng
        )
        pending_command = jnp.where(
            state.standing_transition_pending[:, None],
            jnp.zeros_like(state.command),
            state.command_target,
        )
        requested_command = jnp.where(resample[:, None], sampled, pending_command)
        command, gait_phase, standing_transition_pending = (
            resolve_gait_command_transition(
                state.command,
                requested_command,
                state.gait_phase,
                state.gait_phase,
                state.standing_transition_pending,
                self.command_config,
                self.config.gait,
                state.next_start_left,
                -self._locomotion_right_axis()[:2],
            )
        )
        return (
            command,
            gait_phase,
            standing_transition_pending,
            jnp.where(resample, 0, state.command_resample_step_count),
            resample,
            requested_command,
        )

    def _command_resample_steps(self) -> int:
        return max(
            1, int(round(self.command_config.resample_seconds / self.control_timestep))
        )

    def _resample_batch_command(
        self,
        command: jax.Array,
        batch_size: int,
        rng: jax.Array | None,
    ) -> jax.Array:
        if rng is None:
            return command
        if not self.command_config.enabled:
            return jnp.zeros_like(command)
        if self.command_config.standing_transitions_enabled:
            return self.command_config.sample_stratified_batch(rng, batch_size)
        return self._reset_batch_command(command, batch_size, rng)

    def _reset_batch_command(
        self,
        command: jax.Array,
        batch_size: int,
        rng: jax.Array | None,
    ) -> jax.Array:
        if rng is None:
            return command
        if not self.command_config.enabled:
            return jnp.zeros_like(command)
        standing = command_is_standing(command, self.command_config)
        moving = self.command_config.sample_moving(rng, (batch_size,))
        return jnp.where(standing[:, None], jnp.zeros_like(moving), moving)

    def _maybe_apply_random_push_batch(self, data, steps_remaining, rng):
        return apply_push_batch(
            self.model,
            self.config.random_push,
            self.control_timestep,
            data,
            steps_remaining,
            rng,
            scale=self.random_push_scale,
        )

    @property
    def control_timestep(self) -> float:
        return self.decimation * float(self.model.mj_model.opt.timestep)

    def _frame_terms(
        self, state: BelloState, world_id: jax.Array | int = 0
    ) -> tuple[jax.Array, ...]:
        data = state.data
        policy_idx = jnp.asarray(self.model.policy_ctrl_indices)
        qpos_idx = jnp.asarray(self.model.actuator_qpos_indices)[policy_idx]
        qvel_idx = jnp.asarray(self.model.actuator_qvel_indices)[policy_idx]
        joint_pos = (
            data.qpos[qpos_idx]
            - jnp.asarray(self.model.default_joint_target)[policy_idx]
        )
        joint_vel = data.qvel[qvel_idx]
        actuator_force = data.actuator_force[policy_idx]
        base_lin_vel = self._link_lin_vel(state, self.root_link_name)
        imu_lin_acc = self._imu_linear_acceleration(state)
        imu_ang_vel = self._imu_angular_velocity(state)
        base_height = jnp.asarray(
            [self._link_clearance(state, self.root_link_name, world_id)],
            dtype=jnp.float32,
        )
        projected_gravity = self._link_projected_gravity(
            state, self.model.imu_body_name
        )
        return (
            base_lin_vel,
            imu_lin_acc,
            imu_ang_vel,
            projected_gravity,
            base_height,
            state.command,
            joint_pos,
            joint_vel,
            actuator_force,
            state.last_action,
        )

    def _actuator_indices_named(self, *suffixes: str) -> np.ndarray:
        indices = [
            index
            for index, name in enumerate(self.model.actuator_joint_names)
            if name.endswith(suffixes)
        ]
        return np.asarray(indices, dtype=np.int32)

    def _leg_joint_body_ids(self) -> np.ndarray:
        actuator_indices = self._actuator_indices_named(
            "hip_pitch_joint",
            "hip_roll_joint",
            "hip_yaw_joint",
            "knee_joint",
            "ankle_pitch_joint",
            "ankle_roll_joint",
        )
        policy_indices = np.asarray(self.model.policy_ctrl_indices, dtype=np.int32)
        actuator_indices = actuator_indices[
            np.isin(actuator_indices, policy_indices, assume_unique=True)
        ]
        joint_ids = np.asarray(self.model.actuator_joint_ids)[actuator_indices]
        body_ids = np.asarray(self.model.mj_model.jnt_bodyid)[joint_ids]
        if body_ids.size == 0 or np.unique(body_ids).size != body_ids.size:
            raise ValueError(
                "locomotion joint-force reward requires unique policy-controlled "
                "hip, knee, and ankle child bodies"
            )
        return body_ids.astype(np.int32)

    def _leg_joint_force(self, state: BelloState) -> jax.Array:
        body_ids = jnp.asarray(self._leg_joint_body_ids())
        linear_force = state.data._impl.cfrc_int[body_ids, 3:]
        return jnp.linalg.norm(linear_force, axis=1)

    def _robot_weight(self) -> jax.Array:
        total_mass = float(np.sum(self.model.mj_model.body_mass))
        gravity = float(np.linalg.norm(self.model.mj_model.opt.gravity))
        return jnp.asarray(total_mass * gravity, dtype=jnp.float32)

    def _joint_pose_weights(self) -> jax.Array:
        weights = np.zeros(self.model.mj_model.nu, dtype=np.float32)
        configured = self.reward_config.limits_pose.joint_pose_weights
        if configured:
            policy_indices = np.asarray(self.model.policy_ctrl_indices)
            policy_names = {
                self.model.actuator_joint_names[index] for index in policy_indices
            }
            configured_weights = dict(configured)
            if set(configured_weights) != policy_names:
                raise ValueError(
                    "joint_pose_weights must name every policy joint exactly once; "
                    f"missing={sorted(policy_names - configured_weights.keys())}, "
                    f"extra={sorted(configured_weights.keys() - policy_names)}"
                )
            for index in policy_indices:
                weights[index] = configured_weights[
                    self.model.actuator_joint_names[index]
                ]
            return jnp.asarray(weights)
        weights[np.asarray(self.model.policy_ctrl_indices)] = 1.0
        low_weight_indices = self._actuator_indices_named(
            "hip_pitch_joint", "knee_joint"
        )
        weights[low_weight_indices] = 0.01
        return jnp.asarray(weights)

    @cached_property
    def _motor_continuous_torque(self) -> jax.Array:
        return jnp.asarray(motor_continuous_torque(self.model), dtype=jnp.float32)

    def _action_smoothness_weights(self) -> jax.Array:
        names = self.action_layout.names
        return jnp.asarray(
            [
                (
                    self.reward_config.smoothness.ankle_action_weight
                    if "_ankle_" in name
                    else self.reward_config.smoothness.waist_action_weight
                    if name in {"waist_yaw_joint", "waist_roll_joint"}
                    else 1.0
                )
                for name in names
            ],
            dtype=jnp.float32,
        )

    def requested_actuator_force(self, data: mjx.Data) -> jax.Array:
        """Return the unclipped force requested by the position servos."""

        model = self.model.mj_model
        gain = jnp.asarray(model.actuator_gainprm[:, 0], dtype=jnp.float32)
        bias_position = jnp.asarray(model.actuator_biasprm[:, 1], dtype=jnp.float32)
        bias_velocity = jnp.asarray(model.actuator_biasprm[:, 2], dtype=jnp.float32)
        return (
            gain * data.ctrl
            + bias_position * data.actuator_length
            + bias_velocity * data._impl.actuator_velocity
        )

    def _all_actuator_joint_terms(
        self, state: BelloState
    ) -> tuple[jax.Array, jax.Array]:
        qpos_idx = jnp.asarray(self.model.actuator_qpos_indices)
        qvel_idx = jnp.asarray(self.model.actuator_qvel_indices)
        joint_pos = (
            state.data.qpos[qpos_idx] - jnp.asarray(self.model.default_qpos)[qpos_idx]
        )
        joint_vel = state.data.qvel[qvel_idx]
        return joint_pos, joint_vel

    def _link_body_id(self, link_name: str) -> int:
        return int(
            mujoco.mj_name2id(self.model.mj_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
        )

    def _link_quat(self, state: BelloState, link_name: str) -> jax.Array:
        if link_name == self.model.robot.root_body_name:
            root_qpos = int(self.model.root_qpos_adr)
            return state.data.qpos[root_qpos + 3 : root_qpos + 7]
        return state.data.xquat[self._link_body_id(link_name)]

    def _link_position(self, state: BelloState, link_name: str) -> jax.Array:
        if link_name == self.model.robot.root_body_name:
            root_qpos = int(self.model.root_qpos_adr)
            return state.data.qpos[root_qpos : root_qpos + 3]
        return state.data.xpos[self._link_body_id(link_name)]

    def _link_clearance(
        self,
        state: BelloState,
        link_name: str,
        world_id: jax.Array | int = 0,
    ) -> jax.Array:
        position = self._link_position(state, link_name)
        return position[2] - self._terrain_height(position[:2], world_id)

    def _link_lin_vel(self, state: BelloState, link_name: str) -> jax.Array:
        return self._link_body_lin_vel(state, link_name)

    def _link_body_lin_vel(self, state: BelloState, link_name: str) -> jax.Array:
        if link_name == self.model.robot.root_body_name:
            root_qpos = int(self.model.root_qpos_adr)
            root_qvel = int(self.model.root_qvel_adr)
            return quat_rotate_inverse(
                state.data.qpos[root_qpos + 3 : root_qpos + 7],
                state.data.qvel[root_qvel : root_qvel + 3],
            )
        return quat_rotate_inverse(
            self._link_quat(state, link_name),
            self._link_world_lin_vel(state, link_name),
        )

    def _link_body_lin_acc(self, state: BelloState, link_name: str) -> jax.Array:
        body_id = self._link_body_id(link_name)
        root_body_id = int(self.model.mj_model.body_rootid[body_id])
        offset = state.data.xpos[body_id] - state.data.subtree_com[root_body_id]
        cacc = state.data._impl.cacc[body_id]
        cvel = state.data.cvel[body_id]
        world_lin_acc = cacc[3:6] - jnp.cross(offset, cacc[:3])
        world_lin_vel = cvel[3:6] - jnp.cross(offset, cvel[:3])
        world_lin_acc = world_lin_acc + jnp.cross(cvel[:3], world_lin_vel)
        return quat_rotate_inverse(self._link_quat(state, link_name), world_lin_acc)

    def _imu_linear_acceleration(self, state: BelloState) -> jax.Array:
        """MuJoCo accelerometer-equivalent output at the configured body origin."""

        return self._link_body_lin_acc(state, self.model.imu_body_name)

    def _imu_angular_velocity(self, state: BelloState) -> jax.Array:
        """MuJoCo gyro-equivalent output in the configured IMU body frame."""

        return self._link_body_ang_vel(state, self.model.imu_body_name)

    def _link_world_lin_vel(self, state: BelloState, link_name: str) -> jax.Array:
        if link_name == self.model.robot.root_body_name:
            root_qvel = int(self.model.root_qvel_adr)
            return state.data.qvel[root_qvel : root_qvel + 3]
        return self._body_world_lin_vel(
            state, np.asarray([self._link_body_id(link_name)], dtype=np.int32)
        )[0]

    def _body_world_lin_vel(self, state: BelloState, body_ids: np.ndarray) -> jax.Array:
        body_ids = np.asarray(body_ids, dtype=np.int32)
        root_body_ids = np.asarray(
            self.model.mj_model.body_rootid[body_ids], dtype=np.int32
        )
        body_ids_jax = jnp.asarray(body_ids)
        cvel = state.data.cvel[body_ids_jax]
        offset = (
            state.data.xpos[body_ids_jax]
            - state.data.subtree_com[jnp.asarray(root_body_ids)]
        )
        return cvel[:, 3:6] - jnp.cross(offset, cvel[:, :3])

    def _link_yaw_frame_lin_vel(
        self, state: BelloState, link_name: str, frame_link_name: str | None = None
    ) -> jax.Array:
        frame_link_name = link_name if frame_link_name is None else frame_link_name
        return quat_rotate_inverse(
            yaw_quat(self._link_quat(state, frame_link_name)),
            self._link_world_lin_vel(state, link_name),
        )

    def _link_body_ang_vel(self, state: BelloState, link_name: str) -> jax.Array:
        if link_name == self.model.robot.root_body_name:
            root_qvel = int(self.model.root_qvel_adr)
            return state.data.qvel[root_qvel + 3 : root_qvel + 6]
        body_id = self._link_body_id(link_name)
        return quat_rotate_inverse(
            self._link_quat(state, link_name), state.data.cvel[body_id, :3]
        )

    def _link_projected_gravity(self, state: BelloState, link_name: str) -> jax.Array:
        return quat_rotate_inverse(
            self._link_quat(state, link_name),
            jnp.asarray([0.0, 0.0, -1.0], dtype=jnp.float32),
        )

    def done_flags(
        self, state: BelloState, world_id: jax.Array | int = 0
    ) -> dict[str, jax.Array]:
        config = self.done_config
        bad_height = (
            self._link_clearance(state, self._done_link(config.base_height), world_id)
            < config.base_height.min_height
            if config.base_height.enabled
            else jnp.asarray(False)
        )
        if config.bad_orientation.enabled:
            projected_gravity = self._link_projected_gravity(
                state, self._done_link(config.bad_orientation)
            )
            bad_orientation = (
                projected_gravity_tilt_angle(projected_gravity)
                >= config.bad_orientation.limit_angle
            )
        else:
            bad_orientation = jnp.asarray(False)
        bad_pose = bad_height | bad_orientation
        timeout = (
            state.step_count >= self.max_episode_steps
            if config.timeout.enabled
            else jnp.asarray(False)
        )
        position_fields = (state.data.qpos, state.data.xpos, state.data.xquat)
        velocity_fields = (state.data.qvel, state.data.cvel)
        acceleration_fields = (state.data.qacc, state.data._impl.cacc)
        actuator_force_fields = (state.data.actuator_force,)
        constraint_force_fields = (
            state.data.qfrc_constraint,
            state.data._impl.efc__force,
        )
        all_fields = (
            *position_fields,
            *velocity_fields,
            *acceleration_fields,
            *actuator_force_fields,
            *constraint_force_fields,
        )
        finite = jnp.all(
            jnp.stack([jnp.all(jnp.isfinite(value)) for value in all_fields])
        )
        nonfinite = ~finite if config.nonfinite.enabled else jnp.asarray(False)
        safety = self.config.numerical_safety
        unstable = finite & (
            jnp.any(
                jnp.stack(
                    [
                        jnp.any(jnp.abs(value) > safety.max_abs_position)
                        for value in position_fields
                    ]
                )
            )
            | jnp.any(
                jnp.stack(
                    [
                        jnp.any(jnp.abs(value) > safety.max_abs_velocity)
                        for value in velocity_fields
                    ]
                )
            )
            | jnp.any(
                jnp.stack(
                    [
                        jnp.any(jnp.abs(value) > safety.max_abs_acceleration)
                        for value in acceleration_fields
                    ]
                )
            )
            | jnp.any(
                jnp.stack(
                    [
                        jnp.any(jnp.abs(value) > safety.max_abs_actuator_force)
                        for value in actuator_force_fields
                    ]
                )
            )
            | jnp.any(
                jnp.stack(
                    [
                        jnp.any(jnp.abs(value) > safety.max_abs_constraint_force)
                        for value in constraint_force_fields
                    ]
                )
            )
        )
        terminal = bad_pose | nonfinite | unstable
        done = terminal | timeout
        return {
            "done/any": done,
            "done/terminal": terminal,
            "done/base_height": bad_height,
            "done/bad_orientation": bad_orientation,
            "done/timeout": timeout,
            "done/nonfinite": nonfinite,
            "done/unstable": unstable,
        }

    def _base_height(
        self, state: BelloState, world_id: jax.Array | int = 0
    ) -> jax.Array:
        return self._link_clearance(state, self.root_link_name, world_id)

    def reward(
        self,
        state: BelloState,
        next_state: BelloState,
        action: jax.Array,
        done: jax.Array,
        foot_contact: jax.Array,
        foot_touchdown: jax.Array,
        foot_normal_impulse: jax.Array,
        foot_peak_normal_force: jax.Array,
        touchdown_foot_linear_velocity: jax.Array,
        requested_actuator_force: jax.Array,
        foot_linear_velocity: jax.Array,
        foot_angular_speed: jax.Array,
        world_id: jax.Array | int = 0,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        command = next_state.command
        qpos_idx = jnp.asarray(self.model.actuator_qpos_indices)
        joint_pos_rel, _ = self._all_actuator_joint_terms(next_state)
        lin_vel_xy = self._link_yaw_frame_lin_vel(
            next_state,
            self._reward_link("track_lin_vel_xy"),
            self._reward_frame_link("track_lin_vel_xy"),
        )[:2]
        yaw_vel_z = next_state.tracking_ang_vel_z
        lin_vel_z = self._link_body_lin_vel(next_state, self._reward_link("lin_vel_z"))[
            2
        ]
        ang_vel_xy = self._link_body_ang_vel(
            next_state, self._reward_link("ang_vel_xy")
        )[:2]
        projected_gravity = self._link_projected_gravity(
            next_state, self._reward_link("flat_orientation")
        )
        base_height = self._link_clearance(
            next_state,
            self._reward_link("base_height"),
            world_id,
        )
        return compute_locomotion_reward(
            self.reward_config,
            RewardInputs(
                command=command,
                lin_vel_xy=lin_vel_xy,
                yaw_vel_z=yaw_vel_z,
                lin_vel_z=lin_vel_z,
                ang_vel_xy=ang_vel_xy,
                projected_gravity=projected_gravity,
                base_height=base_height,
                joint_pos=next_state.data.qpos[qpos_idx],
                joint_pos_rel=joint_pos_rel,
                action=action,
                last_action=state.last_action,
                last_action_delta=state.last_action_delta,
                action_smoothness_weights=self._action_smoothness_weights(),
                actuator_force=next_state.data.actuator_force,
                requested_actuator_force=requested_actuator_force,
                actuator_peak_force=jnp.asarray(self.model.actuator_peak_force),
                motor_continuous_torque=(
                    self._motor_continuous_torque
                    if self.reward_config.enabled(
                        "stand_motor_continuous_torque_excess"
                    )
                    else None
                ),
                joint_force=self._leg_joint_force(next_state),
                robot_weight=self._robot_weight(),
                joint_limit_low=self._actuator_joint_limits[:, 0],
                joint_limit_high=self._actuator_joint_limits[:, 1],
                policy_joint_indices=jnp.asarray(self.model.policy_ctrl_indices),
                foot_contact=foot_contact,
                foot_touchdown=foot_touchdown,
                foot_normal_impulse=foot_normal_impulse,
                foot_peak_normal_force=foot_peak_normal_force,
                touchdown_foot_linear_velocity=touchdown_foot_linear_velocity,
                foot_linear_velocity=foot_linear_velocity,
                foot_angular_speed=foot_angular_speed,
                foot_heading_error=self._foot_heading_error(next_state),
                foot_height=self._foot_height(next_state, world_id),
                gait_active=gait_is_active(
                    command,
                    self.command_config,
                    next_state.standing_transition_pending,
                    self.config.gait,
                ),
                standing_reward_blend=next_state.standing_reward_blend,
                **self._foot_placement_inputs(next_state, world_id),
                desired_foot_contact=desired_gait_foot_contact(
                    next_state.gait_phase, self.config.gait
                ),
                joint_pose_weights=self._joint_pose_weights(),
                hip_joint_indices=self._actuator_indices_named(
                    "hip_roll_joint", "hip_yaw_joint"
                ),
                knee_joint_indices=self._actuator_indices_named("knee_joint"),
                done=done,
                control_timestep=jnp.asarray(self.control_timestep, dtype=jnp.float32),
                action_clip_limit=jnp.asarray(self.action_clip, dtype=jnp.float32),
            ),
            self.command_config,
        )

    def _raw_tracking_ang_vel_z(self, state: BelloState) -> jax.Array:
        return self._link_body_ang_vel(state, self._reward_link("track_ang_vel_z"))[2]

    def filter_tracking_ang_vel_z(
        self, previous: jax.Array, measured: jax.Array
    ) -> jax.Array:
        """Low-pass the measured yaw rate used by the tracking reward."""

        alpha = 1.0 - jnp.exp(
            -2.0
            * jnp.pi
            * jnp.asarray(
                self.reward_config.tracking.ang_vel_z_lpf_cutoff_hz,
                dtype=jnp.float32,
            )
            * jnp.asarray(self.control_timestep, dtype=jnp.float32)
        )
        return previous + alpha * (measured - previous)

    def sanitize_nonfinite_reward(
        self,
        reward: jax.Array,
        reward_info: dict[str, jax.Array],
        done_flags: dict[str, jax.Array],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        invalid_physics = done_flags["done/nonfinite"] | done_flags["done/unstable"]
        limit = jnp.asarray(
            self.config.numerical_safety.max_abs_reward_term, dtype=jnp.float32
        )
        sanitized = {}
        for name, value in reward_info.items():
            value = jnp.nan_to_num(value, nan=0.0, posinf=limit, neginf=-limit)
            value = jnp.clip(value, -limit, limit)
            if name == "reward/termination":
                sanitized[name] = value
            else:
                sanitized[name] = jnp.where(
                    invalid_physics, jnp.zeros_like(value), value
                )
        reward = sum(sanitized.values(), jnp.zeros_like(reward))
        return reward, sanitized

    def debounce_foot_contact(self, state: BelloState, raw_foot_contact, done):
        return debounce_contact(
            raw_foot_contact,
            state.last_foot_contact,
            state.foot_contact_debounce_time,
            done,
            self.config.contact,
            self.control_timestep,
        )

    def _reward_link(self, term_name: str) -> str:
        link = getattr(self.reward_config.links, term_name)
        enabled = self.reward_config.enabled(term_name)
        if term_name == "track_lin_vel_xy":
            enabled = self.reward_config.uses_planar_velocity
        elif term_name == "track_ang_vel_z":
            enabled = self.reward_config.uses_yaw_velocity
        if enabled and link is not None:
            return link
        return self.root_link_name

    def _reward_frame_link(self, term_name: str) -> str:
        velocity_link = self._reward_link(term_name)
        frame_link = (
            self.reward_config.links.track_lin_vel_xy_frame
            if term_name == "track_lin_vel_xy"
            else None
        )
        enabled = self.reward_config.enabled(term_name)
        if term_name == "track_lin_vel_xy":
            enabled = self.reward_config.uses_planar_velocity
        if enabled and frame_link is not None:
            return frame_link
        if term_name == "track_lin_vel_xy":
            return self.model.robot.locomotion_command_frame_body_name
        return velocity_link

    def _done_link(self, term) -> str:
        if term.enabled and term.link is not None:
            return term.link
        return self.root_link_name

    def _contact_terms(self, state: BelloState, world_id: jax.Array):
        contact, force, undesired, collision = contact_forces(
            self.model, self.config.contact, state.data, world_id
        )
        return (
            contact,
            force,
            self._foot_world_linear_velocity(state),
            self._foot_angular_speed(state),
            undesired,
            collision,
        )

    def _foot_height(
        self, state: BelloState, world_id: jax.Array | int = 0
    ) -> jax.Array:
        sole_position = self._foot_sole_positions(state, world_id)
        return sole_position[:, 2] - self._terrain_height(
            sole_position[:, :2], world_id
        )

    @cached_property
    def _sole_reward_geometry(self):
        return sole_geometry(self.model, self.reward_config.gait.sole_clearance_margin)

    def _foot_placement_inputs(
        self, state: BelloState, world_id: jax.Array | int = 0
    ) -> dict[str, jax.Array]:
        foot_phase = jnp.mod(state.gait_phase + jnp.asarray([0.0, 0.5]), 1.0)
        progress = jnp.clip(
            (foot_phase - self.config.gait.stance_fraction)
            / (1.0 - self.config.gait.stance_fraction),
            0.0,
            1.0,
        )
        result = {"swing_progress": progress}
        if not (
            self.reward_config.gait.phase_shaped_clearance
            or any(
                self.reward_config.enabled(name)
                for name in ("stand_foot_foreaft", "stand_foot_width", "foot_crossing")
            )
        ):
            return result
        geometry = self._sole_reward_geometry
        bodies = jnp.asarray(self.model.foot_body_ids)
        quats = state.data.xquat[bodies]
        positions = state.data.xpos[bodies]
        centers = positions + jax.vmap(quat_rotate)(
            quats, jnp.asarray(geometry.centers)
        )
        corners = positions[:, None, :] + jax.vmap(
            lambda q, points: jax.vmap(lambda p: quat_rotate(q, p))(points)
        )(quats, jnp.asarray(geometry.corners))
        heading = yaw_quat(
            self._link_quat(state, self._reward_frame_link("track_lin_vel_xy"))
        )
        delta = quat_rotate_inverse(heading, centers[0] - centers[1])
        separation = -jnp.dot(delta, self._locomotion_right_axis())
        left_axis = quat_rotate(heading, -self._locomotion_right_axis())
        ids = jnp.asarray(self.model.foot_geom_ids)
        clearance = sole_lateral_clearance(
            state.data.geom_xpos[ids],
            state.data.geom_xmat[ids].reshape((-1, 3, 3)),
            jnp.asarray(self.model.mj_model.geom_size)[ids],
            jnp.asarray(self.model.mj_model.geom_type)[ids],
            jnp.asarray(self.model.geom_body_ids)[ids],
            bodies,
            left_axis,
            array_module=jnp,
        )
        result.update(
            foot_foreaft=jnp.dot(
                delta, jnp.asarray(self.model.robot.locomotion_forward_axis)
            ),
            foot_separation=separation,
            nominal_stance_width=jnp.asarray(geometry.nominal_width),
            foot_lateral_clearance=clearance,
            foot_clearance_samples=corners[:, :, 2]
            - self._terrain_height(corners[:, :, :2], world_id),
        )
        return result

    def _foot_sole_positions(
        self, state: BelloState, world_id: jax.Array | int = 0
    ) -> jax.Array:
        ids = jnp.asarray(self.model.foot_geom_ids, dtype=jnp.int32)
        position = state.data.geom_xpos[ids]
        return sole_support_positions(
            geom_position=position,
            geom_rotation=state.data.geom_xmat[ids],
            geom_size=jnp.asarray(self.model.mj_model.geom_size)[ids],
            geom_type=jnp.asarray(self.model.mj_model.geom_type)[ids],
            geom_body_ids=jnp.asarray(self.model.geom_body_ids)[ids],
            foot_body_ids=jnp.asarray(self.model.foot_body_ids),
            terrain_up=self._terrain_surface_normal(position[:, :2], world_id),
            terrain_height=lambda xy: self._terrain_height(xy, world_id),
            array_module=jnp,
        )

    def _terrain_height(
        self, world_xy: jax.Array, world_id: jax.Array | int = 0
    ) -> jax.Array:
        if self.terrain_config.mode == "none":
            return jnp.full(
                jnp.asarray(world_xy).shape[:-1],
                self.terrain_config.pos[2],
                dtype=jnp.float32,
            )
        return terrain_surface_height_jax(
            self.model.mj_model,
            int(self.model.terrain_geom_ids[0]),
            world_xy,
            geom_quat=jnp.asarray(self.model.terrain_geom_quat)[
                jnp.minimum(
                    jnp.asarray(world_id, dtype=jnp.int32),
                    self.model.terrain_geom_quat.shape[0] - 1,
                )
            ],
        )

    def _terrain_surface_normal(
        self, world_xy: jax.Array, world_id: jax.Array | int = 0
    ) -> jax.Array:
        world_xy = jnp.asarray(world_xy, dtype=jnp.float32)
        output_shape = world_xy.shape[:-1] + (3,)
        if self.terrain_config.mode == "none":
            return jnp.broadcast_to(
                jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float32), output_shape
            )
        if self.terrain_config.mode == "plane":
            terrain_quat = jnp.asarray(self.model.terrain_geom_quat)[
                jnp.minimum(
                    jnp.asarray(world_id, dtype=jnp.int32),
                    self.model.terrain_geom_quat.shape[0] - 1,
                )
            ]
            terrain_up = quat_rotate(
                terrain_quat,
                jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float32),
            )
            return jnp.broadcast_to(terrain_up, output_shape)

        return heightfield_surface_normal(
            world_xy,
            lambda xy: self._terrain_height(xy, world_id),
            array_module=jnp,
        )

    def _foot_world_linear_velocity(self, state: BelloState) -> jax.Array:
        return self._body_world_lin_vel(state, self.model.foot_body_ids)

    def _foot_angular_speed(self, state: BelloState) -> jax.Array:
        foot_body_ids = jnp.asarray(self.model.foot_body_ids)
        foot_ang_vel = state.data.cvel[foot_body_ids, :3]
        return jnp.linalg.norm(foot_ang_vel, axis=1)

    def _foot_heading_error(self, state: BelloState) -> jax.Array:
        command_forward = quat_rotate(
            self._link_quat(state, self._reward_frame_link("track_lin_vel_xy")),
            jnp.asarray(self.model.robot.locomotion_forward_axis, dtype=jnp.float32),
        )[:2]
        command_forward /= jnp.maximum(jnp.linalg.norm(command_forward), 1.0e-6)
        errors = []
        for link_name, local_forward in self.model.robot.foot_forward_axes:
            foot_forward = quat_rotate(
                self._link_quat(state, link_name),
                jnp.asarray(local_forward, dtype=jnp.float32),
            )[:2]
            foot_forward /= jnp.maximum(jnp.linalg.norm(foot_forward), 1.0e-6)
            cross_z = (
                command_forward[0] * foot_forward[1]
                - command_forward[1] * foot_forward[0]
            )
            dot = jnp.dot(command_forward, foot_forward)
            errors.append(jnp.arctan2(cross_z, dot))
        return jnp.stack(errors)

    def _stance_width(self, state: BelloState, frame_link_name: str) -> jax.Array:
        if self.model.foot_body_ids.size != 2:
            return jnp.asarray(0.0, dtype=jnp.float32)
        foot_position = state.data.xpos[jnp.asarray(self.model.foot_body_ids)]
        foot_delta = foot_position[0] - foot_position[1]
        frame_yaw = yaw_quat(self._link_quat(state, frame_link_name))
        frame_delta = quat_rotate_inverse(frame_yaw, foot_delta)
        return jnp.abs(jnp.dot(frame_delta, self._locomotion_right_axis()))

    @property
    def _actuator_joint_limits(self) -> jax.Array:
        return jnp.asarray(self.model.mj_model.jnt_range)[self.model.actuator_joint_ids]


def require_index_range(name: str, indices: np.ndarray, upper_bound: int) -> None:
    indices = np.asarray(indices)
    if indices.size == 0:
        return
    if np.any(indices < 0) or np.any(indices >= upper_bound):
        raise ValueError(
            f"{name} contains out-of-range references for upper bound {upper_bound}: "
            f"{indices.tolist()}"
        )


def require_disjoint(
    left_name: str,
    left: np.ndarray,
    right_name: str,
    right: np.ndarray,
) -> None:
    overlap = sorted(set(np.asarray(left).tolist()) & set(np.asarray(right).tolist()))
    if overlap:
        raise ValueError(
            f"{left_name} must not reference held actuator controls in {right_name}: "
            f"{overlap}"
        )
