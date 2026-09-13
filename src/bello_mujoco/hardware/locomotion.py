"""Deploy a locomotion checkpoint through bello-stack's hardware client."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, replace
import math
from pathlib import Path
import time
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from bello_mujoco.training.checkpoint import (
    CheckpointNormalizationState,
    LocomotionCheckpointMetadata,
)
from bello_mujoco.envs.locomotion import BelloLocomotionEnv
from bello_mujoco.envs.gait import (
    advance_gait_phase,
    resolve_gait_command_transition,
)
from bello_mujoco.gamepad import axis_to_range, forward_lateral_to_command_axes
from bello_mujoco.hardware.config import HardwareControlConfig
from bello_mujoco.hardware.policy_log import PolicyStepLogger, policy_phase_schema
from bello_mujoco.hardware.remote import (
    RemoteCommandConfig,
    RemoteCommandInput,
    RemoteCommandSource,
)
from bello_mujoco.hardware.runtime import (
    HOLD_CONFIRMATION,
    HardwareFrame,
    actuator_coordinate_names,
    actuator_impedance_gains,
    armed_control_session,
    import_hardware_binding,
    normalized_quaternion_xyzw,
    projected_gravity_from_xyzw,
    require_async_hardware_binding,
    require_nominal_pose,
    require_state_frame_binding,
    specific_force_from_gravity_removed_acceleration,
    tilt_from_xyzw,
    validate_hardware_control_config,
    validate_hardware_frame,
    validate_hardware_inventory,
    wait_for_operator_confirmation_while_holding,
    wait_for_hardware_frame,
)
from bello_mujoco.hardware.reference import ReferenceUnavailable, RobotReferenceFrame
from bello_mujoco.policy import infer_policy_action
from bello_mujoco.locomotion_runtime import load_locomotion_policy_runtime
from bello_mujoco.robot import robot_definition
from bello_mujoco.sim import hardware_imu_to_policy_rotation
from bello_mujoco.training.ppo import normalize_observation


def _upper_body_follow_supported(asset: str, joint_names: frozenset[str]) -> bool:
    expected = frozenset(
        actuator.joint_name
        for actuator in robot_definition(asset).actuators
        if actuator.mode == "hold"
    )
    return bool(expected) and joint_names == expected


def _transform_hardware_imu_frame(
    frame: HardwareFrame, policy_from_hardware: np.ndarray
) -> HardwareFrame:
    orientation_xyzw = normalized_quaternion_xyzw(frame.orientation_xyzw)
    world_from_hardware = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(
        world_from_hardware,
        orientation_xyzw[[3, 0, 1, 2]],
    )
    world_from_policy = world_from_hardware.reshape(3, 3) @ policy_from_hardware.T
    policy_orientation_wxyz = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(policy_orientation_wxyz, world_from_policy.reshape(-1))
    return replace(
        frame,
        orientation_xyzw=policy_orientation_wxyz[[1, 2, 3, 0]],
        angular_velocity=policy_from_hardware @ frame.angular_velocity,
        linear_acceleration=policy_from_hardware @ frame.linear_acceleration,
    )


@dataclass(frozen=True)
class UpperBodyFollowConfig:
    """Fixed safety envelope for locomotion's pose-driven hold coordinates."""

    max_reference_age_ms: int = 1_000
    max_reference_joint_speed_rps: float = 25.0
    max_reference_joint_limit_violation_radians: float = 0.01
    min_live_joint_limit_margin_radians: float = 1.0e-3
    max_target_speed_rps: float = 3.0 * math.pi
    connection_transition_duration_seconds: float = 1.0


@dataclass(frozen=True)
class HardwareLocomotionConfig(HardwareControlConfig):
    """Hardware and visualizer-remote fail-closed deployment limits."""

    client_name: str = "bello-locomotion-policy"
    remote: RemoteCommandConfig = RemoteCommandConfig()
    joystick_command_scale: float = 0.6
    allow_provisional_hardware: bool = False
    upper_body_tracking_enabled: bool = True
    rtmw_endpoint: str = "127.0.0.1:50055"
    rtmw_tls_root_certificate_path: Path | None = None
    rtmw_tls_server_name_override: str | None = None
    rtmw_connect_timeout_seconds: float = 5.0
    rtmw_rpc_timeout_seconds: float = 2.0
    rtmw_reference_qualification_frames: int = 1
    rtmw_min_joint_confidence: float = 0.0
    rtmw_min_rotation_confidence: float = 0.0
    rtmw_max_required_rotation_residual_degrees: float = 35.0
    rtmw_max_pipeline_latency_ms: float = 250.0
    rtmw_transport_latency_budget_ms: float = 10.0
    rtmw_max_source_interval_seconds: float = 0.70
    rtmw_camera_to_world_quaternion_xyzw: tuple[float, float, float, float] = (
        0.5,
        -0.5,
        0.5,
        -0.5,
    )
    rtmw_align_initial_heading: bool = True
    rtmw_align_initial_floor: bool = True
    smplx_neutral_model_path: Path | None = None
    smplx_human_height_m: float = 1.66
    rtmw_human_height_m: float = 1.75
    rtmw_gmr_task_profile: str = "live_upper_body"
    upper_body_follow: UpperBodyFollowConfig = UpperBodyFollowConfig()


@dataclass(frozen=True)
class HardwarePolicyStep:
    action: np.ndarray
    raw_action: np.ndarray
    physical_target: np.ndarray
    observation: np.ndarray
    policy_observation: np.ndarray
    effective_command: np.ndarray
    standing_transition_pending: bool
    tilt_radians: float
    reference_sequence: int | None
    reference_age_ms: float | None
    upper_body_following: bool


class HardwarePolicyAdapter:
    """Reconstruct deployable observations and physical motor targets."""

    def __init__(
        self,
        env: BelloLocomotionEnv,
        policy: Any,
        metadata: LocomotionCheckpointMetadata,
        normalization: CheckpointNormalizationState,
        upper_body_follow: UpperBodyFollowConfig = UpperBodyFollowConfig(),
    ) -> None:
        validate_hardware_checkpoint(env, metadata)
        self.env = env
        self.metadata = metadata
        self.hardware_imu_frame_name = env.model.robot.default_imu_body_name
        self.hardware_imu_to_policy_rotation = hardware_imu_to_policy_rotation(
            env.model
        )
        self.physical_coordinate_names = actuator_coordinate_names(env)
        self.coordinate_index = {
            name: index for index, name in enumerate(self.physical_coordinate_names)
        }
        if len(self.coordinate_index) != len(self.physical_coordinate_names):
            raise ValueError("physical actuator coordinate names must be unique")
        self.transform = np.asarray(
            env.model.actuator_coordinate_transform, dtype=np.float64
        )
        if self.transform.shape != (
            len(self.physical_coordinate_names),
            len(self.physical_coordinate_names),
        ):
            raise ValueError(
                "hardware deployment requires a square actuator coordinate transform"
            )
        condition = float(np.linalg.cond(self.transform))
        if not math.isfinite(condition) or condition > 1.0e6:
            raise ValueError(
                "actuator coordinate transform is not safely invertible: "
                f"condition={condition}"
            )
        self.inverse_transform = np.linalg.inv(self.transform)
        self.policy_indices = np.asarray(env.model.policy_ctrl_indices, dtype=np.int32)
        self.upper_body_indices = np.asarray(
            env.model.hold_ctrl_indices, dtype=np.int32
        )
        self.upper_body_qpos_indices = np.asarray(
            env.model.actuator_qpos_indices[self.upper_body_indices], dtype=np.int32
        )
        self.upper_body_qvel_indices = np.asarray(
            env.model.actuator_qvel_indices[self.upper_body_indices], dtype=np.int32
        )
        upper_body_index_by_name = {
            env.model.actuator_joint_names[int(actuator_index)]: upper_index
            for upper_index, actuator_index in enumerate(self.upper_body_indices)
        }
        self.upper_body_follow_supported = _upper_body_follow_supported(
            env.model.robot.asset,
            frozenset(upper_body_index_by_name),
        )
        self.default_joint_target = np.asarray(
            env.model.default_joint_target, dtype=np.float64
        )
        self._physical_target_limits = np.asarray(
            env.model.mj_model.actuator_ctrlrange, dtype=np.float64
        )
        if self._physical_target_limits.shape != (
            len(self.physical_coordinate_names),
            2,
        ):
            raise ValueError("hardware actuator target limits have an invalid shape")
        self.upper_body_follow = upper_body_follow
        self.kp, self.kd = actuator_impedance_gains(env)
        self._normalization = normalization.observation
        self._infer_action = eqx.filter_jit(
            lambda observation: (
                normalize_observation(
                    observation,
                    self._normalization,
                    metadata.ppo,
                ),
                infer_policy_action(
                    policy,
                    observation,
                    self._normalization,
                    metadata.ppo,
                    env.action_clip,
                ),
            )
        )
        zero_offset = jnp.zeros(len(self.physical_coordinate_names), dtype=jnp.float32)
        self._action_ctrl = jax.jit(
            lambda action: env.action_to_ctrl(action, zero_offset)
        )
        self._obs_stack: np.ndarray | None = None
        self._last_action = np.zeros(env.action_size, dtype=np.float32)
        self._effective_command = np.zeros(3, dtype=np.float32)
        self._gait_phase = 0.0
        self._standing_transition_pending = 0
        self._next_start_left = True
        self._advance_gait_phase_before_step = False

        def resolve_transition(
            current_command,
            requested_command,
            gait_phase,
            standing_transition_pending,
            advance_phase,
            next_start_left,
        ):
            next_phase = jnp.where(
                advance_phase,
                advance_gait_phase(
                    gait_phase,
                    current_command,
                    env.command_config,
                    env.config.gait,
                    env.control_timestep,
                    standing_transition_pending,
                ),
                gait_phase,
            )
            return resolve_gait_command_transition(
                current_command,
                requested_command,
                gait_phase,
                next_phase,
                standing_transition_pending,
                env.command_config,
                env.config.gait,
                next_start_left,
                -env._locomotion_right_axis()[:2],
                jnp.where(advance_phase, env.control_timestep, 0.0),
            )

        self._resolve_gait_transition = jax.jit(resolve_transition)
        self._standing_reference: RobotReferenceFrame | None = None
        self._upper_body_target: np.ndarray | None = None
        self._upper_body_reference_live = False
        self._upper_body_transition_start: np.ndarray | None = None
        self._upper_body_transition_elapsed_seconds = 0.0

    @property
    def control_timestep(self) -> float:
        return self.env.control_timestep

    @property
    def nominal_physical_target(self) -> np.ndarray:
        target = np.asarray(
            jax.device_get(
                self._action_ctrl(jnp.zeros(self.env.action_size, dtype=jnp.float32))
            ),
            dtype=np.float64,
        )
        if self._standing_reference is None:
            return target
        logical_target = self.inverse_transform @ target
        logical_target[self.upper_body_indices] = self._reference_upper_body_target(
            self._standing_reference,
            live=False,
        )
        return self._bounded_physical_target(self.transform @ logical_target)

    def set_upper_body_standing_reference(self, reference: RobotReferenceFrame) -> None:
        """Bind a full fallback frame whose arm joints use the robot nominal pose."""

        qpos = np.asarray(reference.qpos, dtype=np.float64).copy()
        qvel = np.asarray(reference.qvel, dtype=np.float64).copy()
        if qpos.shape == (self.env.model.mj_model.nq,):
            qpos[self.upper_body_qpos_indices] = np.asarray(
                self.env.model.default_qpos, dtype=np.float64
            )[self.upper_body_qpos_indices]
        if qvel.shape == (self.env.model.mj_model.nv,):
            qvel[self.upper_body_qvel_indices] = 0.0
        fallback = replace(reference, qpos=qpos, qvel=qvel)
        self.validate_upper_body_reference(fallback, live=False)
        self._standing_reference = fallback

    def reset(self) -> None:
        self._obs_stack = None
        self._last_action = np.zeros(self.env.action_size, dtype=np.float32)
        self._effective_command = np.zeros(3, dtype=np.float32)
        self._gait_phase = 0.0
        self._standing_transition_pending = 0
        self._next_start_left = True
        self._advance_gait_phase_before_step = False
        self._upper_body_target = None
        self._upper_body_reference_live = False
        self._upper_body_transition_start = None
        self._upper_body_transition_elapsed_seconds = 0.0

    def precompile(self, reference: RobotReferenceFrame | None = None) -> None:
        nominal = self.nominal_physical_target
        frame = HardwareFrame(
            positions=nominal,
            velocities=np.zeros_like(nominal),
            orientation_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
            angular_velocity=np.zeros(3),
            linear_acceleration=np.zeros(3),
            max_motor_age_ms=0,
            imu_age_ms=0,
            max_group_age_ms=0,
            sample_span_ms=0.0,
            state_frame_round_trip_ms=0.0,
            group_states=(),
            group_causes=(),
            group_details=(),
        )
        self.step(frame, np.zeros(3, dtype=np.float32), reference)
        self.reset()

    def policy_imu_frame(self, frame: HardwareFrame) -> HardwareFrame:
        return _transform_hardware_imu_frame(
            frame, self.hardware_imu_to_policy_rotation
        )

    def observation(self, frame: HardwareFrame, command: np.ndarray) -> np.ndarray:
        return self._policy_frame_observation(self.policy_imu_frame(frame), command)

    def _policy_frame_observation(
        self,
        frame: HardwareFrame,
        command: np.ndarray,
        *,
        gait_active: bool | None = None,
    ) -> np.ndarray:
        logical_position = self.inverse_transform @ frame.positions
        logical_velocity = self.inverse_transform @ frame.velocities
        projected_gravity = projected_gravity_from_xyzw(frame.orientation_xyzw)
        specific_force = specific_force_from_gravity_removed_acceleration(
            frame.linear_acceleration, projected_gravity
        )
        scale = self.env.config.observation.scale
        if gait_active is None:
            gait_active = self._gait_command_is_moving(command)
        gait_angle = 2.0 * np.pi * self._gait_phase
        gait_phase = (
            np.asarray((np.sin(gait_angle), np.cos(gait_angle), 1.0), dtype=np.float32)
            if gait_active
            else np.zeros(3, dtype=np.float32)
        )
        components = {
            "imu_linear_acceleration": specific_force * scale.base_lin_acc,
            "imu_angular_velocity": frame.angular_velocity * scale.base_ang_vel,
            "projected_gravity": projected_gravity * scale.projected_gravity,
            "command": command * scale.command,
            "gait_phase": gait_phase * scale.gait_phase,
            "joint_position": (
                logical_position[self.policy_indices]
                - self.default_joint_target[self.policy_indices]
            )
            * scale.joint_pos,
            "joint_velocity": logical_velocity[self.policy_indices] * scale.joint_vel,
            "last_action": self._last_action * scale.last_action,
        }
        parts = []
        for name, size in self.env.observation_layout.components:
            try:
                value = np.asarray(components[name], dtype=np.float32)
            except KeyError as error:
                raise ValueError(
                    f"hardware runner cannot construct observation component {name!r}"
                ) from error
            if value.shape != (size,):
                raise ValueError(
                    f"hardware observation component {name!r} has shape {value.shape}, "
                    f"expected {(size,)}"
                )
            parts.append(value)
        observation = np.concatenate(parts)
        if not np.all(np.isfinite(observation)):
            raise RuntimeError("hardware observation contains non-finite values")
        return observation

    def step(
        self,
        frame: HardwareFrame,
        command: np.ndarray,
        reference: RobotReferenceFrame | None = None,
    ) -> HardwarePolicyStep:
        effective_command = self._resolve_effective_command(command)
        policy_frame = self.policy_imu_frame(frame)
        observation = self._policy_frame_observation(
            policy_frame,
            effective_command,
            gait_active=(
                self._gait_command_is_moving(effective_command)
                or self._standing_transition_pending != 0
            ),
        )
        stack_size = self.metadata.ppo.policy.obs_stack_size
        if self._obs_stack is None:
            self._obs_stack = np.repeat(observation[None, :], stack_size, axis=0)
        else:
            self._obs_stack = np.concatenate(
                (self._obs_stack[1:], observation[None, :]), axis=0
            )
        stacked = jnp.asarray(self._obs_stack.reshape(-1), dtype=jnp.float32)
        policy_observation, inference = self._infer_action(stacked)
        policy_observation = np.asarray(
            jax.device_get(policy_observation), dtype=np.float32
        )
        raw_action = np.asarray(
            jax.device_get(inference.requested_action), dtype=np.float32
        )
        policy_action = np.asarray(
            jax.device_get(inference.applied_action), dtype=np.float32
        )
        if (
            raw_action.shape != (self.env.action_size,)
            or not np.all(np.isfinite(raw_action))
            or policy_action.shape != (self.env.action_size,)
            or not np.all(np.isfinite(policy_action))
        ):
            raise RuntimeError(
                "policy produced an invalid hardware action with shape "
                f"{policy_action.shape}"
            )
        action = policy_action
        physical_target = np.asarray(
            jax.device_get(self._action_ctrl(jnp.asarray(action))), dtype=np.float64
        )
        try:
            physical_target = self._apply_upper_body_reference(
                physical_target,
                frame,
                reference,
            )
        except ReferenceUnavailable:
            reference = None
            physical_target = self._apply_upper_body_reference(
                physical_target,
                frame,
                reference,
            )
        if not np.all(np.isfinite(physical_target)):
            raise RuntimeError("policy produced non-finite physical motor targets")
        self._last_action = action
        self._advance_gait_phase_before_step = True
        return HardwarePolicyStep(
            action=action,
            raw_action=raw_action,
            physical_target=physical_target,
            observation=observation,
            policy_observation=policy_observation,
            effective_command=effective_command,
            standing_transition_pending=bool(self._standing_transition_pending),
            tilt_radians=tilt_from_xyzw(policy_frame.orientation_xyzw),
            reference_sequence=(None if reference is None else reference.sequence),
            reference_age_ms=(None if reference is None else reference.age_ms()),
            upper_body_following=reference is not None,
        )

    def _resolve_effective_command(
        self,
        requested_command: np.ndarray,
    ) -> np.ndarray:
        command, phase, pending = self._resolve_gait_transition(
            jnp.asarray(self._effective_command, dtype=jnp.float32),
            jnp.asarray(requested_command, dtype=jnp.float32),
            jnp.asarray(self._gait_phase, dtype=jnp.float32),
            jnp.asarray(self._standing_transition_pending),
            jnp.asarray(self._advance_gait_phase_before_step),
            jnp.asarray(self._next_start_left),
        )
        if not (
            self._gait_command_is_moving(self._effective_command)
            or self._standing_transition_pending != 0
        ) and self._gait_command_is_moving(np.asarray(command)):
            self._next_start_left = not self._next_start_left
        self._effective_command = np.asarray(command, dtype=np.float32)
        self._gait_phase = float(phase)
        self._standing_transition_pending = int(pending)
        return self._effective_command.copy()

    def validate_upper_body_reference(
        self,
        reference: RobotReferenceFrame,
        *,
        live: bool = True,
    ) -> None:
        if not self.upper_body_follow_supported:
            raise ReferenceUnavailable(
                "upper-body pose following is not supported for robot asset "
                f"{self.env.model.robot.asset!r}"
            )
        qpos = np.asarray(reference.qpos, dtype=np.float64)
        qvel = np.asarray(reference.qvel, dtype=np.float64)
        model = self.env.model.mj_model
        if qpos.shape != (model.nq,) or not np.all(np.isfinite(qpos)):
            raise ReferenceUnavailable(
                f"upper-body reference qpos must contain {model.nq} finite values"
            )
        if qvel.shape != (model.nv,) or not np.all(np.isfinite(qvel)):
            raise ReferenceUnavailable(
                f"upper-body reference qvel must contain {model.nv} finite values"
            )
        if reference.sequence < 0:
            raise ReferenceUnavailable(
                "upper-body reference sequence must be nonnegative"
            )
        age = reference.age_ms()
        if live:
            if age is None:
                raise ReferenceUnavailable(
                    "live upper-body reference has no source age"
                )
            if age > self.upper_body_follow.max_reference_age_ms:
                raise ReferenceUnavailable(
                    f"upper-body reference is stale: {age:.1f} ms exceeds "
                    f"{self.upper_body_follow.max_reference_age_ms} ms"
                )
        speed = float(np.max(np.abs(qvel[self.upper_body_qvel_indices])))
        if speed > self.upper_body_follow.max_reference_joint_speed_rps:
            raise ReferenceUnavailable(
                f"upper-body reference joint speed is unsafe: {speed:.3f} rad/s"
            )
        joint_ids = np.asarray(self.env.model.actuator_joint_ids)[
            self.upper_body_indices
        ]
        if not np.all(model.jnt_limited[joint_ids]):
            raise RuntimeError("upper-body reference joints require hard limits")
        ranges = np.asarray(model.jnt_range, dtype=np.float64)[joint_ids]
        positions = qpos[self.upper_body_qpos_indices]
        violation = np.maximum(ranges[:, 0] - positions, positions - ranges[:, 1])
        index = int(np.argmax(violation))
        if (
            violation[index]
            > self.upper_body_follow.max_reference_joint_limit_violation_radians
        ):
            logical_index = int(self.upper_body_indices[index])
            name = self.env.model.actuator_joint_names[logical_index]
            raise ReferenceUnavailable(
                f"upper-body reference joint {name!r} exceeds its limit by "
                f"{violation[index]:.3f} rad"
            )
        if live:
            margins = np.minimum(
                positions - ranges[:, 0],
                ranges[:, 1] - positions,
            )
            index = int(np.argmin(margins))
            minimum = self.upper_body_follow.min_live_joint_limit_margin_radians
            if margins[index] <= minimum:
                logical_index = int(self.upper_body_indices[index])
                name = self.env.model.actuator_joint_names[logical_index]
                raise ReferenceUnavailable(
                    f"live upper-body reference joint {name!r} has only "
                    f"{margins[index]:.6f} rad of joint-limit margin; requires "
                    f"more than {minimum:.6f} rad"
                )

    def _reference_upper_body_target(
        self,
        reference: RobotReferenceFrame,
        *,
        live: bool,
    ) -> np.ndarray:
        self.validate_upper_body_reference(reference, live=live)
        desired = np.asarray(reference.qpos, dtype=np.float64)[
            self.upper_body_qpos_indices
        ].copy()
        low = self.env.model.joint_target_low[self.upper_body_indices]
        high = self.env.model.joint_target_high[self.upper_body_indices]
        return np.clip(desired, low, high)

    def _apply_upper_body_reference(
        self,
        physical_target: np.ndarray,
        frame: HardwareFrame,
        reference: RobotReferenceFrame | None,
    ) -> np.ndarray:
        fallback = self._standing_reference
        desired_reference = reference if reference is not None else fallback
        if desired_reference is None:
            desired = self.default_joint_target[self.upper_body_indices]
        else:
            desired = self._reference_upper_body_target(
                desired_reference,
                live=reference is not None,
            )
        if self._upper_body_target is None:
            measured = self.inverse_transform @ np.asarray(frame.positions)
            self._upper_body_target = measured[self.upper_body_indices].copy()
        reference_live = reference is not None
        if reference_live != self._upper_body_reference_live:
            self._upper_body_reference_live = reference_live
            self._upper_body_transition_start = self._upper_body_target.copy()
            self._upper_body_transition_elapsed_seconds = 0.0
        if self._upper_body_transition_start is not None:
            duration = self.upper_body_follow.connection_transition_duration_seconds
            self._upper_body_transition_elapsed_seconds = min(
                duration,
                self._upper_body_transition_elapsed_seconds + self.control_timestep,
            )
            blend = self._upper_body_transition_elapsed_seconds / duration
            desired = self._upper_body_transition_start + blend * (
                desired - self._upper_body_transition_start
            )
            if self._upper_body_transition_elapsed_seconds >= duration:
                self._upper_body_transition_start = None
        maximum_step = (
            self.upper_body_follow.max_target_speed_rps * self.control_timestep
        )
        self._upper_body_target += np.clip(
            desired - self._upper_body_target,
            -maximum_step,
            maximum_step,
        )
        logical_target = self.inverse_transform @ physical_target
        logical_target[self.upper_body_indices] = self._upper_body_target
        return self._bounded_physical_target(self.transform @ logical_target)

    def _bounded_physical_target(self, target: np.ndarray) -> np.ndarray:
        return np.clip(
            np.asarray(target, dtype=np.float64),
            self._physical_target_limits[:, 0],
            self._physical_target_limits[:, 1],
        )

    def _gait_command_is_moving(self, command: np.ndarray) -> bool:
        if self.env.config.gait.command_slew_enabled:
            return bool(np.any(command != 0))
        return self._command_is_moving(command)

    def _command_is_moving(self, command: np.ndarray) -> bool:
        command = np.asarray(command, dtype=np.float32)
        return bool(
            np.any(np.abs(command[:2]) >= self.env.command_config.linear_deadband)
            or abs(float(command[2])) >= self.env.command_config.yaw_deadband
        )


def load_hardware_policy(
    checkpoint_path: Path,
    *,
    upper_body_follow: UpperBodyFollowConfig = UpperBodyFollowConfig(),
) -> tuple[HardwarePolicyAdapter, LocomotionCheckpointMetadata]:
    runtime = load_locomotion_policy_runtime(
        checkpoint_path,
        use_viewer_model=False,
    )
    return HardwarePolicyAdapter(
        runtime.env,
        runtime.policy,
        runtime.metadata,
        runtime.normalization,
        upper_body_follow,
    ), runtime.metadata


def validate_hardware_checkpoint(
    env: BelloLocomotionEnv, metadata: LocomotionCheckpointMetadata
) -> None:
    if metadata.task != "locomotion":
        raise ValueError("hardware locomotion requires a locomotion checkpoint")
    if env.config.observation.privileged:
        raise ValueError("hardware actor observation must not be privileged")

    definition = robot_definition(metadata.asset.asset)
    if env.model.robot.asset != definition.asset:
        raise ValueError(
            "checkpoint asset does not match the loaded hardware model: "
            f"checkpoint={definition.asset!r}, model={env.model.robot.asset!r}"
        )
    configured_modes = {
        actuator.joint_name: actuator.mode for actuator in definition.actuators
    }
    model_joint_names = tuple(env.model.actuator_joint_names)
    configured_joint_names = frozenset(configured_modes)
    model_joint_name_set = frozenset(model_joint_names)
    if (
        len(model_joint_names) != len(model_joint_name_set)
        or model_joint_name_set != configured_joint_names
    ):
        raise ValueError(
            "hardware locomotion actuator joints must match the robot-configured "
            f"set exactly; expected {configured_joint_names}, got "
            f"{model_joint_name_set}"
        )

    expected_policy_names = tuple(
        name for name in model_joint_names if configured_modes[name] == "policy"
    )
    expected_hold_names = tuple(
        name for name in model_joint_names if configured_modes[name] == "hold"
    )
    policy_names = tuple(
        model_joint_names[int(index)] for index in env.model.policy_ctrl_indices
    )
    hold_names = tuple(
        model_joint_names[int(index)] for index in env.model.hold_ctrl_indices
    )
    if policy_names != expected_policy_names or env.action_size != len(
        expected_policy_names
    ):
        raise ValueError(
            "hardware locomotion policy joints must match the robot-configured "
            f"name/order exactly; expected {expected_policy_names}, got {policy_names}"
        )
    if hold_names != expected_hold_names:
        raise ValueError(
            "hardware locomotion hold joints must match the robot-configured "
            f"name/order exactly; expected {expected_hold_names}, got {hold_names}"
        )
    if metadata.ppo.policy.action_distribution != "tanh_gaussian":
        raise ValueError("hardware locomotion requires a bounded tanh-Gaussian policy")


def validate_hardware_config(
    config: HardwareLocomotionConfig,
    metadata: LocomotionCheckpointMetadata | None = None,
) -> None:
    validate_hardware_control_config(config)
    if type(config.allow_provisional_hardware) is not bool:
        raise ValueError("allow_provisional_hardware must be a boolean")
    if not config.remote.endpoint.startswith("http://"):
        raise ValueError("remote.endpoint must use plaintext http:// in protocol v1")
    if config.remote.staleness_ms <= 0:
        raise ValueError("remote.staleness_ms must be positive")
    if isinstance(config.joystick_command_scale, bool) or (
        not math.isfinite(config.joystick_command_scale)
        or not 0.0 < config.joystick_command_scale <= 1.0
    ):
        raise ValueError("joystick_command_scale must be finite and in (0, 1]")
    if metadata is not None:
        definition = robot_definition(metadata.asset.asset)
        hold_names = frozenset(
            actuator.joint_name
            for actuator in definition.actuators
            if actuator.mode == "hold"
        )
        follow_supported = _upper_body_follow_supported(
            definition.asset,
            hold_names,
        )
        if definition.physical_hardware_ready and config.allow_provisional_hardware:
            raise ValueError(
                "allow_provisional_hardware is only valid for a provisional robot"
            )
        if (
            config.arm
            and not definition.physical_hardware_ready
            and not config.allow_provisional_hardware
        ):
            raise ValueError(
                f"robot asset {definition.asset!r} is not approved for physical "
                "deployment; emulator-only armed runs require the explicit "
                "allow_provisional_hardware=true acknowledgement"
            )
        if config.upper_body_tracking_enabled and not follow_supported:
            raise ValueError(
                "upper-body pose following is not supported for robot asset "
                f"{definition.asset!r}; set upper_body_tracking_enabled=false"
            )
    if config.upper_body_tracking_enabled:
        from bello_mujoco.hardware.rtmw import RtmwGrpcConfig

        RtmwGrpcConfig.from_hardware(config)
        if (
            not math.isfinite(config.smplx_human_height_m)
            or config.smplx_human_height_m <= 0.0
        ):
            raise ValueError("smplx_human_height_m must be positive and finite")
        if config.rtmw_gmr_task_profile != "live_upper_body":
            raise ValueError(
                "rtmw_gmr_task_profile must be 'live_upper_body' for safe RTMW "
                "locomotion tracking"
            )
        if config.rtmw_reference_qualification_frames < 1:
            raise ValueError("rtmw_reference_qualification_frames must be positive")
    upper = config.upper_body_follow
    positive = {
        "max_reference_joint_speed_rps": upper.max_reference_joint_speed_rps,
        "max_target_speed_rps": upper.max_target_speed_rps,
        "connection_transition_duration_seconds": (
            upper.connection_transition_duration_seconds
        ),
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"upper_body_follow.{name} must be positive and finite")
    if upper.max_reference_age_ms <= 0:
        raise ValueError("upper_body_follow.max_reference_age_ms must be positive")
    if (
        not math.isfinite(upper.max_reference_joint_limit_violation_radians)
        or upper.max_reference_joint_limit_violation_radians < 0.0
    ):
        raise ValueError(
            "upper_body_follow.max_reference_joint_limit_violation_radians must be "
            "finite and nonnegative"
        )
    if (
        not math.isfinite(upper.min_live_joint_limit_margin_radians)
        or upper.min_live_joint_limit_margin_radians <= 0.0
    ):
        raise ValueError(
            "upper_body_follow.min_live_joint_limit_margin_radians must be "
            "positive and finite"
        )
    maximum_rtmw_continuity_age_ms = (
        config.rtmw_max_source_interval_seconds * 1_000.0
        + config.rtmw_max_pipeline_latency_ms
        + config.rtmw_transport_latency_budget_ms
    )
    if (
        config.upper_body_tracking_enabled
        and upper.max_reference_age_ms <= maximum_rtmw_continuity_age_ms
    ):
        raise ValueError(
            "upper_body_follow.max_reference_age_ms must exceed the RTMW source "
            "interval plus pipeline and transport latency limits"
        )
    if metadata is not None:
        command_ranges(metadata)
        command_minimum_moving_speeds(metadata)
        joystick_command_ranges(metadata, config.joystick_command_scale)


def validate_command(
    command: np.ndarray,
    metadata: LocomotionCheckpointMetadata,
) -> None:
    command_config = metadata.command
    if not command_config.enabled:
        if np.any(command != 0.0):
            raise ValueError(
                "checkpoint disables commands; hardware command must be zero"
            )
        return
    if command_config.standing_probability > 0.0 and np.all(command == 0.0):
        return
    bounds = np.asarray(command_config.range_array(), dtype=np.float64)
    if np.any(command < bounds[:, 0]) or np.any(command > bounds[:, 1]):
        raise ValueError(
            f"hardware command {command.tolist()} exceeds checkpoint bounds "
            f"{bounds.tolist()}"
        )


def command_ranges(
    metadata: LocomotionCheckpointMetadata,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    configured_bounds = (
        metadata.command.lin_vel_x,
        metadata.command.lin_vel_y,
        metadata.command.ang_vel_yaw,
    )
    bounds = np.asarray(configured_bounds, dtype=np.float64)
    if bounds.shape != (3, 2) or not np.all(np.isfinite(bounds)):
        raise ValueError("locomotion checkpoint command bounds are invalid")
    return (
        (float(bounds[0, 0]), float(bounds[0, 1])),
        (float(bounds[1, 0]), float(bounds[1, 1])),
        (float(bounds[2, 0]), float(bounds[2, 1])),
    )


def command_minimum_moving_speeds(
    metadata: LocomotionCheckpointMetadata,
) -> tuple[float, float, float]:
    linear = float(metadata.command.minimum_moving_linear_speed)
    yaw = float(metadata.command.minimum_moving_yaw_speed)
    return (linear, linear, yaw)


def joystick_command_ranges(
    metadata: LocomotionCheckpointMetadata,
    scale: float,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    if isinstance(scale, bool) or not math.isfinite(scale) or not 0.0 < scale <= 1.0:
        raise ValueError("joystick command scale must be finite and in (0, 1]")
    ranges = command_ranges(metadata)
    scaled = tuple(
        (
            low if low >= 0.0 else low * scale,
            high if high <= 0.0 else high * scale,
        )
        for low, high in ranges
    )
    minimums = command_minimum_moving_speeds(metadata)
    names = ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")
    for name, bounds, scaled_bounds, minimum in zip(
        names,
        ranges,
        scaled,
        minimums,
        strict=True,
    ):
        low, high = bounds
        scaled_low, scaled_high = scaled_bounds
        if scaled_low > scaled_high:
            raise ValueError(
                f"joystick command scale {scale} leaves no {name} range inside "
                f"the trained bounds {bounds}"
            )
        if low <= -minimum and scaled_low > -minimum:
            raise ValueError(
                f"joystick command scale {scale} removes the trained negative "
                f"{name} range below its minimum moving speed {minimum}"
            )
        if high >= minimum and scaled_high < minimum:
            raise ValueError(
                f"joystick command scale {scale} removes the trained positive "
                f"{name} range below its minimum moving speed {minimum}"
            )
    return scaled


def hardware_command(
    metadata: LocomotionCheckpointMetadata,
    remote_input: RemoteCommandInput,
    *,
    joystick_command_scale: float,
) -> np.ndarray:
    remote_axes = np.asarray(remote_input.normalized_command, dtype=np.float64)
    if (
        remote_axes.shape != (3,)
        or not np.all(np.isfinite(remote_axes))
        or np.any(remote_axes < -1.0)
        or np.any(remote_axes > 1.0)
    ):
        raise RuntimeError("remote service produced an invalid normalized command")
    forward, lateral, yaw = remote_axes
    normalized = np.asarray(
        forward_lateral_to_command_axes(
            forward,
            lateral,
            yaw,
            robot_definition(metadata.asset.asset).locomotion_forward_axis,
        ),
        dtype=np.float64,
    )
    command = np.asarray(
        [
            axis_to_range(value, bounds, minimum)
            for value, bounds, minimum in zip(
                normalized,
                joystick_command_ranges(metadata, joystick_command_scale),
                command_minimum_moving_speeds(metadata),
                strict=True,
            )
        ],
        dtype=np.float32,
    )
    validate_command(command, metadata)
    return command


def report_remote_connection(remote_input: RemoteCommandInput) -> None:
    if remote_input.service_connection_changed:
        state = "connected" if remote_input.service_connected else "disconnected"
        print(
            f"Visualizer remote service {state}; requested command is zero while "
            "disconnected."
        )
    if remote_input.controller_connection_changed:
        state = "connected" if remote_input.controller_connected else "disconnected"
        suffix = (
            f": {remote_input.controller_name}"
            if remote_input.controller_name is not None
            else ""
        )
        print(
            f"Remote controller {state}{suffix}; requested command is zero while "
            "disconnected."
        )
    if (
        remote_input.controller_connection_changed
        and remote_input.controller_connected
        and not remote_input.intent_armed
    ):
        print(
            "Remote controller is waiting for neutral sticks; requested command is "
            "zero."
        )


def run_hardware_locomotion(
    checkpoint_path: Path,
    config: HardwareLocomotionConfig = HardwareLocomotionConfig(),
    *,
    hardware_binding: Any | None = None,
    remote_factory: Any = RemoteCommandSource,
    reference_source_factory: Any | None = None,
    operator_input: Callable[[str], str] | None = None,
) -> None:
    validate_hardware_config(config)
    adapter, metadata = load_hardware_policy(
        checkpoint_path,
        upper_body_follow=config.upper_body_follow,
    )
    validate_hardware_config(config, metadata)
    binding = hardware_binding or import_hardware_binding()
    with ExitStack() as resources:
        remote = remote_factory(config.remote)
        resources.callback(remote.close)
        remote_input = remote.wait_until_ready(config.connection_timeout_seconds)
        report_remote_connection(remote_input)
        if remote_input.abort_active:
            print("Remote controller requested exit before preflight.")
            return
        command = hardware_command(
            metadata,
            remote_input,
            joystick_command_scale=config.joystick_command_scale,
        )
        references = None
        live_reference = None
        if config.upper_body_tracking_enabled:
            if reference_source_factory is None:
                from bello_mujoco.hardware.rtmw import (
                    rtmw_locomotion_reference_source,
                )

                reference_source_factory = rtmw_locomotion_reference_source
            references = resources.enter_context(
                reference_source_factory(adapter, config)
            )
            standing = references.standing()
            adapter.set_upper_body_standing_reference(standing)
        adapter.precompile(live_reference)
        require_state_frame_binding(binding)
        if config.arm:
            require_async_hardware_binding(binding)

        robot = resources.enter_context(
            binding.Robot.connect(
                config.endpoint, timeout=config.connection_timeout_seconds
            )
        )
        groups = validate_hardware_inventory(robot, adapter, config, binding)
        state_reader = robot.states(
            groups=config.groups,
            imus=(config.imu_name,),
        )
        frame = wait_for_hardware_frame(
            state_reader,
            adapter.physical_coordinate_names,
            config.groups,
            config.imu_name,
            config.telemetry_timeout_seconds,
        )
        validate_hardware_frame(frame, config, expected_group_state="IDLE")
        initial_error = float(
            np.max(np.abs(frame.positions - adapter.nominal_physical_target))
        )
        preview = adapter.step(frame, command, live_reference)
        adapter.reset()
        print(
            {
                "hardware_preflight": "passed",
                "endpoint": config.endpoint,
                "groups": config.groups,
                "imu": config.imu_name,
                "coordinates": len(adapter.physical_coordinate_names),
                "policy_joints": adapter.env.action_size,
                "control_rate_hz": 1.0 / adapter.control_timestep,
                "command_source": "visualizer_remote",
                "remote_service_connected": remote_input.service_connected,
                "remote_controller_connected": remote_input.controller_connected,
                "remote_intent_armed": remote_input.intent_armed,
                "joystick_command_scale": config.joystick_command_scale,
                "joystick_command_ranges": joystick_command_ranges(
                    metadata, config.joystick_command_scale
                ),
                "command": tuple(float(value) for value in preview.effective_command),
                "upper_body_pose_source": (
                    "rtmw_smplx52_gmr"
                    if config.upper_body_tracking_enabled
                    else "checkpoint_nominal"
                ),
                "reference_sequence": preview.reference_sequence,
                "reference_age_ms": preview.reference_age_ms,
                "initial_pose_error_max": initial_error,
                "preview_action_abs_max": float(np.max(np.abs(preview.action))),
                "preview_tilt_radians": preview.tilt_radians,
                "group_age_ms": frame.max_group_age_ms,
                "state_sample_span_ms": frame.sample_span_ms,
                "state_frame_round_trip_ms": frame.state_frame_round_trip_ms,
                "armed": config.arm,
            }
        )
        if not config.arm:
            print(
                "Read-only preflight complete; no lease was acquired and no motor command was sent."
            )
            return

        def continue_hold_confirmation() -> bool:
            current = remote.poll()
            report_remote_connection(current)
            if current.abort_active:
                print("Remote controller requested exit before control acquisition.")
                return False
            time.sleep(adapter.control_timestep)
            return True

        confirmed = wait_for_operator_confirmation_while_holding(
            HOLD_CONFIRMATION,
            "Support the robot, then type HOLD to acquire control and begin the "
            "nominal-pose ramp: ",
            continue_hold_confirmation,
            operator_input,
        )
        if not confirmed:
            return
        remote_input = remote.poll()
        report_remote_connection(remote_input)
        if remote_input.abort_active:
            print("Remote controller requested exit before control acquisition.")
            return
        frame = wait_for_hardware_frame(
            state_reader,
            adapter.physical_coordinate_names,
            config.groups,
            config.imu_name,
            config.telemetry_timeout_seconds,
        )
        validate_hardware_frame(frame, config, expected_group_state="IDLE")
        policy_log = (
            resources.enter_context(
                PolicyStepLogger(
                    config.policy_log_path,
                    task="locomotion",
                    checkpoints={"locomotion": checkpoint_path},
                    phases={"locomotion": policy_phase_schema(adapter)},
                )
            )
            if config.policy_log_path is not None
            else None
        )
        control_physical_robot(
            binding,
            groups,
            state_reader,
            frame,
            adapter,
            config,
            remote,
            references,
            policy_log=policy_log,
            operator_input=operator_input,
        )


def control_physical_robot(
    binding: Any,
    groups: dict[str, Any],
    state_reader: Any,
    initial_frame: HardwareFrame,
    adapter: HardwarePolicyAdapter,
    config: HardwareLocomotionConfig,
    remote: RemoteCommandSource,
    references: Any | None,
    *,
    policy_log: PolicyStepLogger | None = None,
    operator_input: Callable[[str], str] | None = None,
) -> None:
    with armed_control_session(
        binding,
        groups,
        state_reader,
        initial_frame,
        adapter,
        config,
        thread_name="bello-group-control",
        operator_input=operator_input,
    ) as control:

        def continue_ramp() -> bool:
            remote_input = remote.poll()
            report_remote_connection(remote_input)
            if remote_input.abort_active:
                print("Remote controller requested exit during the nominal-pose ramp.")
                return False
            return True

        frame = control.ramp_to_nominal(
            initial_frame.positions,
            before_step=continue_ramp,
        )
        if frame is None:
            control.drain()
            return
        require_nominal_pose(frame, adapter)

        def continue_hold() -> bool:
            remote_input = remote.poll()
            report_remote_connection(remote_input)
            if remote_input.abort_active:
                print("Remote controller requested exit before policy start.")
                return False
            return True

        started = control.wait_for_run(
            "Nominal-pose ramp complete. Release external support, let the robot "
            "settle, then type RUN to start policy control: ",
            operator_input=operator_input,
            before_step=continue_hold,
            validate_hold=lambda held_frame: require_nominal_pose(held_frame, adapter),
        )
        if not started:
            control.drain()
            return
        require_nominal_pose(control.read_frame(), adapter)

        adapter.reset()
        start_time = time.monotonic()
        next_status = start_time if config.status_interval_seconds is not None else None
        control.reset_deadline()
        step = 0
        pose_following: bool | None = None
        reference_qualifier = _UpperBodyReferenceQualifier(
            config.rtmw_reference_qualification_frames
        )
        while (
            config.max_duration_seconds is None
            or time.monotonic() - start_time < config.max_duration_seconds
        ):
            loop_started = time.monotonic()
            remote_input = remote.poll()
            report_remote_connection(remote_input)
            if remote_input.abort_active:
                print("Remote controller requested exit; entering damped shutdown.")
                break
            command = hardware_command(
                adapter.metadata,
                remote_input,
                joystick_command_scale=config.joystick_command_scale,
            )
            input_finished = time.monotonic()
            frame = control.read_frame()
            state_finished = time.monotonic()
            reference = None
            if references is not None:
                reference_error: ReferenceUnavailable | None = None
                try:
                    candidate = references.latest()
                    adapter.validate_upper_body_reference(candidate)
                    reference = reference_qualifier.accept(candidate)
                    if reference is None:
                        reference_error = ReferenceUnavailable(
                            "qualifying live upper-body pose: "
                            f"{reference_qualifier.qualified_frames}/"
                            f"{reference_qualifier.required_frames} distinct frames"
                        )
                except ReferenceUnavailable as error:
                    reference_qualifier.reset()
                    reference_error = error
                following = reference is not None
                if following != pose_following:
                    if following:
                        print("Live upper-body pose following is active.")
                    else:
                        print(
                            "Live upper-body pose is unavailable; returning the "
                            f"arms to their nominal pose: {reference_error}"
                        )
                    pose_following = following
            policy_step = adapter.step(frame, command, reference)
            policy_finished = time.monotonic()
            control.send_target(policy_step.physical_target)
            command_finished = time.monotonic()
            step += 1
            now = time.monotonic()
            if policy_log is not None:
                policy_log.write_step(
                    phase="locomotion",
                    step=step,
                    elapsed_seconds=now - start_time,
                    observation=policy_step.observation,
                    policy_observation=policy_step.policy_observation,
                    raw_action=policy_step.raw_action,
                    action=policy_step.action,
                    physical_target=policy_step.physical_target,
                    command=policy_step.effective_command,
                )
            log_finished = time.monotonic()
            if next_status is not None and now >= next_status:
                print(
                    {
                        "hardware_step": step,
                        "elapsed_seconds": now - start_time,
                        "tilt_radians": policy_step.tilt_radians,
                        "action_abs_max": float(np.max(np.abs(policy_step.action))),
                        "command": tuple(
                            float(value) for value in policy_step.effective_command
                        ),
                        "standing_transition_pending": (
                            policy_step.standing_transition_pending
                        ),
                        "remote_service_connected": remote_input.service_connected,
                        "remote_controller_connected": remote_input.controller_connected,
                        "remote_intent_armed": remote_input.intent_armed,
                        "remote_sample_age_ms": remote_input.sample_age_ms,
                        "upper_body_pose_following": (policy_step.upper_body_following),
                        "reference_sequence": policy_step.reference_sequence,
                        "reference_age_ms": policy_step.reference_age_ms,
                        "motor_age_ms": frame.max_motor_age_ms,
                        "imu_age_ms": frame.imu_age_ms,
                        "group_age_ms": frame.max_group_age_ms,
                        "state_sample_span_ms": frame.sample_span_ms,
                        "state_frame_round_trip_ms": (frame.state_frame_round_trip_ms),
                        "policy_log_dropped_records": (
                            None if policy_log is None else policy_log.dropped_records
                        ),
                    }
                )
                assert config.status_interval_seconds is not None
                next_status = now + config.status_interval_seconds
            status_finished = time.monotonic()
            control.pace(
                {
                    "input": (input_finished - loop_started) * 1_000.0,
                    "state": (state_finished - input_finished) * 1_000.0,
                    "policy": (policy_finished - state_finished) * 1_000.0,
                    "command": (command_finished - policy_finished) * 1_000.0,
                    "log": (log_finished - command_finished) * 1_000.0,
                    "status": (status_finished - log_finished) * 1_000.0,
                    "before_pace": (status_finished - loop_started) * 1_000.0,
                }
            )
        control.drain()


@dataclass
class _UpperBodyReferenceQualifier:
    required_frames: int
    qualified_frames: int = 0
    last_sequence: int | None = None
    active: bool = False

    def accept(self, reference: RobotReferenceFrame) -> RobotReferenceFrame | None:
        sequence = reference.sequence
        if self.last_sequence is None:
            self.qualified_frames = 1
            self.last_sequence = sequence
        elif sequence < self.last_sequence:
            self.reset()
            return None
        elif sequence > self.last_sequence:
            self.qualified_frames += 1
            self.last_sequence = sequence
        if self.qualified_frames >= self.required_frames:
            self.active = True
        return reference if self.active else None

    def reset(self) -> None:
        self.qualified_frames = 0
        self.last_sequence = None
        self.active = False
