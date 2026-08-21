"""Deploy a locomotion checkpoint through bello-stack's hardware client."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from bello_mujoco.training.checkpoint import (
    CheckpointNormalizationState,
    LocomotionCheckpointMetadata,
)
from bello_mujoco.envs.locomotion import BelloLocomotionEnv
from bello_mujoco.gamepad import axis_to_range
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
from bello_mujoco.hardware.twist2 import ReferenceUnavailable, Twist2ReferenceFrame
from bello_mujoco.policy import infer_policy_action
from bello_mujoco.locomotion_runtime import load_locomotion_policy_runtime
from bello_mujoco.training.ppo import normalize_observation


@dataclass(frozen=True)
class UpperBodyFollowConfig:
    """Fixed safety envelope for locomotion's pose-driven hold coordinates."""

    max_reference_age_ms: int = 1_000
    max_reference_joint_speed_rps: float = 25.0
    max_reference_joint_limit_violation_radians: float = 0.01
    max_target_speed_rps: float = 3.0 * math.pi


@dataclass(frozen=True)
class HardwareLocomotionConfig(HardwareControlConfig):
    """Hardware and visualizer-remote fail-closed deployment limits."""

    client_name: str = "bello-locomotion-policy"
    remote: RemoteCommandConfig = RemoteCommandConfig()
    fastsam_endpoint: str = "127.0.0.1:50055"
    fastsam_tls_root_certificate_path: Path | None = None
    fastsam_tls_server_name_override: str | None = None
    fastsam_connect_timeout_seconds: float = 5.0
    fastsam_rpc_timeout_seconds: float = 2.0
    fastsam_reference_ready_timeout_seconds: float = 15.0
    fastsam_reference_qualification_frames: int = 5
    fastsam_min_tracking_confidence: float = 0.5
    fastsam_min_joint_confidence: float = 0.25
    fastsam_max_pipeline_latency_ms: float = 120.0
    fastsam_transport_latency_budget_ms: float = 10.0
    fastsam_camera_to_world_quaternion_xyzw: tuple[float, float, float, float] = (
        0.5,
        -0.5,
        0.5,
        -0.5,
    )
    fastsam_align_initial_heading: bool = True
    fastsam_align_initial_floor: bool = True
    smplx_neutral_model_path: Path | None = None
    smplx_human_height_m: float = 1.66
    upper_body_follow: UpperBodyFollowConfig = UpperBodyFollowConfig()


@dataclass(frozen=True)
class HardwarePolicyStep:
    action: np.ndarray
    raw_action: np.ndarray
    physical_target: np.ndarray
    observation: np.ndarray
    policy_observation: np.ndarray
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
        self._gait_phase = 0.0
        self._standing_reference: Twist2ReferenceFrame | None = None
        self._upper_body_target: np.ndarray | None = None

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

    def set_upper_body_standing_reference(
        self, reference: Twist2ReferenceFrame
    ) -> None:
        """Bind the GMR standing arms used by ramp, hold, and pose fallback."""

        self.validate_upper_body_reference(reference, live=False)
        self._standing_reference = reference

    def reset(self) -> None:
        self._obs_stack = None
        self._last_action = np.zeros(self.env.action_size, dtype=np.float32)
        self._gait_phase = 0.0
        self._upper_body_target = None

    def precompile(self, reference: Twist2ReferenceFrame | None = None) -> None:
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

    def observation(self, frame: HardwareFrame, command: np.ndarray) -> np.ndarray:
        logical_position = self.inverse_transform @ frame.positions
        logical_velocity = self.inverse_transform @ frame.velocities
        projected_gravity = projected_gravity_from_xyzw(frame.orientation_xyzw)
        specific_force = specific_force_from_gravity_removed_acceleration(
            frame.linear_acceleration, projected_gravity
        )
        scale = self.env.config.observation.scale
        gait_active = self._command_is_moving(command)
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
        reference: Twist2ReferenceFrame | None = None,
    ) -> HardwarePolicyStep:
        observation = self.observation(frame, command)
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
        if self._command_is_moving(command):
            self._gait_phase = (
                self._gait_phase
                + self.env.config.gait.frequency_hz * self.control_timestep
            ) % 1.0
        else:
            self._gait_phase = 0.0
        return HardwarePolicyStep(
            action=action,
            raw_action=raw_action,
            physical_target=physical_target,
            observation=observation,
            policy_observation=policy_observation,
            tilt_radians=tilt_from_xyzw(frame.orientation_xyzw),
            reference_sequence=(None if reference is None else reference.sequence),
            reference_age_ms=(None if reference is None else reference.age_ms()),
            upper_body_following=reference is not None,
        )

    def validate_upper_body_reference(
        self,
        reference: Twist2ReferenceFrame,
        *,
        live: bool = True,
    ) -> None:
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

    def _reference_upper_body_target(
        self,
        reference: Twist2ReferenceFrame,
        *,
        live: bool,
    ) -> np.ndarray:
        self.validate_upper_body_reference(reference, live=live)
        desired = np.asarray(reference.qpos, dtype=np.float64)[
            self.upper_body_qpos_indices
        ]
        low = self.env.model.joint_target_low[self.upper_body_indices]
        high = self.env.model.joint_target_high[self.upper_body_indices]
        return np.clip(desired, low, high)

    def _apply_upper_body_reference(
        self,
        physical_target: np.ndarray,
        frame: HardwareFrame,
        reference: Twist2ReferenceFrame | None,
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

    def _command_is_moving(self, command: np.ndarray) -> bool:
        command = np.asarray(command, dtype=np.float32)
        return bool(
            np.any(np.abs(command[:2]) >= self.env.command_config.linear_deadband)
            or abs(float(command[2])) >= self.env.command_config.yaw_deadband
        )


def load_hardware_policy(
    checkpoint_path: Path,
    *,
    allow_git_commit_mismatch: bool = False,
    upper_body_follow: UpperBodyFollowConfig = UpperBodyFollowConfig(),
) -> tuple[HardwarePolicyAdapter, LocomotionCheckpointMetadata]:
    runtime = load_locomotion_policy_runtime(
        checkpoint_path,
        use_viewer_model=False,
        allow_git_commit_mismatch=allow_git_commit_mismatch,
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
    availability = metadata.asset.joint_availability
    if not availability.waist_yaw:
        raise ValueError("hardware locomotion requires waist_yaw_joint")
    if env.config.observation.privileged:
        raise ValueError("hardware actor observation must not be privileged")
    if env.action_size != 13:
        raise ValueError(
            f"hardware locomotion requires 13 policy joints, got {env.action_size}"
        )
    if metadata.ppo.policy.action_distribution != "tanh_gaussian":
        raise ValueError("hardware locomotion requires a bounded tanh-Gaussian policy")
    expected_upper_body = {
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow_pitch",
            "elbow_yaw",
            "wrist_pitch",
        )
    }
    hold_names = {
        env.model.actuator_joint_names[index] for index in env.model.hold_ctrl_indices
    }
    if hold_names != expected_upper_body:
        raise ValueError(
            "hardware locomotion hold coordinates must be exactly the 12 arm joints"
        )


def validate_hardware_config(
    config: HardwareLocomotionConfig,
    metadata: LocomotionCheckpointMetadata | None = None,
) -> None:
    validate_hardware_control_config(config)
    if not config.remote.endpoint.startswith("http://"):
        raise ValueError("remote.endpoint must use plaintext http:// in protocol v1")
    if config.remote.staleness_ms <= 0:
        raise ValueError("remote.staleness_ms must be positive")
    from bello_mujoco.hardware.fastsam import FastSamGrpcConfig

    FastSamGrpcConfig.from_hardware(config)
    if (
        not math.isfinite(config.fastsam_reference_ready_timeout_seconds)
        or config.fastsam_reference_ready_timeout_seconds <= 0.0
    ):
        raise ValueError(
            "fastsam_reference_ready_timeout_seconds must be positive and finite"
        )
    if config.fastsam_reference_qualification_frames < 1:
        raise ValueError("fastsam_reference_qualification_frames must be positive")
    upper = config.upper_body_follow
    positive = {
        "max_reference_joint_speed_rps": upper.max_reference_joint_speed_rps,
        "max_target_speed_rps": upper.max_target_speed_rps,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"upper_body_follow.{name} must be positive and finite")
    if (
        not math.isfinite(config.smplx_human_height_m)
        or config.smplx_human_height_m <= 0.0
    ):
        raise ValueError("smplx_human_height_m must be positive and finite")
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
    minimum_age_budget = (
        config.fastsam_max_pipeline_latency_ms
        + config.fastsam_transport_latency_budget_ms
    )
    if upper.max_reference_age_ms <= minimum_age_budget:
        raise ValueError(
            "upper_body_follow.max_reference_age_ms must exceed the Fast-SAM "
            "pipeline and transport latency budget"
        )
    if metadata is not None:
        command_ranges(metadata)
        command_minimum_moving_speeds(metadata)


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


def hardware_command(
    metadata: LocomotionCheckpointMetadata,
    remote_input: RemoteCommandInput,
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
    normalized = np.asarray((lateral, forward, yaw), dtype=np.float64)
    command = np.asarray(
        [
            axis_to_range(value, bounds, minimum)
            for value, bounds, minimum in zip(
                normalized,
                command_ranges(metadata),
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
        print(f"Visualizer remote service {state}; command is zero while disconnected.")
    if remote_input.controller_connection_changed:
        state = "connected" if remote_input.controller_connected else "disconnected"
        suffix = (
            f": {remote_input.controller_name}"
            if remote_input.controller_name is not None
            else ""
        )
        print(f"Remote controller {state}{suffix}; command is zero while disconnected.")
    if (
        remote_input.controller_connection_changed
        and remote_input.controller_connected
        and not remote_input.intent_armed
    ):
        print("Remote controller is waiting for neutral sticks; command is zero.")


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
        allow_git_commit_mismatch=config.allow_git_commit_mismatch,
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
        command = hardware_command(metadata, remote_input)
        if reference_source_factory is None:
            from bello_mujoco.hardware.fastsam import (
                fastsam_locomotion_reference_source,
            )

            reference_source_factory = fastsam_locomotion_reference_source
        references = resources.enter_context(reference_source_factory(adapter, config))
        standing = references.standing()
        adapter.set_upper_body_standing_reference(standing)
        live_reference = wait_for_qualified_upper_body_reference(
            references,
            adapter,
            config,
            remote,
        )
        if live_reference is None:
            print("Remote controller requested exit during pose preflight.")
            return
        remote_input = remote.poll()
        report_remote_connection(remote_input)
        if remote_input.abort_active:
            print("Remote controller requested exit during pose preflight.")
            return
        command = hardware_command(metadata, remote_input)
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
                "command": tuple(float(value) for value in command),
                "upper_body_pose_source": "fastsam_smplx_gmr",
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
    references: Any,
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
        try:
            while (
                config.max_duration_seconds is None
                or time.monotonic() - start_time < config.max_duration_seconds
            ):
                remote_input = remote.poll()
                report_remote_connection(remote_input)
                if remote_input.abort_active:
                    print("Remote controller requested exit; entering damped shutdown.")
                    break
                command = hardware_command(adapter.metadata, remote_input)
                frame = control.read_frame()
                reference_error: ReferenceUnavailable | None = None
                try:
                    reference = references.latest()
                    adapter.validate_upper_body_reference(reference)
                except ReferenceUnavailable as error:
                    reference = None
                    reference_error = error
                following = reference is not None
                if following != pose_following:
                    if following:
                        print("Live upper-body pose following is active.")
                    else:
                        print(
                            "Live upper-body pose is unavailable; returning the arms "
                            f"to GMR standing: {reference_error}"
                        )
                    pose_following = following
                policy_step = adapter.step(frame, command, reference)
                control.send_target(policy_step.physical_target)
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
                        command=command,
                    )
                if next_status is not None and now >= next_status:
                    print(
                        {
                            "hardware_step": step,
                            "elapsed_seconds": now - start_time,
                            "tilt_radians": policy_step.tilt_radians,
                            "action_abs_max": float(np.max(np.abs(policy_step.action))),
                            "command": tuple(float(value) for value in command),
                            "remote_service_connected": remote_input.service_connected,
                            "remote_controller_connected": remote_input.controller_connected,
                            "remote_intent_armed": remote_input.intent_armed,
                            "remote_sample_age_ms": remote_input.sample_age_ms,
                            "upper_body_pose_following": (
                                policy_step.upper_body_following
                            ),
                            "reference_sequence": policy_step.reference_sequence,
                            "reference_age_ms": policy_step.reference_age_ms,
                            "motor_age_ms": frame.max_motor_age_ms,
                            "imu_age_ms": frame.imu_age_ms,
                            "group_age_ms": frame.max_group_age_ms,
                            "state_sample_span_ms": frame.sample_span_ms,
                            "state_frame_round_trip_ms": (
                                frame.state_frame_round_trip_ms
                            ),
                        }
                    )
                    assert config.status_interval_seconds is not None
                    next_status = now + config.status_interval_seconds
                control.pace()
            control.drain()
        except KeyboardInterrupt:
            print("Hardware locomotion interrupted; entering damped shutdown.")


def wait_for_qualified_upper_body_reference(
    references: Any,
    adapter: HardwarePolicyAdapter,
    config: HardwareLocomotionConfig,
    remote: RemoteCommandSource,
) -> Twist2ReferenceFrame | None:
    """Require distinct fresh live poses while preserving the remote abort path."""

    deadline = time.monotonic() + config.fastsam_reference_ready_timeout_seconds
    last_sequence: int | None = None
    qualified_frames = 0
    last_error: ReferenceUnavailable | None = None
    while time.monotonic() < deadline:
        remote_input = remote.poll()
        report_remote_connection(remote_input)
        if remote_input.abort_active:
            return None
        try:
            reference = references.latest()
            adapter.validate_upper_body_reference(reference)
            if last_sequence is None or reference.sequence > last_sequence:
                qualified_frames += 1
                last_sequence = reference.sequence
            elif reference.sequence < last_sequence:
                qualified_frames = 0
                last_sequence = None
            if qualified_frames >= config.fastsam_reference_qualification_frames:
                return reference
        except ReferenceUnavailable as error:
            last_error = error
            qualified_frames = 0
            last_sequence = None
        time.sleep(adapter.control_timestep)
    detail = f": {last_error}" if last_error is not None else ""
    raise TimeoutError(f"timed out waiting for a qualified upper-body pose{detail}")
