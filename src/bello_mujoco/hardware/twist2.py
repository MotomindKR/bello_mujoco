"""Deploy a typed TWIST2 checkpoint with a live whole-body reference."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
import math
import time
from typing import Any, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from bello_mujoco.contracts import ActionLayout, ObservationLayout
from bello_mujoco.envs.twist import TWIST_POLICY_HZ, TWIST_SIMULATION_HZ
from bello_mujoco.envs.twist2 import (
    TWIST2_RAW_ACTION_CLIP,
    Twist2EnvConfig,
    Twist2ObservationContract,
)
from bello_mujoco.hardware.config import HardwareControlConfig
from bello_mujoco.hardware.policy_log import PolicyStepLogger, policy_phase_schema
from bello_mujoco.hardware.runtime import (
    HOLD_CONFIRMATION,
    MOTION_CONFIRMATION,
    HardwareFrame,
    actuator_coordinate_names,
    actuator_impedance_gains,
    armed_control_session,
    import_hardware_binding,
    normalized_quaternion_xyzw,
    require_async_hardware_binding,
    require_operator_confirmation,
    require_state_frame_binding,
    require_target_pose,
    validate_hardware_control_config,
    validate_hardware_frame,
    validate_hardware_inventory,
    wait_for_hardware_frame,
    wait_for_operator_confirmation_while_holding,
)
from bello_mujoco.policy import apply_actor
from bello_mujoco.sim import BelloModel, load_bello_model
from bello_mujoco.spatial import quat_rotate_inverse_wxyz
from bello_mujoco.training.checkpoint import (
    CheckpointNormalizationState,
    Twist2CheckpointMetadata,
    restore_evaluation_checkpoint,
)
from bello_mujoco.training.ppo import normalize_observation


_HARDWARE_LIMIT_ROUNDING_TOLERANCE_RADIANS = 1.0e-5
_ANKLE_HARD_LIMIT_INSET_RADIANS = 1.0e-9
_ANKLE_PROJECTION_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class HardwareTwist2Config(HardwareControlConfig):
    """Physical TWIST2 deployment and fail-closed reference limits."""

    client_name: str = "bello-twist2-policy"
    max_reference_age_ms: int = 250
    max_reference_root_height_m: float = 1.5
    min_reference_root_height_m: float = 0.35
    max_reference_root_speed_mps: float = 5.0
    max_reference_angular_speed_rps: float = 12.0
    max_reference_joint_speed_rps: float = 25.0
    max_reference_joint_limit_violation_radians: float = 0.01
    max_raw_action_abs: float = 20.0
    max_motion_tilt_radians: float = math.radians(70.0)
    policy_transition_seconds: float = 0.2
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


@dataclass(frozen=True)
class Twist2ReferenceFrame:
    """One GMR-retargeted Bello reference in MuJoCo qpos/qvel ordering."""

    qpos: np.ndarray
    qvel: np.ndarray
    sequence: int
    source_age_ms: int | None
    received_at: float

    def age_ms(self, now: float | None = None) -> float | None:
        if self.source_age_ms is None:
            return None
        current = time.monotonic() if now is None else now
        return self.source_age_ms + 1_000.0 * max(0.0, current - self.received_at)


class ReferenceUnavailable(RuntimeError):
    """The live reference is temporarily unsafe or unavailable."""


class ReferenceDiscontinuity(ReferenceUnavailable):
    """The tracked subject or source timeline changed and must be re-anchored."""


class Twist2ReferenceSource(Protocol):
    """Reference boundary implemented by the vision/GMR integration."""

    def standing(self) -> Twist2ReferenceFrame: ...

    def latest(self) -> Twist2ReferenceFrame: ...


@dataclass(frozen=True)
class HardwareTwist2Environment:
    """Native-model subset of BelloTwist2Env needed for actor deployment."""

    model: BelloModel
    config: Twist2EnvConfig
    contract: Twist2ObservationContract

    @property
    def action_size(self) -> int:
        return int(self.model.policy_ctrl_indices.size)

    @property
    def action_clip(self) -> float:
        return self.config.action_clip

    @property
    def control_timestep(self) -> float:
        return 1.0 / TWIST_POLICY_HZ

    @property
    def observation_layout(self) -> ObservationLayout:
        return self.contract.actor_layout

    @property
    def observation_size(self) -> int:
        return self.observation_layout.size

    @property
    def action_layout(self) -> ActionLayout:
        return ActionLayout(
            tuple(
                self.model.actuator_joint_names[index]
                for index in self.model.policy_ctrl_indices
            )
        )


@dataclass(frozen=True)
class HardwareTwist2Step:
    action: np.ndarray
    raw_action: np.ndarray
    physical_target: np.ndarray
    observation: np.ndarray
    policy_observation: np.ndarray
    command: np.ndarray
    action_clipped: bool
    reference_sequence: int
    reference_age_ms: float | None


@dataclass(frozen=True)
class _AnkleTargetEnvelope:
    side: str
    motor_indices: tuple[int, int]
    logical_indices: tuple[int, int]
    motor_to_joint: np.ndarray
    joint_low: np.ndarray
    joint_high: np.ndarray


def _clip_polygon_to_halfspace(
    polygon: np.ndarray,
    normal: np.ndarray,
    upper: float,
) -> np.ndarray:
    """Clip a two-dimensional convex polygon to ``normal @ point <= upper``."""

    if polygon.size == 0:
        return polygon.reshape(0, 2)
    clipped: list[np.ndarray] = []
    previous = polygon[-1]
    previous_value = float(normal @ previous - upper)
    previous_inside = previous_value <= _ANKLE_PROJECTION_TOLERANCE
    for current in polygon:
        current_value = float(normal @ current - upper)
        current_inside = current_value <= _ANKLE_PROJECTION_TOLERANCE
        if current_inside != previous_inside:
            denominator = previous_value - current_value
            if abs(denominator) <= _ANKLE_PROJECTION_TOLERANCE:
                raise RuntimeError("ankle envelope clipping encountered a flat edge")
            fraction = float(np.clip(previous_value / denominator, 0.0, 1.0))
            clipped.append(previous + fraction * (current - previous))
        if current_inside:
            clipped.append(current)
        previous = current
        previous_value = current_value
        previous_inside = current_inside
    if not clipped:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(clipped, dtype=np.float64)


def _ankle_feasible_polygon(
    envelope: _AnkleTargetEnvelope,
    control_limits: np.ndarray,
) -> np.ndarray:
    motor_indices = np.asarray(envelope.motor_indices, dtype=np.int32)
    limits = np.asarray(control_limits, dtype=np.float64)[motor_indices]
    if limits.shape != (2, 2) or not np.all(np.isfinite(limits)):
        raise ValueError(f"{envelope.side} ankle motor limits are invalid")
    low = limits[:, 0]
    high = limits[:, 1]
    if np.any(low >= high):
        raise ValueError(f"{envelope.side} ankle motor limits do not overlap")
    polygon = np.asarray(
        (
            (low[0], low[1]),
            (high[0], low[1]),
            (high[0], high[1]),
            (low[0], high[1]),
        ),
        dtype=np.float64,
    )
    safe_low = envelope.joint_low + _ANKLE_HARD_LIMIT_INSET_RADIANS
    safe_high = envelope.joint_high - _ANKLE_HARD_LIMIT_INSET_RADIANS
    if np.any(safe_low >= safe_high):
        raise ValueError(f"{envelope.side} ankle hard joint limits are invalid")
    for row, joint_low, joint_high in zip(
        envelope.motor_to_joint, safe_low, safe_high, strict=True
    ):
        polygon = _clip_polygon_to_halfspace(polygon, row, float(joint_high))
        polygon = _clip_polygon_to_halfspace(polygon, -row, float(-joint_low))
    if polygon.size == 0:
        raise ValueError(
            f"{envelope.side} ankle motor and hard joint envelopes do not overlap"
        )
    return polygon


def _project_point_to_polygon(point: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    if polygon.shape == (1, 2):
        return polygon[0].copy()
    best: np.ndarray | None = None
    best_distance = math.inf
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        edge = end - start
        denominator = float(edge @ edge)
        if denominator <= _ANKLE_PROJECTION_TOLERANCE:
            candidate = start
        else:
            fraction = float(np.clip((point - start) @ edge / denominator, 0.0, 1.0))
            candidate = start + fraction * edge
        distance = float((candidate - point) @ (candidate - point))
        if distance < best_distance:
            best = candidate
            best_distance = distance
    if best is None:
        raise RuntimeError("ankle envelope projection has no boundary")
    return best.copy()


def _project_ankle_motor_target(
    point: np.ndarray,
    envelope: _AnkleTargetEnvelope,
    control_limits: np.ndarray,
) -> np.ndarray:
    """Return the closest feasible pair of differential-ankle motor targets."""

    point = np.asarray(point, dtype=np.float64)
    polygon = _ankle_feasible_polygon(envelope, control_limits)
    safe_low = envelope.joint_low + _ANKLE_HARD_LIMIT_INSET_RADIANS
    safe_high = envelope.joint_high - _ANKLE_HARD_LIMIT_INSET_RADIANS
    resolved = envelope.motor_to_joint @ point
    motor_limits = np.asarray(control_limits, dtype=np.float64)[
        np.asarray(envelope.motor_indices, dtype=np.int32)
    ]
    if (
        np.all(point >= motor_limits[:, 0])
        and np.all(point <= motor_limits[:, 1])
        and np.all(resolved >= safe_low)
        and np.all(resolved <= safe_high)
    ):
        return point.copy()
    projected = _project_point_to_polygon(point, polygon)
    projected_resolved = envelope.motor_to_joint @ projected
    violation = np.maximum(
        safe_low - projected_resolved,
        projected_resolved - safe_high,
    )
    if float(np.max(violation)) > _ANKLE_PROJECTION_TOLERANCE:
        raise RuntimeError(
            f"{envelope.side} ankle projection remains outside its hard envelope"
        )
    return projected


class HardwareTwist2PolicyAdapter:
    """Reconstruct TWIST2 actor inputs and bounded physical motor targets."""

    def __init__(
        self,
        env: HardwareTwist2Environment,
        policy: Any,
        metadata: Twist2CheckpointMetadata,
        normalization: CheckpointNormalizationState,
        config: HardwareTwist2Config,
    ) -> None:
        validate_hardware_twist2_checkpoint(env, metadata)
        validate_hardware_twist2_config(config)
        self.env = env
        self.metadata = metadata
        self.config = config
        self.physical_coordinate_names = actuator_coordinate_names(env)
        self.coordinate_index = {
            name: index for index, name in enumerate(self.physical_coordinate_names)
        }
        if len(self.coordinate_index) != len(self.physical_coordinate_names):
            raise ValueError("physical actuator coordinate names must be unique")
        coordinate_count = len(self.physical_coordinate_names)
        self.transform = np.asarray(
            env.model.actuator_coordinate_transform, dtype=np.float64
        )
        if self.transform.shape != (coordinate_count, coordinate_count):
            raise ValueError(
                "TWIST2 hardware requires a square actuator coordinate transform"
            )
        condition = float(np.linalg.cond(self.transform))
        if not math.isfinite(condition) or condition > 1.0e6:
            raise ValueError(
                "actuator coordinate transform is not safely invertible: "
                f"condition={condition}"
            )
        self.inverse_transform = np.linalg.inv(self.transform)
        self.policy_indices = np.asarray(env.model.policy_ctrl_indices, dtype=np.int32)
        self.policy_qpos_indices = np.asarray(
            env.model.policy_qpos_indices, dtype=np.int32
        )
        self.policy_qvel_indices = np.asarray(
            env.model.policy_qvel_indices, dtype=np.int32
        )
        self.default_joint_target = np.asarray(
            env.model.default_joint_target, dtype=np.float64
        )
        self.kp, self.kd = actuator_impedance_gains(env)
        self._model_ctrl_limits = np.asarray(
            env.model.mj_model.actuator_ctrlrange, dtype=np.float64
        ).copy()
        if self._model_ctrl_limits.shape != (coordinate_count, 2):
            raise ValueError("TWIST2 actuator control ranges have an invalid shape")
        self._ctrl_limits = self._model_ctrl_limits.copy()
        ankle_names = {
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        }
        policy_names = env.action_layout.names
        self._ankle_policy_indices = np.asarray(
            [index for index, name in enumerate(policy_names) if name in ankle_names],
            dtype=np.int32,
        )
        if self._ankle_policy_indices.size != 4:
            raise ValueError(
                "TWIST2 hardware requires the four configured ankle policy joints"
            )
        self._ankle_target_envelopes = self._build_ankle_target_envelopes()
        for envelope in self._ankle_target_envelopes:
            _ankle_feasible_polygon(envelope, self._model_ctrl_limits)
        observation_norm = normalization.observation

        @eqx.filter_jit
        def infer(observation: jax.Array) -> tuple[jax.Array, jax.Array]:
            policy_observation = normalize_observation(
                observation, observation_norm, metadata.ppo
            )
            return policy_observation, apply_actor(policy, policy_observation)

        self._infer = infer
        self._past_frames: np.ndarray | None = None
        self._reset_frame: np.ndarray | None = None
        self._step_count = 0
        self._last_raw_action = np.zeros(env.action_size, dtype=np.float32)

    @property
    def control_timestep(self) -> float:
        return self.env.control_timestep

    def _build_ankle_target_envelopes(self) -> tuple[_AnkleTargetEnvelope, ...]:
        logical_names = self.env.model.actuator_joint_names
        logical_index = {name: index for index, name in enumerate(logical_names)}
        envelopes = []
        for side in ("left", "right"):
            motor_indices = tuple(
                self.coordinate_index[f"{side}_ankle_motor_{motor}"] for motor in (1, 2)
            )
            logical_indices = tuple(
                logical_index[f"{side}_ankle_{axis}_joint"]
                for axis in ("pitch", "roll")
            )
            motor_to_joint = self.inverse_transform[
                np.ix_(logical_indices, motor_indices)
            ].copy()
            unrelated = self.inverse_transform[np.asarray(logical_indices)].copy()
            unrelated[:, np.asarray(motor_indices)] = 0.0
            if np.max(np.abs(unrelated)) > 1.0e-10:
                raise ValueError(
                    f"{side} ankle joint coordinates depend on unrelated motors"
                )
            condition = float(np.linalg.cond(motor_to_joint))
            if not math.isfinite(condition) or condition > 1.0e6:
                raise ValueError(f"{side} ankle differential is not invertible")
            joint_ids = self.env.model.actuator_joint_ids[
                np.asarray(logical_indices, dtype=np.int32)
            ]
            if not np.all(self.env.model.mj_model.jnt_limited[joint_ids]):
                raise ValueError(f"{side} ankle joints require hard URDF limits")
            joint_range = np.asarray(
                self.env.model.mj_model.jnt_range[joint_ids], dtype=np.float64
            )
            if (
                joint_range.shape != (2, 2)
                or not np.all(np.isfinite(joint_range))
                or np.any(joint_range[:, 0] >= joint_range[:, 1])
            ):
                raise ValueError(f"{side} ankle hard URDF limits are invalid")
            envelopes.append(
                _AnkleTargetEnvelope(
                    side=side,
                    motor_indices=motor_indices,
                    logical_indices=logical_indices,
                    motor_to_joint=motor_to_joint,
                    joint_low=joint_range[:, 0].copy(),
                    joint_high=joint_range[:, 1].copy(),
                )
            )
        return tuple(envelopes)

    def _bounded_physical_target(self, requested_target: np.ndarray) -> np.ndarray:
        target = np.clip(
            requested_target, self._ctrl_limits[:, 0], self._ctrl_limits[:, 1]
        )
        for envelope in self._ankle_target_envelopes:
            indices = np.asarray(envelope.motor_indices, dtype=np.int32)
            target[indices] = _project_ankle_motor_target(
                target[indices], envelope, self._ctrl_limits
            )
        return target

    def _validate_ankle_target(self, physical_target: np.ndarray) -> None:
        for envelope in self._ankle_target_envelopes:
            indices = np.asarray(envelope.motor_indices, dtype=np.int32)
            resolved = envelope.motor_to_joint @ physical_target[indices]
            violation = np.maximum(
                envelope.joint_low - resolved,
                resolved - envelope.joint_high,
            )
            index = int(np.argmax(violation))
            if violation[index] > _ANKLE_PROJECTION_TOLERANCE:
                logical_index = envelope.logical_indices[index]
                name = self.env.model.actuator_joint_names[logical_index]
                raise RuntimeError(
                    f"projected TWIST2 hardware target for {name!r} remains "
                    f"outside its hard URDF envelope by {violation[index]:.9f} rad"
                )

    @property
    def nominal_physical_target(self) -> np.ndarray:
        target = self.transform @ self.default_joint_target
        return self._bounded_physical_target(target)

    def reference_physical_target(self, reference: Twist2ReferenceFrame) -> np.ndarray:
        """Map a retargeted reference pose into physical motor coordinates."""

        validate_twist2_reference(reference, self.env, self.config)
        logical_target = self.default_joint_target.copy()
        logical_target[self.policy_indices] = np.asarray(reference.qpos)[
            self.policy_qpos_indices
        ]
        target = self.transform @ logical_target
        return self._bounded_physical_target(target)

    def reset(self) -> None:
        self._past_frames = None
        self._reset_frame = None
        self._step_count = 0
        self._last_raw_action = np.zeros(self.env.action_size, dtype=np.float32)

    def precompile(self, standing: Twist2ReferenceFrame) -> None:
        nominal = self.reference_physical_target(standing)
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
        self.step(frame, standing)
        self.reset()

    def command(self, reference: Twist2ReferenceFrame) -> np.ndarray:
        validate_twist2_reference(reference, self.env, self.config)
        qpos = np.asarray(reference.qpos, dtype=np.float64)
        qvel = np.asarray(reference.qvel, dtype=np.float64)
        root = self.env.model.root_qpos_adr
        root_velocity = self.env.model.root_qvel_adr
        root_quaternion = qpos[root + 3 : root + 7]
        local_linear_velocity = quat_rotate_inverse_wxyz(
            root_quaternion, qvel[root_velocity : root_velocity + 3]
        )
        local_angular_velocity = quat_rotate_inverse_wxyz(
            root_quaternion, qvel[root_velocity + 3 : root_velocity + 6]
        )
        command = np.concatenate(
            (
                local_linear_velocity[:2],
                qpos[root + 2 : root + 3],
                _roll_pitch_wxyz(root_quaternion),
                local_angular_velocity[2:3],
                qpos[self.policy_qpos_indices],
            )
        ).astype(np.float32)
        if command.shape != (self.env.contract.command_frame_size,):
            raise RuntimeError("TWIST2 hardware command layout has drifted")
        return command

    def proprioception(self, frame: HardwareFrame) -> np.ndarray:
        logical_position = self.inverse_transform @ frame.positions
        logical_velocity = self.inverse_transform @ frame.velocities
        orientation_xyzw = normalized_quaternion_xyzw(frame.orientation_xyzw)
        orientation_wxyz = orientation_xyzw[[3, 0, 1, 2]]
        # The hardware IMU gyro is already expressed in the pelvis frame. The
        # training environment performs its world-to-body conversion upstream.
        base_angular_velocity = 0.25 * np.asarray(
            frame.angular_velocity, dtype=np.float64
        )
        joint_position = (
            logical_position[self.policy_indices]
            - self.default_joint_target[self.policy_indices]
        )
        joint_velocity = 0.05 * logical_velocity[self.policy_indices]
        joint_velocity[self._ankle_policy_indices] = 0.0
        proprioception = np.concatenate(
            (
                base_angular_velocity,
                _roll_pitch_wxyz(orientation_wxyz),
                joint_position,
                joint_velocity,
                self._last_raw_action,
            )
        ).astype(np.float32)
        if proprioception.shape != (self.env.contract.proprioception_frame_size,):
            raise RuntimeError("TWIST2 hardware proprioception layout has drifted")
        if not np.all(np.isfinite(proprioception)):
            raise RuntimeError("TWIST2 hardware proprioception is non-finite")
        return proprioception

    def observation(
        self, frame: HardwareFrame, reference: Twist2ReferenceFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        command = self.command(reference)
        current = np.concatenate((command, self.proprioception(frame)))
        past_count = self.env.contract.past_frames
        if self._step_count == 0:
            history = np.zeros(
                past_count * self.env.contract.actor_frame_size, dtype=np.float32
            )
        elif self._step_count == 1:
            assert self._reset_frame is not None
            history = np.repeat(self._reset_frame[None, :], past_count, axis=0).reshape(
                -1
            )
        else:
            assert self._past_frames is not None
            history = self._past_frames.reshape(-1)
        observation = np.concatenate((current, history, command)).astype(np.float32)
        if observation.shape != (self.env.observation_size,):
            raise RuntimeError(
                "TWIST2 hardware actor observation has shape "
                f"{observation.shape}, expected {(self.env.observation_size,)}"
            )
        if not np.all(np.isfinite(observation)):
            raise RuntimeError("TWIST2 hardware actor observation is non-finite")
        return observation, current, command

    def step(
        self, frame: HardwareFrame, reference: Twist2ReferenceFrame
    ) -> HardwareTwist2Step:
        observation, current, command = self.observation(frame, reference)
        policy_observation, raw_action = self._infer(jnp.asarray(observation))
        policy_observation, raw_action = jax.device_get(
            (policy_observation, raw_action)
        )
        policy_observation = np.asarray(policy_observation, dtype=np.float32)
        raw_action = np.asarray(raw_action, dtype=np.float32)
        if raw_action.shape != (self.env.action_size,) or not np.all(
            np.isfinite(raw_action)
        ):
            raise RuntimeError(
                f"TWIST2 policy produced an invalid raw action {raw_action.shape}"
            )
        raw_abs = float(np.max(np.abs(raw_action)))
        if raw_abs > self.config.max_raw_action_abs:
            raise RuntimeError(
                f"raw TWIST2 action {raw_abs:.3f} exceeds fail-closed limit "
                f"{self.config.max_raw_action_abs:.3f}"
            )
        action = np.clip(raw_action, -self.env.action_clip, self.env.action_clip)
        requested_logical_target = self.default_joint_target.copy()
        requested_logical_target[self.policy_indices] += 0.5 * action
        requested_target = self.transform @ requested_logical_target
        physical_target = self._bounded_physical_target(requested_target)
        if not np.all(np.isfinite(physical_target)):
            raise RuntimeError("TWIST2 policy produced non-finite physical targets")
        self._validate_ankle_target(physical_target)
        action_clipped = not np.array_equal(action, raw_action) or not np.array_equal(
            physical_target, requested_target
        )
        if self._step_count == 0:
            self._reset_frame = current.copy()
        elif self._step_count == 1:
            # The source implementation returns the normally shifted history
            # to the actor, but stores a separately filled post-action history
            # on its first transition.
            self._past_frames = np.repeat(
                current[None, :], self.env.contract.past_frames, axis=0
            )
            self._reset_frame = None
        else:
            assert self._past_frames is not None
            self._past_frames = np.concatenate(
                (self._past_frames[1:], current[None, :]), axis=0
            )
        self._last_raw_action = raw_action
        self._step_count += 1
        return HardwareTwist2Step(
            action=action,
            raw_action=raw_action,
            physical_target=physical_target,
            observation=observation,
            policy_observation=policy_observation,
            command=command,
            action_clipped=action_clipped,
            reference_sequence=reference.sequence,
            reference_age_ms=reference.age_ms(),
        )


def validate_hardware_twist2_checkpoint(
    env: HardwareTwist2Environment,
    metadata: Twist2CheckpointMetadata,
) -> None:
    if metadata.task != "twist2":
        raise ValueError("TWIST2 hardware requires a typed twist2 checkpoint")
    if metadata.ppo.policy.architecture != "twist2":
        raise ValueError("TWIST2 hardware requires the twist2 actor architecture")
    if metadata.ppo.policy.action_distribution != "gaussian":
        raise ValueError("TWIST2 hardware requires the released Gaussian actor")
    if metadata.ppo.policy.obs_stack_size != 1:
        raise ValueError("TWIST2 hardware requires PPO observation stack size one")
    if metadata.asset.joint_availability.enabled_joint_names != {"waist_yaw_joint"}:
        raise ValueError(
            "TWIST2 hardware requires only waist yaw among optional joints"
        )
    if env.action_size != 25 or env.model.mj_model.nu != 25:
        raise ValueError("TWIST2 hardware requires 25 policy and physical coordinates")
    if not math.isclose(env.action_clip, TWIST2_RAW_ACTION_CLIP):
        raise ValueError("TWIST2 hardware action clip differs from the release")
    if env.contract != metadata.observation_contract:
        raise ValueError("TWIST2 hardware observation contract differs from checkpoint")
    if env.observation_layout != metadata.actor_observation_layout:
        raise ValueError("TWIST2 hardware actor layout differs from checkpoint")
    if env.action_layout != metadata.action_layout:
        raise ValueError("TWIST2 hardware action layout differs from checkpoint")
    if env.observation_size != metadata.obs_size:
        raise ValueError("TWIST2 hardware observation size differs from checkpoint")


def validate_hardware_twist2_config(config: HardwareTwist2Config) -> None:
    validate_hardware_control_config(config)
    positive = {
        "max_reference_root_height_m": config.max_reference_root_height_m,
        "min_reference_root_height_m": config.min_reference_root_height_m,
        "max_reference_root_speed_mps": config.max_reference_root_speed_mps,
        "max_reference_angular_speed_rps": config.max_reference_angular_speed_rps,
        "max_reference_joint_speed_rps": config.max_reference_joint_speed_rps,
        "max_raw_action_abs": config.max_raw_action_abs,
        "max_motion_tilt_radians": config.max_motion_tilt_radians,
        "policy_transition_seconds": config.policy_transition_seconds,
        "fastsam_connect_timeout_seconds": config.fastsam_connect_timeout_seconds,
        "fastsam_rpc_timeout_seconds": config.fastsam_rpc_timeout_seconds,
        "fastsam_reference_ready_timeout_seconds": (
            config.fastsam_reference_ready_timeout_seconds
        ),
        "fastsam_max_pipeline_latency_ms": config.fastsam_max_pipeline_latency_ms,
        "smplx_human_height_m": config.smplx_human_height_m,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if config.max_reference_age_ms <= 0:
        raise ValueError("max_reference_age_ms must be positive")
    if config.fastsam_reference_qualification_frames < 1:
        raise ValueError("fastsam_reference_qualification_frames must be positive")
    minimum_age_budget = (
        config.fastsam_max_pipeline_latency_ms
        + config.fastsam_transport_latency_budget_ms
    )
    if config.max_reference_age_ms <= minimum_age_budget:
        raise ValueError(
            "max_reference_age_ms must exceed the configured Fast-SAM pipeline "
            "and transport latency budget"
        )
    if (
        not math.isfinite(config.max_reference_joint_limit_violation_radians)
        or config.max_reference_joint_limit_violation_radians < 0.0
    ):
        raise ValueError(
            "max_reference_joint_limit_violation_radians must be finite and nonnegative"
        )
    if config.min_reference_root_height_m >= config.max_reference_root_height_m:
        raise ValueError("reference root-height bounds must be strictly ordered")
    if config.max_raw_action_abs < TWIST2_RAW_ACTION_CLIP:
        raise ValueError(
            "max_raw_action_abs must cover the released TWIST2 action clip"
        )
    if not config.fastsam_endpoint or "://" in config.fastsam_endpoint:
        raise ValueError("fastsam_endpoint must be a non-empty gRPC host:port target")
    for name, value in (
        ("fastsam_min_tracking_confidence", config.fastsam_min_tracking_confidence),
        ("fastsam_min_joint_confidence", config.fastsam_min_joint_confidence),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    if (
        not math.isfinite(config.fastsam_transport_latency_budget_ms)
        or config.fastsam_transport_latency_budget_ms < 0.0
    ):
        raise ValueError(
            "fastsam_transport_latency_budget_ms must be finite and nonnegative"
        )
    camera_quaternion = np.asarray(
        config.fastsam_camera_to_world_quaternion_xyzw, dtype=np.float64
    )
    if camera_quaternion.shape != (4,) or not np.all(np.isfinite(camera_quaternion)):
        raise ValueError(
            "fastsam_camera_to_world_quaternion_xyzw must contain four finite values"
        )
    if not 0.999 <= float(np.linalg.norm(camera_quaternion)) <= 1.001:
        raise ValueError("fastsam_camera_to_world_quaternion_xyzw must be normalized")


def validate_twist2_reference(
    reference: Twist2ReferenceFrame,
    env: HardwareTwist2Environment,
    config: HardwareTwist2Config,
) -> None:
    qpos = np.asarray(reference.qpos)
    qvel = np.asarray(reference.qvel)
    if qpos.shape != (env.model.mj_model.nq,) or not np.all(np.isfinite(qpos)):
        raise ReferenceUnavailable(
            f"TWIST2 reference qpos must contain {env.model.mj_model.nq} finite values"
        )
    if qvel.shape != (env.model.mj_model.nv,) or not np.all(np.isfinite(qvel)):
        raise ReferenceUnavailable(
            f"TWIST2 reference qvel must contain {env.model.mj_model.nv} finite values"
        )
    if reference.sequence < 0:
        raise ReferenceUnavailable("TWIST2 reference sequence must be nonnegative")
    if reference.source_age_ms is not None and reference.source_age_ms < 0:
        raise ReferenceUnavailable("TWIST2 reference source age must be nonnegative")
    age = reference.age_ms()
    if age is not None and age > config.max_reference_age_ms:
        raise ReferenceUnavailable(
            f"TWIST2 reference is stale: {age:.1f} ms exceeds "
            f"{config.max_reference_age_ms} ms"
        )
    root = env.model.root_qpos_adr
    root_velocity = env.model.root_qvel_adr
    root_height = float(qpos[root + 2])
    if not (
        config.min_reference_root_height_m
        <= root_height
        <= config.max_reference_root_height_m
    ):
        raise ReferenceUnavailable(
            f"TWIST2 reference root height is unsafe: {root_height:.3f} m"
        )
    quaternion = qpos[root + 3 : root + 7]
    quaternion_norm = float(np.linalg.norm(quaternion))
    if not 0.999 <= quaternion_norm <= 1.001:
        raise ReferenceUnavailable(
            f"TWIST2 reference root quaternion norm is invalid: {quaternion_norm}"
        )
    root_speed = float(np.linalg.norm(qvel[root_velocity : root_velocity + 3]))
    if root_speed > config.max_reference_root_speed_mps:
        raise ReferenceUnavailable(
            f"TWIST2 reference root speed is unsafe: {root_speed:.3f} m/s"
        )
    angular_speed = float(np.linalg.norm(qvel[root_velocity + 3 : root_velocity + 6]))
    if angular_speed > config.max_reference_angular_speed_rps:
        raise ReferenceUnavailable(
            f"TWIST2 reference angular speed is unsafe: {angular_speed:.3f} rad/s"
        )
    joint_speed = float(np.max(np.abs(qvel[np.asarray(env.model.policy_qvel_indices)])))
    if joint_speed > config.max_reference_joint_speed_rps:
        raise ReferenceUnavailable(
            f"TWIST2 reference joint speed is unsafe: {joint_speed:.3f} rad/s"
        )
    policy = np.asarray(env.model.policy_ctrl_indices)
    positions = qpos[np.asarray(env.model.policy_qpos_indices)]
    joint_ids = np.asarray(env.model.actuator_joint_ids)[policy]
    if not np.all(env.model.mj_model.jnt_limited[joint_ids]):
        raise RuntimeError("TWIST2 policy reference joints require hard limits")
    ranges = np.asarray(env.model.mj_model.jnt_range)[joint_ids]
    lower = ranges[:, 0]
    upper = ranges[:, 1]
    tolerance = config.max_reference_joint_limit_violation_radians
    violation = np.maximum(lower - positions, positions - upper)
    index = int(np.argmax(violation))
    if violation[index] > tolerance:
        joint_name = env.model.actuator_joint_names[int(policy[index])]
        raise ReferenceUnavailable(
            f"TWIST2 reference joint {joint_name!r} exceeds its limit by "
            f"{violation[index]:.3f} rad"
        )


def bind_twist2_hardware_coordinate_limits(
    groups: dict[str, Any],
    adapter: HardwareTwist2PolicyAdapter,
) -> None:
    """Validate live ranges and bind commands to their exact intersection."""

    motors = {
        motor.coordinate: motor for group in groups.values() for motor in group.motors
    }
    effective_limits = adapter._model_ctrl_limits.copy()
    for index, coordinate in enumerate(adapter.physical_coordinate_names):
        motor = motors[coordinate]
        model_low, model_high = adapter._model_ctrl_limits[index]
        hardware_low = float(motor.limits.position_min)
        hardware_high = float(motor.limits.position_max)
        low_excess = hardware_low - model_low
        high_excess = model_high - hardware_high
        if (
            low_excess > _HARDWARE_LIMIT_ROUNDING_TOLERANCE_RADIANS
            or high_excess > _HARDWARE_LIMIT_ROUNDING_TOLERANCE_RADIANS
        ):
            raise ValueError(
                f"TWIST2 model range for {coordinate!r} "
                f"[{model_low:.9f}, {model_high:.9f}] exceeds the hardware "
                f"range [{hardware_low:.9f}, {hardware_high:.9f}] by more than "
                f"{_HARDWARE_LIMIT_ROUNDING_TOLERANCE_RADIANS:.1e} rad"
            )
        effective_limits[index] = (
            max(model_low, hardware_low),
            min(model_high, hardware_high),
        )
        if effective_limits[index, 0] >= effective_limits[index, 1]:
            raise ValueError(
                f"TWIST2 model and hardware ranges for {coordinate!r} do not overlap"
            )
    for envelope in adapter._ankle_target_envelopes:
        _ankle_feasible_polygon(envelope, effective_limits)
    adapter._ctrl_limits = effective_limits


def load_hardware_twist2_policy(
    checkpoint_path: Path,
    config: HardwareTwist2Config,
) -> tuple[HardwareTwist2PolicyAdapter, Twist2CheckpointMetadata]:
    train_state, metadata, normalization = restore_evaluation_checkpoint(
        checkpoint_path,
        allow_git_commit_mismatch=config.allow_git_commit_mismatch,
    )
    if not isinstance(metadata, Twist2CheckpointMetadata):
        raise ValueError("TWIST2 hardware requires a typed twist2 checkpoint")
    model = load_bello_model(
        metadata.asset,
        warp=metadata.warp,
        batch_size=1,
        terrain=metadata.simulation.terrain,
        load_mjx=False,
        simulation_timestep=1.0 / TWIST_SIMULATION_HZ,
    )
    contract = Twist2ObservationContract(
        joint_names=tuple(
            model.actuator_joint_names[index] for index in model.policy_ctrl_indices
        ),
        key_body_names=metadata.observation_contract.key_body_names,
        privileged_offsets=metadata.observation_contract.privileged_offsets,
        past_frames=metadata.observation_contract.past_frames,
        randomization_size=metadata.observation_contract.randomization_size,
    )
    env = HardwareTwist2Environment(model, metadata.twist2_env, contract)
    return (
        HardwareTwist2PolicyAdapter(
            env, train_state.policy, metadata, normalization, config
        ),
        metadata,
    )


def _roll_pitch_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    return np.asarray((roll, pitch), dtype=np.float64)


def run_hardware_twist2(
    checkpoint_path: Path,
    config: HardwareTwist2Config,
    *,
    reference_source_factory: Callable[
        [HardwareTwist2PolicyAdapter, HardwareTwist2Config], Any
    ]
    | None = None,
    hardware_binding: Any | None = None,
    operator_input: Callable[[str], str] | None = None,
) -> None:
    """Run TWIST2 from the pinned Fast-SAM stream or an injected test source."""

    validate_hardware_twist2_config(config)
    adapter, metadata = load_hardware_twist2_policy(checkpoint_path, config)
    if reference_source_factory is None:
        from bello_mujoco.hardware.fastsam import fastsam_reference_source

        reference_source_factory = fastsam_reference_source
    binding = hardware_binding or import_hardware_binding()
    require_state_frame_binding(binding)
    if config.arm:
        require_async_hardware_binding(binding)
    with (
        reference_source_factory(adapter, config) as references,
        binding.Robot.connect(
            config.endpoint, timeout=config.connection_timeout_seconds
        ) as robot,
    ):
        standing = references.standing()
        validate_twist2_reference(standing, adapter.env, config)
        groups = validate_hardware_inventory(robot, adapter, config, binding)
        bind_twist2_hardware_coordinate_limits(groups, adapter)
        standing_target = adapter.reference_physical_target(standing)
        adapter.precompile(standing)
        state_reader = robot.states(groups=config.groups, imus=(config.imu_name,))
        frame = wait_for_hardware_frame(
            state_reader,
            adapter.physical_coordinate_names,
            config.groups,
            config.imu_name,
            config.telemetry_timeout_seconds,
        )
        validate_hardware_frame(frame, config, expected_group_state="IDLE")
        initial_error = float(np.max(np.abs(frame.positions - standing_target)))
        initial_preview = adapter.step(frame, standing)
        adapter.reset()
        root = adapter.env.model.root_qpos_adr
        standing_orientation_wxyz = np.asarray(standing.qpos)[root + 3 : root + 7]
        settled_preview = adapter.step(
            replace(
                frame,
                positions=standing_target,
                velocities=np.zeros_like(standing_target),
                orientation_xyzw=standing_orientation_wxyz[[1, 2, 3, 0]],
                angular_velocity=np.zeros(3, dtype=np.float64),
            ),
            standing,
        )
        adapter.reset()
        print(
            {
                "twist2_hardware_preflight": "passed",
                "checkpoint_task": metadata.task,
                "endpoint": config.endpoint,
                "groups": config.groups,
                "coordinates": len(adapter.physical_coordinate_names),
                "policy_joints": adapter.env.action_size,
                "actor_observation_size": adapter.env.observation_size,
                "control_rate_hz": 1.0 / adapter.control_timestep,
                "jax_backend": jax.default_backend(),
                "normal_standing_ramp_distance_max": initial_error,
                "initial_state_preview_action_abs_max": float(
                    np.max(np.abs(initial_preview.raw_action))
                ),
                "initial_state_preview_action_clipped": (
                    initial_preview.action_clipped
                ),
                "settled_standing_preview_action_abs_max": float(
                    np.max(np.abs(settled_preview.raw_action))
                ),
                "settled_standing_preview_action_clipped": (
                    settled_preview.action_clipped
                ),
                "armed": config.arm,
            }
        )
        if not config.arm:
            print(
                "Read-only TWIST2 preflight complete; no lease was acquired "
                "and no motor command was sent."
            )
            return
        require_operator_confirmation(
            HOLD_CONFIRMATION,
            "Support the robot, then type HOLD to acquire control and begin the "
            "normal-standing ramp: ",
            operator_input,
        )
        frame = wait_for_hardware_frame(
            state_reader,
            adapter.physical_coordinate_names,
            config.groups,
            config.imu_name,
            config.telemetry_timeout_seconds,
        )
        validate_hardware_frame(frame, config, expected_group_state="IDLE")
        schema = policy_phase_schema(adapter)
        log_context = (
            PolicyStepLogger(
                config.policy_log_path,
                task="twist2",
                checkpoints={"twist2": checkpoint_path},
                phases={
                    "standing": schema,
                    "transition_to_motion": schema,
                    "motion": schema,
                    "transition_to_standing": schema,
                },
            )
            if config.policy_log_path is not None
            else nullcontext(None)
        )
        with log_context as policy_log:
            control_physical_twist2(
                binding,
                groups,
                state_reader,
                references,
                standing,
                frame,
                adapter,
                config,
                policy_log=policy_log,
                operator_input=operator_input,
            )


def control_physical_twist2(
    binding: Any,
    groups: dict[str, Any],
    state_reader: Any,
    references: Twist2ReferenceSource,
    standing: Twist2ReferenceFrame,
    initial_frame: HardwareFrame,
    adapter: HardwareTwist2PolicyAdapter,
    config: HardwareTwist2Config,
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
        thread_name="bello-twist2-control",
        operator_input=operator_input,
    ) as control:
        standing_target = adapter.reference_physical_target(standing)
        frame = control.ramp_to_target(initial_frame.positions, standing_target)
        if frame is None:
            raise RuntimeError("TWIST2 normal-standing ramp ended unexpectedly")
        require_target_pose(
            frame,
            adapter,
            standing_target,
            pose_name="the normal standing pose",
        )

        def validate_hold(held_frame: HardwareFrame) -> None:
            require_target_pose(
                held_frame,
                adapter,
                standing_target,
                pose_name="the normal standing pose",
            )

        control.wait_for_run(
            "Normal-standing ramp complete. Release external support, let "
            "the robot settle, then type RUN to start TWIST2 normal standing: ",
            operator_input=operator_input,
            validate_hold=validate_hold,
        )
        require_target_pose(
            control.read_frame(),
            adapter,
            standing_target,
            pose_name="the normal standing pose",
        )
        adapter.reset()
        run_started = time.monotonic()
        control.reset_deadline()

        def duration_available() -> bool:
            return config.max_duration_seconds is None or (
                time.monotonic() - run_started < config.max_duration_seconds
            )

        standing_steps = 0
        next_standing_status = (
            run_started if config.status_interval_seconds is not None else None
        )

        def standing_step() -> bool:
            nonlocal standing_steps, next_standing_status
            if not duration_available():
                return False
            frame = control.read_frame()
            step = adapter.step(frame, standing)
            control.send_target(step.physical_target)
            standing_steps += 1
            now = time.monotonic()
            _log_twist2_step(
                policy_log,
                phase="standing",
                step_number=standing_steps,
                run_started=run_started,
                step=step,
            )
            control.pace()
            if next_standing_status is not None and now >= next_standing_status:
                print(
                    {
                        "hardware_phase": "standing",
                        "hardware_step": standing_steps,
                        "elapsed_seconds": now - run_started,
                        "action_abs_max": float(np.max(np.abs(step.action))),
                        "action_clipped": step.action_clipped,
                        "motor_age_ms": frame.max_motor_age_ms,
                        "imu_age_ms": frame.imu_age_ms,
                    }
                )
                assert config.status_interval_seconds is not None
                next_standing_status = now + config.status_interval_seconds
            return True

        print("Transitioning from the normal-standing hold to TWIST2 policy standing.")
        _transition_twist2_reference(
            control,
            adapter,
            lambda: standing,
            config,
            phase="transition_to_standing",
            policy_log=policy_log,
            run_started=run_started,
            max_tilt_radians=config.max_tilt_radians,
        )
        control.reset_deadline()
        print(
            "TWIST2 normal standing is active. "
            "Type MOTION to begin live whole-body tracking."
        )
        try:
            while True:
                first_live = None
                while first_live is None:
                    requested_motion = wait_for_operator_confirmation_while_holding(
                        MOTION_CONFIRMATION,
                        "Type MOTION to transition from normal standing to live "
                        "tracking: ",
                        standing_step,
                        operator_input,
                    )
                    if not requested_motion:
                        control.drain()
                        print("Hardware run duration completed before live tracking.")
                        return
                    print(
                        "MOTION confirmed. Waiting for a qualified live SMPL-X "
                        "target while normal standing remains active."
                    )
                    first_live = _wait_for_live_twist2_reference(
                        references,
                        adapter,
                        config,
                        standing_step,
                    )
                    if first_live is None:
                        if not duration_available():
                            control.drain()
                            print(
                                "Hardware run duration completed before live tracking."
                            )
                            return
                        print(
                            "No qualified live SMPL-X target arrived; normal standing "
                            "remains active. Type MOTION to retry."
                        )

                print("Transitioning from normal standing to live TWIST2 tracking.")
                try:
                    _transition_twist2_reference(
                        control,
                        adapter,
                        lambda: references.latest(),
                        config,
                        phase="transition_to_motion",
                        policy_log=policy_log,
                        run_started=run_started,
                        first_reference=first_live,
                    )
                except ReferenceUnavailable as error:
                    print(
                        "Live TWIST2 reference became unavailable during the "
                        f"transition; returning to normal standing: {error}"
                    )
                    _transition_twist2_reference(
                        control,
                        adapter,
                        lambda: standing,
                        config,
                        phase="transition_to_standing",
                        policy_log=policy_log,
                        run_started=run_started,
                    )
                    control.reset_deadline()
                    next_standing_status = (
                        time.monotonic()
                        if config.status_interval_seconds is not None
                        else None
                    )
                    print(
                        "TWIST2 normal standing resumed. Type MOTION to retry live "
                        "tracking."
                    )
                    continue
                break
            motion_started = time.monotonic()
            control.reset_deadline()
            motion_steps = 0
            next_motion_status = (
                motion_started if config.status_interval_seconds is not None else None
            )
            source_lost: ReferenceUnavailable | None = None
            try:
                while duration_available():
                    frame = control.read_frame(
                        max_tilt_radians=config.max_motion_tilt_radians
                    )
                    try:
                        reference = references.latest()
                        step = adapter.step(frame, reference)
                    except ReferenceUnavailable as error:
                        source_lost = error
                        break
                    control.send_target(step.physical_target)
                    motion_steps += 1
                    now = time.monotonic()
                    _log_twist2_step(
                        policy_log,
                        phase="motion",
                        step_number=motion_steps,
                        run_started=run_started,
                        step=step,
                    )
                    control.pace()
                    if next_motion_status is not None and now >= next_motion_status:
                        print(
                            {
                                "hardware_phase": "motion",
                                "hardware_step": motion_steps,
                                "elapsed_seconds": now - run_started,
                                "reference_sequence": step.reference_sequence,
                                "reference_age_ms": step.reference_age_ms,
                                "action_abs_max": float(np.max(np.abs(step.action))),
                                "action_clipped": step.action_clipped,
                            }
                        )
                        assert config.status_interval_seconds is not None
                        next_motion_status = now + config.status_interval_seconds
            except KeyboardInterrupt:
                print("Live TWIST2 tracking interrupted; returning to standing.")
            if source_lost is not None:
                print(
                    "Live TWIST2 reference became unavailable; returning to "
                    f"normal standing: {source_lost}"
                )
            _transition_twist2_reference(
                control,
                adapter,
                lambda: standing,
                config,
                phase="transition_to_standing",
                policy_log=policy_log,
                run_started=run_started,
            )
            control.reset_deadline()
            next_standing_status = (
                time.monotonic() if config.status_interval_seconds is not None else None
            )
            print("TWIST2 normal standing resumed; Ctrl-C enters damped shutdown.")
            while duration_available():
                standing_step()
            control.drain()
        except KeyboardInterrupt:
            print("TWIST2 standing interrupted; entering damped shutdown.")


def _transition_twist2_reference(
    control: Any,
    adapter: HardwareTwist2PolicyAdapter,
    reference: Callable[[], Twist2ReferenceFrame],
    config: HardwareTwist2Config,
    *,
    phase: str,
    policy_log: PolicyStepLogger | None,
    run_started: float,
    first_reference: Twist2ReferenceFrame | None = None,
    max_tilt_radians: float | None = None,
) -> None:
    control.drain()
    initial_target = control.previous_target.copy()
    adapter.reset()
    control.reset_deadline()
    steps = max(
        1, math.ceil(config.policy_transition_seconds / adapter.control_timestep)
    )
    for step_number in range(1, steps + 1):
        frame = control.read_frame(
            max_tilt_radians=(
                config.max_motion_tilt_radians
                if max_tilt_radians is None
                else max_tilt_radians
            )
        )
        current_reference = (
            first_reference
            if step_number == 1 and first_reference is not None
            else reference()
        )
        policy_step = adapter.step(frame, current_reference)
        fraction = step_number / steps
        target = initial_target + fraction * (
            policy_step.physical_target - initial_target
        )
        control.send_target(target)
        if policy_log is not None:
            policy_log.write_step(
                phase=phase,
                step=step_number,
                elapsed_seconds=time.monotonic() - run_started,
                observation=policy_step.observation,
                policy_observation=policy_step.policy_observation,
                raw_action=policy_step.raw_action,
                action=policy_step.action,
                physical_target=target,
                policy_target=policy_step.physical_target,
                command=policy_step.command,
            )
        control.pace()
    control.drain()


def _wait_for_live_twist2_reference(
    references: Twist2ReferenceSource,
    adapter: HardwareTwist2PolicyAdapter,
    config: HardwareTwist2Config,
    standing_step: Callable[[], bool],
) -> Twist2ReferenceFrame | None:
    deadline = time.monotonic() + config.fastsam_reference_ready_timeout_seconds
    last_error: ReferenceUnavailable | None = None
    last_sequence: int | None = None
    qualified_frames = 0
    while time.monotonic() < deadline:
        try:
            reference = references.latest()
            validate_twist2_reference(reference, adapter.env, config)
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
            last_sequence = None
            qualified_frames = 0
        if not standing_step():
            return None
    if last_error is not None:
        print(f"Live SMPL-X target is still unavailable: {last_error}")
    return None


def _log_twist2_step(
    policy_log: PolicyStepLogger | None,
    *,
    phase: str,
    step_number: int,
    run_started: float,
    step: HardwareTwist2Step,
) -> None:
    if policy_log is None:
        return
    policy_log.write_step(
        phase=phase,
        step=step_number,
        elapsed_seconds=time.monotonic() - run_started,
        observation=step.observation,
        policy_observation=step.policy_observation,
        raw_action=step.raw_action,
        action=step.action,
        physical_target=step.physical_target,
        command=step.command,
    )
