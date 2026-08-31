"""Deploy a typed SONIC checkpoint from a clip or live GEM-X reference."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Literal, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from bello_mujoco.actuators import joint_position_action_scales
from bello_mujoco.contracts import ActionLayout, ObservationLayout
from bello_mujoco.envs.sonic import (
    SONIC_ACTION_CLIP,
    SONIC_HISTORY_FRAMES,
    SONIC_POLICY_HZ,
    SONIC_SIMULATION_HZ,
    SonicEnvConfig,
    SonicObservationContract,
)
from bello_mujoco.hardware.config import HardwareControlConfig
from bello_mujoco.hardware.locomotion import (
    HardwarePolicyAdapter as HardwareLocomotionPolicyAdapter,
    load_hardware_policy,
    validate_command,
)
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
    projected_gravity_from_xyzw,
    require_async_hardware_binding,
    require_nominal_pose,
    require_operator_confirmation,
    require_state_frame_binding,
    validate_hardware_control_config,
    validate_hardware_frame,
    validate_hardware_inventory,
    wait_for_hardware_frame,
    wait_for_operator_confirmation_while_holding,
)
from bello_mujoco.hardware.targets import (
    ANKLE_PROJECTION_TOLERANCE,
    AnkleTargetEnvelope,
    ankle_feasible_polygon,
    project_ankle_motor_target,
)
from bello_mujoco.hardware.twist2 import ReferenceUnavailable, Twist2ReferenceFrame
from bello_mujoco.motion.clip import MotionClip, load_motion_clip
from bello_mujoco.motion.sonic_dataset import (
    SONIC_CRITIC_FUTURE_OFFSETS,
)
from bello_mujoco.policy import (
    init_actor_critic,
    sonic_frozen_parameter_hashes,
)
from bello_mujoco.sim import BelloModel, load_bello_model
from bello_mujoco.sonic_policy import (
    BELLO_TO_G1_VECTOR_BASIS,
    SONIC_ACTOR_OBSERVATION_SIZE,
    SONIC_BELLO_ACTION_SIZE,
    SONIC_BELLO_PROPRIO_SIZE,
    SONIC_G1_ACTION_SIZE,
    SONIC_G1_DEFAULT_JOINT_POSITION,
    SONIC_OFFICIAL_CHECKPOINT_SHA256,
    SONIC_ROBOT_INPUT_SIZE,
    build_bello_decoder_initialization,
    load_sonic_release_artifact,
    semantic_g1_from_bello_joint_map,
)
from bello_mujoco.training.checkpoint import (
    CheckpointNormalizationState,
    SonicCheckpointMetadata,
    load_checkpoint_metadata,
    restore_evaluation_checkpoint,
)
from bello_mujoco.training.ppo import normalize_observation


SonicReferenceMode = Literal["clip", "live"]
_HARDWARE_LIMIT_ROUNDING_TOLERANCE_RADIANS = 1.0e-5


@dataclass(frozen=True)
class HardwareSonicConfig(HardwareControlConfig):
    """Physical SONIC deployment and explicit reference-source contract."""

    client_name: str = "bello-sonic-policy"
    reference_mode: SonicReferenceMode = "clip"
    release_artifact_directory: Path = Path("artifacts/sonic/default_release")
    motion_path: Path | None = None
    max_reference_age_ms: int = 250
    max_reference_root_height_m: float = 1.5
    min_reference_root_height_m: float = 0.35
    max_reference_root_speed_mps: float = 5.0
    max_reference_angular_speed_rps: float = 12.0
    max_reference_joint_speed_rps: float = 25.0
    max_reference_joint_limit_violation_radians: float = 0.01
    max_raw_source_action_abs: float = 100.0
    max_motion_tilt_radians: float = math.radians(70.0)
    policy_transition_seconds: float = 0.2
    gemx_endpoint: str = "127.0.0.1:50056"
    gemx_connect_timeout_seconds: float = 5.0
    gemx_rpc_timeout_seconds: float = 2.0
    gemx_reference_ready_timeout_seconds: float = 15.0
    gemx_reference_qualification_frames: int = 5
    gemx_min_tracking_confidence: float = 0.5
    gemx_min_joint_confidence: float = 0.05
    gemx_full_joint_confidence: float = 0.25
    gemx_max_pipeline_latency_ms: float = 220.0
    gemx_transport_latency_budget_ms: float = 10.0
    gemx_camera_to_world_quaternion_xyzw: tuple[float, float, float, float] = (
        0.5,
        -0.5,
        0.5,
        -0.5,
    )
    gemx_align_initial_heading: bool = True
    gemx_align_initial_floor: bool = True
    smplx_neutral_model_path: Path | None = None
    smplx_human_height_m: float = 1.66


@dataclass(frozen=True)
class SonicReferenceWindow:
    """Ten source frames required by the frozen SONIC robot encoder."""

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

    @property
    def current(self) -> Twist2ReferenceFrame:
        return Twist2ReferenceFrame(
            qpos=self.qpos[0],
            qvel=self.qvel[0],
            sequence=self.sequence,
            source_age_ms=self.source_age_ms,
            received_at=self.received_at,
        )


class SonicReferenceSource(Protocol):
    mode: SonicReferenceMode

    def standing(self) -> SonicReferenceWindow: ...

    def latest(self) -> SonicReferenceWindow: ...

    def reset(self) -> None: ...

    def advance(self) -> bool: ...


@dataclass(frozen=True)
class HardwareSonicEnvironment:
    """Native-model subset of BelloSonicEnv needed for actor deployment."""

    model: BelloModel
    config: SonicEnvConfig
    contract: SonicObservationContract
    action_scale: tuple[float, ...]
    g1_from_bello_joint: np.ndarray

    @property
    def action_size(self) -> int:
        return int(self.model.policy_ctrl_indices.size)

    @property
    def action_clip(self) -> float:
        return SONIC_ACTION_CLIP

    @property
    def control_timestep(self) -> float:
        return 1.0 / SONIC_POLICY_HZ

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
class HardwareSonicStep:
    action: np.ndarray
    raw_source_action: np.ndarray
    physical_target: np.ndarray
    observation: np.ndarray
    policy_observation: np.ndarray
    command: np.ndarray
    action_clipped: bool
    reference_sequence: int
    reference_age_ms: float | None


class HardwareSonicPolicyAdapter:
    """Reconstruct SONIC inputs and map its actions to bounded motor targets."""

    def __init__(
        self,
        env: HardwareSonicEnvironment,
        policy: Any,
        metadata: SonicCheckpointMetadata,
        normalization: CheckpointNormalizationState,
        config: HardwareSonicConfig,
    ) -> None:
        validate_hardware_sonic_checkpoint(env, metadata)
        validate_hardware_sonic_config(config)
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
            raise ValueError("SONIC hardware requires a square coordinate transform")
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
        self.action_scale = np.asarray(env.action_scale, dtype=np.float64)
        self.kp, self.kd = actuator_impedance_gains(env)
        self._model_ctrl_limits = np.asarray(
            env.model.mj_model.actuator_ctrlrange, dtype=np.float64
        ).copy()
        if self._model_ctrl_limits.shape != (coordinate_count, 2):
            raise ValueError("SONIC actuator control ranges have an invalid shape")
        self._ctrl_limits = self._model_ctrl_limits.copy()
        self._ankle_target_envelopes = self._build_ankle_target_envelopes()
        for envelope in self._ankle_target_envelopes:
            ankle_feasible_polygon(envelope, self._model_ctrl_limits)
        observation_norm = normalization.observation

        @eqx.filter_jit
        def infer(
            observation: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            policy_observation = normalize_observation(
                observation, observation_norm, metadata.ppo
            )
            raw_source_action = policy.actor(policy_observation)
            clipped_source_action = jnp.clip(
                raw_source_action, -policy.action_limit, policy.action_limit
            )
            action = policy.actor.decoder.bello_action(clipped_source_action)
            return policy_observation, raw_source_action, action

        self._infer = infer
        self._history: np.ndarray | None = None
        self._last_action = np.zeros(env.action_size, dtype=np.float32)
        self._reference_heading: np.ndarray | None = None

    @property
    def control_timestep(self) -> float:
        return self.env.control_timestep

    @property
    def nominal_physical_target(self) -> np.ndarray:
        return self._bounded_physical_target(self.transform @ self.default_joint_target)

    def _build_ankle_target_envelopes(self) -> tuple[AnkleTargetEnvelope, ...]:
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
            joint_ids = self.env.model.actuator_joint_ids[
                np.asarray(logical_indices, dtype=np.int32)
            ]
            if not np.all(self.env.model.mj_model.jnt_limited[joint_ids]):
                raise ValueError(f"{side} ankle joints require hard URDF limits")
            joint_range = np.asarray(
                self.env.model.mj_model.jnt_range[joint_ids], dtype=np.float64
            )
            envelopes.append(
                AnkleTargetEnvelope(
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
            target[indices] = project_ankle_motor_target(
                target[indices], envelope, self._ctrl_limits
            )
        return target

    def _validate_ankle_target(self, physical_target: np.ndarray) -> None:
        for envelope in self._ankle_target_envelopes:
            indices = np.asarray(envelope.motor_indices, dtype=np.int32)
            resolved = envelope.motor_to_joint @ physical_target[indices]
            violation = np.maximum(
                envelope.joint_low - resolved, resolved - envelope.joint_high
            )
            if float(np.max(violation)) > ANKLE_PROJECTION_TOLERANCE:
                raise RuntimeError(
                    f"projected SONIC {envelope.side} ankle target is unsafe"
                )

    def reference_physical_target(self, reference: SonicReferenceWindow) -> np.ndarray:
        self.validate_reference(reference)
        logical = self.default_joint_target.copy()
        logical[self.policy_indices] = reference.qpos[0, self.policy_qpos_indices]
        return self._bounded_physical_target(self.transform @ logical)

    def reset(self) -> None:
        self._history = None
        self._last_action = np.zeros(self.env.action_size, dtype=np.float32)
        self._reference_heading = None

    def precompile(self, reference: SonicReferenceWindow) -> None:
        nominal = self.reference_physical_target(reference)
        root = self.env.model.root_qpos_adr
        orientation_wxyz = reference.qpos[0, root + 3 : root + 7]
        frame = HardwareFrame(
            positions=nominal,
            velocities=np.zeros_like(nominal),
            orientation_xyzw=orientation_wxyz[[1, 2, 3, 0]],
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
        self.step(frame, reference)
        self.reset()

    def validate_reference_frame(self, reference: Twist2ReferenceFrame) -> None:
        validate_sonic_reference_frame(reference, self.env, self.config)

    def validate_reference(self, reference: SonicReferenceWindow) -> None:
        validate_sonic_reference_window(reference, self.env, self.config)

    def _proprioception_frame(self, frame: HardwareFrame) -> np.ndarray:
        logical_position = self.inverse_transform @ frame.positions
        logical_velocity = self.inverse_transform @ frame.velocities
        qpos = (
            logical_position[self.policy_indices]
            - self.default_joint_target[self.policy_indices]
        )
        qvel = logical_velocity[self.policy_indices]
        value = np.concatenate(
            (
                projected_gravity_from_xyzw(frame.orientation_xyzw),
                np.zeros(3, dtype=np.float64),
                np.asarray(frame.angular_velocity, dtype=np.float64),
                qpos,
                qvel,
                self._last_action,
            )
        ).astype(np.float32)
        if value.shape != (self.env.contract.history_frame_size,):
            raise RuntimeError("SONIC hardware history-frame layout has drifted")
        if not np.all(np.isfinite(value)):
            raise RuntimeError("SONIC hardware proprioception is non-finite")
        return value

    def _proprioception(self, history: np.ndarray) -> np.ndarray:
        gravity = history[:, :3]
        angular = history[:, 6:9]
        qpos = history[:, 9 : 9 + self.env.action_size]
        qvel = history[:, 9 + self.env.action_size : 9 + 2 * self.env.action_size]
        action = history[:, 9 + 2 * self.env.action_size :]
        value = np.concatenate(
            (
                gravity.reshape(-1),
                angular.reshape(-1),
                qpos.reshape(-1),
                qvel.reshape(-1),
                action.reshape(-1),
            )
        ).astype(np.float32)
        if value.shape != (SONIC_BELLO_PROPRIO_SIZE,):
            raise RuntimeError("SONIC hardware proprioception packing has drifted")
        return value

    def _robot_encoder_input(
        self, frame: HardwareFrame, reference: SonicReferenceWindow
    ) -> np.ndarray:
        current_wxyz = normalized_quaternion_xyzw(frame.orientation_xyzw)[[3, 0, 1, 2]]
        current_rotation = _quaternion_rotation_matrix(current_wxyz)
        if self._reference_heading is None:
            root = self.env.model.root_qpos_adr
            reference_yaw = _yaw_wxyz(reference.qpos[0, root + 3 : root + 7])
            current_yaw = _yaw_wxyz(current_wxyz)
            delta = current_yaw - reference_yaw
            self._reference_heading = np.asarray(
                (
                    (math.cos(delta), -math.sin(delta), 0.0),
                    (math.sin(delta), math.cos(delta), 0.0),
                    (0.0, 0.0, 1.0),
                ),
                dtype=np.float64,
            )
        basis = np.asarray(BELLO_TO_G1_VECTOR_BASIS, dtype=np.float64)
        joint_map = np.asarray(self.env.g1_from_bello_joint, dtype=np.float64)
        bello_default = self.default_joint_target[self.policy_indices]
        g1_default = np.asarray(SONIC_G1_DEFAULT_JOINT_POSITION, dtype=np.float64)
        g1_joint_positions = np.stack(
            [
                g1_default
                + joint_map @ (qpos[self.policy_qpos_indices] - bello_default)
                for qpos in reference.qpos
            ]
        )
        g1_joint_velocities = np.stack(
            [joint_map @ qvel[self.policy_qvel_indices] for qvel in reference.qvel]
        )
        orientations = np.stack(
            [
                (
                    basis
                    @ current_rotation.T
                    @ self._reference_heading
                    @ _quaternion_rotation_matrix(qpos[3:7])
                    @ basis.T
                )[:, :2].reshape(-1)
                for qpos in reference.qpos
            ]
        )
        command = np.concatenate(
            (g1_joint_positions.reshape(-1), g1_joint_velocities.reshape(-1))
        ).reshape((len(SONIC_CRITIC_FUTURE_OFFSETS), 2 * SONIC_G1_ACTION_SIZE))
        value = np.concatenate((command, orientations), axis=-1).reshape(-1)
        if value.shape != (SONIC_ROBOT_INPUT_SIZE,):
            raise RuntimeError("SONIC hardware robot-input packing has drifted")
        return value.astype(np.float32)

    def observation(
        self, frame: HardwareFrame, reference: SonicReferenceWindow
    ) -> tuple[np.ndarray, np.ndarray]:
        self.validate_reference(reference)
        current = self._proprioception_frame(frame)
        if self._history is None:
            self._history = np.repeat(current[None, :], SONIC_HISTORY_FRAMES, axis=0)
        else:
            self._history = np.concatenate(
                (self._history[1:], current[None, :]), axis=0
            )
        command = self._robot_encoder_input(frame, reference)
        observation = np.concatenate(
            (command, self._proprioception(self._history))
        ).astype(np.float32)
        if observation.shape != (SONIC_ACTOR_OBSERVATION_SIZE,):
            raise RuntimeError("SONIC hardware actor observation shape has drifted")
        return observation, command

    def step(
        self, frame: HardwareFrame, reference: SonicReferenceWindow
    ) -> HardwareSonicStep:
        observation, command = self.observation(frame, reference)
        policy_observation, raw_source_action, action = self._infer(
            jnp.asarray(observation)
        )
        policy_observation, raw_source_action, action = jax.device_get(
            (policy_observation, raw_source_action, action)
        )
        policy_observation = np.asarray(policy_observation, dtype=np.float32)
        raw_source_action = np.asarray(raw_source_action, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        if raw_source_action.shape != (SONIC_G1_ACTION_SIZE,) or not np.all(
            np.isfinite(raw_source_action)
        ):
            raise RuntimeError("SONIC policy produced an invalid G1 source action")
        source_abs = float(np.max(np.abs(raw_source_action)))
        if source_abs > self.config.max_raw_source_action_abs:
            raise RuntimeError(
                f"raw SONIC source action {source_abs:.3f} exceeds fail-closed "
                f"limit {self.config.max_raw_source_action_abs:.3f}"
            )
        if action.shape != (self.env.action_size,) or not np.all(np.isfinite(action)):
            raise RuntimeError("SONIC policy produced an invalid Bello action")
        requested_logical = self.default_joint_target.copy()
        requested_logical[self.policy_indices] += self.action_scale * action
        requested_target = self.transform @ requested_logical
        physical_target = self._bounded_physical_target(requested_target)
        if not np.all(np.isfinite(physical_target)):
            raise RuntimeError("SONIC policy produced non-finite physical targets")
        self._validate_ankle_target(physical_target)
        clipped_source = np.clip(
            raw_source_action, -SONIC_ACTION_CLIP, SONIC_ACTION_CLIP
        )
        action_clipped = not np.array_equal(
            raw_source_action, clipped_source
        ) or not np.array_equal(physical_target, requested_target)
        self._last_action = action.copy()
        return HardwareSonicStep(
            action=action,
            raw_source_action=raw_source_action,
            physical_target=physical_target,
            observation=observation,
            policy_observation=policy_observation,
            command=command,
            action_clipped=action_clipped,
            reference_sequence=reference.sequence,
            reference_age_ms=reference.age_ms(),
        )


class SonicClipReferenceSource:
    """Deterministic 50 Hz SONIC future windows from one runtime motion clip."""

    mode: SonicReferenceMode = "clip"

    def __init__(
        self,
        clip: MotionClip,
        env: HardwareSonicEnvironment,
        config: HardwareSonicConfig,
    ) -> None:
        self.clip = clip
        self.env = env
        self.config = config
        if not math.isclose(clip.fps, SONIC_POLICY_HZ, abs_tol=1.0e-6):
            raise ValueError("SONIC hardware clips must be sampled at exactly 50 Hz")
        if clip.joint_names != env.action_layout.names:
            raise ValueError("SONIC hardware clip joint order differs from checkpoint")
        if clip.num_frames <= max(SONIC_CRITIC_FUTURE_OFFSETS):
            raise ValueError("SONIC hardware clip does not cover the future horizon")
        self._frame = 0

    def standing(self) -> SonicReferenceWindow:
        return _canonical_standing_window(self.env)

    def latest(self) -> SonicReferenceWindow:
        return self._window(self._frame)

    def reset(self) -> None:
        self._frame = 0

    def advance(self) -> bool:
        if self._frame + 1 >= self.clip.num_frames:
            return False
        self._frame += 1
        return True

    def _window(self, frame: int) -> SonicReferenceWindow:
        indices = np.minimum(
            frame + np.asarray(SONIC_CRITIC_FUTURE_OFFSETS),
            self.clip.num_frames - 1,
        )
        return SonicReferenceWindow(
            qpos=np.asarray(self.clip.qpos)[indices].copy(),
            qvel=np.asarray(self.clip.qvel)[indices].copy(),
            sequence=frame,
            source_age_ms=None,
            received_at=time.monotonic(),
        )


class SonicLiveReferenceSource:
    """Causally resample latest-only live GMR output at the policy rate."""

    mode: SonicReferenceMode = "live"

    def __init__(
        self,
        source: Any,
        env: HardwareSonicEnvironment,
        config: HardwareSonicConfig,
    ) -> None:
        self.source = source
        self.env = env
        self.config = config

    def standing(self) -> SonicReferenceWindow:
        return _canonical_standing_window(self.env)

    def latest(self) -> SonicReferenceWindow:
        return self._predict(self.source.latest())

    def reset(self) -> None:
        return None

    def advance(self) -> bool:
        return True

    def _predict(self, reference: Twist2ReferenceFrame) -> SonicReferenceWindow:
        validate_sonic_reference_frame(reference, self.env, self.config)
        age_ms = reference.age_ms()
        latency_seconds = 0.0 if age_ms is None else age_ms / 1_000.0
        qpos_frames = []
        qvel_frames = []
        for offset in SONIC_CRITIC_FUTURE_OFFSETS:
            qpos = np.asarray(reference.qpos, dtype=np.float64).copy()
            prediction_seconds = latency_seconds + offset / SONIC_POLICY_HZ
            if prediction_seconds:
                root = self.env.model.root_qpos_adr
                root_position = qpos[root : root + 3].copy()
                mujoco.mj_integratePos(
                    self.env.model.mj_model,
                    qpos,
                    np.asarray(reference.qvel, dtype=np.float64),
                    prediction_seconds,
                )
                qpos[root : root + 3] = root_position
                _clip_reference_joints(qpos, self.env)
            qpos_frames.append(qpos)
            qvel_frames.append(np.asarray(reference.qvel, dtype=np.float64).copy())
        return SonicReferenceWindow(
            qpos=np.stack(qpos_frames),
            qvel=np.stack(qvel_frames),
            sequence=reference.sequence,
            source_age_ms=reference.source_age_ms,
            received_at=reference.received_at,
        )


def validate_hardware_sonic_checkpoint(
    env: HardwareSonicEnvironment,
    metadata: SonicCheckpointMetadata,
) -> None:
    if metadata.task != "sonic":
        raise ValueError("SONIC hardware requires a typed sonic checkpoint")
    if metadata.ppo.policy.architecture != "sonic":
        raise ValueError("SONIC hardware requires the sonic actor architecture")
    if metadata.ppo.policy.action_distribution != "gaussian":
        raise ValueError("SONIC hardware requires the released Gaussian actor")
    if metadata.ppo.policy.obs_stack_size != 1:
        raise ValueError("SONIC hardware requires PPO observation stack size one")
    if env.action_size != SONIC_BELLO_ACTION_SIZE or env.model.mj_model.nu != 25:
        raise ValueError("SONIC hardware requires 25 policy and physical coordinates")
    if env.contract != metadata.observation_contract:
        raise ValueError("SONIC hardware observation contract differs from checkpoint")
    if env.observation_layout != metadata.actor_observation_layout:
        raise ValueError("SONIC hardware actor layout differs from checkpoint")
    if env.action_layout != metadata.action_layout:
        raise ValueError("SONIC hardware action layout differs from checkpoint")
    if env.observation_size != metadata.obs_size:
        raise ValueError("SONIC hardware observation size differs from checkpoint")


def validate_hardware_sonic_config(config: HardwareSonicConfig) -> None:
    validate_hardware_control_config(config)
    if config.reference_mode not in {"clip", "live"}:
        raise ValueError("SONIC reference_mode must be 'clip' or 'live'")
    if config.reference_mode == "clip" and config.motion_path is None:
        raise ValueError("motion_path is required for SONIC reference_mode=clip")
    if config.reference_mode == "live" and config.motion_path is not None:
        raise ValueError("motion_path is not valid for SONIC reference_mode=live")
    if config.reference_mode == "live" and config.smplx_neutral_model_path is None:
        raise ValueError(
            "smplx_neutral_model_path is required for SONIC reference_mode=live"
        )
    positive = {
        "max_reference_root_height_m": config.max_reference_root_height_m,
        "min_reference_root_height_m": config.min_reference_root_height_m,
        "max_reference_root_speed_mps": config.max_reference_root_speed_mps,
        "max_reference_angular_speed_rps": config.max_reference_angular_speed_rps,
        "max_reference_joint_speed_rps": config.max_reference_joint_speed_rps,
        "max_raw_source_action_abs": config.max_raw_source_action_abs,
        "max_motion_tilt_radians": config.max_motion_tilt_radians,
        "policy_transition_seconds": config.policy_transition_seconds,
        "gemx_connect_timeout_seconds": config.gemx_connect_timeout_seconds,
        "gemx_rpc_timeout_seconds": config.gemx_rpc_timeout_seconds,
        "gemx_reference_ready_timeout_seconds": (
            config.gemx_reference_ready_timeout_seconds
        ),
        "gemx_max_pipeline_latency_ms": config.gemx_max_pipeline_latency_ms,
        "smplx_human_height_m": config.smplx_human_height_m,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if config.max_raw_source_action_abs < SONIC_ACTION_CLIP:
        raise ValueError("max_raw_source_action_abs must cover the SONIC action clip")
    if config.max_reference_age_ms <= 0:
        raise ValueError("max_reference_age_ms must be positive")
    if config.gemx_reference_qualification_frames < 1:
        raise ValueError("gemx_reference_qualification_frames must be positive")
    if config.min_reference_root_height_m >= config.max_reference_root_height_m:
        raise ValueError("reference root-height bounds must be strictly ordered")
    if not config.gemx_endpoint or "://" in config.gemx_endpoint:
        raise ValueError("gemx_endpoint must be a non-empty gRPC host:port target")
    minimum_age_budget = (
        config.gemx_max_pipeline_latency_ms + config.gemx_transport_latency_budget_ms
    )
    if config.max_reference_age_ms <= minimum_age_budget:
        raise ValueError(
            "max_reference_age_ms must exceed the GEM-X pipeline and transport budget"
        )
    if (
        not math.isfinite(config.max_reference_joint_limit_violation_radians)
        or config.max_reference_joint_limit_violation_radians < 0.0
    ):
        raise ValueError(
            "max_reference_joint_limit_violation_radians must be finite and nonnegative"
        )
    for name, value in (
        ("gemx_min_tracking_confidence", config.gemx_min_tracking_confidence),
        ("gemx_min_joint_confidence", config.gemx_min_joint_confidence),
        ("gemx_full_joint_confidence", config.gemx_full_joint_confidence),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    if config.gemx_min_joint_confidence > config.gemx_full_joint_confidence:
        raise ValueError(
            "gemx_min_joint_confidence must not exceed "
            "gemx_full_joint_confidence"
        )
    if (
        not math.isfinite(config.gemx_transport_latency_budget_ms)
        or config.gemx_transport_latency_budget_ms < 0.0
    ):
        raise ValueError(
            "gemx_transport_latency_budget_ms must be finite and nonnegative"
        )
    camera = np.asarray(config.gemx_camera_to_world_quaternion_xyzw)
    if camera.shape != (4,) or not np.all(np.isfinite(camera)):
        raise ValueError(
            "gemx_camera_to_world_quaternion_xyzw must contain four finite values"
        )
    if not 0.999 <= float(np.linalg.norm(camera)) <= 1.001:
        raise ValueError("gemx_camera_to_world_quaternion_xyzw must be normalized")


def validate_sonic_reference_window(
    reference: SonicReferenceWindow,
    env: HardwareSonicEnvironment,
    config: HardwareSonicConfig,
) -> None:
    qpos = np.asarray(reference.qpos)
    qvel = np.asarray(reference.qvel)
    frame_count = len(SONIC_CRITIC_FUTURE_OFFSETS)
    if qpos.shape != (frame_count, env.model.mj_model.nq):
        raise ReferenceUnavailable("SONIC reference window qpos shape is invalid")
    if qvel.shape != (frame_count, env.model.mj_model.nv):
        raise ReferenceUnavailable("SONIC reference window qvel shape is invalid")
    if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(qvel)):
        raise ReferenceUnavailable("SONIC reference window contains non-finite values")
    for index in range(frame_count):
        validate_sonic_reference_frame(
            Twist2ReferenceFrame(
                qpos=qpos[index],
                qvel=qvel[index],
                sequence=reference.sequence,
                source_age_ms=reference.source_age_ms,
                received_at=reference.received_at,
            ),
            env,
            config,
            validate_age=index == 0,
        )


def validate_sonic_reference_frame(
    reference: Twist2ReferenceFrame,
    env: HardwareSonicEnvironment,
    config: HardwareSonicConfig,
    *,
    validate_age: bool = True,
) -> None:
    qpos = np.asarray(reference.qpos)
    qvel = np.asarray(reference.qvel)
    if qpos.shape != (env.model.mj_model.nq,) or not np.all(np.isfinite(qpos)):
        raise ReferenceUnavailable(
            f"SONIC reference qpos must contain {env.model.mj_model.nq} finite values"
        )
    if qvel.shape != (env.model.mj_model.nv,) or not np.all(np.isfinite(qvel)):
        raise ReferenceUnavailable(
            f"SONIC reference qvel must contain {env.model.mj_model.nv} finite values"
        )
    if reference.sequence < 0:
        raise ReferenceUnavailable("SONIC reference sequence must be nonnegative")
    if reference.source_age_ms is not None and reference.source_age_ms < 0:
        raise ReferenceUnavailable("SONIC reference source age must be nonnegative")
    age = reference.age_ms()
    if validate_age and age is not None and age > config.max_reference_age_ms:
        raise ReferenceUnavailable(
            f"SONIC reference is stale: {age:.1f} ms exceeds "
            f"{config.max_reference_age_ms} ms"
        )
    root = env.model.root_qpos_adr
    root_velocity = env.model.root_qvel_adr
    root_height = float(qpos[root + 2])
    if (
        not config.min_reference_root_height_m
        <= root_height
        <= config.max_reference_root_height_m
    ):
        raise ReferenceUnavailable(
            f"SONIC reference root height is unsafe: {root_height:.3f} m"
        )
    quaternion = qpos[root + 3 : root + 7]
    if not 0.999 <= float(np.linalg.norm(quaternion)) <= 1.001:
        raise ReferenceUnavailable("SONIC reference root quaternion norm is invalid")
    root_speed = float(np.linalg.norm(qvel[root_velocity : root_velocity + 3]))
    if root_speed > config.max_reference_root_speed_mps:
        raise ReferenceUnavailable(
            f"SONIC reference root speed is unsafe: {root_speed:.3f} m/s"
        )
    angular_speed = float(np.linalg.norm(qvel[root_velocity + 3 : root_velocity + 6]))
    if angular_speed > config.max_reference_angular_speed_rps:
        raise ReferenceUnavailable(
            f"SONIC reference angular speed is unsafe: {angular_speed:.3f} rad/s"
        )
    joint_speed = float(np.max(np.abs(qvel[env.model.policy_qvel_indices])))
    if joint_speed > config.max_reference_joint_speed_rps:
        raise ReferenceUnavailable(
            f"SONIC reference joint speed is unsafe: {joint_speed:.3f} rad/s"
        )
    policy = np.asarray(env.model.policy_ctrl_indices)
    positions = qpos[np.asarray(env.model.policy_qpos_indices)]
    joint_ids = np.asarray(env.model.actuator_joint_ids)[policy]
    ranges = np.asarray(env.model.mj_model.jnt_range)[joint_ids]
    violation = np.maximum(ranges[:, 0] - positions, positions - ranges[:, 1])
    index = int(np.argmax(violation))
    if violation[index] > config.max_reference_joint_limit_violation_radians:
        name = env.model.actuator_joint_names[int(policy[index])]
        raise ReferenceUnavailable(
            f"SONIC reference joint {name!r} exceeds its limit by "
            f"{violation[index]:.3f} rad"
        )


def bind_sonic_hardware_coordinate_limits(
    groups: dict[str, Any], adapter: HardwareSonicPolicyAdapter
) -> None:
    motors = {
        motor.coordinate: motor for group in groups.values() for motor in group.motors
    }
    effective = adapter._model_ctrl_limits.copy()
    for index, coordinate in enumerate(adapter.physical_coordinate_names):
        motor = motors[coordinate]
        model_low, model_high = adapter._model_ctrl_limits[index]
        hardware_low = float(motor.limits.position_min)
        hardware_high = float(motor.limits.position_max)
        if (
            hardware_low - model_low > _HARDWARE_LIMIT_ROUNDING_TOLERANCE_RADIANS
            or model_high - hardware_high > _HARDWARE_LIMIT_ROUNDING_TOLERANCE_RADIANS
        ):
            raise ValueError(
                f"SONIC model range for {coordinate!r} exceeds its hardware range"
            )
        effective[index] = (
            max(model_low, hardware_low),
            min(model_high, hardware_high),
        )
        if effective[index, 0] >= effective[index, 1]:
            raise ValueError(
                f"SONIC model and hardware ranges for {coordinate!r} do not overlap"
            )
    for envelope in adapter._ankle_target_envelopes:
        ankle_feasible_polygon(envelope, effective)
    adapter._ctrl_limits = effective


def load_hardware_sonic_policy(
    checkpoint_path: Path,
    config: HardwareSonicConfig,
) -> tuple[HardwareSonicPolicyAdapter, SonicCheckpointMetadata]:
    metadata = load_checkpoint_metadata(
        checkpoint_path,
        allow_git_commit_mismatch=config.allow_git_commit_mismatch,
    )
    if not isinstance(metadata, SonicCheckpointMetadata):
        raise ValueError("SONIC hardware requires a typed sonic checkpoint")
    release = load_sonic_release_artifact(config.release_artifact_directory)
    if release.checkpoint_sha256 != SONIC_OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError("SONIC hardware requires NVIDIA's pinned release checkpoint")
    if (
        "sha256:" + release.checkpoint_sha256.removeprefix("sha256:")
        != metadata.release_checkpoint_sha256
        or "sha256:" + release.artifact_sha256.removeprefix("sha256:")
        != metadata.release_artifact_sha256
    ):
        raise ValueError("SONIC release artifact differs from checkpoint metadata")
    model = load_bello_model(
        metadata.asset,
        warp=metadata.warp,
        batch_size=1,
        terrain=metadata.simulation.terrain,
        load_mjx=False,
        simulation_timestep=1.0 / SONIC_SIMULATION_HZ,
    )
    joint_names = tuple(
        model.actuator_joint_names[index] for index in model.policy_ctrl_indices
    )
    contract = SonicObservationContract(joint_names)
    action_scale = joint_position_action_scales(
        metadata.asset.actuator_config, joint_names
    )
    joint_map = semantic_g1_from_bello_joint_map(metadata.dataset.calibration)
    env = HardwareSonicEnvironment(
        model, metadata.sonic_env, contract, action_scale, joint_map
    )
    initialization = build_bello_decoder_initialization(
        release,
        metadata.dataset.calibration,
        bello_joint_names=env.action_layout.names,
        bello_action_scale=env.action_scale,
        bello_default_joint_position=np.asarray(model.default_joint_target)[
            np.asarray(model.policy_ctrl_indices)
        ],
    )
    _, policy_key = jax.random.split(jax.random.PRNGKey(metadata.ppo.seed))
    policy_template = init_actor_critic(
        policy_key,
        metadata.obs_size,
        metadata.action_size,
        metadata.ppo.policy,
        critic_obs_size=metadata.critic_obs_size,
        sonic_release=release,
        sonic_initialization=initialization,
    )
    expected_frozen = sonic_frozen_parameter_hashes(policy_template)
    train_state, restored_metadata, normalization = restore_evaluation_checkpoint(
        checkpoint_path,
        allow_git_commit_mismatch=config.allow_git_commit_mismatch,
        policy=policy_template,
    )
    if restored_metadata != metadata:
        raise RuntimeError("SONIC checkpoint metadata changed during restore")
    if sonic_frozen_parameter_hashes(train_state.policy) != expected_frozen:
        raise ValueError("SONIC checkpoint changed frozen release parameters")
    return (
        HardwareSonicPolicyAdapter(
            env, train_state.policy, metadata, normalization, config
        ),
        metadata,
    )


def run_hardware_sonic(
    checkpoint_path: Path,
    standing_checkpoint_path: Path,
    config: HardwareSonicConfig,
    *,
    reference_source_factory: Callable[
        [HardwareSonicPolicyAdapter, HardwareSonicConfig], Any
    ]
    | None = None,
    hardware_binding: Any | None = None,
    operator_input: Callable[[str], str] | None = None,
) -> None:
    """Run SONIC against an indexed clip or the pinned GEM-X SOMA stream."""

    validate_hardware_sonic_config(config)
    adapter, metadata = load_hardware_sonic_policy(checkpoint_path, config)
    standing_adapter, standing_metadata = load_hardware_policy(
        standing_checkpoint_path,
        allow_git_commit_mismatch=config.allow_git_commit_mismatch,
    )
    validate_sonic_transition_adapters(standing_adapter, adapter)
    standing_command = np.zeros(3, dtype=np.float32)
    validate_command(standing_command, standing_metadata)
    if reference_source_factory is None:
        reference_source_factory = (
            _clip_reference_context
            if config.reference_mode == "clip"
            else _live_reference_context
        )
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
        if references.mode != config.reference_mode:
            raise ValueError("SONIC reference source mode differs from configuration")
        standing = references.standing()
        adapter.validate_reference(standing)
        groups = validate_hardware_inventory(
            robot, standing_adapter, config, binding
        )
        validate_hardware_inventory(robot, adapter, config, binding)
        bind_sonic_hardware_coordinate_limits(groups, adapter)
        adapter.precompile(standing)
        standing_adapter.precompile()
        state_reader = robot.states(groups=config.groups, imus=(config.imu_name,))
        frame = wait_for_hardware_frame(
            state_reader,
            standing_adapter.physical_coordinate_names,
            config.groups,
            config.imu_name,
            config.telemetry_timeout_seconds,
        )
        validate_hardware_frame(frame, config, expected_group_state="IDLE")
        initial_error = float(
            np.max(
                np.abs(frame.positions - standing_adapter.nominal_physical_target)
            )
        )
        print(
            {
                "sonic_hardware_preflight": "passed",
                "checkpoint_task": metadata.task,
                "standing_checkpoint_task": standing_metadata.task,
                "reference_mode": references.mode,
                "endpoint": config.endpoint,
                "groups": config.groups,
                "coordinates": len(adapter.physical_coordinate_names),
                "policy_joints": adapter.env.action_size,
                "source_action_size": SONIC_G1_ACTION_SIZE,
                "actor_observation_size": adapter.env.observation_size,
                "robot_encoder_input_size": SONIC_ROBOT_INPUT_SIZE,
                "control_rate_hz": 1.0 / adapter.control_timestep,
                "standing_control_rate_hz": (
                    1.0 / standing_adapter.control_timestep
                ),
                "jax_backend": jax.default_backend(),
                "normal_standing_ramp_distance_max": initial_error,
                "armed": config.arm,
            }
        )
        if not config.arm:
            print(
                "Read-only SONIC preflight complete; no lease was acquired and "
                "no motor command was sent."
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
            standing_adapter.physical_coordinate_names,
            config.groups,
            config.imu_name,
            config.telemetry_timeout_seconds,
        )
        validate_hardware_frame(frame, config, expected_group_state="IDLE")
        schema = policy_phase_schema(adapter)
        standing_schema = policy_phase_schema(standing_adapter)
        log_context = (
            PolicyStepLogger(
                config.policy_log_path,
                task="sonic",
                checkpoints={
                    "sonic": checkpoint_path,
                    "standing": standing_checkpoint_path,
                },
                phases={
                    "standing": standing_schema,
                    "transition_to_motion": schema,
                    "motion": schema,
                    "transition_to_standing": standing_schema,
                },
            )
            if config.policy_log_path is not None
            else nullcontext(None)
        )
        with log_context as policy_log:
            control_physical_sonic(
                binding,
                groups,
                state_reader,
                references,
                frame,
                standing_adapter,
                adapter,
                config,
                policy_log=policy_log,
                operator_input=operator_input,
            )


def validate_sonic_transition_adapters(
    standing_adapter: HardwareLocomotionPolicyAdapter,
    sonic_adapter: HardwareSonicPolicyAdapter,
) -> None:
    if (
        standing_adapter.physical_coordinate_names
        != sonic_adapter.physical_coordinate_names
    ):
        raise ValueError(
            "standing and SONIC checkpoints require different physical "
            "coordinate mappings"
        )
    arrays = {
        "standing nominal target": standing_adapter.nominal_physical_target,
        "SONIC nominal target": sonic_adapter.nominal_physical_target,
        "standing kp": standing_adapter.kp,
        "standing kd": standing_adapter.kd,
        "SONIC kp": sonic_adapter.kp,
        "SONIC kd": sonic_adapter.kd,
    }
    expected_shape = (len(standing_adapter.physical_coordinate_names),)
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.shape != expected_shape or not np.all(np.isfinite(array)):
            raise ValueError(
                f"{name} must contain one finite value per physical coordinate"
            )
    if not np.array_equal(
        standing_adapter.nominal_physical_target,
        sonic_adapter.nominal_physical_target,
    ):
        raise ValueError("standing and SONIC checkpoints use different nominal poses")
    if not np.array_equal(standing_adapter.kp, sonic_adapter.kp) or not np.array_equal(
        standing_adapter.kd, sonic_adapter.kd
    ):
        raise ValueError(
            "standing and SONIC checkpoints require different impedance gains"
        )
    if not math.isclose(
        standing_adapter.control_timestep,
        sonic_adapter.control_timestep,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError("standing and SONIC checkpoints use different control rates")


def control_physical_sonic(
    binding: Any,
    groups: dict[str, Any],
    state_reader: Any,
    references: SonicReferenceSource,
    initial_frame: HardwareFrame,
    standing_adapter: HardwareLocomotionPolicyAdapter,
    sonic_adapter: HardwareSonicPolicyAdapter,
    config: HardwareSonicConfig,
    *,
    policy_log: PolicyStepLogger | None = None,
    operator_input: Callable[[str], str] | None = None,
) -> None:
    """Run the shared fail-closed lifecycle for both SONIC reference modes."""

    with armed_control_session(
        binding,
        groups,
        state_reader,
        initial_frame,
        standing_adapter,
        config,
        thread_name="bello-sonic-control",
        operator_input=operator_input,
    ) as control:
        frame = control.ramp_to_nominal(initial_frame.positions)
        if frame is None:
            raise RuntimeError("SONIC normal-standing ramp ended unexpectedly")
        require_nominal_pose(frame, standing_adapter)

        def validate_hold(held_frame: HardwareFrame) -> None:
            require_nominal_pose(held_frame, standing_adapter)

        control.wait_for_run(
            "Nominal-pose ramp complete. Keep external support engaged, then type "
            "RUN to start the standing balance policy: ",
            operator_input=operator_input,
            validate_hold=validate_hold,
        )
        require_nominal_pose(control.read_frame(), standing_adapter)
        standing_adapter.reset()
        standing_command = np.zeros(3, dtype=np.float32)
        references.reset()
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
            policy_step = standing_adapter.step(frame, standing_command)
            control.send_target(policy_step.physical_target)
            standing_steps += 1
            now = time.monotonic()
            if policy_log is not None:
                policy_log.write_step(
                    phase="standing",
                    step=standing_steps,
                    elapsed_seconds=now - run_started,
                    observation=policy_step.observation,
                    policy_observation=policy_step.policy_observation,
                    raw_action=policy_step.raw_action,
                    action=policy_step.action,
                    physical_target=policy_step.physical_target,
                    command=standing_command,
                )
            control.pace()
            if next_standing_status is not None and now >= next_standing_status:
                print(
                    {
                        "hardware_phase": "standing",
                        "hardware_step": standing_steps,
                        "elapsed_seconds": now - run_started,
                        "tilt_radians": policy_step.tilt_radians,
                        "action_abs_max": float(
                            np.max(np.abs(policy_step.action))
                        ),
                        "motor_age_ms": frame.max_motor_age_ms,
                        "imu_age_ms": frame.imu_age_ms,
                    }
                )
                assert config.status_interval_seconds is not None
                next_standing_status = now + config.status_interval_seconds
            return True

        if not standing_step():
            control.drain()
            print("Hardware run duration completed before standing control started.")
            return
        source_name = "live GEM-X tracking" if references.mode == "live" else "the clip"
        print(
            "The zero-command standing balance policy is active. Release external "
            "support once it is running and stable. "
            f"Type MOTION to begin {source_name}."
        )

        def resume_standing_control() -> None:
            nonlocal next_standing_status
            _transition_to_sonic_standing_policy(
                control,
                standing_adapter,
                standing_command,
                config,
                policy_log=policy_log,
                run_started=run_started,
            )
            next_standing_status = (
                time.monotonic()
                if config.status_interval_seconds is not None
                else None
            )
            print("The zero-command standing balance policy resumed.")
            while duration_available():
                standing_step()
            control.drain()

        try:
            requested_motion = wait_for_operator_confirmation_while_holding(
                MOTION_CONFIRMATION,
                f"Type MOTION to transition from normal standing to {source_name}: ",
                standing_step,
                operator_input,
            )
            if not requested_motion:
                control.drain()
                print("Hardware run duration completed before SONIC motion.")
                return
            if references.mode == "live":
                print(
                    "MOTION confirmed. Waiting for qualified live GEM-X SOMA "
                    "targets while normal standing remains active."
                )
                first_motion = _wait_for_live_sonic_reference(
                    references, sonic_adapter, config, standing_step
                )
                if first_motion is None:
                    control.drain()
                    print(
                        "No qualified GEM-X target arrived; SONIC motion was not run."
                    )
                    return
            else:
                first_motion = references.latest()
                sonic_adapter.validate_reference(first_motion)

            print(f"Transitioning from standing control to {source_name}.")
            try:
                _transition_sonic_reference(
                    control,
                    sonic_adapter,
                    references.latest,
                    config,
                    phase="transition_to_motion",
                    policy_log=policy_log,
                    run_started=run_started,
                    first_reference=first_motion,
                    advance_reference=(
                        references.advance if references.mode == "clip" else None
                    ),
                )
            except ReferenceUnavailable as error:
                print(
                    "SONIC reference became unavailable during the motion "
                    f"transition; restoring standing control: {error}"
                )
                resume_standing_control()
                return
            motion_started = time.monotonic()
            control.reset_deadline()
            motion_steps = 0
            next_motion_status = (
                motion_started if config.status_interval_seconds is not None else None
            )
            source_lost: ReferenceUnavailable | None = None
            clip_completed = False
            try:
                while duration_available():
                    frame = control.read_frame(
                        max_tilt_radians=config.max_motion_tilt_radians
                    )
                    try:
                        reference = references.latest()
                        step = sonic_adapter.step(frame, reference)
                    except ReferenceUnavailable as error:
                        source_lost = error
                        break
                    control.send_target(step.physical_target)
                    motion_steps += 1
                    now = time.monotonic()
                    _log_sonic_step(
                        policy_log,
                        phase="motion",
                        step_number=motion_steps,
                        run_started=run_started,
                        step=step,
                    )
                    has_next = references.advance()
                    control.pace()
                    if next_motion_status is not None and now >= next_motion_status:
                        print(
                            {
                                "hardware_phase": "motion",
                                "hardware_step": motion_steps,
                                "elapsed_seconds": now - run_started,
                                "reference_sequence": step.reference_sequence,
                                "reference_age_ms": step.reference_age_ms,
                                "source_action_abs_max": float(
                                    np.max(np.abs(step.raw_source_action))
                                ),
                                "action_abs_max": float(np.max(np.abs(step.action))),
                                "action_clipped": step.action_clipped,
                            }
                        )
                        assert config.status_interval_seconds is not None
                        next_motion_status = now + config.status_interval_seconds
                    if not has_next:
                        clip_completed = True
                        break
            except KeyboardInterrupt:
                print("SONIC motion interrupted; returning to standing.")
            if source_lost is not None:
                print(
                    "Live GEM-X reference became unavailable; returning to normal "
                    f"standing: {source_lost}"
                )
            elif clip_completed:
                print("SONIC clip completed; returning to normal standing.")
            print("Transitioning from SONIC motion back to standing control.")
            resume_standing_control()
        except KeyboardInterrupt:
            if control.adapter is not sonic_adapter:
                print("SONIC standing interrupted; entering damped shutdown.")
                return
            control.drain()
            print("SONIC motion interrupted; returning to standing control.")
            try:
                resume_standing_control()
            except KeyboardInterrupt:
                print("SONIC standing interrupted; entering damped shutdown.")


def _transition_sonic_reference(
    control: Any,
    adapter: HardwareSonicPolicyAdapter,
    reference: Callable[[], SonicReferenceWindow],
    config: HardwareSonicConfig,
    *,
    phase: str,
    policy_log: PolicyStepLogger | None,
    run_started: float,
    first_reference: SonicReferenceWindow | None = None,
    advance_reference: Callable[[], bool] | None = None,
) -> None:
    control.drain()
    initial_target = control.previous_target.copy()
    control.adapter = adapter
    adapter.reset()
    control.reset_deadline()
    steps = max(
        1, math.ceil(config.policy_transition_seconds / adapter.control_timestep)
    )
    for step_number in range(1, steps + 1):
        frame = control.read_frame(max_tilt_radians=config.max_motion_tilt_radians)
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
                raw_action=policy_step.raw_source_action,
                action=policy_step.action,
                physical_target=target,
                policy_target=policy_step.physical_target,
                command=policy_step.command,
            )
        if advance_reference is not None and not advance_reference():
            raise ReferenceUnavailable("SONIC clip ended during the policy transition")
        control.pace()
    control.drain()


def _transition_to_sonic_standing_policy(
    control: Any,
    adapter: HardwareLocomotionPolicyAdapter,
    command: np.ndarray,
    config: HardwareSonicConfig,
    *,
    policy_log: PolicyStepLogger | None,
    run_started: float,
) -> None:
    control.drain()
    initial_target = control.previous_target.copy()
    control.adapter = adapter
    adapter.reset()
    control.reset_deadline()
    steps = max(
        1, math.ceil(config.policy_transition_seconds / adapter.control_timestep)
    )
    for step_number in range(1, steps + 1):
        frame = control.read_frame(max_tilt_radians=config.max_motion_tilt_radians)
        policy_step = adapter.step(frame, command)
        fraction = step_number / steps
        target = initial_target + fraction * (
            policy_step.physical_target - initial_target
        )
        control.send_target(target)
        if policy_log is not None:
            policy_log.write_step(
                phase="transition_to_standing",
                step=step_number,
                elapsed_seconds=time.monotonic() - run_started,
                observation=policy_step.observation,
                policy_observation=policy_step.policy_observation,
                raw_action=policy_step.raw_action,
                action=policy_step.action,
                physical_target=target,
                policy_target=policy_step.physical_target,
                command=command,
            )
        control.pace()
    control.drain()


def _wait_for_live_sonic_reference(
    references: SonicReferenceSource,
    adapter: HardwareSonicPolicyAdapter,
    config: HardwareSonicConfig,
    standing_step: Callable[[], bool],
) -> SonicReferenceWindow | None:
    deadline = time.monotonic() + config.gemx_reference_ready_timeout_seconds
    last_error: ReferenceUnavailable | None = None
    last_sequence: int | None = None
    qualified_frames = 0
    while time.monotonic() < deadline:
        try:
            reference = references.latest()
            adapter.validate_reference(reference)
            if last_sequence is None or reference.sequence > last_sequence:
                qualified_frames += 1
                last_sequence = reference.sequence
            elif reference.sequence < last_sequence:
                qualified_frames = 0
                last_sequence = None
            if qualified_frames >= config.gemx_reference_qualification_frames:
                return reference
        except ReferenceUnavailable as error:
            last_error = error
            last_sequence = None
            qualified_frames = 0
        if not standing_step():
            return None
    if last_error is not None:
        print(f"Live GEM-X target is still unavailable: {last_error}")
    return None


def _log_sonic_step(
    policy_log: PolicyStepLogger | None,
    *,
    phase: str,
    step_number: int,
    run_started: float,
    step: HardwareSonicStep,
) -> None:
    if policy_log is None:
        return
    policy_log.write_step(
        phase=phase,
        step=step_number,
        elapsed_seconds=time.monotonic() - run_started,
        observation=step.observation,
        policy_observation=step.policy_observation,
        raw_action=step.raw_source_action,
        action=step.action,
        physical_target=step.physical_target,
        command=step.command,
    )


def _clip_reference_source(
    adapter: HardwareSonicPolicyAdapter,
    config: HardwareSonicConfig,
) -> SonicClipReferenceSource:
    if config.motion_path is None:
        raise ValueError("motion_path is required for SONIC clip playback")
    clip = load_motion_clip(config.motion_path, adapter.env.model)
    return SonicClipReferenceSource(clip, adapter.env, config)


def _clip_reference_context(
    adapter: HardwareSonicPolicyAdapter, config: HardwareSonicConfig
) -> Any:
    return nullcontext(_clip_reference_source(adapter, config))


def _live_reference_context(
    adapter: HardwareSonicPolicyAdapter, config: HardwareSonicConfig
) -> Any:
    from bello_mujoco.hardware.gemx import gemx_sonic_reference_source

    source_context = gemx_sonic_reference_source(adapter, config)

    class _Context:
        def __enter__(self) -> SonicLiveReferenceSource:
            source = source_context.__enter__()
            self.source = source
            return SonicLiveReferenceSource(source, adapter.env, config)

        def __exit__(self, *exception: object) -> bool:
            return source_context.__exit__(*exception)

    return _Context()


def _clip_reference_joints(qpos: np.ndarray, env: HardwareSonicEnvironment) -> None:
    policy = np.asarray(env.model.policy_ctrl_indices)
    joint_ids = np.asarray(env.model.actuator_joint_ids)[policy]
    ranges = np.asarray(env.model.mj_model.jnt_range)[joint_ids]
    indices = np.asarray(env.model.policy_qpos_indices)
    qpos[indices] = np.clip(qpos[indices], ranges[:, 0], ranges[:, 1])


def _canonical_standing_window(
    env: HardwareSonicEnvironment,
) -> SonicReferenceWindow:
    qpos = np.repeat(
        np.asarray(env.model.default_qpos, dtype=np.float64)[None, :],
        len(SONIC_CRITIC_FUTURE_OFFSETS),
        axis=0,
    )
    qvel = np.zeros(
        (len(SONIC_CRITIC_FUTURE_OFFSETS), env.model.mj_model.nv),
        dtype=np.float64,
    )
    return SonicReferenceWindow(
        qpos=qpos,
        qvel=qvel,
        sequence=0,
        source_age_ms=None,
        received_at=time.monotonic(),
    )


def _quaternion_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm < 1.0e-8:
        raise ReferenceUnavailable("SONIC reference quaternion is invalid")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        )
    )


def _yaw_wxyz(quaternion: np.ndarray) -> float:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
