"""Deploy a motion-tracking checkpoint through Bello's hardware services."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
import math
import threading
import time
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from bello_mujoco.training.checkpoint import (
    CheckpointNormalizationState,
    MotionCheckpointMetadata,
)
from bello_mujoco.envs.motion_tracking import (
    BelloMotionTrackingEnv,
    MotionTrackingState,
    quat_distance,
)
from bello_mujoco.hardware import _pose_pb2
from bello_mujoco.hardware.config import HardwareControlConfig
from bello_mujoco.motion.native import NativeContactData, NativeMotionData
from bello_mujoco.motion.runtime import load_motion_policy_runtime
from bello_mujoco.hardware.runtime import (
    HOLD_CONFIRMATION,
    HardwareFrame,
    actuator_coordinate_names,
    actuator_impedance_gains,
    armed_control_session,
    import_hardware_binding,
    require_nominal_pose,
    require_operator_confirmation,
    require_async_hardware_binding,
    validate_hardware_control_config,
    validate_hardware_frame,
    validate_hardware_inventory,
    wait_for_hardware_frame,
)
from bello_mujoco.policy import apply_actor
from bello_mujoco.spatial import quat_rotate_wxyz
from bello_mujoco.training.ppo import normalize_observation


@dataclass(frozen=True)
class HardwareMotionConfig(HardwareControlConfig):
    """Hardware, pose-estimator, and fail-closed motion deployment settings."""

    client_name: str = "bello-motion-policy"
    motion_path: Path | None = None
    pose_endpoint: str = "127.0.0.1:50052"
    pose_reference_frame: str = "odom"
    pose_tracked_frame: str = "pelvis_link"
    pose_timeout_seconds: float = 2.0
    max_pose_age_ms: int = 100
    velocity_filter_alpha: float = 0.2
    start_time_seconds: float = 0.0
    max_action_abs: float = 1.0
    max_raw_action_abs: float = 5.0
    max_pose_imu_orientation_error: float = 0.25
    max_base_linear_velocity: float = 5.0


@dataclass(frozen=True)
class PoseFrame:
    position: np.ndarray
    orientation_xyzw: np.ndarray
    linear_velocity: np.ndarray
    sequence: int
    source_age_ms: int
    received_at: float

    def age_ms(self, now: float | None = None) -> float:
        current = time.monotonic() if now is None else now
        return self.source_age_ms + 1000.0 * max(0.0, current - self.received_at)


@dataclass(frozen=True)
class HardwareMotionStep:
    action: np.ndarray
    raw_action: np.ndarray
    physical_target: np.ndarray
    observation: np.ndarray
    reference_time: float
    anchor_position_error: float
    anchor_orientation_error: float
    action_clipped: bool


@dataclass(frozen=True)
class _MotionInferenceLayout:
    qpos: slice
    qvel: slice
    ctrl: slice
    xpos: slice
    xquat: slice
    last_action: slice
    reference_time: int
    alignment_xy: slice
    alignment_yaw: int
    initialize_stack: int
    size: int


def _motion_inference_layout(env: BelloMotionTrackingEnv) -> _MotionInferenceLayout:
    offset = 0

    def take(size: int) -> slice:
        nonlocal offset
        result = slice(offset, offset + size)
        offset += size
        return result

    qpos = take(env.model.mj_model.nq)
    qvel = take(env.model.mj_model.nv)
    ctrl = take(env.model.mj_model.nu)
    xpos = take(env.model.mj_model.nbody * 3)
    xquat = take(env.model.mj_model.nbody * 4)
    last_action = take(env.action_size)
    reference_time = offset
    offset += 1
    alignment_xy = take(2)
    alignment_yaw = offset
    offset += 1
    initialize_stack = offset
    offset += 1
    return _MotionInferenceLayout(
        qpos=qpos,
        qvel=qvel,
        ctrl=ctrl,
        xpos=xpos,
        xquat=xquat,
        last_action=last_action,
        reference_time=reference_time,
        alignment_xy=alignment_xy,
        alignment_yaw=alignment_yaw,
        initialize_stack=initialize_stack,
        size=offset,
    )


class _MotionInferenceResult(NamedTuple):
    observation_stack: jax.Array
    observation: jax.Array
    raw_action: jax.Array
    action: jax.Array
    physical_target: jax.Array
    done_flags: dict[str, jax.Array]
    anchor_position_error: jax.Array
    anchor_orientation_error: jax.Array


def _build_motion_inference(
    env: BelloMotionTrackingEnv,
    policy: Any,
    normalization: Any,
    metadata: MotionCheckpointMetadata,
    config: HardwareMotionConfig,
) -> Any:
    layout = _motion_inference_layout(env)
    body_count = env.model.mj_model.nbody
    stack_size = metadata.ppo.policy.obs_stack_size
    empty_contacts = NativeContactData(
        contact__geom=jnp.zeros((0, 2), dtype=jnp.int32),
        nacon=jnp.zeros(1, dtype=jnp.int32),
        contact__worldid=jnp.zeros(0, dtype=jnp.int32),
        contact__efc_address=jnp.zeros((0, 1), dtype=jnp.int32),
        efc__force=jnp.zeros(1, dtype=jnp.float32),
    )
    empty_subtree_com = jnp.zeros((0, 3), dtype=jnp.float32)
    empty_cvel = jnp.zeros((body_count, 6), dtype=jnp.float32)
    empty_actuator_force = jnp.zeros(env.model.mj_model.nu, dtype=jnp.float32)
    joint_default_offset = jnp.zeros(env.action_size, dtype=jnp.float32)
    empty_sampling_state = jnp.zeros(env.sampling_bin_count, dtype=jnp.float32)

    def inference(
        hardware_state: jax.Array,
        observation_stack: jax.Array,
    ) -> _MotionInferenceResult:
        qpos = hardware_state[layout.qpos]
        qvel = hardware_state[layout.qvel]
        ctrl = hardware_state[layout.ctrl]
        xpos = hardware_state[layout.xpos].reshape(body_count, 3)
        xquat = hardware_state[layout.xquat].reshape(body_count, 4)
        state = MotionTrackingState(
            data=NativeMotionData(
                qpos=qpos,
                qvel=qvel,
                ctrl=ctrl,
                xpos=xpos,
                xquat=xquat,
                cvel=empty_cvel,
                subtree_com=empty_subtree_com,
                actuator_force=empty_actuator_force,
                _impl=empty_contacts,
            ),
            last_action=hardware_state[layout.last_action],
            reference_time=hardware_state[layout.reference_time],
            alignment_xy=hardware_state[layout.alignment_xy],
            alignment_yaw=hardware_state[layout.alignment_yaw],
            joint_default_offset=joint_default_offset,
            next_push_step=jnp.asarray(0, dtype=jnp.int32),
            sampling_failure_ema=empty_sampling_state,
            step_count=jnp.asarray(0, dtype=jnp.int32),
            episode_return=jnp.asarray(0.0, dtype=jnp.float32),
            episode_length=jnp.asarray(0, dtype=jnp.int32),
            last_episode_return=jnp.asarray(0.0, dtype=jnp.float32),
            last_episode_length=jnp.asarray(0, dtype=jnp.int32),
        )
        done_flags = env.termination_flags(state)
        observation = env.observation(state)
        initial_stack = jnp.repeat(observation[None, :], stack_size, axis=0)
        shifted_stack = jnp.concatenate(
            (observation_stack[1:], observation[None, :]), axis=0
        )
        next_stack = jnp.where(
            hardware_state[layout.initialize_stack] > 0.5,
            initial_stack,
            shifted_stack,
        )
        normalized = normalize_observation(
            next_stack.reshape(-1),
            normalization,
            metadata.ppo,
        )
        raw_action = apply_actor(policy, normalized)
        action = jnp.clip(raw_action, -config.max_action_abs, config.max_action_abs)
        physical_target = env.action_target(state, action)
        reference = env.aligned_reference(
            state.reference_time,
            state.alignment_xy,
            state.alignment_yaw,
        )
        anchor_position_error = jnp.linalg.norm(
            state.data.xpos[env.anchor_body_id]
            - reference.body_pos[env.anchor_body_index]
        )
        anchor_orientation_error = quat_distance(
            state.data.xquat[env.anchor_body_id],
            reference.body_quat[env.anchor_body_index],
        )
        return _MotionInferenceResult(
            observation_stack=next_stack,
            observation=observation,
            raw_action=raw_action,
            action=action,
            physical_target=physical_target,
            done_flags=done_flags,
            anchor_position_error=anchor_position_error,
            anchor_orientation_error=anchor_orientation_error,
        )

    return eqx.filter_jit(inference)


class PoseStream:
    """Background latest-value client for bello-state-estimatord."""

    def __init__(self, config: HardwareMotionConfig) -> None:
        self.config = config
        self._condition = threading.Condition()
        self._latest: PoseFrame | None = None
        self._error: BaseException | None = None
        self._channel: Any | None = None
        self._call: Any | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "PoseStream":
        import grpc

        endpoint = self.config.pose_endpoint.removeprefix("http://")
        if endpoint.startswith("https://"):
            raise ValueError(
                "the motion runner currently requires a local insecure pose endpoint"
            )
        request = _pose_pb2.WatchPoseRequest(
            reference_frame=self.config.pose_reference_frame,
            tracked_frame=self.config.pose_tracked_frame,
        )
        self._channel = grpc.insecure_channel(endpoint)
        method = self._channel.unary_stream(
            "/bello.pose.v1.PoseService/WatchPose",
            request_serializer=lambda value: value.SerializeToString(),
            response_deserializer=_pose_pb2.WatchPoseResponse.FromString,
        )
        self._call = method(request)
        self._thread = threading.Thread(
            target=self._consume,
            name="bello-motion-pose-stream",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        if self._call is not None:
            self._call.cancel()
        if self._channel is not None:
            self._channel.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _consume(self) -> None:
        previous: PoseFrame | None = None
        filtered_velocity = np.zeros(3, dtype=np.float64)
        try:
            for response in self._responses():
                received = time.monotonic()
                if (
                    response.reference_frame != self.config.pose_reference_frame
                    or response.tracked_frame != self.config.pose_tracked_frame
                ):
                    raise RuntimeError(
                        "pose-estimator response frame mismatch: "
                        f"{response.reference_frame}_from_{response.tracked_frame}"
                    )
                sequence = int(response.sequence)
                if previous is not None and sequence <= previous.sequence:
                    raise RuntimeError(
                        "pose-estimator sequence did not increase: "
                        f"{sequence} after {previous.sequence}"
                    )
                position = np.asarray(
                    (
                        response.translation_metres.x,
                        response.translation_metres.y,
                        response.translation_metres.z,
                    ),
                    dtype=np.float64,
                )
                orientation = np.asarray(
                    (
                        response.orientation_xyzw.x,
                        response.orientation_xyzw.y,
                        response.orientation_xyzw.z,
                        response.orientation_xyzw.w,
                    ),
                    dtype=np.float64,
                )
                source_age = int(response.sample_age_ms)
                sample_time = received - source_age / 1000.0
                if previous is not None:
                    previous_time = (
                        previous.received_at - previous.source_age_ms / 1000.0
                    )
                    interval = sample_time - previous_time
                    if 1.0e-4 <= interval <= 0.5:
                        measured = (position - previous.position) / interval
                        alpha = self.config.velocity_filter_alpha
                        filtered_velocity = (
                            alpha * measured + (1.0 - alpha) * filtered_velocity
                        )
                frame = PoseFrame(
                    position=position,
                    orientation_xyzw=orientation,
                    linear_velocity=filtered_velocity.copy(),
                    sequence=sequence,
                    source_age_ms=source_age,
                    received_at=received,
                )
                validate_pose_frame(frame, self.config)
                previous = frame
                with self._condition:
                    self._latest = frame
                    self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                self._error = error
                self._condition.notify_all()

    def _responses(self) -> Iterator[Any]:
        if self._call is None:
            raise RuntimeError("pose stream is not open")
        return iter(self._call)

    def latest(self, timeout: float | None = None) -> PoseFrame:
        timeout = self.config.pose_timeout_seconds if timeout is None else timeout
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._latest is None and self._error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("timed out waiting for pose-estimator telemetry")
                self._condition.wait(remaining)
            if self._error is not None:
                raise RuntimeError("pose-estimator stream failed") from self._error
            assert self._latest is not None
            frame = self._latest
        validate_pose_frame(frame, self.config)
        return frame


def validate_pose_frame(frame: PoseFrame, config: HardwareMotionConfig) -> None:
    values = {
        "position": (frame.position, (3,)),
        "orientation": (frame.orientation_xyzw, (4,)),
        "linear velocity": (frame.linear_velocity, (3,)),
    }
    for name, (value, shape) in values.items():
        array = np.asarray(value)
        if array.shape != shape or not np.all(np.isfinite(array)):
            raise RuntimeError(
                f"pose-estimator {name} must contain {shape[0]} finite values"
            )
    normalized_xyzw(frame.orientation_xyzw)
    if frame.sequence < 0 or frame.source_age_ms < 0:
        raise RuntimeError("pose-estimator sequence and age must be nonnegative")
    age = frame.age_ms()
    if age > config.max_pose_age_ms:
        raise RuntimeError(
            f"pose-estimator telemetry is stale: {age:.1f} ms exceeds "
            f"{config.max_pose_age_ms} ms"
        )
    speed = float(np.linalg.norm(frame.linear_velocity))
    if speed > config.max_base_linear_velocity:
        raise RuntimeError(
            f"estimated base speed {speed:.3f} m/s exceeds "
            f"{config.max_base_linear_velocity:.3f} m/s"
        )


class HardwareMotionPolicyAdapter:
    """Reconstruct a motion actor frame and physical motor target."""

    def __init__(
        self,
        env: BelloMotionTrackingEnv,
        policy: Any,
        metadata: MotionCheckpointMetadata,
        normalization: CheckpointNormalizationState,
        config: HardwareMotionConfig,
    ) -> None:
        validate_hardware_motion_checkpoint(env, metadata)
        validate_hardware_motion_config(config, env)
        self.env = env
        self.metadata = metadata
        self.config = config
        self.physical_coordinate_names = actuator_coordinate_names(env)
        self.coordinate_index = {
            name: index for index, name in enumerate(self.physical_coordinate_names)
        }
        if len(self.coordinate_index) != len(self.physical_coordinate_names):
            raise ValueError("physical actuator coordinate names must be unique")
        self.transform = np.asarray(
            env.model.actuator_coordinate_transform, dtype=np.float64
        )
        if self.transform.shape != (25, 25):
            raise ValueError(
                "motion hardware deployment requires a 25x25 actuator transform"
            )
        condition = float(np.linalg.cond(self.transform))
        if not math.isfinite(condition) or condition > 1.0e6:
            raise ValueError(
                "actuator coordinate transform is not safely invertible: "
                f"condition={condition}"
            )
        self.inverse_transform = np.linalg.inv(self.transform)
        self.kp, self.kd = actuator_impedance_gains(env)
        self._inference = _build_motion_inference(
            env,
            policy,
            normalization.observation,
            metadata,
            config,
        )
        self._inference_layout = _motion_inference_layout(env)
        self._inference_input = np.empty(self._inference_layout.size, dtype=np.float32)
        self._mj_data = mujoco.MjData(env.model.mj_model)
        self._empty_obs_stack = jnp.zeros(
            (metadata.ppo.policy.obs_stack_size, env.observation_size),
            dtype=jnp.float32,
        )
        self._obs_stack: jax.Array | None = None
        self._last_action = np.zeros(env.action_size, dtype=np.float32)
        self._last_physical_target = np.asarray(
            env.model.mj_model.actuator_ctrlrange.mean(axis=1), dtype=np.float64
        )
        self._alignment_xy = np.zeros(2, dtype=np.float32)
        self._alignment_yaw = 0.0
        self._aligned = False

    @property
    def control_timestep(self) -> float:
        return self.env.control_timestep

    @property
    def start_time(self) -> float:
        return self.config.start_time_seconds

    @property
    def nominal_physical_target(self) -> np.ndarray:
        logical = np.asarray(self.env.model.default_joint_target, dtype=np.float64)
        target = self.transform @ logical
        limits = np.asarray(self.env.model.mj_model.actuator_ctrlrange)
        return np.clip(target, limits[:, 0], limits[:, 1])

    def initialize_alignment(self, pose: PoseFrame) -> None:
        pose_quat = xyzw_to_wxyz(pose.orientation_xyzw)
        source = self.env.aligned_reference(
            jnp.asarray(self.start_time, dtype=jnp.float32),
            jnp.zeros(2, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        source_position = np.asarray(source.qpos[:3], dtype=np.float64)
        source_quat = np.asarray(source.qpos[3:7], dtype=np.float64)
        yaw = wrap_angle(quaternion_yaw(pose_quat) - quaternion_yaw(source_quat))
        yaw_quat = np.asarray(
            (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)),
            dtype=np.float64,
        )
        rotated_source = quat_rotate_wxyz(yaw_quat, source_position)
        self._alignment_xy = np.asarray(
            pose.position[:2] - rotated_source[:2], dtype=np.float32
        )
        self._alignment_yaw = yaw
        self._aligned = True
        self.reset()

    def reset(self) -> None:
        self._obs_stack = None
        self._last_action = np.zeros(self.env.action_size, dtype=np.float32)

    def precompile(self) -> None:
        source = self.env.aligned_reference(
            jnp.asarray(self.start_time, dtype=jnp.float32),
            jnp.zeros(2, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        pose = PoseFrame(
            position=np.asarray(source.qpos[:3], dtype=np.float64),
            orientation_xyzw=wxyz_to_xyzw(np.asarray(source.qpos[3:7])),
            linear_velocity=np.asarray(source.qvel[:3], dtype=np.float64),
            sequence=0,
            source_age_ms=0,
            received_at=time.monotonic(),
        )
        logical_position = np.asarray(source.qpos)[self.env.model.actuator_qpos_indices]
        logical_velocity = np.asarray(source.qvel)[self.env.model.actuator_qvel_indices]
        frame = HardwareFrame(
            positions=self.transform @ logical_position,
            velocities=self.transform @ logical_velocity,
            orientation_xyzw=pose.orientation_xyzw,
            angular_velocity=np.zeros(3, dtype=np.float64),
            linear_acceleration=np.zeros(3, dtype=np.float64),
            max_motor_age_ms=0,
            imu_age_ms=0,
            group_states=(),
        )
        self.initialize_alignment(pose)
        self.step(frame, pose, reference_time=self.start_time)
        self._aligned = False
        self.reset()

    def step(
        self,
        frame: HardwareFrame,
        pose: PoseFrame,
        *,
        reference_time: float,
    ) -> HardwareMotionStep:
        if not self._aligned:
            raise RuntimeError("motion alignment must be initialized before policy use")
        validate_pose_imu_agreement(frame, pose, self.config)
        state = self._state(frame, pose, reference_time)
        initialize_stack = self._obs_stack is None
        state[self._inference_layout.initialize_stack] = initialize_stack
        observation_stack = (
            self._empty_obs_stack if initialize_stack else self._obs_stack
        )
        assert observation_stack is not None
        inference = self._inference(
            state,
            observation_stack,
        )
        host = jax.device_get(inference)
        done_flags = host.done_flags
        active_done = {name: bool(value) for name, value in done_flags.items()}
        if active_done["done/nonfinite"]:
            raise RuntimeError("motion policy state became non-finite")
        if active_done["done/timeout"]:
            raise StopIteration("motion reference completed")
        terminal = sorted(
            name.removeprefix("done/")
            for name, active in active_done.items()
            if active and name not in {"done/any", "done/timeout"}
        )
        if terminal:
            raise RuntimeError(
                "motion tracking termination triggered on hardware: "
                + ", ".join(terminal)
            )
        observation = np.asarray(host.observation)
        raw_action = np.asarray(host.raw_action, dtype=np.float32)
        if raw_action.shape != (self.env.action_size,) or not np.all(
            np.isfinite(raw_action)
        ):
            raise RuntimeError(
                f"policy produced an invalid hardware action with shape {raw_action.shape}"
            )
        raw_abs = float(np.max(np.abs(raw_action)))
        if raw_abs > self.config.max_raw_action_abs:
            raise RuntimeError(
                f"raw policy action {raw_abs:.3f} exceeds fail-closed limit "
                f"{self.config.max_raw_action_abs:.3f}"
            )
        action = np.asarray(host.action, dtype=np.float32)
        physical_target = np.asarray(host.physical_target, dtype=np.float64)
        if not np.all(np.isfinite(physical_target)):
            raise RuntimeError("policy produced non-finite physical motor targets")
        self._obs_stack = inference.observation_stack
        self._last_action = action
        self._last_physical_target = physical_target
        return HardwareMotionStep(
            action=action,
            raw_action=raw_action,
            physical_target=physical_target,
            observation=observation,
            reference_time=reference_time,
            anchor_position_error=float(host.anchor_position_error),
            anchor_orientation_error=float(host.anchor_orientation_error),
            action_clipped=not np.array_equal(action, raw_action),
        )

    def _state(self, frame: HardwareFrame, pose: PoseFrame, reference_time: float):
        model = self.env.model
        logical_position = self.inverse_transform @ frame.positions
        logical_velocity = self.inverse_transform @ frame.velocities
        qpos = np.asarray(model.mj_model.qpos0).copy()
        qvel = np.zeros(model.mj_model.nv, dtype=np.float64)
        root = model.root_qpos_adr
        root_velocity = model.root_qvel_adr
        root_quat = xyzw_to_wxyz(pose.orientation_xyzw)
        qpos[root : root + 3] = pose.position
        qpos[root + 3 : root + 7] = root_quat
        qpos[model.actuator_qpos_indices] = logical_position
        qvel[root_velocity : root_velocity + 3] = pose.linear_velocity
        # The hardware gyro and MuJoCo free-joint angular qvel are both root-local.
        qvel[root_velocity + 3 : root_velocity + 6] = frame.angular_velocity
        qvel[model.actuator_qvel_indices] = logical_velocity
        self._mj_data.qpos[:] = qpos
        self._mj_data.qvel[:] = qvel
        self._mj_data.ctrl[:] = self._last_physical_target
        mujoco.mj_kinematics(model.mj_model, self._mj_data)
        layout = self._inference_layout
        inference_input = self._inference_input
        inference_input[layout.qpos] = self._mj_data.qpos
        inference_input[layout.qvel] = self._mj_data.qvel
        inference_input[layout.ctrl] = self._mj_data.ctrl
        inference_input[layout.xpos] = self._mj_data.xpos.reshape(-1)
        inference_input[layout.xquat] = self._mj_data.xquat.reshape(-1)
        inference_input[layout.last_action] = self._last_action
        inference_input[layout.reference_time] = reference_time
        inference_input[layout.alignment_xy] = self._alignment_xy
        inference_input[layout.alignment_yaw] = self._alignment_yaw
        return inference_input


def validate_hardware_motion_checkpoint(
    env: BelloMotionTrackingEnv, metadata: MotionCheckpointMetadata
) -> None:
    if metadata.task != "motion_tracking":
        raise ValueError("motion hardware runner requires a motion-tracking checkpoint")
    if metadata.motion_env.formulation != "beyondmimic_joint_limits":
        raise ValueError(
            "motion hardware deployment requires the bounded BeyondMimic formulation"
        )
    if metadata.ppo.policy.action_distribution != "tanh_gaussian":
        raise ValueError(
            "bounded BeyondMimic hardware deployment requires "
            "'tanh_gaussian' policy actions"
        )
    if metadata.asset.end_effector_sphere_mode != "both":
        raise ValueError("motion hardware runner requires sphere hands")
    availability = metadata.asset.joint_availability
    if availability.enabled_joint_names != {"waist_yaw_joint"}:
        raise ValueError(
            "motion hardware runner requires only waist yaw among optional joints"
        )
    if env.action_size != 25:
        raise ValueError(
            f"motion hardware runner requires 25 policy joints, got {env.action_size}"
        )
    if env.action_layout != metadata.action_layout:
        raise ValueError("motion hardware action layout does not match checkpoint")
    if env.observation_layout != metadata.actor_observation_layout:
        raise ValueError("motion hardware observation layout does not match checkpoint")


def validate_hardware_motion_config(
    config: HardwareMotionConfig,
    env: BelloMotionTrackingEnv | None = None,
) -> None:
    validate_hardware_control_config(config)
    if not config.pose_endpoint.strip():
        raise ValueError("pose_endpoint must not be empty")
    if not config.pose_reference_frame.strip() or not config.pose_tracked_frame.strip():
        raise ValueError("pose frame names must not be empty")
    positive = {
        "pose_timeout_seconds": config.pose_timeout_seconds,
        "max_action_abs": config.max_action_abs,
        "max_raw_action_abs": config.max_raw_action_abs,
        "max_pose_imu_orientation_error": config.max_pose_imu_orientation_error,
        "max_base_linear_velocity": config.max_base_linear_velocity,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if config.max_pose_age_ms <= 0:
        raise ValueError("max_pose_age_ms must be positive")
    if not 0.0 < config.velocity_filter_alpha <= 1.0:
        raise ValueError("velocity_filter_alpha must be in (0, 1]")
    if not math.isfinite(config.start_time_seconds) or config.start_time_seconds < 0.0:
        raise ValueError("start_time_seconds must be finite and nonnegative")
    if env is not None and config.start_time_seconds >= env.motion.duration:
        raise ValueError(
            f"start_time_seconds must be in [0, {env.motion.duration}), got "
            f"{config.start_time_seconds}"
        )
    if config.max_action_abs > config.max_raw_action_abs:
        raise ValueError("max_action_abs must not exceed max_raw_action_abs")


def load_hardware_motion_policy(
    checkpoint_path: Path,
    config: HardwareMotionConfig,
) -> tuple[HardwareMotionPolicyAdapter, MotionCheckpointMetadata]:
    runtime = load_motion_policy_runtime(
        checkpoint_path,
        motion_path=config.motion_path,
        use_viewer_model=False,
        allow_git_commit_mismatch=config.allow_git_commit_mismatch,
    )
    return (
        HardwareMotionPolicyAdapter(
            runtime.env,
            runtime.policy,
            runtime.metadata,
            runtime.normalization,
            config,
        ),
        runtime.metadata,
    )


def run_hardware_motion(
    checkpoint_path: Path,
    config: HardwareMotionConfig = HardwareMotionConfig(),
    *,
    hardware_binding: Any | None = None,
    pose_stream_factory: Any = PoseStream,
    operator_input: Callable[[str], str] | None = None,
) -> None:
    validate_hardware_motion_config(config)
    adapter, metadata = load_hardware_motion_policy(checkpoint_path, config)
    validate_hardware_motion_config(config, adapter.env)
    adapter.precompile()
    binding = hardware_binding or import_hardware_binding()
    if config.arm:
        require_async_hardware_binding(binding)
    with (
        pose_stream_factory(config) as poses,
        binding.Robot.connect(
            config.endpoint, timeout=config.connection_timeout_seconds
        ) as robot,
    ):
        groups = validate_hardware_inventory(robot, adapter, config, binding)
        with robot.states(
            groups=config.groups,
            imus=(config.imu_name,),
            reconnect=False,
        ) as states:
            frame = wait_for_hardware_frame(
                states,
                adapter.physical_coordinate_names,
                config.groups,
                config.imu_name,
                binding.StateUnavailableError,
                config.telemetry_timeout_seconds,
            )
            pose = poses.latest()
            validate_hardware_frame(frame, config, expected_group_state="IDLE")
            validate_pose_imu_agreement(frame, pose, config)
            adapter.initialize_alignment(pose)
            initial_delta = np.abs(frame.positions - adapter.nominal_physical_target)
            initial_error = float(np.max(initial_delta))
            preview = adapter.step(frame, pose, reference_time=adapter.start_time)
            adapter.reset()
            print(
                {
                    "motion_hardware_preflight": "passed",
                    "checkpoint_task": metadata.task,
                    "endpoint": config.endpoint,
                    "pose_endpoint": config.pose_endpoint,
                    "groups": config.groups,
                    "coordinates": len(adapter.physical_coordinate_names),
                    "policy_joints": adapter.env.action_size,
                    "control_rate_hz": 1.0 / adapter.control_timestep,
                    "jax_backend": jax.default_backend(),
                    "motion_start_seconds": adapter.start_time,
                    "nominal_ramp_distance_max": initial_error,
                    "preview_action_abs_max": float(np.max(np.abs(preview.raw_action))),
                    "preview_action_clipped": preview.action_clipped,
                    "armed": config.arm,
                }
            )
            if not config.arm:
                print(
                    "Read-only motion preflight complete; no lease was acquired "
                    "and no motor command was sent."
                )
                return
            require_operator_confirmation(
                HOLD_CONFIRMATION,
                "Support the robot, then type HOLD to acquire control and begin "
                "the nominal-pose ramp: ",
                operator_input,
            )
            frame = wait_for_hardware_frame(
                states,
                adapter.physical_coordinate_names,
                config.groups,
                config.imu_name,
                binding.StateUnavailableError,
                config.telemetry_timeout_seconds,
            )
            pose = poses.latest()
            validate_hardware_frame(frame, config, expected_group_state="IDLE")
            validate_pose_imu_agreement(frame, pose, config)
            control_physical_motion(
                binding,
                groups,
                states,
                poses,
                frame,
                adapter,
                config,
                operator_input=operator_input,
            )


def control_physical_motion(
    binding: Any,
    groups: dict[str, Any],
    states: Any,
    poses: PoseStream,
    initial_frame: HardwareFrame,
    adapter: HardwareMotionPolicyAdapter,
    config: HardwareMotionConfig,
    *,
    operator_input: Callable[[str], str] | None = None,
) -> None:
    with armed_control_session(
        binding,
        groups,
        states,
        initial_frame,
        adapter,
        config,
        thread_name="bello-motion-control",
    ) as control:
        frame = control.ramp_to_nominal(initial_frame.positions)
        if frame is None:
            raise RuntimeError("motion nominal-pose ramp ended unexpectedly")
        require_nominal_pose(frame, adapter)

        def validate_hold(held_frame: HardwareFrame) -> None:
            require_nominal_pose(held_frame, adapter)
            held_pose = poses.latest()
            validate_pose_imu_agreement(held_frame, held_pose, config)

        control.wait_for_run(
            "Nominal-pose ramp complete. Release external support, let the robot "
            "settle, then type RUN to start policy control: ",
            operator_input=operator_input,
            validate_hold=validate_hold,
        )
        frame = control.read_frame()
        pose = poses.latest()
        validate_pose_imu_agreement(frame, pose, config)
        require_nominal_pose(frame, adapter)
        adapter.initialize_alignment(pose)
        start = time.monotonic()
        control.reset_deadline()
        next_status = start
        step = 0
        maximum_inference_ms = 0.0
        maximum_command_submission_ms = 0.0
        maximum_loop_work_ms = 0.0
        try:
            while (
                config.max_duration_seconds is None
                or time.monotonic() - start < config.max_duration_seconds
            ):
                loop_started = time.monotonic()
                frame = control.read_frame()
                pose = poses.latest()
                elapsed = time.monotonic() - start
                reference_time = adapter.start_time + elapsed
                inference_started = time.monotonic()
                try:
                    policy_step = adapter.step(
                        frame, pose, reference_time=reference_time
                    )
                except StopIteration:
                    print("Motion reference completed; stopping and disabling groups.")
                    break
                inference_ms = (time.monotonic() - inference_started) * 1_000.0
                command_started = time.monotonic()
                control.send_target(policy_step.physical_target)
                command_submission_ms = (time.monotonic() - command_started) * 1_000.0
                loop_work_ms = (time.monotonic() - loop_started) * 1_000.0
                maximum_inference_ms = max(maximum_inference_ms, inference_ms)
                maximum_command_submission_ms = max(
                    maximum_command_submission_ms, command_submission_ms
                )
                maximum_loop_work_ms = max(maximum_loop_work_ms, loop_work_ms)
                step += 1
                now = time.monotonic()
                if now >= next_status:
                    print(
                        {
                            "motion_hardware_step": step,
                            "elapsed_seconds": elapsed,
                            "reference_time": policy_step.reference_time,
                            "action_abs_max": float(
                                np.max(np.abs(policy_step.raw_action))
                            ),
                            "action_clipped": policy_step.action_clipped,
                            "anchor_position_error": policy_step.anchor_position_error,
                            "anchor_orientation_error": (
                                policy_step.anchor_orientation_error
                            ),
                            "motor_age_ms": frame.max_motor_age_ms,
                            "imu_age_ms": frame.imu_age_ms,
                            "pose_age_ms": pose.age_ms(now),
                            "inference_ms": inference_ms,
                            "command_submission_ms": command_submission_ms,
                            "loop_work_ms": loop_work_ms,
                            "maximum_inference_ms": maximum_inference_ms,
                            "maximum_command_submission_ms": (
                                maximum_command_submission_ms
                            ),
                            "maximum_loop_work_ms": maximum_loop_work_ms,
                        }
                    )
                    next_status = now + config.status_interval_seconds
                try:
                    control.pace()
                except RuntimeError:
                    print(
                        {
                            "motion_hardware_timing_failure": True,
                            "step": step,
                            "inference_ms": inference_ms,
                            "command_submission_ms": command_submission_ms,
                            "loop_work_ms": loop_work_ms,
                            "maximum_inference_ms": maximum_inference_ms,
                            "maximum_command_submission_ms": (
                                maximum_command_submission_ms
                            ),
                            "maximum_loop_work_ms": maximum_loop_work_ms,
                            "accumulated_lag_ms": (
                                time.monotonic()
                                - (control.deadline + adapter.control_timestep)
                            )
                            * 1_000.0,
                        },
                        flush=True,
                    )
                    raise
            control.drain()
        except KeyboardInterrupt:
            print("Motion hardware run interrupted; stopping and disabling groups.")


def validate_pose_imu_agreement(
    frame: HardwareFrame, pose: PoseFrame, config: HardwareMotionConfig
) -> None:
    difference = quaternion_distance(
        xyzw_to_wxyz(frame.orientation_xyzw),
        xyzw_to_wxyz(pose.orientation_xyzw),
    )
    if difference > config.max_pose_imu_orientation_error:
        raise RuntimeError(
            f"pose/IMU orientation mismatch {difference:.3f} rad exceeds "
            f"{config.max_pose_imu_orientation_error:.3f} rad"
        )


def normalized_xyzw(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise RuntimeError("pose orientation must be a finite xyzw quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise RuntimeError("pose orientation quaternion has near-zero norm")
    return quaternion / norm


def xyzw_to_wxyz(value: np.ndarray) -> np.ndarray:
    normalized = normalized_xyzw(value)
    return normalized[[3, 0, 1, 2]]


def wxyz_to_xyzw(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if quaternion.shape != (4,) or not math.isfinite(norm) or norm <= 1.0e-12:
        raise RuntimeError("motion orientation must be a finite quaternion")
    return (quaternion / norm)[[1, 2, 3, 0]]


def quaternion_yaw(value: np.ndarray) -> float:
    w, x, y, z = np.asarray(value, dtype=np.float64)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quaternion_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    return 2.0 * math.acos(float(np.clip(abs(np.dot(left, right)), 0.0, 1.0)))


def wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi
