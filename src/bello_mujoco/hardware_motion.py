"""Deploy a motion-tracking checkpoint through bello_stack's hardware client."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, replace
import math
from pathlib import Path
import time
from typing import Any

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from bello_mujoco.assets_pipeline.prepare import prepare_asset
from bello_mujoco.checkpoint import (
    CheckpointMetadata,
    CheckpointNormalizationState,
    restore_evaluation_checkpoint,
    validate_motion_clip_contract,
)
from bello_mujoco.envs.motion_tracking import (
    BelloMotionTrackingEnv,
    MotionTrackingState,
)
from bello_mujoco.hardware_locomotion import (
    HardwareFrame,
    actuator_coordinate_names,
    actuator_impedance_gains,
    drain_command_receipts,
    import_hardware_binding,
    limit_target_step,
    normalized_quaternion_xyzw,
    pace_hardware_loop,
    require_async_hardware_binding,
    send_targets,
    start_hardware_controls,
    stop_hardware_controls,
    tilt_from_xyzw,
    validate_hardware_control_config,
    validate_hardware_frame,
    validate_hardware_inventory,
    wait_for_group_state,
    wait_for_hardware_frame,
)
from bello_mujoco.policy import apply_actor
from bello_mujoco.tools.eval_policy import validate_checkpoint_shape
from bello_mujoco.train.ppo import normalize_observation


@dataclass(frozen=True)
class HardwareMotionConfig:
    """Motion source, hardware endpoint, and fail-closed deployment limits."""

    endpoint: str = "http://127.0.0.1:50051"
    groups: tuple[str, ...] = ("lower_body", "upper_body")
    imu_name: str = "imu"
    client_name: str = "bello-motion-policy"
    motion_path: Path | None = None
    start_time_seconds: float = 0.0
    arm: bool = False
    operator_confirmation: str = ""
    connection_timeout_seconds: float = 3.0
    telemetry_timeout_seconds: float = 2.0
    command_validity_seconds: float = 0.08
    acknowledgement_timeout_seconds: float = 0.08
    rpc_timeout_seconds: float = 2.0
    ramp_seconds: float = 3.0
    max_duration_seconds: float = 10.0
    max_motor_age_ms: int = 100
    max_imu_age_ms: int = 100
    max_initial_position_error_radians: float = 0.75
    max_target_step_radians: float = 0.05
    max_tilt_radians: float = 0.70
    max_angular_velocity: float = 20.0
    max_linear_acceleration: float = 100.0
    max_loop_lag_seconds: float = 0.10
    kp_scale: float = 1.0
    kd_scale: float = 1.0
    status_interval_seconds: float = 1.0
    max_estimated_linear_velocity: float = 2.0
    max_estimated_position_error: float = 0.75


@chex.dataclass(frozen=True)
class _HardwareMotionData:
    qpos: jax.Array
    qvel: jax.Array
    xpos: jax.Array
    xquat: jax.Array


@dataclass(frozen=True)
class HardwareMotionStep:
    action: np.ndarray
    physical_target: np.ndarray
    observation: np.ndarray
    reference_time: float
    tilt_radians: float
    estimated_linear_velocity: float
    estimated_position_error: float


class HardwareMotionPolicyAdapter:
    """Build motion observations from joints, IMU, and bounded inertial odometry."""

    def __init__(
        self,
        env: BelloMotionTrackingEnv,
        policy: Any,
        metadata: CheckpointMetadata,
        normalization: CheckpointNormalizationState,
        *,
        start_time_seconds: float = 0.0,
    ) -> None:
        validate_hardware_motion_checkpoint(env, metadata)
        if not math.isfinite(start_time_seconds) or not (
            0.0 <= start_time_seconds < env.motion.duration
        ):
            raise ValueError(
                "start_time_seconds must be finite and inside the motion clip"
            )
        self.env = env
        self.metadata = metadata
        self.start_time_seconds = float(start_time_seconds)
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
        self.kp, self.kd = actuator_impedance_gains(env)
        self._normalization = normalization.observation
        self._actor = eqx.filter_jit(
            lambda observation: apply_actor(policy, observation)
        )
        self._observation = jax.jit(env.observation)
        self._action_ctrl = jax.jit(env._action_to_ctrl)
        self._mj_data = mujoco.MjData(env.model.mj_model)
        self._alignment_xy = np.zeros(2, dtype=np.float32)
        self._alignment_yaw = 0.0
        self._root_position = np.zeros(3, dtype=np.float64)
        self._root_velocity = np.zeros(3, dtype=np.float64)
        self._last_action = np.zeros(env.action_size, dtype=np.float32)
        self._observation_stack: np.ndarray | None = None
        self._step_count = 0
        self._initialized = False

    @property
    def control_timestep(self) -> float:
        return self.env.control_timestep

    @property
    def remaining_duration(self) -> float:
        return self.env.motion.duration - self.start_time_seconds

    @property
    def reference_time(self) -> float:
        return min(
            self.start_time_seconds + self._step_count * self.control_timestep,
            self.env.motion.duration,
        )

    @property
    def nominal_physical_target(self) -> np.ndarray:
        logical = np.asarray(self.env.model.default_joint_target, dtype=np.float64)
        target = self.transform @ logical
        limits = np.asarray(self.env.model.mj_model.actuator_ctrlrange)
        return np.clip(target, limits[:, 0], limits[:, 1])

    def initialize(self, frame: HardwareFrame) -> None:
        orientation = normalized_quaternion_xyzw(frame.orientation_xyzw)
        source = self.env._aligned_reference(
            jnp.asarray(self.start_time_seconds, dtype=jnp.float32),
            jnp.zeros(2, dtype=jnp.float32),
            jnp.zeros((), dtype=jnp.float32),
        )
        reference_yaw = yaw_from_wxyz(np.asarray(source.qpos[3:7]))
        measured_yaw = yaw_from_xyzw(orientation)
        self._alignment_yaw = wrap_angle(measured_yaw - reference_yaw)
        rotated_source_xy = rotate_z(np.asarray(source.qpos[:2]), self._alignment_yaw)
        self._alignment_xy = -rotated_source_xy.astype(np.float32)
        aligned = self.env._aligned_reference(
            jnp.asarray(self.start_time_seconds, dtype=jnp.float32),
            jnp.asarray(self._alignment_xy),
            jnp.asarray(self._alignment_yaw, dtype=jnp.float32),
        )
        self._root_position = np.asarray(aligned.qpos[:3], dtype=np.float64)
        self._root_velocity = np.zeros(3, dtype=np.float64)
        self._last_action = np.zeros(self.env.action_size, dtype=np.float32)
        self._observation_stack = None
        self._step_count = 0
        self._initialized = True

    def precompile(self, frame: HardwareFrame) -> None:
        self.initialize(frame)
        self.step(frame)
        self.initialize(frame)

    def step(self, frame: HardwareFrame) -> HardwareMotionStep:
        if not self._initialized:
            raise RuntimeError("motion adapter must be initialized from hardware")
        dt = self.control_timestep
        acceleration_world = rotate_vector_xyzw(
            frame.orientation_xyzw, frame.linear_acceleration
        )
        previous_velocity = self._root_velocity.copy()
        self._root_velocity = previous_velocity + acceleration_world * dt
        self._root_position = (
            self._root_position
            + previous_velocity * dt
            + 0.5 * acceleration_world * dt * dt
        )
        state = self._state_from_frame(frame)
        observation = np.asarray(
            jax.device_get(self._observation(state)), dtype=np.float32
        )
        if observation.shape != (self.env.observation_size,) or not np.all(
            np.isfinite(observation)
        ):
            raise RuntimeError(
                f"motion hardware observation is invalid: shape={observation.shape}"
            )
        stack_size = self.metadata.ppo.policy.obs_stack_size
        if self._observation_stack is None:
            self._observation_stack = np.repeat(
                observation[None, :], stack_size, axis=0
            )
        else:
            self._observation_stack = np.concatenate(
                (self._observation_stack[1:], observation[None, :]), axis=0
            )
        stacked = jnp.asarray(self._observation_stack.reshape(-1), dtype=jnp.float32)
        normalized = normalize_observation(
            stacked, self._normalization, self.metadata.ppo
        )
        action = np.asarray(jax.device_get(self._actor(normalized)), dtype=np.float32)
        action = np.clip(action, -self.env.action_clip, self.env.action_clip)
        if action.shape != (self.env.action_size,) or not np.all(np.isfinite(action)):
            raise RuntimeError(
                f"policy produced an invalid motion action with shape {action.shape}"
            )
        physical_target = np.asarray(
            jax.device_get(self._action_ctrl(state, jnp.asarray(action))),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(physical_target)):
            raise RuntimeError("policy produced non-finite physical motor targets")
        reference = self.env._aligned_reference(
            jnp.asarray(self.reference_time, dtype=jnp.float32),
            jnp.asarray(self._alignment_xy),
            jnp.asarray(self._alignment_yaw, dtype=jnp.float32),
        )
        estimated_position_error = float(
            np.linalg.norm(self._root_position - np.asarray(reference.qpos[:3]))
        )
        estimated_linear_velocity = float(np.linalg.norm(self._root_velocity))
        result = HardwareMotionStep(
            action=action,
            physical_target=physical_target,
            observation=observation,
            reference_time=self.reference_time,
            tilt_radians=tilt_from_xyzw(frame.orientation_xyzw),
            estimated_linear_velocity=estimated_linear_velocity,
            estimated_position_error=estimated_position_error,
        )
        self._last_action = action
        self._step_count += 1
        return result

    def _state_from_frame(self, frame: HardwareFrame) -> MotionTrackingState:
        logical_position = self.inverse_transform @ frame.positions
        logical_velocity = self.inverse_transform @ frame.velocities
        orientation = normalized_quaternion_xyzw(frame.orientation_xyzw)
        root_qpos = self.env.model.root_qpos_adr
        root_qvel = self.env.model.root_qvel_adr
        self._mj_data.qpos[:] = np.asarray(self.env.model.default_qpos)
        self._mj_data.qvel[:] = 0.0
        self._mj_data.qpos[root_qpos : root_qpos + 3] = self._root_position
        self._mj_data.qpos[root_qpos + 3 : root_qpos + 7] = orientation[[3, 0, 1, 2]]
        self._mj_data.qpos[self.env.model.actuator_qpos_indices] = logical_position
        self._mj_data.qvel[root_qvel : root_qvel + 3] = self._root_velocity
        self._mj_data.qvel[root_qvel + 3 : root_qvel + 6] = rotate_vector_xyzw(
            orientation, frame.angular_velocity
        )
        self._mj_data.qvel[self.env.model.actuator_qvel_indices] = logical_velocity
        mujoco.mj_forward(self.env.model.mj_model, self._mj_data)
        data = _HardwareMotionData(
            qpos=jnp.asarray(np.asarray(self._mj_data.qpos), dtype=jnp.float32),
            qvel=jnp.asarray(np.asarray(self._mj_data.qvel), dtype=jnp.float32),
            xpos=jnp.asarray(np.asarray(self._mj_data.xpos), dtype=jnp.float32),
            xquat=jnp.asarray(np.asarray(self._mj_data.xquat), dtype=jnp.float32),
        )
        return MotionTrackingState(
            data=data,
            last_action=jnp.asarray(self._last_action),
            reference_time=jnp.asarray(self.reference_time, dtype=jnp.float32),
            alignment_xy=jnp.asarray(self._alignment_xy),
            alignment_yaw=jnp.asarray(self._alignment_yaw, dtype=jnp.float32),
            joint_default_offset=jnp.zeros(self.env.action_size, dtype=jnp.float32),
            next_push_step=jnp.asarray(0, dtype=jnp.int32),
            sampling_failure_ema=jnp.zeros(
                self.env.sampling_bin_count, dtype=jnp.float32
            ),
            step_count=jnp.asarray(self._step_count, dtype=jnp.int32),
            episode_return=jnp.asarray(0.0, dtype=jnp.float32),
            episode_length=jnp.asarray(self._step_count, dtype=jnp.int32),
            last_episode_return=jnp.asarray(0.0, dtype=jnp.float32),
            last_episode_length=jnp.asarray(0, dtype=jnp.int32),
        )


def load_hardware_motion_policy(
    checkpoint_path: Path,
    *,
    motion_path: Path | None = None,
    start_time_seconds: float = 0.0,
) -> tuple[HardwareMotionPolicyAdapter, CheckpointMetadata]:
    train_state, metadata, normalization = restore_evaluation_checkpoint(
        checkpoint_path
    )
    if (
        metadata.task != "motion_tracking"
        or metadata.motion_env is None
        or metadata.motion is None
    ):
        raise ValueError(f"checkpoint task is {metadata.task!r}, not motion_tracking")
    resolved_motion_path = (
        metadata.motion.path if motion_path is None else Path(motion_path)
    )
    env_config = replace(metadata.motion_env, motion_path=resolved_motion_path)
    prepared = prepare_asset(metadata.asset)
    env = BelloMotionTrackingEnv.load_prepared_native(
        prepared,
        env_config,
        actuator_config=metadata.asset.actuator_config,
    )
    validate_motion_clip_contract(resolved_motion_path, env.motion, metadata.motion)
    validate_checkpoint_shape(env, metadata)
    return (
        HardwareMotionPolicyAdapter(
            env,
            train_state.policy,
            metadata,
            normalization,
            start_time_seconds=start_time_seconds,
        ),
        metadata,
    )


def validate_hardware_motion_checkpoint(
    env: BelloMotionTrackingEnv, metadata: CheckpointMetadata
) -> None:
    if metadata.task != "motion_tracking":
        raise ValueError(
            "hardware motion playback requires a motion-tracking checkpoint"
        )
    if metadata.motion_env is None or metadata.motion is None:
        raise ValueError("motion-tracking checkpoint is missing its motion contract")
    if metadata.asset.end_effector_sphere_mode != "both":
        raise ValueError("hardware motion playback requires sphere hands")
    availability = metadata.asset.joint_availability
    unexpected = sorted(
        availability.enabled_joint_names
        & {
            "waist_roll_joint",
            "waist_pitch_joint",
            "neck_yaw_joint",
            "head_pitch_joint",
        }
    )
    if unexpected:
        raise ValueError(
            "motion checkpoint contains joints absent from bello_stack hardware: "
            + ", ".join(unexpected)
        )
    if not availability.waist_yaw:
        raise ValueError("hardware motion playback requires waist_yaw_joint")
    if env.action_size != 25 or env.model.mj_model.nu != 25:
        raise ValueError(
            "hardware motion playback requires the 25-coordinate fixed-head model"
        )
    expected_distribution = (
        "gaussian" if env.config.formulation == "beyondmimic" else "tanh_gaussian"
    )
    if metadata.ppo.policy.action_distribution != expected_distribution:
        raise ValueError(
            f"{env.config.formulation} hardware playback requires "
            f"{expected_distribution!r} policy actions"
        )


def validate_hardware_motion_config(
    config: HardwareMotionConfig, env: BelloMotionTrackingEnv | None = None
) -> None:
    validate_hardware_control_config(config)
    if (
        not math.isfinite(config.max_initial_position_error_radians)
        or config.max_initial_position_error_radians <= 0.0
    ):
        raise ValueError(
            "max_initial_position_error_radians must be positive and finite"
        )
    if not math.isfinite(config.start_time_seconds) or config.start_time_seconds < 0.0:
        raise ValueError("start_time_seconds must be finite and nonnegative")
    for name in ("max_estimated_linear_velocity", "max_estimated_position_error"):
        value = getattr(config, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if env is not None and config.start_time_seconds >= env.motion.duration:
        raise ValueError("start_time_seconds must be inside the checkpoint motion")


def run_hardware_motion(
    checkpoint_path: Path,
    config: HardwareMotionConfig = HardwareMotionConfig(),
    *,
    hardware_binding: Any | None = None,
) -> None:
    validate_hardware_motion_config(config)
    adapter, metadata = load_hardware_motion_policy(
        checkpoint_path,
        motion_path=config.motion_path,
        start_time_seconds=config.start_time_seconds,
    )
    validate_hardware_motion_config(config, adapter.env)
    binding = hardware_binding or import_hardware_binding()
    if config.arm:
        require_async_hardware_binding(binding)
    with binding.Robot.connect(
        config.endpoint, timeout=config.connection_timeout_seconds
    ) as robot:
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
            validate_hardware_frame(frame, config, expected_group_state="IDLE")
            adapter.precompile(frame)
            initial_target = adapter.nominal_physical_target
            initial_error = float(np.max(np.abs(frame.positions - initial_target)))
            print(
                {
                    "hardware_motion_preflight": "passed",
                    "endpoint": config.endpoint,
                    "groups": config.groups,
                    "imu": config.imu_name,
                    "coordinates": len(adapter.physical_coordinate_names),
                    "formulation": adapter.env.config.formulation,
                    "motion_sha256": metadata.motion.content_sha256,
                    "start_time_seconds": config.start_time_seconds,
                    "remaining_duration_seconds": adapter.remaining_duration,
                    "nominal_ramp_distance_max": initial_error,
                    "odometry": "gravity-removed IMU dead reckoning",
                    "armed": config.arm,
                }
            )
            if not config.arm:
                print(
                    "Read-only preflight complete; no lease was acquired and no motor command was sent."
                )
                return
            control_physical_motion(binding, groups, states, frame, adapter, config)


def control_physical_motion(
    binding: Any,
    groups: Mapping[str, Any],
    states: Any,
    initial_frame: HardwareFrame,
    adapter: HardwareMotionPolicyAdapter,
    config: HardwareMotionConfig,
) -> None:
    with ExitStack() as resources:
        controls = {
            name: resources.enter_context(
                group.control(
                    config.client_name,
                    command_validity=config.command_validity_seconds,
                    acknowledgement_timeout=config.acknowledgement_timeout_seconds,
                    rpc_timeout=config.rpc_timeout_seconds,
                )
            )
            for name, group in groups.items()
        }
        executor = resources.enter_context(
            ThreadPoolExecutor(
                max_workers=len(config.groups),
                thread_name_prefix="bello-group-control",
            )
        )
        start_hardware_controls(
            binding,
            controls,
            groups,
            initial_frame.positions,
            adapter,
            config,
            executor,
        )
        pending_receipts: list[Any] = []
        wait_for_group_state(
            states,
            adapter.physical_coordinate_names,
            config,
            binding.StateUnavailableError,
            "RUNNING",
        )
        previous_target = initial_frame.positions.copy()
        deadline = time.monotonic()
        ramp_steps = max(1, math.ceil(config.ramp_seconds / adapter.control_timestep))
        for step in range(1, ramp_steps + 1):
            fraction = step / ramp_steps
            requested = initial_frame.positions + fraction * (
                adapter.nominal_physical_target - initial_frame.positions
            )
            previous_target = limit_target_step(
                previous_target, requested, config.max_target_step_radians
            )
            send_targets(
                binding,
                controls,
                groups,
                previous_target,
                adapter,
                config,
                executor,
                pending_receipts,
            )
            deadline = pace_hardware_loop(deadline, adapter.control_timestep, config)
            frame = wait_for_hardware_frame(
                states,
                adapter.physical_coordinate_names,
                config.groups,
                config.imu_name,
                binding.StateUnavailableError,
                config.telemetry_timeout_seconds,
            )
            validate_hardware_frame(frame, config, expected_group_state="RUNNING")

        require_nominal_pose(frame, adapter, config.max_initial_position_error_radians)
        adapter.initialize(frame)
        duration = min(config.max_duration_seconds, adapter.remaining_duration)
        start_time = time.monotonic()
        next_status = start_time
        deadline = start_time
        step = 0
        try:
            while time.monotonic() - start_time < duration:
                frame = wait_for_hardware_frame(
                    states,
                    adapter.physical_coordinate_names,
                    config.groups,
                    config.imu_name,
                    binding.StateUnavailableError,
                    config.telemetry_timeout_seconds,
                )
                validate_hardware_frame(frame, config, expected_group_state="RUNNING")
                policy_step = adapter.step(frame)
                if (
                    policy_step.estimated_linear_velocity
                    > config.max_estimated_linear_velocity
                ):
                    raise RuntimeError(
                        "estimated root velocity exceeds motion safety limit: "
                        f"{policy_step.estimated_linear_velocity:.3f} m/s"
                    )
                if (
                    policy_step.estimated_position_error
                    > config.max_estimated_position_error
                ):
                    raise RuntimeError(
                        "estimated root position error exceeds motion safety limit: "
                        f"{policy_step.estimated_position_error:.3f} m"
                    )
                previous_target = limit_target_step(
                    previous_target,
                    policy_step.physical_target,
                    config.max_target_step_radians,
                )
                send_targets(
                    binding,
                    controls,
                    groups,
                    previous_target,
                    adapter,
                    config,
                    executor,
                    pending_receipts,
                )
                step += 1
                now = time.monotonic()
                if now >= next_status:
                    print(
                        {
                            "hardware_motion_step": step,
                            "elapsed_seconds": now - start_time,
                            "reference_time": policy_step.reference_time,
                            "tilt_radians": policy_step.tilt_radians,
                            "action_abs_max": float(np.max(np.abs(policy_step.action))),
                            "estimated_linear_velocity": (
                                policy_step.estimated_linear_velocity
                            ),
                            "estimated_position_error": (
                                policy_step.estimated_position_error
                            ),
                            "motor_age_ms": frame.max_motor_age_ms,
                            "imu_age_ms": frame.imu_age_ms,
                        }
                    )
                    next_status = now + config.status_interval_seconds
                deadline = pace_hardware_loop(
                    deadline, adapter.control_timestep, config
                )
            drain_command_receipts(pending_receipts)
        except KeyboardInterrupt:
            print("Hardware motion interrupted; stopping and disabling groups.")
        stop_hardware_controls(controls, config, executor)


def require_nominal_pose(
    frame: HardwareFrame,
    adapter: HardwareMotionPolicyAdapter,
    tolerance_radians: float,
) -> None:
    error = np.abs(frame.positions - adapter.nominal_physical_target)
    index = int(np.argmax(error))
    if error[index] > tolerance_radians:
        name = adapter.physical_coordinate_names[index]
        raise RuntimeError(
            "robot did not reach the nominal pose during the ramp: "
            f"{name} remains {error[index]:.3f} rad away "
            f"(measured={frame.positions[index]:.3f}, "
            f"nominal={adapter.nominal_physical_target[index]:.3f}); "
            f"arrival tolerance is {tolerance_radians:.3f} rad"
        )


def yaw_from_xyzw(quaternion: np.ndarray) -> float:
    x, y, z, w = normalized_quaternion_xyzw(quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_from_wxyz(quaternion: np.ndarray) -> float:
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        raise ValueError("reference quaternion must be finite wxyz")
    return yaw_from_xyzw(value[[1, 2, 3, 0]])


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def rotate_z(value_xy: np.ndarray, angle: float) -> np.ndarray:
    value = np.asarray(value_xy, dtype=np.float64)
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        (cosine * value[0] - sine * value[1], sine * value[0] + cosine * value[1])
    )


def rotate_vector_xyzw(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = normalized_quaternion_xyzw(quaternion)
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise RuntimeError("IMU vector must contain three finite values")
    xyz = q[:3]
    return value + 2.0 * np.cross(xyz, np.cross(xyz, value) + q[3] * value)
