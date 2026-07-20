"""Deploy a locomotion checkpoint through bello_stack's hardware client."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
import importlib
import math
from pathlib import Path
import time
from typing import Any

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
)
from bello_mujoco.envs.locomotion import BelloLocomotionEnv, GaitConfig
from bello_mujoco.policy import apply_actor
from bello_mujoco.render import quat_rotate_inverse_np
from bello_mujoco.rerun_eval import GamepadCommandSource, GamepadConfig, GamepadInput
from bello_mujoco.tools.eval_policy import validate_checkpoint_shape
from bello_mujoco.train.ppo import normalize_observation

ARM_CONFIRMATION = "I_HAVE_A_SPOTTER_AND_ESTOP_READY"
FAULT_GROUP_STATES = frozenset({"FAULT", "SHUTDOWN", "UNSPECIFIED"})


@dataclass(frozen=True)
class HardwareLocomotionConfig:
    """Hardware endpoint, command, and fail-closed deployment limits."""

    endpoint: str = "http://127.0.0.1:50051"
    groups: tuple[str, ...] = ("lower_body", "upper_body")
    imu_name: str = "imu"
    client_name: str = "bello-locomotion-policy"
    command: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gamepad: GamepadConfig = GamepadConfig()
    wait_for_start_button: bool = True
    arm: bool = False
    operator_confirmation: str = ""
    connection_timeout_seconds: float = 3.0
    telemetry_timeout_seconds: float = 2.0
    command_validity_seconds: float = 0.08
    acknowledgement_timeout_seconds: float = 0.08
    rpc_timeout_seconds: float = 2.0
    ramp_seconds: float = 10.0
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


@dataclass(frozen=True)
class HardwareFrame:
    positions: np.ndarray
    velocities: np.ndarray
    orientation_xyzw: np.ndarray
    angular_velocity: np.ndarray
    linear_acceleration: np.ndarray
    max_motor_age_ms: int
    imu_age_ms: int
    group_states: tuple[str, ...]


@dataclass(frozen=True)
class HardwarePolicyStep:
    action: np.ndarray
    physical_target: np.ndarray
    observation: np.ndarray
    tilt_radians: float


class HardwarePolicyAdapter:
    """Reconstruct deployable observations and physical motor targets."""

    def __init__(
        self,
        env: BelloLocomotionEnv,
        policy: Any,
        metadata: CheckpointMetadata,
        normalization: CheckpointNormalizationState,
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
        self.default_joint_target = np.asarray(
            env.model.default_joint_target, dtype=np.float64
        )
        self.kp, self.kd = actuator_impedance_gains(env)
        self.gait_frequency = deployment_gait_frequency(env.config.gait)
        self._normalization = normalization.observation
        self._actor = eqx.filter_jit(
            lambda observation: apply_actor(policy, observation)
        )
        zero_offset = jnp.zeros(len(self.physical_coordinate_names), dtype=jnp.float32)
        self._action_ctrl = jax.jit(
            lambda action: env.action_to_ctrl(action, zero_offset)
        )
        self._obs_stack: np.ndarray | None = None
        self._last_action = np.zeros(env.action_size, dtype=np.float32)
        self._step_count = 0

    @property
    def control_timestep(self) -> float:
        return self.env.control_timestep

    @property
    def nominal_physical_target(self) -> np.ndarray:
        return np.asarray(
            jax.device_get(
                self._action_ctrl(jnp.zeros(self.env.action_size, dtype=jnp.float32))
            ),
            dtype=np.float64,
        )

    def reset(self) -> None:
        self._obs_stack = None
        self._last_action = np.zeros(self.env.action_size, dtype=np.float32)
        self._step_count = 0

    def precompile(self) -> None:
        nominal = self.nominal_physical_target
        frame = HardwareFrame(
            positions=nominal,
            velocities=np.zeros_like(nominal),
            orientation_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
            angular_velocity=np.zeros(3),
            linear_acceleration=np.zeros(3),
            max_motor_age_ms=0,
            imu_age_ms=0,
            group_states=(),
        )
        self.step(frame, np.zeros(3, dtype=np.float32))
        self.reset()

    def observation(self, frame: HardwareFrame, command: np.ndarray) -> np.ndarray:
        logical_position = self.inverse_transform @ frame.positions
        logical_velocity = self.inverse_transform @ frame.velocities
        projected_gravity = projected_gravity_from_xyzw(frame.orientation_xyzw)
        foot_phase = foot_phase_sin_cos(
            self.env.config.gait,
            step_count=self._step_count,
            control_timestep=self.control_timestep,
            command=command,
            randomized_frequency=self.gait_frequency,
        )
        scale = self.env.config.observation.scale
        components = {
            "imu_linear_acceleration": frame.linear_acceleration * scale.base_lin_acc,
            "imu_angular_velocity": frame.angular_velocity * scale.base_ang_vel,
            "projected_gravity": projected_gravity * scale.projected_gravity,
            "foot_phase": foot_phase * scale.foot_phase,
            "command": command * scale.command,
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

    def step(self, frame: HardwareFrame, command: np.ndarray) -> HardwarePolicyStep:
        observation = self.observation(frame, command)
        stack_size = self.metadata.ppo.policy.obs_stack_size
        if self._obs_stack is None:
            self._obs_stack = np.repeat(observation[None, :], stack_size, axis=0)
        else:
            self._obs_stack = np.concatenate(
                (self._obs_stack[1:], observation[None, :]), axis=0
            )
        stacked = jnp.asarray(self._obs_stack.reshape(-1), dtype=jnp.float32)
        normalized = normalize_observation(
            stacked,
            self._normalization,
            self.metadata.ppo,
        )
        action = np.asarray(jax.device_get(self._actor(normalized)), dtype=np.float32)
        action = np.clip(action, -self.env.action_clip, self.env.action_clip)
        if action.shape != (self.env.action_size,) or not np.all(np.isfinite(action)):
            raise RuntimeError(
                f"policy produced an invalid hardware action with shape {action.shape}"
            )
        physical_target = np.asarray(
            jax.device_get(self._action_ctrl(jnp.asarray(action))), dtype=np.float64
        )
        if not np.all(np.isfinite(physical_target)):
            raise RuntimeError("policy produced non-finite physical motor targets")
        self._last_action = action
        self._step_count += 1
        return HardwarePolicyStep(
            action=action,
            physical_target=physical_target,
            observation=observation,
            tilt_radians=tilt_from_xyzw(frame.orientation_xyzw),
        )


def load_hardware_policy(
    checkpoint_path: Path,
) -> tuple[HardwarePolicyAdapter, CheckpointMetadata]:
    train_state, metadata, normalization = restore_evaluation_checkpoint(
        checkpoint_path
    )
    if metadata.task != "locomotion":
        raise ValueError(f"checkpoint task is {metadata.task!r}, not 'locomotion'")
    if any(
        value is None
        for value in (metadata.env, metadata.reward, metadata.command, metadata.done)
    ):
        raise ValueError("locomotion checkpoint is missing its environment contract")
    prepared = prepare_asset(metadata.asset)
    env = BelloLocomotionEnv.load_prepared(
        prepared,
        warp=metadata.warp,
        batch_size=1,
        env_config=metadata.env,
        actuator_config=metadata.asset.actuator_config,
        reward_config=metadata.reward,
        command_config=metadata.command,
        done_config=metadata.done,
        load_mjx=False,
    )
    validate_checkpoint_shape(env, metadata)
    return HardwarePolicyAdapter(
        env, train_state.policy, metadata, normalization
    ), metadata


def validate_hardware_checkpoint(
    env: BelloLocomotionEnv, metadata: CheckpointMetadata
) -> None:
    if metadata.task != "locomotion":
        raise ValueError("hardware locomotion requires a locomotion checkpoint")
    if metadata.asset.end_effector_sphere_mode != "both":
        raise ValueError("hardware locomotion requires sphere hands")
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
            "checkpoint morphology contains joints absent from bello_stack hardware: "
            + ", ".join(unexpected)
            + "; train with experiments/train/hardware_locomotion.yaml"
        )
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


def validate_hardware_config(
    config: HardwareLocomotionConfig, metadata: CheckpointMetadata | None = None
) -> None:
    validate_hardware_control_config(config)
    command = np.asarray(config.command, dtype=np.float64)
    if command.shape != (3,) or not np.all(np.isfinite(command)):
        raise ValueError("command must contain three finite values")
    if metadata is not None:
        validate_command(command, metadata)
    if config.gamepad.controller_index < 0:
        raise ValueError("gamepad.controller_index must be nonnegative")
    if not 0.0 <= config.gamepad.deadzone < 1.0:
        raise ValueError("gamepad.deadzone must be in [0, 1)")
    if config.wait_for_start_button and not config.gamepad.enabled:
        raise ValueError("wait_for_start_button requires gamepad.enabled")


def validate_hardware_control_config(config: Any) -> None:
    """Validate controls shared by physical locomotion and motion playback."""

    if not config.endpoint.strip():
        raise ValueError("endpoint must not be empty")
    if not config.groups or any(not group.strip() for group in config.groups):
        raise ValueError("groups must contain non-empty names")
    if len(set(config.groups)) != len(config.groups):
        raise ValueError("groups must not contain duplicates")
    if not config.imu_name.strip() or not config.client_name.strip():
        raise ValueError("imu_name and client_name must not be empty")
    positive = {
        "connection_timeout_seconds": config.connection_timeout_seconds,
        "telemetry_timeout_seconds": config.telemetry_timeout_seconds,
        "command_validity_seconds": config.command_validity_seconds,
        "acknowledgement_timeout_seconds": config.acknowledgement_timeout_seconds,
        "rpc_timeout_seconds": config.rpc_timeout_seconds,
        "ramp_seconds": config.ramp_seconds,
        "max_duration_seconds": config.max_duration_seconds,
        "max_initial_position_error_radians": (
            config.max_initial_position_error_radians
        ),
        "max_target_step_radians": config.max_target_step_radians,
        "max_tilt_radians": config.max_tilt_radians,
        "max_angular_velocity": config.max_angular_velocity,
        "max_linear_acceleration": config.max_linear_acceleration,
        "max_loop_lag_seconds": config.max_loop_lag_seconds,
        "kp_scale": config.kp_scale,
        "kd_scale": config.kd_scale,
        "status_interval_seconds": config.status_interval_seconds,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if config.kp_scale > 1.0 or config.kd_scale > 1.0:
        raise ValueError("kp_scale and kd_scale must not exceed the trained gains")
    if config.max_motor_age_ms <= 0 or config.max_imu_age_ms <= 0:
        raise ValueError("telemetry age limits must be positive")
    if config.arm and config.operator_confirmation != ARM_CONFIRMATION:
        raise ValueError(
            "arming physical hardware requires operator_confirmation="
            f"{ARM_CONFIRMATION}"
        )


def validate_command(command: np.ndarray, metadata: CheckpointMetadata) -> None:
    command_config = metadata.command
    if command_config is None:
        raise ValueError("locomotion checkpoint is missing command metadata")
    if not command_config.enabled:
        if np.any(command != 0.0):
            raise ValueError(
                "checkpoint disables commands; hardware command must be zero"
            )
        return
    bounds = np.asarray(command_config.range_array(), dtype=np.float64)
    if np.any(command < bounds[:, 0]) or np.any(command > bounds[:, 1]):
        raise ValueError(
            f"hardware command {command.tolist()} exceeds checkpoint bounds "
            f"{bounds.tolist()}"
        )


def command_ranges(
    metadata: CheckpointMetadata,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    if metadata.command is None:
        raise ValueError("locomotion checkpoint is missing command metadata")
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


def hardware_command(
    config: HardwareLocomotionConfig,
    metadata: CheckpointMetadata,
    gamepad_input: GamepadInput,
) -> np.ndarray:
    command = (
        gamepad_input.command
        if config.gamepad.enabled
        else np.asarray(config.command, dtype=np.float32)
    )
    command = np.asarray(command, dtype=np.float32)
    if command.shape != (3,) or not np.all(np.isfinite(command)):
        raise RuntimeError("gamepad produced an invalid hardware command")
    validate_command(command, metadata)
    return command


def report_gamepad_connection(gamepad_input: GamepadInput) -> None:
    if not gamepad_input.connection_changed:
        return
    state = "connected" if gamepad_input.connected else "disconnected"
    suffix = (
        f": {gamepad_input.controller_name}"
        if gamepad_input.controller_name is not None
        else ""
    )
    print(f"Hardware gamepad {state}{suffix}; command is zero while disconnected.")


def actuator_coordinate_names(env: BelloLocomotionEnv) -> tuple[str, ...]:
    model = env.model.mj_model
    result = []
    for actuator_id in range(model.nu):
        transmission = int(model.actuator_trntype[actuator_id])
        object_id = int(model.actuator_trnid[actuator_id, 0])
        if transmission == int(mujoco.mjtTrn.mjTRN_JOINT):
            object_type = mujoco.mjtObj.mjOBJ_JOINT
        elif transmission == int(mujoco.mjtTrn.mjTRN_TENDON):
            object_type = mujoco.mjtObj.mjOBJ_TENDON
        else:
            raise ValueError(
                "hardware deployment supports only joint and tendon actuators; "
                f"actuator {actuator_id} uses transmission {transmission}"
            )
        name = mujoco.mj_id2name(model, object_type, object_id)
        if name is None:
            raise ValueError(f"actuator {actuator_id} has an unnamed coordinate")
        result.append(name)
    return tuple(result)


def actuator_impedance_gains(env: BelloLocomotionEnv) -> tuple[np.ndarray, np.ndarray]:
    model = env.model.mj_model
    kp = np.asarray(model.actuator_gainprm[:, 0], dtype=np.float64)
    kd = -np.asarray(model.actuator_biasprm[:, 2], dtype=np.float64)
    if (
        kp.shape != (model.nu,)
        or kd.shape != (model.nu,)
        or not np.all(np.isfinite(kp))
        or not np.all(np.isfinite(kd))
        or np.any(kp < 0.0)
        or np.any(kd < 0.0)
    ):
        raise ValueError("generated hardware actuator gains are invalid")
    return kp, kd


def projected_gravity_from_xyzw(orientation_xyzw: np.ndarray) -> np.ndarray:
    orientation = normalized_quaternion_xyzw(orientation_xyzw)
    wxyz = orientation[[3, 0, 1, 2]]
    return quat_rotate_inverse_np(
        wxyz.astype(np.float32), np.asarray((0.0, 0.0, -1.0), dtype=np.float32)
    )


def tilt_from_xyzw(orientation_xyzw: np.ndarray) -> float:
    projected = projected_gravity_from_xyzw(orientation_xyzw)
    return float(np.arccos(np.clip(-projected[2], -1.0, 1.0)))


def normalized_quaternion_xyzw(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise RuntimeError("hardware IMU orientation is not a finite xyzw quaternion")
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm < 0.9 or norm > 1.1:
        raise RuntimeError(f"hardware IMU quaternion norm is invalid: {norm}")
    return quaternion / norm


def deployment_gait_frequency(config: GaitConfig) -> float:
    if config.randomize_frequency:
        return 0.5 * (config.frequency_range[0] + config.frequency_range[1])
    return 0.0


def foot_phase_sin_cos(
    config: GaitConfig,
    *,
    step_count: int,
    control_timestep: float,
    command: np.ndarray,
    randomized_frequency: float,
) -> np.ndarray:
    if float(np.linalg.norm(command)) <= config.standing_command_threshold:
        return np.zeros(4, dtype=np.float32)
    if config.randomize_frequency:
        frequency = randomized_frequency
    else:
        frequency = np.clip(
            1.0 / config.cycle_time
            + config.command_speed_frequency_gain * float(np.linalg.norm(command[:2])),
            config.min_frequency,
            config.max_frequency,
        )
    phase = (step_count * control_timestep * frequency) % 1.0
    angles = 2.0 * np.pi * (phase + np.asarray(config.phase_offsets, dtype=np.float64))
    return np.concatenate((np.cos(angles), np.sin(angles))).astype(np.float32)


def limit_target_step(
    previous: np.ndarray, requested: np.ndarray, maximum_step: float
) -> np.ndarray:
    previous = np.asarray(previous, dtype=np.float64)
    requested = np.asarray(requested, dtype=np.float64)
    if previous.shape != requested.shape:
        raise ValueError("previous and requested targets must have the same shape")
    return previous + np.clip(requested - previous, -maximum_step, maximum_step)


def import_hardware_binding() -> Any:
    try:
        return importlib.import_module("bello_hardware")
    except ImportError as error:
        raise RuntimeError(
            "bello_hardware is not installed; run with `uv run --with "
            "~/bello_stack/components/bello-hardware-python "
            "bello-run-locomotion-hardware ...`"
        ) from error


def require_async_hardware_binding(binding: Any) -> None:
    control_session = getattr(binding, "ControlSession", None)
    if control_session is None or not hasattr(control_session, "send_async"):
        raise RuntimeError(
            "armed hardware execution requires bello_hardware.ControlSession.send_async; "
            "rebuild the local binding (use `uv run --no-cache --with "
            "~/bello_stack/components/bello-hardware-python ...`)"
        )


def run_hardware_locomotion(
    checkpoint_path: Path,
    config: HardwareLocomotionConfig = HardwareLocomotionConfig(),
    *,
    hardware_binding: Any | None = None,
    gamepad_factory: Any = GamepadCommandSource,
) -> None:
    validate_hardware_config(config)
    adapter, metadata = load_hardware_policy(checkpoint_path)
    validate_hardware_config(config, metadata)
    adapter.precompile()
    binding = hardware_binding or import_hardware_binding()
    if config.arm:
        require_async_hardware_binding(binding)
    with ExitStack() as resources:
        gamepad = gamepad_factory(config.gamepad, command_ranges(metadata))
        resources.callback(gamepad.close)
        gamepad_input = gamepad.poll()
        report_gamepad_connection(gamepad_input)
        command = hardware_command(config, metadata, gamepad_input)

        robot = resources.enter_context(
            binding.Robot.connect(
                config.endpoint, timeout=config.connection_timeout_seconds
            )
        )
        groups = validate_hardware_inventory(robot, adapter, config, binding)
        states = resources.enter_context(
            robot.states(
                groups=config.groups,
                imus=(config.imu_name,),
                reconnect=False,
            )
        )
        frame = wait_for_hardware_frame(
            states,
            adapter.physical_coordinate_names,
            config.groups,
            config.imu_name,
            binding.StateUnavailableError,
            config.telemetry_timeout_seconds,
        )
        validate_hardware_frame(frame, config, expected_group_state="IDLE")
        initial_error = float(
            np.max(np.abs(frame.positions - adapter.nominal_physical_target))
        )
        if initial_error > config.max_initial_position_error_radians:
            raise RuntimeError(
                "robot is too far from the checkpoint nominal pose: "
                f"max error {initial_error:.3f} rad exceeds "
                f"{config.max_initial_position_error_radians:.3f} rad"
            )
        print(
            {
                "hardware_preflight": "passed",
                "endpoint": config.endpoint,
                "groups": config.groups,
                "imu": config.imu_name,
                "coordinates": len(adapter.physical_coordinate_names),
                "policy_joints": adapter.env.action_size,
                "control_rate_hz": 1.0 / adapter.control_timestep,
                "command_source": "gamepad" if config.gamepad.enabled else "fixed",
                "gamepad_connected": gamepad_input.connected,
                "command": tuple(float(value) for value in command),
                "initial_pose_error_max": initial_error,
                "armed": config.arm,
            }
        )
        if not config.arm:
            print(
                "Read-only preflight complete; no lease was acquired and no motor command was sent."
            )
            return
        control_physical_robot(
            binding,
            groups,
            states,
            frame,
            adapter,
            config,
            gamepad,
        )


def validate_hardware_inventory(
    robot: Any,
    adapter: HardwarePolicyAdapter,
    config: HardwareLocomotionConfig,
    binding: Any,
) -> dict[str, Any]:
    unknown = sorted(set(config.groups) - set(robot.group_names))
    if unknown:
        raise ValueError(f"hardware inventory is missing motor groups: {unknown}")
    groups = {name: robot.group(name) for name in config.groups}
    coordinates = tuple(
        coordinate for group in groups.values() for coordinate in group.coordinates
    )
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("hardware groups contain duplicate coordinates")
    expected = set(adapter.physical_coordinate_names)
    actual = set(coordinates)
    if actual != expected:
        raise ValueError(
            "checkpoint/hardware coordinate mismatch; missing from hardware="
            f"{sorted(expected - actual)}, unexpected hardware="
            f"{sorted(actual - expected)}"
        )
    policy_rate = 1.0 / adapter.control_timestep
    for group in groups.values():
        if group.control_rate_hz + 1.0e-6 < policy_rate:
            raise ValueError(
                f"hardware group {group.name!r} runs at {group.control_rate_hz} Hz, "
                f"below the {policy_rate} Hz policy rate"
            )
        if group.initial_state != binding.GroupState.IDLE:
            raise RuntimeError(
                f"hardware group {group.name!r} must start IDLE, got "
                f"{group.initial_state.name}"
            )
        for motor in group.motors:
            if motor.status is None or motor.status.state != binding.DeviceState.ONLINE:
                state = "missing" if motor.status is None else motor.status.state.name
                raise RuntimeError(f"motor {motor.name!r} is not online: {state}")
    matching_imus = [imu for imu in robot.inventory.imus if imu.name == config.imu_name]
    if len(matching_imus) != 1:
        raise ValueError(f"hardware inventory does not contain IMU {config.imu_name!r}")
    imu = matching_imus[0]
    if imu.status is None or imu.status.state != binding.DeviceState.ONLINE:
        state = "missing" if imu.status is None else imu.status.state.name
        raise RuntimeError(f"IMU {config.imu_name!r} is not online: {state}")
    return groups


def wait_for_hardware_frame(
    states: Any,
    coordinate_names: tuple[str, ...],
    group_names: tuple[str, ...],
    imu_name: str,
    unavailable_error: type[BaseException],
    timeout_seconds: float,
) -> HardwareFrame:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            motors = tuple(states.coordinate(name) for name in coordinate_names)
            imu = states.imu(imu_name)
            groups = tuple(states.group(name) for name in group_names)
            return HardwareFrame(
                positions=np.asarray([motor.position for motor in motors]),
                velocities=np.asarray([motor.velocity for motor in motors]),
                orientation_xyzw=np.asarray(
                    (
                        imu.orientation.x,
                        imu.orientation.y,
                        imu.orientation.z,
                        imu.orientation.w,
                    )
                ),
                angular_velocity=np.asarray(
                    (
                        imu.angular_velocity.x,
                        imu.angular_velocity.y,
                        imu.angular_velocity.z,
                    )
                ),
                linear_acceleration=np.asarray(
                    (
                        imu.linear_acceleration.x,
                        imu.linear_acceleration.y,
                        imu.linear_acceleration.z,
                    )
                ),
                max_motor_age_ms=max(motor.feedback_age_ms for motor in motors),
                imu_age_ms=imu.sample_age_ms,
                group_states=tuple(group.state.name for group in groups),
            )
        except unavailable_error:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0 or not states.wait_for_change(timeout=remaining):
                raise TimeoutError("timed out waiting for complete hardware telemetry")


def validate_hardware_frame(
    frame: HardwareFrame,
    config: HardwareLocomotionConfig,
    *,
    expected_group_state: str | None = None,
) -> None:
    values = (
        frame.positions,
        frame.velocities,
        frame.angular_velocity,
        frame.linear_acceleration,
    )
    if any(not np.all(np.isfinite(value)) for value in values):
        raise RuntimeError("hardware telemetry contains non-finite values")
    normalized_quaternion_xyzw(frame.orientation_xyzw)
    if frame.max_motor_age_ms > config.max_motor_age_ms:
        raise RuntimeError(f"motor telemetry is stale: {frame.max_motor_age_ms} ms")
    if frame.imu_age_ms > config.max_imu_age_ms:
        raise RuntimeError(f"IMU telemetry is stale: {frame.imu_age_ms} ms")
    bad_states = sorted(set(frame.group_states) & FAULT_GROUP_STATES)
    if bad_states:
        raise RuntimeError(f"hardware group entered unsafe state: {bad_states}")
    if expected_group_state is not None and any(
        state != expected_group_state for state in frame.group_states
    ):
        raise RuntimeError(
            f"hardware groups must be {expected_group_state}, got {frame.group_states}"
        )
    tilt = tilt_from_xyzw(frame.orientation_xyzw)
    if tilt > config.max_tilt_radians:
        raise RuntimeError(
            f"robot tilt {tilt:.3f} rad exceeds {config.max_tilt_radians:.3f} rad"
        )
    angular_speed = float(np.linalg.norm(frame.angular_velocity))
    if angular_speed > config.max_angular_velocity:
        raise RuntimeError(
            f"IMU angular velocity {angular_speed:.3f} exceeds safety limit"
        )
    acceleration = float(np.linalg.norm(frame.linear_acceleration))
    if acceleration > config.max_linear_acceleration:
        raise RuntimeError(f"IMU acceleration {acceleration:.3f} exceeds safety limit")


def control_physical_robot(
    binding: Any,
    groups: Mapping[str, Any],
    states: Any,
    initial_frame: HardwareFrame,
    adapter: HardwarePolicyAdapter,
    config: HardwareLocomotionConfig,
    gamepad: GamepadCommandSource,
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
            gamepad_input = gamepad.poll()
            report_gamepad_connection(gamepad_input)
            if gamepad_input.exit_requested:
                print("Hardware gamepad requested exit during the nominal-pose ramp.")
                drain_command_receipts(pending_receipts)
                stop_hardware_controls(controls, config, executor)
                return
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

        if config.wait_for_start_button:
            started = wait_for_policy_start(
                binding,
                controls,
                groups,
                states,
                previous_target,
                adapter,
                config,
                gamepad,
                executor,
                pending_receipts,
            )
            if not started:
                drain_command_receipts(pending_receipts)
                stop_hardware_controls(controls, config, executor)
                return

        adapter.reset()
        start_time = time.monotonic()
        next_status = start_time
        deadline = start_time
        step = 0
        try:
            while time.monotonic() - start_time < config.max_duration_seconds:
                gamepad_input = gamepad.poll()
                report_gamepad_connection(gamepad_input)
                if gamepad_input.exit_requested:
                    print(
                        "Hardware gamepad requested exit; stopping and disabling groups."
                    )
                    break
                command = hardware_command(config, adapter.metadata, gamepad_input)
                frame = wait_for_hardware_frame(
                    states,
                    adapter.physical_coordinate_names,
                    config.groups,
                    config.imu_name,
                    binding.StateUnavailableError,
                    config.telemetry_timeout_seconds,
                )
                validate_hardware_frame(frame, config, expected_group_state="RUNNING")
                policy_step = adapter.step(frame, command)
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
                            "hardware_step": step,
                            "elapsed_seconds": now - start_time,
                            "tilt_radians": policy_step.tilt_radians,
                            "action_abs_max": float(np.max(np.abs(policy_step.action))),
                            "command": tuple(float(value) for value in command),
                            "gamepad_connected": gamepad_input.connected,
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
            print("Hardware locomotion interrupted; stopping and disabling groups.")
        stop_hardware_controls(controls, config, executor)


def wait_for_policy_start(
    binding: Any,
    controls: Mapping[str, Any],
    groups: Mapping[str, Any],
    states: Any,
    hold_target: np.ndarray,
    adapter: HardwarePolicyAdapter,
    config: HardwareLocomotionConfig,
    gamepad: GamepadCommandSource,
    executor: Executor,
    pending_receipts: list[Any],
) -> bool:
    print(
        "Nominal-pose ramp complete; policy is held. Release the torso support, "
        "let the robot settle, then press A/Cross to start. Press Start/Options "
        "to abort."
    )
    deadline = time.monotonic()
    while True:
        gamepad_input = gamepad.poll()
        report_gamepad_connection(gamepad_input)
        if gamepad_input.exit_requested:
            print("Hardware gamepad requested exit before policy start.")
            return False
        if gamepad_input.connected and gamepad_input.reset_requested:
            print("Hardware policy start confirmed with A/Cross.")
            return True
        send_targets(
            binding,
            controls,
            groups,
            hold_target,
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


def start_hardware_controls(
    binding: Any,
    controls: Mapping[str, Any],
    groups: Mapping[str, Any],
    physical_target: np.ndarray,
    adapter: HardwarePolicyAdapter,
    config: HardwareLocomotionConfig,
    executor: Executor,
) -> None:
    targets = {
        name: coordinate_targets(
            binding,
            groups[name].coordinates,
            physical_target,
            adapter,
            config,
        )
        for name in config.groups
    }
    starts = {
        name: executor.submit(controls[name].start, targets[name])
        for name in config.groups
    }
    for name in config.groups:
        starts[name].result()


def stop_hardware_controls(
    controls: Mapping[str, Any],
    config: HardwareLocomotionConfig,
    executor: Executor,
) -> None:
    stops = {name: executor.submit(controls[name].stop) for name in config.groups}
    for name in config.groups:
        stops[name].result()


def coordinate_targets(
    binding: Any,
    coordinate_names: tuple[str, ...],
    physical_target: np.ndarray,
    adapter: HardwarePolicyAdapter,
    config: HardwareLocomotionConfig,
) -> dict[str, Any]:
    targets = {}
    for name in coordinate_names:
        index = adapter.coordinate_index[name]
        targets[name] = binding.CoordinateTarget.position(
            float(physical_target[index]),
            kp=float(adapter.kp[index] * config.kp_scale),
            kd=float(adapter.kd[index] * config.kd_scale),
        )
    return targets


def send_targets(
    binding: Any,
    controls: Mapping[str, Any],
    groups: Mapping[str, Any],
    physical_target: np.ndarray,
    adapter: HardwarePolicyAdapter,
    config: HardwareLocomotionConfig,
    executor: Executor,
    pending_receipts: list[Any],
) -> None:
    reap_command_receipts(pending_receipts)
    targets = {
        name: coordinate_targets(
            binding,
            groups[name].coordinates,
            physical_target,
            adapter,
            config,
        )
        for name in config.groups
    }
    submissions = {
        name: executor.submit(controls[name].send_async, targets[name])
        for name in config.groups
    }
    for name in config.groups:
        pending_receipts.append(submissions[name].result())
    maximum_pending = len(config.groups) * (
        math.ceil(config.acknowledgement_timeout_seconds / adapter.control_timestep) + 1
    )
    while len(pending_receipts) > maximum_pending:
        pending_receipts.pop(0).result()


def reap_command_receipts(pending_receipts: list[Any]) -> None:
    unresolved = []
    for receipt in pending_receipts:
        if receipt.done():
            receipt.result()
        else:
            unresolved.append(receipt)
    pending_receipts[:] = unresolved


def drain_command_receipts(pending_receipts: list[Any]) -> None:
    while pending_receipts:
        pending_receipts.pop(0).result()


def wait_for_group_state(
    states: Any,
    coordinate_names: tuple[str, ...],
    config: HardwareLocomotionConfig,
    unavailable_error: type[BaseException],
    expected_state: str,
) -> None:
    deadline = time.monotonic() + config.telemetry_timeout_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutError(
                f"timed out waiting for hardware groups to enter {expected_state}"
            )
        frame = wait_for_hardware_frame(
            states,
            coordinate_names,
            config.groups,
            config.imu_name,
            unavailable_error,
            remaining,
        )
        if all(state == expected_state for state in frame.group_states):
            validate_hardware_frame(frame, config, expected_group_state=expected_state)
            return
        if set(frame.group_states) & FAULT_GROUP_STATES:
            raise RuntimeError(
                f"hardware group fault while arming: {frame.group_states}"
            )
        states.wait_for_change(timeout=remaining)


def pace_hardware_loop(
    previous_deadline: float,
    control_timestep: float,
    config: HardwareLocomotionConfig,
) -> float:
    deadline = previous_deadline + control_timestep
    remaining = deadline - time.monotonic()
    if remaining > 0.0:
        time.sleep(remaining)
    elif -remaining > config.max_loop_lag_seconds:
        raise RuntimeError(
            f"hardware policy loop missed its deadline by {-remaining:.3f} seconds"
        )
    return deadline
