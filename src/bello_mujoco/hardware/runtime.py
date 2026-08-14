"""Shared fail-closed control runtime for physical Bello policies."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import math
from pathlib import Path
import select
import sys
import threading
import time
from typing import Any, Protocol
import warnings

import mujoco
import numpy as np

from bello_mujoco.hardware.config import HardwareControlConfig
from bello_mujoco.spatial import quat_rotate_inverse_wxyz

HOLD_CONFIRMATION = "HOLD"
RUN_CONFIRMATION = "RUN"
MOTION_CONFIRMATION = "MOTION"
IDLE_CONFIRMATION = "IDLE"
FAULT_GROUP_STATES = frozenset({"FAULT", "SHUTDOWN", "UNSPECIFIED"})
STANDARD_GRAVITY = 9.80665
GROUP_STATE_POLL_INTERVAL_SECONDS = 0.01
NOMINAL_POSE_MAX_ERROR_RADIANS = 0.25
ANKLE_NOMINAL_POSE_MAX_ERROR_RADIANS = 1.00
ANKLE_MOTOR_COORDINATES = frozenset(
    {
        "left_ankle_motor_1",
        "left_ankle_motor_2",
        "right_ankle_motor_1",
        "right_ankle_motor_2",
    }
)


@dataclass(frozen=True)
class HardwareFrame:
    positions: np.ndarray
    velocities: np.ndarray
    orientation_xyzw: np.ndarray
    angular_velocity: np.ndarray
    linear_acceleration: np.ndarray
    max_motor_age_ms: int
    imu_age_ms: int
    max_group_age_ms: int
    sample_span_ms: float
    state_frame_round_trip_ms: float
    group_states: tuple[str, ...]
    group_causes: tuple[str, ...]
    group_details: tuple[str, ...]


class HardwarePolicyAdapter(Protocol):
    physical_coordinate_names: tuple[str, ...]
    coordinate_index: dict[str, int]
    kp: np.ndarray
    kd: np.ndarray

    @property
    def control_timestep(self) -> float: ...

    @property
    def nominal_physical_target(self) -> np.ndarray: ...


def validate_hardware_control_config(config: HardwareControlConfig) -> None:
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
        "gain_scale": config.gain_scale,
        "max_tilt_radians": config.max_tilt_radians,
        "max_angular_velocity": config.max_angular_velocity,
        "max_linear_acceleration": config.max_linear_acceleration,
        "max_loop_lag_seconds": config.max_loop_lag_seconds,
        "max_state_sample_span_ms": config.max_state_sample_span_ms,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if config.gain_scale > 1.0:
        raise ValueError("gain_scale must be at most 1.0")
    if config.max_duration_seconds is not None and (
        not math.isfinite(config.max_duration_seconds)
        or config.max_duration_seconds <= 0.0
    ):
        raise ValueError("max_duration_seconds must be positive or null")
    if config.status_interval_seconds is not None and (
        not math.isfinite(config.status_interval_seconds)
        or config.status_interval_seconds <= 0.0
    ):
        raise ValueError("status_interval_seconds must be positive or null")
    if (
        config.max_motor_age_ms <= 0
        or config.max_imu_age_ms <= 0
        or config.max_group_age_ms <= 0
    ):
        raise ValueError("telemetry age limits must be positive")


def actuator_coordinate_names(env: Any) -> tuple[str, ...]:
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


def actuator_impedance_gains(env: Any) -> tuple[np.ndarray, np.ndarray]:
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
    return quat_rotate_inverse_wxyz(
        wxyz.astype(np.float32), np.asarray((0.0, 0.0, -1.0), dtype=np.float32)
    )


def specific_force_from_gravity_removed_acceleration(
    linear_acceleration: np.ndarray, projected_gravity: np.ndarray
) -> np.ndarray:
    acceleration = np.asarray(linear_acceleration, dtype=np.float32)
    gravity = np.asarray(projected_gravity, dtype=np.float32)
    if acceleration.shape != (3,) or gravity.shape != (3,):
        raise ValueError("IMU acceleration and projected gravity must be 3-vectors")
    return acceleration - STANDARD_GRAVITY * gravity


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


def import_hardware_binding() -> Any:
    from bello_mujoco.repository import repository_root

    root = repository_root()
    expected_package = (
        root
        / "deps"
        / "bello-stack"
        / "components"
        / "bello-hardware-python"
        / "src"
        / "bello_hardware"
    ).resolve()
    try:
        binding = importlib.import_module("bello_hardware")
    except ImportError as error:
        raise RuntimeError(
            "bello_hardware is not installed from the pinned bello-stack workspace; "
            "run `git submodule update --init --recursive` and `uv sync --frozen`"
        ) from error
    module_file = getattr(binding, "__file__", None)
    if module_file is None:
        raise RuntimeError("bello_hardware does not expose its installed source path")
    try:
        Path(module_file).resolve().relative_to(expected_package)
    except ValueError as error:
        raise RuntimeError(
            "bello_hardware was not loaded from the pinned bello-stack workspace; "
            "run `uv sync --frozen` in this checkout"
        ) from error
    return binding


def require_async_hardware_binding(binding: Any) -> None:
    control_session = getattr(binding, "ControlSession", None)
    control_state = getattr(binding, "ControlState", None)
    if (
        control_session is None
        or not hasattr(control_session, "send_async")
        or control_state is None
        or not hasattr(control_state, "RUNNING")
    ):
        raise RuntimeError(
            "armed hardware execution requires bello_hardware.ControlSession.send_async "
            "and ControlState.RUNNING; "
            "run `uv sync --frozen` to rebuild the pinned workspace binding"
        )


def require_state_frame_binding(binding: Any) -> None:
    robot = getattr(binding, "Robot", None)
    if robot is None or not hasattr(robot, "states"):
        raise RuntimeError(
            "hardware execution requires the synchronous "
            "bello_hardware.Robot.states API; run `uv sync --frozen` "
            "to rebuild the pinned workspace binding"
        )


def require_operator_confirmation(
    expected: str,
    prompt: str,
    operator_input: Callable[[str], str] | None = None,
) -> None:
    if operator_input is None:
        if not sys.stdin.isatty():
            raise RuntimeError(
                f"operator confirmation requires an interactive terminal; "
                f"expected exact {expected!r}"
            )
        read = input
    else:
        read = operator_input
    try:
        response = read(prompt)
    except (EOFError, OSError) as error:
        raise RuntimeError(
            f"operator confirmation input failed; expected exact {expected!r}"
        ) from error
    if response != expected:
        raise RuntimeError(
            f"operator confirmation rejected; expected exact {expected!r}"
        )
    print(f"Operator confirmed {expected}.")


def require_idle_confirmation(
    prompt: str,
    operator_input: Callable[[str], str] | None = None,
) -> None:
    if operator_input is None:
        if not sys.stdin.isatty():
            raise RuntimeError("IDLE confirmation requires an interactive terminal")
        read = input
    else:
        read = operator_input
    while True:
        try:
            response = read(prompt)
        except (EOFError, OSError) as error:
            raise RuntimeError(
                "operator confirmation input failed while awaiting exact "
                f"{IDLE_CONFIRMATION!r}"
            ) from error
        if response == IDLE_CONFIRMATION:
            print(f"Operator confirmed {IDLE_CONFIRMATION}.")
            return
        print(f"Expected exact {IDLE_CONFIRMATION!r}; hardware controls remain damped.")


def wait_for_operator_confirmation_while_holding(
    expected: str,
    prompt: str,
    hold_step: Callable[[], bool],
    operator_input: Callable[[str], str] | None = None,
) -> bool:
    if operator_input is None:
        if not sys.stdin.isatty():
            raise RuntimeError(
                f"operator confirmation requires an interactive terminal; "
                f"expected exact {expected!r}"
            )
        print(prompt, end="", flush=True)
        while True:
            readable, _, _ = select.select((sys.stdin,), (), (), 0.0)
            if readable:
                response = sys.stdin.readline()
                if response == "":
                    raise RuntimeError(
                        f"operator confirmation input failed; expected exact {expected!r}"
                    )
                response = response.rstrip("\r\n")
                break
            if not hold_step():
                print()
                return False
        if response != expected:
            raise RuntimeError(
                f"operator confirmation rejected; expected exact {expected!r}"
            )
        print(f"Operator confirmed {expected}.")
        return True

    read = operator_input
    response: list[str] = []
    errors: list[BaseException] = []
    complete = threading.Event()

    def read_confirmation() -> None:
        try:
            response.append(read(prompt))
        except BaseException as error:
            errors.append(error)
        finally:
            complete.set()

    threading.Thread(
        target=read_confirmation,
        name="bello-operator-confirmation",
        daemon=True,
    ).start()
    while not complete.is_set():
        if not hold_step():
            return False
    if errors:
        error = errors[0]
        if isinstance(error, KeyboardInterrupt):
            raise error
        raise RuntimeError(
            f"operator confirmation input failed; expected exact {expected!r}"
        ) from error
    if response != [expected]:
        raise RuntimeError(
            f"operator confirmation rejected; expected exact {expected!r}"
        )
    print(f"Operator confirmed {expected}.")
    return True


def validate_hardware_inventory(
    robot: Any,
    adapter: HardwarePolicyAdapter,
    config: HardwareControlConfig,
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
    state_reader: Any,
    coordinate_names: tuple[str, ...],
    group_names: tuple[str, ...],
    imu_name: str,
    timeout_seconds: float,
) -> HardwareFrame:
    frame = state_reader.read(timeout=timeout_seconds)
    motors_by_coordinate = {motor.coordinate: motor for motor in frame.motors}
    groups_by_name = {group.group: group for group in frame.groups}
    imus_by_name = {imu.imu: imu for imu in frame.imus}
    try:
        motors = tuple(motors_by_coordinate[name] for name in coordinate_names)
        groups = tuple(groups_by_name[name] for name in group_names)
        imu = imus_by_name[imu_name]
    except KeyError as error:
        raise RuntimeError(
            f"hardware state frame is missing {error.args[0]!r}"
        ) from error
    received_age_ns = max(0, time.monotonic_ns() - frame.received_monotonic_ns)
    transport_age_ns = frame.round_trip_ns + received_age_ns
    max_motor_age_ms = (
        max(motor.sample_age_ns for motor in motors) + transport_age_ns
    ) // 1_000_000
    imu_age_ms = (imu.sample_age_ns + transport_age_ns) // 1_000_000
    max_group_age_ms = (
        max(group.sample_age_ns for group in groups) + transport_age_ns
    ) // 1_000_000
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
        max_motor_age_ms=max_motor_age_ms,
        imu_age_ms=imu_age_ms,
        group_states=tuple(group.state.name for group in groups),
        group_causes=tuple(group.cause.name for group in groups),
        group_details=tuple(group.detail for group in groups),
        max_group_age_ms=max_group_age_ms,
        sample_span_ms=frame.sample_span_ns / 1_000_000.0,
        state_frame_round_trip_ms=frame.round_trip_ns / 1_000_000.0,
    )


def validate_hardware_frame(
    frame: HardwareFrame,
    config: HardwareControlConfig,
    *,
    expected_group_state: str | None = None,
    max_tilt_radians: float | None = None,
) -> None:
    group_count = len(config.groups)
    if (
        len(frame.group_states) != group_count
        or len(frame.group_causes) != group_count
        or len(frame.group_details) != group_count
    ):
        raise RuntimeError(
            "hardware telemetry does not contain one state, cause, and detail "
            "for every configured group"
        )
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
    if frame.max_group_age_ms > config.max_group_age_ms:
        raise RuntimeError(f"group telemetry is stale: {frame.max_group_age_ms} ms")
    if frame.sample_span_ms > config.max_state_sample_span_ms:
        raise RuntimeError(
            f"hardware state sample span {frame.sample_span_ms:.3f} ms exceeds "
            f"{config.max_state_sample_span_ms:.3f} ms"
        )
    bad_states = sorted(set(frame.group_states) & FAULT_GROUP_STATES)
    if bad_states:
        raise RuntimeError(
            "hardware group entered unsafe state: "
            f"{format_group_statuses(frame, config.groups)}"
        )
    if expected_group_state is not None and any(
        state != expected_group_state for state in frame.group_states
    ):
        raise RuntimeError(
            f"hardware groups must be {expected_group_state}, got "
            f"{format_group_statuses(frame, config.groups)}"
        )
    tilt_limit = (
        config.max_tilt_radians if max_tilt_radians is None else max_tilt_radians
    )
    tilt = tilt_from_xyzw(frame.orientation_xyzw)
    if tilt > tilt_limit:
        raise RuntimeError(f"robot tilt {tilt:.3f} rad exceeds {tilt_limit:.3f} rad")
    angular_speed = float(np.linalg.norm(frame.angular_velocity))
    if angular_speed > config.max_angular_velocity:
        raise RuntimeError(
            f"IMU angular velocity {angular_speed:.3f} exceeds safety limit"
        )
    acceleration = float(np.linalg.norm(frame.linear_acceleration))
    if acceleration > config.max_linear_acceleration:
        raise RuntimeError(f"IMU acceleration {acceleration:.3f} exceeds safety limit")


def format_group_statuses(frame: HardwareFrame, group_names: tuple[str, ...]) -> str:
    statuses = []
    for name, state, cause, detail in zip(
        group_names,
        frame.group_states,
        frame.group_causes,
        frame.group_details,
        strict=True,
    ):
        fields = [f"{name}={state}"]
        if cause != "UNSPECIFIED":
            fields.append(f"cause={cause}")
        if detail:
            fields.append(f"detail={detail!r}")
        statuses.append(" ".join(fields))
    return ", ".join(statuses)


def require_nominal_pose(
    frame: HardwareFrame,
    adapter: HardwarePolicyAdapter,
) -> None:
    """Require measured coordinates to remain near the checkpoint nominal pose."""

    measured = np.asarray(frame.positions, dtype=np.float64)
    nominal = np.asarray(adapter.nominal_physical_target, dtype=np.float64)
    coordinate_count = len(adapter.physical_coordinate_names)
    if coordinate_count == 0:
        raise RuntimeError("nominal-pose validation requires physical coordinates")
    expected_shape = (coordinate_count,)
    if measured.shape != expected_shape or nominal.shape != expected_shape:
        raise RuntimeError(
            "nominal-pose validation requires one measured and target value per "
            "physical coordinate"
        )
    if not np.all(np.isfinite(measured)):
        raise RuntimeError("measured nominal-pose coordinates are non-finite")
    if not np.all(np.isfinite(nominal)):
        raise RuntimeError("checkpoint nominal-pose target is non-finite")
    limits = np.asarray(
        [
            (
                ANKLE_NOMINAL_POSE_MAX_ERROR_RADIANS
                if name in ANKLE_MOTOR_COORDINATES
                else NOMINAL_POSE_MAX_ERROR_RADIANS
            )
            for name in adapter.physical_coordinate_names
        ],
        dtype=np.float64,
    )
    error = np.abs(measured - nominal)
    excess = error - limits
    index = int(np.argmax(excess))
    if excess[index] > 0.0:
        name = adapter.physical_coordinate_names[index]
        raise RuntimeError(
            "robot is too far from the checkpoint nominal pose: "
            f"{name} is {error[index]:.3f} rad away "
            f"(measured={measured[index]:.3f}, nominal={nominal[index]:.3f}); "
            f"maximum allowed error is {limits[index]:.3f} rad"
        )


def coordinate_targets(
    binding: Any,
    coordinate_names: tuple[str, ...],
    physical_target: np.ndarray,
    adapter: HardwarePolicyAdapter,
    *,
    gain_scale: float = 1.0,
) -> dict[str, Any]:
    return {
        name: binding.CoordinateTarget.position(
            float(physical_target[adapter.coordinate_index[name]]),
            kp=float(adapter.kp[adapter.coordinate_index[name]] * gain_scale),
            kd=float(adapter.kd[adapter.coordinate_index[name]] * gain_scale),
        )
        for name in coordinate_names
    }


def start_hardware_controls(
    binding: Any,
    controls: Mapping[str, Any],
    groups: Mapping[str, Any],
    physical_target: np.ndarray,
    adapter: HardwarePolicyAdapter,
    config: HardwareControlConfig,
    executor: Executor,
) -> None:
    starts = {
        name: executor.submit(
            controls[name].start,
            coordinate_targets(
                binding,
                groups[name].coordinates,
                physical_target,
                adapter,
                gain_scale=config.gain_scale,
            ),
        )
        for name in config.groups
    }
    for name in config.groups:
        starts[name].result()


@dataclass(frozen=True)
class _StopResult:
    damped: bool
    errors: tuple[BaseException, ...]


def _stop_hardware_controls(
    binding: Any,
    controls: Mapping[str, Any],
    config: HardwareControlConfig,
    executor: Executor,
) -> _StopResult:
    damped = getattr(binding.ControlState, "DAMPED", None)

    def stop_if_running(control: Any) -> None:
        if control.state == binding.ControlState.RUNNING:
            control.stop()

    stops = {
        name: executor.submit(stop_if_running, controls[name]) for name in config.groups
    }
    errors = []
    for name in config.groups:
        try:
            stops[name].result()
        except BaseException as error:
            errors.append(error)
    return _StopResult(
        damped=damped is not None
        and any(controls[name].state == damped for name in config.groups),
        errors=tuple(errors),
    )


def stop_hardware_controls(
    binding: Any,
    controls: Mapping[str, Any],
    config: HardwareControlConfig,
    executor: Executor,
) -> bool:
    result = _stop_hardware_controls(binding, controls, config, executor)
    if result.errors:
        raise ExceptionGroup("failed to stop hardware controls", result.errors)
    return result.damped


def _disable_damped_hardware_controls(
    binding: Any,
    controls: Mapping[str, Any],
    config: HardwareControlConfig,
    executor: Executor,
) -> tuple[BaseException, ...]:
    damped = getattr(binding.ControlState, "DAMPED", None)
    disables = {
        name: executor.submit(controls[name].disable)
        for name in config.groups
        if damped is not None and controls[name].state == damped
    }
    errors = []
    for name in config.groups:
        if name not in disables:
            continue
        try:
            disables[name].result()
        except BaseException as error:
            errors.append(error)
    return tuple(errors)


def stop_and_confirm_idle(
    binding: Any,
    controls: Mapping[str, Any],
    config: HardwareControlConfig,
    executor: Executor,
    operator_input: Callable[[str], str] | None,
) -> None:
    result = _stop_hardware_controls(binding, controls, config, executor)
    errors = list(result.errors)
    if result.damped:
        require_idle_confirmation(
            "Hardware controls are damped. Support the robot, then type IDLE to "
            "disable motors and release control: ",
            operator_input,
        )
        errors.extend(
            _disable_damped_hardware_controls(
                binding,
                controls,
                config,
                executor,
            )
        )
    if errors:
        raise ExceptionGroup("hardware control cleanup failed", errors)


def _close_preserving_damping(control: Any) -> None:
    try:
        control.close(preserve_damping=True)
    except BaseException as error:
        warnings.warn(
            f"failed to release hardware control while preserving damping: {error}",
            RuntimeWarning,
            stacklevel=2,
        )


def send_targets(
    binding: Any,
    controls: Mapping[str, Any],
    groups: Mapping[str, Any],
    physical_target: np.ndarray,
    adapter: HardwarePolicyAdapter,
    config: HardwareControlConfig,
    executor: Executor,
    pending_receipts: list[Any],
) -> None:
    reap_command_receipts(pending_receipts)
    submissions = {
        name: executor.submit(
            controls[name].send_async,
            coordinate_targets(
                binding,
                groups[name].coordinates,
                physical_target,
                adapter,
                gain_scale=config.gain_scale,
            ),
        )
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
    state_reader: Any,
    coordinate_names: tuple[str, ...],
    config: HardwareControlConfig,
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
            state_reader,
            coordinate_names,
            config.groups,
            config.imu_name,
            remaining,
        )
        if all(state == expected_state for state in frame.group_states):
            validate_hardware_frame(frame, config, expected_group_state=expected_state)
            return
        if set(frame.group_states) & FAULT_GROUP_STATES:
            raise RuntimeError(
                "hardware group fault while arming: "
                f"{format_group_statuses(frame, config.groups)}"
            )
        time.sleep(
            min(
                GROUP_STATE_POLL_INTERVAL_SECONDS,
                max(0.0, deadline - time.monotonic()),
            )
        )


def pace_hardware_loop(
    previous_deadline: float,
    control_timestep: float,
    config: HardwareControlConfig,
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


@dataclass
class ArmedControlSession:
    binding: Any
    controls: Mapping[str, Any]
    groups: Mapping[str, Any]
    state_reader: Any
    adapter: HardwarePolicyAdapter
    config: HardwareControlConfig
    executor: Executor
    pending_receipts: list[Any]
    previous_target: np.ndarray
    deadline: float

    def read_frame(
        self,
        *,
        max_tilt_radians: float | None = None,
    ) -> HardwareFrame:
        frame = wait_for_hardware_frame(
            self.state_reader,
            self.adapter.physical_coordinate_names,
            self.config.groups,
            self.config.imu_name,
            self.config.telemetry_timeout_seconds,
        )
        validate_hardware_frame(
            frame,
            self.config,
            expected_group_state="RUNNING",
            max_tilt_radians=max_tilt_radians,
        )
        return frame

    def send_target(self, target: np.ndarray) -> None:
        send_targets(
            self.binding,
            self.controls,
            self.groups,
            target,
            self.adapter,
            self.config,
            self.executor,
            self.pending_receipts,
        )
        self.previous_target = np.asarray(target, dtype=np.float64)

    def pace(self) -> None:
        self.deadline = pace_hardware_loop(
            self.deadline,
            self.adapter.control_timestep,
            self.config,
        )

    def reset_deadline(self) -> None:
        self.deadline = time.monotonic()

    def ramp_to_nominal(
        self,
        initial_positions: np.ndarray,
        *,
        before_step: Callable[[], bool] | None = None,
    ) -> HardwareFrame | None:
        ramp_steps = max(
            1, math.ceil(self.config.ramp_seconds / self.adapter.control_timestep)
        )
        frame = None
        for step in range(1, ramp_steps + 1):
            if before_step is not None and not before_step():
                return None
            fraction = step / ramp_steps
            target = initial_positions + fraction * (
                self.adapter.nominal_physical_target - initial_positions
            )
            self.send_target(target)
            self.pace()
            frame = self.read_frame()
        return frame

    def wait_for_run(
        self,
        prompt: str,
        *,
        operator_input: Callable[[str], str] | None = None,
        before_step: Callable[[], bool] | None = None,
        validate_hold: Callable[[HardwareFrame], None] | None = None,
    ) -> bool:
        def hold_step() -> bool:
            if before_step is not None and not before_step():
                return False
            self.send_target(self.previous_target)
            self.pace()
            frame = self.read_frame()
            if validate_hold is not None:
                validate_hold(frame)
            return True

        return wait_for_operator_confirmation_while_holding(
            RUN_CONFIRMATION,
            prompt,
            hold_step,
            operator_input,
        )

    def drain(self) -> None:
        drain_command_receipts(self.pending_receipts)


@contextmanager
def armed_control_session(
    binding: Any,
    groups: Mapping[str, Any],
    state_reader: Any,
    initial_frame: HardwareFrame,
    adapter: HardwarePolicyAdapter,
    config: HardwareControlConfig,
    *,
    thread_name: str,
    operator_input: Callable[[str], str] | None = None,
) -> Iterator[ArmedControlSession]:
    controls: dict[str, Any] = {}
    with ThreadPoolExecutor(
        max_workers=len(config.groups),
        thread_name_prefix=thread_name,
    ) as executor:
        try:
            for name, group in groups.items():
                controls[name] = group.control(
                    config.client_name,
                    command_validity=config.command_validity_seconds,
                    acknowledgement_timeout=config.acknowledgement_timeout_seconds,
                    rpc_timeout=config.rpc_timeout_seconds,
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
            wait_for_group_state(
                state_reader,
                adapter.physical_coordinate_names,
                config,
                "RUNNING",
            )
            try:
                yield ArmedControlSession(
                    binding=binding,
                    controls=controls,
                    groups=groups,
                    state_reader=state_reader,
                    adapter=adapter,
                    config=config,
                    executor=executor,
                    pending_receipts=[],
                    previous_target=np.asarray(
                        initial_frame.positions, dtype=np.float64
                    ).copy(),
                    deadline=time.monotonic(),
                )
            except BaseException:
                try:
                    stop_and_confirm_idle(
                        binding,
                        controls,
                        config,
                        executor,
                        operator_input,
                    )
                except BaseException as cleanup:
                    warnings.warn(
                        f"control cleanup failed: {cleanup}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                raise
            else:
                stop_and_confirm_idle(
                    binding,
                    controls,
                    config,
                    executor,
                    operator_input,
                )
        finally:
            for control in controls.values():
                _close_preserving_damping(control)
