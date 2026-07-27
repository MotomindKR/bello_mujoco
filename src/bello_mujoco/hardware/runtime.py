"""Shared fail-closed control runtime for physical Bello policies."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import importlib
import math
import sys
import threading
import time
from typing import Any

import mujoco
import numpy as np

from bello_mujoco.evaluation.render import quat_rotate_inverse_np

HOLD_CONFIRMATION = "HOLD"
RUN_CONFIRMATION = "RUN"
FAULT_GROUP_STATES = frozenset({"FAULT", "SHUTDOWN", "UNSPECIFIED"})
STANDARD_GRAVITY = 9.80665


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


def validate_hardware_control_config(config: Any) -> None:
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
        "max_target_step_radians": config.max_target_step_radians,
        "max_tilt_radians": config.max_tilt_radians,
        "max_angular_velocity": config.max_angular_velocity,
        "max_linear_acceleration": config.max_linear_acceleration,
        "max_loop_lag_seconds": config.max_loop_lag_seconds,
        "status_interval_seconds": config.status_interval_seconds,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if config.max_duration_seconds is not None and (
        not math.isfinite(config.max_duration_seconds)
        or config.max_duration_seconds <= 0.0
    ):
        raise ValueError("max_duration_seconds must be positive or null")
    if config.max_motor_age_ms <= 0 or config.max_imu_age_ms <= 0:
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
    return quat_rotate_inverse_np(
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
            "bello-run-hardware ...`"
        ) from error


def require_async_hardware_binding(binding: Any) -> None:
    control_session = getattr(binding, "ControlSession", None)
    if control_session is None or not hasattr(control_session, "send_async"):
        raise RuntimeError(
            "armed hardware execution requires bello_hardware.ControlSession.send_async; "
            "rebuild the local binding (use `uv run --no-cache --with "
            "~/bello_stack/components/bello-hardware-python ...`)"
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
        read = input
    else:
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
    adapter: Any,
    config: Any,
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
    config: Any,
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


def coordinate_targets(
    binding: Any,
    coordinate_names: tuple[str, ...],
    physical_target: np.ndarray,
    adapter: Any,
) -> dict[str, Any]:
    return {
        name: binding.CoordinateTarget.position(
            float(physical_target[adapter.coordinate_index[name]]),
            kp=float(adapter.kp[adapter.coordinate_index[name]]),
            kd=float(adapter.kd[adapter.coordinate_index[name]]),
        )
        for name in coordinate_names
    }


def start_hardware_controls(
    binding: Any,
    controls: Mapping[str, Any],
    groups: Mapping[str, Any],
    physical_target: np.ndarray,
    adapter: Any,
    config: Any,
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
            ),
        )
        for name in config.groups
    }
    for name in config.groups:
        starts[name].result()


def stop_hardware_controls(
    controls: Mapping[str, Any],
    config: Any,
    executor: Executor,
) -> None:
    stops = {name: executor.submit(controls[name].stop) for name in config.groups}
    for name in config.groups:
        stops[name].result()


def send_targets(
    binding: Any,
    controls: Mapping[str, Any],
    groups: Mapping[str, Any],
    physical_target: np.ndarray,
    adapter: Any,
    config: Any,
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
    states: Any,
    coordinate_names: tuple[str, ...],
    config: Any,
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
    config: Any,
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
    states: Any
    adapter: Any
    config: Any
    executor: Executor
    pending_receipts: list[Any]
    previous_target: np.ndarray
    deadline: float

    def read_frame(self) -> HardwareFrame:
        frame = wait_for_hardware_frame(
            self.states,
            self.adapter.physical_coordinate_names,
            self.config.groups,
            self.config.imu_name,
            self.binding.StateUnavailableError,
            self.config.telemetry_timeout_seconds,
        )
        validate_hardware_frame(frame, self.config, expected_group_state="RUNNING")
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
            requested = initial_positions + fraction * (
                self.adapter.nominal_physical_target - initial_positions
            )
            target = limit_target_step(
                self.previous_target,
                requested,
                self.config.max_target_step_radians,
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
    states: Any,
    initial_frame: HardwareFrame,
    adapter: Any,
    config: Any,
    *,
    thread_name: str,
) -> Iterator[ArmedControlSession]:
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
                thread_name_prefix=thread_name,
            )
        )
        resources.callback(stop_hardware_controls, controls, config, executor)
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
            states,
            adapter.physical_coordinate_names,
            config,
            binding.StateUnavailableError,
            "RUNNING",
        )
        yield ArmedControlSession(
            binding=binding,
            controls=controls,
            groups=groups,
            states=states,
            adapter=adapter,
            config=config,
            executor=executor,
            pending_receipts=[],
            previous_target=np.asarray(
                initial_frame.positions, dtype=np.float64
            ).copy(),
            deadline=time.monotonic(),
        )
