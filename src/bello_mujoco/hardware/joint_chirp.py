"""Fail-closed single-joint chirp diagnostic for Bello hardware or its emulator."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import math
from pathlib import Path
import sys
import time
from typing import IO, Any, Sequence, get_args

import numpy as np

from bello_mujoco.actuators import default_joint_configs
from bello_mujoco.assets_pipeline.mjcf import AnkleDifferentialConfig
from bello_mujoco.hardware.config import HardwareControlConfig
from bello_mujoco.hardware.runtime import (
    HOLD_CONFIRMATION,
    HardwareFrame,
    armed_control_session,
    decode_hardware_frame,
    import_hardware_binding,
    require_async_hardware_binding,
    require_operator_confirmation,
    require_state_frame_binding,
    validate_hardware_control_config,
    validate_hardware_frame,
    validate_hardware_inventory,
    wait_for_hardware_frame,
    wait_for_operator_confirmation_while_holding,
)
from bello_mujoco.robot import AssetName, RobotDefinition, robot_definition

CHIRP_CONFIRMATION = "CHIRP"
MAX_AMPLITUDE_RADIANS = 0.05
MAX_END_FREQUENCY_RATE_FRACTION = 0.4


@dataclass(frozen=True)
class JointChirpConfig:
    joint: str
    asset: AssetName = "full_body"
    rate_hz: float = 50.0
    amplitude_radians: float = 0.02
    start_frequency_hz: float = 0.5
    end_frequency_hz: float = 10.0
    duration_seconds: float = 30.0
    settle_before_seconds: float = 3.0
    settle_after_seconds: float = 2.0
    envelope_seconds: float = 1.0

    def __post_init__(self) -> None:
        robot_definition(self.asset)
        if not self.joint.strip():
            raise ValueError("joint must not be empty")
        positive = {
            "rate_hz": self.rate_hz,
            "amplitude_radians": self.amplitude_radians,
            "start_frequency_hz": self.start_frequency_hz,
            "end_frequency_hz": self.end_frequency_hz,
            "duration_seconds": self.duration_seconds,
            "settle_after_seconds": self.settle_after_seconds,
            "envelope_seconds": self.envelope_seconds,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if (
            not math.isfinite(self.settle_before_seconds)
            or self.settle_before_seconds < 0.0
        ):
            raise ValueError("settle_before_seconds must be nonnegative and finite")
        if self.amplitude_radians > MAX_AMPLITUDE_RADIANS:
            raise ValueError(
                f"amplitude_radians must not exceed {MAX_AMPLITUDE_RADIANS} rad"
            )
        if self.end_frequency_hz < self.start_frequency_hz:
            raise ValueError("end_frequency_hz must be at least start_frequency_hz")
        if self.end_frequency_hz >= self.rate_hz * MAX_END_FREQUENCY_RATE_FRACTION:
            raise ValueError(
                "end_frequency_hz must remain below "
                f"{MAX_END_FREQUENCY_RATE_FRACTION:g} times rate_hz"
            )
        if 2.0 * self.envelope_seconds > self.duration_seconds:
            raise ValueError("envelope_seconds must not exceed half the chirp duration")

    @property
    def timestep(self) -> float:
        return 1.0 / self.rate_hz

    @property
    def total_duration_seconds(self) -> float:
        return (
            self.settle_before_seconds
            + self.duration_seconds
            + self.settle_after_seconds
        )


@dataclass(frozen=True)
class ChirpPoint:
    phase: str
    offset_radians: float
    instantaneous_frequency_hz: float | None


@dataclass(frozen=True)
class ChirpHardwareAdapter:
    asset: AssetName
    physical_coordinate_names: tuple[str, ...]
    logical_joint_names: tuple[str, ...]
    transform: np.ndarray
    inverse_transform: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    rate_hz: float

    @property
    def coordinate_index(self) -> dict[str, int]:
        return {
            name: index for index, name in enumerate(self.physical_coordinate_names)
        }

    @property
    def control_timestep(self) -> float:
        return 1.0 / self.rate_hz

    @property
    def nominal_physical_target(self) -> np.ndarray:
        return np.zeros(len(self.physical_coordinate_names), dtype=np.float64)


def actuator_layout(
    physical_coordinate_names: tuple[str, ...],
    robot: RobotDefinition,
) -> tuple[tuple[str, ...], np.ndarray]:
    if not physical_coordinate_names:
        raise ValueError("hardware groups do not contain any coordinates")
    if len(set(physical_coordinate_names)) != len(physical_coordinate_names):
        raise ValueError("hardware groups contain duplicate coordinates")

    physical_coordinates = set(physical_coordinate_names)
    for group in robot.coupled_actuator_groups:
        tendon_names = {actuator.tendon_name for actuator in group.actuators}
        logical_names = set(group.joint_names)
        selected_logical_names = physical_coordinates & logical_names
        if selected_logical_names:
            raise ValueError(
                f"hardware inventory exposes logical joints for coupled actuator "
                f"group {group.name!r}: {', '.join(sorted(selected_logical_names))}"
            )
        selected_tendon_names = physical_coordinates & tendon_names
        if selected_tendon_names and selected_tendon_names != tendon_names:
            missing = tendon_names - selected_tendon_names
            raise ValueError(
                f"hardware inventory contains incomplete coupled actuator group "
                f"{group.name!r}; missing physical coordinates: "
                f"{', '.join(sorted(missing))}"
            )

    coupled_by_coordinate = {
        actuator.tendon_name: actuator for actuator in robot.coupled_actuators
    }
    logical_joint_names = tuple(
        coupled_by_coordinate[name].logical_joint_name
        if name in coupled_by_coordinate
        else name
        for name in physical_coordinate_names
    )
    if len(set(logical_joint_names)) != len(logical_joint_names):
        raise ValueError("hardware coordinates do not have a unique logical layout")
    logical_index = {name: index for index, name in enumerate(logical_joint_names)}
    transform = np.eye(len(physical_coordinate_names), dtype=np.float64)
    for row, coordinate in enumerate(physical_coordinate_names):
        coupled = coupled_by_coordinate.get(coordinate)
        if coupled is None:
            continue
        missing = [
            name for name, _ in coupled.joint_coefficients if name not in logical_index
        ]
        if missing:
            raise ValueError(
                f"hardware coordinate {coordinate!r} requires coupled logical "
                f"joints: {', '.join(missing)}"
            )
        transform[row] = 0.0
        for joint_name, coefficient in coupled.joint_coefficients:
            transform[row, logical_index[joint_name]] = coefficient
    if not np.all(np.isfinite(transform)) or np.linalg.matrix_rank(transform) != len(
        physical_coordinate_names
    ):
        raise ValueError("hardware actuator transform is not safely invertible")
    return logical_joint_names, transform


def build_hardware_adapter(
    physical_coordinate_names: tuple[str, ...],
    rate_hz: float,
    robot: RobotDefinition,
) -> ChirpHardwareAdapter:
    logical_joint_names, transform = actuator_layout(physical_coordinate_names, robot)
    joint_configs = default_joint_configs(robot.asset)
    unknown = sorted(set(logical_joint_names) - set(joint_configs))
    if unknown:
        raise ValueError(f"hardware inventory contains unknown joints: {unknown}")

    ankle_gain_scale = AnkleDifferentialConfig().motor_gain_scale
    coupled_coordinate_names = {
        actuator.tendon_name for actuator in robot.coupled_actuators
    }
    gains = []
    for physical_name, logical_name in zip(
        physical_coordinate_names, logical_joint_names, strict=True
    ):
        scale = ankle_gain_scale if physical_name in coupled_coordinate_names else 1.0
        gains.append(
            tuple(scale * value for value in joint_configs[logical_name].gains)
        )
    kp = np.asarray([gain[0] for gain in gains], dtype=np.float64)
    kd = np.asarray([gain[1] for gain in gains], dtype=np.float64)
    return ChirpHardwareAdapter(
        asset=robot.asset,
        physical_coordinate_names=physical_coordinate_names,
        logical_joint_names=logical_joint_names,
        transform=transform,
        inverse_transform=np.linalg.inv(transform),
        kp=kp,
        kd=kd,
        rate_hz=rate_hz,
    )


def chirp_point(elapsed_seconds: float, config: JointChirpConfig) -> ChirpPoint:
    if elapsed_seconds < config.settle_before_seconds:
        return ChirpPoint("settle_before", 0.0, None)
    chirp_elapsed = elapsed_seconds - config.settle_before_seconds
    if chirp_elapsed >= config.duration_seconds:
        return ChirpPoint("settle_after", 0.0, None)

    ratio = config.end_frequency_hz / config.start_frequency_hz
    if math.isclose(ratio, 1.0):
        frequency = config.start_frequency_hz
        angle = 2.0 * math.pi * frequency * chirp_elapsed
    else:
        logarithm = math.log(ratio)
        exponent = ratio ** (chirp_elapsed / config.duration_seconds)
        frequency = config.start_frequency_hz * exponent
        angle = (
            2.0
            * math.pi
            * config.start_frequency_hz
            * config.duration_seconds
            * (exponent - 1.0)
            / logarithm
        )

    ramp_in = min(1.0, chirp_elapsed / config.envelope_seconds)
    ramp_out = min(
        1.0,
        (config.duration_seconds - chirp_elapsed) / config.envelope_seconds,
    )
    ramp = min(ramp_in, ramp_out)
    envelope = math.sin(0.5 * math.pi * ramp) ** 2
    offset = config.amplitude_radians * envelope * math.sin(angle)
    return ChirpPoint("chirp", offset, frequency)


def validate_experiment(
    robot: Any,
    adapter: ChirpHardwareAdapter,
    initial_physical: np.ndarray,
    chirp: JointChirpConfig,
    gain_scale: float,
) -> np.ndarray:
    if chirp.joint not in adapter.logical_joint_names:
        raise ValueError(
            f"joint {chirp.joint!r} is not controlled by the selected groups; "
            f"available joints={list(adapter.logical_joint_names)}"
        )
    initial_physical = np.asarray(initial_physical, dtype=np.float64)
    expected_shape = (len(adapter.physical_coordinate_names),)
    if initial_physical.shape != expected_shape or not np.all(
        np.isfinite(initial_physical)
    ):
        raise ValueError("initial hardware position vector is invalid")
    initial_logical = adapter.inverse_transform @ initial_physical
    selected_index = adapter.logical_joint_names.index(chirp.joint)
    selected_initial = float(initial_logical[selected_index])
    selected_config = default_joint_configs(adapter.asset)[chirp.joint]
    if selected_config.soft_limit is not None:
        low, high = selected_config.soft_limit
        if (
            selected_initial - chirp.amplitude_radians < low
            or selected_initial + chirp.amplitude_radians > high
        ):
            raise ValueError(
                f"joint chirp would exceed {chirp.joint} soft limit "
                f"{selected_config.soft_limit}: initial={selected_initial:.6f}, "
                f"amplitude={chirp.amplitude_radians:.6f}"
            )

    selected_delta = adapter.transform[:, selected_index] * chirp.amplitude_radians
    physical_low = initial_physical - np.abs(selected_delta)
    physical_high = initial_physical + np.abs(selected_delta)
    peak_target_velocity = (
        np.abs(adapter.transform[:, selected_index])
        * chirp.amplitude_radians
        * (
            2.0 * math.pi * chirp.end_frequency_hz
            + math.pi / (2.0 * chirp.envelope_seconds)
        )
    )
    motors_by_coordinate = {motor.coordinate: motor for motor in robot.motors}
    for index, coordinate in enumerate(adapter.physical_coordinate_names):
        motor = motors_by_coordinate[coordinate]
        limits = motor.limits
        if (
            physical_low[index] < limits.position_min
            or physical_high[index] > limits.position_max
        ):
            raise ValueError(
                f"chirp target for {coordinate!r} exceeds hardware position limits"
            )
        velocity_limit = min(abs(limits.velocity_min), abs(limits.velocity_max))
        if peak_target_velocity[index] > velocity_limit:
            raise ValueError(
                f"chirp target for {coordinate!r} can slew at "
                f"{peak_target_velocity[index]:.3f} rad/s, above the hardware "
                f"limit {velocity_limit:.3f} rad/s"
            )
        kp = float(adapter.kp[index] * gain_scale)
        kd = float(adapter.kd[index] * gain_scale)
        if not limits.kp_min <= kp <= limits.kp_max:
            raise ValueError(f"kp={kp:g} for {coordinate!r} is outside hardware limits")
        if not limits.kd_min <= kd <= limits.kd_max:
            raise ValueError(f"kd={kd:g} for {coordinate!r} is outside hardware limits")
    return initial_logical


def _enum_name(value: Enum) -> str:
    return value.name.lower()


def _device_status(status: Any | None) -> dict[str, Any] | None:
    if status is None:
        return None
    return {
        "state": _enum_name(status.state),
        "detail": status.detail,
        "transition_count": status.transition_count,
    }


def inventory_record(robot: Any, endpoint: str) -> dict[str, Any]:
    return {
        "record": "inventory",
        "schema_version": 1,
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "groups": [
            {
                "name": name,
                "control_rate_hz": robot.group(name).control_rate_hz,
                "initial_state": _enum_name(robot.group(name).initial_state),
                "coordinates": list(robot.group(name).coordinates),
            }
            for name in robot.group_names
        ],
        "motors": [
            {
                "name": motor.name,
                "coordinate": motor.coordinate,
                "group": motor.group,
                "bus": motor.bus,
                "id": motor.id,
                "model": motor.model,
                "direction": motor.direction,
                "status": _device_status(motor.status),
                "limits": {
                    "position": [
                        motor.limits.position_min,
                        motor.limits.position_max,
                    ],
                    "velocity": [
                        motor.limits.velocity_min,
                        motor.limits.velocity_max,
                    ],
                    "torque": [motor.limits.torque_min, motor.limits.torque_max],
                    "kp": [motor.limits.kp_min, motor.limits.kp_max],
                    "kd": [motor.limits.kd_min, motor.limits.kd_max],
                },
            }
            for motor in robot.motors
        ],
    }


def raw_state_record(frame: Any) -> dict[str, Any]:
    return {
        "captured_monotonic_ns": frame.captured_monotonic_ns,
        "received_monotonic_ns": frame.received_monotonic_ns,
        "round_trip_ns": frame.round_trip_ns,
        "oldest_sample_age_ns": frame.oldest_sample_age_ns,
        "sample_span_ns": frame.sample_span_ns,
        "motors": [
            {
                "group": motor.group,
                "motor": motor.motor,
                "coordinate": motor.coordinate,
                "position": motor.position,
                "velocity": motor.velocity,
                "torque": motor.torque,
                "temperature": motor.temperature,
                "fault_bits": motor.fault_bits,
                "mode_status": motor.mode_status,
                "feedback_age_ms": motor.feedback_age_ms,
                "sample_monotonic_ns": motor.sample_monotonic_ns,
                "sample_age_ns": motor.sample_age_ns,
            }
            for motor in frame.motors
        ],
        "groups": [
            {
                "group": group.group,
                "state": _enum_name(group.state),
                "detail": group.detail,
                "cause": _enum_name(group.cause),
                "command_count": group.command_count,
                "control_tick": group.control_tick,
                "jitter_us": group.jitter_us,
                "sample_monotonic_ns": group.sample_monotonic_ns,
                "sample_age_ns": group.sample_age_ns,
            }
            for group in frame.groups
        ],
        "imus": [
            {
                "imu": imu.imu,
                "orientation_xyzw": [
                    imu.orientation.x,
                    imu.orientation.y,
                    imu.orientation.z,
                    imu.orientation.w,
                ],
                "angular_velocity": [
                    imu.angular_velocity.x,
                    imu.angular_velocity.y,
                    imu.angular_velocity.z,
                ],
                "linear_acceleration": [
                    imu.linear_acceleration.x,
                    imu.linear_acceleration.y,
                    imu.linear_acceleration.z,
                ],
                "device_timestamp_ms": imu.device_timestamp_ms,
                "sample_age_ms": imu.sample_age_ms,
                "sample_monotonic_ns": imu.sample_monotonic_ns,
                "sample_age_ns": imu.sample_age_ns,
            }
            for imu in frame.imus
        ],
    }


def write_record(output: IO[str], record: dict[str, Any]) -> None:
    json.dump(record, output, separators=(",", ":"), allow_nan=False)
    output.write("\n")


def _read_running_frame(session: Any) -> tuple[Any, HardwareFrame, int]:
    raw = session.state_reader.read(timeout=session.config.telemetry_timeout_seconds)
    received_ns = time.monotonic_ns()
    frame = decode_hardware_frame(
        raw,
        session.adapter.physical_coordinate_names,
        session.config.groups,
        session.config.imu_name,
    )
    validate_hardware_frame(frame, session.config, expected_group_state="RUNNING")
    return raw, frame, received_ns


def _send_capture_and_log(
    session: Any,
    output: IO[str],
    adapter: ChirpHardwareAdapter,
    target_logical: np.ndarray,
    point: ChirpPoint,
    cycle: int,
    elapsed_seconds: float,
) -> None:
    target_physical = adapter.transform @ target_logical
    scheduled_deadline_ns = int(
        (session.deadline + adapter.control_timestep) * 1_000_000_000
    )
    command_started_ns = time.monotonic_ns()
    session.send_target(target_physical)
    command_scheduled_ns = time.monotonic_ns()
    session.pace()
    raw, frame, state_returned_ns = _read_running_frame(session)
    measured_logical = adapter.inverse_transform @ frame.positions
    measured_logical_velocity = adapter.inverse_transform @ frame.velocities
    write_record(
        output,
        {
            "record": "chirp_sample",
            "cycle": cycle,
            "phase": point.phase,
            "elapsed_seconds": elapsed_seconds,
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "logged_unix_ns": time.time_ns(),
            "scheduled_deadline_monotonic_ns": scheduled_deadline_ns,
            "command_started_monotonic_ns": command_started_ns,
            "command_submission_scheduled_monotonic_ns": command_scheduled_ns,
            "state_returned_monotonic_ns": state_returned_ns,
            "command_scheduling_call_ns": command_scheduled_ns - command_started_ns,
            "state_return_lag_after_deadline_ns": max(
                0, state_returned_ns - scheduled_deadline_ns
            ),
            "instantaneous_frequency_hz": point.instantaneous_frequency_hz,
            "offset_radians": point.offset_radians,
            "target_logical": target_logical.tolist(),
            "target_physical": target_physical.tolist(),
            "measured_logical": measured_logical.tolist(),
            "measured_logical_velocity": measured_logical_velocity.tolist(),
            "validated_age_ms": {
                "motor_max": frame.max_motor_age_ms,
                "imu": frame.imu_age_ms,
                "group_max": frame.max_group_age_ms,
            },
            "state_frame": raw_state_record(raw),
        },
    )


def default_output_path(joint: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return Path("outputs") / "hardware_identification" / f"{joint}_{stamp}.jsonl"


def _config_record(
    hardware: HardwareControlConfig,
    chirp: JointChirpConfig,
    adapter: ChirpHardwareAdapter,
    initial_physical: np.ndarray,
    initial_logical: np.ndarray,
) -> dict[str, Any]:
    return {
        "record": "chirp_configuration",
        "schema_version": 1,
        "asset": adapter.asset,
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "sample_relation": "each state frame is read after submission of the target in the same record is scheduled and its control deadline passes; group command_count reports daemon application",
        "hardware": {
            "endpoint": hardware.endpoint,
            "groups": list(hardware.groups),
            "imu_name": hardware.imu_name,
            "client_name": hardware.client_name,
            "gain_scale": hardware.gain_scale,
            "command_validity_seconds": hardware.command_validity_seconds,
            "acknowledgement_timeout_seconds": hardware.acknowledgement_timeout_seconds,
        },
        "chirp": {
            "joint": chirp.joint,
            "rate_hz": chirp.rate_hz,
            "amplitude_radians": chirp.amplitude_radians,
            "start_frequency_hz": chirp.start_frequency_hz,
            "end_frequency_hz": chirp.end_frequency_hz,
            "duration_seconds": chirp.duration_seconds,
            "settle_before_seconds": chirp.settle_before_seconds,
            "settle_after_seconds": chirp.settle_after_seconds,
            "envelope_seconds": chirp.envelope_seconds,
        },
        "physical_coordinate_names": list(adapter.physical_coordinate_names),
        "logical_joint_names": list(adapter.logical_joint_names),
        "physical_from_logical_transform": adapter.transform.tolist(),
        "kp": adapter.kp.tolist(),
        "kd": adapter.kd.tolist(),
        "initial_physical": initial_physical.tolist(),
        "initial_logical": initial_logical.tolist(),
    }


def execute_chirp(
    binding: Any,
    groups: dict[str, Any],
    state_reader: Any,
    initial_frame: HardwareFrame,
    adapter: ChirpHardwareAdapter,
    hardware: HardwareControlConfig,
    chirp: JointChirpConfig,
    initial_logical: np.ndarray,
    output: IO[str],
) -> None:
    with armed_control_session(
        binding,
        groups,
        state_reader,
        initial_frame,
        adapter,
        hardware,
        thread_name="bello-joint-chirp",
    ) as session:
        session.reset_deadline()

        def hold_step() -> bool:
            session.send_target(initial_frame.positions)
            session.pace()
            _read_running_frame(session)
            return True

        wait_for_operator_confirmation_while_holding(
            CHIRP_CONFIRMATION,
            "Controls are holding the measured pose. Clear the robot, then type "
            "CHIRP to begin the bounded excitation: ",
            hold_step,
        )
        session.reset_deadline()
        selected_index = adapter.logical_joint_names.index(chirp.joint)
        total_steps = math.ceil(chirp.total_duration_seconds * chirp.rate_hz)
        flush_interval = max(1, round(chirp.rate_hz))
        for cycle in range(total_steps):
            elapsed_seconds = cycle * chirp.timestep
            point = chirp_point(elapsed_seconds, chirp)
            target_logical = initial_logical.copy()
            target_logical[selected_index] += point.offset_radians
            _send_capture_and_log(
                session,
                output,
                adapter,
                target_logical,
                point,
                cycle,
                elapsed_seconds,
            )
            if (cycle + 1) % flush_interval == 0:
                output.flush()
                print(
                    f"chirp {elapsed_seconds + chirp.timestep:.1f}/"
                    f"{chirp.total_duration_seconds:.1f} s ({point.phase})",
                    file=sys.stderr,
                )

        _send_capture_and_log(
            session,
            output,
            adapter,
            initial_logical.copy(),
            ChirpPoint("final_hold", 0.0, None),
            total_steps,
            chirp.total_duration_seconds,
        )
        session.drain()
        output.flush()


def run(
    hardware: HardwareControlConfig,
    chirp: JointChirpConfig,
    output_path: Path,
) -> int:
    validate_hardware_control_config(hardware)
    robot_definition_config = robot_definition(chirp.asset)
    if robot_definition_config.hardware_description is None:
        raise ValueError(
            "joint chirp requires a calibrated hardware description; "
            f"robot asset {chirp.asset!r} has none"
        )
    binding = import_hardware_binding()
    require_state_frame_binding(binding)
    if hardware.arm:
        require_async_hardware_binding(binding)

    with binding.Robot.connect(
        hardware.endpoint, timeout=hardware.connection_timeout_seconds
    ) as robot:
        unknown_groups = sorted(set(hardware.groups) - set(robot.group_names))
        if unknown_groups:
            raise ValueError(f"hardware inventory is missing groups: {unknown_groups}")
        physical_names = tuple(
            coordinate
            for group_name in hardware.groups
            for coordinate in robot.group(group_name).coordinates
        )
        adapter = build_hardware_adapter(
            physical_names,
            chirp.rate_hz,
            robot_definition_config,
        )
        groups = validate_hardware_inventory(robot, adapter, hardware, binding)
        state_reader = robot.states(groups=hardware.groups, imus=(hardware.imu_name,))
        initial_frame = wait_for_hardware_frame(
            state_reader,
            adapter.physical_coordinate_names,
            hardware.groups,
            hardware.imu_name,
            hardware.telemetry_timeout_seconds,
        )
        validate_hardware_frame(initial_frame, hardware, expected_group_state="IDLE")
        initial_logical = validate_experiment(
            robot,
            adapter,
            initial_frame.positions,
            chirp,
            hardware.gain_scale,
        )
        selected_index = adapter.logical_joint_names.index(chirp.joint)
        print(
            json.dumps(
                {
                    "joint_chirp_preflight": "passed",
                    "asset": adapter.asset,
                    "endpoint": hardware.endpoint,
                    "groups": hardware.groups,
                    "joint": chirp.joint,
                    "initial_radians": float(initial_logical[selected_index]),
                    "amplitude_radians": chirp.amplitude_radians,
                    "frequency_hz": [
                        chirp.start_frequency_hz,
                        chirp.end_frequency_hz,
                    ],
                    "rate_hz": chirp.rate_hz,
                    "duration_seconds": chirp.total_duration_seconds,
                    "output": str(output_path),
                    "armed": hardware.arm,
                },
                indent=2,
            )
        )
        if not hardware.arm:
            print(
                "Read-only preflight complete; no lease was acquired and no motor command was sent."
            )
            return 0

        require_operator_confirmation(
            HOLD_CONFIRMATION,
            "Support/restrain the robot and verify the E-stop, then type HOLD "
            "to acquire control at the measured pose: ",
        )
        initial_frame = wait_for_hardware_frame(
            state_reader,
            adapter.physical_coordinate_names,
            hardware.groups,
            hardware.imu_name,
            hardware.telemetry_timeout_seconds,
        )
        validate_hardware_frame(initial_frame, hardware, expected_group_state="IDLE")
        initial_logical = validate_experiment(
            robot,
            adapter,
            initial_frame.positions,
            chirp,
            hardware.gain_scale,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x", encoding="utf-8", buffering=1024 * 1024) as output:
            write_record(output, inventory_record(robot, hardware.endpoint))
            write_record(
                output,
                _config_record(
                    hardware,
                    chirp,
                    adapter,
                    initial_frame.positions,
                    initial_logical,
                ),
            )
            output.flush()
            execute_chirp(
                binding,
                groups,
                state_reader,
                initial_frame,
                adapter,
                hardware,
                chirp,
                initial_logical,
                output,
            )
    print(f"joint chirp complete; wrote {output_path}")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint", required=True, help="logical joint to excite")
    parser.add_argument(
        "--asset",
        choices=get_args(AssetName),
        default="full_body",
        help="robot definition used for joint and coupling contracts",
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:50051",
        help="bello-hardwared endpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["lower_body", "upper_body"],
        help="motor groups to hold (default: lower_body upper_body)",
    )
    parser.add_argument("--imu", default="imu", help="IMU name (default: %(default)s)")
    parser.add_argument("--rate", type=float, default=50.0, help="command rate in Hz")
    parser.add_argument(
        "--amplitude", type=float, default=0.02, help="peak joint offset in radians"
    )
    parser.add_argument("--start-frequency", type=float, default=0.5, help="Hz")
    parser.add_argument("--end-frequency", type=float, default=10.0, help="Hz")
    parser.add_argument("--duration", type=float, default=30.0, help="chirp seconds")
    parser.add_argument("--settle-before", type=float, default=3.0, help="seconds")
    parser.add_argument("--settle-after", type=float, default=2.0, help="seconds")
    parser.add_argument("--envelope", type=float, default=1.0, help="ramp seconds")
    parser.add_argument(
        "--gain-scale", type=float, default=1.0, help="scale repository PD gains"
    )
    parser.add_argument("--connect-timeout", type=float, default=3.0, help="seconds")
    parser.add_argument("--output", type=Path, help="JSONL destination")
    parser.add_argument(
        "--arm",
        action="store_true",
        help="acquire control and run the chirp (default is read-only preflight)",
    )
    args = parser.parse_args(argv)
    try:
        JointChirpConfig(
            joint=args.joint,
            asset=args.asset,
            rate_hz=args.rate,
            amplitude_radians=args.amplitude,
            start_frequency_hz=args.start_frequency,
            end_frequency_hz=args.end_frequency,
            duration_seconds=args.duration,
            settle_before_seconds=args.settle_before,
            settle_after_seconds=args.settle_after,
            envelope_seconds=args.envelope,
        )
        hardware = HardwareControlConfig(
            endpoint=args.endpoint,
            groups=tuple(args.groups),
            imu_name=args.imu,
            client_name="bello-joint-chirp",
            arm=args.arm,
            gain_scale=args.gain_scale,
            connection_timeout_seconds=args.connect_timeout,
        )
        validate_hardware_control_config(hardware)
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    chirp = JointChirpConfig(
        joint=args.joint,
        asset=args.asset,
        rate_hz=args.rate,
        amplitude_radians=args.amplitude,
        start_frequency_hz=args.start_frequency,
        end_frequency_hz=args.end_frequency,
        duration_seconds=args.duration,
        settle_before_seconds=args.settle_before,
        settle_after_seconds=args.settle_after,
        envelope_seconds=args.envelope,
    )
    hardware = HardwareControlConfig(
        endpoint=args.endpoint,
        groups=tuple(args.groups),
        imu_name=args.imu,
        client_name="bello-joint-chirp",
        arm=args.arm,
        gain_scale=args.gain_scale,
        connection_timeout_seconds=args.connect_timeout,
    )
    output_path = args.output or default_output_path(chirp.joint)
    try:
        return run(hardware, chirp, output_path)
    except KeyboardInterrupt:
        print("stopped", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
