"""RobStride private-CAN protocol and actuator device emulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
import math
import socket
import struct
from typing import Any, Literal, Mapping
import xml.etree.ElementTree as ET

from omegaconf import OmegaConf

CAN_EFF_FLAG = 0x80000000
CAN_EFF_MASK = 0x1FFFFFFF
CAN_FRAME = struct.Struct("=IB3x8s")


class CommType(IntEnum):
    GET_DEVICE_ID = 0
    MOTOR_CONTROL = 1
    MOTOR_FEEDBACK = 2
    MOTOR_ENABLE = 3
    MOTOR_STOP = 4
    SET_MECHANICAL_ZERO = 6
    PARAM_READ = 17
    PARAM_WRITE = 18
    DATA_SAVE = 22
    ACTIVE_REPORT = 24


PARAM_RUN_MODE = 0x7005
PARAM_IQ_REF = 0x7006
PARAM_SPEED_REF = 0x700A
PARAM_LIMIT_TORQUE = 0x700B
PARAM_CURRENT_KP = 0x7010
PARAM_CURRENT_KI = 0x7011
PARAM_CURRENT_FILTER_GAIN = 0x7014
PARAM_POSITION_REF = 0x7016
PARAM_LIMIT_SPEED = 0x7017
PARAM_LIMIT_CURRENT = 0x7018
PARAM_MECHANICAL_POSITION = 0x7019
PARAM_FILTERED_CURRENT = 0x701A
PARAM_MECHANICAL_VELOCITY = 0x701B
PARAM_BUS_VOLTAGE = 0x701C
PARAM_POSITION_KP = 0x701E
PARAM_SPEED_KP = 0x701F
PARAM_SPEED_KI = 0x7020
PARAM_SPEED_FILTER_GAIN = 0x7021
PARAM_ACCELERATION = 0x7022
PARAM_MAX_VELOCITY = 0x7024
PARAM_POSITION_ACCELERATION = 0x7025
PARAM_ACTIVE_REPORT_PERIOD = 0x7026
PARAM_CAN_TIMEOUT = 0x7028
PARAM_ZERO_RANGE = 0x7029
PARAM_DAMPER = 0x702A
PARAM_POSITION_OFFSET = 0x702B
PARAM_COGGING_ENABLE = 0x702C
PARAM_CALIBRATION_ENABLE = 0x702D
PARAM_POSITION_DECELERATION = 0x702E

PARAM_TYPES: dict[int, str] = {
    PARAM_RUN_MODE: "B",
    PARAM_IQ_REF: "f",
    PARAM_SPEED_REF: "f",
    PARAM_LIMIT_TORQUE: "f",
    PARAM_CURRENT_KP: "f",
    PARAM_CURRENT_KI: "f",
    PARAM_CURRENT_FILTER_GAIN: "f",
    PARAM_POSITION_REF: "f",
    PARAM_LIMIT_SPEED: "f",
    PARAM_LIMIT_CURRENT: "f",
    PARAM_MECHANICAL_POSITION: "f",
    PARAM_FILTERED_CURRENT: "f",
    PARAM_MECHANICAL_VELOCITY: "f",
    PARAM_BUS_VOLTAGE: "f",
    PARAM_POSITION_KP: "f",
    PARAM_SPEED_KP: "f",
    PARAM_SPEED_KI: "f",
    PARAM_SPEED_FILTER_GAIN: "f",
    PARAM_ACCELERATION: "f",
    PARAM_MAX_VELOCITY: "f",
    PARAM_POSITION_ACCELERATION: "f",
    PARAM_ACTIVE_REPORT_PERIOD: "H",
    PARAM_CAN_TIMEOUT: "I",
    PARAM_ZERO_RANGE: "B",
    PARAM_DAMPER: "B",
    PARAM_POSITION_OFFSET: "f",
    PARAM_COGGING_ENABLE: "B",
    PARAM_CALIBRATION_ENABLE: "B",
    PARAM_POSITION_DECELERATION: "f",
}


@dataclass(frozen=True)
class CanFrame:
    can_id: int
    data: bytes = b""

    def __post_init__(self) -> None:
        if not 0 <= self.can_id <= CAN_EFF_MASK:
            raise ValueError(f"extended CAN id out of range: {self.can_id:#x}")
        if len(self.data) > 8:
            raise ValueError(
                f"classic CAN payload cannot exceed 8 bytes: {len(self.data)}"
            )

    def pack(self) -> bytes:
        return CAN_FRAME.pack(
            self.can_id | CAN_EFF_FLAG,
            len(self.data),
            self.data.ljust(8, b"\0"),
        )

    @classmethod
    def unpack(cls, raw: bytes) -> CanFrame:
        if len(raw) != CAN_FRAME.size:
            raise ValueError(
                f"expected {CAN_FRAME.size}-byte can_frame, got {len(raw)}"
            )
        raw_id, dlc, payload = CAN_FRAME.unpack(raw)
        if raw_id & CAN_EFF_FLAG == 0:
            raise ValueError("RobStride requires a 29-bit extended CAN frame")
        if dlc > 8:
            raise ValueError(f"invalid CAN DLC: {dlc}")
        return cls(raw_id & CAN_EFF_MASK, payload[:dlc])


def build_can_id(comm_type: CommType | int, data_field: int, target_id: int) -> int:
    if not 0 <= int(comm_type) <= 0x1F:
        raise ValueError(f"communication type out of range: {comm_type}")
    if not 0 <= data_field <= 0xFFFF:
        raise ValueError(f"CAN data field out of range: {data_field}")
    if not 0 <= target_id <= 0xFF:
        raise ValueError(f"CAN target id out of range: {target_id}")
    return (int(comm_type) << 24) | (data_field << 8) | target_id


def comm_type(can_id: int) -> int:
    return (can_id >> 24) & 0x1F


def data_field(can_id: int) -> int:
    return (can_id >> 8) & 0xFFFF


def target_id(can_id: int) -> int:
    return can_id & 0xFF


def float_to_uint(value: float, minimum: float, maximum: float) -> int:
    value = min(max(float(value), minimum), maximum)
    return int((value - minimum) * 65535.0 / (maximum - minimum))


def uint_to_float(value: int, minimum: float, maximum: float) -> float:
    return int(value) * (maximum - minimum) / 65535.0 + minimum


@dataclass(frozen=True)
class MotorProtocolLimits:
    position: tuple[float, float] = (-12.57, 12.57)
    velocity: tuple[float, float] = (-33.0, 33.0)
    torque: tuple[float, float] = (-14.0, 14.0)
    kp: tuple[float, float] = (0.0, 500.0)
    kd: tuple[float, float] = (0.0, 5.0)


TargetType = Literal["joint", "tendon"]


@dataclass(frozen=True)
class MotorHardwareLimits:
    no_load_speed_rpm: float
    rated_voltage: float
    peak_phase_current: float


@dataclass(frozen=True)
class MotorModelSpec:
    name: str
    protocol: MotorProtocolLimits
    hardware: MotorHardwareLimits | None


@dataclass(frozen=True)
class MotorTargetSpec:
    type: TargetType
    name: str
    owned_joints: tuple[str, ...] = ()
    coupling_group: str | None = None
    coupling_size: int | None = None


@dataclass(frozen=True)
class MotorSpec:
    name: str
    motor_id: int
    bus: str
    model: MotorModelSpec
    target: MotorTargetSpec
    direction: int


@dataclass(frozen=True)
class BusSpec:
    name: str
    interface: str
    host_id: int


@dataclass(frozen=True)
class ImuSpec:
    name: str
    source_frame: str
    output_frame: str


@dataclass(frozen=True)
class RobstrideConfig:
    buses: tuple[BusSpec, ...]
    motors: tuple[MotorSpec, ...]
    imus: tuple[ImuSpec, ...]


def load_robstride_description(
    hardware_path: Path,
    model_path: Path,
    bus_interfaces: Mapping[str, str],
    *,
    enabled_groups: tuple[str, ...] = ("lower_body", "upper_body"),
) -> RobstrideConfig:
    hardware = _load_mapping(hardware_path, "Bello hardware description")
    expected_sections = {
        "robot_description",
        "logical_buses",
        "serial_devices",
        "imus",
        "motors",
        "motor_groups",
    }
    if set(hardware) != expected_sections:
        raise ValueError(
            "Bello hardware description sections must be exactly "
            + ", ".join(sorted(expected_sections))
        )
    for section in ("serial_devices", "imus"):
        if not isinstance(hardware[section], list) or not hardware[section]:
            raise ValueError(f"Bello {section} must be a nonempty list")
    models = load_motor_models(model_path)
    models_by_name = {model.name: model for model in models}
    robot_description_path = _robot_description_path(
        hardware_path, hardware["robot_description"]
    )
    targets = load_urdf_motor_targets(robot_description_path)
    imus = _imu_specs(hardware["imus"], load_urdf_link_names(robot_description_path))

    bus_descriptions = hardware["logical_buses"]
    if not isinstance(bus_descriptions, list) or not bus_descriptions:
        raise ValueError("Bello logical_buses must be a nonempty list")
    buses_by_name: dict[str, BusSpec] = {}
    for entry in bus_descriptions:
        if not isinstance(entry, dict):
            raise ValueError("Bello logical bus entries must be mappings")
        expected_bus_keys = {
            "name",
            "host_id",
            "bitrate",
            "maximum_utilization",
            "reconnect",
        }
        if set(entry) != expected_bus_keys:
            raise ValueError(
                "Bello logical bus keys must be exactly "
                + ", ".join(sorted(expected_bus_keys))
            )
        name = entry["name"]
        host_id = entry["host_id"]
        if not isinstance(name, str) or not name or name in buses_by_name:
            raise ValueError(f"invalid or duplicate Bello logical bus name {name!r}")
        if isinstance(host_id, bool) or not isinstance(host_id, int):
            raise ValueError(f"Bello logical bus {name!r} host_id must be an integer")
        if not 0 <= host_id <= 0xFF:
            raise ValueError(f"Bello logical bus {name!r} host_id is out of range")
        buses_by_name[name] = BusSpec(
            name=name,
            interface="",
            host_id=host_id,
        )
    if set(bus_interfaces) != set(buses_by_name):
        raise ValueError(
            "emulator bus mappings must exactly match Bello hardware buses; "
            f"expected={sorted(buses_by_name)}, actual={sorted(bus_interfaces)}"
        )
    buses_list = []
    for name, bus in buses_by_name.items():
        interface = bus_interfaces[name]
        if not isinstance(interface, str) or not interface.strip():
            raise ValueError(f"emulator interface for {name!r} must be nonempty")
        buses_list.append(BusSpec(bus.name, interface, bus.host_id))
    buses = tuple(buses_list)

    raw_motors = hardware["motors"]
    if not isinstance(raw_motors, list) or not raw_motors:
        raise ValueError("Bello hardware motors must be a nonempty list")
    motor_entries: dict[str, dict[str, Any]] = {}
    for entry in raw_motors:
        if not isinstance(entry, dict):
            raise ValueError("Bello hardware motor entries must be mappings")
        name = entry.get("name")
        if not isinstance(name, str):
            raise ValueError("Bello hardware motor names must be strings")
        if not name or name in motor_entries:
            raise ValueError(f"invalid or duplicate Bello hardware motor name {name!r}")
        motor_entries[name] = entry
    all_motors = tuple(
        _motor_spec(entry, models_by_name, targets) for entry in motor_entries.values()
    )
    ids_by_bus: set[tuple[str, int]] = set()
    names: set[str] = set()
    bus_names = {bus.name for bus in buses}
    for motor in all_motors:
        key = (motor.bus, motor.motor_id)
        if key in ids_by_bus:
            raise ValueError(f"duplicate motor id {motor.motor_id} on {motor.bus}")
        if motor.name in names:
            raise ValueError(f"duplicate motor name {motor.name!r}")
        if motor.bus not in bus_names:
            raise ValueError(
                f"motor {motor.name!r} references undefined bus {motor.bus!r}"
            )
        ids_by_bus.add(key)
        names.add(motor.name)

    selected_motor_names = _selected_motor_names(
        hardware["motor_groups"],
        enabled_groups,
        available_motors=names,
    )
    motors = tuple(motor for motor in all_motors if motor.name in selected_motor_names)
    _validate_coupling_groups(all_motors)
    _validate_coupling_groups(motors)
    return RobstrideConfig(buses, motors, imus)


def _robot_description_path(hardware_path: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Bello robot_description must be a nonempty path")
    path = Path(value)
    if not path.is_absolute():
        path = hardware_path.parent / path
    if not path.is_file():
        raise ValueError(
            f"Bello robot_description must reference an existing file: {path}"
        )
    return path.resolve()


def load_motor_models(path: Path) -> tuple[MotorModelSpec, ...]:
    raw = _load_mapping(path, "RobStride model catalog")
    if set(raw) != {"models"}:
        raise ValueError(f"RobStride model catalog must contain only 'models': {path}")
    values = raw["models"]
    if not isinstance(values, dict) or not values:
        raise ValueError("models must be a nonempty mapping")
    models = tuple(
        _motor_model_spec(str(name), value) for name, value in values.items()
    )
    if len({model.name for model in models}) != len(models):
        raise ValueError("duplicate RobStride model names")
    return models


def _motor_model_spec(name: str, entry: dict[str, Any]) -> MotorModelSpec:
    if not name or not isinstance(entry, dict):
        raise ValueError(f"motor model {name!r} must be a mapping")
    protocol_fields = {"position", "velocity", "torque", "kp", "kd"}
    unknown = set(entry) - protocol_fields - {"hardware"}
    if unknown:
        raise ValueError(
            f"unknown motor-model keys for {name!r}: {', '.join(sorted(unknown))}"
        )
    protocol_limits = MotorProtocolLimits(
        position=_bounds(entry, "position"),
        velocity=_bounds(entry, "velocity"),
        torque=_bounds(entry, "torque"),
        kp=_bounds(entry, "kp"),
        kd=_bounds(entry, "kd"),
    )
    hardware = entry.get("hardware")
    hardware_limits = None
    if hardware is not None:
        if not isinstance(hardware, dict):
            raise ValueError(f"motor model {name!r} hardware must be a mapping")
        required_hardware = {
            "no_load_speed_rpm",
            "rated_voltage",
            "peak_phase_current",
        }
        if set(hardware) != required_hardware:
            raise ValueError(
                f"motor model {name!r} hardware keys must be exactly "
                + ", ".join(sorted(required_hardware))
            )
        hardware_limits = MotorHardwareLimits(
            no_load_speed_rpm=_positive_float(hardware, "no_load_speed_rpm"),
            rated_voltage=_positive_float(hardware, "rated_voltage"),
            peak_phase_current=_positive_float(hardware, "peak_phase_current"),
        )
    return MotorModelSpec(name, protocol_limits, hardware_limits)


def _motor_spec(
    entry: dict[str, Any],
    models_by_name: dict[str, MotorModelSpec],
    targets: dict[str, MotorTargetSpec],
) -> MotorSpec:
    required = {
        "name",
        "coordinate",
        "bus",
        "id",
        "model",
        "direction",
    }
    allowed = required | {
        "limits",
        "health",
        "command_policy",
        "fault_damping_kd",
    }
    missing = required - set(entry)
    if missing:
        raise ValueError(
            "missing Bello hardware motor keys: " + ", ".join(sorted(missing))
        )
    unknown = set(entry) - allowed
    if unknown:
        raise ValueError(
            "unknown Bello hardware motor keys: " + ", ".join(sorted(unknown))
        )
    if isinstance(entry["id"], bool) or not isinstance(entry["id"], int):
        raise ValueError("motor id must be an integer")
    motor_id = entry["id"]
    if not 0 <= motor_id <= 0xFF:
        raise ValueError(f"motor id out of range: {motor_id}")
    model_name = entry["model"]
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("motor model must be a nonempty string")
    try:
        model = models_by_name[model_name]
    except KeyError as error:
        raise ValueError(f"motor references unknown model {model_name!r}") from error
    coordinate = entry["coordinate"]
    if not isinstance(coordinate, str) or not coordinate:
        raise ValueError("motor coordinate must be a nonempty string")
    try:
        target = targets[coordinate]
    except KeyError as error:
        raise ValueError(
            f"motor coordinate {coordinate!r} is absent from the canonical URDF"
        ) from error
    if isinstance(entry["direction"], bool) or not isinstance(entry["direction"], int):
        raise ValueError("motor direction must be -1 or 1")
    direction = entry["direction"]
    if direction not in {-1, 1}:
        raise ValueError("motor direction must be -1 or 1")
    bus = entry["bus"]
    if not isinstance(bus, str) or not bus:
        raise ValueError("motor bus must be a nonempty string")
    return MotorSpec(
        name=entry["name"],
        motor_id=motor_id,
        bus=bus,
        model=model,
        target=target,
        direction=direction,
    )


def load_urdf_motor_targets(path: Path) -> dict[str, MotorTargetSpec]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    root = ET.parse(path).getroot()
    if root.tag != "robot":
        raise ValueError(f"canonical robot description is not a URDF: {path}")
    targets: dict[str, MotorTargetSpec] = {}
    joint_names: set[str] = set()
    movable_joint_names: set[str] = set()
    for joint in root.findall("joint"):
        name = joint.get("name", "")
        if not name or name in joint_names:
            raise ValueError(f"invalid or duplicate canonical URDF joint {name!r}")
        joint_names.add(name)
        if joint.get("type") != "fixed":
            movable_joint_names.add(name)
            targets[name] = MotorTargetSpec(type="joint", name=name)

    used_coordinates: set[str] = set()
    used_joints: set[str] = set()
    for transmission in root.iter("transmission"):
        transmission_type = next(
            (
                (child.text or "").strip()
                for child in transmission
                if child.tag in {"type", "plugin"}
            ),
            "",
        )
        if not transmission_type.endswith("DifferentialTransmission"):
            continue
        name = transmission.get("name", "unnamed")
        actuator_names = _ordered_transmission_members(
            transmission, "actuator", "actuator", name
        )
        owned_joint_names = _ordered_transmission_members(
            transmission, "joint", "joint", name
        )
        for actuator_name in actuator_names:
            if actuator_name in used_coordinates or actuator_name in targets:
                raise ValueError(f"transmission coordinate {actuator_name!r} is reused")
            used_coordinates.add(actuator_name)
        for joint_name in owned_joint_names:
            if joint_name not in movable_joint_names:
                raise ValueError(
                    f"transmission {name!r} references unknown or fixed joint "
                    f"{joint_name!r}"
                )
            if joint_name in used_joints:
                raise ValueError(f"transmission joint {joint_name!r} is reused")
            used_joints.add(joint_name)
            targets.pop(joint_name)
        for actuator_name in actuator_names:
            targets[actuator_name] = MotorTargetSpec(
                type="tendon",
                name=actuator_name,
                owned_joints=owned_joint_names,
                coupling_group=name,
                coupling_size=len(actuator_names),
            )
    return targets


def load_urdf_link_names(path: Path) -> frozenset[str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    root = ET.parse(path).getroot()
    if root.tag != "robot":
        raise ValueError(f"canonical robot description is not a URDF: {path}")
    names = [link.get("name", "") for link in root.findall("link")]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("canonical URDF links must have unique nonempty names")
    return frozenset(names)


def _imu_specs(raw: Any, link_names: frozenset[str]) -> tuple[ImuSpec, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("Bello hardware imus must be a nonempty list")
    required = {
        "name",
        "serial_device",
        "source_frame",
        "output_frame",
        "stale_timeout_ms",
        "output_rate_hz",
    }
    specs = []
    names: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError("Bello hardware IMU entries must be mappings")
        if set(entry) != required:
            raise ValueError(
                "Bello hardware IMU keys must be exactly " + ", ".join(sorted(required))
            )
        name = entry["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"invalid or duplicate Bello hardware IMU name {name!r}")
        frames = {}
        for frame_field in ("source_frame", "output_frame"):
            frame = entry[frame_field]
            if not isinstance(frame, str) or not frame:
                raise ValueError(
                    f"Bello hardware IMU {name!r} {frame_field} must be nonempty"
                )
            if frame not in link_names:
                raise ValueError(
                    f"Bello hardware IMU {name!r} {frame_field} {frame!r} "
                    "is absent from the canonical URDF"
                )
            frames[frame_field] = frame
        specs.append(
            ImuSpec(
                name=name,
                source_frame=frames["source_frame"],
                output_frame=frames["output_frame"],
            )
        )
        names.add(name)
    return tuple(specs)


def _load_mapping(path: Path, description: str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError(f"{description} must be a mapping: {path}")
    return raw


def _selected_motor_names(
    raw_groups: Any,
    selected: tuple[str, ...],
    *,
    available_motors: set[str],
) -> set[str]:
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("enabled Bello hardware groups must be unique and nonempty")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError("Bello hardware motor_groups must be a nonempty list")
    groups: dict[str, tuple[str, ...]] = {}
    allowed = {
        "name",
        "motors",
        "control_rate_hz",
        "command_refresh_rate_hz",
        "command_timeout_ms",
        "feedback_timeout_ms",
        "enable_timeout_ms",
        "enable_retries",
        "fault_policy",
    }
    for entry in raw_groups:
        if not isinstance(entry, dict):
            raise ValueError("Bello hardware motor-group entries must be mappings")
        unknown = set(entry) - allowed
        if unknown:
            raise ValueError(
                "unknown Bello hardware motor-group keys: " + ", ".join(sorted(unknown))
            )
        name = entry.get("name")
        motors = entry.get("motors")
        if (
            not isinstance(name, str)
            or not name
            or name in groups
            or not isinstance(motors, list)
            or not motors
            or not all(isinstance(item, str) and item for item in motors)
        ):
            raise ValueError(f"invalid Bello hardware motor group {name!r}")
        if len(set(motors)) != len(motors):
            raise ValueError(f"Bello hardware group {name!r} repeats motors")
        groups[name] = tuple(motors)

    memberships = [motor for motors in groups.values() for motor in motors]
    if len(set(memberships)) != len(memberships):
        raise ValueError("Bello hardware motor groups share motors")
    described_motors = set(memberships)
    if described_motors != available_motors:
        details = []
        undefined = described_motors - available_motors
        ungrouped = available_motors - described_motors
        if undefined:
            details.append("undefined: " + ", ".join(sorted(undefined)))
        if ungrouped:
            details.append("ungrouped: " + ", ".join(sorted(ungrouped)))
        raise ValueError(
            "Bello hardware motor groups are incomplete: " + "; ".join(details)
        )

    missing = set(selected) - set(groups)
    if missing:
        raise ValueError(
            "required emulator motor groups are absent: " + ", ".join(sorted(missing))
        )
    return {motor for group in selected for motor in groups[group]}


def _ordered_transmission_members(
    transmission: ET.Element,
    tag: str,
    role_prefix: str,
    transmission_name: str,
) -> tuple[str, str]:
    members: list[tuple[int | None, str]] = []
    for element in transmission.findall(tag):
        name = element.get("name", "")
        if not name:
            raise ValueError(f"transmission {transmission_name!r} has an unnamed {tag}")
        role = element.get("role", "")
        suffix = role.removeprefix(role_prefix) if role.startswith(role_prefix) else ""
        order = int(suffix) if suffix.isdigit() else None
        reduction_text = element.findtext("mechanical_reduction")
        if reduction_text is None:
            reduction_text = element.findtext("mechanicalReduction")
        reduction = 1.0 if reduction_text is None else float(reduction_text)
        offset_text = element.findtext("offset")
        offset = 0.0 if offset_text is None else float(offset_text)
        if not math.isfinite(reduction) or abs(reduction) <= 1.0e-12:
            raise ValueError(
                f"transmission {transmission_name!r} member {name!r} "
                "has invalid reduction"
            )
        if not math.isfinite(offset):
            raise ValueError(
                f"transmission {transmission_name!r} member {name!r} has invalid offset"
            )
        members.append((order, name))
    if len(members) != 2:
        raise ValueError(
            f"differential transmission {transmission_name!r} must have "
            f"exactly two {tag}s"
        )
    if all(order is not None for order, _ in members):
        members.sort(key=lambda item: int(item[0]))
        if [order for order, _ in members] != [1, 2]:
            raise ValueError(
                f"transmission {transmission_name!r} {tag} roles must be "
                f"{role_prefix}1 and {role_prefix}2"
            )
    return members[0][1], members[1][1]


def _validate_coupling_groups(motors: tuple[MotorSpec, ...]) -> None:
    groups: dict[str, list[MotorTargetSpec]] = {}
    for motor in motors:
        if motor.target.coupling_group is not None:
            groups.setdefault(motor.target.coupling_group, []).append(motor.target)
    for name, targets in groups.items():
        sizes = {target.coupling_size for target in targets}
        if len(sizes) != 1:
            raise ValueError(f"coupling group {name!r} has inconsistent sizes")
        expected = sizes.pop()
        if expected is None or len(targets) != expected:
            raise ValueError(
                f"coupling group {name!r} requires {expected} motors; got {len(targets)}"
            )
        if len({target.owned_joints for target in targets}) != 1:
            raise ValueError(f"coupling group {name!r} has inconsistent owned joints")
        if len({target.name for target in targets}) != len(targets):
            raise ValueError(f"coupling group {name!r} repeats a tendon target")


def _bounds(values: dict[str, Any], key: str) -> tuple[float, float]:
    if key not in values:
        raise ValueError(f"missing {key} bounds")
    bounds = values[key]
    if (
        not isinstance(bounds, (list, tuple))
        or len(bounds) != 2
        or not all(math.isfinite(float(value)) for value in bounds)
        or float(bounds[0]) >= float(bounds[1])
    ):
        raise ValueError(f"invalid {key} bounds: {bounds}")
    return float(bounds[0]), float(bounds[1])


def _positive_float(values: dict[str, Any], key: str) -> float:
    value = float(values[key])
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{key} must be finite and positive")
    return value


@dataclass(frozen=True)
class MotorCommand:
    position: float = 0.0
    velocity: float = 0.0
    torque: float = 0.0
    kp: float = 0.0
    kd: float = 0.0


@dataclass
class RobstrideMotor:
    spec: MotorSpec
    temperature: float = 25.0
    mode_status: int = 0
    fault_bits: int = 0
    active_reporting: bool = False
    mechanical_position: float = 0.0
    mechanical_velocity: float = 0.0
    applied_torque: float = 0.0
    zero_offset: float = 0.0
    command: MotorCommand = field(default_factory=MotorCommand)
    params: dict[int, int | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        limits = self.spec.model.protocol
        hardware = self.spec.model.hardware
        maximum_speed = limits.velocity[1]
        # Zero marks unavailable electrical telemetry for protocol-only models.
        limit_current = 0.0
        bus_voltage = 0.0
        if hardware is not None:
            maximum_speed = min(
                maximum_speed, hardware.no_load_speed_rpm * math.tau / 60.0
            )
            limit_current = hardware.peak_phase_current
            bus_voltage = hardware.rated_voltage
        self.params.update(
            {
                PARAM_RUN_MODE: 0,
                PARAM_IQ_REF: 0.0,
                PARAM_SPEED_REF: 0.0,
                PARAM_LIMIT_TORQUE: max(abs(value) for value in limits.torque),
                PARAM_CURRENT_KP: 0.17,
                PARAM_CURRENT_KI: 0.012,
                PARAM_CURRENT_FILTER_GAIN: 0.1,
                PARAM_POSITION_REF: 0.0,
                PARAM_LIMIT_SPEED: maximum_speed,
                PARAM_LIMIT_CURRENT: limit_current,
                PARAM_BUS_VOLTAGE: bus_voltage,
                PARAM_POSITION_KP: 60.0,
                PARAM_SPEED_KP: 6.0,
                PARAM_SPEED_KI: 0.02,
                PARAM_SPEED_FILTER_GAIN: 0.1,
                PARAM_ACCELERATION: 15.0,
                PARAM_MAX_VELOCITY: 10.0,
                PARAM_POSITION_ACCELERATION: 10.0,
                PARAM_ACTIVE_REPORT_PERIOD: 1,
                PARAM_CAN_TIMEOUT: 0,
                PARAM_ZERO_RANGE: 0,
                PARAM_DAMPER: 0,
                PARAM_POSITION_OFFSET: 0.0,
                PARAM_COGGING_ENABLE: 0,
                PARAM_CALIBRATION_ENABLE: 0,
                PARAM_POSITION_DECELERATION: 10.0,
            }
        )

    @property
    def limits(self) -> MotorProtocolLimits:
        return self.spec.model.protocol

    @property
    def reported_position(self) -> float:
        return self.mechanical_position - self.zero_offset

    @property
    def enabled(self) -> bool:
        return self.mode_status == 2

    @property
    def torque_limit(self) -> float:
        return min(
            abs(float(self.params[PARAM_LIMIT_TORQUE])),
            max(abs(item) for item in self.limits.torque),
        )

    def decode_control(self, frame: CanFrame) -> None:
        if len(frame.data) < 8:
            raise ValueError("RobStride motor-control frame must have 8 data bytes")
        limits = self.limits
        direction = self.spec.direction
        self.command = MotorCommand(
            position=uint_to_float(_be16(frame.data, 0), *limits.position) * direction,
            velocity=uint_to_float(_be16(frame.data, 2), *limits.velocity) * direction,
            torque=uint_to_float(data_field(frame.can_id), *limits.torque) * direction,
            kp=uint_to_float(_be16(frame.data, 4), *limits.kp),
            kd=uint_to_float(_be16(frame.data, 6), *limits.kd),
        )

    def feedback_frame(self, response_type: CommType, host_id: int) -> CanFrame:
        limits = self.limits
        direction = self.spec.direction
        status = (self.mode_status & 0x03) << 14
        faults = (self.fault_bits & 0x3F) << 8
        payload = b"".join(
            (
                _pack_be16(
                    float_to_uint(self.reported_position * direction, *limits.position)
                ),
                _pack_be16(
                    float_to_uint(
                        self.mechanical_velocity * direction, *limits.velocity
                    )
                ),
                _pack_be16(
                    float_to_uint(self.applied_torque * direction, *limits.torque)
                ),
                _pack_be16(int(max(0.0, self.temperature) * 10.0)),
            )
        )
        return CanFrame(
            build_can_id(response_type, status | faults | self.spec.motor_id, host_id),
            payload,
        )

    def parameter_value(self, index: int) -> int | float:
        if index == PARAM_MECHANICAL_POSITION:
            return self.reported_position * self.spec.direction
        if index == PARAM_MECHANICAL_VELOCITY:
            return self.mechanical_velocity * self.spec.direction
        if index == PARAM_FILTERED_CURRENT:
            return 0.0
        return self.params.get(index, 0.0)


class SocketCanEndpoint:
    def __init__(self, interface: str) -> None:
        if not hasattr(socket, "AF_CAN"):
            raise RuntimeError("SocketCAN is only available on Linux")
        self.interface = interface
        self.socket = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        try:
            self.socket.setblocking(False)
            self.socket.bind((interface,))
        except Exception:
            self.socket.close()
            raise

    def close(self) -> None:
        self.socket.close()

    def receive(self) -> CanFrame | None:
        try:
            raw = self.socket.recv(CAN_FRAME.size)
        except BlockingIOError:
            return None
        try:
            return CanFrame.unpack(raw)
        except ValueError:
            return None

    def send(self, frame: CanFrame) -> None:
        sent = self.socket.send(frame.pack())
        if sent != CAN_FRAME.size:
            raise OSError(f"short SocketCAN write: {sent}/{CAN_FRAME.size}")

    def __enter__(self) -> SocketCanEndpoint:
        return self

    def __exit__(self, *args) -> None:
        self.close()


class RobstrideBusEmulator:
    def __init__(self, spec: BusSpec, motors: tuple[MotorSpec, ...]) -> None:
        self.spec = spec
        self.motors = {motor.motor_id: RobstrideMotor(motor) for motor in motors}

    def handle_frame(self, frame: CanFrame) -> list[CanFrame]:
        motor = self.motors.get(target_id(frame.can_id))
        if motor is None:
            return []
        frame_type = comm_type(frame.can_id)
        host_id = data_field(frame.can_id) & 0xFF
        if frame_type == CommType.GET_DEVICE_ID:
            unique_id = struct.pack("<Q", _stable_device_id(motor.spec))
            return [
                CanFrame(
                    build_can_id(CommType.GET_DEVICE_ID, motor.spec.motor_id, 0xFE),
                    unique_id,
                )
            ]
        if frame_type == CommType.MOTOR_CONTROL:
            if len(frame.data) < 8:
                return []
            motor.decode_control(frame)
            return [motor.feedback_frame(CommType.MOTOR_FEEDBACK, self.spec.host_id)]
        if frame_type == CommType.MOTOR_ENABLE:
            motor.mode_status = 2
            motor.fault_bits = 0
            return [motor.feedback_frame(CommType.MOTOR_FEEDBACK, host_id)]
        if frame_type == CommType.MOTOR_STOP:
            motor.mode_status = 0
            motor.applied_torque = 0.0
            if frame.data and frame.data[0] != 0:
                motor.fault_bits = 0
            return [motor.feedback_frame(CommType.MOTOR_FEEDBACK, host_id)]
        if frame_type == CommType.SET_MECHANICAL_ZERO:
            if frame.data and frame.data[0] == 1:
                motor.zero_offset = motor.mechanical_position
                motor.command = MotorCommand()
            return [motor.feedback_frame(CommType.MOTOR_FEEDBACK, host_id)]
        if frame_type == CommType.PARAM_READ:
            return self._read_parameter(motor, frame, host_id)
        if frame_type == CommType.PARAM_WRITE:
            return self._write_parameter(motor, frame, host_id)
        if frame_type == CommType.DATA_SAVE:
            return [motor.feedback_frame(CommType.MOTOR_FEEDBACK, host_id)]
        if frame_type == CommType.ACTIVE_REPORT:
            motor.active_reporting = len(frame.data) > 6 and frame.data[6] != 0
            return [motor.feedback_frame(CommType.MOTOR_FEEDBACK, host_id)]
        return []

    def active_feedback(self) -> list[CanFrame]:
        return [
            motor.feedback_frame(CommType.ACTIVE_REPORT, self.spec.host_id)
            for motor in self.motors.values()
            if motor.active_reporting
        ]

    def reset_control_state(self) -> None:
        for motor in self.motors.values():
            motor.mode_status = 0
            motor.applied_torque = 0.0
            motor.command = MotorCommand()

    def _read_parameter(
        self, motor: RobstrideMotor, frame: CanFrame, host_id: int
    ) -> list[CanFrame]:
        if len(frame.data) < 2:
            return []
        index = _le16(frame.data, 0)
        kind = PARAM_TYPES.get(index)
        if kind is None:
            return []
        payload = bytearray(8)
        struct.pack_into("<H", payload, 0, index)
        struct.pack_into("<" + kind, payload, 4, motor.parameter_value(index))
        response = CanFrame(
            build_can_id(CommType.PARAM_READ, motor.spec.motor_id, host_id),
            bytes(payload),
        )
        replies = [response]
        if index == PARAM_MECHANICAL_POSITION:
            replies.append(motor.feedback_frame(CommType.MOTOR_FEEDBACK, host_id))
        return replies

    def _write_parameter(
        self, motor: RobstrideMotor, frame: CanFrame, host_id: int
    ) -> list[CanFrame]:
        if len(frame.data) < 5:
            return []
        index = _le16(frame.data, 0)
        kind = PARAM_TYPES.get(index)
        if kind is None:
            return []
        size = struct.calcsize(kind)
        if len(frame.data) < 4 + size or index in {
            PARAM_MECHANICAL_POSITION,
            PARAM_MECHANICAL_VELOCITY,
            PARAM_FILTERED_CURRENT,
        }:
            return [motor.feedback_frame(CommType.MOTOR_FEEDBACK, host_id)]
        value = struct.unpack_from("<" + kind, frame.data, 4)[0]
        if index == PARAM_RUN_MODE and value != 0:
            return []
        motor.params[index] = value
        return [
            CanFrame(
                build_can_id(CommType.PARAM_WRITE, motor.spec.motor_id, host_id),
                frame.data.ljust(8, b"\0"),
            )
        ]


def _stable_device_id(spec: MotorSpec) -> int:
    value = 0x5253000000000000 | (spec.motor_id << 32)
    for byte in spec.model.name.encode("ascii"):
        value = ((value << 5) - value + byte) & 0xFFFFFFFFFFFFFFFF
    return value


def _be16(data: bytes, offset: int) -> int:
    return struct.unpack_from(">H", data, offset)[0]


def _le16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def _pack_be16(value: int) -> bytes:
    return struct.pack(">H", min(max(int(value), 0), 0xFFFF))
