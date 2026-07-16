"""RobStride private-CAN protocol and actuator device emulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
import math
import socket
import struct
from typing import Any, Literal

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
    SET_CAN_ID = 7
    PARAM_READ = 17
    PARAM_WRITE = 18
    FAULT_FEEDBACK = 21
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


RUN_MODE_NAMES = {"mit": 0, "position": 1, "velocity": 2, "current": 3, "csp": 5}
TargetType = Literal["joint", "tendon"]


@dataclass(frozen=True)
class MotorHardwareLimits:
    rated_torque: float
    peak_torque: float
    rated_speed_rpm: float
    no_load_speed_rpm: float
    rated_voltage: float
    voltage: tuple[float, float]
    rated_phase_current: float
    peak_phase_current: float
    reduction_ratio: float
    temperature_c: tuple[float, float]


@dataclass(frozen=True)
class MotorKindSpec:
    name: str
    protocol: MotorProtocolLimits
    hardware: MotorHardwareLimits


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
    kind: MotorKindSpec
    target: MotorTargetSpec
    direction: int
    run_mode: int


@dataclass(frozen=True)
class BusSpec:
    name: str
    interface: str
    host_id: int


@dataclass(frozen=True)
class RobstrideConfig:
    motor_kinds: tuple[MotorKindSpec, ...]
    buses: tuple[BusSpec, ...]
    motors: tuple[MotorSpec, ...]


def load_robstride_configs(
    motor_kind_path: Path, paths: tuple[Path, ...]
) -> RobstrideConfig:
    if not paths:
        raise ValueError("at least one RobStride actuator config is required")
    motor_kinds = load_motor_kinds(motor_kind_path)
    kinds_by_name = {kind.name: kind for kind in motor_kinds}
    buses: dict[str, BusSpec] = {}
    motors: list[MotorSpec] = []
    ids_by_bus: set[tuple[str, int]] = set()
    names: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        if not isinstance(raw, dict):
            raise ValueError(f"RobStride config must be a mapping: {path}")
        for name, value in (raw.get("buses") or {}).items():
            bus = BusSpec(
                name=str(name),
                interface=str(value.get("interface", name)),
                host_id=int(value.get("host_id", 0)),
            )
            previous = buses.get(bus.name)
            if previous is not None and previous != bus:
                raise ValueError(f"conflicting definitions for bus {bus.name!r}")
            buses[bus.name] = bus
        unknown = set(raw) - {"buses", "motors"}
        if unknown:
            raise ValueError(
                f"unknown RobStride assignment keys in {path}: "
                + ", ".join(sorted(unknown))
            )
        for entry in raw.get("motors") or []:
            motor = _motor_spec(entry, kinds_by_name)
            key = (motor.bus, motor.motor_id)
            if key in ids_by_bus:
                raise ValueError(f"duplicate motor id {motor.motor_id} on {motor.bus}")
            if motor.name in names:
                raise ValueError(f"duplicate motor name {motor.name!r}")
            if motor.bus not in buses:
                raise ValueError(
                    f"motor {motor.name!r} references undefined bus {motor.bus!r}"
                )
            ids_by_bus.add(key)
            names.add(motor.name)
            motors.append(motor)
    _validate_coupling_groups(tuple(motors))
    return RobstrideConfig(motor_kinds, tuple(buses.values()), tuple(motors))


def load_motor_kinds(path: Path) -> tuple[MotorKindSpec, ...]:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict) or set(raw) != {"motor_kinds"}:
        raise ValueError(f"motor-kind config must contain only 'motor_kinds': {path}")
    values = raw["motor_kinds"]
    if not isinstance(values, dict) or not values:
        raise ValueError("motor_kinds must be a nonempty mapping")
    kinds = tuple(_motor_kind_spec(str(name), value) for name, value in values.items())
    if len({kind.name for kind in kinds}) != len(kinds):
        raise ValueError("duplicate RobStride motor-kind names")
    return kinds


def _motor_kind_spec(name: str, entry: dict[str, Any]) -> MotorKindSpec:
    if not isinstance(entry, dict):
        raise ValueError(f"motor kind {name!r} must be a mapping")
    unknown = set(entry) - {"protocol", "hardware"}
    if unknown:
        raise ValueError(
            f"unknown motor-kind keys for {name!r}: {', '.join(sorted(unknown))}"
        )
    protocol = entry.get("protocol")
    hardware = entry.get("hardware")
    if not isinstance(protocol, dict) or not isinstance(hardware, dict):
        raise ValueError(f"motor kind {name!r} requires protocol and hardware mappings")
    protocol_unknown = set(protocol) - {"position", "velocity", "torque", "kp", "kd"}
    if protocol_unknown:
        raise ValueError(
            f"unknown protocol keys for {name!r}: "
            + ", ".join(sorted(protocol_unknown))
        )
    required_hardware = {
        "rated_torque",
        "peak_torque",
        "rated_speed_rpm",
        "no_load_speed_rpm",
        "rated_voltage",
        "voltage",
        "rated_phase_current",
        "peak_phase_current",
        "reduction_ratio",
        "temperature_c",
    }
    if set(hardware) != required_hardware:
        missing = required_hardware - set(hardware)
        unknown = set(hardware) - required_hardware
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown " + ", ".join(sorted(unknown)))
        raise ValueError(f"invalid hardware keys for {name!r}: {'; '.join(details)}")
    protocol_limits = MotorProtocolLimits(
        position=_bounds(protocol, "position"),
        velocity=_bounds(protocol, "velocity"),
        torque=_bounds(protocol, "torque"),
        kp=_bounds(protocol, "kp"),
        kd=_bounds(protocol, "kd"),
    )
    hardware_limits = MotorHardwareLimits(
        rated_torque=_positive_float(hardware, "rated_torque"),
        peak_torque=_positive_float(hardware, "peak_torque"),
        rated_speed_rpm=_positive_float(hardware, "rated_speed_rpm"),
        no_load_speed_rpm=_positive_float(hardware, "no_load_speed_rpm"),
        rated_voltage=_positive_float(hardware, "rated_voltage"),
        voltage=_bounds(hardware, "voltage"),
        rated_phase_current=_positive_float(hardware, "rated_phase_current"),
        peak_phase_current=_positive_float(hardware, "peak_phase_current"),
        reduction_ratio=_positive_float(hardware, "reduction_ratio"),
        temperature_c=_bounds(hardware, "temperature_c"),
    )
    if hardware_limits.rated_torque > hardware_limits.peak_torque:
        raise ValueError(f"motor kind {name!r} rated torque exceeds peak torque")
    if hardware_limits.rated_speed_rpm > hardware_limits.no_load_speed_rpm:
        raise ValueError(f"motor kind {name!r} rated speed exceeds no-load speed")
    if hardware_limits.rated_phase_current > hardware_limits.peak_phase_current:
        raise ValueError(
            f"motor kind {name!r} rated phase current exceeds peak phase current"
        )
    if not (
        hardware_limits.voltage[0]
        <= hardware_limits.rated_voltage
        <= hardware_limits.voltage[1]
    ):
        raise ValueError(
            f"motor kind {name!r} rated voltage is outside its voltage range"
        )
    if protocol_limits.torque != (
        -hardware_limits.peak_torque,
        hardware_limits.peak_torque,
    ):
        raise ValueError(
            f"motor kind {name!r} protocol torque range must match hardware peak torque"
        )
    return MotorKindSpec(name, protocol_limits, hardware_limits)


def _motor_spec(
    entry: dict[str, Any], kinds_by_name: dict[str, MotorKindSpec]
) -> MotorSpec:
    if not isinstance(entry, dict):
        raise ValueError("motor assignment must be a mapping")
    allowed = {"name", "id", "bus", "kind", "target", "direction", "run_mode"}
    unknown = set(entry) - allowed
    if unknown:
        raise ValueError("unknown motor assignment keys: " + ", ".join(sorted(unknown)))
    run_mode = entry.get("run_mode", "mit")
    if isinstance(run_mode, str):
        try:
            run_mode = RUN_MODE_NAMES[run_mode.lower()]
        except KeyError as error:
            raise ValueError(f"unsupported run mode: {run_mode!r}") from error
    motor_id = int(entry["id"])
    if not 0 <= motor_id <= 0xFF:
        raise ValueError(f"motor id out of range: {motor_id}")
    kind_name = str(entry.get("kind", ""))
    try:
        kind = kinds_by_name[kind_name]
    except KeyError as error:
        raise ValueError(f"motor references unknown kind {kind_name!r}") from error
    return MotorSpec(
        name=str(entry.get("name", f"motor_{motor_id}")),
        motor_id=motor_id,
        bus=str(entry.get("bus", "can0")),
        kind=kind,
        target=_motor_target_spec(entry.get("target")),
        direction=1 if int(entry.get("direction", 1)) >= 0 else -1,
        run_mode=int(run_mode),
    )


def _motor_target_spec(value: Any) -> MotorTargetSpec:
    if not isinstance(value, dict):
        raise ValueError("motor target must be a mapping")
    allowed = {"type", "name", "owned_joints", "coupling_group", "coupling_size"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown motor target keys: {', '.join(sorted(unknown))}")
    target_type = str(value.get("type", ""))
    if target_type not in {"joint", "tendon"}:
        raise ValueError("motor target type must be 'joint' or 'tendon'")
    name = str(value.get("name", ""))
    if not name:
        raise ValueError("motor target name must not be empty")
    owned_values = value.get("owned_joints", [])
    if not isinstance(owned_values, list) or not all(
        isinstance(item, str) and item for item in owned_values
    ):
        raise ValueError("motor target owned_joints must be a list of joint names")
    owned_joints = tuple(owned_values)
    coupling_group = value.get("coupling_group")
    coupling_size = value.get("coupling_size")
    if target_type == "joint":
        if owned_joints or coupling_group is not None or coupling_size is not None:
            raise ValueError(
                "joint targets cannot define owned_joints or coupling settings"
            )
    else:
        if not owned_joints:
            raise ValueError("tendon targets must define owned_joints")
        if not isinstance(coupling_group, str) or not coupling_group:
            raise ValueError("tendon targets must define coupling_group")
        if not isinstance(coupling_size, int) or coupling_size < 2:
            raise ValueError("tendon targets must define coupling_size >= 2")
    return MotorTargetSpec(
        type=target_type,
        name=name,
        owned_joints=owned_joints,
        coupling_group=coupling_group,
        coupling_size=coupling_size,
    )


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
        len(bounds) != 2
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
        limits = self.spec.kind.protocol
        hardware = self.spec.kind.hardware
        self.params.update(
            {
                PARAM_RUN_MODE: self.spec.run_mode,
                PARAM_IQ_REF: 0.0,
                PARAM_SPEED_REF: 0.0,
                PARAM_LIMIT_TORQUE: max(abs(value) for value in limits.torque),
                PARAM_CURRENT_KP: 0.17,
                PARAM_CURRENT_KI: 0.012,
                PARAM_CURRENT_FILTER_GAIN: 0.1,
                PARAM_POSITION_REF: 0.0,
                PARAM_LIMIT_SPEED: min(
                    limits.velocity[1], hardware.no_load_speed_rpm * math.tau / 60.0
                ),
                PARAM_LIMIT_CURRENT: hardware.peak_phase_current,
                PARAM_BUS_VOLTAGE: hardware.rated_voltage,
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
        return self.spec.kind.protocol

    @property
    def reported_position(self) -> float:
        return self.mechanical_position - self.zero_offset

    @property
    def enabled(self) -> bool:
        return self.mode_status == 2

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

    def compute_torque(self) -> float:
        if not self.enabled or int(self.params[PARAM_RUN_MODE]) != 0:
            self.applied_torque = 0.0
            return 0.0
        value = (
            self.command.kd * (self.command.velocity - self.mechanical_velocity)
            + self.command.kp * (self.command.position - self.reported_position)
            + self.command.torque
        )
        limit = min(
            abs(float(self.params[PARAM_LIMIT_TORQUE])),
            max(abs(item) for item in self.limits.torque),
        )
        self.applied_torque = min(max(value, -limit), limit)
        return self.applied_torque

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

    def _read_parameter(
        self, motor: RobstrideMotor, frame: CanFrame, host_id: int
    ) -> list[CanFrame]:
        if len(frame.data) < 2:
            return []
        index = _le16(frame.data, 0)
        kind = PARAM_TYPES.get(index, "f")
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
        kind = PARAM_TYPES.get(index, "f")
        size = struct.calcsize(kind)
        if len(frame.data) < 4 + size or index in {
            PARAM_MECHANICAL_POSITION,
            PARAM_MECHANICAL_VELOCITY,
            PARAM_FILTERED_CURRENT,
        }:
            return [motor.feedback_frame(CommType.MOTOR_FEEDBACK, host_id)]
        value = struct.unpack_from("<" + kind, frame.data, 4)[0]
        motor.params[index] = value
        return [
            CanFrame(
                build_can_id(CommType.PARAM_WRITE, motor.spec.motor_id, host_id),
                frame.data.ljust(8, b"\0"),
            )
        ]


def _stable_device_id(spec: MotorSpec) -> int:
    value = 0x5253000000000000 | (spec.motor_id << 32)
    for byte in spec.kind.name.encode("ascii"):
        value = ((value << 5) - value + byte) & 0xFFFFFFFFFFFFFFFF
    return value


def _be16(data: bytes, offset: int) -> int:
    return struct.unpack_from(">H", data, offset)[0]


def _le16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def _pack_be16(value: int) -> bytes:
    return struct.pack(">H", min(max(int(value), 0), 0xFFFF))
