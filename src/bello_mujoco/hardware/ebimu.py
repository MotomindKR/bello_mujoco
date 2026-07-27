"""EBIMU-9DOFV6 serial protocol emulation backed by MuJoCo samples."""

from __future__ import annotations

from dataclasses import dataclass
import errno
import math
import os
from pathlib import Path
import struct
import termios
import tty

from bello_mujoco.imu import ImuSample

STANDARD_GRAVITY = 9.80665
_WORLD_MAGNETIC_FIELD_UT = (20.0, 0.0, -45.0)
_VERSION_RESPONSE = b"<imu9dofv610>"


@dataclass
class _OutputState:
    interval_ms: int = 5
    ascii_output: bool = True
    quaternion: bool = True
    gyro: bool = True
    accel_mode: int = 2
    magnetometer: bool = False
    distance_mode: int = 0
    temperature: bool = False
    timestamp: bool = True
    running: bool = True
    power_on_start: bool = True
    baud_code: int = 7


class EbimuSerialEmulator:
    """Emulate the host-visible UART behavior of an E2BOX EBIMU-9DOFV6."""

    def __init__(
        self,
        device: Path,
        *,
        output_rate_hz: float = 200.0,
        temperature_c: float = 25.0,
    ) -> None:
        interval_ms = round(1000.0 / output_rate_hz)
        if not 1 <= interval_ms <= 1000:
            raise ValueError("EBIMU output rate must be between 1 and 1000 Hz")
        self.device = Path(device)
        self.temperature_c = float(temperature_c)
        self._state = _OutputState(interval_ms=interval_ms)
        self._fd: int | None = None
        self._command = bytearray()
        self._in_command = False
        self._next_output_time: float | None = None
        self._last_simulation_time: float | None = None
        self._distance_origin: tuple[float, float, float] | None = None

    @property
    def is_open(self) -> bool:
        return self._fd is not None

    def open(self) -> None:
        if self._fd is not None:
            return
        try:
            fd = os.open(self.device, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        except OSError as error:
            raise RuntimeError(
                f"cannot open MuJoCo EBIMU peer {str(self.device)!r}; "
                "run ./scripts/setup_emulator.sh first"
            ) from error
        try:
            tty.setraw(fd, termios.TCSANOW)
        except Exception:
            os.close(fd)
            raise
        self._fd = fd

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> EbimuSerialEmulator:
        self.open()
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def update(self, sample: ImuSample) -> None:
        """Process host commands and emit the newest due sensor record."""

        if (
            self._last_simulation_time is not None
            and sample.simulation_time + 1.0e-12 < self._last_simulation_time
        ):
            self._next_output_time = sample.simulation_time
        self._last_simulation_time = sample.simulation_time
        if self._read_host_input(sample):
            return
        if not self._state.running or self._state.interval_ms == 0:
            return
        if self._next_output_time is None:
            self._next_output_time = sample.simulation_time
        if sample.simulation_time + 1.0e-12 < self._next_output_time:
            return
        self._write(self._encode_sample(sample))
        period = self._state.interval_ms / 1000.0
        self._next_output_time += period
        if self._next_output_time < sample.simulation_time - period:
            self._next_output_time = sample.simulation_time + period

    def _read_host_input(self, sample: ImuSample) -> bool:
        fd = self._require_fd()
        received = False
        while True:
            try:
                payload = os.read(fd, 4096)
            except BlockingIOError:
                return received
            except OSError as error:
                if error.errno == errno.EIO:
                    return received
                raise
            if not payload:
                return received
            received = True
            for byte in payload:
                self._consume_byte(byte, sample)

    def _consume_byte(self, byte: int, sample: ImuSample) -> None:
        if self._in_command:
            if byte == ord("<"):
                self._command.clear()
            elif byte == ord(">"):
                command = self._command.decode("ascii", errors="ignore")
                self._command.clear()
                self._in_command = False
                self._handle_command(command, sample)
            elif len(self._command) < 128:
                self._command.append(byte)
            else:
                self._command.clear()
                self._in_command = False
                self._write(b"<er>")
            return

        if byte == ord("<"):
            self._in_command = True
            self._command.clear()
        elif byte == ord("*") and self._state.interval_ms == 0:
            self._write(self._encode_sample(sample))
        elif byte == ord("/"):
            self._distance_origin = sample.world_position_xyz

    def _handle_command(self, command: str, sample: ImuSample) -> None:
        if command == "ver":
            self._write(_VERSION_RESPONSE)
            return
        if command == "start":
            self._state.running = True
            self._next_output_time = sample.simulation_time
            self._write(b"<ok>")
            return
        if command == "stop":
            self._state.running = False
            self._write(b"<ok>")
            return
        if command == "reset":
            self._state.running = self._state.power_on_start
            self._next_output_time = sample.simulation_time
            self._write(b"<ok>")
            return
        if command == "lf":
            self._state = _OutputState(
                interval_ms=10,
                ascii_output=True,
                quaternion=False,
                gyro=False,
                accel_mode=0,
                timestamp=False,
                baud_code=5,
            )
            self._next_output_time = sample.simulation_time
            self._write(b"<ok>")
            return
        if command == "posz":
            self._distance_origin = sample.world_position_xyz
            self._write(b"<ok>")
            return

        setters = (
            ("sb", range(1, 9), "baud_code"),
            ("soc", range(1, 3), "ascii_output"),
            ("sof", range(1, 3), "quaternion"),
            ("sog", range(0, 2), "gyro"),
            ("soa", range(0, 6), "accel_mode"),
            ("som", range(0, 2), "magnetometer"),
            ("sod", range(0, 3), "distance_mode"),
            ("sots", range(0, 2), "timestamp"),
            ("sot", range(0, 2), "temperature"),
            ("pons", range(0, 2), "power_on_start"),
        )
        for prefix, allowed, attribute in setters:
            if not command.startswith(prefix):
                continue
            raw_value = command[len(prefix) :]
            if not raw_value.isdigit() or int(raw_value) not in allowed:
                self._write(b"<er>")
                return
            value = int(raw_value)
            if attribute == "ascii_output":
                value = value == 1
            elif attribute == "quaternion":
                value = value == 2
            elif attribute in {
                "ascii_output",
                "gyro",
                "magnetometer",
                "temperature",
                "timestamp",
                "power_on_start",
            }:
                value = bool(value)
            setattr(self._state, attribute, value)
            self._write(b"<ok>")
            return

        if command.startswith("sor"):
            raw_value = command[3:]
            if not raw_value.isdigit() or not 0 <= int(raw_value) <= 1000:
                self._write(b"<er>")
                return
            self._state.interval_ms = int(raw_value)
            self._next_output_time = sample.simulation_time
            self._write(b"<ok>")
            return

        accepted_sensor_settings = {
            "sem": {"0", "1", "2"},
            "ssg": {"1", "2", "3", "4", "5"},
            "ssa": {"1", "2", "3", "4"},
            "lpfa": {str(value) for value in range(10)},
        }
        for prefix, allowed in accepted_sensor_settings.items():
            if command.startswith(prefix):
                self._write(b"<ok>" if command[len(prefix) :] in allowed else b"<er>")
                return
        self._write(b"<er>")

    def _encode_sample(self, sample: ImuSample) -> bytes:
        values = self._sample_fields(sample)
        if self._state.ascii_output:
            return (
                "*" + ",".join(text for text, _value, _scale in values) + "\r\n"
            ).encode("ascii")

        payload = bytearray((0x55, 0x55))
        for _text, value, scale in values:
            raw = int(round(value * scale)) & 0xFFFF
            payload.extend(struct.pack(">H", raw))
        payload.extend(struct.pack(">H", sum(payload) & 0xFFFF))
        return bytes(payload)

    def _sample_fields(self, sample: ImuSample) -> list[tuple[str, float, float]]:
        state = self._state
        orientation = _ebimu_attitude_quaternion(sample.orientation_xyzw)
        fields: list[tuple[str, float, float]] = []
        if state.quaternion:
            x, y, z, w = orientation
            fields.extend((f"{value:.4f}", value, 10_000.0) for value in (z, y, x, w))
        else:
            fields.extend(
                (f"{value:.2f}", value, 100.0)
                for value in _quaternion_to_euler_degrees(orientation)
            )

        if state.gyro:
            gyro_dps = tuple(
                math.degrees(value) for value in sample.angular_velocity_xyz
            )
            fields.extend((f"{value:.2f}", value, 10.0) for value in gyro_dps)

        if state.accel_mode:
            local_linear = _local_linear_acceleration(sample)
            if state.accel_mode == 1:
                vector = sample.linear_acceleration_xyz
                scale = STANDARD_GRAVITY
            elif state.accel_mode == 2:
                vector = local_linear
                scale = STANDARD_GRAVITY
            elif state.accel_mode == 3:
                vector = _local_to_world(sample.orientation_xyzw, local_linear)
                scale = STANDARD_GRAVITY
            elif state.accel_mode == 4:
                vector = sample.local_velocity_xyz
                scale = 1.0
            else:
                vector = _local_to_world(
                    sample.orientation_xyzw, sample.local_velocity_xyz
                )
                scale = 1.0
            values = tuple(value / scale for value in vector)
            fields.extend((f"{value:.3f}", value, 1000.0) for value in values)

        if state.magnetometer:
            magnetic = _world_to_local(
                sample.orientation_xyzw, _WORLD_MAGNETIC_FIELD_UT
            )
            fields.extend((f"{value:.1f}", value, 10.0) for value in magnetic)

        if state.distance_mode:
            if self._distance_origin is None:
                self._distance_origin = sample.world_position_xyz
            displacement = tuple(
                value - origin
                for value, origin in zip(
                    sample.world_position_xyz, self._distance_origin, strict=True
                )
            )
            if state.distance_mode == 1:
                displacement = _world_to_local(sample.orientation_xyzw, displacement)
            fields.extend((f"{value:.3f}", value, 1000.0) for value in displacement)

        if state.temperature:
            fields.append((f"{self.temperature_c:.1f}", self.temperature_c, 10.0))
        if state.timestamp:
            timestamp_ms = int(round(sample.simulation_time * 1000.0)) % 60_001
            fields.append((str(timestamp_ms), float(timestamp_ms), 1.0))
        return fields

    def _write(self, payload: bytes) -> None:
        try:
            os.write(self._require_fd(), payload)
        except BlockingIOError:
            pass
        except OSError as error:
            if error.errno != errno.EIO:
                raise

    def _require_fd(self) -> int:
        if self._fd is None:
            raise RuntimeError("EBIMU serial emulator is not open")
        return self._fd


def _ebimu_attitude_quaternion(
    sensor_quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x, y, z, w = sensor_quaternion
    values = x, -y, -z, w
    return tuple(0.0 if value == 0.0 else value for value in values)


def _local_linear_acceleration(sample: ImuSample) -> tuple[float, float, float]:
    local_gravity = _world_to_local(
        sample.orientation_xyzw, (0.0, 0.0, -STANDARD_GRAVITY)
    )
    return tuple(
        specific_force + gravity
        for specific_force, gravity in zip(
            sample.linear_acceleration_xyz, local_gravity, strict=True
        )
    )


def _local_to_world(
    quaternion: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    rotation = _rotation_matrix(quaternion)
    return tuple(
        sum(rotation[row][column] * vector[column] for column in range(3))
        for row in range(3)
    )


def _world_to_local(
    quaternion: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    rotation = _rotation_matrix(quaternion)
    return tuple(
        sum(rotation[row][column] * vector[row] for row in range(3))
        for column in range(3)
    )


def _rotation_matrix(
    quaternion: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], ...]:
    x, y, z, w = quaternion
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )


def _quaternion_to_euler_degrees(
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    x, y, z, w = quaternion
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sin_pitch)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)
