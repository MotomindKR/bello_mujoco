from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import struct

import pytest

from bello_mujoco.ebimu_serial import EbimuSerialEmulator, STANDARD_GRAVITY
from bello_mujoco.imu import ImuSample


@contextmanager
def open_emulator():
    host_fd, device_fd = os.openpty()
    device = Path(os.ttyname(device_fd))
    os.close(device_fd)
    flags = fcntl.fcntl(host_fd, fcntl.F_GETFL)
    fcntl.fcntl(host_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    try:
        with EbimuSerialEmulator(device) as emulator:
            yield emulator, host_fd
    finally:
        os.close(host_fd)


def sample(
    simulation_time: float = 0.0,
    *,
    orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> ImuSample:
    return ImuSample(
        simulation_time=simulation_time,
        orientation_xyzw=orientation_xyzw,
        angular_velocity_xyz=(0.1, -0.2, 0.3),
        linear_acceleration_xyz=(0.0, 0.0, STANDARD_GRAVITY),
        local_velocity_xyz=(1.0, 2.0, 3.0),
        world_position_xyz=(4.0, 5.0, 6.0),
    )


def read_available(fd: int) -> bytes:
    chunks = []
    while True:
        try:
            chunks.append(os.read(fd, 4096))
        except BlockingIOError:
            return b"".join(chunks)


def send_command(
    emulator: EbimuSerialEmulator, host_fd: int, command: bytes, value: ImuSample
) -> bytes:
    read_available(host_fd)
    os.write(host_fd, command)
    emulator.update(value)
    return read_available(host_fd)


def test_default_ascii_stream_matches_motomind_policy_layout() -> None:
    with open_emulator() as (emulator, host_fd):
        emulator.update(sample())
        fields = read_available(host_fd).decode("ascii").strip()[1:].split(",")

    assert fields[:4] == ["0.0000", "0.0000", "0.0000", "1.0000"]
    assert [float(value) for value in fields[4:7]] == pytest.approx(
        [5.73, -11.46, 17.19], abs=0.01
    )
    assert [float(value) for value in fields[7:10]] == pytest.approx([0.0, 0.0, 0.0])
    assert fields[10] == "0"


def test_stream_rebases_after_simulation_time_resets() -> None:
    with open_emulator() as (emulator, host_fd):
        emulator.update(sample(10.0))
        assert read_available(host_fd)

        emulator.update(sample(0.0))
        reset_record = read_available(host_fd).decode("ascii")

    assert reset_record
    assert reset_record.strip().split(",")[-1] == "0"


def test_commands_reconfigure_stream_and_support_polling() -> None:
    value = sample(1.25)
    with open_emulator() as (emulator, host_fd):
        assert send_command(emulator, host_fd, b"<stop>", value) == b"<ok>"
        commands = b"<soc1><sof1><sog0><soa1><som1><sod2><sot1><sots1><sor0>"
        responses = send_command(emulator, host_fd, commands, value)
        assert responses == b"<ok>" * 9
        assert send_command(emulator, host_fd, b"<start>", value) == b"<ok>"

        os.write(host_fd, b"*")
        emulator.update(value)
        record = read_available(host_fd).decode("ascii")

    fields = record.strip()[1:].split(",")
    assert len(fields) == 3 + 3 + 3 + 3 + 1 + 1
    assert [float(entry) for entry in fields[:3]] == pytest.approx([0.0, 0.0, 0.0])
    assert [float(entry) for entry in fields[3:6]] == pytest.approx([0.0, 0.0, 1.0])
    assert fields[-2:] == ["25.0", "1250"]


def test_quaternion_serial_order_inverts_motomind_sensor_frame_conversion() -> None:
    half_sqrt = 2.0**-0.5
    value = sample(orientation_xyzw=(0.0, 0.0, half_sqrt, half_sqrt))
    with open_emulator() as (emulator, host_fd):
        emulator.update(value)
        fields = read_available(host_fd).decode("ascii").strip()[1:].split(",")

    assert [float(entry) for entry in fields[:4]] == pytest.approx(
        [-half_sqrt, 0.0, 0.0, half_sqrt], abs=1.0e-4
    )


def test_binary_mode_uses_sop_big_endian_fields_and_checksum() -> None:
    value = sample()
    with open_emulator() as (emulator, host_fd):
        send_command(
            emulator, host_fd, b"<stop><soc2><sog0><soa0><sots0><sor0><start>", value
        )
        read_available(host_fd)
        os.write(host_fd, b"*")
        emulator.update(value)
        payload = read_available(host_fd)

    assert payload[:2] == b"\x55\x55"
    assert len(payload) == 12
    assert struct.unpack(">4h", payload[2:10]) == (0, 0, 0, 10_000)
    assert struct.unpack(">H", payload[-2:])[0] == sum(payload[:-2]) & 0xFFFF


def test_invalid_command_returns_error_response() -> None:
    value = sample()
    with open_emulator() as (emulator, host_fd):
        response = send_command(emulator, host_fd, b"<sor1001>", value)

    assert response == b"<er>"
