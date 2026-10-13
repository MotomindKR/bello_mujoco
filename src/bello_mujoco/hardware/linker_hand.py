"""Classic SocketCAN primitives for Linker Hand O6 and L6 devices.

This wire protocol is specific to Linker Hand O6/L6 hardware.  It is not the
32-byte CAN-FD protocol used by the HITBOT eHand-6.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import socket
import struct
import time
from typing import Protocol


CAN_EFF_FLAG = 0x80000000
CAN_RTR_FLAG = 0x40000000
CAN_ERR_FLAG = 0x20000000
CAN_SFF_MASK = 0x7FF
CAN_FRAME = struct.Struct("=IB3x8s")
CAN_FILTER = struct.Struct("=II")

# Match the complete standard identifier while excluding extended and remote
# frames.  CAN_ERR_FLAG shares its value with CAN_INV_FILTER in ``can_filter``;
# putting that bit in this mask prevents normal hand responses from matching.
# Error frames are disabled separately by CAN_RAW_ERR_FILTER's default value.
EXACT_STANDARD_DATA_MASK = CAN_SFF_MASK | CAN_EFF_FLAG | CAN_RTR_FLAG

RIGHT_CAN_ID = 0x27
LEFT_CAN_ID = 0x28
L6_RESPONSE_ID_OFFSET = 0x08

POSITION_OPCODE = 0x01
TORQUE_LIMIT_OPCODE = 0x02
SPEED_OPCODE = 0x05
ACTUATOR_COUNT = 6


class RawCanSocket(Protocol):
    def setsockopt(self, level: int, option: int, value: bytes) -> None: ...

    def setblocking(self, flag: bool) -> None: ...

    def bind(self, address: tuple[str]) -> None: ...

    def send(self, data: bytes) -> int: ...

    def recv(self, size: int) -> bytes: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class LinkerHandAddress:
    """CAN addressing for one Linker Hand O6 or L6."""

    model: str
    side: str

    def __post_init__(self) -> None:
        if self.model not in {"O6", "L6"}:
            raise ValueError(f"unsupported Linker Hand model: {self.model!r}")
        if self.side not in {"left", "right"}:
            raise ValueError(f"unsupported hand side: {self.side!r}")

    @property
    def command_id(self) -> int:
        return LEFT_CAN_ID if self.side == "left" else RIGHT_CAN_ID

    @property
    def response_ids(self) -> tuple[int, ...]:
        if self.model == "L6":
            return (self.command_id, self.command_id + L6_RESPONSE_ID_OFFSET)
        return (self.command_id,)


@dataclass(frozen=True)
class StandardCanFrame:
    can_id: int
    data: bytes

    def __post_init__(self) -> None:
        if not 0 <= self.can_id <= CAN_SFF_MASK:
            raise ValueError(f"standard CAN id out of range: {self.can_id:#x}")
        if not isinstance(self.data, bytes):
            raise TypeError("CAN frame data must be bytes")
        if len(self.data) > 8:
            raise ValueError(
                f"classic CAN payload cannot exceed 8 bytes: {len(self.data)}"
            )

    def pack(self) -> bytes:
        return CAN_FRAME.pack(
            self.can_id,
            len(self.data),
            self.data.ljust(8, b"\0"),
        )

    @classmethod
    def unpack(cls, raw: bytes) -> StandardCanFrame:
        if len(raw) != CAN_FRAME.size:
            raise ValueError(
                f"expected {CAN_FRAME.size}-byte can_frame, got {len(raw)}"
            )
        raw_id, dlc, payload = CAN_FRAME.unpack(raw)
        if raw_id & CAN_EFF_FLAG:
            raise ValueError("Linker Hand requires an 11-bit standard CAN frame")
        if raw_id & CAN_RTR_FLAG:
            raise ValueError("Linker Hand does not use remote CAN frames")
        if raw_id & CAN_ERR_FLAG:
            raise ValueError("CAN error frame is not a Linker Hand response")
        if raw_id > CAN_SFF_MASK:
            raise ValueError(f"standard CAN id out of range: {raw_id:#x}")
        if dlc > 8:
            raise ValueError(f"invalid classic CAN DLC: {dlc}")
        return cls(raw_id, payload[:dlc])


@dataclass(frozen=True)
class LinkerHandPositionResponse:
    can_id: int
    positions: tuple[int, int, int, int, int, int]


def _actuator_values(values: Sequence[int]) -> tuple[int, ...]:
    if len(values) != ACTUATOR_COUNT:
        raise ValueError(
            f"expected {ACTUATOR_COUNT} actuator values, got {len(values)}"
        )
    checked: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"actuator value {index} must be an integer")
        if not 0 <= value <= 0xFF:
            raise ValueError(f"actuator value {index} out of range: {value}")
        checked.append(value)
    return tuple(checked)


def build_query(address: LinkerHandAddress, opcode: int) -> StandardCanFrame:
    """Build a read-only, one-byte query for a six-actuator register."""

    _validate_actuator_opcode(opcode)
    return StandardCanFrame(address.command_id, bytes((opcode,)))


def build_write(
    address: LinkerHandAddress,
    opcode: int,
    values: Sequence[int],
) -> StandardCanFrame:
    """Build an opcode plus six-value Classic CAN command."""

    _validate_actuator_opcode(opcode)
    checked = _actuator_values(values)
    return StandardCanFrame(address.command_id, bytes((opcode, *checked)))


def build_position_query(address: LinkerHandAddress) -> StandardCanFrame:
    return build_query(address, POSITION_OPCODE)


def build_position_write(
    address: LinkerHandAddress,
    positions: Sequence[int],
) -> StandardCanFrame:
    return build_write(address, POSITION_OPCODE, positions)


def parse_position_response(
    address: LinkerHandAddress,
    frame: StandardCanFrame,
) -> LinkerHandPositionResponse:
    if frame.can_id not in address.response_ids:
        expected = ", ".join(f"{can_id:#x}" for can_id in address.response_ids)
        raise ValueError(
            f"unexpected Linker Hand response id {frame.can_id:#x}; expected {expected}"
        )
    if len(frame.data) != 1 + ACTUATOR_COUNT:
        raise ValueError(
            "position response must contain opcode plus six actuator values"
        )
    if frame.data[0] != POSITION_OPCODE:
        raise ValueError(f"unexpected position response opcode: {frame.data[0]:#04x}")
    positions = tuple(frame.data[1:])
    return LinkerHandPositionResponse(frame.can_id, positions)  # type: ignore[arg-type]


def socketcan_filter_bytes(address: LinkerHandAddress) -> bytes:
    """Pack exact Linux CAN_RAW filters for this hand's response IDs."""

    return b"".join(
        CAN_FILTER.pack(can_id, EXACT_STANDARD_DATA_MASK)
        for can_id in address.response_ids
    )


def _validate_actuator_opcode(opcode: int) -> None:
    if isinstance(opcode, bool) or not isinstance(opcode, int):
        raise TypeError("Linker Hand opcode must be an integer")
    if opcode not in {POSITION_OPCODE, TORQUE_LIMIT_OPCODE, SPEED_OPCODE}:
        raise ValueError(f"unsupported six-actuator opcode: {opcode:#04x}")


class LinkerHandCanEndpoint:
    """Nonblocking raw-SocketCAN endpoint that never configures the interface."""

    def __init__(
        self,
        interface: str,
        address: LinkerHandAddress,
        *,
        can_socket: RawCanSocket | None = None,
        socket_factory: Callable[[int, int, int], RawCanSocket] | None = None,
    ) -> None:
        if not isinstance(interface, str) or not interface:
            raise ValueError("SocketCAN interface must be a non-empty string")
        if "\0" in interface:
            raise ValueError("SocketCAN interface cannot contain NUL")
        if can_socket is not None and socket_factory is not None:
            raise ValueError("provide either can_socket or socket_factory, not both")
        if can_socket is None:
            if not hasattr(socket, "AF_CAN") or not hasattr(socket, "CAN_RAW"):
                raise RuntimeError("SocketCAN is only available on Linux")
            factory = socket_factory or socket.socket
            can_socket = factory(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)

        self.interface = interface
        self.address = address
        self._socket = can_socket
        self._closed = False
        try:
            sol_can_raw = getattr(socket, "SOL_CAN_RAW", 101)
            can_raw_filter = getattr(socket, "CAN_RAW_FILTER", 1)
            self._socket.setsockopt(
                sol_can_raw,
                can_raw_filter,
                socketcan_filter_bytes(address),
            )
            self._socket.setblocking(False)
            self._socket.bind((interface,))
        except Exception:
            self._socket.close()
            self._closed = True
            raise

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        self._socket.close()
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Linker Hand CAN endpoint is closed")

    def _send(self, frame: StandardCanFrame) -> None:
        self._require_open()
        if frame.can_id != self.address.command_id:
            raise ValueError(
                f"command id {frame.can_id:#x} does not match endpoint id "
                f"{self.address.command_id:#x}"
            )
        packed = frame.pack()
        sent = self._socket.send(packed)
        if sent != CAN_FRAME.size:
            raise OSError(f"short SocketCAN write: {sent}/{CAN_FRAME.size}")

    def query_positions(self) -> None:
        """Transmit only a read-only position query."""

        self._send(build_position_query(self.address))

    def write_positions(self, positions: Sequence[int]) -> None:
        self._send(build_position_write(self.address, positions))

    def receive_frame(self) -> StandardCanFrame | None:
        self._require_open()
        try:
            raw = self._socket.recv(CAN_FRAME.size)
        except BlockingIOError:
            return None
        frame = StandardCanFrame.unpack(raw)
        if frame.can_id not in self.address.response_ids:
            raise ValueError(f"unexpected Linker Hand CAN id: {frame.can_id:#x}")
        return frame

    def receive_positions(self) -> LinkerHandPositionResponse | None:
        """Return the next position response, ignoring other valid hand opcodes."""

        while (frame := self.receive_frame()) is not None:
            if not frame.data:
                raise ValueError("Linker Hand response has an empty payload")
            if frame.data[0] == POSITION_OPCODE:
                if len(frame.data) == 1:
                    # Other local clients can query the same shared hand ID.
                    continue
                return parse_position_response(self.address, frame)
        return None

    def discard_pending_frames(self) -> int:
        """Drain responses that predate the next request."""

        discarded = 0
        while self.receive_frame() is not None:
            discarded += 1
        return discarded

    def probe_positions(
        self,
        *,
        timeout_s: float = 0.1,
        poll_interval_s: float = 0.001,
    ) -> LinkerHandPositionResponse:
        """Query positions without writing actuator state and await one response."""

        if timeout_s < 0.0:
            raise ValueError("position probe timeout cannot be negative")
        if poll_interval_s <= 0.0:
            raise ValueError("position probe poll interval must be positive")
        # A response carries no transaction sequence. Discard anything queued
        # before this query so old feedback cannot masquerade as a fresh probe.
        self.discard_pending_frames()
        self.query_positions()
        deadline = time.monotonic() + timeout_s
        while True:
            response = self.receive_positions()
            if response is not None:
                return response
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(
                    f"no Linker Hand position response within {timeout_s:.3f} seconds"
                )
            time.sleep(min(poll_interval_s, remaining))

    def __enter__(self) -> LinkerHandCanEndpoint:
        self._require_open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
