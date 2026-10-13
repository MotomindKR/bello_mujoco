from __future__ import annotations

from collections.abc import Sequence
import socket

import pytest

from bello_mujoco.hardware.linker_hand import (
    ACTUATOR_COUNT,
    CAN_EFF_FLAG,
    CAN_ERR_FLAG,
    CAN_FILTER,
    CAN_FRAME,
    CAN_RTR_FLAG,
    CAN_SFF_MASK,
    EXACT_STANDARD_DATA_MASK,
    LEFT_CAN_ID,
    POSITION_OPCODE,
    RIGHT_CAN_ID,
    SPEED_OPCODE,
    LinkerHandAddress,
    LinkerHandCanEndpoint,
    LinkerHandPositionResponse,
    StandardCanFrame,
    build_position_query,
    build_position_write,
    build_query,
    build_write,
    parse_position_response,
    socketcan_filter_bytes,
)


class FakeRawCanSocket:
    def __init__(
        self,
        received: Sequence[bytes] = (),
        *,
        received_after_send: Sequence[bytes] = (),
        send_size: int | None = None,
        bind_error: Exception | None = None,
    ) -> None:
        self.received = list(received)
        self.received_after_send = list(received_after_send)
        self.send_size = send_size
        self.bind_error = bind_error
        self.socket_options: list[tuple[int, int, bytes]] = []
        self.blocking: bool | None = None
        self.bound_address: tuple[str] | None = None
        self.sent: list[bytes] = []
        self.close_count = 0

    def setsockopt(self, level: int, option: int, value: bytes) -> None:
        self.socket_options.append((level, option, value))

    def setblocking(self, flag: bool) -> None:
        self.blocking = flag

    def bind(self, address: tuple[str]) -> None:
        self.bound_address = address
        if self.bind_error is not None:
            raise self.bind_error

    def send(self, data: bytes) -> int:
        self.sent.append(data)
        self.received.extend(self.received_after_send)
        self.received_after_send.clear()
        return len(data) if self.send_size is None else self.send_size

    def recv(self, size: int) -> bytes:
        if not self.received:
            raise BlockingIOError
        value = self.received.pop(0)
        assert size == CAN_FRAME.size
        return value

    def close(self) -> None:
        self.close_count += 1


def test_position_queries_on_shared_id_are_not_malformed_feedback():
    address = LinkerHandAddress("O6", "left")
    fake = FakeRawCanSocket(
        received=[
            build_position_query(address).pack(),
            build_position_write(address, (1, 2, 3, 4, 5, 6)).pack(),
        ]
    )
    with LinkerHandCanEndpoint("can2", address, can_socket=fake) as endpoint:
        assert endpoint.receive_positions().positions == (1, 2, 3, 4, 5, 6)


@pytest.mark.parametrize(
    ("model", "side", "command_id", "response_ids"),
    [
        ("O6", "right", RIGHT_CAN_ID, (RIGHT_CAN_ID,)),
        ("O6", "left", LEFT_CAN_ID, (LEFT_CAN_ID,)),
        ("L6", "right", RIGHT_CAN_ID, (RIGHT_CAN_ID, RIGHT_CAN_ID + 8)),
        ("L6", "left", LEFT_CAN_ID, (LEFT_CAN_ID, LEFT_CAN_ID + 8)),
    ],
)
def test_linker_hand_addressing(
    model: str,
    side: str,
    command_id: int,
    response_ids: tuple[int, ...],
) -> None:
    address = LinkerHandAddress(model, side)

    assert address.command_id == command_id
    assert address.response_ids == response_ids


@pytest.mark.parametrize(
    ("model", "side"),
    [("eHand-6", "right"), ("O6", "RIGHT"), ("L7", "left")],
)
def test_linker_hand_address_rejects_unsupported_variants(
    model: str,
    side: str,
) -> None:
    with pytest.raises(ValueError):
        LinkerHandAddress(model, side)


def test_standard_can_frame_round_trip_has_no_extended_flag() -> None:
    frame = StandardCanFrame(RIGHT_CAN_ID, bytes((POSITION_OPCODE, *range(6))))

    packed = frame.pack()
    raw_id, dlc, payload = CAN_FRAME.unpack(packed)

    assert len(packed) == CAN_FRAME.size == 16
    assert raw_id == RIGHT_CAN_ID
    assert dlc == 7
    assert payload == bytes((POSITION_OPCODE, *range(6), 0))
    assert StandardCanFrame.unpack(packed) == frame


@pytest.mark.parametrize(
    ("flag", "message"),
    [
        (CAN_EFF_FLAG, "11-bit standard"),
        (CAN_RTR_FLAG, "remote"),
        (CAN_ERR_FLAG, "error frame"),
    ],
)
def test_standard_can_frame_rejects_non_data_frame_flags(
    flag: int,
    message: str,
) -> None:
    raw = CAN_FRAME.pack(
        RIGHT_CAN_ID | flag,
        1,
        b"\x01".ljust(8, b"\0"),
    )

    with pytest.raises(ValueError, match=message):
        StandardCanFrame.unpack(raw)


def test_standard_can_frame_rejects_invalid_sizes_and_identifiers() -> None:
    with pytest.raises(ValueError, match="CAN id"):
        StandardCanFrame(0x800, b"")
    with pytest.raises(ValueError, match="cannot exceed"):
        StandardCanFrame(RIGHT_CAN_ID, bytes(9))
    with pytest.raises(TypeError, match="must be bytes"):
        StandardCanFrame(RIGHT_CAN_ID, bytearray(1))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="16-byte"):
        StandardCanFrame.unpack(bytes(15))
    with pytest.raises(ValueError, match="DLC"):
        StandardCanFrame.unpack(CAN_FRAME.pack(RIGHT_CAN_ID, 9, bytes(8)))


def test_position_query_is_one_byte_and_position_write_is_seven_bytes() -> None:
    address = LinkerHandAddress("O6", "right")

    query = build_position_query(address)
    command = build_position_write(address, (10, 20, 30, 40, 50, 60))

    assert query == StandardCanFrame(RIGHT_CAN_ID, b"\x01")
    assert len(query.data) == 1
    assert command == StandardCanFrame(
        RIGHT_CAN_ID,
        bytes((POSITION_OPCODE, 10, 20, 30, 40, 50, 60)),
    )
    assert len(command.data) == 7


def test_generic_six_actuator_codec_accepts_only_known_registers() -> None:
    address = LinkerHandAddress("L6", "left")

    assert build_query(address, SPEED_OPCODE).data == b"\x05"
    assert build_write(address, SPEED_OPCODE, (1, 2, 3, 4, 5, 6)).data == bytes(
        (SPEED_OPCODE, 1, 2, 3, 4, 5, 6)
    )

    with pytest.raises(ValueError, match="unsupported"):
        build_query(address, 0x33)
    with pytest.raises(TypeError, match="opcode"):
        build_query(address, True)


@pytest.mark.parametrize(
    "values",
    [
        (),
        (0, 1, 2, 3, 4),
        (0, 1, 2, 3, 4, 5, 6),
        (0, 1, 2, 3, 4, 256),
        (0, 1, 2, 3, 4, -1),
    ],
)
def test_position_write_rejects_wrong_length_or_range(values: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        build_position_write(LinkerHandAddress("O6", "right"), values)


@pytest.mark.parametrize("invalid", [1.0, True, "1", None])
def test_position_write_rejects_non_integer_values(invalid: object) -> None:
    values = [0] * ACTUATOR_COUNT
    values[-1] = invalid  # type: ignore[list-item]

    with pytest.raises(TypeError, match="must be an integer"):
        build_position_write(LinkerHandAddress("O6", "right"), values)


def test_position_response_parser_accepts_o6_primary_id() -> None:
    address = LinkerHandAddress("O6", "left")
    frame = StandardCanFrame(LEFT_CAN_ID, bytes((POSITION_OPCODE, 6, 5, 4, 3, 2, 1)))

    response = parse_position_response(address, frame)

    assert response == LinkerHandPositionResponse(
        LEFT_CAN_ID,
        (6, 5, 4, 3, 2, 1),
    )


def test_position_response_parser_accepts_l6_offset_id_only_for_l6() -> None:
    l6 = LinkerHandAddress("L6", "right")
    frame = StandardCanFrame(
        RIGHT_CAN_ID + 8,
        bytes((POSITION_OPCODE, 1, 2, 3, 4, 5, 6)),
    )

    assert parse_position_response(l6, frame).positions == (1, 2, 3, 4, 5, 6)

    with pytest.raises(ValueError, match="response id"):
        parse_position_response(LinkerHandAddress("O6", "right"), frame)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\x01",
        bytes((POSITION_OPCODE, 1, 2, 3, 4, 5)),
        bytes((POSITION_OPCODE, 1, 2, 3, 4, 5, 6, 7)),
        bytes((SPEED_OPCODE, 1, 2, 3, 4, 5, 6)),
    ],
)
def test_position_response_parser_rejects_bad_payload(data: bytes) -> None:
    frame = StandardCanFrame(RIGHT_CAN_ID, data)

    with pytest.raises(ValueError):
        parse_position_response(LinkerHandAddress("O6", "right"), frame)


def test_socketcan_filters_match_only_exact_standard_data_ids() -> None:
    address = LinkerHandAddress("L6", "left")
    packed = socketcan_filter_bytes(address)

    assert EXACT_STANDARD_DATA_MASK == CAN_SFF_MASK | CAN_EFF_FLAG | CAN_RTR_FLAG
    assert EXACT_STANDARD_DATA_MASK & CAN_ERR_FLAG == 0
    assert len(packed) == 2 * CAN_FILTER.size
    filters = tuple(
        CAN_FILTER.unpack_from(packed, offset)
        for offset in range(0, len(packed), CAN_FILTER.size)
    )
    assert filters == (
        (LEFT_CAN_ID, EXACT_STANDARD_DATA_MASK),
        (LEFT_CAN_ID + 8, EXACT_STANDARD_DATA_MASK),
    )


def test_endpoint_configures_only_socket_and_sends_position_frames() -> None:
    fake = FakeRawCanSocket()
    address = LinkerHandAddress("O6", "right")
    endpoint = LinkerHandCanEndpoint("can1", address, can_socket=fake)

    assert fake.socket_options == [
        (
            getattr(socket, "SOL_CAN_RAW", 101),
            getattr(socket, "CAN_RAW_FILTER", 1),
            socketcan_filter_bytes(address),
        )
    ]
    assert fake.blocking is False
    assert fake.bound_address == ("can1",)

    endpoint.query_positions()
    endpoint.write_positions((10, 20, 30, 40, 50, 60))

    assert tuple(StandardCanFrame.unpack(value) for value in fake.sent) == (
        build_position_query(address),
        build_position_write(address, (10, 20, 30, 40, 50, 60)),
    )
    endpoint.close()
    endpoint.close()
    assert fake.close_count == 1
    assert endpoint.closed


def test_endpoint_closes_socket_if_initialization_fails() -> None:
    fake = FakeRawCanSocket(bind_error=OSError("bind failed"))

    with pytest.raises(OSError, match="bind failed"):
        LinkerHandCanEndpoint(
            "can1",
            LinkerHandAddress("O6", "right"),
            can_socket=fake,
        )

    assert fake.close_count == 1


def test_endpoint_rejects_short_socketcan_write() -> None:
    fake = FakeRawCanSocket(send_size=CAN_FRAME.size - 1)
    endpoint = LinkerHandCanEndpoint(
        "can1",
        LinkerHandAddress("O6", "right"),
        can_socket=fake,
    )

    with pytest.raises(OSError, match="short SocketCAN write"):
        endpoint.query_positions()


def test_endpoint_position_probe_is_read_only_and_returns_response() -> None:
    address = LinkerHandAddress("L6", "left")
    response_frame = StandardCanFrame(
        LEFT_CAN_ID + 8,
        bytes((POSITION_OPCODE, 11, 22, 33, 44, 55, 66)),
    )
    fake = FakeRawCanSocket(received_after_send=(response_frame.pack(),))
    endpoint = LinkerHandCanEndpoint("can1", address, can_socket=fake)

    response = endpoint.probe_positions(timeout_s=0.0)

    assert response.positions == (11, 22, 33, 44, 55, 66)
    assert [StandardCanFrame.unpack(value) for value in fake.sent] == [
        build_position_query(address)
    ]


def test_endpoint_position_probe_ignores_other_register_responses() -> None:
    address = LinkerHandAddress("O6", "right")
    speed = StandardCanFrame(
        RIGHT_CAN_ID,
        bytes((SPEED_OPCODE, 1, 2, 3, 4, 5, 6)),
    )
    position = StandardCanFrame(
        RIGHT_CAN_ID,
        bytes((POSITION_OPCODE, 6, 5, 4, 3, 2, 1)),
    )
    fake = FakeRawCanSocket(received_after_send=(speed.pack(), position.pack()))
    endpoint = LinkerHandCanEndpoint("can1", address, can_socket=fake)

    assert endpoint.probe_positions(timeout_s=0.0).positions == (6, 5, 4, 3, 2, 1)


def test_endpoint_position_probe_times_out_after_read_only_query() -> None:
    address = LinkerHandAddress("O6", "right")
    fake = FakeRawCanSocket()
    endpoint = LinkerHandCanEndpoint("can1", address, can_socket=fake)

    with pytest.raises(TimeoutError, match="no Linker Hand position response"):
        endpoint.probe_positions(timeout_s=0.0)

    assert [StandardCanFrame.unpack(value) for value in fake.sent] == [
        build_position_query(address)
    ]


def test_endpoint_position_probe_discards_queued_feedback_before_query() -> None:
    address = LinkerHandAddress("O6", "right")
    stale = StandardCanFrame(
        RIGHT_CAN_ID,
        bytes((POSITION_OPCODE, 1, 1, 1, 1, 1, 1)),
    )
    fresh = StandardCanFrame(
        RIGHT_CAN_ID,
        bytes((POSITION_OPCODE, 2, 2, 2, 2, 2, 2)),
    )
    fake = FakeRawCanSocket(
        (stale.pack(),),
        received_after_send=(fresh.pack(),),
    )
    endpoint = LinkerHandCanEndpoint("can1", address, can_socket=fake)

    response = endpoint.probe_positions(timeout_s=0.0)

    assert response.positions == (2, 2, 2, 2, 2, 2)


def test_endpoint_rejects_unfiltered_response_from_injected_socket() -> None:
    fake = FakeRawCanSocket(
        (
            StandardCanFrame(
                LEFT_CAN_ID,
                bytes((POSITION_OPCODE, 1, 2, 3, 4, 5, 6)),
            ).pack(),
        )
    )
    endpoint = LinkerHandCanEndpoint(
        "can1",
        LinkerHandAddress("O6", "right"),
        can_socket=fake,
    )

    with pytest.raises(ValueError, match="unexpected Linker Hand CAN id"):
        endpoint.receive_frame()


def test_closed_endpoint_rejects_io() -> None:
    endpoint = LinkerHandCanEndpoint(
        "can1",
        LinkerHandAddress("O6", "right"),
        can_socket=FakeRawCanSocket(),
    )
    endpoint.close()

    with pytest.raises(RuntimeError, match="closed"):
        endpoint.query_positions()
    with pytest.raises(RuntimeError, match="closed"):
        endpoint.receive_frame()
