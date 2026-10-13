from __future__ import annotations

from dataclasses import replace
import errno

import pytest

from bello_mujoco.hardware.hand_control import (
    LinkerHandsCanConfig,
    LinkerHandsCanTransport,
)
from bello_mujoco.hardware.linker_hand import LinkerHandPositionResponse


class FakeClock:
    def __init__(self) -> None:
        self.now = 10.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeEndpoint:
    def __init__(self, interface, address) -> None:
        self.interface = interface
        self.address = address
        self.queries = 0
        self.writes: list[tuple[int, ...]] = []
        self.responses: list[LinkerHandPositionResponse] = []
        self.probe_position = (250, 200, 240, 230, 220, 210)
        self.probes = 0
        self.closed = False

    def probe_positions(self, *, timeout_s: float):
        del timeout_s
        self.probes += 1
        return LinkerHandPositionResponse(
            self.address.command_id,
            self.probe_position,
        )

    def query_positions(self) -> None:
        self.queries += 1

    def write_positions(self, positions) -> None:
        self.writes.append(tuple(positions))

    def receive_positions(self):
        return self.responses.pop(0) if self.responses else None

    def close(self) -> None:
        self.closed = True


def make_transport(config: LinkerHandsCanConfig, clock: FakeClock):
    endpoints = {}

    def factory(interface, address):
        endpoint = FakeEndpoint(interface, address)
        endpoints[address.side] = endpoint
        return endpoint

    transport = LinkerHandsCanTransport(
        config,
        endpoint_factory=factory,
        clock=clock,
        sleep=clock.sleep,
    )
    return transport, endpoints


@pytest.mark.parametrize("interfaces", (("can2", "can3"), ("can1", "can1")))
def test_transport_probes_both_hands_before_exposing_positions(interfaces) -> None:
    clock = FakeClock()
    config = LinkerHandsCanConfig(
        left_interface=interfaces[0], right_interface=interfaces[1]
    )
    transport, endpoints = make_transport(config, clock)

    with transport:
        assert set(endpoints) == {"left", "right"}
        assert all(endpoint.probes == 1 for endpoint in endpoints.values())
        assert endpoints["left"].interface == interfaces[0]
        assert endpoints["right"].interface == interfaces[1]
        assert transport.initial_positions["left"] == (250, 200, 240, 230, 220, 210)

    assert all(endpoint.closed for endpoint in endpoints.values())


def test_transport_writes_only_the_requested_side_and_staggers_frames() -> None:
    clock = FakeClock()
    config = replace(
        LinkerHandsCanConfig(),
        feedback_poll_interval_seconds=1.0,
        feedback_timeout_seconds=2.0,
        tx_spacing_seconds=0.001,
    )
    transport, endpoints = make_transport(config, clock)

    with transport:
        transport.set_positions("left", (1, 2, 3, 4, 5, 6))
        transport.set_positions("right", (6, 5, 4, 3, 2, 1))

    assert endpoints["left"].writes == [(1, 2, 3, 4, 5, 6)]
    assert endpoints["right"].writes == [(6, 5, 4, 3, 2, 1)]
    assert clock.sleeps == pytest.approx([0.001, 0.001, 0.001])


def test_transport_rejects_write_when_feedback_is_stale() -> None:
    clock = FakeClock()
    config = replace(
        LinkerHandsCanConfig(),
        right_enabled=False,
        feedback_poll_interval_seconds=0.1,
        feedback_timeout_seconds=0.2,
    )
    transport, endpoints = make_transport(config, clock)

    with transport:
        clock.now += 0.3
        with pytest.raises(TimeoutError, match="refresh and reseed"):
            transport.set_positions("left", (1, 2, 3, 4, 5, 6))

    assert endpoints["left"].probes == 1
    assert endpoints["left"].writes == []


def test_transport_polls_feedback_without_blocking_when_current() -> None:
    clock = FakeClock()
    config = replace(
        LinkerHandsCanConfig(),
        right_enabled=False,
        feedback_poll_interval_seconds=0.1,
        feedback_timeout_seconds=1.0,
    )
    transport, endpoints = make_transport(config, clock)

    with transport:
        clock.now += 0.15
        transport.set_positions("left", (1, 2, 3, 4, 5, 6))

    assert endpoints["left"].queries == 1


def test_transport_drains_queued_feedback_before_stale_check() -> None:
    clock = FakeClock()
    config = replace(
        LinkerHandsCanConfig(),
        right_enabled=False,
        feedback_poll_interval_seconds=0.2,
        feedback_timeout_seconds=0.5,
    )
    transport, endpoints = make_transport(config, clock)

    with transport:
        clock.now += 0.6
        endpoints["left"].responses.append(
            LinkerHandPositionResponse(
                endpoints["left"].address.command_id,
                (10, 20, 30, 40, 50, 60),
            )
        )
        transport.set_positions("left", (1, 2, 3, 4, 5, 6))

    assert endpoints["left"].writes == [(1, 2, 3, 4, 5, 6)]


def test_transport_refresh_returns_new_physical_position() -> None:
    clock = FakeClock()
    config = replace(LinkerHandsCanConfig(), right_enabled=False)
    transport, endpoints = make_transport(config, clock)

    with transport:
        endpoints["left"].probe_position = (10, 20, 30, 40, 50, 60)
        actual = transport.refresh_positions("left")

    assert actual == (10, 20, 30, 40, 50, 60)
    assert endpoints["left"].closed


@pytest.mark.parametrize(
    "changes",
    (
        {"left_interface": ""},
        {"right_interface": "can3\0"},
        {"left_enabled": False, "right_enabled": False},
        {"probe_timeout_seconds": 0.0},
        {
            "feedback_poll_interval_seconds": 0.6,
            "feedback_timeout_seconds": 0.5,
        },
        {"tx_spacing_seconds": -1.0},
    ),
)
def test_transport_config_rejects_unsafe_values(changes) -> None:
    with pytest.raises(ValueError):
        replace(LinkerHandsCanConfig(), **changes)


def test_suspended_hands_poll_and_report_feedback_without_position_writes():
    clock = FakeClock()
    transport, endpoints = make_transport(LinkerHandsCanConfig(), clock)
    with transport:
        initial, initial_at = transport.feedback("left")
        clock.now += 0.3
        transport.poll_feedback("left")
        assert endpoints["left"].queries == 1
        assert transport.feedback("left") == (initial, initial_at)
        endpoints["left"].responses.append(
            LinkerHandPositionResponse(0x28, (10, 20, 30, 40, 50, 60))
        )
        transport.poll_feedback("left")
        measured, stamp = transport.feedback("left")
        assert measured == (10, 20, 30, 40, 50, 60)
        assert stamp == clock()
        assert endpoints["left"].writes == []
        assert endpoints["right"].writes == []


@pytest.mark.parametrize(
    "operation", ("query_positions", "receive_positions", "write_positions")
)
def test_cutoff_closes_socket_and_rebinds_before_any_new_write(operation):
    clock = FakeClock()
    transport, endpoints = make_transport(LinkerHandsCanConfig(), clock)
    with transport:
        old = endpoints["left"]

        def disconnected(*_):
            raise OSError(errno.ENODEV, "CAN adapter removed")

        setattr(old, operation, disconnected)
        clock.now += 0.3
        with pytest.raises(OSError, match="adapter removed"):
            transport.set_positions("left", (1,) * 6)
        assert old.closed
        assert transport.feedback("left") == ((), 0.0)
        transport.poll_feedback("right")
        for _ in range(3):
            clock.now += 0.04
            with pytest.raises(ConnectionError, match="reconnect pending"):
                transport.poll_feedback("left")
            assert endpoints["left"] is old
        clock.now += 0.1
        transport.poll_feedback("left")
        new = endpoints["left"]
        assert new is not old and new.queries == 1 and new.writes == []
        with pytest.raises(TimeoutError, match="reseed"):
            transport.set_positions("left", (1,) * 6)
        assert new.writes == []
        clock.now += 0.3
        assert transport.refresh_positions("left") == (250, 200, 240, 230, 220, 210)
        transport.set_positions("left", (1,) * 6)
        assert endpoints["left"].writes == [(1,) * 6]


@pytest.mark.parametrize("failure", ("bind", "probe"))
def test_server_transport_starts_with_missing_hand_and_polls_until_recovery(failure):
    clock = FakeClock()
    endpoints = {}
    missing = True

    def factory(interface, address):
        if address.side == "left" and missing and failure == "bind":
            raise OSError(errno.ENODEV, "CAN adapter absent")
        endpoint = FakeEndpoint(interface, address)
        endpoints[address.side] = endpoint
        if address.side == "left" and missing:

            def timeout(**_):
                raise TimeoutError("hand unplugged")

            endpoint.probe_positions = timeout
        return endpoint

    transport = LinkerHandsCanTransport(
        LinkerHandsCanConfig(),
        endpoint_factory=factory,
        clock=clock,
        sleep=clock.sleep,
        require_initial_feedback=False,
    )
    with transport:
        assert transport.feedback("left") == ((), 0.0)
        assert len(transport.feedback("right")[0]) == 6
        missing = False
        clock.now += 0.3
        transport.poll_feedback("left")
        endpoint = endpoints["left"]
        endpoint.responses.append(LinkerHandPositionResponse(0x28, (90,) * 6))
        transport.poll_feedback("left")
        assert transport.feedback("left")[0] == (90,) * 6
        assert all(not endpoint.writes for endpoint in endpoints.values())


def test_strict_transport_still_rejects_missing_hand_and_cleans_up():
    clock = FakeClock()
    endpoint = FakeEndpoint("can2", None)

    def factory(interface, address):
        if address.side == "right":
            raise OSError("right adapter absent")
        endpoint.address = address
        return endpoint

    transport = LinkerHandsCanTransport(
        LinkerHandsCanConfig(),
        endpoint_factory=factory,
        clock=clock,
        sleep=clock.sleep,
    )
    with pytest.raises(OSError, match="adapter absent"):
        with transport:
            pass
    assert endpoint.closed
