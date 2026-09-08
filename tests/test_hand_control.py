from __future__ import annotations

from dataclasses import replace

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


def test_transport_probes_both_hands_before_exposing_positions() -> None:
    clock = FakeClock()
    transport, endpoints = make_transport(LinkerHandsCanConfig(), clock)

    with transport:
        assert set(endpoints) == {"left", "right"}
        assert all(endpoint.probes == 1 for endpoint in endpoints.values())
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
        {"interface": ""},
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
