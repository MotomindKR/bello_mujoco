from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
import threading

import grpc
import pytest

from bello_mujoco.hardware import hand_pb2 as pb
from bello_mujoco.hardware.hand_server import (
    HandController,
    HandControlClient,
    HandServerConfig,
    hand_service_handler,
    run_hand_server,
)
from bello_mujoco.hardware.pico_grpc import PicoGrpcConfig
from bello_mujoco.hardware.xrobot_hands import (
    XRobotHandsConfig,
    XRobotHandsFrame,
    XRobotHandTarget,
)


class Clock:
    now = 10.0

    def __call__(self):
        return self.now


class Transport:
    config = SimpleNamespace(feedback_timeout_seconds=0.5)

    def __init__(self, clock):
        self.clock = clock
        self.writes = []
        self.probes = []
        self.measured = {"left": (200,) * 6, "right": (150,) * 6}
        self.stamps = {side: clock() for side in self.measured}
        self.fault_side = None
        self.stale_side = None
        self.probe_delay = 0

    def feedback(self, side):
        return self.measured[side], self.stamps[side]

    def poll_feedback(self, side):
        if side == self.fault_side:
            raise OSError("CAN failure")
        if side != self.stale_side:
            self.stamps[side] = self.clock()

    def refresh_positions(self, side):
        self.probes.append(side)
        self.clock.now += self.probe_delay
        self.stamps[side] = self.clock()
        return self.measured[side]

    def set_positions(self, side, positions):
        self.writes.append((side, positions))


def setup_controller(**changes):
    clock = Clock()
    config = XRobotHandsConfig(
        xrobot=PicoGrpcConfig(endpoint="127.0.0.1:1"),
        arm=True,
        calibration_confirmed=True,
        max_raw_rate_per_second=100,
        smoothing_time_constant_ms=0,
    )
    config = replace(config, **changes)
    transport = Transport(clock)
    controller = HandController(config, HandServerConfig(), transport, clock=clock)
    return controller, transport, clock


def command(lease, source=pb.APPLICATION, **kwargs):
    return pb.Command(side=lease.side, token=lease.token, source=source, **kwargs)


def frame(clock, *, left=0.0, right=0.0, track_id="epoch"):
    return XRobotHandsFrame(
        sequence=1,
        source_timestamp_ns=1,
        source_age_ms=0,
        received_at=clock(),
        track_id=track_id,
        left=None if left is None else XRobotHandTarget("left", (left,) * 6),
        right=None if right is None else XRobotHandTarget("right", (right,) * 6),
    )


def test_startup_only_polls_and_reports_measured_positions():
    controller, transport, _ = setup_controller()
    controller.tick()
    assert transport.writes == transport.probes == []
    status = controller.status()
    assert [hand.source for hand in status.hands] == [pb.HOLD, pb.HOLD]
    assert list(status.hands[0].measured_positions) == [200] * 6
    assert not status.grasp_feedback_supported
    assert not status.force_control_supported


def test_ownership_conflicts_and_invalid_tokens_do_not_change_control():
    controller, _, clock = setup_controller(right_enabled=False)
    lease = controller.acquire(pb.Hand(side=pb.LEFT))
    with pytest.raises(RuntimeError, match="owner"):
        controller.acquire(pb.Hand(side=pb.LEFT))
    with pytest.raises(ValueError, match="enabled"):
        controller.acquire(pb.Hand(side=pb.RIGHT))
    for token in ("", "wrong", "\u00e9"):
        with pytest.raises(PermissionError):
            controller.command(pb.Command(side=pb.LEFT, token=token, source=pb.HOLD))
    clock.now += 1.1
    with pytest.raises(PermissionError):
        controller.keep_alive(lease)
    new_lease = controller.acquire(pb.Hand(side=pb.LEFT))
    assert new_lease.token != lease.token


def test_application_target_ignores_tracking_and_reseeds_with_bounded_rate():
    controller, transport, clock = setup_controller()
    lease = controller.acquire(pb.Hand(side=pb.LEFT))
    controller.command(command(lease, close=True, max_raw_rate_per_second=50))
    controller.publish_tracking(frame(clock, left=0))
    controller.tick()
    assert transport.writes == [("left", (190,) * 6)]
    controller.publish_tracking(None)
    clock.now += 0.2
    controller.tick()
    assert transport.writes[-1] == ("left", (180,) * 6)
    assert transport.probes == ["left"]
    status = controller.status().hands[0]
    assert status.source == pb.APPLICATION
    assert list(status.target) == [1] * 6
    assert list(status.measured_positions) == [200] * 6
    assert list(status.commanded_positions) == [180] * 6


def test_expiry_suspends_without_opening_or_restoring_tracking():
    controller, transport, clock = setup_controller()
    lease = controller.acquire(pb.Hand(side=pb.LEFT))
    controller.command(command(lease, close=True))
    controller.tick()
    clock.now += 1.1
    controller.publish_tracking(frame(clock))
    controller.tick()
    assert len(transport.writes) == 1
    state = controller.status().hands[0]
    assert state.source == pb.HOLD and not state.owned
    assert "expired" in state.suspension_reason
    with pytest.raises(PermissionError):
        controller.command(command(lease, open=True))


def test_keepalive_extends_lease_but_does_not_change_target():
    controller, transport, clock = setup_controller()
    lease = controller.acquire(pb.Hand(side=pb.LEFT))
    controller.command(command(lease, positions=pb.Positions(values=[0.5] * 6)))
    clock.now += 0.8
    controller.keep_alive(lease)
    clock.now += 0.8
    controller.tick()
    assert len(transport.writes) == 1
    assert list(controller.status().hands[0].target) == [0.5] * 6
    controller.release(lease)
    controller.tick()
    assert len(transport.writes) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"open": False},
        {"positions": pb.Positions(values=[0.5])},
        {"positions": pb.Positions(values=[float("nan")] * 6)},
        {"positions": pb.Positions(values=[1.1] * 6)},
        {"open": True, "max_raw_rate_per_second": 101},
        {"open": True, "max_raw_rate_per_second": 0},
        {"open": True, "max_raw_rate_per_second": float("inf")},
    ],
)
def test_invalid_commands_do_not_mutate_state_or_renew_lease(kwargs):
    controller, transport, clock = setup_controller()
    lease = controller.acquire(pb.Hand(side=pb.LEFT))
    clock.now += 0.5
    before = controller.status()
    with pytest.raises(ValueError):
        controller.command(command(lease, **kwargs))
    assert controller.status() == before
    controller.tick()
    assert transport.writes == []


def test_tracking_needs_new_frame_after_probe_and_recovers_from_missing_side():
    controller, transport, clock = setup_controller()
    lease = controller.acquire(pb.Hand(side=pb.LEFT))
    controller.command(command(lease, pb.TRACKING))
    controller.publish_tracking(frame(clock, left=1))
    controller.tick()
    assert not transport.writes
    clock.now += 0.1
    controller.publish_tracking(frame(clock, left=1))
    controller.tick()
    assert transport.writes == [("left", (180,) * 6)]
    controller.publish_tracking(frame(clock, left=None))
    clock.now += 0.1
    controller.publish_tracking(frame(clock, left=1))
    controller.tick()
    assert transport.probes == ["left", "left"]
    assert len(transport.writes) == 1
    clock.now += 0.1
    controller.publish_tracking(frame(clock, left=1))
    controller.tick()
    assert transport.writes[-1] == ("left", (180,) * 6)


def test_tracking_staleness_and_epoch_changes_require_reseed():
    controller, transport, clock = setup_controller()
    lease = controller.acquire(pb.Hand(side=pb.LEFT))
    controller.command(command(lease, pb.TRACKING))
    controller.publish_tracking(frame(clock))
    controller.tick()
    clock.now += 0.1
    controller.publish_tracking(frame(clock))
    controller.tick()
    clock.now += 0.6
    controller.tick()
    assert len(transport.writes) == 1
    assert "stale" in controller.status().hands[0].suspension_reason
    controller.publish_tracking(frame(clock, track_id="new"))
    controller.tick()
    assert transport.probes == ["left", "left"]
    assert len(transport.writes) == 1


def test_expiry_during_probe_prevents_write():
    controller, transport, _ = setup_controller()
    lease = controller.acquire(pb.Hand(side=pb.LEFT))
    controller.command(command(lease, close=True))
    transport.probe_delay = 1.1
    controller.tick()
    assert not transport.writes
    assert controller.status().hands[0].source == pb.HOLD


def test_can_cutoff_recovers_with_reseed_and_other_hand_continues():
    controller, transport, clock = setup_controller()
    left = controller.acquire(pb.Hand(side=pb.LEFT))
    right = controller.acquire(pb.Hand(side=pb.RIGHT))
    controller.command(command(left, close=True))
    controller.command(command(right, open=True))
    transport.fault_side = "left"
    controller.tick()
    assert transport.writes == [("right", (170,) * 6)]
    assert controller.status().hands[0].transport_fault == "CAN failure"
    assert not controller.status().hands[0].feedback_fresh
    assert controller.status().hands[0].source == pb.APPLICATION
    transport.fault_side = None
    transport.measured["left"] = (80,) * 6
    clock.now += 0.2
    controller.tick()
    assert ("left", (60,) * 6) in transport.writes
    assert not controller.status().hands[0].transport_fault
    assert transport.probes.count("left") == 1


def test_can_recovery_after_lease_expiry_only_clears_diagnostics():
    controller, transport, clock = setup_controller()
    lease = controller.acquire(pb.Hand(side=pb.LEFT))
    controller.command(command(lease, close=True))
    transport.fault_side = "left"
    controller.tick()
    clock.now += 1.1
    transport.fault_side = None
    controller.tick()
    assert not transport.writes
    assert not controller.status().hands[0].transport_fault
    assert controller.status().hands[0].source == pb.HOLD
    assert not controller.status().hands[0].owned


def test_tracking_can_recovery_requires_observation_after_reseed():
    controller, transport, clock = setup_controller()
    lease = controller.acquire(pb.Hand(side=pb.LEFT))
    controller.command(command(lease, pb.TRACKING))
    controller.publish_tracking(frame(clock, left=1))
    controller.tick()
    clock.now += 0.1
    controller.publish_tracking(frame(clock, left=1))
    controller.tick()
    assert len(transport.writes) == 1
    transport.fault_side = "left"
    clock.now += 0.1
    controller.tick()
    transport.fault_side = None
    transport.measured["left"] = (80,) * 6
    clock.now += 0.1
    controller.publish_tracking(frame(clock, left=1))
    controller.tick()
    assert len(transport.writes) == 1
    clock.now += 0.1
    controller.publish_tracking(frame(clock, left=1))
    controller.tick()
    assert transport.writes[-1] == ("left", (60,) * 6)
    assert transport.probes == ["left", "left"]


def test_missing_replies_suspend_then_reseed_without_accumulating_slew():
    controller, transport, clock = setup_controller()
    lease = controller.acquire(pb.Hand(side=pb.LEFT))
    controller.command(command(lease, close=True))
    controller.tick()
    transport.stale_side = "left"
    clock.now += 0.6
    controller.keep_alive(lease)
    controller.tick()
    assert len(transport.writes) == 1
    assert not controller.status().hands[0].feedback_fresh
    assert controller.status().hands[0].source == pb.APPLICATION
    transport.stale_side = None
    transport.measured["left"] = (80,) * 6
    clock.now += 0.1
    controller.tick()
    assert transport.writes[-1] == ("left", (60,) * 6)
    assert transport.probes == ["left", "left"]


def test_protocol_error_still_requires_explicit_source_selection():
    controller, transport, _ = setup_controller()
    lease = controller.acquire(pb.Hand(side=pb.LEFT))
    controller.command(command(lease, close=True))
    original = transport.poll_feedback

    def malformed(side):
        if side == "left":
            raise ValueError("malformed response")
        original(side)

    transport.poll_feedback = malformed
    controller.tick()
    assert controller.status().hands[0].source == pb.HOLD
    assert controller.status().hands[0].transport_fault
    transport.poll_feedback = original
    controller.tick()
    assert not transport.writes
    assert not controller.status().hands[0].transport_fault
    controller.command(command(lease, close=True))
    controller.tick()
    assert len(transport.writes) == 1


def test_switching_from_application_to_tracking_cannot_use_old_target():
    controller, transport, clock = setup_controller()
    lease = controller.acquire(pb.Hand(side=pb.LEFT))
    controller.command(command(lease, close=True))
    controller.tick()
    controller.publish_tracking(frame(clock))
    controller.command(command(lease, pb.TRACKING))
    controller.tick()
    assert len(transport.writes) == 1
    clock.now += 0.1
    controller.publish_tracking(frame(clock))
    controller.tick()
    assert transport.writes[-1] == ("left", (220,) * 6)


def test_grpc_round_trip_and_error_codes():
    controller, transport, _ = setup_controller()
    with ThreadPoolExecutor(max_workers=2) as executor:
        server = grpc.server(executor)
        server.add_generic_rpc_handlers((hand_service_handler(controller),))
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        try:
            with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                client = HandControlClient(channel)
                assert len(client.status().hands) == 2
                lease = client.acquire(pb.LEFT)
                with pytest.raises(grpc.RpcError) as error:
                    client.acquire(pb.LEFT)
                assert error.value.code() == grpc.StatusCode.FAILED_PRECONDITION
                with pytest.raises(grpc.RpcError) as error:
                    client.command(command(lease))
                assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
                with pytest.raises(grpc.RpcError) as error:
                    client.release(pb.Lease(side=pb.LEFT, token="wrong"))
                assert error.value.code() == grpc.StatusCode.PERMISSION_DENIED
                client.command(command(lease, close=True))
                assert transport.writes == []  # RPC only changes intent.
                controller.tick()
                assert len(transport.writes) == 1
                client.keep_alive(lease)
                client.release(lease)
                assert not client.status().hands[0].owned
        finally:
            server.stop(0).wait()


def test_server_runs_and_closes_transport_without_tracking():
    controller, transport, _ = setup_controller()
    events = []
    stop = threading.Event()

    class Context:
        def __enter__(self):
            events.append("CAN opened")
            return transport

        def __exit__(self, *_):
            events.append("CAN closed")

    def unavailable(_):
        events.append("tracking attempted")
        raise ConnectionError("bridge offline")

    run_hand_server(
        replace(controller.config, max_duration_seconds=0.03, control_rate_hz=100),
        HandServerConfig(listen="127.0.0.1:0"),
        lambda _: Context(),
        stop_event=stop,
        stream_factory=unavailable,
    )
    assert events == ["CAN opened", "tracking attempted", "CAN closed"]
    assert transport.writes == []
    assert stop.is_set()
