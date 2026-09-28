from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import queue
import time

import grpc
import numpy as np
import pytest

from bello_mujoco.hardware import _smplx_pb2 as pb
from bello_mujoco.hardware.pico_grpc import PicoGrpcConfig
from bello_mujoco.hardware.reference import ReferenceDiscontinuity, ReferenceUnavailable
from bello_mujoco.hardware.xrobot_hand_stream import (
    HAND_NAMES,
    XRobotHandsGrpcStream,
    parse_frame,
)


def config(**overrides):
    return replace(PicoGrpcConfig("localhost:1"), **overrides)


def hand():
    return pb.XRobotHand(
        joint_names=HAND_NAMES,
        positions=np.arange(78) / 1000,
        orientations=np.tile([1, 0, 0, 0], 26),
        position_valid=[True] * 26,
        orientation_valid=[True] * 26,
        frame="pico_native_y_up",
        active=True,
    )


def packet(sequence=1, **overrides):
    values = dict(
        sequence=sequence,
        source_timestamp_ns=sequence * 33_333_333,
        device_id="headset",
        upstream_connected=True,
        xrobot_left_hand=hand(),
        xrobot_right_hand=hand(),
    )
    values.update(overrides)
    return pb.PoseFrame(**values)


def test_direct_hands_need_no_smplx_model_or_calibration_or_body():
    message = packet(packet_age_seconds=0.02)
    result = parse_frame(message, config())
    assert not message.calibration_id and not message.body_valid
    assert result.left is not None and result.right is not None
    assert result.source_age_ms == 30
    np.testing.assert_array_equal(
        result.left.positions, np.arange(78).reshape(26, 3) / 1000
    )
    assert not result.left.positions.flags.writeable


def test_position_validity_is_independent_from_orientation_and_other_hand():
    message = packet()
    message.xrobot_left_hand.position_valid[7] = False
    message.xrobot_right_hand.orientation_valid[:] = [False] * 26
    result = parse_frame(message, config())
    assert result.left is None and result.right is not None


def test_frozen_hand_does_not_become_fresh_from_body_packets():
    message = packet()
    message.xrobot_left_hand.age_seconds = 0.6
    result = parse_frame(message, config())
    assert result.left is None and result.right is not None
    message.xrobot_right_hand.age_seconds = 0.6
    with pytest.raises(ReferenceUnavailable):
        parse_frame(message, config())


def test_hand_joint_order_is_explicit_and_reordered_by_name():
    message = packet()
    message.xrobot_left_hand.joint_names.reverse()
    message.xrobot_left_hand.positions[:] = (
        np.arange(78).reshape(26, 3)[::-1].ravel() / 1000
    )
    np.testing.assert_array_equal(
        parse_frame(message, config()).left.positions,
        np.arange(78).reshape(26, 3) / 1000,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("positions", [0] * 77),
        ("positions", [float("nan")] * 78),
        ("joint_names", ["wrist"] * 26),
        ("position_valid", [True] * 25),
        ("frame", "unity"),
        ("age_seconds", float("nan")),
        ("age_seconds", -1),
    ],
)
def test_malformed_hand_fails_closed(field, value):
    message = packet()
    if isinstance(value, list):
        getattr(message.xrobot_left_hand, field)[:] = value
    else:
        setattr(message.xrobot_left_hand, field, value)
    with pytest.raises(ValueError):
        parse_frame(message, config())


def test_missing_typed_hands_requires_upgrade_no_smplx_fallback():
    message = packet()
    message.ClearField("xrobot_left_hand")
    with pytest.raises(ValueError, match="upgrade"):
        parse_frame(message, config())


@pytest.mark.parametrize(
    "overrides",
    [
        {"upstream_connected": False},
        {"sequence": 0},
        {"packet_age_seconds": 1},
    ],
)
def test_unavailable_tracking_is_recoverable(overrides):
    with pytest.raises(ReferenceUnavailable):
        parse_frame(packet(**overrides), config())


@pytest.mark.parametrize(
    "overrides",
    [
        {"source_timestamp_ns": 0},
        {"device_id": ""},
        {"packet_age_seconds": -1},
        {"packet_age_seconds": float("nan")},
    ],
)
def test_valid_tracking_requires_identity_timestamp_and_age(overrides):
    with pytest.raises(ValueError):
        parse_frame(packet(**overrides), config())


class PoseServer:
    def __init__(self):
        self.packets = queue.Queue()
        self.subscriptions = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.server = grpc.server(self.executor)
        self.server.add_generic_rpc_handlers(
            (
                grpc.method_handlers_generic_handler(
                    "pico.smplx.v1.PoseStream",
                    {
                        "StreamPoses": grpc.unary_stream_rpc_method_handler(
                            self.stream,
                            request_deserializer=pb.StreamRequest.FromString,
                            response_serializer=pb.PoseFrame.SerializeToString,
                        )
                    },
                ),
            )
        )
        self.port = self.server.add_insecure_port("127.0.0.1:0")
        self.server.start()

    def stream(self, request, context):
        self.subscriptions.put(request)
        while context.is_active():
            try:
                value = self.packets.get(timeout=0.05)
            except queue.Empty:
                continue
            if isinstance(value, grpc.StatusCode):
                context.abort(value, "injected failure")
            yield value

    def close(self):
        self.server.stop(0).wait(2)
        self.executor.shutdown(wait=True)


@pytest.fixture
def server():
    value = PoseServer()
    try:
        yield value
    finally:
        value.close()


def read(stream, after=None, timeout=1):
    return stream.wait_next(after_sequence=after, timeout_seconds=timeout)


def test_direct_body_uses_shared_outage_barriers_without_smplx_calibration(server):
    from bello_mujoco.hardware.pico_grpc import PicoGrpcConfig
    from bello_mujoco.hardware.xrobot_stream import BODY_NAMES, XRobotGrpcStream

    def raw(sequence):
        message = packet(sequence, calibration_id="", body_calibrated=False)
        message.xrobot_body.CopyFrom(
            pb.XRobotBody(
                joint_names=BODY_NAMES,
                positions=np.tile([0, 1, 0], 24),
                orientations=np.tile([1, 0, 0, 0], 24),
                frame="pico_native_y_up",
                valid=True,
            )
        )
        return message

    with XRobotGrpcStream(PicoGrpcConfig(f"127.0.0.1:{server.port}")) as stream:
        stale = raw(10)
        stale.source_timestamp_ns = 0
        stale.packet_age_seconds = 53_000
        stale.xrobot_body.valid = False
        server.packets.put(stale)
        with pytest.raises(ReferenceUnavailable):
            read(stream)
        server.packets.put(raw(10))
        assert read(stream).sequence == 10
        server.packets.put(raw(10))
        assert read(stream, timeout=0.03) is None
        server.packets.put(raw(1))
        with pytest.raises(ReferenceDiscontinuity):
            read(stream, after=10)
        assert read(stream).sequence == 1
        invalid = raw(2)
        invalid.xrobot_body.valid = False
        invalid.ClearField("source_timestamp_ns")
        invalid.ClearField("device_id")
        server.packets.put(invalid)
        with pytest.raises(ReferenceUnavailable):
            read(stream, after=1)
        server.packets.put(raw(3))
        assert read(stream, after=1).sequence == 3


def test_local_grpc_single_delivery_and_missing_body_does_not_spin(server):
    with XRobotHandsGrpcStream(config(endpoint=f"127.0.0.1:{server.port}")) as stream:
        assert server.subscriptions.get(timeout=2).max_rate_hz == 60
        server.packets.put(packet())
        first = read(stream)
        assert first is not None and first.left is not None
        assert read(stream, timeout=0.03) is None
        server.packets.put(packet())  # Same observation is not new data.
        assert read(stream, timeout=0.03) is None
        server.packets.put(packet(2))
        assert read(stream).sequence == 2
    assert not stream._thread.is_alive()
    assert not stream._watchdog.is_alive()


@pytest.mark.parametrize(
    "changed",
    [
        packet(1),
        packet(11, device_id="new"),
        packet(11, source_timestamp_ns=1),
        packet(11, source_timestamp_ns=2_000_000_000),
    ],
)
def test_epoch_and_clock_discontinuities_precede_new_frame(server, changed):
    with XRobotHandsGrpcStream(config(endpoint=f"127.0.0.1:{server.port}")) as stream:
        server.packets.put(packet(10))
        assert read(stream).sequence == 10
        server.packets.put(changed)
        with pytest.raises(ReferenceDiscontinuity):
            read(stream, after=10)
        assert read(stream).sequence == changed.sequence


def test_duplicates_do_not_refresh_age_and_stale_error_is_delivered_once(server):
    with XRobotHandsGrpcStream(
        config(endpoint=f"127.0.0.1:{server.port}", max_source_age_ms=100)
    ) as stream:
        server.packets.put(packet())
        assert read(stream).sequence == 1
        time.sleep(0.11)
        server.packets.put(packet())
        with pytest.raises(ReferenceUnavailable, match="stale"):
            read(stream)
        assert read(stream, timeout=0.03) is None


def test_rpc_outage_resets_sequence_and_recovers(server):
    with XRobotHandsGrpcStream(config(endpoint=f"127.0.0.1:{server.port}")) as stream:
        server.packets.put(packet(20))
        assert read(stream).sequence == 20
        server.subscriptions.get(timeout=1)
        server.packets.put(grpc.StatusCode.UNAVAILABLE)
        with pytest.raises(ReferenceDiscontinuity):
            read(stream, after=20)
        server.subscriptions.get(timeout=2)
        server.packets.put(packet(1))
        assert read(stream).sequence == 1


def test_slow_consumer_gets_latest_frame_after_latched_identity_barrier(server):
    with XRobotHandsGrpcStream(config(endpoint=f"127.0.0.1:{server.port}")) as stream:
        server.packets.put(packet(1))
        assert read(stream).sequence == 1
        for sequence in range(2, 21):
            server.packets.put(packet(sequence, device_id="new"))
        with stream._condition:
            assert stream._condition.wait_for(
                lambda: stream._latest is not None and stream._latest.sequence == 20,
                timeout=2,
            )
        with pytest.raises(ReferenceDiscontinuity):
            read(stream, after=1)
        assert read(stream).sequence == 20
        assert read(stream, timeout=0.03) is None


def test_close_cancels_waiting_for_a_connection():
    # Allocate a real unused port without making an external connection.
    import socket

    with socket.socket() as socket_:
        socket_.bind(("127.0.0.1", 0))
        port = socket_.getsockname()[1]
        stream = XRobotHandsGrpcStream(config(endpoint=f"127.0.0.1:{port}"))
        with stream:
            assert read(stream, timeout=0.03) is None
        assert not stream._thread.is_alive()


def test_watchdog_reconnects_a_silent_stream(server):
    with XRobotHandsGrpcStream(
        config(endpoint=f"127.0.0.1:{server.port}", rpc_timeout_seconds=0.2)
    ) as stream:
        server.subscriptions.get(timeout=1)
        server.packets.put(packet(10))
        assert read(stream).sequence == 10
        with pytest.raises(ReferenceDiscontinuity):
            read(stream)
        server.subscriptions.get(timeout=2)
        server.packets.put(packet())
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            try:
                frame = read(stream, timeout=0.1)
                if frame:
                    assert frame.sequence == 1
                    break
            except ReferenceDiscontinuity:
                pass
        else:
            pytest.fail("stream did not recover after watchdog cancellation")


@pytest.mark.parametrize(
    "value",
    [
        packet(xrobot_left_hand=pb.XRobotHand(active=True, frame="pico_native_y_up")),
        grpc.StatusCode.UNIMPLEMENTED,
    ],
)
def test_malformed_data_and_protocol_failure_are_terminal(server, value):
    with XRobotHandsGrpcStream(config(endpoint=f"127.0.0.1:{server.port}")) as stream:
        server.packets.put(value)
        with pytest.raises(RuntimeError, match="stream failed"):
            read(stream)
