from __future__ import annotations

from pathlib import Path
import queue
import sys
import threading
import time
from types import SimpleNamespace

import grpc
import numpy as np
import pytest

from bello_mujoco.hardware import rtmw
from bello_mujoco.hardware.rtmw import (
    RTMW_COORDINATE_SPACE,
    RTMW_REPRESENTATION,
    RTMW_ROTATION_SEMANTICS,
    RTMW_SCHEMA_ID,
    RTMW_SMPLX52_NAMES,
    RTMW_SMPLX52_PARENTS,
    RtmwCoordinateAdapter,
    RtmwGrpcConfig,
    RtmwSmplxStream,
    _validate_stream_info,
    import_pinned_rtmw_proto,
    validate_pinned_rtmw_contract,
)
from bello_mujoco.hardware.smplx_retarget import (
    SMPLX_MODEL_TO_WORLD_WXYZ,
    SmplxTargetFrame,
)
from bello_mujoco.hardware.twist2 import (
    ReferenceDiscontinuity,
    ReferenceUnavailable,
)
from bello_mujoco.repository import repository_root
from bello_mujoco.robot_tasks import motion_task_definition


GMR_SOURCE_BODY_NAMES = motion_task_definition("bello").gmr.source_body_names
_END_STREAM = object()


class FakeRpcError(grpc.RpcError):
    def __init__(self, code: grpc.StatusCode, detail: str) -> None:
        self._code = code
        self._detail = detail

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return self._detail


class FakeCall:
    def __init__(self) -> None:
        self._messages: queue.Queue[object] = queue.Queue()
        self.read_started = threading.Event()
        self.cancelled = False

    def __iter__(self) -> FakeCall:
        return self

    def __next__(self) -> object:
        self.read_started.set()
        payload = self._messages.get()
        if payload is _END_STREAM:
            raise StopIteration
        if isinstance(payload, BaseException):
            raise payload
        return payload

    def send(self, payload: object) -> None:
        self._messages.put(payload)

    def end(self) -> None:
        self._messages.put(_END_STREAM)

    def cancel(self) -> None:
        self.cancelled = True
        self._messages.put(FakeRpcError(grpc.StatusCode.CANCELLED, "cancelled"))


class FakeReadyFuture:
    def __init__(self, *, ready: bool = True) -> None:
        self.started = threading.Event()
        self.available = threading.Event()
        self.cancelled = threading.Event()
        if ready:
            self.available.set()

    def result(self, timeout: float) -> None:
        self.started.set()
        deadline = time.monotonic() + timeout
        while not self.cancelled.is_set():
            if self.available.wait(min(0.01, max(0.0, deadline - time.monotonic()))):
                return
            if time.monotonic() >= deadline:
                raise grpc.FutureTimeoutError()
        if self.cancelled.is_set():
            raise grpc.FutureCancelledError()
        if self.available.is_set():
            return
        raise grpc.FutureTimeoutError()

    def make_ready(self) -> None:
        self.available.set()

    def cancel(self) -> bool:
        self.cancelled.set()
        return True


class FakeChannel:
    def __init__(self, stub: object, *, ready: bool = True) -> None:
        self.stub = stub
        self.ready_future = FakeReadyFuture(ready=ready)
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeStub:
    def __init__(
        self,
        info: object,
        call: FakeCall,
        *,
        status: object | None = None,
    ) -> None:
        self.info = info
        self.call = call
        self.status = status or SimpleNamespace(
            terminal_failure="",
            ready=True,
            source_running=True,
            inference_running=True,
        )
        self.info_requests = 0
        self.status_requests = 0

    def GetStreamInfo(self, _request: object, *, timeout: float) -> object:
        del timeout
        self.info_requests += 1
        return self.info

    def GetStatus(self, _request: object, *, timeout: float) -> object:
        del timeout
        self.status_requests += 1
        return self.status

    def Subscribe(self, _request: object) -> FakeCall:
        return self.call


class FailingLivenessStub(FakeStub):
    def GetStatus(self, _request: object, *, timeout: float) -> object:
        del timeout
        self.status_requests += 1
        if self.status_requests > 1:
            raise FakeRpcError(grpc.StatusCode.DEADLINE_EXCEEDED, "half-open")
        return self.status


class AdvancingStatusStub(FakeStub):
    def GetStatus(self, _request: object, *, timeout: float) -> object:
        del timeout
        self.status_requests += 1
        return SimpleNamespace(
            terminal_failure="",
            ready=True,
            source_running=True,
            inference_running=True,
            latest_frame_id=self.status_requests,
        )


class SparseAdvancingStatusStub(FakeStub):
    def GetStatus(self, _request: object, *, timeout: float) -> object:
        del timeout
        self.status_requests += 1
        frame_ids = (1, 2, 2, 3, 3)
        frame_id = frame_ids[min(self.status_requests - 1, len(frame_ids) - 1)]
        return SimpleNamespace(
            terminal_failure="",
            ready=True,
            source_running=True,
            inference_running=True,
            latest_frame_id=frame_id,
        )


def rtmw_config(tmp_path: Path, **overrides: object) -> RtmwGrpcConfig:
    values = {
        "endpoint": "127.0.0.1:50055",
        "neutral_model_path": tmp_path / "SMPLX_NEUTRAL.npz",
        "tls_root_certificate_path": None,
        "tls_server_name_override": None,
        "connect_timeout_seconds": 5.0,
        "rpc_timeout_seconds": 2.0,
        "human_height_m": 1.75,
        "min_joint_confidence": 0.0,
        "min_rotation_confidence": 0.0,
        "max_required_rotation_residual_degrees": 35.0,
        "max_pipeline_latency_ms": 250.0,
        "transport_latency_budget_ms": 10.0,
        "max_source_interval_seconds": 0.70,
        "max_source_age_ms": 1_000,
        "camera_to_world_quaternion_xyzw": (0.0, 0.0, 0.0, 1.0),
        "align_initial_heading": False,
        "align_initial_floor": False,
    }
    values.update(overrides)
    return RtmwGrpcConfig(**values)


def standing_sample(
    _path: Path, source_body_names: tuple[str, ...]
) -> SmplxTargetFrame:
    identity = np.asarray((1.0, 0.0, 0.0, 0.0))
    return SmplxTargetFrame(
        transforms={name: (np.zeros(3), identity.copy()) for name in source_body_names},
        sequence=0,
        source_timestamp_ns=0,
        source_age_ms=0,
        received_at=time.monotonic(),
    )


def rtmw_frame(
    captured_at_ns: int,
    *,
    frame_id: int = 7,
    track_id: str = "3",
) -> SimpleNamespace:
    count = len(RTMW_SMPLX52_NAMES)
    positions = np.stack(
        (
            np.linspace(0.0, 0.51, count),
            np.linspace(0.2, -0.31, count),
            np.linspace(1.0, 0.49, count),
        ),
        axis=1,
    )
    identity = np.tile(np.asarray((0.0, 0.0, 0.0, 1.0)), (count, 1))
    confidence = np.full(count, 0.9)
    rotation_valid = np.zeros(count, dtype=bool)
    rotation_residual = np.full(count, np.nan)
    rotation_rank = np.zeros(count, dtype=np.uint8)
    for name in (
        "pelvis",
        "spine3",
        "left_shoulder",
        "left_elbow",
        "left_wrist",
        "right_shoulder",
        "right_elbow",
        "right_wrist",
    ):
        index = RTMW_SMPLX52_NAMES.index(name)
        rotation_valid[index] = True
        rotation_residual[index] = 0.0
        rotation_rank[index] = 2
    return SimpleNamespace(
        frame_id=frame_id,
        source_timestamp_ns=captured_at_ns,
        captured_monotonic_ns=captured_at_ns,
        output_monotonic_ns=captured_at_ns + 10_000_000,
        stream_instance_id="test-stream",
        track_id=track_id,
        tracking_state="tracked",
        root_translation=positions[0].tolist(),
        joint_positions=positions.reshape(-1).tolist(),
        local_joint_quaternions=identity.reshape(-1).tolist(),
        global_joint_quaternions=identity.reshape(-1).tolist(),
        joint_confidence=confidence.tolist(),
        joint_valid=np.ones(count, dtype=bool).tolist(),
        rotation_valid=rotation_valid.tolist(),
        rotation_confidence=np.where(rotation_valid, 0.8, 0.0).tolist(),
        rotation_residual_degrees=rotation_residual.tolist(),
        rotation_constraint_rank=rotation_rank.tolist(),
        pipeline_latency_ms=10.0,
    )


def stream_info(**overrides: object) -> SimpleNamespace:
    values = {
        "schema_version": 1,
        "stream_instance_id": "test-stream",
        "config_sha256": "1" * 64,
        "joint_names": RTMW_SMPLX52_NAMES,
        "joint_parents": RTMW_SMPLX52_PARENTS,
        "root_index": 0,
        "coordinate_space": RTMW_COORDINATE_SPACE,
        "quaternion_order": "xyzw",
        "rotation_semantics": RTMW_ROTATION_SEMANTICS,
        "position_units": "meters",
        "representation": RTMW_REPRESENTATION,
        "schema_id": RTMW_SCHEMA_ID,
        "depth_provenance": "calibrated_camera+assumed_subject_height",
        "metric_scale": True,
        "global_translation": True,
        "global_depth_is_measured": False,
        "subject_height_m": 1.75,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_stream(tmp_path: Path, **config_overrides: object) -> RtmwSmplxStream:
    config = rtmw_config(tmp_path, **config_overrides)
    config.neutral_model_path.write_bytes(b"test-model")
    stream = RtmwSmplxStream(
        config,
        GMR_SOURCE_BODY_NAMES,
        standing_sample_loader=standing_sample,
    )
    stream.stream_instance_id = "test-stream"
    return stream


def configure_fake_transport(
    monkeypatch: pytest.MonkeyPatch,
    stream: RtmwSmplxStream,
    channels: list[FakeChannel],
) -> None:
    pending_channels = iter(channels)
    monkeypatch.setattr(
        rtmw,
        "import_pinned_rtmw_proto",
        lambda: (
            SimpleNamespace(SubscribeRequest=lambda: object()),
            SimpleNamespace(SkeletonServiceStub=lambda channel: channel.stub),
        ),
    )
    monkeypatch.setattr(
        grpc,
        "channel_ready_future",
        lambda channel: channel.ready_future,
    )
    monkeypatch.setattr(stream, "_create_channel", lambda: next(pending_channels))


def test_rtmw_grpc_accepts_the_published_schema(tmp_path: Path) -> None:
    captured_at_ns = 5_000_000_000
    stream = make_stream(tmp_path)

    frame = stream._parse_frame(rtmw_frame(captured_at_ns))

    assert frame.sequence == 7
    assert frame.source_timestamp_ns == captured_at_ns
    assert frame.source_age_ms == 20
    assert tuple(frame.transforms) == GMR_SOURCE_BODY_NAMES
    assert frame.transforms["pelvis"][0] == pytest.approx((0.0, 0.0, 1.0))
    assert frame.transforms["pelvis"][1] == pytest.approx((1.0, 0.0, 0.0, 0.0))
    assert frame.foot_contacts is not None


def test_rtmw_liveness_probe_preserves_a_healthy_idle_stream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(rtmw, "_LIVENESS_PROBE_INTERVAL_SECONDS", 0.01)
    stream = make_stream(tmp_path)
    call = FakeCall()
    stub = FakeStub(stream_info(), call)
    channel = FakeChannel(stub)
    configure_fake_transport(monkeypatch, stream, [channel])

    with stream:
        deadline = time.monotonic() + 1.0
        while stub.status_requests < 3:
            assert time.monotonic() < deadline
            time.sleep(0.001)

        assert not call.cancelled
        assert not channel.closed
        assert stream.wait_next(after_sequence=None, timeout_seconds=0.02) is None


def test_rtmw_liveness_failure_cancels_and_reconnects_half_open_stream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(rtmw, "_LIVENESS_PROBE_INTERVAL_SECONDS", 0.01)
    stream = make_stream(tmp_path)
    stale_call = FakeCall()
    stale_channel = FakeChannel(FailingLivenessStub(stream_info(), stale_call))
    recovered_call = FakeCall()
    recovered_channel = FakeChannel(
        FakeStub(stream_info(stream_instance_id="recovered"), recovered_call)
    )
    configure_fake_transport(
        monkeypatch,
        stream,
        [stale_channel, recovered_channel],
    )

    with stream:
        with pytest.raises(ReferenceDiscontinuity, match="reconnecting to RTMW"):
            stream.wait_next(after_sequence=None, timeout_seconds=1.0)
        recovered = rtmw_frame(8_000_000_000, frame_id=1, track_id="recovered")
        recovered.stream_instance_id = "recovered"
        recovered_call.send(recovered)

        frame = stream.wait_next(after_sequence=None, timeout_seconds=1.0)

        assert frame is not None
        assert frame.sequence == 1
        assert stale_call.cancelled
        assert stale_channel.closed


def test_rtmw_liveness_probe_ignores_failures_while_frames_arrive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(rtmw, "_LIVENESS_PROBE_INTERVAL_SECONDS", 0.01)
    stream = make_stream(tmp_path)
    call = FakeCall()
    stub = FailingLivenessStub(stream_info(), call)
    channel = FakeChannel(stub)
    configure_fake_transport(monkeypatch, stream, [channel])

    with stream:
        for frame_id in range(1, 51):
            call.send(
                rtmw_frame(
                    5_000_000_000 + frame_id * 10_000_000,
                    frame_id=frame_id,
                )
            )
            time.sleep(0.002)

        assert not call.cancelled
        assert not channel.closed


@pytest.mark.parametrize(
    "stub_type",
    (AdvancingStatusStub, SparseAdvancingStatusStub),
)
def test_rtmw_liveness_reconnects_when_producer_advances_but_stream_stalls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_type: type[FakeStub],
) -> None:
    monkeypatch.setattr(rtmw, "_LIVENESS_PROBE_INTERVAL_SECONDS", 0.01)
    stream = make_stream(tmp_path)
    stale_call = FakeCall()
    stale_channel = FakeChannel(stub_type(stream_info(), stale_call))
    recovered_call = FakeCall()
    recovered_channel = FakeChannel(
        FakeStub(stream_info(stream_instance_id="recovered"), recovered_call)
    )
    configure_fake_transport(
        monkeypatch,
        stream,
        [stale_channel, recovered_channel],
    )

    with stream:
        with pytest.raises(ReferenceDiscontinuity, match="reconnecting to RTMW"):
            stream.wait_next(after_sequence=None, timeout_seconds=1.0)
        recovered = rtmw_frame(8_000_000_000, frame_id=1, track_id="recovered")
        recovered.stream_instance_id = "recovered"
        recovered_call.send(recovered)

        frame = stream.wait_next(after_sequence=None, timeout_seconds=1.0)

        assert frame is not None
        assert frame.sequence == 1
        assert stale_call.cancelled
        assert stale_channel.closed


@pytest.mark.parametrize("outage", ("unavailable", "clean-eof"))
def test_rtmw_fast_reconnect_cannot_overwrite_outage_discontinuity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    outage: str,
) -> None:
    stream = make_stream(tmp_path, connect_timeout_seconds=0.2)
    first_call = FakeCall()
    second_call = FakeCall()
    first_channel = FakeChannel(FakeStub(stream_info(), first_call))
    second_channel = FakeChannel(
        FakeStub(
            stream_info(stream_instance_id="restarted-stream", config_sha256="2" * 64),
            second_call,
        )
    )
    configure_fake_transport(
        monkeypatch,
        stream,
        [first_channel, second_channel],
    )

    with stream:
        first_call.send(rtmw_frame(5_000_000_000, frame_id=21))
        first = stream.wait_next(after_sequence=None, timeout_seconds=1.0)
        assert first is not None
        assert first.sequence == 21

        restarted = rtmw_frame(6_000_000_000, frame_id=22, track_id="new-track")
        restarted.stream_instance_id = "restarted-stream"
        second_call.send(restarted)
        if outage == "unavailable":
            first_call.send(
                FakeRpcError(grpc.StatusCode.UNAVAILABLE, "producer stopped")
            )
        else:
            first_call.end()
        deadline = time.monotonic() + 1.0
        while not first_channel.closed:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        time.sleep(0.02)
        assert not second_call.read_started.is_set()
        with pytest.raises(ReferenceDiscontinuity, match="reconnecting to RTMW"):
            stream.wait_next(after_sequence=first.sequence, timeout_seconds=1.0)

        second = stream.wait_next(after_sequence=first.sequence, timeout_seconds=1.0)

        assert second is not None
        assert second.sequence == 22
        assert stream.stream_instance_id == "restarted-stream"
        assert stream.stream_config_sha256 == "2" * 64

    assert first_channel.closed
    assert second_channel.closed


def test_rtmw_replacement_producer_can_reset_its_frame_sequence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stream = make_stream(tmp_path)
    first_call = FakeCall()
    replacement_call = FakeCall()
    configure_fake_transport(
        monkeypatch,
        stream,
        [
            FakeChannel(FakeStub(stream_info(), first_call)),
            FakeChannel(
                FakeStub(
                    stream_info(stream_instance_id="replacement"),
                    replacement_call,
                )
            ),
        ],
    )

    with stream:
        first_call.send(rtmw_frame(5_000_000_000, frame_id=21))
        first = stream.wait_next(after_sequence=None, timeout_seconds=1.0)
        assert first is not None
        first_call.send(FakeRpcError(grpc.StatusCode.UNAVAILABLE, "restart"))
        with pytest.raises(ReferenceDiscontinuity):
            stream.wait_next(after_sequence=first.sequence, timeout_seconds=1.0)

        replacement = rtmw_frame(
            6_000_000_000,
            frame_id=1,
            track_id="replacement-track",
        )
        replacement.stream_instance_id = "replacement"
        replacement_call.send(replacement)
        resumed = stream.wait_next(after_sequence=None, timeout_seconds=1.0)

        assert resumed is not None
        assert resumed.sequence == 1
        assert stream.stream_instance_id == "replacement"


def test_rtmw_nonretryable_grpc_failure_remains_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stream = make_stream(tmp_path)
    call = FakeCall()
    channel = FakeChannel(FakeStub(stream_info(), call))
    configure_fake_transport(monkeypatch, stream, [channel])

    with stream:
        call.send(FakeRpcError(grpc.StatusCode.DATA_LOSS, "corrupt stream"))
        with pytest.raises(RuntimeError, match="RTMW gRPC stream failed") as raised:
            stream.wait_next(after_sequence=None, timeout_seconds=1.0)

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert "DATA_LOSS: corrupt stream" in str(raised.value.__cause__)


def test_rtmw_reconnect_revalidates_the_producer_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stream = make_stream(tmp_path)
    first_call = FakeCall()
    incompatible_call = FakeCall()
    channels = [
        FakeChannel(FakeStub(stream_info(), first_call)),
        FakeChannel(
            FakeStub(
                stream_info(
                    stream_instance_id="incompatible",
                    coordinate_space="root_relative",
                ),
                incompatible_call,
            ),
            ready=False,
        ),
    ]
    configure_fake_transport(monkeypatch, stream, channels)

    with stream:
        first_call.send(FakeRpcError(grpc.StatusCode.UNAVAILABLE, "restart"))
        with pytest.raises(ReferenceDiscontinuity):
            stream.wait_next(after_sequence=None, timeout_seconds=1.0)
        channels[1].ready_future.make_ready()
        with pytest.raises(RuntimeError, match="RTMW gRPC stream failed") as raised:
            stream.wait_next(after_sequence=None, timeout_seconds=1.0)

    assert isinstance(raised.value.__cause__, ValueError)
    assert "coordinate_space" in str(raised.value.__cause__)


def test_rtmw_reconnect_retries_while_the_service_is_starting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stream = make_stream(tmp_path, connect_timeout_seconds=0.2)
    first_call = FakeCall()
    warming_call = FakeCall()
    recovered_call = FakeCall()
    warming_channel = FakeChannel(
        FakeStub(
            stream_info(stream_instance_id="warming"),
            warming_call,
            status=SimpleNamespace(
                terminal_failure="",
                ready=False,
                source_running=False,
                inference_running=False,
            ),
        )
    )
    recovered_channel = FakeChannel(
        FakeStub(
            stream_info(stream_instance_id="recovered"),
            recovered_call,
        )
    )
    configure_fake_transport(
        monkeypatch,
        stream,
        [
            FakeChannel(FakeStub(stream_info(), first_call)),
            warming_channel,
            recovered_channel,
        ],
    )

    with stream:
        first_call.send(FakeRpcError(grpc.StatusCode.UNAVAILABLE, "restart"))
        with pytest.raises(ReferenceDiscontinuity):
            stream.wait_next(after_sequence=None, timeout_seconds=1.0)
        recovered = rtmw_frame(8_000_000_000, frame_id=1, track_id="recovered")
        recovered.stream_instance_id = "recovered"
        recovered_call.send(recovered)

        frame = stream.wait_next(after_sequence=None, timeout_seconds=1.0)

        assert frame is not None
        assert frame.sequence == 1
        assert stream.stream_instance_id == "recovered"

    assert warming_channel.closed
    assert recovered_channel.closed


def test_rtmw_reconnect_backoff_persists_until_a_frame_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stream = make_stream(tmp_path)
    waits: list[float] = []
    monkeypatch.setattr(
        stream._stop,
        "wait",
        lambda timeout: waits.append(timeout) or False,
    )
    first_call = FakeCall()
    first_half_alive_call = FakeCall()
    first_half_alive_call.end()
    second_half_alive_call = FakeCall()
    second_half_alive_call.end()
    recovered_call = FakeCall()
    recovered = rtmw_frame(9_000_000_000, frame_id=1, track_id="recovered")
    recovered.stream_instance_id = "recovered"
    recovered_call.send(recovered)
    reset_half_alive_call = FakeCall()
    reset_half_alive_call.end()
    final_call = FakeCall()
    final = rtmw_frame(10_000_000_000, frame_id=2, track_id="final")
    final.stream_instance_id = "final"
    final_call.send(final)
    configure_fake_transport(
        monkeypatch,
        stream,
        [
            FakeChannel(FakeStub(stream_info(), first_call)),
            FakeChannel(
                FakeStub(
                    stream_info(stream_instance_id="half-alive-1"),
                    first_half_alive_call,
                )
            ),
            FakeChannel(
                FakeStub(
                    stream_info(stream_instance_id="half-alive-2"),
                    second_half_alive_call,
                )
            ),
            FakeChannel(
                FakeStub(
                    stream_info(stream_instance_id="recovered"),
                    recovered_call,
                )
            ),
            FakeChannel(
                FakeStub(
                    stream_info(stream_instance_id="half-alive-after-frame"),
                    reset_half_alive_call,
                )
            ),
            FakeChannel(
                FakeStub(
                    stream_info(stream_instance_id="final"),
                    final_call,
                )
            ),
        ],
    )

    with stream:
        first_call.send(FakeRpcError(grpc.StatusCode.UNAVAILABLE, "restart"))
        with pytest.raises(ReferenceDiscontinuity):
            stream.wait_next(after_sequence=None, timeout_seconds=1.0)

        frame = stream.wait_next(after_sequence=None, timeout_seconds=1.0)

        assert frame is not None
        assert frame.sequence == 1
        recovered_call.send(FakeRpcError(grpc.StatusCode.UNAVAILABLE, "restart again"))
        with pytest.raises(ReferenceDiscontinuity):
            stream.wait_next(after_sequence=frame.sequence, timeout_seconds=1.0)

        final_frame = stream.wait_next(after_sequence=None, timeout_seconds=1.0)

        assert final_frame is not None
        assert final_frame.sequence == 2
        assert waits == pytest.approx((0.1, 0.2, 0.1))


def test_rtmw_cleanup_interrupts_an_in_progress_reconnect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stream = make_stream(tmp_path, connect_timeout_seconds=30.0)
    first_call = FakeCall()
    first_channel = FakeChannel(FakeStub(stream_info(), first_call))
    reconnect_channel = FakeChannel(object(), ready=False)
    configure_fake_transport(
        monkeypatch,
        stream,
        [first_channel, reconnect_channel],
    )

    stream.__enter__()
    try:
        first_call.send(FakeRpcError(grpc.StatusCode.UNAVAILABLE, "restart"))
        with pytest.raises(ReferenceDiscontinuity):
            stream.wait_next(after_sequence=None, timeout_seconds=1.0)
        assert reconnect_channel.ready_future.started.wait(timeout=1.0)

        started = time.monotonic()
        stream.__exit__(None, None, None)
        elapsed = time.monotonic() - started

        assert elapsed < 1.0
        assert reconnect_channel.ready_future.cancelled.is_set()
        assert reconnect_channel.closed
    finally:
        if stream._thread is not None:
            stream.__exit__(None, None, None)


def test_initial_heading_is_relative_to_the_smplx_world_basis(tmp_path: Path) -> None:
    camera_to_world_xyzw = (0.5, -0.5, 0.5, -0.5)
    base = rtmw_config(
        tmp_path,
        camera_to_world_quaternion_xyzw=camera_to_world_xyzw,
        align_initial_heading=False,
    )
    aligned = rtmw_config(
        tmp_path,
        camera_to_world_quaternion_xyzw=camera_to_world_xyzw,
        align_initial_heading=True,
    )
    positions = np.zeros((len(RTMW_SMPLX52_NAMES), 3), dtype=np.float64)
    positions[:, 2] = 2.0
    positions[RTMW_SMPLX52_NAMES.index("left_shoulder"), 0] = 0.3
    positions[RTMW_SMPLX52_NAMES.index("right_shoulder"), 0] = -0.3
    # A front-facing neutral SMPL-X frame maps model axes into OpenCV camera
    # axes with a 180-degree rotation around camera X.
    quaternions_xyzw = np.tile(
        np.asarray((1.0, 0.0, 0.0, 0.0)),
        (len(RTMW_SMPLX52_NAMES), 1),
    )

    expected_positions, expected_quaternions = RtmwCoordinateAdapter(base).transform(
        positions, quaternions_xyzw, track_id="neutral"
    )
    actual_positions, actual_quaternions = RtmwCoordinateAdapter(aligned).transform(
        positions, quaternions_xyzw, track_id="neutral"
    )

    assert actual_positions == pytest.approx(expected_positions)
    assert actual_quaternions == pytest.approx(expected_quaternions)
    shoulder_delta = (
        actual_positions[RTMW_SMPLX52_NAMES.index("left_shoulder")]
        - actual_positions[RTMW_SMPLX52_NAMES.index("right_shoulder")]
    )
    assert abs(shoulder_delta[1]) == pytest.approx(0.6)
    assert shoulder_delta[0] == pytest.approx(0.0)


def test_initial_heading_removes_nonzero_subject_yaw(tmp_path: Path) -> None:
    camera_to_world_xyzw = (0.5, -0.5, 0.5, -0.5)
    config = rtmw_config(
        tmp_path,
        camera_to_world_quaternion_xyzw=camera_to_world_xyzw,
        align_initial_heading=True,
    )
    yaw = 0.4
    yaw_quaternion = np.asarray((np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)))
    canonical = np.asarray(SMPLX_MODEL_TO_WORLD_WXYZ, dtype=np.float64)
    pelvis_world = rtmw._quat_multiply(yaw_quaternion, canonical)
    camera_to_world = np.asarray(camera_to_world_xyzw)[[3, 0, 1, 2]]
    pelvis_camera = rtmw._quat_multiply(
        rtmw._quat_conjugate(camera_to_world), pelvis_world
    )
    quaternions_xyzw = np.tile(
        pelvis_camera[[1, 2, 3, 0]],
        (len(RTMW_SMPLX52_NAMES), 1),
    )
    positions = np.zeros((len(RTMW_SMPLX52_NAMES), 3), dtype=np.float64)

    _, aligned_quaternions = RtmwCoordinateAdapter(config).transform(
        positions, quaternions_xyzw, track_id="yawed"
    )

    pelvis = RTMW_SMPLX52_NAMES.index("pelvis")
    assert aligned_quaternions[pelvis] == pytest.approx(canonical)


def test_pinned_rtmw_contract_and_protobuf_match_consumer(monkeypatch) -> None:
    submodule = repository_root() / "deps" / "RTMW-SMPLX-Teleop"
    validate_pinned_rtmw_contract(submodule)
    monkeypatch.setattr(rtmw, "require_pinned_rtmw_revision", lambda: submodule)
    for name in tuple(sys.modules):
        if name == "rtmw_teleop" or name.startswith("rtmw_teleop."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    skeleton_pb2, skeleton_pb2_grpc = import_pinned_rtmw_proto()

    assert skeleton_pb2.StreamInfo.DESCRIPTOR.full_name == "rtmw.teleop.v1.StreamInfo"
    assert hasattr(skeleton_pb2_grpc, "SkeletonServiceStub")


@pytest.mark.parametrize(
    "asset",
    (
        pytest.param("bello", id="bello"),
        pytest.param("bello_mini", id="bello-mini"),
    ),
)
def test_rtmw_reference_source_uses_selected_robot_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    asset: str,
) -> None:
    source_config = SimpleNamespace(
        endpoint="127.0.0.1:50055",
        human_height_m=1.75,
        max_source_interval_seconds=0.70,
    )
    identity = object()
    calls: dict[str, object] = {}

    class FakeStream:
        neutral_model_sha256 = "a" * 64
        stream_instance_id = "test-stream"
        stream_config_sha256 = "b" * 64
        depth_provenance = "test-depth"

        def __enter__(self) -> FakeStream:
            return self

        def __exit__(self, *_exception: object) -> bool:
            return False

    stream = FakeStream()

    class FakeSource:
        def __init__(self, actual_stream: object, retargeter: object) -> None:
            calls["source"] = (actual_stream, retargeter)

        def __enter__(self) -> FakeSource:
            return self

        def __exit__(self, *_exception: object) -> bool:
            return False

    monkeypatch.setattr(
        rtmw.RtmwGrpcConfig,
        "from_hardware",
        classmethod(lambda _cls, _config: source_config),
    )
    monkeypatch.setattr(rtmw, "require_pinned_rtmw_revision", lambda: tmp_path)
    monkeypatch.setattr(rtmw, "validate_pinned_rtmw_contract", lambda _root: None)

    def make_stream(actual_config: object, source_body_names: tuple[str, ...]):
        calls["stream"] = (actual_config, source_body_names)
        return stream

    def source_identity(asset: str) -> object:
        calls["identity_asset"] = asset
        return identity

    def make_retargeter(
        env: object,
        actual_identity: object,
        config: object,
        *,
        validate_reference: object,
    ) -> object:
        retargeter = object()
        calls["retargeter"] = (
            env,
            actual_identity,
            config,
            validate_reference,
            retargeter,
        )
        return retargeter

    monkeypatch.setattr(rtmw, "RtmwSmplxStream", make_stream)
    monkeypatch.setattr(rtmw, "repository_gmr_source_identity", source_identity)
    monkeypatch.setattr(rtmw, "GmrRetargeter", make_retargeter)
    monkeypatch.setattr(rtmw, "LiveGmrReferenceSource", FakeSource)

    validation = object()
    env = SimpleNamespace(model=SimpleNamespace(robot=SimpleNamespace(asset=asset)))
    adapter = SimpleNamespace(
        env=env,
        validate_upper_body_reference=validation,
    )
    hardware_config = SimpleNamespace(
        rtmw_gmr_task_profile="live_upper_body",
        smplx_human_height_m=1.66,
    )
    with rtmw.rtmw_locomotion_reference_source(adapter, hardware_config) as result:
        assert isinstance(result, FakeSource)

    assert calls["stream"] == (
        source_config,
        motion_task_definition(asset).gmr.source_body_names,
    )
    assert calls["identity_asset"] == asset
    retargeter_call = calls["retargeter"]
    assert isinstance(retargeter_call, tuple)
    assert retargeter_call[0] is env
    assert retargeter_call[1] is identity
    assert retargeter_call[2].human_height_m == pytest.approx(1.66)
    assert retargeter_call[2].task_profile == "live_upper_body"
    assert retargeter_call[2].max_source_interval_seconds == pytest.approx(0.70)
    assert retargeter_call[3] is validation
    assert calls["source"] == (stream, retargeter_call[4])


def test_rtmw_rejects_an_invalid_required_rotation(tmp_path: Path) -> None:
    stream = make_stream(tmp_path)
    message = rtmw_frame(8_000_000_000)
    index = RTMW_SMPLX52_NAMES.index("pelvis")
    message.rotation_valid[index] = False

    with pytest.raises(ReferenceUnavailable, match="pelvis"):
        stream._parse_frame(message)


def test_rtmw_requires_observable_low_residual_pelvis_rotation(tmp_path: Path) -> None:
    stream = make_stream(tmp_path)
    rank_one = rtmw_frame(8_000_000_000)
    pelvis = RTMW_SMPLX52_NAMES.index("pelvis")
    rank_one.rotation_constraint_rank[pelvis] = 1
    with pytest.raises(ReferenceUnavailable, match="pelvis.*rank 1"):
        stream._parse_frame(rank_one)

    stream = make_stream(tmp_path)
    high_residual = rtmw_frame(9_000_000_000)
    high_residual.rotation_residual_degrees[pelvis] = 35.1
    with pytest.raises(ReferenceUnavailable, match="pelvis.*residual"):
        stream._parse_frame(high_residual)


def test_rtmw_unreliable_wrist_rotation_uses_ancestor(tmp_path: Path) -> None:
    stream = make_stream(tmp_path)
    message = rtmw_frame(8_000_000_000)
    wrist = RTMW_SMPLX52_NAMES.index("right_wrist")
    turn_xyzw = np.asarray((0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)))
    local = np.asarray(message.local_joint_quaternions).reshape(-1, 4)
    global_ = np.asarray(message.global_joint_quaternions).reshape(-1, 4)
    local[wrist] = turn_xyzw
    for index in range(len(RTMW_SMPLX52_NAMES)):
        ancestor = index
        while ancestor >= 0 and ancestor != wrist:
            ancestor = RTMW_SMPLX52_PARENTS[ancestor]
        if ancestor == wrist:
            global_[index] = turn_xyzw
    message.local_joint_quaternions = local.reshape(-1).tolist()
    message.global_joint_quaternions = global_.reshape(-1).tolist()
    message.rotation_constraint_rank[wrist] = 1
    message.rotation_residual_degrees[wrist] = 35.1

    frame = stream._parse_frame(message)

    assert frame.sequence == message.frame_id
    np.testing.assert_allclose(
        frame.transforms["right_wrist"][1],
        (1.0, 0.0, 0.0, 0.0),
        atol=1e-7,
    )


def test_rtmw_accepts_rank_one_elbow_for_position_only_live_profile(
    tmp_path: Path,
) -> None:
    stream = make_stream(tmp_path)
    message = rtmw_frame(8_000_000_000)
    elbow = RTMW_SMPLX52_NAMES.index("left_elbow")
    message.rotation_constraint_rank[elbow] = 1

    frame = stream._parse_frame(message)

    assert frame.sequence == message.frame_id


@pytest.mark.parametrize(
    ("rotation_confidence", "rotation_residual"),
    [(0.05, 0.0), (0.8, 30.1)],
)
def test_rtmw_low_quality_optional_rotation_uses_ancestor(
    tmp_path: Path,
    rotation_confidence: float,
    rotation_residual: float,
) -> None:
    stream = make_stream(
        tmp_path,
        min_rotation_confidence=0.10,
        max_required_rotation_residual_degrees=30.0,
    )
    message = rtmw_frame(8_000_000_000)
    shoulder = RTMW_SMPLX52_NAMES.index("left_shoulder")
    turn_xyzw = np.asarray((0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)))
    local = np.asarray(message.local_joint_quaternions).reshape(-1, 4)
    global_ = np.asarray(message.global_joint_quaternions).reshape(-1, 4)
    local[shoulder] = turn_xyzw
    for index in range(len(RTMW_SMPLX52_NAMES)):
        ancestor = index
        while ancestor >= 0 and ancestor != shoulder:
            ancestor = RTMW_SMPLX52_PARENTS[ancestor]
        if ancestor == shoulder:
            global_[index] = turn_xyzw
    message.local_joint_quaternions = local.reshape(-1).tolist()
    message.global_joint_quaternions = global_.reshape(-1).tolist()
    message.rotation_confidence[shoulder] = rotation_confidence
    message.rotation_residual_degrees[shoulder] = rotation_residual

    frame = stream._parse_frame(message)

    np.testing.assert_allclose(
        frame.transforms["left_shoulder"][1],
        (1.0, 0.0, 0.0, 0.0),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        frame.transforms["left_elbow"][1],
        (np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)),
        atol=1e-7,
    )


def test_rtmw_rejects_root_relative_or_uncalibrated_stream(tmp_path: Path) -> None:
    config = rtmw_config(tmp_path)
    info = stream_info(
        coordinate_space="root_relative_opencv_x_right_y_down_z_forward",
        depth_provenance="rtmw_root_relative_only",
        metric_scale=False,
        global_translation=False,
    )

    with pytest.raises(ValueError, match="coordinate_space"):
        _validate_stream_info(info, config)


def test_rtmw_requires_versioned_anatomical_arm_and_palm_rotation_semantics(
    tmp_path: Path,
) -> None:
    config = rtmw_config(tmp_path)
    assert RTMW_ROTATION_SEMANTICS == (
        "derived_tree_consistent_landmark_ik;"
        "canonical_smplx_upper_body_projection_v1;"
        "anatomical_elbow_hinge_arm_chain_v2;"
        "palm_frame_wrist_orientation_v2;"
        "lower_body_hand_shape_and_unobserved_axial_twist_approximate"
    )
    _validate_stream_info(stream_info(), config)

    previous = stream_info(
        rotation_semantics=(
            "derived_tree_consistent_landmark_ik;"
            "canonical_smplx_upper_body_projection_v1;"
            "anatomical_elbow_hinge_upper_arm_frame_v1;"
            "palm_frame_forearm_twist_v1;"
            "lower_body_hand_shape_and_unobserved_axial_twist_approximate"
        )
    )
    with pytest.raises(ValueError, match="rotation_semantics"):
        _validate_stream_info(previous, config)


def test_rtmw_rejects_latency_mismatch_or_excess(tmp_path: Path) -> None:
    stream = make_stream(tmp_path)
    mismatch = rtmw_frame(8_000_000_000)
    mismatch.pipeline_latency_ms = 1.0
    with pytest.raises(ReferenceUnavailable, match="disagrees"):
        stream._parse_frame(mismatch)

    stream = make_stream(tmp_path, max_pipeline_latency_ms=5.0)
    with pytest.raises(ReferenceUnavailable, match="too high"):
        stream._parse_frame(rtmw_frame(9_000_000_000))


def test_rtmw_default_camera_rotation_maps_opencv_axes(tmp_path: Path) -> None:
    config = rtmw_config(
        tmp_path,
        camera_to_world_quaternion_xyzw=(0.5, -0.5, 0.5, -0.5),
    )
    adapter = RtmwCoordinateAdapter(config)
    positions = np.zeros((len(RTMW_SMPLX52_NAMES), 3))
    positions[1] = (1.0, 2.0, 3.0)
    quaternions = np.tile(
        np.asarray((0.0, 0.0, 0.0, 1.0)),
        (len(RTMW_SMPLX52_NAMES), 1),
    )

    world, _ = adapter.transform(positions, quaternions, track_id="1")

    assert world[1] == pytest.approx((3.0, -1.0, -2.0))
