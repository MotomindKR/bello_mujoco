"""Latest-only PICO gRPC transport shared by raw body and calibrated hands."""

from collections.abc import Callable
from dataclasses import dataclass
import math
from pathlib import Path
import threading
import time
from typing import Protocol

import grpc

from bello_mujoco.hardware import _smplx_pb2 as pb
from bello_mujoco.hardware.reference import ReferenceDiscontinuity, ReferenceUnavailable


@dataclass(frozen=True)
class PicoGrpcConfig:
    endpoint: str
    tls_root_certificate_path: Path | None = None
    tls_server_name_override: str | None = None
    connect_timeout_seconds: float = 5.0
    rpc_timeout_seconds: float = 2.0
    max_source_age_ms: int = 500
    max_source_interval_seconds: float = 0.70
    transport_latency_budget_ms: float = 10.0
    max_rate_hz: float = 60.0

    def __post_init__(self) -> None:
        if not self.endpoint or "://" in self.endpoint:
            raise ValueError("PICO endpoint must be a gRPC host:port target")
        for name in (
            "connect_timeout_seconds",
            "rpc_timeout_seconds",
            "max_source_age_ms",
            "max_source_interval_seconds",
            "max_rate_hz",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.max_rate_hz > 120:
            raise ValueError("max_rate_hz must not exceed 120")
        if (
            not math.isfinite(self.transport_latency_budget_ms)
            or self.transport_latency_budget_ms < 0
        ):
            raise ValueError(
                "transport_latency_budget_ms must be finite and nonnegative"
            )
        if self.tls_server_name_override and not self.tls_root_certificate_path:
            raise ValueError("TLS server-name override requires a CA file")


class TrackedFrame(Protocol):
    sequence: int
    source_timestamp_ns: int
    track_id: str

    def age_ms(self) -> float: ...


class PicoGrpcStream[Frame: TrackedFrame]:
    """Bounded latest-frame delivery with explicit outage and epoch barriers."""

    def __init__(
        self,
        config: PicoGrpcConfig,
        parser: Callable[[pb.PoseFrame, PicoGrpcConfig], Frame],
    ):
        self.config = config
        self.parser = parser
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._latest: Frame | None = None
        self._delivered: Frame | None = None
        self._pending: ReferenceUnavailable | None = None
        self._terminal: Exception | None = None
        self._channel = None
        self._thread = None
        self._watchdog = None
        self._call = None
        self._last_delivery = 0.0
        self._received_on_call = False
        self._last_identity = None
        self._last_sequence = None
        self._last_timestamp = None

    def __enter__(self):
        if self._thread is not None:
            raise RuntimeError("PICO stream is already running")
        options = [("grpc.max_receive_message_length", 1024 * 1024)]
        if self.config.tls_root_certificate_path:
            credentials = grpc.ssl_channel_credentials(
                self.config.tls_root_certificate_path.read_bytes()
            )
            if self.config.tls_server_name_override:
                options.append(
                    (
                        "grpc.ssl_target_name_override",
                        self.config.tls_server_name_override,
                    )
                )
            self._channel = grpc.secure_channel(
                self.config.endpoint, credentials, options=options
            )
        else:
            self._channel = grpc.insecure_channel(self.config.endpoint, options=options)
        self._thread = threading.Thread(
            target=self._consume, name="pico-grpc", daemon=True
        )
        self._watchdog = threading.Thread(
            target=self._monitor, name="pico-liveness", daemon=True
        )
        self._thread.start()
        self._watchdog.start()
        return self

    def __exit__(self, *_exception):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._channel:
            self._channel.close()
        for thread in (self._thread, self._watchdog):
            if thread:
                thread.join(timeout=self.config.connect_timeout_seconds + 1)
                if thread.is_alive():
                    raise RuntimeError("PICO transport worker did not stop")
        return False

    def _unavailable(self, message, *, discontinuity=False):
        error_type = ReferenceDiscontinuity if discontinuity else ReferenceUnavailable
        with self._condition:
            self._latest = None
            if self._pending is None or discontinuity:
                self._pending = error_type(message)
            self._condition.notify_all()

    def _consume(self):
        assert self._channel is not None
        subscribe = self._channel.unary_stream(
            "/pico.smplx.v1.PoseStream/StreamPoses",
            request_serializer=pb.StreamRequest.SerializeToString,
            response_deserializer=pb.PoseFrame.FromString,
        )
        retryable = {
            grpc.StatusCode.UNAVAILABLE,
            grpc.StatusCode.CANCELLED,
            grpc.StatusCode.DEADLINE_EXCEEDED,
        }
        while not self._stop.is_set():
            try:
                call = subscribe(
                    pb.StreamRequest(max_rate_hz=self.config.max_rate_hz),
                    wait_for_ready=True,
                )
                with self._condition:
                    self._call = call
                    self._last_delivery = time.monotonic()
                    self._received_on_call = False
                for message in call:
                    if self._stop.is_set():
                        return
                    with self._condition:
                        self._last_delivery = time.monotonic()
                        self._received_on_call = True
                    try:
                        frame = self.parser(message, self.config)
                    except ReferenceUnavailable as error:
                        self._unavailable(str(error))
                        continue
                    if frame.track_id != self._last_identity or (
                        self._last_sequence is not None
                        and frame.sequence < self._last_sequence
                    ):
                        if self._last_identity is not None:
                            self._unavailable(
                                "PICO tracking epoch changed", discontinuity=True
                            )
                        self._last_sequence = self._last_timestamp = None
                    if frame.sequence == self._last_sequence:
                        continue  # Duplicate delivery must not refresh observation age.
                    if self._last_timestamp is not None:
                        interval = (
                            frame.source_timestamp_ns - self._last_timestamp
                        ) / 1e9
                        if not 0 < interval <= self.config.max_source_interval_seconds:
                            self._unavailable(
                                "PICO source timestamp discontinuity",
                                discontinuity=True,
                            )
                    self._last_identity = frame.track_id
                    self._last_sequence = frame.sequence
                    self._last_timestamp = frame.source_timestamp_ns
                    with self._condition:
                        self._latest = frame
                        self._condition.notify_all()
                if not self._stop.is_set():
                    self._unavailable("PICO subscription ended", discontinuity=True)
            except grpc.RpcError as error:
                if error.code() not in retryable:
                    with self._condition:
                        self._terminal = error
                        self._condition.notify_all()
                    return
                if not self._stop.is_set():
                    self._unavailable(
                        "PICO stream disconnected; reconnecting", discontinuity=True
                    )
            except Exception as error:
                with self._condition:
                    self._terminal = error
                    self._condition.notify_all()
                return
            finally:
                with self._condition:
                    self._call = None
            self._last_identity = self._last_sequence = self._last_timestamp = None
            self._stop.wait(0.1)

    def _monitor(self):
        while not self._stop.wait(0.1):
            with self._condition:
                call, elapsed = self._call, time.monotonic() - self._last_delivery
                timeout = (
                    self.config.rpc_timeout_seconds
                    if self._received_on_call
                    else self.config.connect_timeout_seconds
                )
            if call is not None and elapsed > timeout:
                self._unavailable("PICO stream stalled", discontinuity=True)
                call.cancel()

    def wait_next(self, *, after_sequence, timeout_seconds):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("poll timeout must be positive and finite")
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while not self._stop.is_set():
                if self._terminal:
                    raise RuntimeError("PICO stream failed") from self._terminal
                if self._pending:
                    error, self._pending = self._pending, None
                    raise error
                frame = self._latest
                if frame is not None:
                    if frame.age_ms() > self.config.max_source_age_ms:
                        self._latest = None
                        raise ReferenceUnavailable("PICO latest frame became stale")
                    if frame is not self._delivered and (
                        after_sequence is None or frame.sequence > after_sequence
                    ):
                        self._delivered = frame
                        return frame
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
        return None
