"""Nonblocking client for the visualizer's normalized remote-command stream."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time

import grpc

from bello_mujoco.hardware import _remote_pb2


@dataclass(frozen=True)
class RemoteCommandConfig:
    endpoint: str = "http://127.0.0.1:50054"
    staleness_ms: int = 200

    def __post_init__(self) -> None:
        if not self.endpoint.strip() or self.endpoint.strip() != self.endpoint:
            raise ValueError(
                "remote endpoint must be non-empty without surrounding whitespace"
            )
        if not self.endpoint.startswith("http://"):
            raise ValueError(
                "remote endpoint must use plaintext http:// in protocol v1"
            )
        if isinstance(self.staleness_ms, bool) or not isinstance(
            self.staleness_ms, int
        ):
            raise TypeError("remote staleness_ms must be an integer")
        if self.staleness_ms <= 0:
            raise ValueError("remote staleness_ms must be positive")


@dataclass(frozen=True)
class RemoteCommandInput:
    normalized_command: tuple[float, float, float]
    service_connected: bool
    controller_connected: bool
    intent_armed: bool
    abort_active: bool
    service_connection_changed: bool = False
    controller_connection_changed: bool = False
    controller_name: str | None = None
    sample_age_ms: float | None = None


@dataclass(frozen=True)
class _ReceivedCommand:
    normalized_command: tuple[float, float, float]
    controller_connected: bool
    intent_armed: bool
    controller_name: str | None
    received_at: float


class RemoteCommandSource:
    def __init__(self, config: RemoteCommandConfig) -> None:
        self.config = config
        self._staleness_seconds = config.staleness_ms / 1000.0
        self._condition = threading.Condition()
        self._closed = threading.Event()
        self._active_call: grpc.Call | None = None
        self._received: _ReceivedCommand | None = None
        self._fatal_error: RuntimeError | None = None
        self._abort_latched = False
        self._rearm_required = False
        self._reported_service_connected = False
        self._reported_controller_connected = False
        self._thread = threading.Thread(
            target=self._run,
            name="bello-remote-command",
            daemon=True,
        )
        self._thread.start()

    def wait_until_ready(self, timeout_seconds: float) -> RemoteCommandInput:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise ValueError("remote readiness timeout must be positive and finite")
        deadline = time.monotonic() + timeout_seconds
        while True:
            with self._condition:
                while (
                    self._received is None
                    and self._fatal_error is None
                    and not self._closed.is_set()
                ):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        raise TimeoutError(
                            "timed out waiting for a fresh visualizer remote-command "
                            f"sample from {self.config.endpoint}"
                        )
                    self._condition.wait(remaining)
                if self._closed.is_set():
                    raise RuntimeError("remote command source closed while waiting")
                result = self._poll_locked()
            if result.service_connected:
                return result

    def poll(self) -> RemoteCommandInput:
        with self._condition:
            return self._poll_locked()

    def _poll_locked(self) -> RemoteCommandInput:
        if self._fatal_error is not None:
            raise self._fatal_error
        now = time.monotonic()
        received = self._received
        fresh = (
            received is not None
            and now - received.received_at <= self._staleness_seconds
        )
        service_connected = fresh
        controller_connected = bool(
            fresh and received is not None and received.controller_connected
        )
        intent_armed = bool(fresh and received is not None and received.intent_armed)
        normalized = (
            received.normalized_command
            if controller_connected and intent_armed and received is not None
            else (0.0, 0.0, 0.0)
        )
        service_changed = service_connected != self._reported_service_connected
        controller_changed = controller_connected != self._reported_controller_connected
        self._reported_service_connected = service_connected
        self._reported_controller_connected = controller_connected
        result = RemoteCommandInput(
            normalized_command=normalized,
            service_connected=service_connected,
            controller_connected=controller_connected,
            intent_armed=intent_armed,
            abort_active=self._abort_latched,
            service_connection_changed=service_changed,
            controller_connection_changed=controller_changed,
            controller_name=(
                received.controller_name
                if controller_connected and received is not None
                else None
            ),
            sample_age_ms=(
                max(0.0, now - received.received_at) * 1000.0
                if received is not None
                else None
            ),
        )
        if not fresh:
            if received is not None:
                self._rearm_required = True
            self._received = None
        return result

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        with self._condition:
            call = self._active_call
            self._condition.notify_all()
        if call is not None:
            call.cancel()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        channel = grpc.insecure_channel(
            self.config.endpoint.removeprefix("http://")
        )
        method = channel.unary_stream(
            "/bello.remote.v1.RemoteCommandService/WatchCommand",
            request_serializer=_remote_pb2.WatchCommandRequest.SerializeToString,
            response_deserializer=_remote_pb2.WatchCommandResponse.FromString,
        )
        backoff_seconds = 0.1
        try:
            while not self._closed.is_set():
                call = method(_remote_pb2.WatchCommandRequest(), wait_for_ready=True)
                with self._condition:
                    self._active_call = call
                    self._received = None
                    self._condition.notify_all()
                    close_requested = self._closed.is_set()
                if close_requested:
                    call.cancel()
                    with self._condition:
                        if self._active_call is call:
                            self._active_call = None
                    return
                last_sequence: int | None = None
                client_armed = False
                try:
                    for response in call:
                        normalized = _validated_response(response, last_sequence)
                        last_sequence = int(response.sequence)
                        with self._condition:
                            if self._rearm_required:
                                client_armed = False
                                self._rearm_required = False
                            if (
                                not response.controller_connected
                                or not response.intent_armed
                            ):
                                client_armed = False
                            if (
                                not client_armed
                                and response.controller_connected
                                and response.intent_armed
                                and normalized == (0.0, 0.0, 0.0)
                            ):
                                client_armed = True
                            effective = (
                                normalized if client_armed else (0.0, 0.0, 0.0)
                            )
                            if response.abort_active:
                                self._abort_latched = True
                            self._received = _ReceivedCommand(
                                normalized_command=effective,
                                controller_connected=bool(
                                    response.controller_connected
                                ),
                                intent_armed=bool(
                                    response.intent_armed and client_armed
                                ),
                                controller_name=(
                                    response.controller_name
                                    if response.controller_connected
                                    else None
                                ),
                                received_at=time.monotonic(),
                            )
                            self._condition.notify_all()
                        backoff_seconds = 0.1
                        if self._closed.is_set():
                            break
                except grpc.RpcError:
                    pass
                except (TypeError, ValueError) as error:
                    with self._condition:
                        self._fatal_error = RuntimeError(
                            f"invalid visualizer remote-command stream: {error}"
                        )
                        self._condition.notify_all()
                    return
                finally:
                    with self._condition:
                        if self._active_call is call:
                            self._active_call = None
                        self._received = None
                        self._condition.notify_all()
                if self._closed.wait(backoff_seconds):
                    break
                backoff_seconds = min(1.0, backoff_seconds * 2.0)
        finally:
            channel.close()


def _validated_response(
    response: _remote_pb2.WatchCommandResponse,
    last_sequence: int | None,
) -> tuple[float, float, float]:
    sequence = int(response.sequence)
    if last_sequence is not None and sequence <= last_sequence:
        raise ValueError(
            f"sequence did not increase: previous={last_sequence}, current={sequence}"
        )
    normalized = (
        float(response.normalized_forward),
        float(response.normalized_lateral),
        float(response.normalized_yaw),
    )
    if not all(math.isfinite(value) and -1.0 <= value <= 1.0 for value in normalized):
        raise ValueError(
            f"normalized axes must be finite and in [-1, 1], got {normalized}"
        )
    if not response.controller_connected and normalized != (0.0, 0.0, 0.0):
        raise ValueError("disconnected controller supplied nonzero intent")
    if not response.intent_armed and normalized != (0.0, 0.0, 0.0):
        raise ValueError("unarmed controller supplied nonzero intent")
    return normalized
