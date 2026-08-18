from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import time

import grpc
import pytest

from bello_mujoco.hardware import _remote_pb2
from bello_mujoco.hardware import remote as remote_module
from bello_mujoco.hardware.remote import RemoteCommandConfig, RemoteCommandSource


class RemoteService:
    def __init__(self) -> None:
        self._responses: queue.Queue[object] = queue.Queue()
        self.calls = 0
        self._server = grpc.server(ThreadPoolExecutor(max_workers=4))
        handler = grpc.unary_stream_rpc_method_handler(
            self._watch,
            request_deserializer=_remote_pb2.WatchCommandRequest.FromString,
            response_serializer=_remote_pb2.WatchCommandResponse.SerializeToString,
        )
        self._server.add_generic_rpc_handlers(
            (
                grpc.method_handlers_generic_handler(
                    "bello.remote.v1.RemoteCommandService",
                    {"WatchCommand": handler},
                ),
            )
        )
        port = self._server.add_insecure_port("127.0.0.1:0")
        assert port != 0
        self.endpoint = f"http://127.0.0.1:{port}"

    def __enter__(self) -> RemoteService:
        self._server.start()
        return self

    def __exit__(self, *_args) -> None:
        self._server.stop(0).wait()

    def send(self, sequence: int, axes=(0.0, 0.0, 0.0), **kwargs) -> None:
        self._responses.put(
            _remote_pb2.WatchCommandResponse(
                sequence=sequence,
                sample_monotonic_ns=sequence * 1_000_000,
                normalized_forward=axes[0],
                normalized_lateral=axes[1],
                normalized_yaw=axes[2],
                **kwargs,
            )
        )

    def end_stream(self) -> None:
        self._responses.put(None)

    def _watch(self, _request, context):
        self.calls += 1
        while context.is_active():
            try:
                response = self._responses.get(timeout=0.05)
            except queue.Empty:
                continue
            if response is None:
                return
            yield response


def wait_for(check, timeout: float = 1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(0.005)
    raise AssertionError("condition did not become true")


def test_remote_requires_neutral_and_latches_abort() -> None:
    with RemoteService() as service:
        source = RemoteCommandSource(RemoteCommandConfig(endpoint=service.endpoint))
        try:
            service.send(
                1,
                (0.8, 0.0, 0.0),
                controller_connected=True,
                controller_name="controller",
                intent_armed=True,
                abort_active=True,
            )
            held = source.wait_until_ready(1.0)
            assert held.normalized_command == (0.0, 0.0, 0.0)
            assert not held.intent_armed
            assert held.abort_active

            service.send(
                2,
                controller_connected=True,
                controller_name="controller",
                intent_armed=True,
            )
            wait_for(lambda: source.poll().intent_armed)
            service.send(
                3,
                (0.5, -0.25, 1.0),
                controller_connected=True,
                controller_name="controller",
                intent_armed=True,
            )
            moving = wait_for(
                lambda: (
                    value
                    if (value := source.poll()).normalized_command[0] == 0.5
                    else None
                )
            )
            assert moving.normalized_command == (0.5, -0.25, 1.0)
            assert moving.abort_active
        finally:
            source.close()


def test_stale_stream_commands_zero_and_recovers_in_place_requires_neutral() -> None:
    with RemoteService() as service:
        source = RemoteCommandSource(
            RemoteCommandConfig(endpoint=service.endpoint, staleness_ms=30)
        )
        try:
            service.send(
                1,
                controller_connected=True,
                controller_name="controller",
                intent_armed=True,
            )
            source.wait_until_ready(1.0)
            service.send(
                2,
                (0.5, 0.0, 0.0),
                controller_connected=True,
                controller_name="controller",
                intent_armed=True,
            )
            wait_for(lambda: source.poll().normalized_command[0] == 0.5)
            stale = wait_for(
                lambda: (
                    value if not (value := source.poll()).service_connected else None
                )
            )
            assert stale.normalized_command == (0.0, 0.0, 0.0)
            assert service.calls == 1

            service.send(
                3,
                (0.5, 0.0, 0.0),
                controller_connected=True,
                controller_name="controller",
                intent_armed=True,
            )
            held = source.wait_until_ready(1.0)
            assert held.normalized_command == (0.0, 0.0, 0.0)
            assert not held.intent_armed
            service.send(
                4,
                controller_connected=True,
                controller_name="controller",
                intent_armed=True,
            )
            wait_for(lambda: source.poll().intent_armed)
            assert service.calls == 1
        finally:
            source.close()


def test_rpc_termination_reconnects_on_existing_channel(monkeypatch) -> None:
    created_channels = 0
    original_insecure_channel = grpc.insecure_channel

    def counted_channel(target):
        nonlocal created_channels
        created_channels += 1
        return original_insecure_channel(target)

    monkeypatch.setattr(remote_module.grpc, "insecure_channel", counted_channel)
    with RemoteService() as service:
        source = RemoteCommandSource(RemoteCommandConfig(endpoint=service.endpoint))
        try:
            service.send(
                1,
                controller_connected=True,
                controller_name="controller",
                intent_armed=True,
            )
            source.wait_until_ready(1.0)
            service.end_stream()
            wait_for(lambda: service.calls >= 2)
            service.send(
                1,
                controller_connected=True,
                controller_name="controller",
                intent_armed=True,
            )

            reconnected = source.wait_until_ready(1.0)

            assert reconnected.service_connected
            assert reconnected.controller_connected
            assert reconnected.intent_armed
            assert created_channels == 1
        finally:
            source.close()


def test_invalid_sequence_is_terminal() -> None:
    with RemoteService() as service:
        source = RemoteCommandSource(RemoteCommandConfig(endpoint=service.endpoint))
        try:
            service.send(1)
            source.wait_until_ready(1.0)
            service.send(1)

            def invalid():
                try:
                    source.poll()
                except RuntimeError as error:
                    return error
                return None

            error = wait_for(invalid)
            assert "sequence did not increase" in str(error)
        finally:
            source.close()


def test_remote_config_rejects_tls_and_nonpositive_staleness() -> None:
    with pytest.raises(ValueError, match="plaintext"):
        RemoteCommandSource(RemoteCommandConfig(endpoint="https://remote:50054"))
    with pytest.raises(ValueError, match="staleness_ms"):
        RemoteCommandSource(RemoteCommandConfig(staleness_ms=0))
    with pytest.raises(TypeError, match="integer"):
        RemoteCommandConfig(staleness_ms=True)


def test_close_cancels_a_call_created_during_shutdown(monkeypatch) -> None:
    entered_channel_creation = threading.Event()
    release_channel_creation = threading.Event()
    original_insecure_channel = grpc.insecure_channel

    def delayed_channel(target):
        entered_channel_creation.set()
        assert release_channel_creation.wait(timeout=1.0)
        return original_insecure_channel(target)

    monkeypatch.setattr(remote_module.grpc, "insecure_channel", delayed_channel)
    source = RemoteCommandSource(RemoteCommandConfig(endpoint="http://127.0.0.1:1"))
    assert entered_channel_creation.wait(timeout=1.0)

    close_thread = threading.Thread(target=source.close)
    close_thread.start()
    assert wait_for(source._closed.is_set)
    release_channel_creation.set()
    close_thread.join(timeout=1.0)

    assert not close_thread.is_alive()
    assert not source._thread.is_alive()
