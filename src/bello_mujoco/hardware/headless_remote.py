"""Internal zero-only RemoteCommandService for Linux vCAN policy tests."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import platform
import subprocess
import sys
from threading import Event
import time
from types import TracebackType
from typing import Callable, Iterator, Sequence

import grpc

from bello_mujoco.hardware import _remote_pb2


_LISTEN_ADDRESS = "127.0.0.1:50054"
_REQUIRED_VCAN_INTERFACES = ("vcan0", "vcan1")
_PUBLISH_PERIOD_SECONDS = 0.05

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
InterfaceVerifier = Callable[[], None]


def verify_vcan_interfaces(
    command_runner: CommandRunner = subprocess.run,
) -> None:
    """Fail unless both required Linux interfaces are actual vCAN links."""

    if platform.system() != "Linux":
        raise RuntimeError("headless remote service requires Linux")
    try:
        result = command_runner(
            ["ip", "-details", "-json", "link", "show"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "cannot verify vCAN interfaces: ip command not found"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("timed out while inspecting Linux links with ip") from error
    except OSError as error:
        raise RuntimeError(f"cannot inspect Linux links with ip: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit status {result.returncode}"
        raise RuntimeError(f"cannot inspect Linux links with ip: {detail}")
    try:
        links = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError) as error:
        raise RuntimeError(
            "ip returned invalid JSON while inspecting Linux links"
        ) from error
    if not isinstance(links, list):
        raise RuntimeError("ip returned invalid link data: expected a JSON array")

    links_by_name: dict[str, dict[str, object]] = {}
    for link in links:
        if not isinstance(link, dict):
            continue
        name = link.get("ifname")
        if isinstance(name, str):
            links_by_name[name] = link
    for interface in _REQUIRED_VCAN_INTERFACES:
        link = links_by_name.get(interface)
        if link is None:
            raise RuntimeError(f"required vCAN interface {interface!r} is missing")
        link_info = link.get("linkinfo")
        kind = link_info.get("info_kind") if isinstance(link_info, dict) else None
        if kind != "vcan":
            raise RuntimeError(
                f"required interface {interface!r} is not vCAN (kind={kind!r})"
            )
        flags = link.get("flags")
        if not isinstance(flags, list) or "UP" not in flags:
            raise RuntimeError(f"required vCAN interface {interface!r} is down")


class HeadlessRemoteCommandService:
    """A loopback-only gRPC service that can emit only armed zero commands."""

    def __init__(
        self,
        interface_verifier: InterfaceVerifier | None = None,
    ) -> None:
        verifier = (
            verify_vcan_interfaces if interface_verifier is None else interface_verifier
        )
        verifier()
        self._stopped = Event()
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._server = grpc.server(
            self._executor,
            options=(("grpc.so_reuseport", 0),),
        )
        handler = grpc.method_handlers_generic_handler(
            "bello.remote.v1.RemoteCommandService",
            {
                "WatchCommand": grpc.unary_stream_rpc_method_handler(
                    self._watch_command,
                    request_deserializer=(_remote_pb2.WatchCommandRequest.FromString),
                    response_serializer=(
                        _remote_pb2.WatchCommandResponse.SerializeToString
                    ),
                )
            },
        )
        self._server.add_generic_rpc_handlers((handler,))
        try:
            self.port = self._server.add_insecure_port(_LISTEN_ADDRESS)
        except RuntimeError as error:
            self._executor.shutdown(wait=False, cancel_futures=True)
            raise RuntimeError(
                f"cannot bind headless RemoteCommandService to {_LISTEN_ADDRESS!r}"
            ) from error
        if self.port == 0:
            self._executor.shutdown(wait=False, cancel_futures=True)
            raise RuntimeError(
                f"cannot bind headless RemoteCommandService to {_LISTEN_ADDRESS!r}"
            )
        self._started = False
        self._closed = False

    def __enter__(self) -> HeadlessRemoteCommandService:
        self.start()
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("headless RemoteCommandService is closed")
        if not self._started:
            self._server.start()
            self._started = True

    def wait_for_termination(self) -> None:
        if not self._started:
            raise RuntimeError("headless RemoteCommandService has not started")
        self._server.wait_for_termination()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stopped.set()
        if self._started:
            self._server.stop(grace=0).wait(timeout=2.0)
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _watch_command(
        self,
        _request: _remote_pb2.WatchCommandRequest,
        context: grpc.ServicerContext,
    ) -> Iterator[_remote_pb2.WatchCommandResponse]:
        sequence = 0
        while context.is_active() and not self._stopped.is_set():
            sequence += 1
            yield _remote_pb2.WatchCommandResponse(
                sequence=sequence,
                sample_monotonic_ns=time.monotonic_ns(),
                controller_connected=True,
                controller_name="headless-neutral",
                intent_armed=True,
                normalized_forward=0.0,
                normalized_lateral=0.0,
                normalized_yaw=0.0,
                abort_active=False,
            )
            self._stopped.wait(_PUBLISH_PERIOD_SECONDS)


def run() -> None:
    """Verify the vCAN sandbox, then serve neutral commands until stopped."""

    with HeadlessRemoteCommandService() as service:
        print(
            f"Serving armed neutral-only commands on {_LISTEN_ADDRESS}; "
            "press Ctrl-C to stop.",
            flush=True,
        )
        service.wait_for_termination()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        run()
    except KeyboardInterrupt:
        return 130
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
