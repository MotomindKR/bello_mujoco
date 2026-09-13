from __future__ import annotations

import json
import subprocess

import grpc
import pytest

from bello_mujoco.hardware import _remote_pb2
from bello_mujoco.hardware import headless_remote


def _completed_process(links: object) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["ip"],
        returncode=0,
        stdout=json.dumps(links),
        stderr="",
    )


def test_verify_vcan_interfaces_uses_structured_link_kinds(monkeypatch) -> None:
    monkeypatch.setattr(headless_remote.platform, "system", lambda: "Linux")
    calls: list[tuple[object, dict[str, object]]] = []

    def command_runner(command, **kwargs):
        calls.append((command, kwargs))
        return _completed_process(
            [
                {
                    "ifname": "vcan0",
                    "flags": ["UP"],
                    "linkinfo": {"info_kind": "vcan"},
                },
                {
                    "ifname": "vcan1",
                    "flags": ["UP"],
                    "linkinfo": {"info_kind": "vcan"},
                },
            ]
        )

    headless_remote.verify_vcan_interfaces(command_runner)

    assert calls == [
        (
            ["ip", "-details", "-json", "link", "show"],
            {
                "check": False,
                "capture_output": True,
                "text": True,
                "timeout": 5.0,
            },
        )
    ]


@pytest.mark.parametrize(
    ("links", "message"),
    [
        (
            [
                {
                    "ifname": "vcan0",
                    "flags": ["UP"],
                    "linkinfo": {"info_kind": "vcan"},
                }
            ],
            "vcan1.*missing",
        ),
        (
            [
                {
                    "ifname": "vcan0",
                    "flags": ["UP"],
                    "linkinfo": {"info_kind": "vcan"},
                },
                {
                    "ifname": "vcan1",
                    "flags": ["UP"],
                    "linkinfo": {"info_kind": "can"},
                },
            ],
            "vcan1.*not vCAN",
        ),
        (
            [
                {
                    "ifname": "vcan0",
                    "flags": ["UP"],
                    "linkinfo": {"info_kind": "vcan"},
                },
                {
                    "ifname": "vcan1",
                    "flags": [],
                    "linkinfo": {"info_kind": "vcan"},
                },
            ],
            "vcan1.*down",
        ),
    ],
)
def test_verify_vcan_interfaces_rejects_missing_or_wrong_links(
    monkeypatch, links, message
) -> None:
    monkeypatch.setattr(headless_remote.platform, "system", lambda: "Linux")

    with pytest.raises(RuntimeError, match=message):
        headless_remote.verify_vcan_interfaces(
            lambda *_args, **_kwargs: _completed_process(links)
        )


def test_verify_vcan_interfaces_rejects_non_linux(monkeypatch) -> None:
    monkeypatch.setattr(headless_remote.platform, "system", lambda: "Darwin")

    with pytest.raises(RuntimeError, match="requires Linux"):
        headless_remote.verify_vcan_interfaces()


def test_service_stream_is_armed_and_permanently_neutral(monkeypatch) -> None:
    monkeypatch.setattr(headless_remote, "_LISTEN_ADDRESS", "127.0.0.1:0")
    with headless_remote.HeadlessRemoteCommandService(lambda: None) as service:
        channel = grpc.insecure_channel(f"127.0.0.1:{service.port}")
        method = channel.unary_stream(
            "/bello.remote.v1.RemoteCommandService/WatchCommand",
            request_serializer=_remote_pb2.WatchCommandRequest.SerializeToString,
            response_deserializer=_remote_pb2.WatchCommandResponse.FromString,
        )
        call = method(_remote_pb2.WatchCommandRequest())
        try:
            responses = (next(call), next(call))
        finally:
            call.cancel()
            channel.close()
        assert [response.sequence for response in responses] == [1, 2]
        for response in responses:
            assert response.controller_connected
            assert response.intent_armed
            assert response.controller_name == "headless-neutral"
            assert response.normalized_forward == 0.0
            assert response.normalized_lateral == 0.0
            assert response.normalized_yaw == 0.0
            assert not response.abort_active


def test_service_uses_loopback_fixed_port() -> None:
    assert headless_remote._LISTEN_ADDRESS == "127.0.0.1:50054"


def test_service_rejects_a_second_producer_on_the_fixed_port(monkeypatch) -> None:
    monkeypatch.setattr(headless_remote, "_LISTEN_ADDRESS", "127.0.0.1:0")
    with headless_remote.HeadlessRemoteCommandService(lambda: None) as first:
        monkeypatch.setattr(
            headless_remote,
            "_LISTEN_ADDRESS",
            f"127.0.0.1:{first.port}",
        )
        with pytest.raises(RuntimeError, match="cannot bind"):
            headless_remote.HeadlessRemoteCommandService(lambda: None)


def test_main_fails_closed_before_constructing_server(monkeypatch, capsys) -> None:
    def reject_interfaces() -> None:
        raise RuntimeError("not a vCAN sandbox")

    constructed = False

    def construct_service():
        nonlocal constructed
        constructed = True
        raise AssertionError("must not construct server")

    monkeypatch.setattr(headless_remote, "verify_vcan_interfaces", reject_interfaces)
    monkeypatch.setattr(headless_remote.grpc, "server", construct_service)

    assert headless_remote.main([]) == 1
    assert not constructed
    assert "not a vCAN sandbox" in capsys.readouterr().err
