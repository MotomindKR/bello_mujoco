"""Operator-assisted lowering through the real runner, using loopback hardware."""

import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import pytest
import yaml

from bello_mujoco.hardware import _operator_pb2 as pb
from native_operator import OperatorClient, operator_endpoint
from test_native_control import bundle, start_services
from test_native_control import hardware_proto as hardware_proto
from test_native_startup import AssistedHardware

pytestmark = pytest.mark.integration


@pytest.fixture
def sit_runner(rust_policy, hardware_proto, tmp_path):
    _, _, source = rust_policy
    names = [
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in ("hip_pitch", "knee", "ankle_pitch", "ankle_roll")
    ]
    nominal = np.array([-0.15, 0.45, -0.3, 0.0] * 2)
    mapping = np.eye(8)
    for i in (2, 6):
        mapping[i : i + 2, i : i + 2] = [[1, 1], [1, -1]]
    fake = AssistedHardware(hardware_proto, names=names, positions=mapping @ nominal)
    server, pool, endpoint = start_services(fake)
    policy = tmp_path / "policy.zip"
    policy.write_bytes(
        bundle(
            source,
            metadata_updates=dict(
                coordinates=names,
                joint_names=names,
                default_position=(mapping @ nominal).tolist(),
                default_joint_position=nominal.tolist(),
                kp=[2.0] * 8,
                kd=[0.1] * 8,
                joint_from_motor=np.linalg.inv(mapping).tolist(),
                motor_from_joint=mapping.tolist(),
                joint_limits=[[-3.0, 3.0]] * 8,
                motor_limits=[[-3.0, 3.0]] * 8,
                position_step_limits=[0.02] * 8,
            ),
        )
    )
    settings = yaml.safe_load(
        (
            Path(__file__).parents[1] / "rust/bello-policy/config/hardware.yaml"
        ).read_text()
    )
    address = operator_endpoint()
    policy_log = tmp_path / "policy.jsonl"
    settings.update(endpoint=endpoint, groups=["limb"])
    settings["control"].update(
        operator_listen=address,
        ramp_seconds=0.1,
        policy_log_path=str(policy_log),
        lying_start=dict(
            hip_pitch_radians=-0.35,
            knee_radians=0.85,
            straighten_speed_rps=0.1,
            ankle_engage_seconds=2.0,
        ),
    )
    config = tmp_path / "control.json"
    config.write_text(json.dumps(settings))
    clients = []

    def connect(**kwargs):
        client = OperatorClient(address, **kwargs)
        clients.append(client)
        return client

    with (tmp_path / "runner.log").open("w+") as log:
        process = subprocess.Popen(
            [
                os.environ["BELLO_POLICY_BIN"],
                "--policy",
                str(policy),
                "--source",
                str(source),
                "--hardware-config",
                str(config),
                "--arm",
            ],
            stdout=log,
            stderr=log,
        )
        try:
            client = connect()
            client.stand()
            client.select("Sit")
            client.press(1.1, a=True)
            client.wait(
                lambda s: s.mode == pb.MODE_SIT and s.phase == "awaiting support"
            )
            yield fake, client, connect, policy_log
        finally:
            for client in clients:
                client.close()
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
            fake.closed.set()
            server.stop(0).wait()
            pool.shutdown(wait=True)
            log.seek(0)
            print(log.read())


def bend(client):
    client.press(1.1, a=True)
    client.wait(lambda s: s.phase == "upright")
    client.buttons = {"forward": -1.0}
    client.wait(lambda s: s.phase == "bent")
    client.buttons = {}
    time.sleep(0.1)


def command_frame(fake):
    return [
        (c.position, c.kp, c.kd, c.torque_feedforward) for c in fake.command_frames[-1]
    ]


def test_sit_completion_requires_separate_support_confirmations(
    sit_runner, hardware_proto
):
    fake, client, _, log = sit_runner
    # Stand continues balancing until a separate support confirmation.
    assert all(c.kp == 2 for c in fake.command_frames[-1])
    client.buttons = {"a": True}
    client.wait(lambda s: s.phase == "settling to hold")
    fake.orientation = hardware_proto.Quaternion(y=-(2**-0.5), w=2**-0.5)
    client.wait(lambda s: s.phase == "upright")
    time.sleep(1.2)
    assert client.status.phase == "upright"
    client.buttons = {}
    time.sleep(0.1)
    mark = len(fake.command_frames)
    client.buttons = {"forward": -1.0}
    client.wait(lambda s: s.phase == "bent")
    client.buttons = {}
    time.sleep(0.1)
    frames = fake.command_frames[mark:]
    for frame in frames:
        assert all(
            frame[i].kp == frame[i].kd == frame[i].torque_feedforward == 0
            for i in (2, 3, 6, 7)
        )
    legs = np.array([[frame[i].position for i in (0, 1, 4, 5)] for frame in frames])
    assert np.max(np.abs(np.diff(legs, axis=0))) <= 0.002001
    np.testing.assert_allclose(legs[-1], [-0.35, 0.85] * 2, atol=1e-6)
    client.buttons = {"a": True}
    client.wait(lambda s: s.phase == "lowering onto back")
    time.sleep(1.2)
    assert client.status.phase == "lowering onto back"
    assert "disable" not in fake.calls
    assert fake.command_frames[-1][0].kp == 2
    client.buttons = {}
    time.sleep(0.1)
    client.press(1.1, a=True)
    client.wait(lambda s: s.mode == pb.MODE_DISARMED)
    assert fake.state == hardware_proto.GROUP_STATE_IDLE
    assert all(
        c.kp == c.kd == c.torque_feedforward == 0 for c in fake.command_frames[-1]
    )
    assert [call for call in fake.calls if call in ("stop", "disable", "release")] == [
        "stop",
        "disable",
        "release",
    ]
    assert any(
        json.loads(line).get("event") == "sit_completed"
        for line in log.read_text().splitlines()
    )
    client.wait(lambda s: not s.neutral_required)
    client.press(dpad_right=True)
    client.wait(lambda s: s.selected_startup == "Back-down")
    client.wait(lambda s: not s.fault and not s.neutral_required)
    client.press(2.1, back=True, start=True)
    client.wait(lambda s: s.phase == "awaiting lift")
    assert fake.state == hardware_proto.GROUP_STATE_RUNNING


def test_sit_reverses_to_stand_with_measured_ankle_engagement(
    sit_runner, hardware_proto
):
    fake, client, _, _ = sit_runner
    client.press(1.1, a=True)
    client.wait(lambda s: s.phase == "upright")
    client.press(0.4, forward=-1.0)
    assert client.status.phase == "crouching"
    held = command_frame(fake)
    time.sleep(0.2)
    assert command_frame(fake) == held
    client.press(1.1, a=True)
    assert client.status.phase == "crouching"
    client.buttons = {"forward": 1.0}
    client.wait(lambda s: s.phase == "upright")
    client.buttons = {}
    fake.orientation = hardware_proto.Quaternion(y=-(2**-0.5), w=2**-0.5)
    client.wait(lambda s: not s.startup_ready)
    client.press(1.1, a=True)
    assert client.status.phase == "upright"
    assert all(fake.command_frames[-1][i].kp == 0 for i in (2, 3, 6, 7))
    fake.orientation = hardware_proto.Quaternion(w=1.0)
    client.wait(lambda s: s.startup_ready)
    for i in (2, 3, 6, 7):
        fake.positions[i] = 0.1
    time.sleep(0.1)
    mark = len(fake.command_frames)
    client.press(1.1, a=True)
    client.wait(lambda s: s.mode == pb.MODE_STAND)
    frames = fake.command_frames[mark:]
    first = next(frame for frame in frames if frame[2].kp > 0)
    assert abs(first[2].position - 0.1) < 0.01
    assert 0 < first[2].kp < 2
    assert frames[-1][2].kp == 2
    assert "release" not in fake.calls


@pytest.mark.parametrize("loss", ["stream", "controller"])
def test_sit_disconnect_freezes_each_stage_and_requires_neutral(
    sit_runner, hardware_proto, loss
):
    fake, client, connect, _ = sit_runner

    def reconnect():
        nonlocal client
        phase = client.status.phase
        if loss == "stream":
            client.close()
        else:
            client.connected = False
        time.sleep(0.4)
        held = command_frame(fake)
        if loss == "stream":
            client = connect(buttons={"a": True, "forward": -1.0}, wait_neutral=False)
        else:
            client.buttons = {"a": True, "forward": -1.0}
            client.connected = True
        client.wait(lambda s: s.neutral_required)
        time.sleep(0.3)
        assert command_frame(fake) == held
        assert client.status.phase == phase
        assert not {"stop", "disable", "release"}.intersection(fake.calls)
        assert fake.state == fake.pb.GROUP_STATE_RUNNING
        client.buttons = {}
        client.wait(lambda s: not s.neutral_required)

    client.buttons = {"a": True}
    client.wait(lambda s: s.phase == "releasing ankles")
    fake.orientation = hardware_proto.Quaternion(y=-(2**-0.5), w=2**-0.5)
    reconnect()
    client.wait(lambda s: s.phase == "upright")
    client.press(0.3, forward=-1.0)
    reconnect()
    client.buttons = {"forward": -1.0}
    client.wait(lambda s: s.phase == "bent")
    client.buttons = {}
    time.sleep(0.1)
    client.press(1.1, a=True)
    client.wait(lambda s: s.phase == "lowering onto back")
    reconnect()
    client.buttons = {"a": True}
    client.wait(lambda s: s.phase == "unloading")
    reconnect()
    client.wait(lambda s: s.mode == pb.MODE_DISARMED)


@pytest.mark.parametrize("phase", ["crouching", "lowering onto back", "unloading"])
def test_sit_abort_damps_without_disabling(sit_runner, hardware_proto, phase):
    fake, client, _, _ = sit_runner
    if phase == "crouching":
        client.press(1.1, a=True)
        client.wait(lambda s: s.phase == "upright")
        client.press(0.3, forward=-1.0)
    else:
        bend(client)
        client.press(1.1, a=True)
        client.wait(lambda s: s.phase == "lowering onto back")
        if phase == "unloading":
            client.press(1.1, a=True)
            client.wait(lambda s: s.phase == "unloading")
    client.press(1.1, l2=True, r2=True)
    client.wait(lambda s: s.mode == pb.MODE_FAULT and s.can_clear_fault)
    assert fake.state == hardware_proto.GROUP_STATE_DAMPED
    assert "disable" not in fake.calls


@pytest.mark.parametrize("phase", ["awaiting support", "engaging ankles", "lowering"])
def test_sit_retains_hardware_safety_checks(sit_runner, hardware_proto, phase):
    fake, client, _, _ = sit_runner
    fault = "tilt"
    if phase == "engaging ankles":
        client.press(1.1, a=True)
        client.wait(lambda s: s.phase == "upright")
        client.press(1.1, a=True)
        client.wait(lambda s: s.phase == "engaging ankles")
    if phase != "lowering":
        fake.orientation = hardware_proto.Quaternion(y=2**-0.5, w=2**-0.5)
    else:
        bend(client)
        client.press(1.1, a=True)
        client.wait(lambda s: s.phase == "lowering onto back")
        fake.orientation = hardware_proto.Quaternion(y=2**-0.5, w=2**-0.5)
        fault = "stale"
        fake.failure = "stale"
    client.wait(lambda s: s.mode == pb.MODE_FAULT and s.can_clear_fault)
    assert fault in client.status.fault
    assert fake.state == hardware_proto.GROUP_STATE_DAMPED
    assert "disable" not in fake.calls
