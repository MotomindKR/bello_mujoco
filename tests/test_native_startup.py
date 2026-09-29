"""Assisted startup through the real runner, with loopback hardware only."""

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
from test_native_control import SyntheticHardware, bundle, start_services
from test_native_control import hardware_proto as hardware_proto

pytestmark = pytest.mark.integration


class AssistedHardware(SyntheticHardware):
    def apply(self, commands):
        measured = self.positions.copy()
        super().apply(commands)
        for command in commands:
            if command.kp == command.kd == 0:
                i = int(command.motor_name.removeprefix("motor_"))
                self.positions[i] = measured[i]


@pytest.mark.parametrize(
    "scenario",
    [
        "complete",
        "tight_steps",
        "abort_lift",
        "tilt_after_balance",
        "invalid_speed",
        "invalid_pose",
        "missing_joint",
    ],
)
def test_front_down_startup(rust_policy, hardware_proto, tmp_path, scenario):
    _, _, source = rust_policy
    names = [
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in ("hip_pitch", "knee", "ankle_pitch", "ankle_roll")
    ]
    if scenario == "missing_joint":
        names[0] = "missing_hip"
    nominal = np.array([-0.15, 0.45, -0.3, 0.0] * 2)
    mapping = np.eye(8)
    for i in (2, 6):
        mapping[i : i + 2, i : i + 2] = [[1, 1], [1, -1]]
    fake = AssistedHardware(hardware_proto, names=names, positions=[0.0] * 8)
    fake.orientation = hardware_proto.Quaternion(y=2**-0.5, w=2**-0.5)
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
                position_step_limits=[
                    0.001 if scenario == "tight_steps" and i in (0, 1, 4, 5) else 0.02
                    for i in range(8)
                ],
            ),
        )
    )
    settings = yaml.safe_load(
        (
            Path(__file__).parents[1] / "rust/bello-policy/config/hardware.yaml"
        ).read_text()
    )
    address = operator_endpoint()
    settings.update(endpoint=endpoint, groups=["limb"])
    settings["control"].update(
        operator_listen=address,
        ramp_seconds=0.1,
        lying_start=dict(
            hip_pitch_radians=-0.35,
            knee_radians=0.85,
            straighten_speed_rps=0.1,
            ankle_engage_seconds=1.0,
        ),
    )
    if scenario == "invalid_speed":
        settings["control"]["lying_start"]["straighten_speed_rps"] = 1.0
    if scenario == "invalid_pose":
        settings["control"]["lying_start"]["knee_radians"] = 4.0
    config = tmp_path / "control.json"
    config.write_text(json.dumps(settings))
    operator = None
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
            if scenario in ("invalid_speed", "invalid_pose", "missing_joint"):
                assert process.wait(timeout=5) != 0
                assert not fake.calls
                log.flush()
                log.seek(0)
                assert {
                    "invalid_speed": "speed must",
                    "invalid_pose": "pose exceeds",
                    "missing_joint": "requires left_hip",
                }[scenario] in log.read()
                return
            operator = OperatorClient(address)
            assert not fake.calls
            assert list(operator.status.startups) == ["Crane", "Front-down"]
            operator.press(dpad_right=True)
            operator.wait(lambda s: s.selected_startup == "Front-down")
            operator.press(2.1, back=True, start=True)
            operator.wait(lambda s: s.phase == "awaiting lift", timeout=25)
            if scenario == "abort_lift":
                operator.press(1.1, l1=True, r1=True)
                operator.wait(lambda s: s.mode == pb.MODE_FAULT and s.can_clear_fault)
                assert fake.state == hardware_proto.GROUP_STATE_DAMPED
                assert "disable" not in fake.calls
                operator.press(clear_fault=True)
                operator.wait(lambda s: s.mode == pb.MODE_DISARMED)
                assert fake.state == hardware_proto.GROUP_STATE_IDLE
                return
            assert not operator.status.startup_ready
            operator.press(1.1, a=True)
            assert operator.status.phase == "awaiting lift"
            fake.orientation = hardware_proto.Quaternion(w=1)
            operator.wait(lambda s: s.startup_ready)
            operator.press(1.1, a=True)
            operator.wait(lambda s: s.phase == "straightening")
            if scenario == "tilt_after_balance":
                fake.orientation = hardware_proto.Quaternion(y=2**-0.5, w=2**-0.5)
                operator.wait(lambda s: s.mode == pb.MODE_FAULT and s.can_clear_fault)
                assert "tilt" in operator.status.fault
                assert fake.state == hardware_proto.GROUP_STATE_DAMPED
                return
            operator.press(1.1, a=True)
            assert operator.status.phase == "straightening"
            assert operator.status.progress == 0
            operator.press(0.3, forward=0.5)
            before = fake.positions.copy()
            progress = operator.status.progress
            assert 0 < progress < 0.2
            time.sleep(0.3)
            np.testing.assert_allclose(fake.positions, before)
            operator.close()
            time.sleep(0.4)
            assert "release" not in fake.calls
            operator = OperatorClient(
                address, buttons={"forward": 1.0}, wait_neutral=False
            )
            operator.wait(lambda s: s.neutral_required)
            time.sleep(0.3)
            np.testing.assert_allclose(fake.positions, before)
            operator.buttons = {}
            operator.wait(lambda s: not s.neutral_required)
            mark = len(fake.targets)
            operator.buttons = {"forward": 1.0}
            operator.wait(lambda s: s.phase == "upright", timeout=12)
            operator.buttons = {}
            time.sleep(0.1)
            for frame in fake.command_frames:
                for command in frame:
                    if int(command.motor_name.removeprefix("motor_")) in (2, 3, 6, 7):
                        assert (
                            command.kp == command.kd == command.torque_feedforward == 0
                        )
            joints = np.asarray(fake.targets[mark:]) @ np.linalg.inv(mapping).T
            assert np.max(np.abs(np.diff(joints[:, [0, 1, 4, 5]], axis=0))) <= 0.002001
            np.testing.assert_allclose(
                np.asarray(fake.positions)[[0, 1, 4, 5]],
                nominal[[0, 1, 4, 5]],
                atol=1e-6,
            )
            assert operator.status.mode == pb.MODE_HOLD
            # Free ankles may have moved; engagement must use their measured pose.
            for i in (2, 3, 6, 7):
                fake.positions[i] = 0.1
            mark = len(fake.command_frames)
            operator.press(1.1, a=True)
            operator.wait(lambda s: s.mode == pb.MODE_STAND)
            frames = fake.command_frames[mark:]
            assert any(0 < frame[2].kp < 2 for frame in frames)
            first_engaged = next(frame for frame in frames if frame[2].kp > 0)
            assert abs(first_engaged[2].position - 0.1) < 0.01
            assert frames[-1][2].kp == 2
            operator.press(1.1, l1=True, r1=True)
            operator.wait(lambda s: s.mode == pb.MODE_FAULT and s.can_clear_fault)
            assert fake.state == hardware_proto.GROUP_STATE_DAMPED
            assert "disable" not in fake.calls
        except Exception:
            log.flush()
            log.seek(0)
            print(log.read())
            raise
        finally:
            if operator:
                operator.close()
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
