"""Stand and Locomotion keep one continuous policy session."""

import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import pytest
import yaml

from bello_mujoco.envs.gait import GaitConfig
from bello_mujoco.envs.locomotion_config import BelloEnvConfig
from bello_mujoco.hardware import _operator_pb2 as pb
from bello_mujoco.training.tracking import to_builtin
from native_operator import OperatorClient, operator_endpoint
from test_native_control import SyntheticHardware, bundle, start_services
from test_native_control import hardware_proto as hardware_proto

pytestmark = pytest.mark.integration


def test_stand_locomotion_and_disconnect_preserve_policy_state(
    rust_policy, hardware_proto, tmp_path
):
    _, _, source = rust_policy
    fake = SyntheticHardware(hardware_proto)
    server, pool, endpoint = start_services(fake)
    policy = tmp_path / "policy.zip"
    policy.write_bytes(
        bundle(
            source,
            action=[0.25, -0.2],
            metadata_updates=dict(
                observation_frame=[
                    ["last_action", 2],
                    ["command", 3],
                    ["gait_phase", 3],
                ],
                observation_size=24,
                observation_stack_size=3,
                environment=to_builtin(
                    BelloEnvConfig(gait=GaitConfig(command_slew_enabled=True))
                ),
            ),
        )
    )
    settings = yaml.safe_load(
        (
            Path(__file__).parents[1] / "rust/bello-policy/config/hardware.yaml"
        ).read_text()
    )
    address = operator_endpoint()
    log_path = tmp_path / "policy.jsonl"
    settings.update(endpoint=endpoint, groups=["limb"])
    settings["control"].update(
        operator_listen=address, ramp_seconds=0.08, policy_log_path=str(log_path)
    )
    config = tmp_path / "hardware.yaml"
    config.write_text(yaml.safe_dump(settings))
    command = [
        os.environ["BELLO_POLICY_BIN"],
        "--policy",
        str(policy),
        "--source",
        str(source),
    ]
    operator = None
    with (tmp_path / "runner.log").open("w+") as output:
        process = subprocess.Popen(
            command + ["--hardware-config", str(config), "--arm"],
            stdout=output,
            stderr=output,
        )
        try:
            operator = OperatorClient(address)
            operator.stand()
            for _ in range(2):
                operator.select("Locomotion")
                operator.press(1.1, a=True)
                operator.wait(lambda s: s.mode == pb.MODE_LOCOMOTION)
                operator.press(0.5, forward=0.5)
                operator.press(1.1, a=True)
                operator.wait(lambda s: s.mode == pb.MODE_STAND)
            operator.select("Locomotion")
            operator.press(1.1, a=True)
            operator.wait(lambda s: s.mode == pb.MODE_LOCOMOTION)
            operator.buttons = {"forward": 0.5}
            time.sleep(0.5)
            operator.close()
            operator = None
            time.sleep(0.35)
            operator = OperatorClient(address)
            operator.wait(lambda s: s.mode == pb.MODE_STAND)
            time.sleep(0.5)
            assert fake.state == hardware_proto.GROUP_STATE_RUNNING
            assert "release" not in fake.calls
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=5)
            rows = [json.loads(line) for line in log_path.read_text().splitlines()]
            ticks = [row for row in rows if "policy" in row]
            assert len(ticks) > 100
            # Replay exactly the measured inputs without operator mode changes.
            # A mode switch must have the same policy/gait trajectory as this
            # uninterrupted session, including the zero-command stop on loss.
            replay = subprocess.run(
                command + ["--telemetry"],
                input="".join(json.dumps(row["feedback"]) + "\n" for row in ticks),
                text=True,
                capture_output=True,
                check=True,
                timeout=15,
            )
            expected = [json.loads(line) for line in replay.stdout.splitlines()]
            assert len(expected) == len(ticks)
            for row, reference in zip(ticks, expected, strict=True):
                for field in ("observation", "action", "motor_target"):
                    np.testing.assert_allclose(
                        row["policy"][field], reference[field], atol=1e-7
                    )
                for field in ("command", "phase"):
                    np.testing.assert_allclose(
                        row["policy"]["gait"][field],
                        reference["gait"][field],
                        atol=1e-7,
                    )
                for field in ("pending", "next_start_left", "started"):
                    assert row["policy"]["gait"][field] == reference["gait"][field]
        finally:
            if operator is not None:
                operator.close()
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            fake.closed.set()
            server.stop(0).wait()
            pool.shutdown(wait=True)
