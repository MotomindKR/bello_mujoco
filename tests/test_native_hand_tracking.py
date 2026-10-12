"""Locomotion's upper-body toggle owns hand tracking through the real hand API."""

import json
import os
from pathlib import Path
import signal
import subprocess
import time

import pytest
import yaml

from bello_mujoco.hardware import hand_pb2 as pb, _operator_pb2 as op
from native_hands import hand_service as hand_service, wait_hands
from native_operator import OperatorClient, operator_endpoint
from test_native_control import SyntheticHardware, bundle, start_services
from test_native_control import hardware_proto as hardware_proto

pytestmark = pytest.mark.integration


class FollowingHardware(SyntheticHardware):
    feedback_delay = 0

    def GetStateFrame(self, request, context):
        time.sleep(self.feedback_delay)
        return super().GetStateFrame(request, context)


@pytest.mark.parametrize(
    "interruption",
    ["toggle", "disconnect", "abort", "terminate", "busy", "server_loss", "body_stall"],
)
def test_locomotion_hand_tracking_ownership(
    rust_policy, hardware_proto, tmp_path, hand_service, interruption
):
    _, _, source = rust_policy
    fake = FollowingHardware(hardware_proto)
    fake.reference_mode = "follow"
    server, pool, endpoint = start_services(fake)
    policy = tmp_path / "policy.zip"
    policy.write_bytes(bundle(source))
    settings = yaml.safe_load(
        (
            Path(__file__).parents[1] / "rust/bello-policy/config/hardware.yaml"
        ).read_text()
    )
    settings.update(endpoint=endpoint, groups=["limb"])
    address = operator_endpoint()
    settings["control"].update(
        operator_listen=address, ramp_seconds=0.06, hand_tracking=hand_service.config
    )
    settings["control"]["following"].update(enabled=True, endpoint=endpoint)
    config = tmp_path / "hardware.json"
    config.write_text(json.dumps(settings))
    operator = None
    occupied = None
    with (tmp_path / "runner.log").open("w") as log:
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
        operator = OperatorClient(address, fake)
        operator.stand()
        assert not hand_service.transport.writes
        operator.select("Locomotion")
        operator.press(1.1, a=True)
        operator.wait(lambda s: s.mode == op.MODE_LOCOMOTION)
        assert not any(h.owned for h in hand_service.controller.status().hands)
        if interruption == "busy":
            occupied = hand_service.controller.acquire(pb.Hand(side=pb.RIGHT))
        operator.press(y=True)
        operator.wait(lambda s: s.tracking_enabled and s.tracking_live)
        if occupied:
            time.sleep(0.3)
            assert not hand_service.controller.status().hands[0].owned
            assert not hand_service.transport.writes
            hand_service.controller.release(occupied)
            occupied = None
        wait_hands(
            hand_service,
            lambda hands: all(h.owned and h.source == pb.TRACKING for h in hands),
        )
        time.sleep(1.2)
        assert all(h.owned for h in hand_service.controller.status().hands)
        assert hand_service.transport.writes
        hand_service.tracking = False
        wait_hands(hand_service, lambda hands: all(not h.tracking_fresh for h in hands))
        time.sleep(0.1)
        writes = len(hand_service.transport.writes)
        time.sleep(0.3)
        assert len(hand_service.transport.writes) == writes
        assert operator.status.mode == op.MODE_LOCOMOTION
        hand_service.tracking = True
        wait_hands(
            hand_service,
            lambda hands: all(
                h.tracking_fresh and not h.suspension_reason for h in hands
            ),
        )
        if interruption in ("toggle", "busy"):
            operator.press(y=True)
            operator.wait(lambda s: not s.tracking_enabled)
        elif interruption == "disconnect":
            operator.connected = False
            operator.wait(lambda s: s.mode == op.MODE_STAND)
        elif interruption == "abort":
            operator.press(1.1, l2=True, r2=True)
            operator.wait(lambda s: s.mode == op.MODE_FAULT)
        elif interruption == "body_stall":
            fake.feedback_delay = 0.6
            wait_hands(
                hand_service, lambda hands: all(not h.owned for h in hands), timeout=0.4
            )
            assert operator.status.mode == op.MODE_LOCOMOTION
            operator.wait(lambda s: s.mode == op.MODE_FAULT)
        elif interruption == "server_loss":
            hand_service.server.stop(0).wait()
            count = len(fake.targets)
            time.sleep(1.3)
            assert (
                len(fake.targets) > count and operator.status.mode == op.MODE_LOCOMOTION
            )
        else:
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=4)
        wait_hands(
            hand_service,
            lambda hands: all(not h.owned and h.source == pb.HOLD for h in hands),
        )
        writes = len(hand_service.transport.writes)
        time.sleep(0.2)
        assert len(hand_service.transport.writes) == writes
    finally:
        if occupied:
            hand_service.controller.release(occupied)
        if operator:
            operator.close()
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=5)
        fake.closed.set()
        server.stop(0).wait()
        pool.shutdown(wait=True)
