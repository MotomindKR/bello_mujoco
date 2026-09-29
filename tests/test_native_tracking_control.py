"""Buffered SONIC lifecycle over synthetic loopback services, never CAN."""

from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import time
from zipfile import ZipFile

import grpc
import numpy as np
import pytest
import yaml

from bello_mujoco.hardware import _operator_pb2 as operator_pb
from bello_mujoco.hardware import reference_pb2 as reference_pb
from native_operator import OperatorClient, operator_endpoint
from test_native_control import SyntheticHardware, bundle, start_services
from test_native_control import hardware_proto as hardware_proto

pytestmark = pytest.mark.integration


class TrackingHardware(SyntheticHardware):
    reference_failure = None

    def reference(self, request, context):
        if self.reference_failure == "outage":
            context.abort(grpc.StatusCode.UNAVAILABLE, "reference disconnected")
        return reference_pb.GetReferenceResponse(
            robot="synthetic",
            available=True,
            sequence=time.monotonic_ns(),
            source_age_ms=1000 if self.reference_failure == "stale" else 0,
            full_body=reference_pb.FullBodyReference(
                joint_names=self.names,
                position=[float("nan") if self.reference_failure == "invalid" else 0.0]
                * 28,
                velocity=[0.0] * 28,
                root_orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
            ),
        )


def tracking_bundles(source):
    names = [f"joint_{i}" for i in range(28)]
    common = dict(
        coordinates=names,
        joint_names=names,
        policy_indices=list(range(28)),
        default_position=[0.0] * 28,
        default_joint_position=[0.0] * 28,
        kp=[2.0] * 28,
        kd=[0.1] * 28,
        joint_from_motor=np.eye(28).tolist(),
        motor_from_joint=np.eye(28).tolist(),
        joint_limits=[[-3.0, 3.0]] * 28,
        motor_limits=[[-3.0, 3.0]] * 28,
        position_step_limits=[0.1] * 28,
    )
    standing = bundle(
        source,
        metadata_updates=common
        | dict(
            observation_size=28,
            observation_frame=[["last_action", 28]],
        ),
    )
    sonic = bundle(
        source,
        metadata_updates=common
        | dict(
            task="sonic_transfer",
            observation_size=7300,
            observation_stack_size=10,
            observation_frame=[
                ["sonic_future_reference", 640],
                ["gravity_direction", 3],
                ["base_angular_velocity", 3],
                ["joint_position", 28],
                ["joint_velocity", 28],
                ["last_action", 28],
            ],
            reference_clips=["unused"],
            future_frames=10,
            future_dt=0.1,
            source_joint_indices=list(range(28)),
            source_signs=[1.0] * 28,
            source_neutral=[0.0] * 29,
            root_from_hardware_imu_xyzw=[0.0, 0.0, 0.0, 1.0],
        ),
    )
    output = BytesIO()
    with ZipFile(BytesIO(sonic)) as original, ZipFile(output, "w") as archive:
        for name in original.namelist():
            archive.writestr(name, original.read(name))
        archive.writestr("clips.json", json.dumps({"clips": ["unused"]}))
        # Live mode deliberately does not load a prerecorded clip.
        archive.writestr("clips/unused.npz", b"")
    return names, standing, output.getvalue()


@pytest.mark.parametrize("failure", ["outage", "stale", "invalid", "console"])
def test_tracking_returns_to_stand_and_can_reenter_after_rebuffering(
    rust_policy,
    hardware_proto,
    tmp_path,
    failure,
):
    _, _, source = rust_policy
    names, standing, sonic = tracking_bundles(source)
    fake = TrackingHardware(hardware_proto, names, [0.0] * 28)
    fake.reference_failure = "outage"
    server, pool, endpoint = start_services(fake)
    policy = tmp_path / "standing.zip"
    policy.write_bytes(standing)
    tracking = tmp_path / "tracking.zip"
    tracking.write_bytes(sonic)
    settings = yaml.safe_load(
        (
            Path(__file__).parents[1] / "rust/bello-policy/config/hardware.yaml"
        ).read_text()
    )
    settings.update(
        endpoint=endpoint,
        groups=["limb"],
        tracking=dict(
            policy=str(tracking),
            source=str(source),
            endpoint=endpoint,
            rpc_timeout_seconds=0.1,
            poll_period_seconds=0.02,
            max_reference_age_ms=100.0,
            max_sample_gap_seconds=0.1,
            delay_seconds=1.05,
            transition_seconds=0.1,
            max_tilt_error_radians=0.5,
            max_raw_action_abs=5.0,
        ),
    )
    address = operator_endpoint()
    settings["control"].update(ramp_seconds=0.06, operator_listen=address)
    config = tmp_path / "hardware.json"
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
            operator = OperatorClient(address)
            assert not fake.calls
            operator.stand()
            operator.select("Tracking")
            operator.press(1.1, a=True)
            assert operator.status.mode == operator_pb.MODE_STAND
            assert not operator.status.sonic_ready
            fake.reference_failure = None
            operator.wait(lambda s: s.sonic_ready)
            operator.press(1.1, a=True)
            operator.wait(
                lambda s: s.mode == operator_pb.MODE_TRACKING and s.phase == "motion"
            )
            if failure == "console":
                operator.paused.set()
                time.sleep(0.5)
                operator.close()
                operator = OperatorClient(
                    address, buttons={"a": True}, wait_neutral=False
                )
                operator.wait(lambda s: s.neutral_required)
                operator.buttons = {}
            else:
                fake.reference_failure = failure
            operator.wait(lambda s: s.mode == operator_pb.MODE_STAND)
            assert "stop" not in fake.calls and "disable" not in fake.calls
            assert not operator.status.fault
            operator.paused.clear()
            operator.wait(lambda s: not s.neutral_required)
            fake.reference_failure = None
            operator.wait(lambda s: s.sonic_ready)
            operator.press(1.1, a=True)
            operator.wait(
                lambda s: s.mode == operator_pb.MODE_TRACKING and s.phase == "motion"
            )
            operator.press(1.1, a=True)
            operator.wait(lambda s: s.mode == operator_pb.MODE_STAND)
            operator.press(1.1, a=True)
            operator.wait(lambda s: s.mode == operator_pb.MODE_TRACKING)
            operator.press(1.1, l1=True, r1=True)
            operator.wait(
                lambda s: s.mode == operator_pb.MODE_FAULT and s.can_clear_fault
            )
            assert fake.state == hardware_proto.GROUP_STATE_DAMPED
            assert "disable" not in fake.calls
        except Exception:
            log.flush()
            log.seek(0)
            print(log.read())
            raise
        finally:
            if operator is not None:
                operator.close()
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            fake.closed.set()
            server.stop(0).wait()
            pool.shutdown()
