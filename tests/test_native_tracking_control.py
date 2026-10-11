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
from native_runtime import floating_joint_physics
from test_native_control import SyntheticHardware, bundle, start_services
from test_native_control import hardware_proto as hardware_proto

pytestmark = pytest.mark.integration


class TrackingHardware(SyntheticHardware):
    reference_failure = None
    reference_positions = tuple(0.4 + i * 0.001 for i in range(28))
    reference_velocities = tuple(i * 0.01 for i in range(28))

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
                position=[
                    float("nan") if self.reference_failure == "invalid" else value
                    for value in self.reference_positions
                ],
                velocity=self.reference_velocities,
                root_orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
            ),
        )


def tracking_bundles(source, *, sonic_target=None, sonic_action=None):
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
        physics=floating_joint_physics(reversed(names)),
        target=sonic_target or [0.05] * 28,
        action=sonic_action,
        metadata_updates=common
        | dict(
            task="sonic_transfer",
            action_clip=20.0,
            kp=[4.0] * 28,
            kd=[0.2] * 28,
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


@pytest.mark.parametrize(
    "failure",
    [
        "outage",
        "stale",
        "brief_outage",
        "brief_stale",
        "invalid",
        "console",
        "return",
        "clipped_action",
        "settle_timeout",
        "abort_return",
    ],
)
def test_tracking_returns_to_stand_and_can_reenter_after_rebuffering(
    rust_policy,
    hardware_proto,
    tmp_path,
    failure,
):
    _, _, source = rust_policy
    names, standing, sonic = tracking_bundles(
        source,
        sonic_target=[0.5] * 28 if failure == "settle_timeout" else None,
        sonic_action=[25.0, -25.0] * 14 if failure == "clipped_action" else None,
    )
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
            neutral_return_seconds=2.0 if failure == "abort_return" else 0.3,
            neutral_joint_tolerance_radians=0.15,
            neutral_settle_timeout_seconds=0.4,
            standing_transition_seconds=0.06,
            max_tilt_error_radians=0.5,
        ),
    )
    address = operator_endpoint()
    settings["control"].update(ramp_seconds=0.06, operator_listen=address)
    policy_log = tmp_path / "policy.jsonl"
    settings["control"]["policy_log_path"] = str(policy_log)
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
            if failure in ("brief_outage", "brief_stale"):
                fake.reference_failure = failure.removeprefix("brief_")
                time.sleep(0.06)
                fake.reference_failure = None
                time.sleep(0.3)
                assert operator.status.mode == operator_pb.MODE_TRACKING
                assert operator.status.phase == "motion"
                operator.press(1.1, a=True)
            elif failure == "console":
                operator.paused.set()
                time.sleep(0.5)
                operator.close()
                operator = OperatorClient(
                    address, buttons={"a": True}, wait_neutral=False
                )
                operator.wait(lambda s: s.neutral_required)
                operator.buttons = {}
            elif failure in ("return", "clipped_action"):
                operator.press()
                operator.press(1.1, a=True)
            elif failure in ("settle_timeout", "abort_return"):
                fake.reference_failure = "outage"
            else:
                fake.reference_failure = failure
            if failure == "abort_return":
                operator.wait(lambda s: s.phase == "returning to neutral")
                operator.press(1.1, l2=True, r2=True)
                operator.wait(
                    lambda s: s.mode == operator_pb.MODE_FAULT and s.can_clear_fault
                )
                assert fake.state == hardware_proto.GROUP_STATE_DAMPED
                assert "disable" not in fake.calls
                samples = [
                    json.loads(line) for line in policy_log.read_text().splitlines()
                ]
                assert not any(
                    s.get("motion", {}).get("phase") == "transition_to_standing"
                    for s in samples
                )
                return
            operator.wait(lambda s: s.mode == operator_pb.MODE_STAND)
            # Inspect the large observation log after control stops. Parsing it
            # here can starve the synthetic Python gRPC server of the GIL.
            first_return_log_end = policy_log.stat().st_size
            operator.paused.clear()
            operator.wait(lambda s: not s.neutral_required)
            fake.reference_failure = None
            operator.wait(lambda s: s.sonic_ready)
            operator.press()
            operator.press(1.1, a=True)
            operator.wait(
                lambda s: s.mode == operator_pb.MODE_TRACKING and s.phase == "motion"
            )
            # Confirmation must be released after entering the new phase.
            operator.press()
            operator.press(1.1, a=True)
            operator.wait(lambda s: s.mode == operator_pb.MODE_STAND)
            # Reentry needs a released sample after the mode transition.
            operator.press()
            operator.press(1.1, a=True)
            operator.wait(lambda s: s.mode == operator_pb.MODE_TRACKING)
            operator.press(1.1, l2=True, r2=True)
            operator.wait(
                lambda s: s.mode == operator_pb.MODE_FAULT and s.can_clear_fault
            )
            assert fake.state == hardware_proto.GROUP_STATE_DAMPED
            assert "disable" not in fake.calls
            with policy_log.open("rb") as recorded:
                completed = recorded.read(first_return_log_end).rsplit(b"\n", 1)[0]
            samples = [json.loads(line) for line in completed.splitlines()]
            neutral = [
                s
                for s in samples
                if s.get("motion", {}).get("phase") == "returning_to_neutral"
            ]
            handover = [
                s
                for s in samples
                if s.get("motion", {}).get("phase") == "transition_to_standing"
            ]
            if failure == "clipped_action":
                for sample in neutral:
                    policy = sample["motion"]["policy"]
                    np.testing.assert_array_equal(
                        policy["raw_action"], [25.0, -25.0] * 14
                    )
                    np.testing.assert_array_equal(policy["action"], [20.0, -20.0] * 14)
                    np.testing.assert_array_equal(
                        policy["observation"][-28:], [20.0, -20.0] * 14
                    )
            assert len(neutral) >= 10
            assert 2 <= len(handover) <= 4
            assert neutral[-1]["elapsed_seconds"] < handover[0]["elapsed_seconds"]
            first = samples.index(neutral[0])
            live_frame = np.asarray(
                samples[first - 1]["motion"]["policy"]["observation"][-730:]
            )
            commands = live_frame[:640].reshape(10, 64)[:, :58].reshape(-1)
            # The physics model stores joints in reverse policy order. Verify
            # the actual live service/runner path uses the checkpoint layout.
            np.testing.assert_allclose(
                commands[:290].reshape(10, 29),
                np.tile([*reversed(fake.reference_positions), 0.0], (10, 1)),
                atol=1e-6,
            )
            np.testing.assert_allclose(
                commands[290:].reshape(10, 29),
                np.tile([*reversed(fake.reference_velocities), 0.0], (10, 1)),
                atol=1e-6,
            )
            np.testing.assert_array_equal(
                neutral[0]["motion"]["policy"]["observation"][:-730],
                samples[first - 1]["motion"]["policy"]["observation"][730:],
            )
            reference_positions = [
                s["motion"]["policy"]["observation"][-730] for s in neutral
            ]
            assert reference_positions[0] > 0.3
            assert reference_positions[-1] < 0.01
            assert np.all(np.diff(reference_positions) <= 0.0)
            if failure == "settle_timeout":
                assert "neutral settling timed out" in handover[0]["motion"]["failure"]
                assert (
                    neutral[-1]["elapsed_seconds"] - neutral[0]["elapsed_seconds"]
                    >= 0.6
                )
            # SONIC alone produces the targets and gains throughout the neutral return.
            for sample in neutral:
                motion = sample["motion"]
                np.testing.assert_allclose(
                    motion["motor_target"], motion["policy"]["motor_target"]
                )
                np.testing.assert_allclose(motion["motor_gains"], [[4.0, 0.2]] * 28)
                assert len(motion["policy"]["observation"]) == 7300
            assert "stop" not in fake.calls and "disable" not in fake.calls
            assert not any(s.get("event") == "control_fault" for s in samples)
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
