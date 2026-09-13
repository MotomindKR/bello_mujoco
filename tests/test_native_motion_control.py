"""Motion lifecycle and pose failures over synthetic gRPC, never physical CAN."""

import json
import os
from pathlib import Path
import pty
import select
import signal
import subprocess
import time

import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
import yaml

from bello_mujoco.envs.motion_tracking import MotionTrackingEnvConfig
from bello_mujoco.hardware import _pose_pb2 as pb
from bello_mujoco.motion.clip import MotionClip
from bello_mujoco.training.tracking import to_builtin
from test_native_control import SyntheticHardware, bundle, start_services
from test_native_control import hardware_proto as hardware_proto

pytestmark = pytest.mark.integration

XML = """<mujoco><worldbody><body name="root"><freejoint/>
<geom type="sphere" size=".1" mass="1"/>
<body name="imu" quat=".9800665778412416 0 0 .19866933079506122"/>
<body name="a"><joint name="first" axis="1 0 0"/>
<geom type="sphere" size=".04" mass=".1"/>
<body name="b" pos="0 0 -.2"><joint name="held" axis="0 1 0"/>
<geom type="sphere" size=".04" mass=".1"/>
<body name="tip" pos="0 0 -.2"><joint name="last" axis="1 0 0"/>
<geom type="sphere" size=".04" mass=".1"/>
</body></body></body></body></worldbody></mujoco>"""


class MotionHardware(SyntheticHardware):
    pose_failure = None

    def GetStateFrame(self, request, context):
        response = super().GetStateFrame(request, context)
        response.imus[0].orientation.CopyFrom(
            self.pb.Quaternion(w=np.cos(0.2), z=np.sin(0.2))
        )
        return response

    def pose(self, request, context):
        sequence = 0
        stamp = 0
        while context.is_active() and not self.closed.is_set():
            if self.pose_failure == "outage":
                self.closed.wait(0.01)
                continue
            if self.pose_failure != "repeated":
                sequence += 1
                stamp = time.monotonic_ns()
            response = pb.WatchPoseResponse(
                reference_frame=request.reference_frame,
                tracked_frame="wrong"
                if self.pose_failure == "frame"
                else request.tracked_frame,
                sequence=sequence,
                sample_monotonic_ns=stamp,
                sample_age_ms=1000 if self.pose_failure == "stale" else 0,
                translation_metres=pb.Translation(z=0.9),
                orientation_xyzw=pb.Orientation(w=np.cos(0.4), x=np.sin(0.4))
                if self.pose_failure == "orientation"
                else pb.Orientation(w=1),
                linear_velocity_metres_per_second=pb.LinearVelocity(),
                angular_velocity_radians_per_second=pb.AngularVelocity(),
            )
            if self.pose_failure == "missing_velocity":
                response.ClearField("linear_velocity_metres_per_second")
            yield response
            self.closed.wait(0.005)


def motion_bundle(source):
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    data.qpos[:] = [0, 0, 0.9, 1, 0, 0, 0, 0.2, -0.3, 0.4]
    mujoco.mj_kinematics(model, data)
    ids = [model.body(name).id for name in ("root", "tip")]

    def repeat(a):
        return jnp.array(np.repeat(a[None], 3, axis=0))

    clip = MotionClip(
        fps=10.0,
        joint_names=("first", "held", "last"),
        body_names=("root", "tip"),
        qpos=repeat(data.qpos),
        qvel=repeat(data.qvel),
        body_pos=repeat(data.xpos[ids]),
        body_quat=repeat(data.xquat[ids]),
        body_lin_vel=jnp.zeros((3, 2, 3)),
        body_ang_vel=jnp.zeros((3, 2, 3)),
        foot_contacts=jnp.zeros((3, 2)),
    )
    layout = [
        ("reference_joint_position", 3),
        ("reference_joint_velocity", 3),
        ("anchor_position_error", 3),
        ("anchor_orientation_error_6d", 6),
        ("base_linear_velocity", 3),
        ("base_angular_velocity", 3),
        ("joint_position", 3),
        ("joint_velocity", 3),
        ("last_action", 3),
    ]
    return bundle(
        source,
        physics=XML.encode(),
        motion=clip,
        target=[0.26, -0.34, 0.44],
        metadata_updates=dict(
            task="motion_tracking",
            policy_indices=[0, 1, 2],
            observation_size=sum(n for _, n in layout),
            observation_frame=layout,
            control_timestep=0.01,
            environment=to_builtin(MotionTrackingEnvConfig(anchor_body_name="root")),
            end_effector_names=["tip"],
        ),
    )


def control_settings(endpoint, standing, log=None):
    settings = yaml.safe_load(
        (
            Path(__file__).parents[1] / "rust/bello-policy/config/hardware.yaml"
        ).read_text()
    )
    settings.update(
        endpoint=endpoint,
        groups=["limb"],
        motion=dict(
            standing_policy=str(standing),
            start_time_seconds=0.0,
            transition_seconds=0.08,
            max_tilt_radians=1.2,
            max_raw_action_abs=5.0,
            pose=dict(
                endpoint=endpoint,
                reference_frame="odom",
                tracked_frame="root",
                timeout_seconds=1.0,
                max_age_ms=80.0,
                max_linear_velocity=5.0,
                max_angular_velocity=20.0,
                max_imu_orientation_error=0.1,
            ),
        ),
    )
    settings["control"].update(
        ramp_seconds=0.06,
        max_duration_seconds=0.8,
        policy_log_path=str(log) if log is not None else None,
    )
    settings["control"]["remote"]["endpoint"] = endpoint
    return settings


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "interrupt",
        "second_interrupt",
        "terminate",
        "stale",
        "outage",
        "repeated",
        "frame",
        "orientation",
        "missing_velocity",
        "reject",
        "abort",
    ],
)
def test_motion_lifecycle_returns_to_standing_or_stops_safely(
    rust_policy, hardware_proto, tmp_path, failure
):
    _, _, source = rust_policy
    fake = MotionHardware(hardware_proto)
    server, pool, endpoint = start_services(fake)
    standing = tmp_path / "standing.zip"
    standing.write_bytes(bundle(source))
    motion = tmp_path / "motion.zip"
    motion.write_bytes(motion_bundle(source))
    log = tmp_path / "policy.jsonl"
    settings = control_settings(endpoint, standing, log)
    config = tmp_path / "hardware.json"
    config.write_text(json.dumps(settings))
    master, slave = pty.openpty()
    process = subprocess.Popen(
        [
            os.environ["BELLO_POLICY_BIN"],
            "--source",
            str(source),
            "--policy",
            str(motion),
            "--hardware-config",
            str(config),
            "--arm",
        ],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        start_new_session=True,
    )
    os.close(slave)
    transcript = bytearray()

    def wait_for(text):
        deadline = time.monotonic() + 8
        while text.encode() not in transcript:
            assert time.monotonic() < deadline, transcript.decode(errors="replace")
            if select.select([master], [], [], 0.03)[0]:
                try:
                    transcript.extend(os.read(master, 8192))
                except OSError:
                    pytest.fail(transcript.decode(errors="replace"))
            assert process.poll() is None or text.encode() in transcript, (
                transcript.decode(errors="replace")
            )

    try:
        wait_for("Type HOLD")
        assert not fake.calls
        os.write(master, b"HOLD\n")
        wait_for("then type RUN")
        os.write(master, b"RUN\n")
        wait_for("Type MOTION")
        os.write(master, b"MOTION\n")
        wait_for("Motion tracking policy is active")
        if failure in ("interrupt", "second_interrupt"):
            os.kill(process.pid, signal.SIGINT)
            wait_for("requesting return to standing")
            if failure == "second_interrupt":
                os.kill(process.pid, signal.SIGINT)
        elif failure == "terminate":
            os.kill(process.pid, signal.SIGTERM)
        elif failure == "reject":
            fake.failure = failure
        elif failure == "abort":
            fake.abort = True
        else:
            fake.pose_failure = failure
        if failure in (None, "interrupt"):
            wait_for("Standing policy resumed")
            assert "stop" not in fake.calls
        wait_for("Type IDLE")
        assert "stop" in fake.calls and "disable" not in fake.calls
        os.write(master, b"IDLE\n")
        assert (process.wait(timeout=3) == 0) == (failure in (None, "interrupt")), (
            transcript.decode(errors="replace")
        )
        assert fake.calls[-2:] == ["disable", "release"]
        rows = [json.loads(line)["motion"] for line in log.read_text().splitlines()]
        phases = [row["phase"] for row in rows]
        assert "transition_to_motion" in phases
        targets = np.asarray([row["motor_target"] for row in rows])
        assert np.abs(np.diff(targets, axis=0)).max() <= 0.015001
        if failure in (None, "interrupt"):
            assert "transition_to_standing" in phases
            assert phases[-1] == "standing"
            np.testing.assert_allclose(targets[-1], [0.2, -0.3, 0.4], atol=1e-6)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=3)
        os.close(master)
        fake.closed.set()
        server.stop(0).wait()
        pool.shutdown(wait=True)


@pytest.mark.parametrize("failure", [None, "frame", "orientation", "gains"])
def test_motion_preflight_never_acquires_control(
    rust_policy, hardware_proto, tmp_path, failure
):
    run, _, source = rust_policy
    fake = MotionHardware(hardware_proto)
    fake.pose_failure = failure
    server, pool, endpoint = start_services(fake)
    standing = tmp_path / "standing.zip"
    standing.write_bytes(
        bundle(
            source, metadata_updates={"kp": [3, 3, 3]} if failure == "gains" else None
        )
    )
    try:
        result = run(
            content=motion_bundle(source),
            hardware_config=control_settings(endpoint, standing),
        )
        assert (result.returncode == 0) == (failure is None), result.stderr
        assert not fake.calls
        if failure is None:
            assert json.loads(result.stdout)["armed"] is False
        else:
            assert failure in result.stderr
    finally:
        fake.closed.set()
        server.stop(0).wait()
        pool.shutdown(wait=True)
