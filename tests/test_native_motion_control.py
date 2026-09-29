"""Motion lifecycle and pose failures over synthetic gRPC, never physical CAN."""

import json
from io import BytesIO
import os
from pathlib import Path
import pty
import select
import signal
import subprocess
import time
from zipfile import ZipFile

import jax.numpy as jnp
import mujoco
import grpc
import numpy as np
import onnx
from onnx import helper, numpy_helper
import pytest
import yaml
from scipy.spatial.transform import Rotation

from bello_mujoco.envs.motion_tracking import MotionTrackingEnvConfig
from bello_mujoco.hardware import _pose_pb2 as pb
from bello_mujoco.motion.clip import MotionClip
from bello_mujoco.training.tracking import to_builtin
from bello_mujoco.hardware import _operator_pb2 as operator_pb
from native_operator import OperatorClient, operator_endpoint
from test_native_control import (
    SyntheticHardware,
    assert_motor_rail_log,
    bundle,
    start_services,
)
from test_native_control import hardware_proto as hardware_proto

pytestmark = pytest.mark.integration

XML = """<mujoco><worldbody><body name="root"><freejoint/>
<geom type="sphere" size=".1" mass="1"/>
<body name="offset" pos=".01 0 0"><body name="coincident" pos="-.01 0 0"/></body>
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
    pose_requests = 0
    pitch_ramp = (0.0, 0.0, 0.0, 1.0)

    def __init__(self, pb):
        super().__init__(pb)
        self.applied_gains = []

    def apply(self, commands):
        super().apply(commands)
        self.applied_gains.append(
            [[c.kp, c.kd] for c in sorted(commands, key=lambda c: c.motor_name)]
        )

    def root_orientation(self):
        initial, target, started, duration = self.pitch_ramp
        fraction = np.clip((time.monotonic() - started) / duration, 0.0, 1.0)
        return Rotation.from_euler("y", initial + fraction * (target - initial))

    def ListHardware(self, request, context):
        response = super().ListHardware(request, context)
        response.motors[0].limits.position_max = 0.65
        return response

    def GetStateFrame(self, request, context):
        response = super().GetStateFrame(request, context)
        q = (self.root_orientation() * Rotation.from_euler("z", 0.4)).as_quat()
        response.imus[0].orientation.CopyFrom(
            self.pb.Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        )
        response.imus[0].angular_velocity.CopyFrom(
            self.pb.Vector3(x=0.2, y=-0.1, z=0.3)
        )
        return response

    def pose(self, request, context):
        self.pose_requests += 1
        sequence = 0
        stamp = 0
        while context.is_active() and not self.closed.is_set():
            if self.pose_failure == "outage":
                self.closed.wait(0.01)
                continue
            if self.pose_failure != "repeated":
                sequence += 1
                stamp = time.monotonic_ns()
            q = self.root_orientation().as_quat()
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
                else pb.Orientation(x=q[0], y=q[1], z=q[2], w=q[3]),
                linear_velocity_metres_per_second=pb.LinearVelocity(),
                angular_velocity_radians_per_second=pb.AngularVelocity(),
            )
            if self.pose_failure == "missing_velocity":
                response.ClearField("linear_velocity_metres_per_second")
            yield response
            self.closed.wait(0.005)


def motion_bundle(
    source,
    *,
    root_pitch=0.0,
    frames=3,
    observation_mode="full_state",
    kp=(4.0, 1.0, 0.0),
    kd=(0.3, 0.05, 0.5),
):
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    data.qpos[:] = [0, 0, 0.9, 1, 0, 0, 0, 0.2, -0.3, 0.4]
    data.qpos[3:7] = Rotation.from_euler("y", root_pitch).as_quat()[[3, 0, 1, 2]]
    mujoco.mj_kinematics(model, data)
    ids = [model.body(name).id for name in ("root", "tip")]

    def repeat(a):
        return jnp.array(np.repeat(a[None], frames, axis=0))

    clip = MotionClip(
        fps=10.0,
        joint_names=("first", "held", "last"),
        body_names=("root", "tip"),
        qpos=repeat(data.qpos),
        qvel=repeat(data.qvel),
        body_pos=repeat(data.xpos[ids]),
        body_quat=repeat(data.xquat[ids]),
        body_lin_vel=jnp.zeros((frames, 2, 3)),
        body_ang_vel=jnp.zeros((frames, 2, 3)),
        foot_contacts=jnp.zeros((frames, 2)),
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
    if observation_mode == "proprioceptive":
        layout = [
            term
            for term in layout
            if term[0] not in {"anchor_position_error", "base_linear_velocity"}
        ]
    return bundle(
        source,
        physics=XML.encode(),
        motion=clip,
        target=[0.75, -0.34, 0.44],
        metadata_updates=dict(
            task="motion_tracking",
            kp=kp,
            kd=kd,
            policy_indices=[0, 1, 2],
            observation_size=sum(n for _, n in layout),
            observation_frame=layout,
            control_timestep=0.01,
            motor_limits=[[0.0, 0.9], [-3, 3], [-3, 3]],
            joint_limits=[[0.0, 0.8], [-3, 3], [-3, 3]],
            position_step_limits=[0.02, 0.02, 0.02],
            environment=to_builtin(
                MotionTrackingEnvConfig(
                    anchor_body_name="root", observation_mode=observation_mode
                )
            ),
            end_effector_names=["tip"],
        ),
    )


def control_settings(endpoint, standing, source, log=None):
    settings = yaml.safe_load(
        (
            Path(__file__).parents[1] / "rust/bello-policy/config/hardware.yaml"
        ).read_text()
    )
    settings.update(
        endpoint=endpoint,
        groups=["limb"],
        motion=dict(
            start_time_seconds=0.0,
            transition_seconds=0.08,
            max_tilt_error_radians=0.5,
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
        policy_log_path=str(log) if log is not None else None,
    )
    settings["control"]["operator_listen"] = operator_endpoint()
    settings["motions"] = [
        dict(
            name="Wave",
            policy=str(standing.parent / "motion.zip"),
            source=str(source),
            settings=settings.pop("motion"),
        )
    ]
    return settings


def run_motion(run, standing, *, content, hardware_config):
    Path(hardware_config["motions"][0]["policy"]).write_bytes(content)
    return run(content=standing.read_bytes(), hardware_config=hardware_config)


def failing_motion_bundle(source, failure, observation_mode):
    content = motion_bundle(source, frames=20, observation_mode=observation_mode)
    result = BytesIO()
    with ZipFile(BytesIO(content)) as original, ZipFile(result, "w") as modified:
        for name in original.namelist():
            data = original.read(name)
            if name == "policy.onnx":
                model = onnx.load_model_from_string(data)
                if failure == "action":
                    node = next(n for n in model.graph.node if n.output == ["action"])
                    node.attribute[0].t.CopyFrom(
                        numpy_helper.from_array(np.full((1, 3), 6, dtype=np.float32))
                    )
                elif failure == "inference":
                    # Warm-up uses zeros. Fail only on live observations so this
                    # exercises recovery after acquisition, not bundle rejection.
                    node = next(
                        n for n in model.graph.node if n.output == ["motor_target"]
                    )
                    node.output[0] = "valid_target"
                    model.graph.initializer.extend(
                        [
                            numpy_helper.from_array(np.array(0, np.float32), "zero"),
                            numpy_helper.from_array(
                                np.full((1, 3), np.nan, np.float32), "invalid_target"
                            ),
                        ]
                    )
                    model.graph.node.extend(
                        [
                            helper.make_node("Abs", ["observation"], ["absolute"]),
                            helper.make_node("ReduceSum", ["absolute"], ["sum"]),
                            helper.make_node("Greater", ["sum", "zero"], ["live"]),
                            helper.make_node(
                                "Where",
                                ["live", "invalid_target", "valid_target"],
                                ["motor_target"],
                            ),
                        ]
                    )
                elif failure in ("tracking", "tracking_active"):
                    # The synthetic tip is displaced by motor feedback while
                    # hardware tilt, pose, and IMU remain valid.
                    metadata = json.loads(model.metadata_props[0].value)
                    threshold = (
                        "root_orientation_error"
                        if failure == "tracking_active"
                        else "end_effector_position_error"
                    )
                    metadata["environment"]["done"][threshold] = 0.00001
                    model.metadata_props[0].value = json.dumps(metadata)
                onnx.checker.check_model(model)
                data = model.SerializeToString()
            modified.writestr(name, data)
    return result.getvalue()


@pytest.mark.parametrize(
    "failure",
    ["action", "inference", "tracking", "tracking_active", "tilt_entry", "tilt_active"],
)
@pytest.mark.parametrize("observation_mode", ["full_state", "proprioceptive"])
def test_failed_motion_blends_to_stand_without_releasing_control(
    rust_policy, hardware_proto, tmp_path, failure, observation_mode
):
    if failure == "tracking" and observation_mode == "proprioceptive":
        pytest.skip("absolute-height tracking requires full-state feedback")
    _, _, source = rust_policy
    fake = MotionHardware(hardware_proto)
    server, pool, endpoint = start_services(fake)
    standing = tmp_path / "standing.zip"
    standing.write_bytes(bundle(source))
    motion = tmp_path / "motion.zip"
    motion.write_bytes(failing_motion_bundle(source, failure, observation_mode))
    standing.write_bytes(
        bundle(source, metadata_updates={"position_step_limits": [0.02] * 3})
    )
    log = tmp_path / "policy.jsonl"
    settings = control_settings(endpoint, standing, source, log)
    if failure == "tilt_entry":
        settings["motions"][0]["settings"]["transition_seconds"] = 0.3
        settings["motions"][0]["settings"]["max_tilt_error_radians"] = 0.05
    if observation_mode == "proprioceptive":
        del settings["motions"][0]["settings"]["pose"]
    config = tmp_path / "hardware.json"
    config.write_text(json.dumps(settings))
    output = tmp_path / "runner.log"
    with output.open("w") as stream:
        process = subprocess.Popen(
            [
                os.environ["BELLO_POLICY_BIN"],
                "--source",
                str(source),
                "--policy",
                str(standing),
                "--hardware-config",
                str(config),
                "--arm",
            ],
            stdout=stream,
            stderr=stream,
            start_new_session=True,
        )
    operator = None
    try:
        try:
            operator = OperatorClient(settings["control"]["operator_listen"], fake)
        except grpc.FutureTimeoutError:
            pytest.fail(output.read_text())
        operator.stand()
        for attempt in range(2):
            fake.pitch_ramp = (0.0, 0.0, 0.0, 1.0)
            operator.select("Wave")
            if failure == "tilt_entry":
                operator.buttons = {"a": True}
                operator.wait(lambda status: status.phase == "transition_to_motion")
                operator.buttons = {}
                fake.pitch_ramp = (0.0, 0.6, time.monotonic(), 0.15)
            else:
                operator.press(1.1, a=True)
            if failure == "tracking_active":
                operator.wait(
                    lambda status: (
                        status.mode == operator_pb.MODE_MOTION
                        and status.phase == "motion"
                    )
                )
                fake.pitch_ramp = (0.1, 0.1, 0.0, 1.0)
            if failure == "tilt_active":
                operator.wait(lambda status: status.phase == "motion")
                fake.pitch_ramp = (0.0, 0.6, time.monotonic(), 0.15)
            operator.wait(lambda status: status.mode == operator_pb.MODE_STAND)
            deadline = time.monotonic() + 2
            while not np.allclose(fake.targets[-1], [0.2, -0.3, 0.4], atol=1e-6):
                assert time.monotonic() < deadline
                time.sleep(0.01)
            assert (
                output.read_text().count("Motion failed; returning to Stand")
                == attempt + 1
            )
            assert not operator.status.fault
            assert fake.state == hardware_proto.GROUP_STATE_RUNNING
            assert "release" not in fake.calls
            assert "disable" not in fake.calls
            if failure.startswith("tilt_"):
                fake.pitch_ramp = (0.6, 0.0, time.monotonic(), 0.15)
                time.sleep(0.2)
        events = [json.loads(line) for line in log.read_text().splitlines()]
        returns = [row for row in events if row.get("motion", {}).get("failure")]
        assert returns
        assert any(
            row["motion"]["phase"] == "transition_to_standing" for row in returns
        )
        assert returns[-1]["motion"]["phase"] == "standing"
        expected = {
            "action": "implausible motion action",
            "inference": "nonfinite",
            "tracking": "motion termination",
            "tracking_active": "motion termination: anchor orientation",
            "tilt_entry": "motion tilt deviation",
            "tilt_active": "motion tilt deviation",
        }[failure]
        assert expected in returns[0]["motion"]["failure"]
        if failure == "tilt_entry":
            assert not any(
                row.get("motion", {}).get("phase") == "motion" for row in events
            )
        if failure.startswith("tilt_"):
            assert "damp" not in fake.calls
        np.testing.assert_allclose(fake.targets[-1], [0.2, -0.3, 0.4], atol=1e-6)
        assert np.max(np.abs(np.diff(np.asarray(fake.targets), axis=0))) <= 0.020001
        # Recovery retains control, so a subsequent abort must still damp.
        operator.press(1.1, l1=True, r1=True)
        operator.wait(lambda status: status.mode == operator_pb.MODE_FAULT)
        deadline = time.monotonic() + 3
        while fake.state != hardware_proto.GROUP_STATE_DAMPED:
            assert time.monotonic() < deadline
            time.sleep(0.01)
    finally:
        if operator:
            operator.close()
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=5)
        fake.closed.set()
        server.stop(0).wait()
        pool.shutdown(wait=True)


@pytest.mark.parametrize(
    "frame,error",
    [
        ("root", None),
        ("coincident", None),
        ("offset", "coincide"),
        ("imu", "coincide"),
        ("tip", "fixed"),
        ("missing", "missing"),
    ],
)
def test_motion_pose_frame_requires_fixed_coincident_root(
    rust_policy, tmp_path, hardware_proto, frame, error
):
    run, _, source = rust_policy
    fake = MotionHardware(hardware_proto)
    server, pool, endpoint = start_services(fake)
    standing = tmp_path / "standing.zip"
    standing.write_bytes(bundle(source))
    settings = control_settings(endpoint, standing, source)
    settings["motions"][0]["settings"]["pose"]["tracked_frame"] = frame
    try:
        result = run_motion(
            run, standing, content=motion_bundle(source), hardware_config=settings
        )
        assert (result.returncode == 0) == (error is None), result.stderr
        if error is not None:
            assert error in result.stderr
        assert not fake.calls
    finally:
        fake.closed.set()
        server.stop(0).wait()
        pool.shutdown(wait=True)


@pytest.mark.parametrize("observation_mode", ["full_state", "proprioceptive"])
@pytest.mark.parametrize(
    "failure",
    [
        None,
        "entry_interrupt",
        "interrupt",
        "second_interrupt",
        "terminate",
        "stale",
        "outage",
        "repeated",
        "frame",
        "orientation",
        "missing_velocity",
        "hardware_stale",
        "reject",
        "abort",
        "deep_tilt",
        "wrong_direction",
        "return_still_tilted",
    ],
)
def test_motion_lifecycle_returns_to_standing_or_stops_safely(
    rust_policy, hardware_proto, tmp_path, failure, observation_mode
):
    _, _, source = rust_policy
    fake = MotionHardware(hardware_proto)
    tilted = failure in ("deep_tilt", "wrong_direction", "return_still_tilted")
    if tilted:
        fake.pitch_ramp = (0.6, 0.6, 0.0, 1.0)
    server, pool, endpoint = start_services(
        fake, pose_enabled=observation_mode == "full_state"
    )
    succeeds = failure in (
        None,
        "entry_interrupt",
        "interrupt",
        "deep_tilt",
        "wrong_direction",
    ) or (
        observation_mode == "proprioceptive"
        and failure
        in ("stale", "outage", "repeated", "frame", "orientation", "missing_velocity")
    )
    standing = tmp_path / "standing.zip"
    standing.write_bytes(
        bundle(
            source,
            metadata_updates=dict(
                motor_limits=[[0.0, 0.3], [-3, 3], [-3, 3]],
                joint_limits=[[0.0, 0.35], [-3, 3], [-3, 3]],
                position_step_limits=[0.025, 0.025, 0.025],
            ),
        )
    )
    motion = tmp_path / "motion.zip"
    motion.write_bytes(
        motion_bundle(
            source,
            root_pitch=1.3 if tilted else 0.0,
            frames=20 if tilted else 3,
            observation_mode=observation_mode,
        )
    )
    log = tmp_path / "policy.jsonl"
    settings = control_settings(endpoint, standing, source, log)
    if failure == "entry_interrupt":
        settings["motions"][0]["settings"]["transition_seconds"] = 0.3
    if observation_mode == "proprioceptive":
        del settings["motions"][0]["settings"]["pose"]
    if tilted:
        settings["motions"][0]["settings"]["transition_seconds"] = 0.5
    config = tmp_path / "hardware.json"
    config.write_text(json.dumps(settings))
    master, slave = pty.openpty()
    process = subprocess.Popen(
        [
            os.environ["BELLO_POLICY_BIN"],
            "--source",
            str(source),
            "--policy",
            str(standing),
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

    operator = None
    try:
        wait_for("Operator service listening")
        assert not fake.calls
        operator = OperatorClient(settings["control"]["operator_listen"], fake)
        operator.stand()
        operator.select("Wave")
        operator.buttons = {"a": True}
        wait_for("Transitioning to motion tracking")
        operator.buttons = {}
        if tilted:
            fake.pitch_ramp = (0.6, 1.3, time.monotonic(), 0.5)
        if failure == "entry_interrupt":
            wait_for("Transitioning to motion tracking")
            time.sleep(0.04)
            os.kill(process.pid, signal.SIGINT)
            wait_for("requesting return to standing")
        else:
            wait_for("Motion tracking policy is active")
        if tilted:
            # Remain beyond the absolute upright limit for several policy ticks.
            time.sleep(0.1)
            assert "Control stopped:" not in transcript.decode()
            if failure == "wrong_direction":
                fake.pitch_ramp = (1.3, -0.4, time.monotonic(), 0.6)
            else:
                os.kill(process.pid, signal.SIGINT)
                wait_for("requesting return to standing")
                if failure == "deep_tilt":
                    fake.pitch_ramp = (1.3, 0.0, time.monotonic(), 0.5)
        elif failure in ("interrupt", "second_interrupt"):
            os.kill(process.pid, signal.SIGINT)
            wait_for("requesting return to standing")
            if failure == "second_interrupt":
                os.kill(process.pid, signal.SIGINT)
        elif failure == "terminate":
            os.kill(process.pid, signal.SIGTERM)
        elif failure == "reject":
            fake.failure = failure
        elif failure == "hardware_stale":
            fake.failure = "stale"
        elif failure == "abort":
            fake.abort = True
        else:
            fake.pose_failure = failure
        if succeeds:
            wait_for("Standing policy resumed")
            operator.wait(lambda status: status.mode == operator_pb.MODE_STAND)
            deadline = time.monotonic() + 2
            while not np.allclose(fake.targets[-1], [0.2, -0.3, 0.4], atol=1e-6):
                assert time.monotonic() < deadline
                time.sleep(0.01)
            assert "stop" not in fake.calls
            process.send_signal(signal.SIGTERM)
        deadline = time.monotonic() + 12
        while "release" not in fake.calls:
            assert time.monotonic() < deadline, transcript.decode(errors="replace")
            time.sleep(0.02)
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
        deadline = time.monotonic() + 5
        while process.poll() is None and time.monotonic() < deadline:
            if select.select([master], [], [], 0.05)[0]:
                try:
                    transcript.extend(os.read(master, 65536))
                except OSError:
                    break
        assert process.wait(timeout=3) != 0, transcript.decode(errors="replace")
        assert fake.calls[-1] == "release"
        assert "disable" not in fake.calls
        assert fake.state == fake.pb.GROUP_STATE_DAMPED
        events = [json.loads(line) for line in log.read_text().splitlines()]
        if failure is None:
            assert_motor_rail_log([row for row in events if "motion" in row])
        rows = [row["motion"] for row in events if "motion" in row]
        assert all(row["observation_mode"] == observation_mode for row in rows)
        gains = np.asarray([row["motor_gains"] for row in rows])
        starting_gains = np.asarray([[2.0, 0.1]] * 3)
        motion_gains = np.asarray([[4.0, 0.3], [1.0, 0.05], [0.0, 0.5]])
        low = np.minimum(starting_gains, motion_gains)
        high = np.maximum(starting_gains, motion_gains)
        assert np.all(gains >= low - 1e-6)
        assert np.all(gains <= high + 1e-6)
        for row in rows:
            if row["phase"] in ("standing", "motion"):
                expected = (
                    starting_gains if row["phase"] == "standing" else motion_gains
                )
                np.testing.assert_allclose(row["motor_gains"], expected, atol=1e-6)
        wire_gains = np.asarray(fake.applied_gains)
        np.testing.assert_allclose(wire_gains[0], starting_gains, atol=1e-6)
        assert np.all(wire_gains >= low - 1e-6)
        assert np.all(wire_gains <= high + 1e-6)
        if succeeds:
            np.testing.assert_allclose(wire_gains[-1], starting_gains, atol=1e-6)
            reached_motion_gains = np.any(
                np.all(np.isclose(wire_gains, motion_gains), axis=(1, 2))
            )
            assert reached_motion_gains == (failure != "entry_interrupt")
            bound = (
                abs(motion_gains - starting_gains)
                * 0.02
                / settings["motions"][0]["settings"]["transition_seconds"]
            )
            assert np.all(np.abs(np.diff(wire_gains, axis=0)) <= bound + 1e-6)
        if observation_mode == "proprioceptive":
            assert fake.pose_requests == 0
            for row in rows:
                if row["phase"] == "motion":
                    np.testing.assert_allclose(
                        row["policy"]["observation"][12:15],
                        Rotation.from_euler("z", 0.4).apply([0.2, -0.1, 0.3]),
                        atol=1e-6,
                    )
        if tilted:
            motion_rows = [row for row in rows if row["phase"] == "motion"]
            assert len(motion_rows) >= 3
            assert all(row["tilt_error_radians"] < 0.1 for row in motion_rows[:3])
            if failure == "wrong_direction":
                assert "motion tilt deviation" in transcript.decode()
            if failure == "return_still_tilted":
                assert (
                    "robot tilt exceeds safety limit after return to standing"
                    in transcript.decode()
                )
                assert "Standing policy resumed" not in transcript.decode()
        if failure == "reject":
            assert any(row.get("event") == "command_rejected" for row in events)
        phases = [row["phase"] for row in rows]
        assert "transition_to_motion" in phases
        applied = [row for row in events if "motion" in row]
        targets = np.asarray([row["command"]["motor_target"] for row in applied])
        assert np.abs(np.diff(targets, axis=0)).max() <= 0.025001
        assert targets[:, 0].max() <= 0.65
        for row in applied:
            if row["motion"]["phase"] == "standing":
                assert row["command"]["motor_target"][0] <= 0.3
        if succeeds:
            assert "transition_to_standing" in phases
            assert phases[-1] == "standing"
            np.testing.assert_allclose(fake.targets[-1], [0.2, -0.3, 0.4], atol=1e-6)
        if failure is None:
            assert targets[:, 0].max() > 0.6
            returning = [
                row
                for row in applied
                if row["motion"]["phase"] == "transition_to_standing"
            ]
            assert (
                len(returning) * 0.02
                > settings["motions"][0]["settings"]["transition_seconds"]
            )
    finally:
        if operator is not None:
            operator.close()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=3)
        os.close(master)
        fake.closed.set()
        server.stop(0).wait()
        pool.shutdown(wait=True)


@pytest.mark.parametrize(
    "failure", [None, "frame", "orientation", "coordinates", "kp", "kd"]
)
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
            source,
            metadata_updates={"coordinates": ["other", "held", "last"]}
            if failure == "coordinates"
            else None,
        )
    )
    try:
        result = run_motion(
            run,
            standing,
            content=motion_bundle(source, **{failure: [101.0] * 3})
            if failure in ("kp", "kd")
            else motion_bundle(source),
            hardware_config=control_settings(endpoint, standing, source),
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


@pytest.mark.parametrize("mode", ["full_state", "proprioceptive"])
def test_motion_rejects_pose_configuration_inconsistent_with_bundle(
    rust_policy, hardware_proto, tmp_path, mode
):
    run, _, source = rust_policy
    fake = MotionHardware(hardware_proto)
    server, pool, endpoint = start_services(fake, pose_enabled=False)
    standing = tmp_path / "standing.zip"
    standing.write_bytes(bundle(source))
    settings = control_settings(endpoint, standing, source)
    if mode == "full_state":
        del settings["motions"][0]["settings"]["pose"]
    try:
        result = run_motion(
            run,
            standing,
            content=motion_bundle(source, observation_mode=mode),
            hardware_config=settings,
        )
        assert result.returncode != 0
        assert "motions[].settings.pose" in result.stderr
        assert not fake.calls
        assert fake.pose_requests == 0
    finally:
        fake.closed.set()
        server.stop(0).wait()
        pool.shutdown(wait=True)


@pytest.mark.parametrize("connection_loss", ["close", "stall", "invalid_sample"])
def test_two_named_motions_and_operator_loss(
    rust_policy, hardware_proto, tmp_path, connection_loss
):
    """Exercise the complete operator path without a CAN device or physical robot."""
    _, _, source = rust_policy
    fake = MotionHardware(hardware_proto)
    server, pool, endpoint = start_services(fake, pose_enabled=False)
    standing = tmp_path / "standing.zip"
    standing.write_bytes(bundle(source))
    motion = tmp_path / "motion.zip"
    motion.write_bytes(
        motion_bundle(source, frames=8, observation_mode="proprioceptive")
    )
    log = tmp_path / "operator.jsonl"
    settings = control_settings(endpoint, standing, source, log)
    settings["control"]["ramp_seconds"] = 0.6
    entry = settings["motions"][0]
    entry["name"] = "Wave"
    del entry["settings"]["pose"]
    settings["motions"].append(dict(entry, name="Bow"))
    config = tmp_path / "operator.json"
    config.write_text(json.dumps(settings))
    output = tmp_path / "runner.log"
    operator = None
    with output.open("w") as stream:
        process = subprocess.Popen(
            [
                os.environ["BELLO_POLICY_BIN"],
                "--source",
                str(source),
                "--policy",
                str(standing),
                "--hardware-config",
                str(config),
                "--arm",
            ],
            stdout=stream,
            stderr=stream,
            start_new_session=True,
        )
    address = settings["control"]["operator_listen"]

    def lose_connection():
        if connection_loss == "stall":
            operator.paused.set()
        elif connection_loss == "invalid_sample":
            operator.buttons = {"forward": float("nan")}
        if connection_loss != "close":
            deadline = time.monotonic() + 2
            while operator.error is None:
                assert time.monotonic() < deadline
                time.sleep(0.01)
        operator.close()

    def assert_control_retained():
        assert fake.state == fake.pb.GROUP_STATE_RUNNING
        assert not {"damp", "stop", "disable", "release"}.intersection(fake.calls)

    try:
        operator = OperatorClient(address, fake)
        operator.press(0.2, back=True, start=True)
        assert not fake.calls
        lose_connection()
        operator = OperatorClient(
            address, fake, buttons={"back": True, "start": True}, wait_neutral=False
        )
        operator.wait(lambda status: status.neutral_required)
        time.sleep(0.3)
        assert not fake.calls
        operator.buttons = {}
        operator.wait(lambda status: not status.neutral_required)
        other = OperatorClient(address, wait_neutral=False)
        try:
            deadline = time.monotonic() + 2
            while other.error is None and time.monotonic() < deadline:
                time.sleep(0.01)
            assert other.error.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
        finally:
            other.close()
        operator.press(2.1, back=True, start=True)
        operator.wait(lambda status: status.phase == "ramping")
        lose_connection()
        time.sleep(0.3)
        held = np.array(fake.targets[-1])
        time.sleep(0.2)
        np.testing.assert_array_equal(fake.targets[-1], held)
        assert_control_retained()
        operator = OperatorClient(address, fake)
        operator.wait(lambda status: status.phase == "holding")
        lose_connection()
        time.sleep(0.3)
        held = np.array(fake.targets[-1])
        time.sleep(0.2)
        np.testing.assert_array_equal(fake.targets[-1], held)
        assert_control_retained()
        operator = OperatorClient(address, fake)
        operator.press(1.1, a=True)
        operator.wait(lambda status: status.mode == operator_pb.MODE_STAND)
        lose_connection()
        time.sleep(0.3)
        assert_control_retained()
        operator = OperatorClient(address, fake)
        operator.wait(lambda status: status.mode == operator_pb.MODE_STAND)
        operator.select("Locomotion")
        operator.press(1.1, a=True)
        operator.wait(lambda status: status.mode == operator_pb.MODE_LOCOMOTION)
        lose_connection()
        time.sleep(0.3)
        operator = OperatorClient(
            address, fake, buttons={"forward": 1.0}, wait_neutral=False
        )
        operator.wait(
            lambda status: (
                status.mode == operator_pb.MODE_STAND and status.neutral_required
            )
        )
        operator.buttons = {}
        operator.wait(lambda status: not status.neutral_required)
        for name in ("Wave", "Bow"):
            operator.select(name)
            operator.buttons = {"a": True}
            operator.wait(
                lambda status, selected=name: (
                    status.active_motion == selected and status.phase == "motion"
                )
            )
            lose_connection()
            time.sleep(0.3)
            operator = OperatorClient(address, fake)
            operator.wait(lambda status: status.mode == operator_pb.MODE_STAND)
            assert_control_retained()
        operator.press(1.1, l1=True, r1=True)
        operator.wait(lambda status: status.mode == operator_pb.MODE_FAULT)
        deadline = time.monotonic() + 3
        while "release" not in fake.calls and time.monotonic() < deadline:
            time.sleep(0.01)
        assert fake.calls[-1] == "release"
        assert "disable" not in fake.calls
        assert fake.state == fake.pb.GROUP_STATE_DAMPED
        process.terminate()
        assert process.wait(timeout=5) != 0
    finally:
        if operator:
            operator.close()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        fake.closed.set()
        server.stop(0).wait()
        pool.shutdown(wait=True)
