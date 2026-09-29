"""CPU-only lifecycle checks against loopback synthetic hardware, never CAN."""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import pty
import select
import shutil
import signal
import subprocess
import threading
import time

import grpc
import numpy as np
import onnx
from google.protobuf.message_factory import GetMessageClass
from onnx import helper, numpy_helper
import pytest
import yaml

from bello_mujoco.deployment import deployment_bundle
from bello_mujoco.envs.locomotion_config import BelloEnvConfig
from bello_mujoco.envs.rewards import CommandConfig, TorsoCommandConfig
from bello_mujoco.hardware import _operator_pb2 as operator_pb
from native_operator import OperatorClient, operator_endpoint
from bello_mujoco.hardware import reference_pb2 as reference_pb
from bello_mujoco.hardware import _pose_pb2 as pose_pb
from bello_mujoco.training.tracking import to_builtin

pytestmark = pytest.mark.integration

FOLLOWING_CASES = (
    "follow",
    "outage",
    "bad_reference",
    "missing_torso",
    "nan_torso",
    "stale_reference",
    "unsafe_reference",
)


@pytest.fixture(scope="module")
def hardware_proto(tmp_path_factory):
    compiler = shutil.which("protoc")
    if compiler is None:
        pytest.skip("native lifecycle integration requires the Nix protoc tool")
    output = tmp_path_factory.mktemp("hardware-proto")
    source = (
        Path(__file__).resolve().parents[1]
        / "deps/bello-stack/components/bello-hardware-api/proto"
    )
    subprocess.run(
        [
            compiler,
            f"-I{source}",
            f"--python_out={output}",
            str(source / "bello/hardware/v1/hardware.proto"),
        ],
        check=True,
    )
    spec = importlib.util.spec_from_file_location(
        "native_test_hardware_pb2", output / "bello/hardware/v1/hardware_pb2.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SyntheticHardware:
    def __init__(self, pb, names=("first", "held", "last"), positions=(0.2, -0.3, 0.4)):
        self.pb = pb
        self.state = pb.GROUP_STATE_IDLE
        self.names = names
        self.positions = list(positions)
        self.calls = []
        self.failure = None
        self.lease_duration_ms = 10000
        self.abort = False
        self.closed = threading.Event()
        self.reference_mode = None
        self.reference_started = None
        self.targets = []

    def ListHardware(self, request, context):
        p = self.pb
        status = p.DeviceStatus(state=p.DEVICE_STATE_ONLINE)
        addresses = [
            p.MotorAddress(bus="synthetic", id=i + 1) for i in range(len(self.names))
        ]
        return p.ListHardwareResponse(
            buses=[p.BusInfo(name="synthetic", interface="none", status=status)],
            groups=[
                p.GroupInfo(
                    name="limb", control_rate_hz=100, state=self.state, motors=addresses
                )
            ],
            motors=[
                p.MotorInfo(
                    name=f"motor_{i}",
                    coordinate=name,
                    group="limb",
                    address=addresses[i],
                    model="synthetic",
                    direction=1,
                    status=status,
                    limits=p.MotorLimits(
                        position_min=-3,
                        position_max=3,
                        velocity_min=-10,
                        velocity_max=10,
                        torque_min=-10,
                        torque_max=10,
                        kp_max=100,
                        kd_max=10,
                    ),
                )
                for i, name in enumerate(self.names)
            ],
            serial_devices=[
                p.SerialDeviceInfo(
                    name="serial", port="none", baud_rate=115200, status=status
                ),
                p.SerialDeviceInfo(
                    name="power_serial",
                    port="power-none",
                    baud_rate=115200,
                    status=status,
                ),
            ],
            imus=[
                p.ImuInfo(
                    name="imu",
                    serial_device="serial",
                    source_frame="imu",
                    output_frame="imu",
                    status=status,
                )
            ],
            power_boards=[
                p.PowerBoardInfo(
                    name="supply",
                    serial_device="power_serial",
                    profile="agx",
                    stale_timeout_ms=3000,
                    sync_period_ms=1000,
                    status=status,
                )
            ],
        )

    def WatchTelemetry(self, request, context):
        assert request.include_power_boards
        assert list(request.power_boards) == ["supply"]
        assert not request.include_motors and not request.include_imu
        p = self.pb
        while context.is_active() and not self.closed.is_set():
            yield p.WatchTelemetryResponse(
                monotonic_ns=time.monotonic_ns(),
                power_board=p.PowerBoardTelemetry(
                    board_name="supply",
                    connection_generation=1,
                    status=p.DeviceStatus(state=p.DEVICE_STATE_ONLINE),
                    samples=[
                        p.PowerBoardSample(
                            sequence=42,
                            motor_rail=p.PowerVoltageCurrent(
                                voltage_v=48.5, current_a=-1.25
                            ),
                        )
                    ],
                ),
            )
            self.closed.wait(0.01)

    def GetStateFrame(self, request, context):
        p = self.pb
        now = time.monotonic_ns()
        age = 1_000_000_000 if self.failure == "stale" else 0
        stamp = now - age
        return p.GetStateFrameResponse(
            captured_monotonic_ns=now,
            oldest_sample_age_ns=age,
            motors=[
                p.StateMotor(
                    group="limb",
                    motor_name=f"motor_{i}",
                    position=q,
                    temperature=25,
                    sample_monotonic_ns=stamp,
                    sample_age_ns=age,
                )
                for i, q in enumerate(self.positions)
            ],
            groups=[
                p.StateGroup(
                    group="limb",
                    state=self.state,
                    sample_monotonic_ns=stamp,
                    sample_age_ns=age,
                )
            ],
            imus=[
                p.StateImu(
                    imu_name="imu",
                    orientation=p.Quaternion(w=1),
                    angular_velocity=p.Vector3(),
                    linear_acceleration=p.Vector3(),
                    sample_monotonic_ns=stamp,
                    sample_age_ns=age,
                )
            ],
        )

    def AcquireLease(self, request, context):
        self.calls.append("acquire")
        return self.pb.AcquireLeaseResponse(
            group="limb", token="synthetic-token", expires_in_ms=self.lease_duration_ms
        )

    def RenewLease(self, request, context):
        if self.failure == "lease":
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION, "synthetic lease expired"
            )
        return self.pb.RenewLeaseResponse(
            group="limb", token="synthetic-token", expires_in_ms=self.lease_duration_ms
        )

    def ReleaseLease(self, request, context):
        self.calls.append("release")
        if self.state == self.pb.GROUP_STATE_RUNNING:
            self.state = self.pb.GROUP_STATE_DAMPED
        return self.pb.ReleaseLeaseResponse(state=self.state)

    def EnableGroup(self, request, context):
        self.calls.append("enable")
        self.state = self.pb.GROUP_STATE_DAMPED
        return self.pb.EnableGroupResponse(state=self.state)

    def StartControl(self, request, context):
        self.calls.append("start")
        self.state = self.pb.GROUP_STATE_RUNNING
        self.apply(request.commands)
        if self.failure == "start_timeout":
            time.sleep(0.5)
        return self.pb.StartControlResponse(state=self.state)

    def StopControl(self, request, context):
        self.calls.append("stop")
        self.state = self.pb.GROUP_STATE_DAMPED
        return self.pb.StopControlResponse(state=self.state)

    def DisableGroup(self, request, context):
        self.calls.append("disable")
        self.state = self.pb.GROUP_STATE_IDLE
        return self.pb.DisableGroupResponse(state=self.state)

    def apply(self, commands):
        assert len(commands) == len(self.names)
        assert {c.motor_name for c in commands} == {
            f"motor_{i}" for i in range(len(self.names))
        }
        for command in commands:
            self.positions[int(command.motor_name.removeprefix("motor_"))] = (
                command.position
            )
        self.targets.append(self.positions.copy())

    def reference(self, request, context):
        if self.reference_started is None:
            self.reference_started = time.monotonic()
        live = self.reference_mode is not None
        if (
            self.reference_mode == "outage"
            and time.monotonic() - self.reference_started > 0.3
        ):
            context.abort(grpc.StatusCode.UNAVAILABLE, "synthetic producer outage")
        return reference_pb.GetReferenceResponse(
            robot="synthetic",
            available=live,
            sequence=time.monotonic_ns(),
            source_age_ms=1000 if self.reference_mode == "stale_reference" else 0,
            joint_names=["wrong" if self.reference_mode == "bad_reference" else "held"],
            position=[-0.1],
            velocity=[100.0 if self.reference_mode == "unsafe_reference" else 0.0],
            torso_command=(
                []
                if self.reference_mode == "missing_torso"
                else [
                    float("nan") if self.reference_mode == "nan_torso" else 1.0,
                    -1.0,
                    1.0,
                ]
            ),
        )

    def StreamCommands(self, requests, context):
        context.send_initial_metadata(())
        for request in requests:
            self.calls.append("command")
            self.apply(request.commands)
            yield self.pb.StreamCommandsResponse(
                group="limb",
                sequence=request.sequence,
                state=self.state,
                result=self.pb.COMMAND_RESULT_UNSAFE
                if self.failure == "reject"
                else self.pb.COMMAND_RESULT_ACCEPTED,
            )


def start_services(fake, *, pose_enabled=True):
    pool = ThreadPoolExecutor(max_workers=8)
    server = grpc.server(pool)
    methods = {}
    for method in fake.pb.DESCRIPTOR.services_by_name["HardwareService"].methods:
        handler = getattr(fake, method.name, None)
        if handler is None:
            continue
        make = (
            grpc.stream_stream_rpc_method_handler
            if method.client_streaming
            else grpc.unary_stream_rpc_method_handler
            if method.server_streaming
            else grpc.unary_unary_rpc_method_handler
        )
        methods[method.name] = make(
            handler,
            request_deserializer=GetMessageClass(method.input_type).FromString,
            response_serializer=lambda m: m.SerializeToString(),
        )
    server.add_generic_rpc_handlers(
        [
            grpc.method_handlers_generic_handler(
                "bello.hardware.v1.HardwareService", methods
            ),
            grpc.method_handlers_generic_handler(
                "bello.reference.v1.ReferenceService",
                {
                    "GetReference": grpc.unary_unary_rpc_method_handler(
                        fake.reference,
                        request_deserializer=reference_pb.GetReferenceRequest.FromString,
                        response_serializer=lambda m: m.SerializeToString(),
                    )
                },
            ),
        ]
    )
    if pose_enabled and hasattr(fake, "pose"):
        server.add_generic_rpc_handlers(
            [
                grpc.method_handlers_generic_handler(
                    "bello.pose.v1.PoseService",
                    {
                        "WatchPose": grpc.unary_stream_rpc_method_handler(
                            fake.pose,
                            request_deserializer=pose_pb.WatchPoseRequest.FromString,
                            response_serializer=lambda m: m.SerializeToString(),
                        ),
                    },
                ),
            ]
        )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, pool, f"http://127.0.0.1:{port}"


def bundle(
    source, *, metadata_updates=None, physics=b"<mujoco/>", motion=None, target=None
):
    metadata = dict(
        git_commit=subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        task="locomotion",
        robot="synthetic",
        observation_size=2,
        observation_frame=[["last_action", 2]],
        observation_stack_size=1,
        observation_max_abs=100,
        coordinates=["first", "held", "last"],
        policy_indices=[0, 2],
        default_position=[0.2, -0.3, 0.4],
        kp=[2] * 3,
        kd=[0.1] * 3,
        control_timestep=0.02,
        joint_from_motor=np.eye(3).tolist(),
        motor_from_joint=np.eye(3).tolist(),
        joint_names=["first", "held", "last"],
        joint_limits=[[-3, 3]] * 3,
        motor_limits=[[-3, 3]] * 3,
        position_step_limits=[2.0] * 3,
        default_joint_position=[0.2, -0.3, 0.4],
        policy_from_hardware=np.eye(3).tolist(),
        left_axis=[0, 1],
        hardware_imu_frame="imu",
        action_clip=1,
        environment=to_builtin(BelloEnvConfig()),
        command=to_builtin(CommandConfig()),
    )
    metadata.update(metadata_updates or {})
    count = len(metadata["policy_indices"])
    nodes = [
        helper.make_node(
            "Constant",
            [],
            [name],
            value=numpy_helper.from_array(np.asarray([values], dtype=np.float32)),
        )
        for name, values in [
            ("action", [0] * count),
            ("motor_target", target or metadata["default_position"]),
        ]
    ]
    graph = helper.make_graph(
        nodes,
        "synthetic-hold",
        [
            helper.make_tensor_value_info(
                "observation", onnx.TensorProto.FLOAT, [1, metadata["observation_size"]]
            )
        ],
        [
            helper.make_tensor_value_info("action", onnx.TensorProto.FLOAT, [1, count]),
            helper.make_tensor_value_info(
                "motor_target",
                onnx.TensorProto.FLOAT,
                [1, len(metadata["coordinates"])],
            ),
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10
    )
    helper.set_model_props(model, {"bello": json.dumps(metadata)})
    return deployment_bundle(model.SerializeToString(), physics=physics, motion=motion)


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "reject",
        "stale",
        "abort",
        "interrupt",
        "terminate",
        "start_timeout",
        "lease",
        *FOLLOWING_CASES,
        "disconnect",
        "limited",
        "startup_quantization",
        "startup_margin",
    ],
)
def test_native_operator_lifecycle_and_failure_cleanup(
    rust_policy, hardware_proto, tmp_path, failure
):
    _, _, source = rust_policy
    fake = SyntheticHardware(hardware_proto)
    if failure == "lease":
        fake.lease_duration_ms = 500
    if failure == "startup_quantization":
        fake.positions[0] = -0.00019180590524214836
    if failure == "startup_margin":
        fake.positions[0] = -0.002
    server, pool, endpoint = start_services(fake)
    policy = tmp_path / "policy.zip"
    limits = (
        dict(
            joint_from_motor=[[0.5, 0.5, 0], [-0.5, 0.5, 0], [0, 0, 1]],
            joint_limits=[[-0.4, 0.6], [-0.3, 0.5], [-1, 2]],
            position_step_limits=[0.1, 0.2, 0.3],
        )
        if failure == "limited"
        else None
    )
    if failure in ("startup_quantization", "startup_margin"):
        limits = dict(motor_limits=[[0, 1.2], [-3, 3], [-3, 3]])
    if failure in FOLLOWING_CASES:
        limits = dict(
            command=to_builtin(
                CommandConfig(
                    torso=TorsoCommandConfig(
                        roll=(-0.1, 0.1),
                        pitch=(-0.2, 0.2),
                        yaw=(-0.3, 0.3),
                        slew_rate=(1.0, 2.0, 3.0),
                    )
                )
            ),
            observation_size=5,
            observation_frame=[["last_action", 2], ["torso_command", 3]],
        )
    policy.write_bytes(
        bundle(
            source,
            metadata_updates=limits,
            target=[1.5, 1.5, 0.4] if failure == "limited" else None,
        )
    )
    settings = yaml.safe_load(
        (
            Path(__file__).parents[1] / "rust/bello-policy/config/hardware.yaml"
        ).read_text()
    )
    settings.update(endpoint=endpoint, groups=["limb"])
    if failure == "startup_margin":
        settings["startup_feedback_margin_radians"] = 0.002
    settings["control"].update(ramp_seconds=0.08)
    log_path = tmp_path / "policy.jsonl"
    settings["control"]["policy_log_path"] = str(log_path)
    operator_address = operator_endpoint()
    settings["control"]["operator_listen"] = operator_address
    if failure in FOLLOWING_CASES:
        settings["control"].update(
            following=dict(
                enabled=True,
                endpoint=endpoint,
                rpc_timeout_seconds=0.05,
                poll_period_seconds=0.01,
                max_reference_age_ms=100.0,
                max_reference_joint_speed_rps=2.0,
                min_live_joint_limit_margin_radians=0.001,
                max_target_speed_rps=1.0,
                connection_transition_duration_seconds=0.1,
            ),
        )
    if failure == "start_timeout":
        settings["control"]["rpc_timeout_seconds"] = 0.2
        fake.failure = failure
    config = tmp_path / "control.json"
    config.write_text(json.dumps(settings))
    master, slave = pty.openpty()
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
        stdin=slave,
        stdout=slave,
        stderr=slave,
        start_new_session=True,
    )
    os.close(slave)
    transcript = bytearray()

    def wait_for(text):
        deadline = time.monotonic() + 5
        while text.encode() not in transcript:
            assert time.monotonic() < deadline, transcript.decode(errors="replace")
            assert process.poll() is None, transcript.decode(errors="replace")
            if select.select([master], [], [], 0.05)[0]:
                transcript.extend(os.read(master, 4096))

    operator = None
    try:
        wait_for("Operator service listening")
        assert not fake.calls
        operator = OperatorClient(operator_address, fake)
        if failure == "start_timeout":
            operator.press(2.1, back=True, start=True)
        else:
            operator.stand()
            assert "start" in fake.calls and "disable" not in fake.calls
            operator.select("Locomotion")
            operator.press(1.1, a=True)
            operator.wait(lambda status: status.mode == operator_pb.MODE_LOCOMOTION)
            fake.failure = failure if failure in ("reject", "stale", "lease") else None
            fake.abort = failure == "abort"
            if failure in FOLLOWING_CASES:
                fake.reference_mode = failure
                fake.reference_started = time.monotonic()
                assert operator.status.tracking_available
                assert not operator.status.tracking_enabled
                operator.press(y=True)
                if failure == "follow":
                    operator.wait(lambda status: status.tracking_live)
                    time.sleep(0.3)
                    operator.press(y=True)
                    operator.wait(lambda status: not status.tracking_enabled)
                    time.sleep(0.4)
                    np.testing.assert_allclose(
                        fake.targets[-1], [0.2, -0.3, 0.4], atol=1e-6
                    )
                    operator.press(y=True)
                    operator.wait(lambda status: status.tracking_live)
            if failure == "disconnect":
                operator.close()
                operator = None
            if failure in ("interrupt", "terminate"):
                os.kill(
                    process.pid,
                    signal.SIGINT if failure == "interrupt" else signal.SIGTERM,
                )
        if failure in (
            None,
            "follow",
            "outage",
            "stale_reference",
            "unsafe_reference",
            "disconnect",
            "limited",
            "startup_quantization",
            "startup_margin",
        ):
            time.sleep(1.5)
            assert "release" not in fake.calls
            process.send_signal(signal.SIGTERM)
        deadline = time.monotonic() + 8
        while "release" not in fake.calls:
            assert time.monotonic() < deadline, fake.calls
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
        # PTY EOF can arrive before waitpid observes process exit.
        code = process.wait(timeout=3)
        assert code != 0, transcript.decode(errors="replace")
        assert fake.calls[-1] == "release"
        assert "disable" not in fake.calls
        assert fake.state == fake.pb.GROUP_STATE_DAMPED
        log = [json.loads(line) for line in log_path.read_text().splitlines()]
        if failure is None:
            assert_motor_rail_log([row for row in log if "policy" in row])
        if failure in ("startup_quantization", "startup_margin"):
            targets = np.asarray(fake.targets)
            assert np.all(targets[:, 0] >= 0)
            assert 0 <= targets[0, 0] < 0.001
            startup = next(row for row in log if row.get("event") == "startup_target")
            assert startup["measured_motor_position"][0] < 0
            assert startup["motor_target"][0] == targets[0, 0]
        if failure == "reject":
            assert any(
                row.get("event") == "command_rejected"
                and "requested_motor_target" in row["command"]
                for row in log
            )
        if failure == "limited":
            targets = np.asarray(fake.targets)
            joint = targets @ np.asarray(limits["joint_from_motor"]).T
            assert np.all(joint >= np.asarray(limits["joint_limits"])[:, 0])
            assert np.all(joint <= np.asarray(limits["joint_limits"])[:, 1])
            assert np.all(
                np.abs(np.diff(targets, axis=0))
                <= np.asarray(limits["position_step_limits"]) + 1e-7
            )
            assert any(row.get("event") == "target_limited" for row in log)
        if failure in ("follow", "outage"):
            targets = np.asarray(fake.targets)
            np.testing.assert_allclose(targets[:, 0], 0.2)
            np.testing.assert_allclose(targets[:, 2], 0.4)
            assert targets[:, 1].max() > -0.25
            assert np.abs(np.diff(targets[:, 1])).max() <= 0.020001
            assert abs(targets[-1, 1] - (-0.3 if failure == "outage" else -0.1)) < 0.02
            ticks = [row for row in log if "following_live" in row]
            desired = np.asarray([row["feedback"]["torso_command"] for row in ticks])
            actual = np.asarray([row["policy"]["torso_command"] for row in ticks])
            live = np.asarray([row["following_live"] for row in ticks])
            np.testing.assert_allclose(
                desired[live], np.tile([0.1, -0.2, 0.3], (live.sum(), 1))
            )
            np.testing.assert_array_equal(desired[~live], 0.0)
            assert np.all(
                np.abs(np.diff(actual, axis=0)) <= np.array([0.02, 0.04, 0.06]) + 1e-6
            )
            assert np.max(np.abs(actual)) > 0.1
            np.testing.assert_allclose(
                actual[-1],
                [0, 0, 0] if failure == "outage" else [0.1, -0.2, 0.3],
                atol=1e-6,
            )
            for row in ticks:
                np.testing.assert_allclose(
                    row["policy"]["observation"][-3:], row["policy"]["torso_command"]
                )
        if failure in ("stale_reference", "unsafe_reference"):
            ticks = [row for row in log if "following_live" in row]
            assert ticks and all(not row["following_live"] for row in ticks)
            for row in ticks:
                np.testing.assert_array_equal(row["policy"]["torso_command"], 0.0)
            np.testing.assert_allclose(
                fake.targets,
                np.tile([0.2, -0.3, 0.4], (len(fake.targets), 1)),
                atol=1e-6,
            )
    finally:
        if operator is not None:
            operator.close()
        if process.poll() is None:
            os.kill(process.pid, signal.SIGKILL)
        process.wait(timeout=3)
        os.close(master)
        fake.closed.set()
        server.stop(0).wait()
        pool.shutdown(wait=True)


def assert_motor_rail_log(rows):
    assert rows and all("motor_rails" in row for row in rows)
    readings = [row["motor_rails"]["supply"] for row in rows]
    received = [value for value in readings if value is not None]
    assert received
    for value in received:
        assert value["voltage_v"] == 48.5
        assert value["current_a"] == -1.25
        assert 0 <= value["sample_age_ms"] < 3000
        assert value["stale"] is False
        assert value["sequence"] == 42
        assert value["connection_generation"] == 1


@pytest.mark.parametrize("position", [-0.00019180590524214836, -0.002])
def test_readonly_preflight_quantization_tolerance(
    rust_policy, hardware_proto, position, tmp_path
):
    run, _, source = rust_policy
    fake = SyntheticHardware(hardware_proto)
    fake.positions[0] = position
    server, pool, endpoint = start_services(fake)
    settings = yaml.safe_load(
        (
            Path(__file__).parents[1] / "rust/bello-policy/config/hardware.yaml"
        ).read_text()
    )
    settings.update(endpoint=endpoint, groups=["limb"])
    settings["control"]["operator_listen"] = operator_endpoint()
    try:
        result = run(
            content=bundle(
                source, metadata_updates=dict(motor_limits=[[0, 1.2], [-3, 3], [-3, 3]])
            ),
            hardware_config=settings,
        )
        if position > -0.001:
            assert result.returncode == 0, result.stderr
            assert json.loads(result.stdout)["startup_motor_target"][0] >= 0
        else:
            assert result.returncode != 0
            assert "startup feedback constraint" in result.stderr
            armed = subprocess.run(
                [
                    os.environ["BELLO_POLICY_BIN"],
                    "--policy",
                    str(tmp_path / "deployment.zip"),
                    "--source",
                    str(source),
                    "--hardware-config",
                    str(tmp_path / "hardware.yaml"),
                    "--arm",
                ],
                stdin=subprocess.DEVNULL,
                text=True,
                capture_output=True,
                timeout=5,
            )
            assert armed.returncode != 0
            assert "startup feedback constraint" in armed.stderr
        assert not fake.calls
    finally:
        fake.closed.set()
        server.stop(0).wait()
        pool.shutdown(wait=True)
