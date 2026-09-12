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
from bello_mujoco.envs.rewards import CommandConfig
from bello_mujoco.hardware import _remote_pb2 as remote_pb
from bello_mujoco.training.tracking import to_builtin

pytestmark = pytest.mark.integration


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
    def __init__(self, pb):
        self.pb = pb
        self.state = pb.GROUP_STATE_IDLE
        self.positions = [0.2, -0.3, 0.4]
        self.calls = []
        self.failure = None
        self.abort = False
        self.closed = threading.Event()

    def ListHardware(self, request, context):
        p = self.pb
        status = p.DeviceStatus(state=p.DEVICE_STATE_ONLINE)
        addresses = [p.MotorAddress(bus="synthetic", id=i + 1) for i in range(3)]
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
                for i, name in enumerate(("first", "held", "last"))
            ],
            serial_devices=[
                p.SerialDeviceInfo(
                    name="serial", port="none", baud_rate=115200, status=status
                )
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
        )

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
            group="limb", token="synthetic-token", expires_in_ms=10000
        )

    def RenewLease(self, request, context):
        return self.pb.RenewLeaseResponse(
            group="limb", token="synthetic-token", expires_in_ms=10000
        )

    def ReleaseLease(self, request, context):
        self.calls.append("release")
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
        assert len(commands) == 3
        assert {c.motor_name for c in commands} == {f"motor_{i}" for i in range(3)}
        for command in commands:
            self.positions[int(command.motor_name[-1])] = command.position

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

    def remote(self, request, context):
        sequence = 0
        while context.is_active() and not self.closed.is_set():
            sequence += 1
            yield remote_pb.WatchCommandResponse(
                sequence=sequence,
                sample_monotonic_ns=time.monotonic_ns(),
                controller_connected=True,
                intent_armed=True,
                abort_active=self.abort,
            )
            self.closed.wait(0.01)


def start_services(fake):
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
                "bello.remote.v1.RemoteCommandService",
                {
                    "WatchCommand": grpc.unary_stream_rpc_method_handler(
                        fake.remote,
                        request_deserializer=remote_pb.WatchCommandRequest.FromString,
                        response_serializer=lambda m: m.SerializeToString(),
                    )
                },
            ),
        ]
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, pool, f"http://127.0.0.1:{port}"


def bundle(source):
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
        default_joint_position=[0.2, -0.3, 0.4],
        policy_from_hardware=np.eye(3).tolist(),
        left_axis=[0, 1],
        hardware_imu_frame="imu",
        action_clip=1,
        environment=to_builtin(BelloEnvConfig()),
        command=to_builtin(CommandConfig()),
    )
    nodes = [
        helper.make_node(
            "Constant",
            [],
            [name],
            value=numpy_helper.from_array(np.asarray([values], dtype=np.float32)),
        )
        for name, values in [("action", [0, 0]), ("motor_target", [0.2, -0.3, 0.4])]
    ]
    graph = helper.make_graph(
        nodes,
        "synthetic-hold",
        [helper.make_tensor_value_info("observation", onnx.TensorProto.FLOAT, [1, 2])],
        [
            helper.make_tensor_value_info("action", onnx.TensorProto.FLOAT, [1, 2]),
            helper.make_tensor_value_info(
                "motor_target", onnx.TensorProto.FLOAT, [1, 3]
            ),
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10
    )
    helper.set_model_props(model, {"bello": json.dumps(metadata)})
    return deployment_bundle(model.SerializeToString(), physics=b"<mujoco/>")


@pytest.mark.parametrize(
    "failure",
    [None, "reject", "stale", "abort", "interrupt", "terminate", "start_timeout"],
)
def test_native_operator_lifecycle_and_failure_cleanup(
    rust_policy, hardware_proto, tmp_path, failure
):
    _, _, source = rust_policy
    fake = SyntheticHardware(hardware_proto)
    server, pool, endpoint = start_services(fake)
    policy = tmp_path / "policy.zip"
    policy.write_bytes(bundle(source))
    settings = yaml.safe_load(
        (
            Path(__file__).parents[1] / "rust/bello-policy/config/hardware.yaml"
        ).read_text()
    )
    settings.update(endpoint=endpoint, groups=["limb"])
    settings["control"].update(ramp_seconds=0.08, max_duration_seconds=0.15)
    settings["control"]["remote"]["endpoint"] = endpoint
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

    try:
        wait_for("Type HOLD")
        assert not fake.calls
        os.write(master, b"HOLD\n")
        if failure != "start_timeout":
            wait_for("then type RUN")
            assert "start" in fake.calls and "disable" not in fake.calls
            fake.failure = failure if failure in ("reject", "stale") else None
            fake.abort = failure == "abort"
            os.write(master, b"RUN\n")
            if failure in ("interrupt", "terminate"):
                os.kill(
                    process.pid,
                    signal.SIGINT if failure == "interrupt" else signal.SIGTERM,
                )
        wait_for("Type IDLE")
        assert "stop" in fake.calls and "disable" not in fake.calls
        os.write(master, b"wrong\n")
        wait_for("Expected exact IDLE")
        assert "disable" not in fake.calls
        os.write(master, b"IDLE\n")
        code = process.wait(timeout=3)
        assert (code == 0) == (failure is None), transcript.decode(errors="replace")
        assert fake.calls[-2:] == ["disable", "release"]
    finally:
        if process.poll() is None:
            os.kill(process.pid, signal.SIGKILL)
        process.wait(timeout=3)
        os.close(master)
        fake.closed.set()
        server.stop(0).wait()
        pool.shutdown(wait=True)
