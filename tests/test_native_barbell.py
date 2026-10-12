"""Console intent -> Rust sequence -> real hand API, over loopback synthetic devices."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
import subprocess
import threading
import time
from types import SimpleNamespace

import grpc
import pytest

from bello_mujoco.hardware import _operator_pb2 as pb
from bello_mujoco.hardware import hand_pb2 as hand_pb
from bello_mujoco.hardware.hand_server import (
    HandController,
    HandServerConfig,
    hand_service_handler,
)
from bello_mujoco.hardware.xrobot_hands import XRobotHandsConfig
from bello_mujoco.hardware.pico_grpc import PicoGrpcConfig
from native_operator import OperatorClient
from test_native_control import bundle, start_services
from test_native_control import hardware_proto as hardware_proto
from test_native_motion_control import MotionHardware, motion_bundle, control_settings

pytestmark = pytest.mark.integration


class Hands:
    config = SimpleNamespace(feedback_timeout_seconds=0.5)

    def __init__(self):
        self.positions = {side: (200,) * 6 for side in ("left", "right")}
        self.stamps = {side: time.monotonic() for side in self.positions}
        self.writes = []
        self.failed = False

    def feedback(self, side):
        return self.positions[side], self.stamps[side]

    def poll_feedback(self, side):
        if self.failed:
            raise OSError("synthetic CAN failure")
        self.stamps[side] = time.monotonic()

    def refresh_positions(self, side):
        self.poll_feedback(side)
        return self.positions[side]

    def set_positions(self, side, positions):
        self.positions[side] = tuple(positions)
        self.writes.append((side, tuple(positions)))


@pytest.mark.parametrize(
    "failure", [None, "cancel", "disconnect", "hand_fault", "busy"]
)
def test_console_barbell_cycle_and_ownership(
    rust_policy, hardware_proto, tmp_path, failure
):
    _, _, source = rust_policy
    fake = MotionHardware(hardware_proto)
    server, pool, endpoint = start_services(fake)
    transport = Hands()
    hand_controller = HandController(
        replace(
            XRobotHandsConfig(xrobot=PicoGrpcConfig(endpoint="127.0.0.1:1")),
            arm=True,
            calibration_confirmed=True,
            smoothing_time_constant_ms=0,
            max_raw_rate_per_second=1000,
        ),
        HandServerConfig(),
        transport,
    )
    hand_controller.tick()
    occupied = (
        hand_controller.acquire(hand_pb.Hand(side=hand_pb.RIGHT))
        if failure == "busy"
        else None
    )
    closed = threading.Event()

    def tick():
        while not closed.wait(0.01):
            hand_controller.tick()

    thread = threading.Thread(target=tick, daemon=True)
    thread.start()
    hand_pool = ThreadPoolExecutor(max_workers=4)
    hand_server = grpc.server(hand_pool)
    hand_server.add_generic_rpc_handlers((hand_service_handler(hand_controller),))
    port = hand_server.add_insecure_port("127.0.0.1:0")
    hand_server.start()
    standing = tmp_path / "standing.zip"
    standing.write_bytes(bundle(source))
    v2 = tmp_path / "v2.zip"
    v2.write_bytes(
        bundle(
            source,
            metadata_updates=dict(
                observation_size=9,
                observation_frame=[
                    ["joint_position", 3],
                    ["joint_velocity", 3],
                    ["last_action", 2],
                    ["arm_target", 1],
                ],
                default_position=[0.2, 0.1, 0.4],
                default_joint_position=[0.2, 0.1, 0.4],
            ),
        )
    )
    motion = tmp_path / "motion.zip"
    # Match the two standing policies' 20-ms period.
    from io import BytesIO
    from zipfile import ZipFile
    import onnx

    content = motion_bundle(source, frames=11, observation_mode="proprioceptive")
    modified = BytesIO()
    with ZipFile(BytesIO(content)) as original, ZipFile(modified, "w") as archive:
        for name in original.namelist():
            data = original.read(name)
            if name == "policy.onnx":
                graph = onnx.load_model_from_string(data)
                metadata = json.loads(graph.metadata_props[0].value)
                metadata["control_timestep"] = 0.02
                graph.metadata_props[0].value = json.dumps(metadata)
                data = graph.SerializeToString()
            archive.writestr(name, data)
    motion.write_bytes(modified.getvalue())
    poses = tmp_path / "poses.json"
    poses.write_text(
        json.dumps(
            dict(
                coordinates=["first", "held", "last"],
                normal=[0.2, -0.3, 0.4],
                unloaded_forward=[0.2, 0.0, 0.4],
                loaded_forward=[0.2, 0.1, 0.4],
            )
        )
    )
    settings = control_settings(endpoint, standing, source, tmp_path / "policy.jsonl")
    settings["motions"] = []
    settings["control"]["max_loop_lag_seconds"] = 0.5
    settings["barbell"] = dict(
        standing=dict(policy=str(v2), source=str(source)),
        motion=dict(policy=str(motion), source=str(source)),
        motion_settings=dict(
            start_time_seconds=0,
            transition_seconds=0.2,
            max_tilt_error_radians=0.5,
            max_raw_action_abs=5,
        ),
        arm_poses=str(poses),
        grip_adjustment_per_second=0.5,
        timing=dict(
            initial_stand_seconds=0.2,
            reach_seconds=0.2,
            load_transfer_seconds=0.2,
            policy_blend_seconds=0.2,
            motion_seconds=0.6,
            retract_seconds=0.2,
            final_stand_seconds=0.2,
        ),
        hands=dict(
            endpoint=f"http://127.0.0.1:{port}",
            rpc_timeout_seconds=0.05,
            update_seconds=0.05,
            stale_seconds=0.2,
            acquisition_timeout_seconds=2,
            max_raw_rate_per_second=1000,
        ),
    )
    config = tmp_path / "hardware.json"
    config.write_text(json.dumps(settings))
    output = tmp_path / "runner.log"
    client = None
    with output.open("w") as log:
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
            stdout=log,
            stderr=log,
        )
        try:
            client = OperatorClient(settings["control"]["operator_listen"], fake)
            client.stand()
            if occupied:
                # Renew immediately before entry; hold confirmation lasts one second.
                def renew():
                    while not closed.wait(0.1):
                        hand_controller.keep_alive(occupied)

                # Reacquire after normal startup (the initial lease has expired).
                occupied = hand_controller.acquire(hand_pb.Hand(side=hand_pb.RIGHT))
                threading.Thread(target=renew, daemon=True).start()
            client.select("Barbell")
            client.press(1.1, a=True)
            if failure == "busy":
                client.wait(lambda s: s.mode == pb.MODE_FAULT)
                assert not hand_controller.status().hands[0].owned
                return
            client.wait(
                lambda s: (
                    s.barbell.awaiting == "bar_ready" and s.barbell.confirmation_ready
                )
            )
            client.press(1.1, a=True)
            client.wait(
                lambda s: (
                    s.barbell.awaiting == "grasp_secured" and s.barbell.grip_adjustable
                )
            )
            assert client.status.barbell.grip_closure == 0
            assert not client.status.barbell.confirmation_ready
            if failure is None:
                client.press(1.1, a=True)
                assert client.status.barbell.awaiting == "grasp_secured"
            client.buttons = {"forward": 1.0}
            client.wait(lambda s: s.barbell.grip_closure >= 0.7)
            client.buttons = {"forward": -0.5}
            client.wait(lambda s: s.barbell.grip_closure <= 0.55)
            client.press()
            client.wait(lambda s: s.barbell.confirmation_ready)
            grip = client.status.barbell.grip_closure
            assert 0.4 < grip < 0.6
            time.sleep(0.2)
            assert client.status.barbell.grip_closure == grip
            assert all(
                list(h.target) == pytest.approx([grip] * 6)
                for h in hand_controller.status().hands
            )
            if failure == "hand_fault":
                before_fault = len(transport.writes)
                transport.failed = True
                client.wait(lambda s: s.mode == pb.MODE_FAULT)
                time.sleep(0.1)
                assert len(transport.writes) == before_fault
                assert all(not h.owned for h in hand_controller.status().hands)
                return
            if failure in ("cancel", "disconnect"):
                if failure == "cancel":
                    client.press(1.1, y=True)
                else:
                    client.connected = False
                    client.wait(lambda s: s.phase == "interrupted")
                    client.connected = True
                    client.wait(lambda s: not s.neutral_required)
                client.wait(lambda s: s.barbell.awaiting == "bar_supported")
            else:
                # A held across a phase change must not confirm the next gate.
                client.buttons = {"a": True}
                client.wait(lambda s: not s.barbell.grip_adjustable)
                # Stick motion cannot change the latched grip or command walking.
                client.buttons = {"a": True, "forward": -1.0, "lateral": 1.0}
                client.wait(lambda s: s.barbell.awaiting == "bar_supported")
                client.buttons = {"a": True}
                time.sleep(1.3)
                assert client.status.barbell.awaiting == "bar_supported"
                assert all(
                    list(h.target) == pytest.approx([grip] * 6)
                    for h in hand_controller.status().hands
                )
                client.press()
            assert client.status.barbell.grip_closure == grip
            assert not client.status.barbell.grip_adjustable
            for event in ("bar_supported", "hands_open", "bar_clear"):
                client.wait(
                    lambda s, event=event: (
                        s.barbell.awaiting == event and s.barbell.confirmation_ready
                    )
                )
                generation = client.status.barbell.phase_id
                client.buttons = {"a": True}
                client.wait(
                    lambda s, generation=generation: s.barbell.phase_id != generation,
                    timeout=4,
                )
                client.press()
            client.wait(lambda s: s.mode == pb.MODE_STAND)
            assert all(
                not h.owned and h.source == hand_pb.HOLD
                for h in hand_controller.status().hands
            )
            rows = [
                json.loads(line)
                for line in (tmp_path / "policy.jsonl").read_text().splitlines()
            ]
            phases = {
                r["barbell"]["sequence"]["phase"]
                for r in rows
                if "sequence" in r.get("barbell", {})
            }
            assert {"reach", "close", "unload", "retract", "finish"} <= phases
            assert ("motion" in phases) == (failure is None)
            for row in rows:
                command = row.get("barbell", {}).get("sequence")
                if command and command["phase"] in {
                    "load",
                    "enter_motion",
                    "motion",
                    "return",
                    "support",
                    "unload",
                    "interrupted",
                }:
                    assert command["closure"] == grip
                    assert row["feedback"]["command"] == [0.0, 0.0, 0.0]

        except Exception as error:
            raise AssertionError(
                output.read_text()
                + "\n"
                + str(client.status if client else "no operator")
            ) from error
        finally:
            if client:
                client.close()
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            closed.set()
            thread.join(timeout=2)
            hand_server.stop(0).wait()
            hand_pool.shutdown(wait=True)
            fake.closed.set()
            server.stop(0).wait()
            pool.shutdown(wait=True)


@pytest.mark.parametrize("invalid", [None, "mass", "screen", "mapping"])
def test_motion_bundle_with_bar_before_robot_validates_and_infers(rust_policy, invalid):
    """Exercise nonzero root addresses and object quaternions, not just loading."""
    from io import BytesIO
    from zipfile import ZipFile
    import xml.etree.ElementTree as ET

    import numpy as np
    import onnx

    run, _, source = rust_policy
    content = motion_bundle(source, frames=3, observation_mode="proprioceptive")
    modified = BytesIO()
    with ZipFile(BytesIO(content)) as original, ZipFile(modified, "w") as archive:
        for name in original.namelist():
            data = original.read(name)
            if name == "physics.xml":
                root = ET.fromstring(data)
                bar = ET.Element("body", name="barbell")
                ET.SubElement(bar, "freejoint")
                ET.SubElement(bar, "geom", type="sphere", size=".1", mass="10")
                root.find("worldbody").insert(0, bar)
                data = ET.tostring(root)
            elif name == "policy.onnx":
                graph = onnx.load_model_from_string(data)
                metadata = json.loads(graph.metadata_props[0].value)
                metadata["root_qpos_adr"] = 7
                graph.metadata_props[0].value = json.dumps(metadata)
                data = graph.SerializeToString()
            elif name == "motion.npz":
                arrays = dict(np.load(BytesIO(data)))
                arrays["qpos"] = np.concatenate(
                    (
                        np.tile(np.array([0, 0, 1, 1, 0, 0, 0], np.float32), (3, 1)),
                        arrays["qpos"],
                    ),
                    axis=1,
                )
                arrays["qvel"] = np.concatenate(
                    (np.zeros((3, 6), np.float32), arrays["qvel"]), axis=1
                )
                arrays.update(
                    free_joint_qpos=np.array([0, 7], np.int64),
                    joint_qpos_indices=np.array([14, 15, 16], np.int64),
                    joint_qvel_indices=np.array([12, 13, 14], np.int64),
                    bar_mass_kg=np.array(
                        9.0 if invalid == "mass" else 10.0, np.float64
                    ),
                    screen_passed=np.array(invalid != "screen"),
                )
                if invalid == "mapping":
                    arrays["joint_qpos_indices"] = np.array([15, 14, 16], np.int64)
                buffer = BytesIO()
                np.savez(buffer, **arrays)
                data = buffer.getvalue()
            archive.writestr(name, data)
    content = modified.getvalue()
    checked = run(content=content, check=True)
    if invalid:
        assert checked.returncode != 0
        assert {"mass": "mass", "screen": "screen", "mapping": "mapping"}[
            invalid
        ] in checked.stderr
        return
    assert checked.returncode == 0, checked.stderr
    result = run(
        [
            dict(
                motor_position=[0.2, -0.3, 0.4],
                motor_velocity=[0.0, 0.0, 0.0],
                orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
                angular_velocity_local=[0.0, 0.0, 0.0],
                reference_time=0.05,
                initialize=True,
            )
        ],
        content=content,
        telemetry=True,
    )
    assert result.returncode == 0, result.stderr
    assert np.isfinite(json.loads(result.stdout)["observation"]).all()
