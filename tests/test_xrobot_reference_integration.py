"""Opt-in direct XRoboToolkit -> GMR -> reference RPC; no licensed model or IO."""

from concurrent.futures import ThreadPoolExecutor
import threading
import time

import grpc
import numpy as np
import pytest

from bello_mujoco.hardware import _smplx_pb2 as pose_pb
from bello_mujoco.hardware import reference_pb2 as ref_pb
from bello_mujoco.hardware.reference_producer import ReferenceService
from bello_mujoco.hardware.pico_grpc import PicoGrpcConfig
from bello_mujoco.hardware.gmr_retarget import (
    GmrLiveRetargetConfig,
    GmrRetargeter,
    LiveGmrReferenceSource,
)
from bello_mujoco.hardware.xrobot_stream import BODY_NAMES, XRobotGrpcStream


@pytest.mark.integration
def test_pose_stream_retargets_and_serves_named_robot_references():
    from bello_mujoco.envs.locomotion import BelloLocomotionEnv
    from bello_mujoco.models.compiler import physical_model_config
    from bello_mujoco.robot_tasks import motion_task_definition

    env = BelloLocomotionEnv.load(
        physical_model_config(asset="bello_mini"), load_mjx=False
    )
    profile = motion_task_definition("bello_mini").gmr.for_xrobot()
    model = env.model
    held = model.hold_ctrl_indices
    changed = threading.Event()

    def poses(request, context):
        sequence = 1
        while context.is_active():
            from scipy.spatial.transform import Rotation

            rotations = np.zeros((24, 3))
            if changed.is_set():
                rotations[18:22, 1] = 0.7
            quaternions = Rotation.from_rotvec(rotations).as_quat(scalar_first=True)
            yield pose_pb.PoseFrame(
                sequence=sequence,
                source_timestamp_ns=sequence * 33_333_333,
                device_id="synthetic",
                xrobot_body=pose_pb.XRobotBody(
                    joint_names=BODY_NAMES,
                    positions=np.tile([0, 1, 0], 24),
                    orientations=quaternions.ravel(),
                    frame="pico_native_y_up",
                    valid=True,
                ),
                upstream_connected=True,
                global_orient=[0] * 3,
                body_pose=[0] * 63,
                left_hand_pose=[0] * 45,
                right_hand_pose=[0] * 45,
                root_translation=[0] * 3,
                body_valid=True,
                body_calibrated=True,
                left_hand_joint_valid=[False] * 15,
                right_hand_joint_valid=[False] * 15,
            )
            sequence += 1
            time.sleep(1 / 30)

    with ThreadPoolExecutor(max_workers=4) as executor:
        server = grpc.server(executor)
        server.add_generic_rpc_handlers(
            (
                grpc.method_handlers_generic_handler(
                    "pico.smplx.v1.PoseStream",
                    {
                        "StreamPoses": grpc.unary_stream_rpc_method_handler(
                            poses,
                            request_deserializer=pose_pb.StreamRequest.FromString,
                            response_serializer=pose_pb.PoseFrame.SerializeToString,
                        )
                    },
                ),
            )
        )
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        try:
            with XRobotGrpcStream(PicoGrpcConfig(f"127.0.0.1:{port}")) as stream:
                retargeter = GmrRetargeter(
                    env, GmrLiveRetargetConfig(), profile=profile
                )
                with LiveGmrReferenceSource(stream, retargeter) as source:
                    source.wait_until_ready(10)
                    service = ReferenceService(
                        "bello_mini",
                        source,
                        [model.actuator_joint_names[i] for i in held],
                        model.actuator_qpos_indices[held],
                        model.actuator_qvel_indices[held],
                        model.mj_model.nq,
                        model.mj_model.nv,
                    )
                    server.add_generic_rpc_handlers((service.handler(),))
                    with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                        get = channel.unary_unary(
                            "/bello.reference.v1.ReferenceService/GetReference",
                            request_serializer=ref_pb.GetReferenceRequest.SerializeToString,
                            response_deserializer=ref_pb.GetReferenceResponse.FromString,
                        )
                        request = ref_pb.GetReferenceRequest(robot="bello_mini")
                        first = get(request, timeout=2)
                        assert first.available and first.joint_names
                        changed.set()
                        deadline = time.monotonic() + 5
                        while time.monotonic() < deadline:
                            latest = get(request, timeout=2)
                            if (
                                latest.available
                                and latest.sequence > first.sequence + 5
                            ):
                                break
                            time.sleep(0.03)
                        else:
                            pytest.fail("no advancing retargeted reference")
                        assert latest.joint_names == first.joint_names
                        assert (
                            np.isfinite(latest.position).all()
                            and np.isfinite(latest.velocity).all()
                        )
                        assert 0 <= latest.source_age_ms < 500
                        assert (
                            np.max(np.abs(np.array(latest.position) - first.position))
                            > 0.01
                        )
        finally:
            server.stop(0).wait(2)
