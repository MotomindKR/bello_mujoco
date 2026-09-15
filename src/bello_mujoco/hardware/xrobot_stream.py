"""Direct XRoboToolkit body targets for upstream GMR's xrobot convention."""

import math
import time
from types import MappingProxyType

import numpy as np
from scipy.spatial.transform import Rotation

from bello_mujoco.hardware import _smplx_pb2 as pb
from bello_mujoco.hardware.gmr_retarget import HumanTargetFrame
from bello_mujoco.hardware.pico_grpc import PicoGrpcConfig, PicoGrpcStream
from bello_mujoco.hardware.reference import ReferenceUnavailable


BODY_NAMES = (
    "Pelvis",
    "Left_Hip",
    "Right_Hip",
    "Spine1",
    "Left_Knee",
    "Right_Knee",
    "Spine2",
    "Left_Ankle",
    "Right_Ankle",
    "Spine3",
    "Left_Foot",
    "Right_Foot",
    "Neck",
    "Left_Collar",
    "Right_Collar",
    "Head",
    "Left_Shoulder",
    "Right_Shoulder",
    "Left_Elbow",
    "Right_Elbow",
    "Left_Wrist",
    "Right_Wrist",
    "Left_Hand",
    "Right_Hand",
)
# Same proper rotation as upstream XRobotStreamer: (x, y, z) -> (x, -z, y).
# The installed client already cancels the native/Unity reflections on the wire.
TRACKING_TO_GMR = Rotation.from_matrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]])


def parse_frame(message: pb.PoseFrame, config: PicoGrpcConfig) -> HumanTargetFrame:
    if not message.upstream_connected or message.sequence == 0:
        raise ReferenceUnavailable("XRoboToolkit upstream is unavailable")
    if not message.HasField("xrobot_body"):
        raise ValueError(
            "Tracking bridge must publish typed xrobot_body; upgrade it first"
        )
    if message.source_timestamp_ns <= 0 or not message.device_id:
        raise ValueError("XRoboToolkit frame requires a timestamp and device identity")
    body = message.xrobot_body
    if body.frame != "pico_native_y_up":
        raise ValueError("XRoboToolkit body must use pico_native_y_up coordinates")
    for age in (message.packet_age_seconds, body.age_seconds):
        if not math.isfinite(age) or age < 0:
            raise ValueError("XRoboToolkit age must be finite and nonnegative")
    age = max(message.packet_age_seconds, body.age_seconds) * 1000
    age += config.transport_latency_budget_ms
    if not body.valid or age > config.max_source_age_ms:
        raise ReferenceUnavailable("XRoboToolkit body is invalid or stale")
    if len(body.joint_names) != len(BODY_NAMES) or set(body.joint_names) != set(
        BODY_NAMES
    ):
        raise ValueError(
            "XRoboToolkit frame must contain each of the 24 named body joints"
        )
    positions = np.asarray(body.positions, dtype=np.float64)
    quaternions = np.asarray(body.orientations, dtype=np.float64)
    if positions.shape != (72,) or not np.isfinite(positions).all():
        raise ValueError("XRoboToolkit body positions must contain 72 finite values")
    if quaternions.shape != (96,) or not np.isfinite(quaternions).all():
        raise ValueError("XRoboToolkit body orientations must contain 96 finite values")
    quaternions = quaternions.reshape(24, 4)
    if np.any(np.abs(np.linalg.norm(quaternions, axis=1) - 1) > 0.001):
        raise ValueError("XRoboToolkit body orientations must be unit WXYZ quaternions")
    positions = TRACKING_TO_GMR.apply(positions.reshape(24, 3))
    quaternions = (
        TRACKING_TO_GMR * Rotation.from_quat(quaternions, scalar_first=True)
    ).as_quat(scalar_first=True)
    positions.setflags(write=False)
    quaternions.setflags(write=False)
    return HumanTargetFrame(
        transforms=MappingProxyType(
            dict(
                zip(
                    body.joint_names,
                    zip(positions, quaternions, strict=True),
                    strict=True,
                )
            )
        ),
        sequence=message.sequence,
        source_timestamp_ns=message.source_timestamp_ns,
        source_age_ms=age,
        received_at=time.monotonic(),
        # A reference recapture is an epoch barrier even though raw poses do not
        # depend on SMPL-X calibration. Empty calibration is valid for this path.
        track_id=f"{message.device_id}:{message.calibration_id}",
    )


class XRobotGrpcStream(PicoGrpcStream[HumanTargetFrame]):
    def __init__(self, config: PicoGrpcConfig):
        super().__init__(config, parse_frame)
