"""Calibrated SMPL-X pose decoding for finger control."""

from dataclasses import dataclass, field
import math
from pathlib import Path
import time

import numpy as np

from bello_mujoco.hardware import _smplx_pb2 as pb
from bello_mujoco.hardware.pico_grpc import PicoGrpcConfig, PicoGrpcStream
from bello_mujoco.hardware.reference import ReferenceUnavailable


@dataclass(frozen=True)
class SmplxGrpcConfig(PicoGrpcConfig):
    neutral_model_path: Path = field(kw_only=True)

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.neutral_model_path, Path):
            raise TypeError("neutral_model_path must be a pathlib.Path")


@dataclass(frozen=True)
class SmplxPoseFrame:
    sequence: int
    source_timestamp_ns: int
    source_age_ms: float
    received_at: float
    track_id: str
    rotations: np.ndarray
    root_translation: np.ndarray
    body_valid: bool
    left_valid: bool
    right_valid: bool

    def age_ms(self) -> float:
        return self.source_age_ms + (time.monotonic() - self.received_at) * 1000


def parse_frame(message: pb.PoseFrame, config: PicoGrpcConfig) -> SmplxPoseFrame:
    if (
        not message.upstream_connected
        or not message.sequence
        or message.source_timestamp_ns <= 0
    ):
        raise ReferenceUnavailable("SMPL-X upstream tracking is unavailable")
    if not message.device_id or not message.calibration_id:
        raise ReferenceUnavailable("SMPL-X stream has no calibrated tracking identity")
    age = message.packet_age_seconds * 1000 + config.transport_latency_budget_ms
    if not math.isfinite(age) or message.packet_age_seconds < 0:
        raise ValueError("SMPL-X packet age must be finite and nonnegative")
    if age > config.max_source_age_ms:
        raise ReferenceUnavailable("SMPL-X source frame is stale")

    def array(values, count, label):
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (count,) or not np.all(np.isfinite(result)):
            raise ValueError(f"SMPL-X {label} must contain {count} finite values")
        return result

    root = array(message.global_orient, 3, "global orientation")
    body = array(message.body_pose, 63, "body pose")
    left = array(message.left_hand_pose, 45, "left hand pose")
    right = array(message.right_hand_pose, 45, "right hand pose")
    translation = array(message.root_translation, 3, "root translation")
    for mask in (message.left_hand_joint_valid, message.right_hand_joint_valid):
        if len(mask) != 15:
            raise ValueError("SMPL-X hand validity must contain 15 flags")
    body_valid = message.body_valid and message.body_calibrated
    left_valid = message.left_hand_calibrated and all(message.left_hand_joint_valid)
    right_valid = message.right_hand_calibrated and all(message.right_hand_joint_valid)
    rotations = np.concatenate((root, body, np.zeros(9), left, right)).reshape(55, 3)
    if not body_valid:
        rotations[:22] = 0
        translation = np.zeros(3)
    if not left_valid:
        rotations[25:40] = 0
    if not right_valid:
        rotations[40:55] = 0
    rotations.setflags(write=False)
    translation.setflags(write=False)
    return SmplxPoseFrame(
        message.sequence,
        message.source_timestamp_ns,
        age,
        time.monotonic(),
        f"{message.device_id}:{message.calibration_id}",
        rotations,
        translation,
        body_valid,
        left_valid,
        right_valid,
    )


class SmplxGrpcStream(PicoGrpcStream[SmplxPoseFrame]):
    def __init__(self, config: SmplxGrpcConfig):
        super().__init__(config, parse_frame)
