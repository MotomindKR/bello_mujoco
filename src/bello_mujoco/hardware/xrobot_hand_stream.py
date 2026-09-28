"""Typed native hand landmarks, independent of SMPL-X and body tracking."""

from dataclasses import dataclass
import math
import time

import numpy as np

from bello_mujoco.hardware import _smplx_pb2 as pb
from bello_mujoco.hardware.pico_grpc import PicoGrpcConfig, PicoGrpcStream
from bello_mujoco.hardware.reference import ReferenceUnavailable


HAND_NAMES = (
    "palm",
    "wrist",
    "thumb_metacarpal",
    "thumb_proximal",
    "thumb_distal",
    "thumb_tip",
    "index_metacarpal",
    "index_proximal",
    "index_intermediate",
    "index_distal",
    "index_tip",
    "middle_metacarpal",
    "middle_proximal",
    "middle_intermediate",
    "middle_distal",
    "middle_tip",
    "ring_metacarpal",
    "ring_proximal",
    "ring_intermediate",
    "ring_distal",
    "ring_tip",
    "little_metacarpal",
    "little_proximal",
    "little_intermediate",
    "little_distal",
    "little_tip",
)
FINGER_JOINTS = {
    "thumb": ("thumb_metacarpal", "thumb_proximal", "thumb_distal"),
    **{
        finger: tuple(
            f"{name}_{joint}" for joint in ("proximal", "intermediate", "distal")
        )
        for finger, name in (
            ("index", "index"),
            ("middle", "middle"),
            ("ring", "ring"),
            ("pinky", "little"),
        )
    },
}
FINGER_TIPS = {
    finger: ("little_tip" if finger == "pinky" else f"{finger}_tip")
    for finger in FINGER_JOINTS
}
REQUIRED_JOINTS = (
    ("wrist",)
    + tuple(name for names in FINGER_JOINTS.values() for name in names)
    + tuple(FINGER_TIPS.values())
)


@dataclass(frozen=True)
class HandLandmarks:
    positions: np.ndarray
    source_age_ms: float


@dataclass(frozen=True)
class XRobotHandsFrame:
    sequence: int
    source_timestamp_ns: int
    source_age_ms: float
    received_at: float
    track_id: str
    left: HandLandmarks | None
    right: HandLandmarks | None

    def age_ms(self) -> float:
        return self.source_age_ms + (time.monotonic() - self.received_at) * 1000


def parse_frame(message: pb.PoseFrame, config: PicoGrpcConfig) -> XRobotHandsFrame:
    if not message.upstream_connected or not message.sequence:
        raise ReferenceUnavailable("XRoboToolkit upstream tracking is unavailable")
    if not math.isfinite(message.packet_age_seconds) or message.packet_age_seconds < 0:
        raise ValueError("XRoboToolkit packet age must be finite and nonnegative")
    packet_age = message.packet_age_seconds * 1000 + config.transport_latency_budget_ms
    if packet_age > config.max_source_age_ms:
        raise ReferenceUnavailable("XRoboToolkit packet is stale")
    hands = {}
    for side in ("left", "right"):
        field = f"xrobot_{side}_hand"
        if not message.HasField(field):
            raise ValueError(
                f"Tracking bridge must publish typed {field}; upgrade it first"
            )
        hand = getattr(message, field)
        if hand.frame != "pico_native_y_up":
            raise ValueError("XRoboToolkit hands must use pico_native_y_up coordinates")
        if not math.isfinite(hand.age_seconds) or hand.age_seconds < 0:
            raise ValueError("XRoboToolkit hand age must be finite and nonnegative")
        age = max(
            packet_age, hand.age_seconds * 1000 + config.transport_latency_budget_ms
        )
        hands[side] = None
        if not hand.active or age > config.max_source_age_ms:
            continue
        if len(hand.joint_names) != 26 or set(hand.joint_names) != set(HAND_NAMES):
            raise ValueError("XRoboToolkit hand must contain all 26 named PICO joints")
        if len(hand.position_valid) != 26:
            raise ValueError("XRoboToolkit hand requires 26 position validity flags")
        positions = np.asarray(hand.positions, dtype=np.float64)
        if positions.shape != (78,) or not np.isfinite(positions).all():
            raise ValueError(
                "XRoboToolkit hand positions must contain 78 finite values"
            )
        indices = {name: index for index, name in enumerate(hand.joint_names)}
        if not all(hand.position_valid[indices[name]] for name in REQUIRED_JOINTS):
            continue
        positions = positions.reshape(26, 3)[[indices[name] for name in HAND_NAMES]]
        positions.setflags(write=False)
        hands[side] = HandLandmarks(positions, age)
    if not any(hands.values()):
        raise ReferenceUnavailable("No hand has fresh valid tracking")
    if message.source_timestamp_ns <= 0 or not message.device_id:
        raise ValueError("XRoboToolkit frame requires a timestamp and device identity")
    return XRobotHandsFrame(
        message.sequence,
        message.source_timestamp_ns,
        max(hand.source_age_ms for hand in hands.values() if hand is not None),
        time.monotonic(),
        message.device_id,
        hands["left"],
        hands["right"],
    )


class XRobotHandsGrpcStream(PicoGrpcStream[XRobotHandsFrame]):
    def __init__(self, config: PicoGrpcConfig):
        super().__init__(config, parse_frame)
