"""Lightweight SMPL-X body topology shared by live hardware transports."""

from __future__ import annotations

import numpy as np


SMPLX_BODY_JOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)

SMPLX_BODY_JOINT_PARENTS = np.asarray(
    (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19),
    dtype=np.int32,
)

SMPLX_FINGER_TIP_NAMES = (
    "left_index_tip",
    "left_middle_tip",
    "left_pinky_tip",
    "left_ring_tip",
    "left_thumb_tip",
    "right_index_tip",
    "right_middle_tip",
    "right_pinky_tip",
    "right_ring_tip",
    "right_thumb_tip",
)


__all__ = (
    "SMPLX_BODY_JOINT_NAMES",
    "SMPLX_BODY_JOINT_PARENTS",
    "SMPLX_FINGER_TIP_NAMES",
)
