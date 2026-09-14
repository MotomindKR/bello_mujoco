"""Neutral SMPL-X skeleton FK; no mesh renderer, Torch, or device SDK."""

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from bello_mujoco.hardware.smplx_contract import SMPLX_JOINT_PARENTS

# Canonical SMPL-X mesh landmark indices from vchoutas/smplx/vertex_ids.py.
_TIP_VERTICES = (4933, 5058, 5286, 5169, 5361, 7669, 7794, 8022, 7905, 8079)
_TIP_PARENTS = (27, 30, 33, 36, 39, 42, 45, 48, 51, 54)
MODEL_TO_WORLD = np.asarray(((0.0, 0.0, -1.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
MODEL_TO_WORLD_WXYZ = (0.5, 0.5, -0.5, -0.5)


def rotation_matrix_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = vector / angle
    skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def quaternion_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.asarray((1.0, 0.0, 0.0, 0.0))
    return np.concatenate(
        ([math.cos(0.5 * angle)], math.sin(0.5 * angle) / angle * vector)
    )


def quaternion_multiply(left, right) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    value = np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )
    return value / np.linalg.norm(value)


@dataclass(frozen=True)
class SkeletonPose:
    positions: np.ndarray
    quaternions: np.ndarray
    tips: np.ndarray


class SmplxKinematics:
    def __init__(self, path: Path) -> None:
        with np.load(path.expanduser().resolve(), allow_pickle=False) as model:
            template = np.asarray(model["v_template"], dtype=np.float64)
            regressor = np.asarray(model["J_regressor"], dtype=np.float64)
            parents = np.asarray(model["kintree_table"][0], dtype=np.int64).copy()
        if parents.shape != (55,):
            raise ValueError("SMPL-X model must have 55 joint parents")
        parents[0] = -1
        if tuple(parents) != SMPLX_JOINT_PARENTS:
            raise ValueError("SMPL-X model must have the canonical 55-joint hierarchy")
        if (
            template.ndim != 2
            or template.shape[1] != 3
            or regressor.shape != (55, len(template))
        ):
            raise ValueError(
                "SMPL-X model has invalid template or joint regressor dimensions"
            )
        if (
            len(template) <= max(_TIP_VERTICES)
            or not np.isfinite(template).all()
            or not np.isfinite(regressor).all()
        ):
            raise ValueError("SMPL-X model geometry is invalid")
        self.rest = regressor @ template
        self.tip_offsets = template[list(_TIP_VERTICES)] - self.rest[list(_TIP_PARENTS)]
        self.parents = parents
        # Reference pelvis height in the selected neutral model. Translation on
        # the wire is pelvis displacement, not SMPL-X's model-origin transl.
        self.pelvis_height = self.rest[0, 1] - template[:, 1].min()

    def forward(
        self, rotvecs: np.ndarray, translation: np.ndarray, *, world: bool = False
    ) -> SkeletonPose:
        rotvecs = np.asarray(rotvecs, dtype=np.float64)
        translation = np.asarray(translation, dtype=np.float64)
        if rotvecs.shape != (55, 3) or translation.shape != (3,):
            raise ValueError(
                "SMPL-X FK requires 55 rotation vectors and 3 translation values"
            )
        if not np.isfinite(rotvecs).all() or not np.isfinite(translation).all():
            raise ValueError("SMPL-X FK input must be finite")
        positions = np.empty((55, 3))
        rotations = np.empty((55, 3, 3))
        quaternions = np.empty((55, 4))
        for i, parent in enumerate(self.parents):
            rotation = rotation_matrix_from_rotvec(rotvecs[i])
            quaternion = quaternion_from_rotvec(rotvecs[i])
            if parent < 0:
                positions[i] = translation + [0.0, self.pelvis_height, 0.0]
                rotations[i] = rotation
                quaternions[i] = quaternion
            else:
                positions[i] = positions[parent] + rotations[parent] @ (
                    self.rest[i] - self.rest[parent]
                )
                rotations[i] = rotations[parent] @ rotation
                quaternions[i] = quaternion_multiply(quaternions[parent], quaternion)
        tips = np.stack(
            [
                positions[parent] + rotations[parent] @ offset
                for parent, offset in zip(_TIP_PARENTS, self.tip_offsets, strict=True)
            ]
        )
        if world:
            positions = positions @ MODEL_TO_WORLD.T
            tips = tips @ MODEL_TO_WORLD.T
            quaternions = np.stack(
                [quaternion_multiply(MODEL_TO_WORLD_WXYZ, q) for q in quaternions]
            )
        return SkeletonPose(positions, quaternions, tips)
