"""Validated human-reference contract for the released frozen SMPL encoder."""

from dataclasses import dataclass
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from bello_mujoco.envs.geometry import quat_rotate_inverse
from bello_mujoco.envs.motion_tracking import quat_conjugate, quat_multiply, rotation_6d

SMPL_JOINT_NAMES = (
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
    "left_hand",
    "right_hand",
)
SMPL_PARENTS = (
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
)
SMPL_BASE_ROTATION = Rotation.from_quat([0.5, 0.5, 0.5, 0.5])
SMPL_GEOMETRY_FORMAT = "sonic_smplx_body22_thumbs_v1"


@dataclass(frozen=True)
class SmplSkeleton:
    rest: np.ndarray
    shape_directions: np.ndarray

    @classmethod
    def load(cls, path: Path):
        """Read the neutral geometry prepared by sonic.smpl_model."""
        with np.load(path, allow_pickle=False) as data:
            if "format" not in data or str(data["format"]) != SMPL_GEOMETRY_FORMAT:
                raise ValueError(
                    "prepare SONIC neutral geometry with sonic.smpl_model first"
                )
            regressor = np.asarray(data["J_regressor"], dtype=np.float64)
            parents = np.asarray(data["kintree_table"])[0].astype(np.int64)
            parents[0] = -1
            if regressor.shape[0] != 24 or tuple(parents) != SMPL_PARENTS:
                raise ValueError(
                    "SONIC human geometry requires its prepared 24-joint tree"
                )
            rest = regressor @ data["v_template"]
            directions = np.einsum("jv,vck->jck", regressor, data["shapedirs"])
        if not np.isfinite(rest).all() or not np.isfinite(directions).all():
            raise ValueError("nonfinite SMPL model")
        return cls(rest, directions)

    def forward(self, local_quaternions, betas):
        """SMPL skeletal FK; inputs are XYZW rotations, with a world Z-up root."""
        local = np.asarray(local_quaternions, dtype=np.float64)
        betas = np.asarray(betas, dtype=np.float64)
        if local.ndim != 3 or local.shape[1:] != (24, 4) or betas.ndim != 1:
            raise ValueError("invalid SMPL pose/shape dimensions")
        if len(betas) > self.shape_directions.shape[-1]:
            raise ValueError("SMPL model has fewer shape coefficients than the source")
        if not np.isfinite(local).all() or not np.isfinite(betas).all():
            raise ValueError("nonfinite SMPL pose/shape")
        rest = self.rest + np.einsum(
            "jck,k->jc", self.shape_directions[..., : len(betas)], betas
        )
        rotations = (
            Rotation.from_quat(local.reshape(-1, 4)).as_matrix().reshape(-1, 24, 3, 3)
        )
        positions = np.zeros((len(local), 24, 3))
        world = np.empty_like(rotations)
        for joint, parent in enumerate(SMPL_PARENTS):
            if parent < 0:
                world[:, joint] = rotations[:, joint]
            else:
                world[:, joint] = world[:, parent] @ rotations[:, joint]
                positions[:, joint] = positions[:, parent] + np.einsum(
                    "fij,j->fi", world[:, parent], rest[joint] - rest[parent]
                )
        roots = (Rotation.from_quat(local[:, 0]) * SMPL_BASE_ROTATION.inv()).as_quat(
            scalar_first=True
        )
        validate_smpl_motion(positions, roots, len(local))
        return positions.astype(np.float32), roots.astype(np.float32)

    def from_xrobot(self, transforms):
        """Match the pinned SONIC PICO conversion, using calibrated SMPL geometry."""
        # XRoboToolkit capitalizes each body-role word.
        names = [
            "_".join(part.capitalize() for part in name.split("_"))
            for name in SMPL_JOINT_NAMES
        ]
        world = Rotation.from_quat(
            [transforms[name][1] for name in names], scalar_first=True
        )
        world = world * Rotation.from_euler("y", 180, degrees=True)
        local = np.empty((1, 24, 4))
        for joint, parent in enumerate(SMPL_PARENTS):
            local[0, joint] = (
                world[joint] if parent < 0 else world[parent].inv() * world[joint]
            ).as_quat()
        # The released live path uses body_pose[:63], holding the terminal hands neutral.
        local[0, 22:] = [0, 0, 0, 1]
        return self.forward(local, np.zeros(0))


@dataclass(frozen=True)
class SonicInputConfig:
    smpl_probability: float = 0.0
    smpl_joint_noise_m: float = 0.05

    def __post_init__(self):
        if (
            not math.isfinite(self.smpl_probability)
            or not 0 <= self.smpl_probability <= 1
        ):
            raise ValueError("SMPL sampling probability must be in [0, 1]")
        if not math.isfinite(self.smpl_joint_noise_m) or self.smpl_joint_noise_m < 0:
            raise ValueError("SMPL joint noise must be finite and nonnegative")

    @property
    def smpl_enabled(self) -> bool:
        return self.smpl_probability > 0


def validate_smpl_motion(smpl_joints, smpl_root_quat, frames: int) -> None:
    joints, root = np.asarray(smpl_joints), np.asarray(smpl_root_quat)
    if (
        joints.shape != (frames, 24, 3)
        or root.shape != (frames, 4)
        or not np.isfinite(joints).all()
        or not np.isfinite(root).all()
        or not np.allclose(joints[:, 0], 0, atol=1e-5)
        or not np.allclose(np.linalg.norm(root, axis=-1), 1, atol=1e-4)
    ):
        raise ValueError(
            "SMPL motion requires finite pelvis-relative 24-joint positions and unit WXYZ roots"
        )
    if np.any(np.linalg.norm(joints[:, 15], axis=-1) < 0.1):
        raise ValueError("SMPL motion contains placeholder/degenerate human geometry")


def smpl_reference(joints, roots, wrists, robot_root, alignment):
    """Release layout: 10 frames x (24 local XYZ + root 6D + six G1 wrists)."""
    local = jax.vmap(lambda q, p: jax.vmap(lambda v: quat_rotate_inverse(q, v))(p))(
        roots, joints
    )
    relative = jax.vmap(
        lambda q: rotation_6d(
            quat_multiply(quat_conjugate(robot_root), quat_multiply(alignment, q))
        )
    )(roots)
    return jnp.concatenate((local.reshape(10, 72), relative, wrists), axis=-1).reshape(
        840
    )
