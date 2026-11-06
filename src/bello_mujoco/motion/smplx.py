"""Read native SMPL-X motion into named human joint transforms."""

from fractions import Fraction
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from bello_mujoco.hardware.smplx_contract import SMPLX_BODY_JOINT_PARENTS
from bello_mujoco.motion.interpolation import resample_quaternions


def resample_rotations(rotvecs: np.ndarray, frame: np.ndarray) -> np.ndarray:
    rotations = (
        Rotation.from_rotvec(rotvecs.reshape(-1, 3))
        .as_quat()
        .reshape(rotvecs.shape[:-1] + (4,))
    )
    return resample_quaternions(rotations, frame)


def skeleton_forward(
    rest: np.ndarray, local_quaternions: np.ndarray, translation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """SMPL-X shape-dependent joint FK, before skinning; AMASS is already Z-up."""
    frames, joints = local_quaternions.shape[:2]
    positions = np.empty((frames, joints, 3))
    rotations = np.empty((frames, joints, 3, 3))
    local = (
        Rotation.from_quat(local_quaternions.reshape(-1, 4))
        .as_matrix()
        .reshape(frames, joints, 3, 3)
    )
    for joint, parent in enumerate(SMPLX_BODY_JOINT_PARENTS):
        if parent == -1:
            positions[:, joint] = rest[joint] + translation
            rotations[:, joint] = local[:, joint]
        else:
            positions[:, joint] = positions[:, parent] + np.einsum(
                "fij,j->fi", rotations[:, parent], rest[joint] - rest[parent]
            )
            rotations[:, joint] = rotations[:, parent] @ local[:, joint]
    quaternions = (
        Rotation.from_matrix(rotations.reshape(-1, 3, 3))
        .as_quat(scalar_first=True)
        .reshape(frames, joints, 4)
    )
    return positions, quaternions


class SmplxMotionReader:
    """Shape-dependent native SMPL-X FK with a shared per-gender model cache."""

    def __init__(self, model_directory: Path):
        self.model_directory = model_directory
        self.body_models = {}

    def read(self, path: Path, output_fps: float):
        if not math.isfinite(output_fps) or output_fps <= 0:
            raise ValueError("SMPL-X output frame rate must be positive and finite")
        with np.load(path, allow_pickle=False) as data:
            if str(data["surface_model_type"]) != "smplx":
                raise ValueError("requires AMASS SMPL-X, not SMPL/SMPL-H")
            gender = str(data["gender"])
            fps = float(data["mocap_frame_rate"])
            root = np.asarray(data["root_orient"])
            pose = np.asarray(data["pose_body"]).reshape(len(root), 21, 3)
            trans = np.asarray(data["trans"])
            betas = np.asarray(data["betas"]).reshape(-1)
        if (
            gender not in {"male", "female", "neutral"}
            or not math.isfinite(fps)
            or fps <= 0
        ):
            raise ValueError("invalid AMASS gender or frame rate")
        if (
            root.shape != (len(root), 3)
            or trans.shape != root.shape
            or len(root) < 2
            or not 1 <= len(betas) <= 16
        ):
            raise ValueError(
                "invalid native SMPL-X pose, translation or beta dimensions"
            )
        if not all(np.isfinite(v).all() for v in (root, pose, trans, betas)):
            raise ValueError("nonfinite AMASS motion")
        if gender not in self.body_models:
            with np.load(
                self.model_directory / f"SMPLX_{gender.upper()}.npz", allow_pickle=False
            ) as body:
                regressor = np.asarray(body["J_regressor"])[:22]
                parents = np.asarray(body["kintree_table"])[0, :22].astype(int)
                parents[0] = -1
                if not np.array_equal(parents, SMPLX_BODY_JOINT_PARENTS):
                    raise ValueError("unexpected SMPL-X hierarchy")
                self.body_models[gender] = (
                    regressor @ body["v_template"],
                    np.einsum("jv,vck->jck", regressor, body["shapedirs"][:, :, :16]),
                )
        rest, directions = self.body_models[gender]
        rest = rest + np.einsum("jck,k->jc", directions[:, :, : len(betas)], betas)
        rate_ratio = Fraction(str(output_fps)) / Fraction(str(fps))
        last_output_frame = (
            (len(root) - 1) * rate_ratio.numerator // rate_ratio.denominator
        )
        frame = np.minimum(
            np.arange(last_output_frame + 1, dtype=np.float64)
            * rate_ratio.denominator
            / rate_ratio.numerator,
            len(root) - 1,
        )
        local = resample_rotations(np.concatenate((root[:, None], pose), axis=1), frame)
        translation = np.stack(
            [
                np.interp(frame, np.arange(len(trans)), trans[:, axis])
                for axis in range(3)
            ],
            axis=-1,
        )
        positions, orientations = skeleton_forward(rest, local, translation)
        return positions, orientations, float(1.66 + 0.1 * betas[0])
