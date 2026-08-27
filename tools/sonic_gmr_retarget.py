#!/usr/bin/env python3
"""Create the synchronized G1 and SMPL sidecar for a Bello AMASS clip.

Run this tool with the pinned GMR environment.  The existing TWIST converter
continues to own the Bello retarget; this tool deliberately emits only the
paired representation needed by SONIC transfer.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GMR_ROOT = ROOT / "deps" / "GMR"
G1_RETARGET_FORMAT_VERSION = 2
SMPL_COORDINATE_CONVENTION = "sonic_g1_root_local_z_up"
G1_QPOS_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
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
    "jaw",
    "left_eye_smplhf",
)


def main() -> None:
    sys.path.insert(0, str(GMR_ROOT))
    from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting
    from general_motion_retargeting.params import IK_CONFIG_DICT, ROBOT_XML_DICT
    from general_motion_retargeting.utils.smpl import get_smplx_data_offline_fast
    from scipy.spatial.transform import Rotation
    from smplx.joint_names import JOINT_NAMES
    import smplx
    import torch

    from twist_gmr_retarget import _differentiate, _load_smplx_bounded

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--provenance-source", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--body-model-root", type=Path, required=True)
    parser.add_argument("--target-fps", type=float, default=50.0)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    provenance_source = (
        source
        if args.provenance_source is None
        else args.provenance_source.expanduser().resolve()
    )
    output = args.output.expanduser().resolve()
    body_model_root = args.body_model_root.expanduser().resolve()
    for name, path in (
        ("source", source),
        ("provenance source", provenance_source),
        ("body-model root", body_model_root),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")
    if not np.isfinite(args.target_fps) or args.target_fps <= 0.0:
        raise ValueError("target FPS must be positive and finite")

    with torch.inference_mode():
        smplx_data, body_model, smplx_output, human_height = _load_smplx_bounded(
            source,
            body_model_root,
            smplx_module=smplx,
            torch_module=torch,
        )
        frames, aligned_fps = get_smplx_data_offline_fast(
            smplx_data,
            body_model,
            smplx_output,
            tgt_fps=args.target_fps,
        )
    del body_model, smplx_output
    if not np.isclose(float(aligned_fps), args.target_fps):
        raise ValueError("GMR returned a different aligned frame rate")
    smpl_joint_names = tuple(JOINT_NAMES[:24])
    if smpl_joint_names != SMPL_JOINT_NAMES:
        raise ValueError("pinned SMPL-X joint order differs from SONIC")
    missing_joints = sorted(name for name in smpl_joint_names if name not in frames[0])
    if missing_joints:
        raise ValueError(f"GMR SMPL frames are missing joints: {missing_joints}")
    smpl_joint_world = np.asarray(
        [[frame[name][0] for name in smpl_joint_names] for frame in frames],
        dtype=np.float64,
    )
    smpl_root_quat_raw = np.asarray(
        [frame["pelvis"][1] for frame in frames], dtype=np.float64
    )
    smpl_root_quat_raw /= np.maximum(
        np.linalg.norm(smpl_root_quat_raw, axis=-1, keepdims=True), 1.0e-12
    )
    source_translation = _resample_translation(
        np.asarray(smplx_data["trans"], dtype=np.float64),
        source_fps=float(np.asarray(smplx_data["mocap_frame_rate"]).item()),
        target_fps=float(aligned_fps),
        target_frames=len(frames),
    )

    retargeter = GeneralMotionRetargeting(
        src_human="smplx",
        tgt_robot="unitree_g1",
        actual_human_height=float(human_height),
        verbose=False,
        use_velocity_limit=True,
    )
    if "pelvis" not in retargeter.rot_offsets1:
        raise ValueError("pinned G1 retarget config has no pelvis rotation offset")
    root_rotation = (
        Rotation.from_quat(smpl_root_quat_raw, scalar_first=True)
        * retargeter.rot_offsets1["pelvis"]
    )
    smpl_root_quat = root_rotation.as_quat(scalar_first=True)
    # The release corpus retains the SMPL body-model joint offset while
    # removing only global translation. Subtracting the pelvis joint would
    # silently shift all 24 inputs away from the frozen encoder's distribution.
    smpl_joint_pos = _canonicalize_smpl_joints(
        smpl_joint_world,
        source_translation,
        root_rotation.as_matrix(),
    )
    # Settling the first target removes dependence on Mink's zero configuration.
    previous = None
    for _ in range(20):
        current = np.asarray(retargeter.retarget(frames[0]), dtype=np.float64)
        if previous is not None and np.max(np.abs(current - previous)) < 1.0e-5:
            break
        previous = current.copy()
    qpos = []
    for frame in frames:
        value = np.asarray(retargeter.retarget(frame), dtype=np.float64).copy()
        if value.shape != (retargeter.model.nq,):
            raise ValueError("GMR returned an invalid G1 qpos vector")
        qpos.append(value)
    qpos = np.stack(qpos)
    actual_joint_names = tuple(
        retargeter.model.joint(joint_id).name
        for joint_id in range(retargeter.model.njnt)
        if retargeter.model.jnt_qposadr[joint_id] >= 7
    )
    if actual_joint_names != G1_QPOS_JOINT_NAMES:
        raise ValueError("pinned GMR G1 qpos joint order has drifted")
    qvel = _differentiate(retargeter.model, qpos, float(aligned_fps))
    timestamps = np.arange(len(frames), dtype=np.float64) / float(aligned_fps)
    gmr_revision = subprocess.run(
        ["git", "-C", str(GMR_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    config_path = Path(IK_CONFIG_DICT["smplx"]["unitree_g1"]).resolve()
    robot_path = Path(ROBOT_XML_DICT["unitree_g1"]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        timestamp_seconds=timestamps,
        source_timestamp_seconds=timestamps,
        g1_qpos=qpos.astype(np.float32),
        g1_qvel=qvel.astype(np.float32),
        g1_qpos_joint_names=np.asarray(G1_QPOS_JOINT_NAMES),
        smpl_joint_pos=smpl_joint_pos.astype(np.float32),
        smpl_root_quat_wxyz=smpl_root_quat.astype(np.float32),
        smpl_root_translation=source_translation.astype(np.float32),
        smpl_joint_names=np.asarray(smpl_joint_names),
        source_sha256=np.asarray(_sha256(provenance_source)),
        retarget_format_version=np.asarray(G1_RETARGET_FORMAT_VERSION, dtype=np.int32),
        gmr_revision=np.asarray(gmr_revision),
        gmr_config_sha256=np.asarray(_sha256(config_path)),
        g1_robot_sha256=np.asarray(_sha256(robot_path)),
        coordinate_convention=np.asarray(SMPL_COORDINATE_CONVENTION),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resample_translation(
    translation: np.ndarray,
    *,
    source_fps: float,
    target_fps: float,
    target_frames: int,
) -> np.ndarray:
    """Match GMR's linear frame resampling for the source root translation."""

    if translation.ndim != 2 or translation.shape[1] != 3 or len(translation) < 1:
        raise ValueError("normalized SMPL-X translation must have shape (frames, 3)")
    if not np.all(np.isfinite(translation)):
        raise ValueError("normalized SMPL-X translation contains non-finite values")
    if not np.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError("source FPS must be positive and finite")
    if not np.isfinite(target_fps) or target_fps <= 0.0:
        raise ValueError("target FPS must be positive and finite")
    if isinstance(target_frames, bool) or not isinstance(target_frames, int):
        raise TypeError("target frame count must be an integer")
    if target_frames < 1:
        raise ValueError("target frame count must be positive")
    source_coordinates = np.arange(len(translation), dtype=np.float64)
    target_coordinates = np.arange(target_frames, dtype=np.float64) * (
        source_fps / target_fps
    )
    target_coordinates = np.clip(target_coordinates, 0.0, len(translation) - 1)
    return np.stack(
        [
            np.interp(target_coordinates, source_coordinates, translation[:, axis])
            for axis in range(3)
        ],
        axis=-1,
    )


def _canonicalize_smpl_joints(
    joint_world: np.ndarray,
    source_translation: np.ndarray,
    processed_root_rotation: np.ndarray,
) -> np.ndarray:
    """Remove global translation and express joints in each processed root frame."""

    joint_world = np.asarray(joint_world, dtype=np.float64)
    source_translation = np.asarray(source_translation, dtype=np.float64)
    processed_root_rotation = np.asarray(processed_root_rotation, dtype=np.float64)
    frames = len(joint_world)
    if joint_world.shape != (frames, len(SMPL_JOINT_NAMES), 3):
        raise ValueError("SMPL joint positions have an invalid shape")
    if source_translation.shape != (frames, 3):
        raise ValueError("SMPL source translation has an invalid shape")
    if processed_root_rotation.shape != (frames, 3, 3):
        raise ValueError("SMPL processed root rotations have an invalid shape")
    if not all(
        np.all(np.isfinite(value))
        for value in (joint_world, source_translation, processed_root_rotation)
    ):
        raise ValueError("SMPL canonicalization inputs must be finite")
    return np.einsum(
        "tnj,tjk->tnk",
        joint_world - source_translation[:, None, :],
        processed_root_rotation,
    )


if __name__ == "__main__":
    main()
