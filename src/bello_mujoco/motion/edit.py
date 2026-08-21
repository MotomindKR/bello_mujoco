"""Auditable edits to canonical Bello motion references."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

import mujoco
import numpy as np

from bello_mujoco.motion.clip import load_motion_clip, motion_tracking_asset_config
from bello_mujoco.motion.import_gmr import (
    _body_trajectories,
    _differentiate_qpos,
    _self_collision_report,
    _validate_joint_ranges,
)
from bello_mujoco.sim import load_bello_model

SYMMETRIC_DOWNWARD_ARM_POSE = {
    "shoulder_pitch_joint": 0.0,
    "shoulder_roll_joint": 0.20,
    "shoulder_yaw_joint": 0.0,
    "elbow_pitch_joint": 0.15,
    "elbow_yaw_joint": 0.0,
    "wrist_pitch_joint": 0.0,
}
SYMMETRIC_DOWNWARD_ARM_EDIT_VERSION = "symmetric-downward-arms-v1"


def edit_symmetric_downward_arm_pose(source: Path, output: Path) -> Path:
    """Replace both arm trajectories with a fixed down-and-out carrying pose."""

    source = Path(source).resolve()
    output = Path(output).resolve()
    model = load_bello_model(
        motion_tracking_asset_config(),
        load_mjx=False,
    )
    motion = load_motion_clip(source, model)
    source_bytes = source.read_bytes()
    with np.load(source, allow_pickle=False) as archive:
        if "arm_pose_edit_version" in archive.files:
            raise ValueError(f"motion is already arm-pose edited: {source}")
        fields = {name: np.asarray(archive[name]).copy() for name in archive.files}

    qpos = np.asarray(motion.qpos, dtype=np.float64).copy()
    qpos_index = dict(
        zip(model.actuator_joint_names, model.actuator_qpos_indices, strict=True)
    )
    targets: dict[str, float] = {}
    for side in ("left", "right"):
        for suffix, target in SYMMETRIC_DOWNWARD_ARM_POSE.items():
            name = f"{side}_{suffix}"
            if name not in qpos_index:
                raise ValueError(f"motion model is missing arm joint {name!r}")
            qpos[:, qpos_index[name]] = target
            targets[name] = target

    _validate_joint_ranges(model.mj_model, qpos)
    qvel = _differentiate_qpos(model.mj_model, qpos, motion.fps)
    body_ids = np.asarray(
        [
            mujoco.mj_name2id(
                model.mj_model,
                mujoco.mjtObj.mjOBJ_BODY,
                name,
            )
            for name in motion.body_names
        ],
        dtype=np.int32,
    )
    if np.any(body_ids < 0):
        raise ValueError("motion tracked-body layout is absent from the Bello model")
    body_pos, body_quat, body_lin_vel, body_ang_vel = _body_trajectories(
        model.mj_model, qpos, qvel, body_ids
    )
    collision_report = _self_collision_report(model, qpos)

    fields.update(
        {
            "qpos": qpos.astype(np.float32),
            "qvel": qvel.astype(np.float32),
            "body_pos": body_pos.astype(np.float32),
            "body_quat_wxyz": body_quat.astype(np.float32),
            "body_lin_vel": body_lin_vel.astype(np.float32),
            "body_ang_vel": body_ang_vel.astype(np.float32),
            "self_collision_frames": np.asarray(
                collision_report["frames"], dtype=np.int32
            ),
            "max_self_collision_penetration_m": np.asarray(
                collision_report["max_penetration_m"], dtype=np.float32
            ),
            "deepest_self_collision_pair": np.asarray(collision_report["deepest_pair"]),
            "arm_pose_edit_version": np.asarray(SYMMETRIC_DOWNWARD_ARM_EDIT_VERSION),
            "arm_pose_edit_source_content_sha256": np.asarray(
                hashlib.sha256(source_bytes).hexdigest()
            ),
            "arm_pose_edit_joint_names": np.asarray(tuple(targets)),
            "arm_pose_edit_joint_positions": np.asarray(
                tuple(targets.values()), dtype=np.float32
            ),
        }
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".npz", dir=output.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        np.savez_compressed(temporary_path, **fields)
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output
