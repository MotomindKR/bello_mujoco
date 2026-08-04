#!/usr/bin/env python3
"""Run pinned GMR against a generated canonical Bello MJCF.

Execute this script in a dedicated environment containing GMR's optional
retargeting dependencies and licensed SMPL-X body models. It never downloads
data or models.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GMR_ROOT = ROOT / "deps" / "GMR"
sys.path.insert(0, str(GMR_ROOT))

from general_motion_retargeting import params as gmr_params  # noqa: E402
from general_motion_retargeting.motion_retarget import (  # noqa: E402
    GeneralMotionRetargeting,
)
from general_motion_retargeting.utils.smpl import (  # noqa: E402
    get_smplx_data_offline_fast,
    load_smplx_file,
)

BODY_NAMES = (
    "pelvis_link",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "l_shoulder_roll_link",
    "l_elbow_link",
    "l_wrist_link",
    "r_shoulder_roll_link",
    "r_elbow_link",
    "r_wrist_link",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--provenance-source",
        type=Path,
        default=None,
        help="Original licensed source when --source is a normalized derivative.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--body-model-root", type=Path, required=True)
    parser.add_argument("--robot-xml", type=Path, required=True)
    parser.add_argument("--target-fps", type=float, default=50.0)
    args = parser.parse_args()
    for name, path in (
        ("source", args.source),
        ("robot XML", args.robot_xml),
        ("body-model root", args.body_model_root),
    ):
        if not path.expanduser().exists():
            raise FileNotFoundError(f"{name} not found: {path}")
    if args.provenance_source is not None and not args.provenance_source.exists():
        raise FileNotFoundError(
            f"provenance source not found: {args.provenance_source}"
        )
    config_path = ROOT / "configs" / "twist" / "gmr_smplx_to_bello.json"
    gmr_params.ROBOT_XML_DICT["bello"] = args.robot_xml.expanduser().resolve()
    gmr_params.IK_CONFIG_DICT.setdefault("smplx", {})["bello"] = config_path
    smplx_data, body_model, smplx_output, human_height = load_smplx_file(
        args.source.expanduser().resolve(),
        args.body_model_root.expanduser().resolve(),
    )
    frames, fps = get_smplx_data_offline_fast(
        smplx_data,
        body_model,
        smplx_output,
        tgt_fps=args.target_fps,
    )
    retargeter = GeneralMotionRetargeting(
        src_human="smplx",
        tgt_robot="bello",
        actual_human_height=float(human_height),
        verbose=False,
    )
    qpos = np.stack(
        [
            retargeter.retarget(frame, offset_to_ground=index == 0)
            for index, frame in enumerate(frames)
        ]
    )
    model = retargeter.model
    if qpos.shape[1] != model.nq:
        raise ValueError(f"GMR returned qpos shape {qpos.shape}; model nq={model.nq}")
    timestamps = np.arange(len(qpos), dtype=np.float64) / float(fps)
    qvel = _differentiate(model, qpos, float(fps))
    body_ids = np.asarray([model.body(name).id for name in BODY_NAMES])
    body_pos, body_quat, body_lin_vel, body_ang_vel = _body_state(
        model, qpos, qvel, body_ids
    )
    joint_names = tuple(
        model.joint(int(model.actuator_trnid[index, 0])).name
        for index in range(model.nu)
    )
    joint_ids = np.asarray([model.joint(name).id for name in joint_names])
    joint_ranges = model.jnt_range[joint_ids]
    foot_indices = [
        BODY_NAMES.index("left_ankle_roll_link"),
        BODY_NAMES.index("right_ankle_roll_link"),
    ]
    foot_speed = np.linalg.norm(body_lin_vel[:, foot_indices], axis=-1)
    foot_height = body_pos[:, foot_indices, 2]
    contacts = ((foot_height < 0.04) & (foot_speed < 0.35)).astype(np.float32)
    revision = subprocess.run(
        ["git", "-C", str(GMR_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.expanduser().resolve(),
        timestamp_seconds=timestamps,
        source_timestamp_seconds=timestamps,
        qpos=qpos.astype(np.float32),
        qvel=qvel.astype(np.float32),
        body_pos=body_pos.astype(np.float32),
        body_quat_wxyz=body_quat.astype(np.float32),
        body_lin_vel=body_lin_vel.astype(np.float32),
        body_ang_vel=body_ang_vel.astype(np.float32),
        foot_contacts=contacts,
        joint_names=np.asarray(joint_names),
        body_names=np.asarray(BODY_NAMES),
        joint_lower=joint_ranges[:, 0].astype(np.float32),
        joint_upper=joint_ranges[:, 1].astype(np.float32),
        source_sha256=np.asarray(
            _sha256(
                args.provenance_source
                if args.provenance_source is not None
                else args.source
            )
        ),
        gmr_revision=np.asarray(revision),
        gmr_config_sha256=np.asarray(_sha256(config_path)),
        robot_fingerprint=np.asarray(
            "sha256:"
            + _sha256(
                ROOT
                / "deps"
                / "bello-stack"
                / "descriptions"
                / "bello"
                / "full_body"
                / "urdf"
                / "bello_full_body.urdf"
            )
        ),
    )


def _differentiate(model, qpos: np.ndarray, fps: float) -> np.ndarray:
    qvel = np.empty((len(qpos), model.nv))
    dt = 1.0 / fps
    for index in range(len(qpos) - 1):
        mujoco.mj_differentiatePos(model, qvel[index], dt, qpos[index], qpos[index + 1])
    qvel[-1] = qvel[-2]
    return qvel


def _body_state(model, qpos, qvel, body_ids):
    shapes = (len(qpos), len(body_ids))
    position = np.empty(shapes + (3,))
    quaternion = np.empty(shapes + (4,))
    linear = np.empty(shapes + (3,))
    angular = np.empty(shapes + (3,))
    data = mujoco.MjData(model)
    velocity = np.empty(6)
    for frame in range(len(qpos)):
        data.qpos[:] = qpos[frame]
        data.qvel[:] = qvel[frame]
        mujoco.mj_forward(model, data)
        position[frame] = data.xpos[body_ids]
        quaternion[frame] = data.xquat[body_ids]
        for body_index, body_id in enumerate(body_ids):
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY, int(body_id), velocity, 0
            )
            angular[frame, body_index] = velocity[:3]
            linear[frame, body_index] = velocity[3:]
    return position, quaternion, linear, angular


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
