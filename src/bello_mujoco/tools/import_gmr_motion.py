"""Import a trusted legacy GMR Bello pickle into the canonical clip format."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import pickle

import mujoco
import numpy as np

from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.motion import (
    DEFAULT_TRACKED_BODY_NAMES,
    GMR_BELLO_JOINT_NAMES,
    MOTION_FORMAT_VERSION,
)
from bello_mujoco.sim import load_bello_model


def import_gmr_motion(
    source: Path,
    output: Path,
    *,
    collision_mode: str = "box",
) -> Path:
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    with source.open("rb") as source_file:
        motion = pickle.load(source_file)  # noqa: S301 - explicit trusted local input

    model = load_bello_model(
        AssetPrepConfig(
            asset="full_body",
            collision_mode=collision_mode,
            end_effector_sphere_mode="both",
            fixed_joint_names=("waist_roll_joint", "waist_pitch_joint"),
        ),
        load_mjx=False,
    )
    if set(model.actuator_joint_names) != set(GMR_BELLO_JOINT_NAMES):
        raise ValueError(
            "fixed-waist Bello joints have drifted from the legacy GMR schema:\n"
            f"model={model.actuator_joint_names}\nGMR={GMR_BELLO_JOINT_NAMES}"
        )
    arrays = _validate_legacy_motion(motion)
    qpos = _build_qpos(model, arrays["root_pos"], arrays["root_rot"], arrays["dof_pos"])
    _validate_joint_ranges(model.mj_model, qpos)
    _validate_self_collisions(model, qpos)
    qvel = _differentiate_qpos(model.mj_model, qpos, float(motion["fps"]))
    body_ids = np.asarray(
        [
            mujoco.mj_name2id(model.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in DEFAULT_TRACKED_BODY_NAMES
        ],
        dtype=np.int32,
    )
    if np.any(body_ids < 0):
        missing = [
            name
            for name, body_id in zip(DEFAULT_TRACKED_BODY_NAMES, body_ids, strict=True)
            if body_id < 0
        ]
        raise ValueError(f"tracked bodies missing from fixed-waist model: {missing}")
    body_pos, body_quat, body_lin_vel, body_ang_vel = _body_trajectories(
        model.mj_model, qpos, qvel, body_ids
    )
    contacts = arrays["foot_contacts"]
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        format_version=np.asarray(MOTION_FORMAT_VERSION, dtype=np.int32),
        fps=np.asarray(float(motion["fps"]), dtype=np.float32),
        joint_names=np.asarray(model.actuator_joint_names),
        body_names=np.asarray(DEFAULT_TRACKED_BODY_NAMES),
        source_sha256=np.asarray(hashlib.sha256(source.read_bytes()).hexdigest()),
        qpos=qpos.astype(np.float32),
        qvel=qvel.astype(np.float32),
        body_pos=body_pos.astype(np.float32),
        body_quat_wxyz=body_quat.astype(np.float32),
        body_lin_vel=body_lin_vel.astype(np.float32),
        body_ang_vel=body_ang_vel.astype(np.float32),
        foot_contacts=contacts,
    )
    return output


def _validate_legacy_motion(motion: object) -> dict[str, np.ndarray]:
    if not isinstance(motion, dict):
        raise ValueError(f"expected legacy motion dict, got {type(motion).__name__}")
    for key in ("fps", "root_pos", "root_rot", "dof_pos"):
        if key not in motion:
            raise ValueError(f"legacy motion is missing {key!r}")
    fps = float(motion["fps"])
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"legacy fps must be positive and finite, got {fps}")
    arrays = {
        "root_pos": np.asarray(motion["root_pos"], dtype=np.float64),
        "root_rot": np.asarray(motion["root_rot"], dtype=np.float64),
        "dof_pos": np.asarray(motion["dof_pos"], dtype=np.float64),
    }
    frames = arrays["root_pos"].shape[0]
    expected = {
        "root_pos": (frames, 3),
        "root_rot": (frames, 4),
        "dof_pos": (frames, len(GMR_BELLO_JOINT_NAMES)),
    }
    if frames < 2:
        raise ValueError("legacy motion must contain at least two frames")
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(
                f"legacy {name} has shape {arrays[name].shape}; expected {shape}"
            )
        if not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"legacy {name} contains nonfinite values")
    contacts = motion.get("foot_contacts")
    if contacts is None:
        arrays["foot_contacts"] = np.zeros((frames, 2), dtype=np.float32)
    else:
        if not isinstance(contacts, dict):
            raise ValueError("legacy foot_contacts must be a dict")
        contact_columns = []
        for foot in ("left_foot", "right_foot"):
            values = np.asarray(contacts.get(foot), dtype=np.float32)
            if values.shape != (frames,):
                raise ValueError(
                    f"legacy foot_contacts[{foot!r}] has shape {values.shape}; "
                    f"expected {(frames,)}"
                )
            contact_columns.append(values)
        arrays["foot_contacts"] = np.stack(contact_columns, axis=-1)
    return arrays


def _build_qpos(model, root_pos, root_quat_xyzw, joint_pos) -> np.ndarray:
    frames = root_pos.shape[0]
    qpos = np.repeat(model.default_qpos[None, :], frames, axis=0)
    root = model.root_qpos_adr
    qpos[:, root : root + 3] = root_pos
    root_quat = root_quat_xyzw[:, [3, 0, 1, 2]].copy()
    root_quat /= np.linalg.norm(root_quat, axis=-1, keepdims=True)
    for frame in range(1, frames):
        if np.dot(root_quat[frame - 1], root_quat[frame]) < 0.0:
            root_quat[frame] *= -1.0
    qpos[:, root + 3 : root + 7] = root_quat
    source_column = {name: index for index, name in enumerate(GMR_BELLO_JOINT_NAMES)}
    reordered_joint_pos = np.stack(
        [joint_pos[:, source_column[name]] for name in model.actuator_joint_names],
        axis=-1,
    )
    qpos[:, model.actuator_qpos_indices] = reordered_joint_pos
    return qpos


def _validate_joint_ranges(mj_model, qpos: np.ndarray) -> None:
    for joint_id in range(mj_model.njnt):
        if not mj_model.jnt_limited[joint_id]:
            continue
        address = int(mj_model.jnt_qposadr[joint_id])
        low, high = mj_model.jnt_range[joint_id]
        values = qpos[:, address]
        if values.min() < low - 1.0e-4 or values.max() > high + 1.0e-4:
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            raise ValueError(
                f"motion joint {name!r} exceeds hard range [{low}, {high}]: "
                f"observed [{values.min()}, {values.max()}]"
            )


def _validate_self_collisions(model, qpos: np.ndarray) -> None:
    mj_model = model.mj_model
    robot_geom_ids = set(model.robot_geom_ids.tolist())
    data = mujoco.MjData(mj_model)
    collision_frames = 0
    deepest_distance = 0.0
    deepest_pair = ("", "")
    for frame_qpos in qpos:
        data.qpos[:] = frame_qpos
        mujoco.mj_forward(mj_model, data)
        frame_collides = False
        for contact in data.contact[: data.ncon]:
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if (
                geom1 not in robot_geom_ids
                or geom2 not in robot_geom_ids
                or contact.dist >= -1.0e-7
            ):
                continue
            frame_collides = True
            if contact.dist < deepest_distance:
                deepest_distance = float(contact.dist)
                deepest_pair = (
                    mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom1),
                    mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom2),
                )
        collision_frames += int(frame_collides)
    if collision_frames:
        raise ValueError(
            "motion intersects the configured training collision model in "
            f"{collision_frames} frames; deepest penetration is "
            f"{-deepest_distance:.6f} m between {deepest_pair[0]!r} and "
            f"{deepest_pair[1]!r}"
        )


def _differentiate_qpos(mj_model, qpos: np.ndarray, fps: float) -> np.ndarray:
    qvel = np.empty((qpos.shape[0], mj_model.nv), dtype=np.float64)
    dt = 1.0 / fps
    for frame in range(qpos.shape[0] - 1):
        mujoco.mj_differentiatePos(
            mj_model, qvel[frame], dt, qpos[frame], qpos[frame + 1]
        )
    mujoco.mj_differentiatePos(mj_model, qvel[-1], dt, qpos[-2], qpos[-1])
    return qvel


def _body_trajectories(mj_model, qpos, qvel, body_ids):
    frames = qpos.shape[0]
    bodies = body_ids.size
    pos = np.empty((frames, bodies, 3), dtype=np.float64)
    quat = np.empty((frames, bodies, 4), dtype=np.float64)
    lin_vel = np.empty((frames, bodies, 3), dtype=np.float64)
    ang_vel = np.empty((frames, bodies, 3), dtype=np.float64)
    data = mujoco.MjData(mj_model)
    velocity = np.empty(6, dtype=np.float64)
    for frame in range(frames):
        data.qpos[:] = qpos[frame]
        data.qvel[:] = qvel[frame]
        mujoco.mj_forward(mj_model, data)
        pos[frame] = data.xpos[body_ids]
        quat[frame] = data.xquat[body_ids]
        for output_index, body_id in enumerate(body_ids):
            mujoco.mj_objectVelocity(
                mj_model,
                data,
                mujoco.mjtObj.mjOBJ_BODY,
                int(body_id),
                velocity,
                0,
            )
            ang_vel[frame, output_index] = velocity[:3]
            lin_vel[frame, output_index] = velocity[3:]
    return pos, quat, lin_vel, ang_vel


def entrypoint() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--collision-mode",
        choices=("box", "foot-mesh", "mesh"),
        default="box",
    )
    args = parser.parse_args()
    output = import_gmr_motion(
        args.source, args.output, collision_mode=args.collision_mode
    )
    print(output)


if __name__ == "__main__":
    entrypoint()
