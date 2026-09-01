"""Import trusted legacy GMR Bello pickles into canonical clips."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from pathlib import Path
import pickle

import mujoco
import numpy as np

from bello_mujoco.motion.clip import (
    BEYONDMIMIC_TRACKED_BODY_NAMES,
    DEFAULT_TRACKED_BODY_NAMES,
    FIXED_TRACKING_JOINT_NAMES,
    GMR_BELLO_JOINT_NAMES,
    MOTION_FORMAT_VERSION,
    motion_tracking_asset_config,
)
from bello_mujoco.sim import load_bello_model


def import_gmr_motion(
    source: Path,
    output: Path,
    *,
    collision_mode: str = "box",
    tracked_body_names: tuple[str, ...] = DEFAULT_TRACKED_BODY_NAMES,
    reject_self_collisions: bool = True,
) -> Path:
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    with source.open("rb") as source_file:
        motion = pickle.load(source_file)  # noqa: S301 - explicit trusted local input

    model = load_bello_model(
        motion_tracking_asset_config(collision_mode=collision_mode),
        load_mjx=False,
    )
    expected_joint_names = set(GMR_BELLO_JOINT_NAMES) - set(FIXED_TRACKING_JOINT_NAMES)
    if set(model.actuator_joint_names) != expected_joint_names:
        raise ValueError(
            "fixed-joint Bello model has drifted from the legacy GMR schema:\n"
            f"model={model.actuator_joint_names}\nGMR={GMR_BELLO_JOINT_NAMES}"
        )
    arrays = _validate_legacy_motion(motion)
    qpos = _build_qpos(model, arrays["root_pos"], arrays["root_rot"], arrays["dof_pos"])
    _validate_joint_ranges(model.mj_model, qpos)
    collision_report = _self_collision_report(model, qpos)
    if reject_self_collisions:
        _raise_for_self_collisions(collision_report)
    qvel = _differentiate_qpos(model.mj_model, qpos, float(motion["fps"]))
    body_ids = np.asarray(
        [
            mujoco.mj_name2id(model.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in tracked_body_names
        ],
        dtype=np.int32,
    )
    if np.any(body_ids < 0):
        missing = [
            name
            for name, body_id in zip(tracked_body_names, body_ids, strict=True)
            if body_id < 0
        ]
        raise ValueError(f"tracked bodies missing from fixed-joint model: {missing}")
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
        body_names=np.asarray(tracked_body_names),
        source_sha256=np.asarray(hashlib.sha256(source.read_bytes()).hexdigest()),
        qpos=qpos.astype(np.float32),
        qvel=qvel.astype(np.float32),
        body_pos=body_pos.astype(np.float32),
        body_quat_wxyz=body_quat.astype(np.float32),
        body_lin_vel=body_lin_vel.astype(np.float32),
        body_ang_vel=body_ang_vel.astype(np.float32),
        foot_contacts=contacts,
        self_collision_frames=np.asarray(collision_report["frames"], dtype=np.int32),
        max_self_collision_penetration_m=np.asarray(
            collision_report["max_penetration_m"], dtype=np.float32
        ),
        deepest_self_collision_pair=np.asarray(collision_report["deepest_pair"]),
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
    _source_joint_names(arrays["dof_pos"].shape, frames)
    if not np.all(np.isfinite(arrays["dof_pos"])):
        raise ValueError("legacy dof_pos contains nonfinite values")
    contacts = motion.get("foot_contacts")
    if contacts is None:
        raise ValueError(
            "legacy motion is missing foot_contacts; regenerate it with the "
            "maintained GMR retargeter"
        )
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
    source_joint_names = _source_joint_names(joint_pos.shape, frames)
    source_column = {name: index for index, name in enumerate(source_joint_names)}
    reordered_joint_pos = np.stack(
        [joint_pos[:, source_column[name]] for name in model.actuator_joint_names],
        axis=-1,
    )
    qpos[:, model.actuator_qpos_indices] = reordered_joint_pos
    return qpos


def _source_joint_names(shape: tuple[int, ...], frames: int) -> tuple[str, ...]:
    fixed_head_joint_names = tuple(
        name for name in GMR_BELLO_JOINT_NAMES if name not in FIXED_TRACKING_JOINT_NAMES
    )
    schemas = {
        len(GMR_BELLO_JOINT_NAMES): GMR_BELLO_JOINT_NAMES,
        len(fixed_head_joint_names): fixed_head_joint_names,
    }
    if len(shape) != 2 or shape[0] != frames or shape[1] not in schemas:
        expected = [(frames, columns) for columns in sorted(schemas)]
        raise ValueError(
            f"legacy dof_pos has shape {shape}; expected one of {expected}"
        )
    return schemas[shape[1]]


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


def _self_collision_report(model, qpos: np.ndarray) -> dict:
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
    return {
        "frames": collision_frames,
        "max_penetration_m": -deepest_distance,
        "deepest_pair": deepest_pair,
    }


def _raise_for_self_collisions(report: dict) -> None:
    if not report["frames"]:
        return
    geom1, geom2 = report["deepest_pair"]
    raise ValueError(
        "motion intersects the configured training collision model in "
        f"{report['frames']} frames; deepest penetration is "
        f"{report['max_penetration_m']:.6f} m between {geom1!r} and {geom2!r}"
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


def _import_one(source: str, output: str, overwrite: bool) -> dict:
    source_path = Path(source)
    output_path = Path(output)
    if overwrite or not output_path.exists():
        import_gmr_motion(
            source_path,
            output_path,
            collision_mode="box",
            tracked_body_names=BEYONDMIMIC_TRACKED_BODY_NAMES,
            reject_self_collisions=False,
        )
    with np.load(output_path) as motion:
        frames = int(motion["qpos"].shape[0])
        fps = float(motion["fps"])
        source_sha256 = str(motion["source_sha256"])
        body_names = [str(name) for name in motion["body_names"]]
        collision_frames = int(motion["self_collision_frames"])
        max_penetration = float(motion["max_self_collision_penetration_m"])
        deepest_pair = [str(name) for name in motion["deepest_self_collision_pair"]]
    return {
        "sequence": source_path.stem,
        "source_file": source_path.name,
        "source_sha256": source_sha256,
        "motion_file": output_path.name,
        "frames": frames,
        "fps": fps,
        "duration_seconds": frames / fps,
        "tracked_bodies": body_names,
        "collision_clean": collision_frames == 0,
        "self_collision_frames": collision_frames,
        "max_self_collision_penetration_m": max_penetration,
        "deepest_self_collision_pair": deepest_pair,
    }


def import_dataset(
    source_dir: Path,
    output_dir: Path,
    *,
    jobs: int,
    overwrite: bool,
) -> tuple[list[dict], list[dict]]:
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    sources = sorted(source_dir.resolve().glob("*.pkl"))
    if not sources:
        raise ValueError(f"no GMR .pkl motions found in {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    motions = []
    failures = []
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                _import_one,
                str(source),
                str((output_dir / f"{source.stem}.npz").resolve()),
                overwrite,
            ): source
            for source in sources
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            source = futures[future]
            try:
                motions.append(future.result())
                print(f"[{completed}/{len(futures)}] {source.stem}", flush=True)
            except Exception as error:  # noqa: BLE001 - report every bad sequence
                failures.append({"sequence": source.stem, "error": repr(error)})
                print(
                    f"[{completed}/{len(futures)}] FAILED {source.stem}: {error}",
                    flush=True,
                )
    return (
        sorted(motions, key=lambda item: item["sequence"]),
        sorted(failures, key=lambda item: item["sequence"]),
    )
