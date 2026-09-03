#!/usr/bin/env python3
"""Run pinned GMR against a generated robot MJCF.

Execute this script in a dedicated environment containing GMR's optional
retargeting dependencies and licensed SMPL-X body models. It never downloads
data or models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
GMR_ROOT = ROOT / "deps" / "GMR"
sys.path.insert(0, str(ROOT / "src"))

from bello_mujoco.assets_pipeline import robot_descriptions_root  # noqa: E402
from bello_mujoco.motion.twist_dataset import (  # noqa: E402
    TWIST_RETARGET_FORMAT_VERSION,
)
from bello_mujoco.robot import robot_definition  # noqa: E402
from bello_mujoco.robot_tasks import (  # noqa: E402
    GmrLegDefinition,
    GmrKneePoleDefinition,
    motion_task_definition,
)
from bello_mujoco.sim import logical_actuator_joint_names  # noqa: E402

CONTACT_MAX_SPEED = 0.2
CONTACT_HEIGHT_MARGIN = 0.03


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
    parser.add_argument("--asset", required=True)
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
    robot = robot_definition(args.asset)
    profile = motion_task_definition(args.asset).gmr
    config_path = profile.config_path(GMR_ROOT)
    if not config_path.is_file():
        raise FileNotFoundError(f"GMR profile config not found: {config_path}")
    sys.path.insert(0, str(GMR_ROOT))
    from general_motion_retargeting import params as gmr_params
    from general_motion_retargeting.offline_retarget import retarget_offline_frames

    gmr_params.ROBOT_XML_DICT[profile.target_robot] = (
        args.robot_xml.expanduser().resolve()
    )
    gmr_params.IK_CONFIG_DICT.setdefault(profile.source_human, {})[
        profile.target_robot
    ] = config_path
    source_path = args.source.expanduser().resolve()
    if _is_fastsam_source(source_path):
        frames, fps, human_height, source_contacts, source_timestamps = (
            _load_fastsam_source(source_path)
        )
    else:
        from general_motion_retargeting.utils.smpl import get_smplx_data_offline_fast
        import smplx
        import torch

        with torch.inference_mode():
            smplx_data, body_model, smplx_output, human_height = _load_smplx_bounded(
                source_path,
                args.body_model_root.expanduser().resolve(),
                smplx_module=smplx,
                torch_module=torch,
            )
            frames, fps = get_smplx_data_offline_fast(
                smplx_data,
                body_model,
                smplx_output,
                tgt_fps=args.target_fps,
            )
        del body_model, smplx_output
        source_contacts = _human_foot_contacts(
            frames, float(fps), profile.source_foot_body_names
        )
        source_timestamps = np.arange(len(frames), dtype=np.float64) / float(fps)
    frames = _select_source_targets(frames, profile.source_body_names)
    result = retarget_offline_frames(
        frames,
        src_human=profile.source_human,
        tgt_robot=profile.target_robot,
        actual_human_height=float(human_height),
        task_profile=profile.task_profile,
        verbose=False,
    )
    retargeter = result.retargeter
    model = retargeter.model
    joint_names = logical_actuator_joint_names(model, robot)
    qpos_joint_names = _qpos_ordered_joint_names(model, joint_names)
    joint_ranges = _qpos_ordered_joint_ranges(model, joint_names)
    knee_pole_local_axis, knee_pole_world_direction = _knee_pole_diagnostics(
        frames, profile.knee_pole
    )
    qpos = result.qpos
    if qpos.shape[1] != model.nq:
        raise ValueError(f"GMR returned qpos shape {qpos.shape}; model nq={model.nq}")
    foot_sole_z = _foot_sole_z(model, qpos, retargeter.ground_clearance_geom_ids)
    qpos, foot_sole_z = _ground_root_to_support(qpos, foot_sole_z, source_contacts)
    timestamps = np.arange(len(qpos), dtype=np.float64) / float(fps)
    qvel = _differentiate(model, qpos, float(fps))
    body_ids = np.asarray([model.body(name).id for name in profile.output_body_names])
    body_pos, body_quat, body_lin_vel, body_ang_vel = _body_state(
        model, qpos, qvel, body_ids
    )
    contacts = (source_contacts & (foot_sole_z < 0.03)).astype(np.float32)
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
        source_timestamp_seconds=source_timestamps,
        qpos=qpos.astype(np.float32),
        qvel=qvel.astype(np.float32),
        body_pos=body_pos.astype(np.float32),
        body_quat_wxyz=body_quat.astype(np.float32),
        body_lin_vel=body_lin_vel.astype(np.float32),
        body_ang_vel=body_ang_vel.astype(np.float32),
        foot_contacts=contacts,
        foot_sole_z=foot_sole_z.astype(np.float32),
        knee_pole_local_axis=knee_pole_local_axis.astype(np.float32),
        knee_pole_world_direction=knee_pole_world_direction.astype(np.float32),
        knee_pole_sides=np.asarray(tuple(leg.name for leg in profile.knee_pole.legs)),
        retarget_format_version=np.asarray(
            TWIST_RETARGET_FORMAT_VERSION, dtype=np.int32
        ),
        joint_names=np.asarray(joint_names),
        qpos_joint_names=np.asarray(qpos_joint_names),
        body_names=np.asarray(profile.output_body_names),
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
        gmr_task_profile=np.asarray(retargeter.task_profile or ""),
        gmr_use_velocity_limit=np.asarray(result.use_velocity_limit),
        gmr_passes_per_frame=np.asarray(result.passes_per_frame, dtype=np.int32),
        gmr_initial_settle_passes=np.asarray(
            result.initial_settle_passes, dtype=np.int32
        ),
        gmr_joint_smoothing_kernel=result.joint_smoothing_kernel,
        gmr_root_orientation_smoothing_kernel=(
            result.root_orientation_smoothing_kernel
        ),
        gmr_joint_smoothing_passes=np.asarray(
            result.joint_smoothing_passes, dtype=np.int32
        ),
        gmr_smoothing_alpha=result.smoothing_alpha,
        robot_fingerprint=np.asarray(
            "sha256:" + _sha256(robot.source_urdf_path(robot_descriptions_root()))
        ),
    )


def _is_fastsam_source(path: Path) -> bool:
    with np.load(path, allow_pickle=False) as archive:
        return (
            "source_format" in archive.files
            and str(np.asarray(archive["source_format"]).item())
            == "fastsam-smplx-body-22-v1"
        )


def _select_source_targets(frames, source_body_names: tuple[str, ...]):
    selected = []
    for frame_index, frame in enumerate(frames):
        missing = sorted(set(source_body_names) - set(frame))
        if missing:
            raise ValueError(
                f"source frame {frame_index} is missing configured GMR bodies: "
                + ", ".join(missing)
            )
        selected.append(
            {
                name: [np.asarray(value).copy() for value in frame[name]]
                for name in source_body_names
            }
        )
    if not selected:
        raise ValueError("retargeting requires at least one source frame")
    return selected


def _load_fastsam_source(path: Path):
    required = {
        "format_version",
        "source_format",
        "timestamp_seconds",
        "source_timestamp_seconds",
        "joint_names",
        "joint_parents",
        "joint_positions",
        "global_joint_quaternions_wxyz",
        "foot_contacts",
        "human_height_m",
        "root_conditioning_version",
        "root_conditioning_config",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"Fast-SAM source is missing fields: {missing}")
        if int(np.asarray(archive["format_version"]).item()) != 1:
            raise ValueError("unsupported Fast-SAM source format version")
        names = tuple(str(name) for name in np.asarray(archive["joint_names"]))
        parents = np.asarray(archive["joint_parents"], dtype=np.int32)
        positions = np.asarray(archive["joint_positions"], dtype=np.float64)
        quaternions = np.asarray(
            archive["global_joint_quaternions_wxyz"], dtype=np.float64
        )
        timestamps = np.asarray(archive["timestamp_seconds"], dtype=np.float64)
        source_timestamps = np.asarray(
            archive["source_timestamp_seconds"], dtype=np.float64
        )
        contacts = np.asarray(archive["foot_contacts"], dtype=bool)
        human_height = float(np.asarray(archive["human_height_m"]).item())
        conditioning_version = str(
            np.asarray(archive["root_conditioning_version"]).item()
        )
        conditioning_config = str(
            np.asarray(archive["root_conditioning_config"]).item()
        )
    if conditioning_version != "causal-support-foot-xy-v2":
        raise ValueError(
            "Fast-SAM source does not use the live-compatible root conditioner"
        )
    try:
        conditioning_document = json.loads(conditioning_config)
    except json.JSONDecodeError as error:
        raise ValueError("Fast-SAM root-conditioning config is invalid") from error
    if not isinstance(conditioning_document, dict) or not conditioning_document:
        raise ValueError("Fast-SAM root-conditioning config must be a mapping")
    expected_names = (
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
    expected_parents = np.asarray(
        (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19),
        dtype=np.int32,
    )
    frames = len(timestamps)
    if names != expected_names or not np.array_equal(parents, expected_parents):
        raise ValueError("Fast-SAM source has an incompatible SMPL-X body contract")
    if positions.shape != (frames, len(names), 3) or quaternions.shape != (
        frames,
        len(names),
        4,
    ):
        raise ValueError("Fast-SAM source joint arrays have invalid shapes")
    if contacts.shape != (frames, 2):
        raise ValueError("Fast-SAM source foot contacts have an invalid shape")
    if source_timestamps.shape != (frames,) or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("Fast-SAM source timestamps are invalid")
    if not all(
        np.all(np.isfinite(value))
        for value in (positions, quaternions, timestamps, source_timestamps)
    ):
        raise ValueError("Fast-SAM source contains non-finite values")
    dt = np.diff(timestamps)
    fps = 1.0 / float(np.median(dt))
    targets = []
    for frame_index in range(frames):
        targets.append(
            {
                name: [
                    positions[frame_index, joint_index].copy(),
                    quaternions[frame_index, joint_index].copy(),
                ]
                for joint_index, name in enumerate(names)
            }
        )
    return targets, fps, human_height, contacts, source_timestamps


def _load_smplx_bounded(
    source: Path,
    body_model_root: Path,
    *,
    smplx_module,
    torch_module,
    chunk_frames: int = 512,
):
    if chunk_frames < 1:
        raise ValueError("SMPL-X inference chunk must contain at least one frame")
    with np.load(source, allow_pickle=False) as archive:
        smplx_data = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "betas",
        "gender",
        "mocap_frame_rate",
        "pose_body",
        "root_orient",
        "trans",
    }
    missing = sorted(required - smplx_data.keys())
    if missing:
        raise ValueError(f"normalized SMPL-X source is missing fields: {missing}")
    pose_body = np.asarray(smplx_data["pose_body"], dtype=np.float32)
    root_orient = np.asarray(smplx_data["root_orient"], dtype=np.float32)
    translation = np.asarray(smplx_data["trans"], dtype=np.float32)
    frames = pose_body.shape[0]
    if pose_body.shape != (frames, 63):
        raise ValueError(f"normalized pose_body has invalid shape {pose_body.shape}")
    if root_orient.shape != (frames, 3) or translation.shape != (frames, 3):
        raise ValueError("normalized root orientation or translation shape is invalid")
    gender = str(np.asarray(smplx_data["gender"]).item())
    body_model = smplx_module.create(
        body_model_root,
        "smplx",
        gender=gender,
        use_pca=False,
    )
    betas = torch_module.as_tensor(
        np.asarray(smplx_data["betas"], dtype=np.float32).reshape(1, -1)
    )
    outputs = {"global_orient": [], "full_pose": [], "joints": []}
    for start in range(0, frames, chunk_frames):
        stop = min(start + chunk_frames, frames)
        count = stop - start
        zeros_45 = torch_module.zeros((count, 45), dtype=torch_module.float32)
        zeros_3 = torch_module.zeros((count, 3), dtype=torch_module.float32)
        output = body_model(
            betas=betas,
            global_orient=torch_module.as_tensor(root_orient[start:stop]),
            body_pose=torch_module.as_tensor(pose_body[start:stop]),
            transl=torch_module.as_tensor(translation[start:stop]),
            left_hand_pose=zeros_45,
            right_hand_pose=zeros_45,
            jaw_pose=zeros_3,
            leye_pose=zeros_3,
            reye_pose=zeros_3,
            return_full_pose=True,
            return_verts=False,
        )
        for name in outputs:
            outputs[name].append(getattr(output, name).detach())
    smplx_output = SimpleNamespace(
        **{name: torch_module.cat(chunks, dim=0) for name, chunks in outputs.items()}
    )
    if np.asarray(smplx_data["betas"]).ndim == 1:
        human_height = 1.66 + 0.1 * float(np.asarray(smplx_data["betas"])[0])
    else:
        human_height = 1.66 + 0.1 * float(np.asarray(smplx_data["betas"])[0, 0])
    return smplx_data, body_model, smplx_output, human_height


def _knee_pole_diagnostics(
    frames,
    config: GmrKneePoleDefinition,
) -> tuple[np.ndarray, np.ndarray]:
    """Return stable source knee-pole axes for retargeting quality checks."""
    if not frames:
        raise ValueError("retargeting requires at least one source frame")
    local_axes = tuple(
        _infer_positive_knee_pole_trajectory(frames, leg, config) for leg in config.legs
    )
    world_directions = []
    for frame_index, frame in enumerate(frames):
        frame_world_directions = []
        for leg_index, leg in enumerate(config.legs):
            hip_rotation = Rotation.from_quat(
                frame[leg.source_hip_body_name][1], scalar_first=True
            )
            pole_direction = hip_rotation.apply(local_axes[leg_index][frame_index])
            pole_direction /= np.linalg.norm(pole_direction)
            frame_world_directions.append(pole_direction)
        world_directions.append(frame_world_directions)
    return (
        np.stack(local_axes, axis=1),
        np.asarray(world_directions),
    )


def _infer_positive_knee_pole_trajectory(
    frames,
    leg: GmrLegDefinition,
    config: GmrKneePoleDefinition,
) -> np.ndarray:
    hip_name, knee_name, foot_name = leg.source_body_names
    thigh_offsets = []
    shin_offsets = []
    sample_axes = np.full((len(frames), 3), np.nan, dtype=np.float64)
    flexions = np.empty(len(frames), dtype=np.float64)
    pole_distances = np.empty(len(frames), dtype=np.float64)
    for frame_index, frame in enumerate(frames):
        hip_position = np.asarray(frame[hip_name][0], dtype=np.float64)
        knee_position = np.asarray(frame[knee_name][0], dtype=np.float64)
        foot_position = np.asarray(frame[foot_name][0], dtype=np.float64)
        hip_rotation = Rotation.from_quat(frame[hip_name][1], scalar_first=True)
        knee_rotation = Rotation.from_quat(frame[knee_name][1], scalar_first=True)
        thigh_offsets.append(hip_rotation.inv().apply(knee_position - hip_position))
        shin_offsets.append(knee_rotation.inv().apply(foot_position - knee_position))
        local_knee_rotation = (hip_rotation.inv() * knee_rotation).as_rotvec()
        flexion = float(local_knee_rotation[0])
        flexions[frame_index] = flexion
        pole = _knee_pole_vector(hip_position, knee_position, foot_position)
        pole_distance = float(np.linalg.norm(pole))
        pole_distances[frame_index] = pole_distance
        if pole_distance >= config.sample_minimum_distance_meters:
            sample_axes[frame_index] = hip_rotation.inv().apply(pole / pole_distance)
    thigh_offset = np.median(np.asarray(thigh_offsets), axis=0)
    shin_offset = np.median(np.asarray(shin_offsets), axis=0)
    reference_foot = thigh_offset + Rotation.from_rotvec(
        (config.reference_flexion_radians, 0.0, 0.0)
    ).apply(shin_offset)
    reference_axis = _normalized_knee_pole(np.zeros(3), thigh_offset, reference_foot)
    finite = np.all(np.isfinite(sample_axes), axis=1)
    hemisphere = np.where(finite, sample_axes @ reference_axis > 0.0, False)
    valid = (
        finite
        & hemisphere
        & (flexions >= config.sample_minimum_flexion_radians)
        & (pole_distances >= config.sample_minimum_distance_meters)
    )
    if not np.any(valid):
        return np.repeat(reference_axis[None, :], len(frames), axis=0)
    valid_indices = np.flatnonzero(valid)
    frame_indices = np.arange(len(frames))
    trajectory = np.stack(
        [
            np.interp(
                frame_indices,
                valid_indices,
                sample_axes[valid_indices, component],
            )
            for component in range(3)
        ],
        axis=1,
    )
    trajectory = _normalize_vector_rows(trajectory, reference_axis)
    window = config.smoothing_frames
    if window > 1:
        radius = window // 2
        padded = np.pad(trajectory, ((radius, radius), (0, 0)), mode="edge")
        kernel = np.full(window, 1.0 / window)
        trajectory = np.stack(
            [
                np.convolve(padded[:, component], kernel, mode="valid")
                for component in range(3)
            ],
            axis=1,
        )
        trajectory = _normalize_vector_rows(trajectory, reference_axis)
    return trajectory


def _normalize_vector_rows(vectors: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    valid = np.isfinite(norms) & (norms >= 1.0e-8)
    result = np.repeat(np.asarray(fallback)[None, :], len(vectors), axis=0)
    result[valid] = vectors[valid] / norms[valid, None]
    return result


def _knee_pole_vector(
    hip_position: np.ndarray,
    knee_position: np.ndarray,
    foot_position: np.ndarray,
) -> np.ndarray:
    chord = np.asarray(foot_position) - np.asarray(hip_position)
    chord_norm = float(np.linalg.norm(chord))
    if not math.isfinite(chord_norm) or chord_norm < 1.0e-8:
        return np.zeros(3, dtype=np.float64)
    chord_axis = chord / chord_norm
    thigh = np.asarray(knee_position) - np.asarray(hip_position)
    return thigh - np.dot(thigh, chord_axis) * chord_axis


def _normalized_knee_pole(
    hip_position: np.ndarray,
    knee_position: np.ndarray,
    foot_position: np.ndarray,
) -> np.ndarray:
    pole = _knee_pole_vector(hip_position, knee_position, foot_position)
    norm = float(np.linalg.norm(pole))
    if not math.isfinite(norm) or norm < 1.0e-8:
        raise ValueError("positive-flex knee reference has no bend plane")
    return pole / norm


def _qpos_ordered_joint_ranges(
    model, actuator_joint_names: tuple[str, ...]
) -> np.ndarray:
    qpos_joint_names = _qpos_ordered_joint_names(model, actuator_joint_names)
    joint_ids = [model.joint(name).id for name in qpos_joint_names]
    return np.asarray(model.jnt_range[joint_ids])


def _qpos_ordered_joint_names(
    model, actuator_joint_names: tuple[str, ...]
) -> tuple[str, ...]:
    joint_ids = sorted(
        (
            joint_id
            for joint_id in range(model.njnt)
            if model.jnt_type[joint_id]
            in {mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE}
        ),
        key=lambda joint_id: int(model.jnt_qposadr[joint_id]),
    )
    qpos_joint_names = tuple(model.joint(joint_id).name for joint_id in joint_ids)
    if len(joint_ids) != model.nq - 7:
        raise ValueError(
            "robot model must contain one free root followed by scalar joints"
        )
    if set(qpos_joint_names) != set(actuator_joint_names):
        raise ValueError(
            "robot qpos and logical actuator joints do not describe the same joints"
        )
    return qpos_joint_names


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


def _foot_sole_z(model, qpos: np.ndarray, geom_ids) -> np.ndarray:
    validated_geom_ids = []
    for geom_id in geom_ids:
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
            name = model.geom(geom_id).name
            raise ValueError(f"GMR ground-clearance geom {name!r} must be a box")
        validated_geom_ids.append(geom_id)
    if len(validated_geom_ids) != 2:
        raise ValueError(
            "TWIST retargeting requires exactly two ground-clearance geoms"
        )
    heights = np.empty((len(qpos), len(validated_geom_ids)))
    data = mujoco.MjData(model)
    for frame, position in enumerate(qpos):
        data.qpos[:] = position
        mujoco.mj_forward(model, data)
        for foot, geom_id in enumerate(validated_geom_ids):
            rotation = data.geom_xmat[geom_id].reshape(3, 3)
            vertical_extent = np.abs(rotation[2]) @ model.geom_size[geom_id]
            heights[frame, foot] = data.geom_xpos[geom_id, 2] - vertical_extent
    return heights


def _human_foot_contacts(
    frames,
    fps: float,
    foot_body_names: tuple[str, str],
) -> np.ndarray:
    positions = np.asarray(
        [[np.asarray(frame[name][0]) for name in foot_body_names] for frame in frames]
    )
    if positions.shape != (len(frames), len(foot_body_names), 3):
        raise ValueError(f"SMPL-X foot positions have invalid shape {positions.shape}")
    velocity = np.gradient(positions, 1.0 / fps, axis=0)
    speed = np.linalg.norm(velocity, axis=-1)
    contacts = np.zeros(speed.shape, dtype=bool)
    for foot in range(len(foot_body_names)):
        stationary = speed[:, foot] < CONTACT_MAX_SPEED
        if not np.any(stationary):
            continue
        support_height = float(np.median(positions[stationary, foot, 2]))
        contacts[:, foot] = stationary & (
            positions[:, foot, 2] <= support_height + CONTACT_HEIGHT_MARGIN
        )
    return contacts


def _ground_root_to_support(
    qpos: np.ndarray,
    foot_sole_z: np.ndarray,
    source_contacts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if foot_sole_z.shape != source_contacts.shape:
        raise ValueError("sole heights and source contacts must have matching shapes")
    support_height = np.full(len(qpos), np.nan)
    for frame in range(len(qpos)):
        if np.any(source_contacts[frame]):
            support_height[frame] = np.min(foot_sole_z[frame, source_contacts[frame]])
    anchors = np.flatnonzero(np.isfinite(support_height))
    if len(anchors):
        ground_offset = np.interp(
            np.arange(len(qpos)), anchors, support_height[anchors]
        )
        ground_offset = np.minimum(ground_offset, np.min(foot_sole_z, axis=1))
    else:
        ground_offset = np.full(len(qpos), np.min(foot_sole_z))
    grounded_qpos = np.array(qpos, copy=True)
    grounded_qpos[:, 2] -= ground_offset
    return grounded_qpos, foot_sole_z - ground_offset[:, None]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
