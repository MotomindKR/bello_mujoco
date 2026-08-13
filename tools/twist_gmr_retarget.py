#!/usr/bin/env python3
"""Run pinned GMR against a generated canonical Bello MJCF.

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
from typing import NamedTuple

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
GMR_ROOT = ROOT / "deps" / "GMR"
RETARGET_FORMAT_VERSION = 9

ANKLE_TENDON_LOGICAL_JOINT = {
    "left_ankle_motor_1": "left_ankle_pitch_joint",
    "left_ankle_motor_2": "left_ankle_roll_joint",
    "right_ankle_motor_1": "right_ankle_pitch_joint",
    "right_ankle_motor_2": "right_ankle_roll_joint",
}

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
    "head_link",
)

FOOT_SOLE_GEOM_NAMES = (
    "left_ankle_roll_link_collision_box_1",
    "right_ankle_roll_link_collision_box_1",
)
HUMAN_FOOT_NAMES = ("left_foot", "right_foot")
CONTACT_MAX_SPEED = 0.2
CONTACT_HEIGHT_MARGIN = 0.03


class OfflineSolverConfig(NamedTuple):
    passes_per_frame: int
    max_joint_velocity_radians_per_second: float
    max_joint_acceleration_radians_per_second_squared: float
    knee_pole_reference_flexion_radians: float
    knee_target_minimum_flexion_radians: float
    knee_pole_sample_minimum_flexion_radians: float
    knee_pole_sample_minimum_distance_meters: float
    knee_pole_smoothing_frames: int


def main() -> None:
    sys.path.insert(0, str(GMR_ROOT))
    from general_motion_retargeting import params as gmr_params
    from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting

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
    config_path = (
        GMR_ROOT / "general_motion_retargeting" / "ik_configs" / "smplx_to_bello.json"
    )
    solver_config = _load_offline_solver_config(config_path)
    gmr_params.ROBOT_XML_DICT["bello"] = args.robot_xml.expanduser().resolve()
    gmr_params.IK_CONFIG_DICT.setdefault("smplx", {})["bello"] = config_path
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
        source_contacts = _human_foot_contacts(frames, float(fps))
        source_timestamps = np.arange(len(frames), dtype=np.float64) / float(fps)
    retargeter = GeneralMotionRetargeting(
        src_human="smplx",
        tgt_robot="bello",
        actual_human_height=float(human_height),
        verbose=False,
        use_velocity_limit=True,
    )
    retargeter.max_iter = 0
    model = retargeter.model
    joint_names = _logical_actuator_joint_names(model)
    qpos_joint_names = _qpos_ordered_joint_names(model, joint_names)
    joint_ranges = _qpos_ordered_joint_ranges(model, joint_names)
    (
        frames,
        knee_pole_local_axis,
        knee_pole_world_direction,
    ) = _stabilize_knee_bend_targets(
        retargeter,
        frames,
        solver_config,
    )
    _settle_first_frame(retargeter, frames[0], solver_config.passes_per_frame)
    qpos = _retarget_frames(
        retargeter,
        frames,
        fps=float(fps),
        joint_ranges=joint_ranges,
        solver_config=solver_config,
    )
    if qpos.shape[1] != model.nq:
        raise ValueError(f"GMR returned qpos shape {qpos.shape}; model nq={model.nq}")
    foot_sole_z = _foot_sole_z(model, qpos)
    qpos, foot_sole_z = _ground_root_to_support(qpos, foot_sole_z, source_contacts)
    timestamps = np.arange(len(qpos), dtype=np.float64) / float(fps)
    qvel = _differentiate(model, qpos, float(fps))
    body_ids = np.asarray([model.body(name).id for name in BODY_NAMES])
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
        knee_pole_sides=np.asarray(("left", "right")),
        retarget_format_version=np.asarray(RETARGET_FORMAT_VERSION, dtype=np.int32),
        joint_names=np.asarray(joint_names),
        qpos_joint_names=np.asarray(qpos_joint_names),
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


def _is_fastsam_source(path: Path) -> bool:
    with np.load(path, allow_pickle=False) as archive:
        return (
            "source_format" in archive.files
            and str(np.asarray(archive["source_format"]).item())
            == "fastsam-smplx-body-22-v1"
        )


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


def _load_offline_solver_config(path: Path) -> OfflineSolverConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    try:
        solver = raw["offline_solver"]
    except KeyError as error:
        raise ValueError(f"{path}: missing offline_solver configuration") from error
    expected = {
        "knee_pole_reference_flexion_radians",
        "knee_target_minimum_flexion_radians",
        "knee_pole_sample_minimum_distance_meters",
        "knee_pole_sample_minimum_flexion_radians",
        "knee_pole_smoothing_frames",
        "passes_per_frame",
        "max_joint_velocity_radians_per_second",
        "max_joint_acceleration_radians_per_second_squared",
    }
    if not isinstance(solver, dict) or set(solver) != expected:
        raise ValueError(
            f"{path}: offline_solver fields must be exactly {sorted(expected)}"
        )
    config = OfflineSolverConfig(
        passes_per_frame=solver["passes_per_frame"],
        max_joint_velocity_radians_per_second=solver[
            "max_joint_velocity_radians_per_second"
        ],
        max_joint_acceleration_radians_per_second_squared=solver[
            "max_joint_acceleration_radians_per_second_squared"
        ],
        knee_pole_reference_flexion_radians=solver[
            "knee_pole_reference_flexion_radians"
        ],
        knee_target_minimum_flexion_radians=solver[
            "knee_target_minimum_flexion_radians"
        ],
        knee_pole_sample_minimum_flexion_radians=solver[
            "knee_pole_sample_minimum_flexion_radians"
        ],
        knee_pole_sample_minimum_distance_meters=solver[
            "knee_pole_sample_minimum_distance_meters"
        ],
        knee_pole_smoothing_frames=solver["knee_pole_smoothing_frames"],
    )
    if not isinstance(config.passes_per_frame, int) or config.passes_per_frame < 1:
        raise ValueError(f"{path}: offline_solver.passes_per_frame must be positive")
    for name, value in (
        (
            "max_joint_velocity_radians_per_second",
            config.max_joint_velocity_radians_per_second,
        ),
        (
            "max_joint_acceleration_radians_per_second_squared",
            config.max_joint_acceleration_radians_per_second_squared,
        ),
        (
            "knee_pole_reference_flexion_radians",
            config.knee_pole_reference_flexion_radians,
        ),
        (
            "knee_target_minimum_flexion_radians",
            config.knee_target_minimum_flexion_radians,
        ),
        (
            "knee_pole_sample_minimum_flexion_radians",
            config.knee_pole_sample_minimum_flexion_radians,
        ),
        (
            "knee_pole_sample_minimum_distance_meters",
            config.knee_pole_sample_minimum_distance_meters,
        ),
    ):
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(
                f"{path}: offline_solver.{name} must be finite and positive"
            )
    if config.knee_target_minimum_flexion_radians >= math.pi:
        raise ValueError(
            f"{path}: offline_solver.knee_target_minimum_flexion_radians "
            "must be smaller than pi"
        )
    if (
        not isinstance(config.knee_pole_smoothing_frames, int)
        or config.knee_pole_smoothing_frames < 1
        or config.knee_pole_smoothing_frames % 2 == 0
    ):
        raise ValueError(
            f"{path}: offline_solver.knee_pole_smoothing_frames must be a positive odd integer"
        )
    return config


def _stabilize_knee_bend_targets(
    retargeter,
    frames,
    solver_config: OfflineSolverConfig,
) -> tuple[list[dict[str, list[np.ndarray]]], np.ndarray, np.ndarray]:
    """Keep Bello's one-direction knees on the SMPL-X positive-flex branch."""
    if not frames:
        raise ValueError("retargeting requires at least one source frame")
    sides = ("left", "right")
    local_axes = {
        side: _infer_positive_knee_pole_trajectory(frames, side, solver_config)
        for side in sides
    }
    segment_lengths = _robot_leg_segment_lengths(retargeter, sides)
    scaled_root_factor = retargeter.human_scale_table[retargeter.human_root_name]
    stabilized = []
    world_directions = []
    for frame_index, frame in enumerate(frames):
        source = {
            name: [np.asarray(value[0]).copy(), np.asarray(value[1]).copy()]
            for name, value in frame.items()
        }
        scaled = retargeter.scale_human_data(
            {
                name: [value[0].copy(), value[1].copy()]
                for name, value in source.items()
            },
            retargeter.human_root_name,
            retargeter.human_scale_table,
        )
        scaled = retargeter.offset_human_data(
            scaled,
            retargeter.pos_offsets1,
            retargeter.rot_offsets1,
        )
        scaled = retargeter.align_human_data_to_ground(scaled)
        scaled = retargeter.apply_ground_offset(scaled)
        root_position = source[retargeter.human_root_name][0]
        scaled_root_position = scaled_root_factor * root_position
        frame_world_directions = []
        for side in sides:
            hip_name = f"{side}_hip"
            knee_name = f"{side}_knee"
            foot_name = f"{side}_foot"
            hip_position = np.asarray(scaled[hip_name][0])
            foot_position = np.asarray(scaled[foot_name][0])
            hip_rotation = Rotation.from_quat(source[hip_name][1], scalar_first=True)
            pole_direction = hip_rotation.apply(local_axes[side][frame_index])
            pole_direction /= np.linalg.norm(pole_direction)
            frame_world_directions.append(pole_direction)
            knee_target, foot_target = _reachable_leg_targets(
                hip_position,
                foot_position,
                pole_direction,
                *segment_lengths[side],
                solver_config.knee_target_minimum_flexion_radians,
            )
            for body_name, target_position in (
                (knee_name, knee_target),
                (foot_name, foot_target),
            ):
                body_rotation = (
                    Rotation.from_quat(source[body_name][1], scalar_first=True)
                    * retargeter.rot_offsets1[body_name]
                )
                position_offset = body_rotation.apply(
                    retargeter.pos_offsets1[body_name]
                )
                body_scale = retargeter.human_scale_table[body_name]
                if not math.isfinite(body_scale) or body_scale <= 0.0:
                    raise ValueError(f"invalid human scale for {body_name}")
                source[body_name][0] = (
                    root_position
                    + (target_position - position_offset - scaled_root_position)
                    / body_scale
                )
        stabilized.append(source)
        world_directions.append(frame_world_directions)
    return (
        stabilized,
        np.stack([local_axes[side] for side in sides], axis=1),
        np.asarray(world_directions),
    )


def _infer_positive_knee_pole_trajectory(
    frames,
    side: str,
    solver_config: OfflineSolverConfig,
) -> np.ndarray:
    hip_name = f"{side}_hip"
    knee_name = f"{side}_knee"
    foot_name = f"{side}_foot"
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
        if pole_distance >= solver_config.knee_pole_sample_minimum_distance_meters:
            sample_axes[frame_index] = hip_rotation.inv().apply(pole / pole_distance)
    thigh_offset = np.median(np.asarray(thigh_offsets), axis=0)
    shin_offset = np.median(np.asarray(shin_offsets), axis=0)
    reference_foot = thigh_offset + Rotation.from_rotvec(
        (solver_config.knee_pole_reference_flexion_radians, 0.0, 0.0)
    ).apply(shin_offset)
    reference_axis = _normalized_knee_pole(np.zeros(3), thigh_offset, reference_foot)
    finite = np.all(np.isfinite(sample_axes), axis=1)
    hemisphere = np.where(finite, sample_axes @ reference_axis > 0.0, False)
    valid = (
        finite
        & hemisphere
        & (flexions >= solver_config.knee_pole_sample_minimum_flexion_radians)
        & (pole_distances >= solver_config.knee_pole_sample_minimum_distance_meters)
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
    window = solver_config.knee_pole_smoothing_frames
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


def _robot_leg_segment_lengths(retargeter, sides) -> dict[str, tuple[float, float]]:
    mujoco.mj_forward(retargeter.model, retargeter.configuration.data)
    positions = retargeter.configuration.data.xpos
    result = {}
    for side in sides:
        hip_id = retargeter.model.body(f"{side}_hip_roll_link").id
        knee_id = retargeter.model.body(f"{side}_knee_link").id
        foot_id = retargeter.model.body(f"{side}_ankle_roll_link").id
        thigh = float(np.linalg.norm(positions[knee_id] - positions[hip_id]))
        shin = float(np.linalg.norm(positions[foot_id] - positions[knee_id]))
        if min(thigh, shin) <= 0.0 or not np.isfinite((thigh, shin)).all():
            raise ValueError(f"invalid Bello leg segment lengths for {side}")
        result[side] = (thigh, shin)
    return result


def _reachable_leg_targets(
    hip_position: np.ndarray,
    foot_position: np.ndarray,
    pole_direction: np.ndarray,
    thigh_length: float,
    shin_length: float,
    minimum_flexion_radians: float,
) -> tuple[np.ndarray, np.ndarray]:
    chord = np.asarray(foot_position) - np.asarray(hip_position)
    distance = float(np.linalg.norm(chord))
    if not math.isfinite(distance) or distance < 1.0e-8:
        raise ValueError("hip and foot targets must have distinct finite positions")
    chord_axis = chord / distance
    pole = np.asarray(pole_direction) - np.dot(pole_direction, chord_axis) * chord_axis
    pole_norm = float(np.linalg.norm(pole))
    if not math.isfinite(pole_norm) or pole_norm < 1.0e-8:
        basis = np.eye(3)[int(np.argmin(np.abs(chord_axis)))]
        pole = basis - np.dot(basis, chord_axis) * chord_axis
        pole_norm = float(np.linalg.norm(pole))
    pole /= pole_norm
    minimum_reach = abs(thigh_length - shin_length) + 1.0e-6
    maximum_reach = math.sqrt(
        thigh_length**2
        + shin_length**2
        + 2.0 * thigh_length * shin_length * math.cos(minimum_flexion_radians)
    )
    effective_distance = float(np.clip(distance, minimum_reach, maximum_reach))
    along = (thigh_length**2 - shin_length**2 + effective_distance**2) / (
        2.0 * effective_distance
    )
    radial = math.sqrt(max(thigh_length**2 - along**2, 0.0))
    hip_position = np.asarray(hip_position)
    knee_target = hip_position + along * chord_axis + radial * pole
    foot_target = hip_position + effective_distance * chord_axis
    return knee_target, foot_target


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


def _solve_frame(retargeter, frame, passes_per_frame: int) -> np.ndarray:
    for _ in range(passes_per_frame):
        qpos = retargeter.retarget(frame)
    return np.asarray(qpos, dtype=np.float64)


def _settle_first_frame(retargeter, frame, passes_per_frame: int) -> None:
    previous = _solve_frame(retargeter, frame, passes_per_frame)
    for _ in range(9):
        current = _solve_frame(retargeter, frame, passes_per_frame)
        if np.max(np.abs(current - previous)) < 1.0e-4:
            break
        previous = current


def _retarget_frames(
    retargeter,
    frames,
    *,
    fps: float,
    joint_ranges: np.ndarray,
    solver_config: OfflineSolverConfig,
) -> np.ndarray:
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("retargeting frame rate must be finite and positive")
    qpos_frames = []
    previous_qpos = None
    previous_joint_velocity = None
    for frame in frames:
        candidate = _solve_frame(retargeter, frame, solver_config.passes_per_frame)
        if previous_qpos is None:
            limited = candidate
            joint_velocity = np.zeros(candidate.shape[0] - 7, dtype=np.float64)
        else:
            limited, joint_velocity = _limit_scalar_joint_candidate(
                candidate,
                previous_qpos,
                previous_joint_velocity,
                fps=fps,
                joint_ranges=joint_ranges,
                solver_config=solver_config,
            )
        retargeter.configuration.update(limited)
        qpos_frames.append(limited)
        previous_qpos = limited
        previous_joint_velocity = joint_velocity
    return np.stack(qpos_frames)


def _limit_scalar_joint_candidate(
    candidate: np.ndarray,
    previous_qpos: np.ndarray,
    previous_joint_velocity: np.ndarray,
    *,
    fps: float,
    joint_ranges: np.ndarray,
    solver_config: OfflineSolverConfig,
) -> tuple[np.ndarray, np.ndarray]:
    candidate = np.asarray(candidate, dtype=np.float64)
    previous_qpos = np.asarray(previous_qpos, dtype=np.float64)
    previous_joint_velocity = np.asarray(previous_joint_velocity, dtype=np.float64)
    joint_ranges = np.asarray(joint_ranges, dtype=np.float64)
    joint_count = candidate.shape[0] - 7
    if candidate.shape != previous_qpos.shape or candidate.ndim != 1:
        raise ValueError("candidate and previous qpos must have matching vector shapes")
    if previous_joint_velocity.shape != (joint_count,):
        raise ValueError("previous joint velocity has an incompatible shape")
    if joint_ranges.shape != (joint_count, 2):
        raise ValueError("joint ranges have an incompatible shape")

    dt = 1.0 / fps
    maximum_velocity = solver_config.max_joint_velocity_radians_per_second
    maximum_acceleration = (
        solver_config.max_joint_acceleration_radians_per_second_squared
    )
    velocity_step = maximum_acceleration * dt
    desired_velocity = (candidate[7:] - previous_qpos[7:]) / dt

    lower_distance = np.maximum(previous_qpos[7:] - joint_ranges[:, 0], 0.0)
    upper_distance = np.maximum(joint_ranges[:, 1] - previous_qpos[7:], 0.0)
    acceleration_step = maximum_acceleration * dt
    lower_braking_velocity = acceleration_step - np.sqrt(
        acceleration_step**2 + 2.0 * maximum_acceleration * lower_distance
    )
    upper_braking_velocity = -acceleration_step + np.sqrt(
        acceleration_step**2 + 2.0 * maximum_acceleration * upper_distance
    )
    lower_velocity = np.maximum.reduce(
        (
            np.full(joint_count, -maximum_velocity),
            previous_joint_velocity - velocity_step,
            lower_braking_velocity,
        )
    )
    upper_velocity = np.minimum.reduce(
        (
            np.full(joint_count, maximum_velocity),
            previous_joint_velocity + velocity_step,
            upper_braking_velocity,
        )
    )
    if np.any(lower_velocity > upper_velocity + 1.0e-10):
        raise ValueError("joint velocity and position limits have no feasible solution")

    joint_velocity = np.clip(desired_velocity, lower_velocity, upper_velocity)
    limited = candidate.copy()
    limited[7:] = previous_qpos[7:] + joint_velocity * dt
    if np.any(limited[7:] < joint_ranges[:, 0] - 1.0e-10) or np.any(
        limited[7:] > joint_ranges[:, 1] + 1.0e-10
    ):
        raise ValueError("temporal limiter produced a joint-limit violation")
    return limited, joint_velocity


def _logical_actuator_joint_names(model) -> tuple[str, ...]:
    names: list[str] = []
    for actuator_id in range(model.nu):
        transmission = model.actuator_trntype[actuator_id]
        target_id = int(model.actuator_trnid[actuator_id, 0])
        if transmission in {
            mujoco.mjtTrn.mjTRN_JOINT,
            mujoco.mjtTrn.mjTRN_JOINTINPARENT,
        }:
            name = model.joint(target_id).name
        elif transmission == mujoco.mjtTrn.mjTRN_TENDON:
            name = ANKLE_TENDON_LOGICAL_JOINT.get(model.tendon(target_id).name)
        else:
            name = None
        if name is None:
            raise ValueError(
                f"actuator {model.actuator(actuator_id).name!r} has no supported "
                "logical joint mapping"
            )
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("logical actuator joint mappings must be unique")
    return tuple(names)


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
            "Bello model must contain one free root followed by scalar joints"
        )
    if set(qpos_joint_names) != set(actuator_joint_names):
        raise ValueError(
            "Bello qpos and logical actuator joints do not describe the same joints"
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


def _foot_sole_z(model, qpos: np.ndarray) -> np.ndarray:
    geom_ids = []
    for name in FOOT_SOLE_GEOM_NAMES:
        geom_id = model.geom(name).id
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError(f"Bello sole geometry {name!r} must be a box")
        geom_ids.append(geom_id)
    heights = np.empty((len(qpos), len(geom_ids)))
    data = mujoco.MjData(model)
    for frame, position in enumerate(qpos):
        data.qpos[:] = position
        mujoco.mj_forward(model, data)
        for foot, geom_id in enumerate(geom_ids):
            rotation = data.geom_xmat[geom_id].reshape(3, 3)
            vertical_extent = np.abs(rotation[2]) @ model.geom_size[geom_id]
            heights[frame, foot] = data.geom_xpos[geom_id, 2] - vertical_extent
    return heights


def _human_foot_contacts(frames, fps: float) -> np.ndarray:
    positions = np.asarray(
        [[np.asarray(frame[name][0]) for name in HUMAN_FOOT_NAMES] for frame in frames]
    )
    if positions.shape != (len(frames), len(HUMAN_FOOT_NAMES), 3):
        raise ValueError(f"SMPL-X foot positions have invalid shape {positions.shape}")
    velocity = np.gradient(positions, 1.0 / fps, axis=0)
    speed = np.linalg.norm(velocity, axis=-1)
    contacts = np.zeros(speed.shape, dtype=bool)
    for foot in range(len(HUMAN_FOOT_NAMES)):
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
