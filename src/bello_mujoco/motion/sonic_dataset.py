"""Synchronized AMASS references for the Bello SONIC transfer task.

The released SONIC SMPL encoder does not consume the Bello retarget alone.  A
single source frame must retain three views with identical timestamps:

* Bello generalized coordinates and body states for simulation tracking;
* 24 SMPL joint positions plus the SMPL root orientation for the frozen encoder;
* the paired 29-DoF G1 retarget for the six wrist inputs and edge calibration.

This module owns that contract.  Licensed AMASS and SMPL-X inputs remain
outside Git; only deterministic manifests and code are repository content.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

import numpy as np

from bello_mujoco.assets_pipeline import repo_root, stack_description_path
from bello_mujoco.motion.twist_dataset import (
    AssignedSequence,
    AssignedSplit,
    ClipDecision,
    QualityConfig,
    SourceSequence,
    assign_splits,
    clip_quality_reasons,
    load_and_normalize_retargeted_clip,
    load_source_manifest,
    portable_manifest,
)

SONIC_DATASET_SCHEMA_VERSION = 3
SONIC_G1_RETARGET_FORMAT_VERSION = 2
SONIC_SMPL_COORDINATE_CONVENTION = "sonic_g1_root_local_z_up"
SONIC_POLICY_FPS = 50.0
SONIC_SMPL_FUTURE_OFFSETS = tuple(range(10))
SONIC_CRITIC_FUTURE_OFFSETS = tuple(range(0, 50, 5))
SONIC_REQUIRED_FUTURE_HORIZON = max(
    max(SONIC_SMPL_FUTURE_OFFSETS), max(SONIC_CRITIC_FUTURE_OFFSETS)
)
SONIC_TRAIN_SAMPLING = "frame_count_times_manifest_weight"
SONIC_DEFAULT_SHARD_FRAME_CAPACITY = 65_536
SONIC_DEFAULT_SHARD_CLIP_CAPACITY = 256
SONIC_SMPL_JOINT_NAMES = (
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

# MuJoCo qpos order in the pinned GMR G1 asset.
G1_MUJOCO_JOINT_NAMES = (
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

# Action/observation order used by the released SONIC checkpoint.
G1_ISAACLAB_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

G1_WRIST_JOINT_NAMES = G1_ISAACLAB_JOINT_NAMES[23:]


@dataclass(frozen=True)
class SonicCalibrationConfig:
    minimum_samples: int = 1_000
    ridge_regularization: float = 0.1
    minimum_joint_standard_deviation: float = 2.0e-3
    minimum_mean_explained_variance: float = 0.4
    maximum_absolute_coefficient: float = 20.0
    maximum_rms_residual_radians: float = 0.75

    def __post_init__(self) -> None:
        if self.minimum_samples < 2:
            raise ValueError("minimum_samples must be at least two")
        for name in (
            "ridge_regularization",
            "minimum_joint_standard_deviation",
            "minimum_mean_explained_variance",
            "maximum_absolute_coefficient",
            "maximum_rms_residual_radians",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.minimum_mean_explained_variance > 1.0:
            raise ValueError("minimum_mean_explained_variance must not exceed one")


@dataclass(frozen=True)
class SonicShardConfig:
    """Fixed training-shard dimensions shared by every split archive."""

    frame_capacity: int = SONIC_DEFAULT_SHARD_FRAME_CAPACITY
    clip_capacity: int = SONIC_DEFAULT_SHARD_CLIP_CAPACITY

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.frame_capacity, self.clip_capacity)
        ):
            raise TypeError("SONIC shard capacities must be integers")
        if self.frame_capacity <= SONIC_REQUIRED_FUTURE_HORIZON:
            raise ValueError("SONIC shard frame capacity is too small")
        if self.clip_capacity <= 0:
            raise ValueError("SONIC shard clip capacity must be positive")


@dataclass(frozen=True)
class SonicJointCalibration:
    """Bidirectional coupled joint maps fitted on the training split only."""

    g1_joint_names: tuple[str, ...]
    bello_joint_names: tuple[str, ...]
    g1_from_bello_matrix: tuple[tuple[float, ...], ...]
    bello_from_g1_matrix: tuple[tuple[float, ...], ...]
    g1_explained_variance: tuple[float, ...]
    g1_rms_residual_radians: tuple[float, ...]
    bello_explained_variance: tuple[float, ...]
    bello_rms_residual_radians: tuple[float, ...]
    sample_count: int
    training_content_sha256: str

    def __post_init__(self) -> None:
        if len(set(self.g1_joint_names)) != len(self.g1_joint_names):
            raise ValueError("G1 calibration joint names must be unique")
        if len(set(self.bello_joint_names)) != len(self.bello_joint_names):
            raise ValueError("Bello calibration joint names must be unique")
        if self.g1_joint_names != G1_ISAACLAB_JOINT_NAMES:
            raise ValueError(
                "SONIC calibration G1 joint order differs from the release"
            )
        if len(self.bello_joint_names) != 25:
            raise ValueError("SONIC calibration requires Bello's 25 action joints")
        g1_count = len(self.g1_joint_names)
        bello_count = len(self.bello_joint_names)
        g1_from_bello = np.asarray(self.g1_from_bello_matrix, dtype=np.float64)
        bello_from_g1 = np.asarray(self.bello_from_g1_matrix, dtype=np.float64)
        if g1_from_bello.shape != (g1_count, bello_count):
            raise ValueError("SONIC G1-from-Bello calibration matrix has invalid shape")
        if bello_from_g1.shape != (bello_count, g1_count):
            raise ValueError("SONIC Bello-from-G1 calibration matrix has invalid shape")
        vector_shapes = {
            "g1_explained_variance": (self.g1_explained_variance, g1_count),
            "g1_rms_residual_radians": (
                self.g1_rms_residual_radians,
                g1_count,
            ),
            "bello_explained_variance": (
                self.bello_explained_variance,
                bello_count,
            ),
            "bello_rms_residual_radians": (
                self.bello_rms_residual_radians,
                bello_count,
            ),
        }
        for name, (values, expected) in vector_shapes.items():
            if len(values) != expected:
                raise ValueError(f"SONIC calibration {name} has invalid length")
        numeric = (
            g1_from_bello,
            bello_from_g1,
            np.asarray(self.g1_explained_variance),
            np.asarray(self.g1_rms_residual_radians),
            np.asarray(self.bello_explained_variance),
            np.asarray(self.bello_rms_residual_radians),
        )
        if any(not np.all(np.isfinite(value)) for value in numeric):
            raise ValueError("SONIC calibration contains non-finite values")
        explained = self.g1_explained_variance + self.bello_explained_variance
        if any(value > 1.0 + 1.0e-6 for value in explained):
            raise ValueError("SONIC calibration explained variance exceeds one")
        residuals = self.g1_rms_residual_radians + self.bello_rms_residual_radians
        if any(value < 0.0 for value in residuals):
            raise ValueError("SONIC calibration residuals must be nonnegative")
        if self.sample_count < 2:
            raise ValueError("joint calibration requires at least two samples")
        _require_sha256(self.training_content_sha256, "training_content_sha256")

    def to_document(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_document(cls, value: dict[str, Any]) -> SonicJointCalibration:
        expected = {field.name for field in cls.__dataclass_fields__.values()}
        if set(value) != expected:
            raise ValueError(
                "SONIC joint calibration fields differ from the schema: "
                f"missing={sorted(expected - set(value))}, "
                f"extra={sorted(set(value) - expected)}"
            )
        matrix_fields = {"g1_from_bello_matrix", "bello_from_g1_matrix"}
        tuple_fields = expected - {
            "sample_count",
            "training_content_sha256",
            *matrix_fields,
        }
        converted = {}
        for name, raw in value.items():
            if name in matrix_fields:
                converted[name] = tuple(tuple(row) for row in raw)
            elif name in tuple_fields:
                converted[name] = tuple(raw)
            else:
                converted[name] = raw
        return cls(**converted)


@dataclass(frozen=True)
class SonicReferenceBank:
    """One fixed-capacity synchronized shard at the JAX device boundary."""

    qpos: np.ndarray
    qvel: np.ndarray
    body_pos: np.ndarray
    body_quat_wxyz: np.ndarray
    body_lin_vel: np.ndarray
    body_ang_vel: np.ndarray
    foot_contacts: np.ndarray
    smpl_joint_pos: np.ndarray
    smpl_root_quat_wxyz: np.ndarray
    smpl_root_translation: np.ndarray
    g1_joint_pos: np.ndarray
    offsets: np.ndarray
    lengths: np.ndarray
    sampling_probabilities: np.ndarray
    sampling_bin_clip_indices: np.ndarray
    sampling_bin_starts: np.ndarray
    sampling_bin_widths: np.ndarray
    sampling_bin_valid: np.ndarray
    sampling_clip_bin_offsets: np.ndarray
    sampling_bin_size: int
    valid_clip_count: int
    source_ids: tuple[str, ...]
    sequence_ids: tuple[str, ...]
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    calibration: SonicJointCalibration


@dataclass(frozen=True)
class _AcceptedSonicClip:
    assigned: AssignedSequence
    bello_path: Path
    g1_path: Path
    content_sha256: str
    frames: int
    duration_seconds: float


_PACKED_ARRAY_NAMES = (
    "qpos",
    "qvel",
    "body_pos",
    "body_quat_wxyz",
    "body_lin_vel",
    "body_ang_vel",
    "foot_contacts",
    "smpl_joint_pos",
    "smpl_root_quat_wxyz",
    "smpl_root_translation",
    "g1_joint_pos",
)

_SONIC_INDEX_FIELDS = {
    "schema_version",
    "dataset_fingerprint",
    "portable_manifest_fingerprint",
    "policy_fps",
    "smpl_future_offsets",
    "critic_future_offsets",
    "train_sampling",
    "shard_frame_capacity",
    "shard_clip_capacity",
    "shards",
    "joint_names",
    "body_names",
    "g1_joint_names",
    "g1_wrist_joint_names",
    "robot_fingerprint",
    "gmr_revision",
    "bello_gmr_config_sha256",
    "g1_gmr_revision",
    "g1_gmr_config_sha256",
    "g1_robot_sha256",
    "joint_calibration",
    "clips",
    "fingerprint_payload",
}

_SONIC_SHARD_FIELDS = {
    "file",
    "sha256",
    "clip_count",
    "data_frames",
    "sampling_mass",
}
_SONIC_CLIP_FIELDS = {
    "source_kind",
    "source_id",
    "sequence_id",
    "subject_id",
    "split",
    "sampling_weight",
    "shard_index",
    "offset",
    "frames",
    "duration_seconds",
    "source_sha256",
    "synchronized_sha256",
}
_SONIC_TRAIN_CLIP_FIELDS = _SONIC_CLIP_FIELDS | {
    "sampling_probability_numerator",
    "sampling_probability_denominator",
}
_SONIC_EVALUATION_CLIP_FIELDS = _SONIC_CLIP_FIELDS | {"evaluation_order"}


def load_sonic_g1_retarget(path: Path) -> dict[str, np.ndarray]:
    """Load one G1/SMPL sidecar and enforce its immutable frame contract."""

    path = Path(path).expanduser().resolve()
    required = {
        "timestamp_seconds",
        "source_timestamp_seconds",
        "g1_qpos",
        "g1_qvel",
        "g1_qpos_joint_names",
        "smpl_joint_pos",
        "smpl_root_quat_wxyz",
        "smpl_root_translation",
        "smpl_joint_names",
        "source_sha256",
        "retarget_format_version",
        "gmr_revision",
        "gmr_config_sha256",
        "g1_robot_sha256",
        "coordinate_convention",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        extra = sorted(set(archive.files) - required)
        if missing or extra:
            raise ValueError(
                f"{path.name}: sidecar fields differ from schema: "
                f"missing={missing}, extra={extra}"
            )
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    version = int(np.asarray(arrays["retarget_format_version"]).item())
    if version != SONIC_G1_RETARGET_FORMAT_VERSION:
        raise ValueError(
            f"{path.name}: unsupported SONIC G1 retarget version {version}"
        )
    timestamp_array = np.asarray(arrays["timestamp_seconds"])
    if timestamp_array.ndim != 1:
        raise ValueError(f"{path.name}: timestamp_seconds must be one-dimensional")
    frames = len(timestamp_array)
    expected_shapes = {
        "timestamp_seconds": (frames,),
        "source_timestamp_seconds": (frames,),
        "g1_qpos": (frames, 36),
        "g1_qvel": (frames, 35),
        "smpl_joint_pos": (frames, 24, 3),
        "smpl_root_quat_wxyz": (frames, 4),
        "smpl_root_translation": (frames, 3),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(
                f"{path.name}: {name} has shape {arrays[name].shape}; expected {shape}"
            )
    if frames < 2:
        raise ValueError(f"{path.name}: sidecar contains fewer than two frames")
    qpos_names = tuple(str(name) for name in arrays["g1_qpos_joint_names"].tolist())
    if qpos_names != G1_MUJOCO_JOINT_NAMES:
        raise ValueError(f"{path.name}: G1 qpos joint order differs from pinned GMR")
    smpl_joint_names = tuple(str(name) for name in arrays["smpl_joint_names"].tolist())
    if smpl_joint_names != SONIC_SMPL_JOINT_NAMES:
        raise ValueError(f"{path.name}: SMPL joint order differs from the release")
    timestamps = np.asarray(arrays["timestamp_seconds"], dtype=np.float64)
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError(f"{path.name}: timestamps must be finite and increasing")
    if not np.isclose(timestamps[0], 0.0, atol=1.0e-9, rtol=0.0):
        raise ValueError(f"{path.name}: timestamps must start at zero")
    source_timestamps = np.asarray(arrays["source_timestamp_seconds"], dtype=np.float64)
    if not np.all(np.isfinite(source_timestamps)) or np.any(
        np.diff(source_timestamps) <= 0.0
    ):
        raise ValueError(
            f"{path.name}: source timestamps must be finite and increasing"
        )
    if not np.allclose(timestamps, source_timestamps, atol=1.0e-6, rtol=0.0):
        raise ValueError(
            f"{path.name}: G1 and SMPL outputs do not share exact source frames"
        )
    expected_dt = 1.0 / SONIC_POLICY_FPS
    if not np.allclose(np.diff(timestamps), expected_dt, atol=1.0e-6, rtol=0.0):
        raise ValueError(f"{path.name}: sidecar must be sampled at 50 Hz")
    numeric = tuple(expected_shapes)
    if any(not np.all(np.isfinite(arrays[name])) for name in numeric):
        raise ValueError(f"{path.name}: sidecar contains non-finite values")
    quaternion_norm = np.linalg.norm(arrays["smpl_root_quat_wxyz"], axis=-1)
    if not np.allclose(quaternion_norm, 1.0, atol=1.0e-4):
        raise ValueError(f"{path.name}: SMPL root quaternions are not normalized")
    g1_root_quaternion_norm = np.linalg.norm(arrays["g1_qpos"][:, 3:7], axis=-1)
    if not np.allclose(g1_root_quaternion_norm, 1.0, atol=1.0e-4):
        raise ValueError(f"{path.name}: G1 root quaternions are not normalized")
    if (
        str(np.asarray(arrays["coordinate_convention"]).item())
        != SONIC_SMPL_COORDINATE_CONVENTION
    ):
        raise ValueError(f"{path.name}: unsupported SMPL coordinate convention")
    joint_extent = np.linalg.norm(arrays["smpl_joint_pos"], axis=-1)
    if np.max(joint_extent) > 3.0:
        raise ValueError(f"{path.name}: SMPL root-local joint extent is implausible")
    for name in (
        "source_sha256",
        "gmr_config_sha256",
        "g1_robot_sha256",
    ):
        _require_sha256(str(np.asarray(arrays[name]).item()), name)
    revision = str(np.asarray(arrays["gmr_revision"]).item())
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError(f"{path.name}: GMR revision must be 40 lowercase hex digits")
    return arrays


def synchronize_retargeted_clip(
    bello_path: Path,
    g1_path: Path,
    quality: QualityConfig,
) -> dict[str, np.ndarray]:
    """Join Bello and G1/SMPL views without interpolating either side again."""

    bello = load_and_normalize_retargeted_clip(bello_path, quality)
    g1 = load_sonic_g1_retarget(g1_path)
    bello_time = np.asarray(bello["source_timestamp_seconds"], dtype=np.float64)
    g1_time = np.asarray(g1["source_timestamp_seconds"], dtype=np.float64)
    if bello_time.shape != g1_time.shape or not np.allclose(
        bello_time, g1_time, atol=1.0e-6, rtol=0.0
    ):
        raise ValueError("Bello and G1/SMPL retargets do not share exact source frames")
    bello_source = str(np.asarray(bello["source_sha256"]).item()).removeprefix(
        "sha256:"
    )
    g1_source = str(np.asarray(g1["source_sha256"]).item()).removeprefix("sha256:")
    if bello_source != g1_source:
        raise ValueError("Bello and G1/SMPL retargets have different source hashes")
    g1_index = {name: index for index, name in enumerate(G1_MUJOCO_JOINT_NAMES)}
    g1_joint_pos = np.stack(
        [g1["g1_qpos"][:, 7 + g1_index[name]] for name in G1_ISAACLAB_JOINT_NAMES],
        axis=-1,
    ).astype(np.float32)
    with np.load(
        Path(bello_path).expanduser().resolve(), allow_pickle=False
    ) as archive:
        raw_bello_root_quaternion = np.asarray(
            archive["qpos"][0, 3:7], dtype=np.float64
        )
    smpl_root_quaternion = _align_smpl_root_to_bello_world(
        np.asarray(g1["smpl_root_quat_wxyz"], dtype=np.float64),
        raw_bello_root_quaternion,
    )
    # Retain the complete Bello QC/provenance surface while adding the packed
    # SONIC fields.  Only ``_PACKED_ARRAY_NAMES`` are written to the device
    # dataset, but quality validation still needs sole heights, joint limits,
    # and knee-pole metadata here.
    synchronized = {name: np.asarray(value) for name, value in bello.items()}
    synchronized.update(
        {
            name: np.asarray(bello[name], dtype=np.float32)
            for name in _PACKED_ARRAY_NAMES[:7]
        }
    )
    synchronized.update(
        {
            "smpl_joint_pos": np.asarray(g1["smpl_joint_pos"], dtype=np.float32),
            "smpl_root_quat_wxyz": np.asarray(smpl_root_quaternion, dtype=np.float32),
            "smpl_root_translation": np.asarray(
                g1["smpl_root_translation"], dtype=np.float32
            ),
            "g1_joint_pos": g1_joint_pos,
            "timestamp_seconds": np.asarray(bello["timestamp_seconds"]),
            "source_timestamp_seconds": bello_time,
            "joint_names": np.asarray(bello["joint_names"]),
            "qpos_joint_names": np.asarray(bello["qpos_joint_names"]),
            "body_names": np.asarray(bello["body_names"]),
            "source_sha256": np.asarray(bello_source),
            "robot_fingerprint": np.asarray(bello["robot_fingerprint"]),
            "gmr_revision": np.asarray(bello["gmr_revision"]),
            "bello_gmr_config_sha256": np.asarray(bello["gmr_config_sha256"]),
            "g1_gmr_revision": np.asarray(g1["gmr_revision"]),
            "g1_gmr_config_sha256": np.asarray(g1["gmr_config_sha256"]),
            "g1_robot_sha256": np.asarray(g1["g1_robot_sha256"]),
        }
    )
    return synchronized


def _align_smpl_root_to_bello_world(
    smpl_root_quat_wxyz: np.ndarray,
    raw_bello_initial_root_quat_wxyz: np.ndarray,
) -> np.ndarray:
    """Apply Bello's first-heading removal to sidecar root orientations."""

    roots = np.asarray(smpl_root_quat_wxyz, dtype=np.float64)
    initial = np.asarray(raw_bello_initial_root_quat_wxyz, dtype=np.float64)
    if roots.ndim != 2 or roots.shape[1] != 4 or initial.shape != (4,):
        raise ValueError("SONIC root-orientation alignment received invalid shapes")
    if not np.all(np.isfinite(roots)) or not np.all(np.isfinite(initial)):
        raise ValueError("SONIC root-orientation alignment requires finite values")
    initial_norm = float(np.linalg.norm(initial))
    if initial_norm < 1.0e-8:
        raise ValueError("Bello initial root quaternion has near-zero norm")
    initial /= initial_norm
    yaw = math.atan2(
        2.0 * (initial[0] * initial[3] + initial[1] * initial[2]),
        1.0 - 2.0 * (initial[2] ** 2 + initial[3] ** 2),
    )
    alignment = np.asarray(
        (math.cos(-0.5 * yaw), 0.0, 0.0, math.sin(-0.5 * yaw)),
        dtype=np.float64,
    )
    aligned = _numpy_quat_multiply(np.broadcast_to(alignment, roots.shape), roots)
    return aligned / np.maximum(
        np.linalg.norm(aligned, axis=-1, keepdims=True), 1.0e-12
    )


def _numpy_quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def fit_joint_calibration(
    clips: tuple[dict[str, np.ndarray], ...],
    *,
    bello_joint_names: tuple[str, ...],
    content_hashes: tuple[str, ...],
    config: SonicCalibrationConfig = SonicCalibrationConfig(),
) -> SonicJointCalibration:
    """Fit deterministic coupled edge maps on synchronized training data."""

    if not clips or len(clips) != len(content_hashes):
        raise ValueError("joint calibration requires clips and matching content hashes")
    moments = _JointCalibrationMoments(bello_joint_names)
    for clip in clips:
        moments.update(clip)
    return moments.finalize(content_hashes, config)


class _JointCalibrationMoments:
    """Numerically stable sufficient statistics for corpus-scale ridge fits."""

    def __init__(self, bello_joint_names: tuple[str, ...]):
        if len(bello_joint_names) != 25 or len(set(bello_joint_names)) != 25:
            raise ValueError("joint calibration requires 25 unique Bello joints")
        self.bello_joint_names = bello_joint_names
        bello_count = len(bello_joint_names)
        g1_count = len(G1_ISAACLAB_JOINT_NAMES)
        self.sample_count = 0
        self.bello_mean = np.zeros(bello_count, dtype=np.float64)
        self.g1_mean = np.zeros(g1_count, dtype=np.float64)
        self.bello_m2 = np.zeros((bello_count, bello_count), dtype=np.float64)
        self.g1_m2 = np.zeros((g1_count, g1_count), dtype=np.float64)
        self.cross_m2 = np.zeros((bello_count, g1_count), dtype=np.float64)

    def update(self, clip: dict[str, np.ndarray]) -> None:
        qpos_joint_names = tuple(
            str(value) for value in np.asarray(clip["qpos_joint_names"]).tolist()
        )
        qpos_index = {name: index for index, name in enumerate(qpos_joint_names)}
        missing = sorted(set(self.bello_joint_names) - set(qpos_index))
        if missing:
            raise ValueError(f"Bello calibration qpos joints are missing: {missing}")
        qpos = np.asarray(clip["qpos"], dtype=np.float64)
        bello_values = np.stack(
            [
                qpos[:, 7 + qpos_index[bello_name]]
                for bello_name in self.bello_joint_names
            ],
            axis=-1,
        )
        g1_values = np.asarray(clip["g1_joint_pos"], dtype=np.float64)
        if (
            g1_values.shape != (len(bello_values), len(G1_ISAACLAB_JOINT_NAMES))
            or bello_values.shape != (len(g1_values), len(self.bello_joint_names))
            or bello_values.shape[0] == 0
        ):
            raise ValueError("joint calibration clip arrays are inconsistent")
        if not np.all(np.isfinite(bello_values)) or not np.all(np.isfinite(g1_values)):
            raise ValueError("joint calibration requires finite trajectories")
        clip_count = bello_values.shape[0]
        bello_mean = np.mean(bello_values, axis=0)
        g1_mean = np.mean(g1_values, axis=0)
        bello_centered = bello_values - bello_mean
        g1_centered = g1_values - g1_mean
        bello_m2 = bello_centered.T @ bello_centered
        g1_m2 = g1_centered.T @ g1_centered
        cross_m2 = bello_centered.T @ g1_centered
        if self.sample_count == 0:
            self.sample_count = clip_count
            self.bello_mean = bello_mean
            self.g1_mean = g1_mean
            self.bello_m2 = bello_m2
            self.g1_m2 = g1_m2
            self.cross_m2 = cross_m2
            return
        previous_count = self.sample_count
        combined_count = previous_count + clip_count
        merge_weight = previous_count * clip_count / combined_count
        bello_delta = bello_mean - self.bello_mean
        g1_delta = g1_mean - self.g1_mean
        self.bello_mean += bello_delta * clip_count / combined_count
        self.g1_mean += g1_delta * clip_count / combined_count
        self.bello_m2 += bello_m2 + np.outer(bello_delta, bello_delta) * merge_weight
        self.g1_m2 += g1_m2 + np.outer(g1_delta, g1_delta) * merge_weight
        self.cross_m2 += cross_m2 + np.outer(bello_delta, g1_delta) * merge_weight
        self.sample_count = combined_count

    def finalize(
        self,
        content_hashes: tuple[str, ...],
        config: SonicCalibrationConfig,
    ) -> SonicJointCalibration:
        if self.sample_count < config.minimum_samples:
            raise ValueError(
                f"joint calibration has {self.sample_count} frames; "
                f"requires at least {config.minimum_samples}"
            )
        bello_covariance = self.bello_m2 / self.sample_count
        g1_covariance = self.g1_m2 / self.sample_count
        cross_covariance = self.cross_m2 / self.sample_count
        bello_covariance = 0.5 * (bello_covariance + bello_covariance.T)
        g1_covariance = 0.5 * (g1_covariance + g1_covariance.T)
        bello_std = np.sqrt(np.maximum(np.diag(bello_covariance), 0.0))
        g1_std = np.sqrt(np.maximum(np.diag(g1_covariance), 0.0))
        insufficient_bello = [
            name
            for name, standard_deviation in zip(
                self.bello_joint_names, bello_std, strict=True
            )
            if standard_deviation < config.minimum_joint_standard_deviation
        ]
        insufficient_g1 = [
            name
            for name, standard_deviation in zip(
                G1_ISAACLAB_JOINT_NAMES, g1_std, strict=True
            )
            if standard_deviation < config.minimum_joint_standard_deviation
        ]
        if insufficient_bello or insufficient_g1:
            raise ValueError(
                "joint calibration lacks motion variance: "
                f"Bello={insufficient_bello}, G1={insufficient_g1}"
            )
        try:
            g1_from_bello = _standardized_ridge_map(
                bello_covariance,
                cross_covariance,
                bello_std,
                g1_std,
                config.ridge_regularization,
            )
            bello_from_g1 = _standardized_ridge_map(
                g1_covariance,
                cross_covariance.T,
                g1_std,
                bello_std,
                config.ridge_regularization,
            )
        except np.linalg.LinAlgError as error:
            raise ValueError("joint calibration ridge system is singular") from error
        g1_rms, g1_explained = _regression_metrics(
            g1_from_bello,
            source_covariance=bello_covariance,
            target_covariance=g1_covariance,
            cross_covariance=cross_covariance,
        )
        bello_rms, bello_explained = _regression_metrics(
            bello_from_g1,
            source_covariance=g1_covariance,
            target_covariance=bello_covariance,
            cross_covariance=cross_covariance.T,
        )
        maximum_coefficient = max(
            float(np.max(np.abs(g1_from_bello))),
            float(np.max(np.abs(bello_from_g1))),
        )
        if maximum_coefficient > config.maximum_absolute_coefficient:
            raise ValueError(
                "joint calibration coefficient exceeds configured limit: "
                f"{maximum_coefficient:.3f}"
            )
        for name, explained, rms in (
            ("G1-from-Bello", g1_explained, g1_rms),
            ("Bello-from-G1", bello_explained, bello_rms),
        ):
            if float(np.mean(explained)) < config.minimum_mean_explained_variance:
                raise ValueError(
                    f"{name} calibration mean explained variance is "
                    f"{np.mean(explained):.3f}"
                )
            if float(np.max(rms)) > config.maximum_rms_residual_radians:
                raise ValueError(
                    f"{name} calibration maximum RMS residual is {np.max(rms):.3f} rad"
                )
        return SonicJointCalibration(
            g1_joint_names=G1_ISAACLAB_JOINT_NAMES,
            bello_joint_names=self.bello_joint_names,
            g1_from_bello_matrix=_matrix_tuple(g1_from_bello),
            bello_from_g1_matrix=_matrix_tuple(bello_from_g1),
            g1_explained_variance=tuple(float(value) for value in g1_explained),
            g1_rms_residual_radians=tuple(float(value) for value in g1_rms),
            bello_explained_variance=tuple(float(value) for value in bello_explained),
            bello_rms_residual_radians=tuple(float(value) for value in bello_rms),
            sample_count=self.sample_count,
            training_content_sha256=_json_sha256(sorted(content_hashes)),
        )


def _standardized_ridge_map(
    source_covariance: np.ndarray,
    cross_covariance: np.ndarray,
    source_std: np.ndarray,
    target_std: np.ndarray,
    regularization: float,
) -> np.ndarray:
    source_correlation = source_covariance / np.outer(source_std, source_std)
    source_correlation = 0.5 * (source_correlation + source_correlation.T)
    cross_correlation = cross_covariance / np.outer(source_std, target_std)
    standardized = np.linalg.solve(
        source_correlation + regularization * np.eye(len(source_std), dtype=np.float64),
        cross_correlation,
    )
    return target_std[:, None] * standardized.T / source_std[None, :]


def _regression_metrics(
    matrix: np.ndarray,
    *,
    source_covariance: np.ndarray,
    target_covariance: np.ndarray,
    cross_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    residual_variance = (
        np.diag(target_covariance)
        - 2.0 * np.diag(matrix @ cross_covariance)
        + np.diag(matrix @ source_covariance @ matrix.T)
    )
    residual_variance = np.maximum(residual_variance, 0.0)
    target_variance = np.diag(target_covariance)
    explained_variance = np.minimum(
        1.0,
        1.0 - residual_variance / target_variance,
    )
    return np.sqrt(residual_variance), explained_variance


def _matrix_tuple(value: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(item) for item in row) for row in value)


def build_sonic_dataset(
    manifest_path: Path,
    bello_retargeted_root: Path,
    g1_retargeted_root: Path,
    output_root: Path,
    *,
    quality: QualityConfig = QualityConfig(
        future_horizon_steps=SONIC_REQUIRED_FUTURE_HORIZON
    ),
    calibration: SonicCalibrationConfig = SonicCalibrationConfig(),
    sharding: SonicShardConfig = SonicShardConfig(),
) -> dict[str, Any]:
    """Build immutable, fixed-capacity synchronized shards and an index."""

    if quality.policy_fps != SONIC_POLICY_FPS:
        raise ValueError("SONIC synchronized data must use 50 Hz policy frames")
    if quality.future_horizon_steps < SONIC_REQUIRED_FUTURE_HORIZON:
        raise ValueError(
            "SONIC quality horizon must cover the ten 0.1-second critic frames"
        )
    manifest_path = Path(manifest_path).expanduser().resolve()
    bello_root = Path(bello_retargeted_root).expanduser().resolve()
    g1_root = Path(g1_retargeted_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"SONIC dataset output is not empty: {output_root}; builds never overwrite"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    assignments = assign_splits(
        load_source_manifest(
            manifest_path,
            verify_files=True,
            verify_body_models=False,
        )
    )
    decisions: list[ClipDecision] = []
    accepted: list[_AcceptedSonicClip] = []
    joint_names: tuple[str, ...] | None = None
    body_names: tuple[str, ...] | None = None
    provenance: dict[str, str] | None = None
    calibration_moments: _JointCalibrationMoments | None = None
    provenance_fields = (
        "robot_fingerprint",
        "gmr_revision",
        "bello_gmr_config_sha256",
        "g1_gmr_revision",
        "g1_gmr_config_sha256",
        "g1_robot_sha256",
    )
    pinned_provenance = _pinned_sonic_provenance()
    for assigned in assignments:
        sequence = assigned.sequence
        stem = f"{sequence.source_id}__{sequence.sequence_id}.npz"
        bello_path = bello_root / stem
        g1_path = g1_root / stem
        if not bello_path.is_file() or not g1_path.is_file():
            missing = []
            if not bello_path.is_file():
                missing.append("Bello")
            if not g1_path.is_file():
                missing.append("G1/SMPL")
            decisions.append(
                ClipDecision(
                    sequence.source_id,
                    sequence.sequence_id,
                    "rejected",
                    (f"missing {' and '.join(missing)} retarget",),
                )
            )
            continue
        try:
            arrays = synchronize_retargeted_clip(bello_path, g1_path, quality)
            _validate_source_identity(arrays, sequence)
            _validate_pinned_provenance(arrays, pinned_provenance)
            reasons = clip_quality_reasons(arrays, quality)
        except (OSError, TypeError, ValueError, KeyError) as error:
            decisions.append(
                ClipDecision(
                    sequence.source_id,
                    sequence.sequence_id,
                    "rejected",
                    (str(error),),
                )
            )
            continue
        digest = _arrays_sha256(arrays, _PACKED_ARRAY_NAMES)
        frames = int(arrays["qpos"].shape[0])
        duration = float(arrays["timestamp_seconds"][-1])
        if reasons:
            decisions.append(
                ClipDecision(
                    sequence.source_id,
                    sequence.sequence_id,
                    "quarantined",
                    tuple(reasons),
                    frames,
                    duration,
                    digest,
                )
            )
            continue
        decisions.append(
            ClipDecision(
                sequence.source_id,
                sequence.sequence_id,
                "accepted",
                (),
                frames,
                duration,
                digest,
            )
        )
        clip_joint_names = tuple(str(name) for name in arrays["joint_names"].tolist())
        clip_body_names = tuple(str(name) for name in arrays["body_names"].tolist())
        qpos_names = tuple(str(name) for name in arrays["qpos_joint_names"].tolist())
        clip_provenance = {
            name: str(np.asarray(arrays[name]).item()) for name in provenance_fields
        }
        if set(qpos_names) != set(clip_joint_names):
            raise ValueError("SONIC Bello action and qpos joint sets differ")
        if joint_names is None:
            joint_names = clip_joint_names
            body_names = clip_body_names
            provenance = clip_provenance
            calibration_moments = _JointCalibrationMoments(joint_names)
        else:
            if clip_joint_names != joint_names:
                raise ValueError(
                    "SONIC clips do not share one Bello action joint order"
                )
            if clip_body_names != body_names:
                raise ValueError("SONIC clips do not share one Bello body order")
            for name, expected in provenance.items():
                if clip_provenance[name] != expected:
                    raise ValueError(f"SONIC clips do not share one {name}")
        if assigned.split == "train":
            calibration_moments.update(arrays)
        accepted.append(
            _AcceptedSonicClip(
                assigned=assigned,
                bello_path=bello_path,
                g1_path=g1_path,
                content_sha256=digest,
                frames=frames,
                duration_seconds=duration,
            )
        )
        del arrays
    if not accepted:
        _write_conversion_report(output_root / "conversion_report.json", decisions)
        raise ValueError("SONIC dataset build accepted no synchronized clips")
    train = tuple(item for item in accepted if item.assigned.split == "train")
    evaluation = tuple(item for item in accepted if item.assigned.split == "evaluation")
    if not train or not evaluation:
        _write_conversion_report(output_root / "conversion_report.json", decisions)
        raise ValueError("SONIC dataset requires both training and evaluation clips")
    if (
        joint_names is None
        or body_names is None
        or provenance is None
        or calibration_moments is None
    ):
        raise RuntimeError("SONIC accepted-clip contract was not initialized")
    joint_calibration = calibration_moments.finalize(
        tuple(item.content_sha256 for item in train), calibration
    )
    clip_entries: list[dict[str, Any]] = []
    shard_entries: dict[str, list[dict[str, Any]]] = {}
    for split, items in (("train", train), ("evaluation", evaluation)):
        descriptors, split_entries = _write_shards(
            output_root,
            split,
            items,
            quality=quality,
            joint_names=joint_names,
            body_names=body_names,
            provenance=provenance,
            sharding=sharding,
        )
        shard_entries[split] = descriptors
        clip_entries.extend(split_entries)
    _assign_sampling_metadata(clip_entries)
    portable = portable_manifest(assignments)
    heldout = portable_manifest(
        tuple(item for item in assignments if item.split == "evaluation")
    )
    _write_json_once(output_root / "heldout_manifest.json", heldout)
    fingerprint_payload = {
        "schema_version": SONIC_DATASET_SCHEMA_VERSION,
        "portable_manifest": portable,
        "accepted_clip_hashes": sorted(
            (
                f"{item.assigned.sequence.source_id}/"
                f"{item.assigned.sequence.sequence_id}",
                item.content_sha256,
            )
            for item in accepted
        ),
        "quality": asdict(quality),
        "calibration_config": asdict(calibration),
        "sharding": asdict(sharding),
        "joint_calibration": joint_calibration.to_document(),
        "smpl_future_offsets": SONIC_SMPL_FUTURE_OFFSETS,
        "critic_future_offsets": SONIC_CRITIC_FUTURE_OFFSETS,
        "train_sampling": SONIC_TRAIN_SAMPLING,
        "shards": shard_entries,
        "joint_names": joint_names,
        "body_names": body_names,
        "g1_joint_names": G1_ISAACLAB_JOINT_NAMES,
        "g1_wrist_joint_names": G1_WRIST_JOINT_NAMES,
        "provenance": provenance,
        "clips": clip_entries,
    }
    index = {
        "schema_version": SONIC_DATASET_SCHEMA_VERSION,
        "dataset_fingerprint": _json_sha256(fingerprint_payload),
        "portable_manifest_fingerprint": _json_sha256(portable),
        "policy_fps": SONIC_POLICY_FPS,
        "smpl_future_offsets": list(SONIC_SMPL_FUTURE_OFFSETS),
        "critic_future_offsets": list(SONIC_CRITIC_FUTURE_OFFSETS),
        "train_sampling": SONIC_TRAIN_SAMPLING,
        "shard_frame_capacity": sharding.frame_capacity,
        "shard_clip_capacity": sharding.clip_capacity,
        "shards": shard_entries,
        "joint_names": list(joint_names),
        "body_names": list(body_names),
        "g1_joint_names": list(G1_ISAACLAB_JOINT_NAMES),
        "g1_wrist_joint_names": list(G1_WRIST_JOINT_NAMES),
        **provenance,
        "joint_calibration": joint_calibration.to_document(),
        "clips": clip_entries,
        "fingerprint_payload": fingerprint_payload,
    }
    _write_json_once(output_root / "index.json", index)
    _write_conversion_report(output_root / "conversion_report.json", decisions)
    return index


def load_sonic_dataset_index(index_path: Path) -> dict[str, Any]:
    """Load and verify the self-authenticating SONIC dataset index."""

    path = Path(index_path).expanduser().resolve()
    index = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(index, dict):
        raise TypeError("SONIC dataset index root must be a mapping")
    missing = sorted(_SONIC_INDEX_FIELDS - set(index))
    extra = sorted(set(index) - _SONIC_INDEX_FIELDS)
    if missing or extra:
        raise ValueError(
            "SONIC dataset index fields differ from schema: "
            f"missing={missing}, extra={extra}"
        )
    if index["schema_version"] != SONIC_DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported SONIC packed dataset schema")
    if index["policy_fps"] != SONIC_POLICY_FPS:
        raise ValueError("SONIC dataset policy frequency differs from 50 Hz")
    if tuple(index["smpl_future_offsets"]) != SONIC_SMPL_FUTURE_OFFSETS:
        raise ValueError("SONIC SMPL future-frame contract has drifted")
    if tuple(index["critic_future_offsets"]) != SONIC_CRITIC_FUTURE_OFFSETS:
        raise ValueError("SONIC critic future-frame contract has drifted")
    if index["train_sampling"] != SONIC_TRAIN_SAMPLING:
        raise ValueError("SONIC training-sampling contract has drifted")
    if index["g1_joint_names"] != list(G1_ISAACLAB_JOINT_NAMES):
        raise ValueError("SONIC G1 joint order differs from the release")
    if index["g1_wrist_joint_names"] != list(G1_WRIST_JOINT_NAMES):
        raise ValueError("SONIC G1 wrist order differs from the release")
    sharding = SonicShardConfig(
        frame_capacity=index["shard_frame_capacity"],
        clip_capacity=index["shard_clip_capacity"],
    )
    if not isinstance(index["shards"], dict) or set(index["shards"]) != {
        "train",
        "evaluation",
    }:
        raise ValueError("SONIC dataset must define train and evaluation shards")
    for split, descriptors in index["shards"].items():
        if not isinstance(descriptors, list) or not descriptors:
            raise ValueError(f"SONIC dataset has no {split} shards")
        for shard_index, descriptor in enumerate(descriptors):
            _validate_shard_descriptor(
                descriptor,
                split=split,
                shard_index=shard_index,
                sharding=sharding,
            )
    for name in (
        "dataset_fingerprint",
        "portable_manifest_fingerprint",
        "robot_fingerprint",
        "bello_gmr_config_sha256",
        "g1_gmr_config_sha256",
        "g1_robot_sha256",
    ):
        _require_sha256(index[name], name)
    for name in ("gmr_revision", "g1_gmr_revision"):
        revision = index[name]
        if (
            not isinstance(revision, str)
            or len(revision) != 40
            or any(char not in "0123456789abcdef" for char in revision)
        ):
            raise ValueError(f"SONIC {name} must be a 40-character Git revision")
    payload = index["fingerprint_payload"]
    if (
        not isinstance(payload, dict)
        or _json_sha256(payload) != index["dataset_fingerprint"]
    ):
        raise ValueError("SONIC dataset fingerprint does not match its payload")
    if (
        _json_sha256(payload.get("portable_manifest"))
        != index["portable_manifest_fingerprint"]
    ):
        raise ValueError("SONIC portable-manifest fingerprint has drifted")
    duplicate_contract = {
        "smpl_future_offsets": index["smpl_future_offsets"],
        "critic_future_offsets": index["critic_future_offsets"],
        "train_sampling": index["train_sampling"],
        "sharding": {
            "frame_capacity": index["shard_frame_capacity"],
            "clip_capacity": index["shard_clip_capacity"],
        },
        "shards": index["shards"],
        "joint_names": index["joint_names"],
        "body_names": index["body_names"],
        "g1_joint_names": index["g1_joint_names"],
        "g1_wrist_joint_names": index["g1_wrist_joint_names"],
        "provenance": {
            name: index[name]
            for name in (
                "robot_fingerprint",
                "gmr_revision",
                "bello_gmr_config_sha256",
                "g1_gmr_revision",
                "g1_gmr_config_sha256",
                "g1_robot_sha256",
            )
        },
        "clips": index["clips"],
        "joint_calibration": index["joint_calibration"],
    }
    mismatches = sorted(
        name for name, value in duplicate_contract.items() if payload.get(name) != value
    )
    if mismatches:
        raise ValueError(
            "SONIC index differs from its fingerprint payload for: "
            + ", ".join(mismatches)
        )
    if (
        not isinstance(index["joint_names"], list)
        or len(index["joint_names"]) != 25
        or any(not isinstance(name, str) or not name for name in index["joint_names"])
        or len(set(index["joint_names"])) != 25
    ):
        raise ValueError("SONIC dataset requires 25 unique Bello action joints")
    if (
        not isinstance(index["body_names"], list)
        or not index["body_names"]
        or any(not isinstance(name, str) or not name for name in index["body_names"])
        or len(set(index["body_names"])) != len(index["body_names"])
    ):
        raise ValueError("SONIC dataset body names must be nonempty and unique")
    SonicJointCalibration.from_document(index["joint_calibration"])
    _validate_index_clip_bounds(index, sharding)
    return index


def load_sonic_reference_bank(
    index_path: Path,
    *,
    split: AssignedSplit,
    shard_index: int,
    sampling_bin_size: int = 50,
) -> SonicReferenceBank:
    """Load one verified shard without materializing the rest of the corpus."""

    index_path = Path(index_path).expanduser().resolve()
    index = load_sonic_dataset_index(index_path)
    if split not in {"train", "evaluation"}:
        raise ValueError(f"invalid SONIC dataset split {split!r}")
    if isinstance(shard_index, bool) or not isinstance(shard_index, int):
        raise TypeError("SONIC shard index must be an integer")
    descriptors = index["shards"][split]
    if not 0 <= shard_index < len(descriptors):
        raise IndexError(f"SONIC {split} shard index is out of range: {shard_index}")
    descriptor = descriptors[shard_index]
    clips = [
        clip
        for clip in index["clips"]
        if clip["split"] == split and clip["shard_index"] == shard_index
    ]
    if not clips:
        raise ValueError(f"SONIC {split} shard contains no clips")
    frame_capacity = int(index["shard_frame_capacity"])
    clip_capacity = int(index["shard_clip_capacity"])
    actual_lengths = np.asarray([int(clip["frames"]) for clip in clips], dtype=np.int32)
    actual_offsets = np.asarray([int(clip["offset"]) for clip in clips], dtype=np.int32)
    lengths = np.full(
        (clip_capacity,), SONIC_REQUIRED_FUTURE_HORIZON + 1, dtype=np.int32
    )
    offsets = np.zeros((clip_capacity,), dtype=np.int32)
    lengths[: len(clips)] = actual_lengths
    offsets[: len(clips)] = actual_offsets
    shard_path = _shard_path(index_path, descriptor)
    _validate_shard_file(descriptor, shard_path, split=split, shard_index=shard_index)
    with np.load(shard_path, allow_pickle=False) as archive:
        missing = sorted(set(_PACKED_ARRAY_NAMES) - set(archive.files))
        extra = sorted(set(archive.files) - set(_PACKED_ARRAY_NAMES))
        if missing or extra:
            raise ValueError(
                "SONIC shard fields differ from the schema: "
                f"missing={missing}, extra={extra}"
            )
        arrays = {name: np.asarray(archive[name]) for name in _PACKED_ARRAY_NAMES}
    _validate_loaded_shard_arrays(
        arrays,
        frame_capacity=frame_capacity,
        data_frames=int(descriptor["data_frames"]),
        body_count=len(index["body_names"]),
        split=split,
        shard_index=shard_index,
    )
    actual_weights = (
        np.ones(len(clips), dtype=np.float64)
        if split == "evaluation"
        else np.asarray(
            [clip["sampling_probability_numerator"] for clip in clips],
            dtype=np.float64,
        )
    )
    probabilities = np.zeros((clip_capacity,), dtype=np.float32)
    probabilities[: len(clips)] = (actual_weights / actual_weights.sum()).astype(
        np.float32
    )
    bins = build_padded_sonic_sampling_bins(
        actual_lengths,
        bin_size=sampling_bin_size,
        frame_capacity=frame_capacity,
        clip_capacity=clip_capacity,
    )
    padding = ("",) * (clip_capacity - len(clips))
    return SonicReferenceBank(
        **arrays,
        offsets=offsets,
        lengths=lengths,
        sampling_probabilities=probabilities,
        sampling_bin_clip_indices=bins[0],
        sampling_bin_starts=bins[1],
        sampling_bin_widths=bins[2],
        sampling_bin_valid=bins[3],
        sampling_clip_bin_offsets=bins[4],
        sampling_bin_size=sampling_bin_size,
        valid_clip_count=len(clips),
        source_ids=tuple(clip["source_id"] for clip in clips) + padding,
        sequence_ids=tuple(clip["sequence_id"] for clip in clips) + padding,
        joint_names=tuple(index["joint_names"]),
        body_names=tuple(index["body_names"]),
        calibration=SonicJointCalibration.from_document(index["joint_calibration"]),
    )


def materialize_sonic_clip(
    index_path: Path,
    output_path: Path,
    *,
    split: AssignedSplit,
    source_id: str | None = None,
    sequence_id: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Materialize one synchronized Bello view for environment construction."""

    index_path = Path(index_path).expanduser().resolve()
    index = load_sonic_dataset_index(index_path)
    if split not in {"train", "evaluation"}:
        raise ValueError(f"invalid SONIC dataset split {split!r}")
    candidates = [
        clip
        for clip in index["clips"]
        if clip["split"] == split
        and (source_id is None or clip["source_id"] == source_id)
        and (sequence_id is None or clip["sequence_id"] == sequence_id)
    ]
    if not candidates:
        raise ValueError(f"SONIC dataset contains no matching {split} clip")
    clip = sorted(
        candidates, key=lambda value: (value["source_id"], value["sequence_id"])
    )[0]
    shard_index = int(clip["shard_index"])
    descriptor = index["shards"][split][shard_index]
    shard_path = _shard_path(index_path, descriptor)
    _validate_shard_file(descriptor, shard_path, split=split, shard_index=shard_index)
    start = int(clip["offset"])
    stop = start + int(clip["frames"])
    with np.load(shard_path, allow_pickle=False) as packed:
        arrays = {
            name: np.asarray(packed[name][start:stop])
            for name in _PACKED_ARRAY_NAMES[:7]
        }
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_runtime_clip(
        output_path,
        policy_fps=float(index["policy_fps"]),
        joint_names=tuple(index["joint_names"]),
        body_names=tuple(index["body_names"]),
        source_sha256=clip["source_sha256"],
        arrays=arrays,
    )
    return output_path, clip


def materialize_sonic_reference_clip(
    bank: SonicReferenceBank,
    output_path: Path,
    *,
    clip_index: int,
    source_sha256: str,
    policy_fps: float = SONIC_POLICY_FPS,
) -> Path:
    """Write one native-motion clip from an already verified in-memory bank."""

    if clip_index < 0 or clip_index >= bank.valid_clip_count:
        raise IndexError(f"SONIC clip index is out of range: {clip_index}")
    if policy_fps != SONIC_POLICY_FPS:
        raise ValueError("SONIC runtime clips must use 50 Hz policy frames")
    _require_sha256(source_sha256, "source_sha256")
    start = int(bank.offsets[clip_index])
    stop = start + int(bank.lengths[clip_index])
    arrays = {
        name: np.asarray(getattr(bank, name))[start:stop]
        for name in _PACKED_ARRAY_NAMES[:7]
    }
    return _write_runtime_clip(
        output_path,
        policy_fps=policy_fps,
        joint_names=bank.joint_names,
        body_names=bank.body_names,
        source_sha256=source_sha256,
        arrays=arrays,
    )


def select_sonic_reference_clip(
    bank: SonicReferenceBank, clip_index: int
) -> SonicReferenceBank:
    """Return one clip with rebased offsets for deterministic evaluation."""

    if clip_index < 0 or clip_index >= bank.valid_clip_count:
        raise IndexError(f"SONIC clip index is out of range: {clip_index}")
    start = int(bank.offsets[clip_index])
    length = int(bank.lengths[clip_index])
    stop = start + length
    arrays = {
        name: np.asarray(getattr(bank, name))[start:stop].copy()
        for name in _PACKED_ARRAY_NAMES
    }
    bins = build_padded_sonic_sampling_bins(
        np.asarray((length,), dtype=np.int32),
        bin_size=bank.sampling_bin_size,
        frame_capacity=length,
        clip_capacity=1,
    )
    return SonicReferenceBank(
        **arrays,
        offsets=np.asarray((0,), dtype=np.int32),
        lengths=np.asarray((length,), dtype=np.int32),
        sampling_probabilities=np.asarray((1.0,), dtype=np.float32),
        sampling_bin_clip_indices=bins[0],
        sampling_bin_starts=bins[1],
        sampling_bin_widths=bins[2],
        sampling_bin_valid=bins[3],
        sampling_clip_bin_offsets=bins[4],
        sampling_bin_size=bank.sampling_bin_size,
        valid_clip_count=1,
        source_ids=(bank.source_ids[clip_index],),
        sequence_ids=(bank.sequence_ids[clip_index],),
        joint_names=bank.joint_names,
        body_names=bank.body_names,
        calibration=bank.calibration,
    )


def select_padded_sonic_reference_clip(
    bank: SonicReferenceBank, clip_index: int
) -> SonicReferenceBank:
    """Select one clip without changing fixed shard-array capacities."""

    if clip_index < 0 or clip_index >= bank.valid_clip_count:
        raise IndexError(f"SONIC clip index is out of range: {clip_index}")
    frame_capacity = int(bank.qpos.shape[0])
    clip_capacity = int(len(bank.lengths))
    length = int(bank.lengths[clip_index])
    lengths = np.full(
        (clip_capacity,), SONIC_REQUIRED_FUTURE_HORIZON + 1, dtype=np.int32
    )
    offsets = np.zeros((clip_capacity,), dtype=np.int32)
    probabilities = np.zeros((clip_capacity,), dtype=np.float32)
    lengths[0] = length
    offsets[0] = int(bank.offsets[clip_index])
    probabilities[0] = 1.0
    bins = build_padded_sonic_sampling_bins(
        np.asarray((length,), dtype=np.int32),
        bin_size=bank.sampling_bin_size,
        frame_capacity=frame_capacity,
        clip_capacity=clip_capacity,
    )
    padding = ("",) * (clip_capacity - 1)
    return SonicReferenceBank(
        **{name: getattr(bank, name) for name in _PACKED_ARRAY_NAMES},
        offsets=offsets,
        lengths=lengths,
        sampling_probabilities=probabilities,
        sampling_bin_clip_indices=bins[0],
        sampling_bin_starts=bins[1],
        sampling_bin_widths=bins[2],
        sampling_bin_valid=bins[3],
        sampling_clip_bin_offsets=bins[4],
        sampling_bin_size=bank.sampling_bin_size,
        valid_clip_count=1,
        source_ids=(bank.source_ids[clip_index],) + padding,
        sequence_ids=(bank.sequence_ids[clip_index],) + padding,
        joint_names=bank.joint_names,
        body_names=bank.body_names,
        calibration=bank.calibration,
    )


def sample_sonic_start_frames(rng, lengths, probabilities, *, batch_size: int):
    """JAX-compatible clip/start sampling that preserves every future window."""

    import jax
    import jax.numpy as jnp

    clip_key, frame_key = jax.random.split(rng)
    clip_indices = jax.random.categorical(
        clip_key, jnp.log(jnp.asarray(probabilities)), shape=(batch_size,)
    ).astype(jnp.int32)
    maximum_start = (
        jnp.asarray(lengths)[clip_indices] - SONIC_REQUIRED_FUTURE_HORIZON - 1
    )
    start_frames = jnp.floor(
        jax.random.uniform(frame_key, (batch_size,)) * (maximum_start + 1)
    ).astype(jnp.int32)
    return clip_indices, start_frames


def _write_shards(
    output_root: Path,
    split: AssignedSplit,
    items: tuple[_AcceptedSonicClip, ...],
    *,
    quality: QualityConfig,
    joint_names: tuple[str, ...],
    body_names: tuple[str, ...],
    provenance: dict[str, str],
    sharding: SonicShardConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups = _partition_shards(items, sharding)
    shard_root = output_root / "shards" / split
    shard_root.mkdir(parents=True, exist_ok=False)
    descriptors: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    for shard_index, group in enumerate(groups):
        relative_path = Path("shards") / split / f"{shard_index:05d}.npz"
        path = output_root / relative_path
        shard_entries = _write_shard(
            path,
            group,
            shard_index=shard_index,
            quality=quality,
            joint_names=joint_names,
            body_names=body_names,
            provenance=provenance,
            frame_capacity=sharding.frame_capacity,
        )
        data_frames = sum(item.frames for item in group)
        sampling_mass = sum(
            item.frames * item.assigned.sequence.sampling_weight for item in group
        )
        descriptors.append(
            {
                "file": relative_path.as_posix(),
                "sha256": "sha256:" + _file_sha256(path),
                "clip_count": len(group),
                "data_frames": data_frames,
                "sampling_mass": sampling_mass,
            }
        )
        entries.extend(shard_entries)
    return descriptors, entries


def _partition_shards(
    items: tuple[_AcceptedSonicClip, ...], sharding: SonicShardConfig
) -> tuple[tuple[_AcceptedSonicClip, ...], ...]:
    groups: list[tuple[_AcceptedSonicClip, ...]] = []
    current: list[_AcceptedSonicClip] = []
    current_frames = 0
    for item in items:
        identity = (
            f"{item.assigned.sequence.source_id}/{item.assigned.sequence.sequence_id}"
        )
        if item.frames > sharding.frame_capacity:
            raise ValueError(
                f"SONIC clip {identity} has {item.frames} frames, exceeding shard "
                f"capacity {sharding.frame_capacity}"
            )
        full = current and (
            current_frames + item.frames > sharding.frame_capacity
            or len(current) == sharding.clip_capacity
        )
        if full:
            groups.append(tuple(current))
            current = []
            current_frames = 0
        current.append(item)
        current_frames += item.frames
    if current:
        groups.append(tuple(current))
    if not groups:
        raise ValueError("SONIC shard partition contains no clips")
    return tuple(groups)


def _write_shard(
    path: Path,
    items: tuple[_AcceptedSonicClip, ...],
    *,
    shard_index: int,
    quality: QualityConfig,
    joint_names: tuple[str, ...],
    body_names: tuple[str, ...],
    provenance: dict[str, str],
    frame_capacity: int,
) -> list[dict[str, Any]]:
    first = _reload_accepted_clip(
        items[0],
        quality=quality,
        joint_names=joint_names,
        body_names=body_names,
        provenance=provenance,
    )
    entries: list[dict[str, Any]] = []
    archive_descriptor, archive_name = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".npz", dir=path.parent
    )
    os.close(archive_descriptor)
    temporary_archive = Path(archive_name)
    with tempfile.TemporaryDirectory(prefix=f".{path.stem}-", dir=path.parent) as tmp:
        try:
            temporary_root = Path(tmp)
            packed = {
                name: np.lib.format.open_memmap(
                    temporary_root / f"{name}.npy",
                    mode="w+",
                    dtype=np.float32,
                    shape=(frame_capacity,) + np.asarray(first[name]).shape[1:],
                )
                for name in _PACKED_ARRAY_NAMES
            }
            offset = 0
            for item_index, item in enumerate(items):
                arrays = (
                    first
                    if item_index == 0
                    else _reload_accepted_clip(
                        item,
                        quality=quality,
                        joint_names=joint_names,
                        body_names=body_names,
                        provenance=provenance,
                    )
                )
                frames = int(arrays["qpos"].shape[0])
                stop = offset + frames
                for name, destination in packed.items():
                    value = np.asarray(arrays[name], dtype=np.float32)
                    if value.shape != (frames,) + destination.shape[1:]:
                        raise ValueError(
                            f"{item.assigned.sequence.sequence_id}: {name} shape "
                            f"{value.shape} differs from the shard contract"
                        )
                    destination[offset:stop] = value
                entries.append(_clip_entry(item, offset, shard_index))
                offset = stop
                if item_index != 0:
                    del arrays
            del first
            for destination in packed.values():
                destination.flush()
            np.savez_compressed(temporary_archive, **packed)
            del packed
            os.replace(temporary_archive, path)
        except BaseException:
            temporary_archive.unlink(missing_ok=True)
            raise
    return entries


def _reload_accepted_clip(
    item: _AcceptedSonicClip,
    *,
    quality: QualityConfig,
    joint_names: tuple[str, ...],
    body_names: tuple[str, ...],
    provenance: dict[str, str],
) -> dict[str, np.ndarray]:
    arrays = synchronize_retargeted_clip(item.bello_path, item.g1_path, quality)
    _validate_source_identity(arrays, item.assigned.sequence)
    reasons = clip_quality_reasons(arrays, quality)
    if reasons:
        raise ValueError(
            f"{item.assigned.sequence.sequence_id}: quality changed between "
            f"dataset passes: {', '.join(reasons)}"
        )
    if _arrays_sha256(arrays, _PACKED_ARRAY_NAMES) != item.content_sha256:
        raise ValueError(
            f"{item.assigned.sequence.sequence_id}: synchronized content changed "
            "between dataset passes"
        )
    if int(arrays["qpos"].shape[0]) != item.frames or not math.isclose(
        float(arrays["timestamp_seconds"][-1]),
        item.duration_seconds,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError(
            f"{item.assigned.sequence.sequence_id}: clip bounds changed between "
            "dataset passes"
        )
    actual_joint_names = tuple(
        str(name) for name in np.asarray(arrays["joint_names"]).tolist()
    )
    actual_qpos_names = tuple(
        str(name) for name in np.asarray(arrays["qpos_joint_names"]).tolist()
    )
    actual_body_names = tuple(
        str(name) for name in np.asarray(arrays["body_names"]).tolist()
    )
    if actual_joint_names != joint_names or set(actual_qpos_names) != set(joint_names):
        raise ValueError("SONIC Bello joint order changed between dataset passes")
    if actual_body_names != body_names:
        raise ValueError("SONIC Bello body order changed between dataset passes")
    for name, expected in provenance.items():
        if str(np.asarray(arrays[name]).item()) != expected:
            raise ValueError(f"SONIC {name} changed between dataset passes")
    return arrays


def _clip_entry(
    item: _AcceptedSonicClip, offset: int, shard_index: int
) -> dict[str, Any]:
    assigned = item.assigned
    sequence = assigned.sequence
    return {
        "source_kind": sequence.source_kind,
        "source_id": sequence.source_id,
        "sequence_id": sequence.sequence_id,
        "subject_id": sequence.subject_id,
        "split": assigned.split,
        "sampling_weight": sequence.sampling_weight,
        "shard_index": shard_index,
        "offset": offset,
        "frames": item.frames,
        "duration_seconds": item.duration_seconds,
        "source_sha256": "sha256:" + sequence.source_sha256.removeprefix("sha256:"),
        "synchronized_sha256": item.content_sha256,
    }


def _assign_sampling_metadata(entries: list[dict[str, Any]]) -> None:
    train_mass = sum(
        entry["frames"] * entry["sampling_weight"]
        for entry in entries
        if entry["split"] == "train"
    )
    evaluation_order = {
        (entry["source_id"], entry["sequence_id"]): index
        for index, entry in enumerate(
            sorted(
                (entry for entry in entries if entry["split"] == "evaluation"),
                key=lambda item: (item["source_id"], item["sequence_id"]),
            )
        )
    }
    for entry in entries:
        if entry["split"] == "train":
            entry["sampling_probability_numerator"] = (
                entry["frames"] * entry["sampling_weight"]
            )
            entry["sampling_probability_denominator"] = train_mass
        else:
            entry["evaluation_order"] = evaluation_order[
                (entry["source_id"], entry["sequence_id"])
            ]


def _validate_source_identity(
    arrays: dict[str, np.ndarray], sequence: SourceSequence
) -> None:
    expected = sequence.source_sha256.removeprefix("sha256:")
    actual = str(np.asarray(arrays["source_sha256"]).item()).removeprefix("sha256:")
    if actual != expected:
        raise ValueError("synchronized retarget source hash differs from the manifest")


def _pinned_sonic_provenance() -> dict[str, str]:
    repository = repo_root()
    gmr_root = repository / "deps" / "GMR"
    paths = {
        "robot_fingerprint": stack_description_path(
            "bello/full_body/urdf/bello_full_body.urdf"
        ),
        "bello_gmr_config_sha256": (
            gmr_root
            / "general_motion_retargeting"
            / "ik_configs"
            / "smplx_to_bello.json"
        ),
        "g1_gmr_config_sha256": (
            gmr_root / "general_motion_retargeting" / "ik_configs" / "smplx_to_g1.json"
        ),
        "g1_robot_sha256": gmr_root / "assets" / "unitree_g1" / "g1_mocap_29dof.xml",
    }
    missing = sorted(str(path) for path in paths.values() if not path.is_file())
    if missing:
        raise FileNotFoundError(
            "pinned SONIC provenance input is missing: " + ", ".join(missing)
        )
    revision = _git_revision(gmr_root)
    return {
        **{name: "sha256:" + _file_sha256(path) for name, path in paths.items()},
        "gmr_revision": revision,
        "g1_gmr_revision": revision,
    }


def _validate_pinned_provenance(
    arrays: dict[str, np.ndarray], expected: dict[str, str]
) -> None:
    actual = {name: str(np.asarray(arrays[name]).item()) for name in expected}
    for name in (
        "robot_fingerprint",
        "bello_gmr_config_sha256",
        "g1_gmr_config_sha256",
        "g1_robot_sha256",
    ):
        actual[name] = "sha256:" + actual[name].removeprefix("sha256:")
    mismatches = sorted(name for name in expected if actual[name] != expected[name])
    if mismatches:
        raise ValueError(
            "synchronized retarget provenance differs from the pinned stack for: "
            + ", ".join(mismatches)
        )


def _arrays_sha256(arrays: dict[str, np.ndarray], names: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name in names:
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return "sha256:" + digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(root: Path) -> str:
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError(f"invalid GMR revision from {root}: {revision!r}")
    return revision


def _validate_shard_descriptor(
    descriptor: Any,
    *,
    split: str,
    shard_index: int,
    sharding: SonicShardConfig,
) -> None:
    if not isinstance(descriptor, dict) or set(descriptor) != _SONIC_SHARD_FIELDS:
        raise ValueError(
            f"SONIC {split} shard {shard_index} descriptor differs from the schema"
        )
    expected_file = f"shards/{split}/{shard_index:05d}.npz"
    if descriptor["file"] != expected_file:
        raise ValueError(
            f"SONIC {split} shard {shard_index} path is not canonical: "
            f"{descriptor['file']!r}"
        )
    _require_sha256(descriptor["sha256"], f"{split} shard {shard_index} sha256")
    for name, capacity in (
        ("clip_count", sharding.clip_capacity),
        ("data_frames", sharding.frame_capacity),
    ):
        value = descriptor[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 < value <= capacity
        ):
            raise ValueError(
                f"SONIC {split} shard {shard_index} {name} is out of range"
            )
    mass = descriptor["sampling_mass"]
    if isinstance(mass, bool) or not isinstance(mass, (int, float)):
        raise ValueError(f"SONIC {split} shard {shard_index} sampling mass is invalid")
    if not math.isfinite(float(mass)) or float(mass) <= 0.0:
        raise ValueError(
            f"SONIC {split} shard {shard_index} sampling mass must be positive"
        )


def _validate_index_clip_bounds(
    index: dict[str, Any], sharding: SonicShardConfig
) -> None:
    clips = index["clips"]
    if not isinstance(clips, list) or not clips:
        raise ValueError("SONIC dataset clips must be a nonempty list")
    split_counts = {"train": 0, "evaluation": 0}
    for clip in clips:
        if not isinstance(clip, dict):
            raise ValueError("SONIC clip metadata must be a mapping")
        split = clip.get("split")
        if split not in split_counts:
            raise ValueError(f"SONIC clip has invalid split {split!r}")
        expected_fields = (
            _SONIC_TRAIN_CLIP_FIELDS
            if split == "train"
            else _SONIC_EVALUATION_CLIP_FIELDS
        )
        if set(clip) != expected_fields:
            raise ValueError(
                f"SONIC {split} clip fields differ from the dataset schema"
            )
        split_counts[split] += 1
        if clip["source_kind"] not in {"amass", "omomo", "in_house"}:
            raise ValueError("SONIC clip source kind is invalid")
        for name in ("source_id", "sequence_id", "subject_id"):
            if not isinstance(clip[name], str) or not clip[name]:
                raise ValueError(f"SONIC clip {name} must be a nonempty string")
        frames = clip["frames"]
        offset = clip["offset"]
        shard_index = clip["shard_index"]
        if (
            isinstance(frames, bool)
            or not isinstance(frames, int)
            or frames <= SONIC_REQUIRED_FUTURE_HORIZON
            or isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
            or isinstance(shard_index, bool)
            or not isinstance(shard_index, int)
            or not 0 <= shard_index < len(index["shards"][split])
        ):
            raise ValueError("SONIC clip bounds are invalid")
        duration = clip["duration_seconds"]
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or not math.isclose(
                float(duration),
                (frames - 1) / SONIC_POLICY_FPS,
                rel_tol=0.0,
                abs_tol=1.0e-6,
            )
        ):
            raise ValueError("SONIC clip duration differs from its 50 Hz frame count")
        weight = clip["sampling_weight"]
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) <= 0.0
        ):
            raise ValueError("SONIC clip sampling weight is invalid")
        _require_sha256(clip["source_sha256"], "clip source_sha256")
        _require_sha256(clip["synchronized_sha256"], "clip synchronized_sha256")
    if any(count == 0 for count in split_counts.values()):
        raise ValueError("SONIC dataset requires training and evaluation clips")
    identities = [(clip["source_id"], clip["sequence_id"]) for clip in clips]
    if len(set(identities)) != len(identities):
        raise ValueError("SONIC dataset clip identities must be unique")
    train_mass = sum(
        float(clip["frames"]) * float(clip["sampling_weight"])
        for clip in clips
        if clip["split"] == "train"
    )
    if not math.isfinite(train_mass) or train_mass <= 0.0:
        raise ValueError("SONIC dataset has invalid training sampling mass")
    evaluation_orders: list[int] = []
    for split in ("train", "evaluation"):
        for shard_index, descriptor in enumerate(index["shards"][split]):
            clips = [
                clip
                for clip in index["clips"]
                if clip["split"] == split and clip["shard_index"] == shard_index
            ]
            if len(clips) != descriptor["clip_count"]:
                raise ValueError(
                    f"SONIC {split} shard {shard_index} clip count has drifted"
                )
            offset = 0
            shard_mass = 0.0
            for clip in clips:
                frames = clip.get("frames")
                clip_offset = clip.get("offset")
                weight = clip.get("sampling_weight")
                if clip_offset != offset:
                    raise ValueError(
                        f"SONIC {split} shard {shard_index} clip bounds are invalid"
                    )
                offset += frames
                shard_mass += frames * float(weight)
                if split == "train":
                    expected_numerator = frames * float(weight)
                    numerator = clip["sampling_probability_numerator"]
                    denominator = clip["sampling_probability_denominator"]
                    if (
                        isinstance(numerator, bool)
                        or not isinstance(numerator, (int, float))
                        or isinstance(denominator, bool)
                        or not isinstance(denominator, (int, float))
                        or not math.isclose(
                            float(numerator),
                            expected_numerator,
                            rel_tol=1.0e-12,
                            abs_tol=0.0,
                        )
                        or not math.isclose(
                            float(denominator),
                            train_mass,
                            rel_tol=1.0e-12,
                            abs_tol=0.0,
                        )
                    ):
                        raise ValueError("SONIC training sampling metadata has drifted")
                else:
                    order = clip.get("evaluation_order")
                    if (
                        isinstance(order, bool)
                        or not isinstance(order, int)
                        or order < 0
                    ):
                        raise ValueError("SONIC evaluation order is invalid")
                    evaluation_orders.append(order)
            if offset != descriptor["data_frames"] or offset > sharding.frame_capacity:
                raise ValueError(
                    f"SONIC {split} shard {shard_index} frame count has drifted"
                )
            if not math.isclose(
                shard_mass,
                float(descriptor["sampling_mass"]),
                rel_tol=1.0e-12,
                abs_tol=0.0,
            ):
                raise ValueError(
                    f"SONIC {split} shard {shard_index} sampling mass has drifted"
                )
    if sorted(evaluation_orders) != list(range(len(evaluation_orders))):
        raise ValueError("SONIC evaluation ordering is not contiguous")


def _shard_path(index_path: Path, descriptor: dict[str, Any]) -> Path:
    root = index_path.parent.resolve()
    path = (root / descriptor["file"]).resolve()
    if path.parent.parent.parent != root:
        raise ValueError("SONIC shard path escapes the dataset root")
    return path


def _validate_shard_file(
    descriptor: dict[str, Any], path: Path, *, split: str, shard_index: int
) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"SONIC {split} shard {shard_index} file not found: {path}"
        )
    actual = "sha256:" + _file_sha256(path)
    if actual != descriptor["sha256"]:
        raise ValueError(
            f"SONIC {split} shard {shard_index} file differs from its index digest"
        )


def _validate_loaded_shard_arrays(
    arrays: dict[str, np.ndarray],
    *,
    frame_capacity: int,
    data_frames: int,
    body_count: int,
    split: str,
    shard_index: int,
) -> None:
    expected_shapes = {
        "qpos": (frame_capacity, 32),
        "qvel": (frame_capacity, 31),
        "body_pos": (frame_capacity, body_count, 3),
        "body_quat_wxyz": (frame_capacity, body_count, 4),
        "body_lin_vel": (frame_capacity, body_count, 3),
        "body_ang_vel": (frame_capacity, body_count, 3),
        "foot_contacts": (frame_capacity, 2),
        "smpl_joint_pos": (frame_capacity, len(SONIC_SMPL_JOINT_NAMES), 3),
        "smpl_root_quat_wxyz": (frame_capacity, 4),
        "smpl_root_translation": (frame_capacity, 3),
        "g1_joint_pos": (frame_capacity, len(G1_ISAACLAB_JOINT_NAMES)),
    }
    for name, expected_shape in expected_shapes.items():
        value = arrays[name]
        if value.dtype != np.float32 or value.shape != expected_shape:
            raise ValueError(
                f"SONIC {split} shard {shard_index} {name} must be float32 with "
                f"shape {expected_shape}; got {value.dtype} {value.shape}"
            )
        if not np.all(np.isfinite(value[:data_frames])):
            raise ValueError(
                f"SONIC {split} shard {shard_index} {name} contains non-finite data"
            )
    for name, quaternions in (
        ("qpos", arrays["qpos"][:data_frames, 3:7]),
        ("body_quat_wxyz", arrays["body_quat_wxyz"][:data_frames]),
        (
            "smpl_root_quat_wxyz",
            arrays["smpl_root_quat_wxyz"][:data_frames],
        ),
    ):
        if not np.allclose(
            np.linalg.norm(quaternions, axis=-1), 1.0, atol=1.0e-4, rtol=0.0
        ):
            raise ValueError(
                f"SONIC {split} shard {shard_index} {name} has non-unit quaternions"
            )
    contacts = arrays["foot_contacts"][:data_frames]
    if np.any((contacts < 0.0) | (contacts > 1.0)):
        raise ValueError(
            f"SONIC {split} shard {shard_index} foot contacts are outside [0, 1]"
        )


def build_padded_sonic_sampling_bins(
    lengths: np.ndarray,
    *,
    bin_size: int,
    frame_capacity: int,
    clip_capacity: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lengths = np.asarray(lengths, dtype=np.int32)
    if bin_size <= 0:
        raise ValueError("SONIC sampling bin size must be positive")
    if lengths.ndim != 1 or not 0 < len(lengths) <= clip_capacity:
        raise ValueError("SONIC shard clip count is incompatible with its capacity")
    if np.any(lengths <= SONIC_REQUIRED_FUTURE_HORIZON):
        raise ValueError("SONIC shard clips do not cover the future-frame horizon")
    if int(lengths.sum(dtype=np.int64)) > frame_capacity:
        raise ValueError("SONIC shard frame count exceeds its capacity")
    clip_indices: list[int] = []
    starts: list[int] = []
    widths: list[int] = []
    clip_bin_offsets = [0]
    for clip_index, length in enumerate(lengths.tolist()):
        valid_frames = int(length) - SONIC_REQUIRED_FUTURE_HORIZON
        for start in range(0, valid_frames, bin_size):
            clip_indices.append(clip_index)
            starts.append(start)
            widths.append(min(bin_size, valid_frames - start))
        clip_bin_offsets.append(len(starts))
    clip_bin_offsets.extend([len(starts)] * (clip_capacity + 1 - len(clip_bin_offsets)))
    bin_capacity = frame_capacity // bin_size + clip_capacity
    if len(starts) > bin_capacity:
        raise RuntimeError("SONIC sampling-bin capacity calculation is inconsistent")
    padded_clips = np.zeros((bin_capacity,), dtype=np.int32)
    padded_starts = np.zeros((bin_capacity,), dtype=np.int32)
    padded_widths = np.ones((bin_capacity,), dtype=np.int32)
    valid = np.zeros((bin_capacity,), dtype=bool)
    count = len(starts)
    padded_clips[:count] = clip_indices
    padded_starts[:count] = starts
    padded_widths[:count] = widths
    valid[:count] = True
    return (
        padded_clips,
        padded_starts,
        padded_widths,
        valid,
        np.asarray(clip_bin_offsets, dtype=np.int32),
    )


def _write_runtime_clip(
    output_path: Path,
    *,
    policy_fps: float,
    joint_names: tuple[str, ...],
    body_names: tuple[str, ...],
    source_sha256: str,
    arrays: dict[str, np.ndarray],
) -> Path:
    if policy_fps != SONIC_POLICY_FPS:
        raise ValueError("SONIC runtime clips must use 50 Hz policy frames")
    _require_sha256(source_sha256, "source_sha256")
    if set(arrays) != set(_PACKED_ARRAY_NAMES[:7]):
        raise ValueError(
            "SONIC runtime clip arrays differ from the native-motion schema"
        )
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}-", suffix=".npz", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        np.savez_compressed(
            temporary_path,
            format_version=np.asarray(1, dtype=np.int32),
            fps=np.asarray(policy_fps, dtype=np.float32),
            joint_names=np.asarray(joint_names),
            body_names=np.asarray(body_names),
            source_sha256=np.asarray(source_sha256.removeprefix("sha256:")),
            **arrays,
        )
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path


def _require_sha256(value: str, name: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    normalized = value.removeprefix("sha256:")
    if len(normalized) != 64 or any(
        char not in "0123456789abcdef" for char in normalized
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _write_json_once(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable dataset file: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_conversion_report(path: Path, decisions: list[ClipDecision]) -> None:
    document = {
        "accepted": sum(item.status == "accepted" for item in decisions),
        "quarantined": sum(item.status == "quarantined" for item in decisions),
        "rejected": sum(item.status == "rejected" for item in decisions),
        "clips": [asdict(item) for item in decisions],
    }
    _write_json_once(path, document)
