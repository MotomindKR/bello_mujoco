"""Pinned GMR boundary for live SMPL-X-to-Bello reference retargeting."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import redirect_stdout
from dataclasses import dataclass
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
import time
import types
from typing import Any, Protocol

import mujoco
import numpy as np

from bello_mujoco.hardware.twist2 import (
    HardwareTwist2Environment,
    ReferenceDiscontinuity,
    ReferenceUnavailable,
    Twist2ReferenceFrame,
)
from bello_mujoco.repository import repository_root
from bello_mujoco.training.checkpoint import Twist2CheckpointMetadata


SMPLX_GMR_BODY_NAMES = (
    "pelvis",
    "spine3",
    "left_hip",
    "left_knee",
    "left_foot",
    "right_hip",
    "right_knee",
    "right_foot",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
)

SMPLX_BODY_JOINT_NAMES = (
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

SMPLX_BODY_JOINT_PARENTS = np.asarray(
    (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19),
    dtype=np.int32,
)

# SMPL-X's model frame has +y up. This proper rotation produces the same
# z-up, x-forward, y-left convention used by the offline AMASS pipeline.
_SMPLX_MODEL_TO_WORLD = np.asarray(
    ((0.0, 0.0, -1.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    dtype=np.float64,
)
_SMPLX_MODEL_TO_WORLD_WXYZ = np.asarray((0.5, 0.5, -0.5, -0.5))

# Repository-owned, hand-authored relaxed stand in SMPL-X body-pose order.
# This is deliberately independent of licensed motion datasets.
_SMPLX_RELAXED_STANDING_BODY_POSE = np.asarray(
    (
        (-0.255, 0.04, 0.00),  # left_hip
        (-0.255, -0.04, 0.00),  # right_hip
        (0.06, 0.00, 0.00),  # spine1
        (0.71, -0.04, 0.00),  # left_knee
        (0.71, 0.04, 0.00),  # right_knee
        (0.04, 0.00, 0.00),  # spine2
        (-0.435, 0.08, 0.00),  # left_ankle
        (-0.435, -0.08, 0.00),  # right_ankle
        (0.02, 0.00, 0.00),  # spine3
        (0.00, 0.00, 0.00),  # left_foot
        (0.00, 0.00, 0.00),  # right_foot
        (0.00, 0.00, 0.00),  # neck
        (0.02, 0.06, -0.30),  # left_collar
        (0.02, -0.06, 0.30),  # right_collar
        (0.00, 0.00, 0.00),  # head
        (0.10, -0.18, -0.95),  # left_shoulder
        (0.10, 0.18, 0.95),  # right_shoulder
        (0.12, -0.30, 0.08),  # left_elbow
        (0.12, 0.30, -0.08),  # right_elbow
        (-0.10, -0.06, -0.10),  # left_wrist
        (-0.10, 0.06, 0.10),  # right_wrist
    ),
    dtype=np.float64,
)


@dataclass(frozen=True)
class SmplxTargetFrame:
    """Global SMPL-X joint transforms in world coordinates and wxyz order."""

    transforms: Mapping[str, tuple[np.ndarray, np.ndarray]]
    sequence: int
    source_timestamp_ns: int
    source_age_ms: int
    received_at: float
    foot_contacts: np.ndarray | None = None


class SmplxTargetStream(Protocol):
    """Blocking source boundary implemented by the corrected gRPC adapter."""

    def standing_sample(self) -> SmplxTargetFrame: ...

    def wait_next(
        self,
        *,
        after_sequence: int | None,
        timeout_seconds: float,
    ) -> SmplxTargetFrame | None: ...


def load_smplx_standing_sample(model_path: Path) -> SmplxTargetFrame:
    """Pose the supplied neutral SMPL-X skeleton in a relaxed normal stand."""

    path = model_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"neutral SMPL-X model does not exist: {path}")
    try:
        with np.load(path, allow_pickle=False) as body:
            template = np.asarray(body["v_template"], dtype=np.float64)
            regressor = np.asarray(body["J_regressor"], dtype=np.float64)
            parents = np.asarray(body["kintree_table"][0], dtype=np.int64)
    except (KeyError, OSError, ValueError) as error:
        raise ValueError(f"invalid neutral SMPL-X model {path}: {error}") from error
    if template.ndim != 2 or template.shape[1] != 3:
        raise ValueError("neutral SMPL-X v_template must have shape (vertices, 3)")
    joint_count = len(SMPLX_BODY_JOINT_NAMES)
    if regressor.ndim != 2 or regressor.shape[0] < joint_count:
        raise ValueError("neutral SMPL-X J_regressor has too few body joints")
    if regressor.shape[1] != template.shape[0] or parents.size < joint_count:
        raise ValueError("neutral SMPL-X model arrays have incompatible dimensions")
    body_parents = parents[:joint_count].astype(np.int32, copy=True)
    body_parents[0] = -1
    if not np.array_equal(body_parents, SMPLX_BODY_JOINT_PARENTS):
        raise ValueError("neutral SMPL-X model has an unexpected body hierarchy")
    rest_positions = regressor[:joint_count] @ template
    local_rotvecs = np.concatenate(
        (np.zeros((1, 3), dtype=np.float64), _SMPLX_RELAXED_STANDING_BODY_POSE),
        axis=0,
    )
    positions = np.empty((joint_count, 3), dtype=np.float64)
    rotations = np.empty((joint_count, 3, 3), dtype=np.float64)
    quaternions = np.empty((joint_count, 4), dtype=np.float64)
    for index, parent in enumerate(body_parents):
        local_rotation = _rotation_matrix_from_rotvec(local_rotvecs[index])
        local_quaternion = _quaternion_from_rotvec(local_rotvecs[index])
        if parent < 0:
            positions[index] = 0.0
            rotations[index] = local_rotation
            quaternions[index] = local_quaternion
            continue
        positions[index] = positions[parent] + rotations[parent] @ (
            rest_positions[index] - rest_positions[parent]
        )
        rotations[index] = rotations[parent] @ local_rotation
        quaternions[index] = _quaternion_multiply(quaternions[parent], local_quaternion)
    positions = positions @ _SMPLX_MODEL_TO_WORLD.T
    quaternions = np.stack(
        [
            _quaternion_multiply(_SMPLX_MODEL_TO_WORLD_WXYZ, quaternion)
            for quaternion in quaternions
        ]
    )
    foot_indices = tuple(
        SMPLX_BODY_JOINT_NAMES.index(name) for name in ("left_foot", "right_foot")
    )
    positions[:, 2] -= float(np.min(positions[foot_indices, 2]))
    if not np.all(np.isfinite(positions)):
        raise ValueError("neutral SMPL-X model produced non-finite body joints")
    position_lookup = dict(zip(SMPLX_BODY_JOINT_NAMES, positions, strict=True))
    quaternion_lookup = dict(zip(SMPLX_BODY_JOINT_NAMES, quaternions, strict=True))
    return SmplxTargetFrame(
        transforms={
            name: (
                position_lookup[name].copy(),
                quaternion_lookup[name].copy(),
            )
            for name in SMPLX_GMR_BODY_NAMES
        },
        sequence=0,
        source_timestamp_ns=0,
        source_age_ms=0,
        received_at=time.monotonic(),
    )


def _rotation_matrix_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle < 1.0e-12:
        return np.eye(3, dtype=np.float64)
    axis = vector / angle
    x, y, z = axis
    skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _quaternion_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle < 1.0e-12:
        return np.asarray((1.0, 0.0, 0.0, 0.0))
    scale = math.sin(0.5 * angle) / angle
    return np.concatenate((np.asarray((math.cos(0.5 * angle),)), scale * vector))


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.asarray(left, dtype=np.float64)
    rw, rx, ry, rz = np.asarray(right, dtype=np.float64)
    result = np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )
    return result / np.linalg.norm(result)


@dataclass(frozen=True)
class GmrLiveRetargetConfig:
    """Real-time settings layered on the pinned Bello GMR solver contract."""

    human_height_m: float = 1.66
    settling_rounds: int = 10
    convergence_tolerance: float = 1.0e-4
    max_source_interval_seconds: float = 0.2
    contact_max_speed_mps: float = 0.2
    contact_height_margin_m: float = 0.03

    def __post_init__(self) -> None:
        positive = {
            "human_height_m": self.human_height_m,
            "convergence_tolerance": self.convergence_tolerance,
            "max_source_interval_seconds": self.max_source_interval_seconds,
            "contact_max_speed_mps": self.contact_max_speed_mps,
            "contact_height_margin_m": self.contact_height_margin_m,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.settling_rounds < 1:
            raise ValueError("GMR settling_rounds must be positive")


@dataclass(frozen=True)
class _GmrSolverSettings:
    passes_per_frame: int
    max_joint_velocity_rps: float
    max_joint_acceleration_rps2: float
    knee_pole_reference_flexion_radians: float
    knee_target_minimum_flexion_radians: float
    knee_pole_sample_minimum_flexion_radians: float
    knee_pole_sample_minimum_distance_meters: float
    knee_pole_smoothing_frames: int
    ground_clearance_geoms: tuple[str, str]


def _load_gmr_solver_settings(path: Path) -> _GmrSolverSettings:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        solver = document["offline_solver"]
        ground_geoms = tuple(document["ground_clearance_geoms"])
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ValueError(
            f"invalid pinned Bello GMR configuration {path}: {error}"
        ) from error
    expected = {
        "passes_per_frame",
        "max_joint_velocity_radians_per_second",
        "max_joint_acceleration_radians_per_second_squared",
        "knee_pole_reference_flexion_radians",
        "knee_target_minimum_flexion_radians",
        "knee_pole_sample_minimum_flexion_radians",
        "knee_pole_sample_minimum_distance_meters",
        "knee_pole_smoothing_frames",
    }
    if not isinstance(solver, dict) or set(solver) != expected:
        raise ValueError("pinned Bello GMR offline_solver fields are incompatible")
    if len(ground_geoms) != 2 or not all(
        isinstance(name, str) and name for name in ground_geoms
    ):
        raise ValueError("pinned Bello GMR must configure two foot clearance geoms")
    settings = _GmrSolverSettings(
        passes_per_frame=solver["passes_per_frame"],
        max_joint_velocity_rps=solver["max_joint_velocity_radians_per_second"],
        max_joint_acceleration_rps2=solver[
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
        ground_clearance_geoms=ground_geoms,
    )
    if not isinstance(settings.passes_per_frame, int) or settings.passes_per_frame < 1:
        raise ValueError("pinned Bello GMR passes_per_frame must be positive")
    numeric = (
        settings.max_joint_velocity_rps,
        settings.max_joint_acceleration_rps2,
        settings.knee_pole_reference_flexion_radians,
        settings.knee_target_minimum_flexion_radians,
        settings.knee_pole_sample_minimum_flexion_radians,
        settings.knee_pole_sample_minimum_distance_meters,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in numeric):
        raise ValueError("pinned Bello GMR solver values must be positive and finite")
    if (
        not isinstance(settings.knee_pole_smoothing_frames, int)
        or settings.knee_pole_smoothing_frames < 1
        or settings.knee_pole_smoothing_frames % 2 == 0
    ):
        raise ValueError(
            "pinned Bello GMR knee smoothing window must be positive and odd"
        )
    return settings


class _CausalKneeTargetConditioner:
    """Causal form of the offline positive-flex knee target conditioning."""

    def __init__(self, retargeter: Any, settings: _GmrSolverSettings) -> None:
        self.retargeter = retargeter
        self.settings = settings
        self._segment_lengths = _robot_leg_segment_lengths(retargeter)
        self._reference_axes: dict[str, np.ndarray] = {}
        self._local_axes: dict[str, np.ndarray] = {}

    def reset(self, axes: Mapping[str, np.ndarray] | None = None) -> None:
        copied = (
            {}
            if axes is None
            else {side: np.asarray(axis).copy() for side, axis in axes.items()}
        )
        self._reference_axes = {side: axis.copy() for side, axis in copied.items()}
        self._local_axes = copied

    def restore(self, axes: Mapping[str, np.ndarray]) -> None:
        self._local_axes = {
            side: np.asarray(axis).copy() for side, axis in axes.items()
        }

    def axes(self) -> dict[str, np.ndarray]:
        return {side: axis.copy() for side, axis in self._local_axes.items()}

    def condition(
        self,
        targets: Mapping[str, tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        source = {
            name: [np.asarray(position).copy(), np.asarray(quaternion).copy()]
            for name, (position, quaternion) in targets.items()
        }
        scaled = self.retargeter.scale_human_data(
            _copy_mutable_targets(source),
            self.retargeter.human_root_name,
            self.retargeter.human_scale_table,
        )
        scaled = self.retargeter.offset_human_data(
            scaled,
            self.retargeter.pos_offsets1,
            self.retargeter.rot_offsets1,
        )
        scaled = self.retargeter.align_human_data_to_ground(scaled)
        scaled = self.retargeter.apply_ground_offset(scaled)
        root_position = source[self.retargeter.human_root_name][0]
        scaled_root_position = (
            self.retargeter.human_scale_table[self.retargeter.human_root_name]
            * root_position
        )
        for side in ("left", "right"):
            hip_name = f"{side}_hip"
            knee_name = f"{side}_knee"
            foot_name = f"{side}_foot"
            hip_position, hip_quaternion = source[hip_name]
            knee_position, knee_quaternion = source[knee_name]
            foot_position = source[foot_name][0]
            inferred_reference_axis = _positive_knee_reference_axis(
                hip_position,
                hip_quaternion,
                knee_position,
                knee_quaternion,
                foot_position,
                self.settings.knee_pole_reference_flexion_radians,
            )
            reference_axis = self._reference_axes.setdefault(
                side, inferred_reference_axis
            )
            observed_pole = _knee_pole_vector(
                hip_position, knee_position, foot_position
            )
            observed_distance = float(np.linalg.norm(observed_pole))
            local_knee_quaternion = _quaternion_multiply(
                _quaternion_conjugate(hip_quaternion), knee_quaternion
            )
            flexion = float(_rotation_vector(local_knee_quaternion)[0])
            candidate_axis = reference_axis
            if (
                observed_distance
                >= self.settings.knee_pole_sample_minimum_distance_meters
                and flexion >= self.settings.knee_pole_sample_minimum_flexion_radians
            ):
                observed_axis = _quaternion_rotate(
                    _quaternion_conjugate(hip_quaternion),
                    observed_pole / observed_distance,
                )
                if float(np.dot(observed_axis, reference_axis)) > 0.0:
                    candidate_axis = observed_axis
            previous_axis = self._local_axes.get(side, reference_axis)
            if float(np.dot(candidate_axis, previous_axis)) < 0.0:
                candidate_axis = -candidate_axis
            alpha = 2.0 / (self.settings.knee_pole_smoothing_frames + 1.0)
            local_axis = _normalized_vector(
                (1.0 - alpha) * previous_axis + alpha * candidate_axis,
                fallback=reference_axis,
            )
            self._local_axes[side] = local_axis
            pole_direction = _quaternion_rotate(hip_quaternion, local_axis)
            pole_direction = _normalized_vector(
                pole_direction,
                fallback=np.asarray((1.0, 0.0, 0.0)),
            )
            knee_target, foot_target = _reachable_leg_targets(
                np.asarray(scaled[hip_name][0]),
                np.asarray(scaled[foot_name][0]),
                pole_direction,
                *self._segment_lengths[side],
                self.settings.knee_target_minimum_flexion_radians,
            )
            for body_name, target_position in (
                (knee_name, knee_target),
                (foot_name, foot_target),
            ):
                offset_quaternion = self.retargeter.rot_offsets1[body_name].as_quat(
                    scalar_first=True
                )
                body_quaternion = _quaternion_multiply(
                    source[body_name][1], offset_quaternion
                )
                position_offset = _quaternion_rotate(
                    body_quaternion, self.retargeter.pos_offsets1[body_name]
                )
                body_scale = self.retargeter.human_scale_table[body_name]
                if not math.isfinite(body_scale) or body_scale <= 0.0:
                    raise ReferenceDiscontinuity(
                        f"invalid pinned GMR scale for {body_name}"
                    )
                source[body_name][0] = (
                    root_position
                    + (target_position - position_offset - scaled_root_position)
                    / body_scale
                )
        return {
            name: (np.asarray(position).copy(), np.asarray(quaternion).copy())
            for name, (position, quaternion) in source.items()
        }


def _copy_mutable_targets(
    targets: Mapping[str, list[np.ndarray]],
) -> dict[str, list[np.ndarray]]:
    return {
        name: [np.asarray(position).copy(), np.asarray(quaternion).copy()]
        for name, (position, quaternion) in targets.items()
    }


def _robot_leg_segment_lengths(retargeter: Any) -> dict[str, tuple[float, float]]:
    mujoco.mj_forward(retargeter.model, retargeter.configuration.data)
    positions = retargeter.configuration.data.xpos
    lengths = {}
    for side in ("left", "right"):
        hip = retargeter.model.body(f"{side}_hip_roll_link").id
        knee = retargeter.model.body(f"{side}_knee_link").id
        foot = retargeter.model.body(f"{side}_ankle_roll_link").id
        thigh = float(np.linalg.norm(positions[knee] - positions[hip]))
        shin = float(np.linalg.norm(positions[foot] - positions[knee]))
        if not np.all(np.isfinite((thigh, shin))) or min(thigh, shin) <= 0.0:
            raise ValueError(f"invalid Bello leg segment lengths for {side}")
        lengths[side] = (thigh, shin)
    return lengths


def _positive_knee_reference_axis(
    hip_position: np.ndarray,
    hip_quaternion: np.ndarray,
    knee_position: np.ndarray,
    knee_quaternion: np.ndarray,
    foot_position: np.ndarray,
    reference_flexion: float,
) -> np.ndarray:
    inverse_hip = _quaternion_conjugate(hip_quaternion)
    inverse_knee = _quaternion_conjugate(knee_quaternion)
    thigh_offset = _quaternion_rotate(
        inverse_hip, np.asarray(knee_position) - np.asarray(hip_position)
    )
    shin_offset = _quaternion_rotate(
        inverse_knee, np.asarray(foot_position) - np.asarray(knee_position)
    )
    reference_foot = thigh_offset + _quaternion_rotate(
        _quaternion_from_rotvec(np.asarray((reference_flexion, 0.0, 0.0))),
        shin_offset,
    )
    return _normalized_knee_pole(np.zeros(3), thigh_offset, reference_foot)


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
        raise ReferenceUnavailable("SMPL-X hip and foot targets overlap")
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
    return (
        hip_position + along * chord_axis + radial * pole,
        hip_position + effective_distance * chord_axis,
    )


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
        raise ReferenceUnavailable("positive-flex knee reference has no bend plane")
    return pole / norm


def _normalized_vector(vector: np.ndarray, *, fallback: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm < 1.0e-8:
        vector = np.asarray(fallback, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
    return vector / norm


def _quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternion, dtype=np.float64).copy()
    result[1:] *= -1.0
    return result / np.linalg.norm(result)


def _quaternion_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    cross = np.cross(quaternion[1:], vector)
    return vector + 2.0 * (quaternion[0] * cross + np.cross(quaternion[1:], cross))


def _rotation_vector(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion = quaternion / np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    vector_norm = float(np.linalg.norm(quaternion[1:]))
    if vector_norm < 1.0e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(vector_norm, float(quaternion[0]))
    return angle * quaternion[1:] / vector_norm


class GmrBelloRetargeter:
    """Stateful, source-rate GMR retargeter producing TWIST2 command frames."""

    def __init__(
        self,
        env: HardwareTwist2Environment,
        metadata: Twist2CheckpointMetadata,
        config: GmrLiveRetargetConfig = GmrLiveRetargetConfig(),
        validate_reference: Callable[[Twist2ReferenceFrame], None] | None = None,
    ) -> None:
        gmr_root = require_pinned_gmr_revision(metadata.dataset.gmr_revision)
        require_robot_fingerprint(metadata.robot_fingerprint)
        solver_path = (
            gmr_root
            / "general_motion_retargeting"
            / "ik_configs"
            / "smplx_to_bello.json"
        )
        solver_settings = _load_gmr_solver_settings(solver_path)
        retargeter_type = import_pinned_gmr_retargeter()
        with redirect_stdout(io.StringIO()):
            retargeter = retargeter_type(
                src_human="smplx",
                tgt_robot="bello",
                actual_human_height=config.human_height_m,
                solver="daqp",
                damping=5.0e-1,
                verbose=False,
                use_velocity_limit=True,
                robot_xml_path=env.model.xml_path,
            )
        retargeter.max_iter = 0
        self.env = env
        self.metadata = metadata
        self.config = config
        self.validate_reference = validate_reference
        self.solver_settings = solver_settings
        self.retargeter = retargeter
        self.model = retargeter.model
        _validate_model_identity(self.model, env.model.mj_model)
        self._joint_ranges = _scalar_joint_ranges(self.model)
        self._knee_conditioner = _CausalKneeTargetConditioner(
            retargeter, solver_settings
        )
        self._standing_knee_axes: dict[str, np.ndarray] | None = None
        self._foot_geom_ids = tuple(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in solver_settings.ground_clearance_geoms
        )
        if any(geom_id < 0 for geom_id in self._foot_geom_ids):
            raise ValueError("pinned Bello GMR foot clearance geom is missing")
        self._ground_data = mujoco.MjData(self.model)
        self._previous_qpos: np.ndarray | None = None
        self._previous_joint_velocity: np.ndarray | None = None
        self._previous_timestamp_ns: int | None = None
        self._previous_source_feet: np.ndarray | None = None
        self._source_floor_height: float | None = None
        self._live_ground_offset: float | None = None
        self._last_sequence: int | None = None
        self._cached: Twist2ReferenceFrame | None = None
        self._standing_qpos: np.ndarray | None = None

    def standing(self, frame: SmplxTargetFrame) -> Twist2ReferenceFrame:
        """Settle the canonical SMPL-X standing sample through pinned GMR."""

        targets = validate_smplx_target_frame(frame, live=False)
        self._knee_conditioner.reset()
        targets = self._knee_conditioner.condition(targets)
        previous = self._solve(targets)
        for _ in range(self.config.settling_rounds - 1):
            current = self._solve(targets)
            if np.max(np.abs(current - previous)) < self.config.convergence_tolerance:
                previous = current
                break
            previous = current
        previous = self._normalize_standing(previous)
        previous = self._ground_standing(previous)
        self.retargeter.configuration.update(previous)
        self._standing_qpos = previous.copy()
        self._standing_knee_axes = self._knee_conditioner.axes()
        self._reset_live_state()
        return Twist2ReferenceFrame(
            qpos=previous,
            qvel=np.zeros(self.model.nv, dtype=np.float64),
            sequence=frame.sequence,
            source_age_ms=None,
            received_at=frame.received_at,
        )

    def retarget(self, frame: SmplxTargetFrame) -> Twist2ReferenceFrame:
        """Retarget one new live source frame, caching repeated source frames."""

        source_targets = validate_smplx_target_frame(frame, live=True)
        if self._last_sequence is not None:
            if frame.sequence == self._last_sequence:
                assert self._cached is not None
                return self._cached
            if frame.sequence < self._last_sequence:
                raise ReferenceDiscontinuity(
                    "SMPL-X source sequence moved backwards: "
                    f"{frame.sequence} < {self._last_sequence}"
                )
        if (
            self._previous_timestamp_ns is not None
            and frame.source_timestamp_ns <= self._previous_timestamp_ns
        ):
            raise ReferenceDiscontinuity("SMPL-X source timestamp did not advance")
        dt = None
        if self._previous_timestamp_ns is not None:
            dt = (frame.source_timestamp_ns - self._previous_timestamp_ns) / 1.0e9
            if not math.isfinite(dt) or not (
                0.0 < dt <= self.config.max_source_interval_seconds
            ):
                raise ReferenceDiscontinuity(
                    f"SMPL-X source interval is invalid: {dt:.6f} seconds"
                )
        solver_qpos = self.retargeter.configuration.data.qpos.copy()
        knee_axes = self._knee_conditioner.axes()
        previous_source_feet = (
            None
            if self._previous_source_feet is None
            else self._previous_source_feet.copy()
        )
        source_floor_height = self._source_floor_height
        live_ground_offset = self._live_ground_offset
        try:
            targets = self._knee_conditioner.condition(source_targets)
            candidate = self._solve(targets)
            candidate = self._ground_live(
                candidate,
                source_targets,
                dt=dt,
                source_contacts=frame.foot_contacts,
            )
            if self._previous_qpos is None:
                qpos = candidate
                joint_velocity = np.zeros(self.model.nq - 7, dtype=np.float64)
                qvel = np.zeros(self.model.nv, dtype=np.float64)
            else:
                assert self._previous_joint_velocity is not None
                assert dt is not None
                qpos, joint_velocity = _limit_scalar_joint_candidate(
                    candidate,
                    self._previous_qpos,
                    self._previous_joint_velocity,
                    dt=dt,
                    joint_ranges=self._joint_ranges,
                    maximum_velocity=self.solver_settings.max_joint_velocity_rps,
                    maximum_acceleration=(
                        self.solver_settings.max_joint_acceleration_rps2
                    ),
                )
                qvel = np.empty(self.model.nv, dtype=np.float64)
                mujoco.mj_differentiatePos(
                    self.model, qvel, dt, self._previous_qpos, qpos
                )
            if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(qvel)):
                raise ReferenceDiscontinuity(
                    "GMR produced a non-finite Bello reference"
                )
            result = Twist2ReferenceFrame(
                qpos=qpos,
                qvel=qvel,
                sequence=frame.sequence,
                source_age_ms=frame.source_age_ms,
                received_at=frame.received_at,
            )
            if self.validate_reference is not None:
                self.validate_reference(result)
        except BaseException:
            self.retargeter.configuration.update(solver_qpos)
            self._knee_conditioner.restore(knee_axes)
            self._previous_source_feet = previous_source_feet
            self._source_floor_height = source_floor_height
            self._live_ground_offset = live_ground_offset
            raise
        self.retargeter.configuration.update(qpos)
        self._previous_qpos = qpos
        self._previous_joint_velocity = joint_velocity
        self._previous_timestamp_ns = frame.source_timestamp_ns
        self._last_sequence = frame.sequence
        self._cached = result
        return result

    def reset_live(self) -> None:
        """Re-anchor GMR after a tracking discontinuity."""

        if self._standing_qpos is None:
            raise RuntimeError("GMR standing reference has not been initialized")
        self.retargeter.configuration.update(self._standing_qpos)
        self._reset_live_state()

    def _solve(self, targets: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        qpos = None
        for _ in range(self.solver_settings.passes_per_frame):
            qpos = self.retargeter.retarget(_copy_targets(targets))
        assert qpos is not None
        result = np.asarray(qpos, dtype=np.float64)
        if result.shape != (self.model.nq,):
            raise ReferenceDiscontinuity(
                f"GMR returned qpos shape {result.shape}; expected {(self.model.nq,)}"
            )
        if not np.all(np.isfinite(result)):
            raise ReferenceDiscontinuity("GMR returned non-finite Bello qpos")
        return result

    def _ground_standing(self, qpos: np.ndarray) -> np.ndarray:
        sole_z = self._foot_sole_z(qpos)
        grounded = np.asarray(qpos, dtype=np.float64).copy()
        grounded[self.env.model.root_qpos_adr + 2] -= float(np.min(sole_z))
        return grounded

    def _normalize_standing(self, qpos: np.ndarray) -> np.ndarray:
        """Use GMR arms with the policy's canonical, symmetric Bello leg stance."""

        normalized = np.asarray(qpos, dtype=np.float64).copy()
        leg_joint_names = (
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "left_hip_yaw_joint",
        )
        for joint_name in leg_joint_names:
            address = int(self.model.jnt_qposadr[self.model.joint(joint_name).id])
            normalized[address] = self.env.model.default_qpos[address]
        root = self.env.model.root_qpos_adr
        w, x, y, z = normalized[root + 3 : root + 7]
        yaw = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
        normalized[root + 3 : root + 7] = (
            math.cos(0.5 * yaw),
            0.0,
            0.0,
            math.sin(0.5 * yaw),
        )
        return normalized

    def _ground_live(
        self,
        qpos: np.ndarray,
        targets: Mapping[str, tuple[np.ndarray, np.ndarray]],
        *,
        dt: float | None,
        source_contacts: np.ndarray | None,
    ) -> np.ndarray:
        source_feet = np.stack(
            [np.asarray(targets[f"{side}_foot"][0]) for side in ("left", "right")]
        )
        if self._source_floor_height is None:
            self._source_floor_height = float(np.min(source_feet[:, 2]))
        if self._previous_source_feet is None or dt is None:
            source_speed = np.zeros(2, dtype=np.float64)
        else:
            source_speed = (
                np.linalg.norm(source_feet - self._previous_source_feet, axis=1) / dt
            )
        if source_contacts is None:
            contacts = (source_speed < self.config.contact_max_speed_mps) & (
                source_feet[:, 2]
                <= self._source_floor_height + self.config.contact_height_margin_m
            )
        else:
            contacts = np.asarray(source_contacts, dtype=bool)
            if contacts.shape != (2,):
                raise ReferenceDiscontinuity(
                    "SMPL-X source foot contacts must have shape (2,)"
                )
        self._previous_source_feet = source_feet
        sole_z = self._foot_sole_z(qpos)
        if np.any(contacts):
            self._live_ground_offset = float(np.min(sole_z[contacts]))
        elif self._live_ground_offset is None:
            self._live_ground_offset = float(np.min(sole_z))
        ground_offset = min(self._live_ground_offset, float(np.min(sole_z)))
        grounded = np.asarray(qpos, dtype=np.float64).copy()
        grounded[self.env.model.root_qpos_adr + 2] -= ground_offset
        return grounded

    def _foot_sole_z(self, qpos: np.ndarray) -> np.ndarray:
        self._ground_data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self._ground_data)
        heights = np.empty(2, dtype=np.float64)
        for output, geom_id in enumerate(self._foot_geom_ids):
            rotation = self._ground_data.geom_xmat[geom_id].reshape(3, 3)
            vertical_extent = np.abs(rotation[2]) @ self.model.geom_size[geom_id]
            heights[output] = self._ground_data.geom_xpos[geom_id, 2] - vertical_extent
        return heights

    def _reset_live_state(self) -> None:
        self._knee_conditioner.reset(self._standing_knee_axes)
        self._previous_qpos = None
        self._previous_joint_velocity = None
        self._previous_timestamp_ns = None
        self._previous_source_feet = None
        self._source_floor_height = None
        self._live_ground_offset = None
        self._last_sequence = None
        self._cached = None


class LiveGmrReferenceSource:
    """Run source-rate GMR in a worker while the policy reads latest-only data."""

    def __init__(
        self,
        stream: SmplxTargetStream,
        retargeter: GmrBelloRetargeter,
        *,
        poll_timeout_seconds: float = 0.1,
    ) -> None:
        if not math.isfinite(poll_timeout_seconds) or poll_timeout_seconds <= 0.0:
            raise ValueError("GMR worker poll timeout must be positive and finite")
        self.stream = stream
        self.retargeter = retargeter
        self.poll_timeout_seconds = poll_timeout_seconds
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._standing: Twist2ReferenceFrame | None = None
        self._latest: Twist2ReferenceFrame | None = None
        self._unavailable: ReferenceUnavailable | None = None
        self._terminal_error: BaseException | None = None

    def __enter__(self) -> LiveGmrReferenceSource:
        if self._thread is not None:
            raise RuntimeError("GMR reference source is already running")
        self._standing = self.retargeter.standing(self.stream.standing_sample())
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="bello-live-gmr",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *_exception: object) -> bool:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(2.0, 2.0 * self.poll_timeout_seconds))
            if thread.is_alive():
                raise RuntimeError("live GMR worker did not stop")
        self._thread = None
        return False

    def standing(self) -> Twist2ReferenceFrame:
        if self._standing is None:
            raise RuntimeError("live GMR source is not running")
        return self._standing

    def latest(self) -> Twist2ReferenceFrame:
        with self._condition:
            if self._terminal_error is not None:
                raise RuntimeError("live GMR worker failed") from self._terminal_error
            if self._latest is not None:
                return self._latest
            if self._unavailable is not None:
                raise self._unavailable
            raise ReferenceUnavailable("no live SMPL-X frame has been retargeted")

    def wait_until_ready(self, timeout_seconds: float) -> Twist2ReferenceFrame:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise ValueError("GMR ready timeout must be positive and finite")
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while self._latest is None and self._terminal_error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("timed out waiting for a live GMR reference")
                self._condition.wait(remaining)
            if self._terminal_error is not None:
                raise RuntimeError("live GMR worker failed") from self._terminal_error
            assert self._latest is not None
            return self._latest

    def _run(self) -> None:
        after_sequence = None
        while not self._stop.is_set():
            try:
                frame = self.stream.wait_next(
                    after_sequence=after_sequence,
                    timeout_seconds=self.poll_timeout_seconds,
                )
                if frame is None:
                    continue
                reference = self.retargeter.retarget(frame)
            except ReferenceDiscontinuity as error:
                try:
                    self.retargeter.reset_live()
                except BaseException as reset_error:
                    self._set_terminal(reset_error)
                    return
                after_sequence = None
                with self._condition:
                    self._latest = None
                    self._unavailable = error
                    self._condition.notify_all()
                continue
            except ReferenceUnavailable as error:
                with self._condition:
                    self._unavailable = error
                    self._condition.notify_all()
                continue
            except BaseException as error:
                self._set_terminal(error)
                return
            after_sequence = frame.sequence
            with self._condition:
                self._latest = reference
                self._unavailable = None
                self._condition.notify_all()

    def _set_terminal(self, error: BaseException) -> None:
        with self._condition:
            self._terminal_error = error
            self._condition.notify_all()


def validate_smplx_target_frame(
    frame: SmplxTargetFrame,
    *,
    live: bool,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    missing = sorted(set(SMPLX_GMR_BODY_NAMES) - set(frame.transforms))
    if missing:
        raise ReferenceUnavailable(
            "SMPL-X frame is missing GMR target joints: " + ", ".join(missing)
        )
    if frame.sequence < 0 or frame.source_age_ms < 0:
        raise ReferenceUnavailable("SMPL-X sequence and source age must be nonnegative")
    if live and frame.source_timestamp_ns <= 0:
        raise ReferenceUnavailable("live SMPL-X source timestamp must be positive")
    result = {}
    for name in SMPLX_GMR_BODY_NAMES:
        position, quaternion = frame.transforms[name]
        position = np.asarray(position, dtype=np.float64)
        quaternion = np.asarray(quaternion, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ReferenceUnavailable(
                f"SMPL-X joint {name!r} position must be a finite 3-vector"
            )
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            raise ReferenceUnavailable(
                f"SMPL-X joint {name!r} orientation must be a finite wxyz quaternion"
            )
        norm = float(np.linalg.norm(quaternion))
        if not 0.999 <= norm <= 1.001:
            raise ReferenceUnavailable(
                f"SMPL-X joint {name!r} quaternion norm is invalid: {norm}"
            )
        result[name] = (position.copy(), (quaternion / norm).copy())
    return result


def require_pinned_gmr_revision(expected: str) -> Path:
    root = repository_root()
    gmr_root = root / "deps" / "GMR"
    if len(expected) != 40 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise ValueError("checkpoint GMR revision must be 40 lowercase hex digits")
    gitlink = _git_output(root, "ls-tree", "HEAD", "--", "deps/GMR").split()
    if len(gitlink) < 3 or gitlink[1] != "commit":
        raise RuntimeError("deps/GMR is not a pinned Git submodule")
    pinned = gitlink[2]
    current = _git_output(gmr_root, "rev-parse", "HEAD")
    if pinned != expected or current != expected:
        raise ValueError(
            "TWIST2 checkpoint, repository gitlink, and checked-out GMR revision "
            f"differ: checkpoint={expected}, gitlink={pinned}, checkout={current}"
        )
    status = _git_output(
        gmr_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise RuntimeError("hardware GMR submodule must be clean")
    return gmr_root


def require_robot_fingerprint(expected: str) -> Path:
    path = (
        repository_root()
        / "deps"
        / "bello-stack"
        / "descriptions"
        / "bello"
        / "full_body"
        / "urdf"
        / "bello_full_body.urdf"
    )
    actual = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(
            "TWIST2 checkpoint robot fingerprint differs from the canonical "
            f"Bello description: checkpoint={expected}, current={actual}"
        )
    return path


def import_pinned_gmr_retargeter() -> type[Any]:
    """Import only GMR's solver module, avoiding its optional UI/torch surface."""

    package_name = "_bello_pinned_gmr"
    module_name = f"{package_name}.motion_retarget"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing.GeneralMotionRetargeting
    gmr_package = repository_root() / "deps" / "GMR" / "general_motion_retargeting"
    package = types.ModuleType(package_name)
    package.__path__ = [str(gmr_package)]
    package.__package__ = package_name
    sys.modules[package_name] = package
    try:
        spec = importlib.util.spec_from_file_location(
            module_name, gmr_package / "motion_retarget.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load pinned GMR motion_retarget module")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        sys.modules.pop(package_name, None)
        raise
    module_path = Path(module.__file__).resolve()
    try:
        module_path.relative_to(gmr_package.resolve())
    except ValueError as error:
        raise RuntimeError("GMR retargeter was not imported from deps/GMR") from error
    return module.GeneralMotionRetargeting


def _validate_model_identity(gmr_model: mujoco.MjModel, model: mujoco.MjModel) -> None:
    fields = ("nq", "nv", "nu", "njnt", "nbody")
    mismatched = [
        name for name in fields if getattr(gmr_model, name) != getattr(model, name)
    ]
    if mismatched:
        raise ValueError(
            "GMR and TWIST2 hardware models have different dimensions: "
            + ", ".join(mismatched)
        )
    for object_type, count in (
        (mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
        (mujoco.mjtObj.mjOBJ_BODY, model.nbody),
        (mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu),
    ):
        left = tuple(
            mujoco.mj_id2name(gmr_model, object_type, index) for index in range(count)
        )
        right = tuple(
            mujoco.mj_id2name(model, object_type, index) for index in range(count)
        )
        if left != right:
            raise ValueError("GMR and TWIST2 hardware model ordering differs")


def _scalar_joint_ranges(model: mujoco.MjModel) -> np.ndarray:
    joint_ids = sorted(
        (
            joint_id
            for joint_id in range(model.njnt)
            if model.jnt_type[joint_id]
            in {mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE}
        ),
        key=lambda joint_id: int(model.jnt_qposadr[joint_id]),
    )
    if len(joint_ids) != model.nq - 7:
        raise ValueError("Bello GMR model must have a free root and scalar joints")
    return np.asarray(model.jnt_range[joint_ids], dtype=np.float64)


def _limit_scalar_joint_candidate(
    candidate: np.ndarray,
    previous_qpos: np.ndarray,
    previous_joint_velocity: np.ndarray,
    *,
    dt: float,
    joint_ranges: np.ndarray,
    maximum_velocity: float,
    maximum_acceleration: float,
) -> tuple[np.ndarray, np.ndarray]:
    candidate = np.asarray(candidate, dtype=np.float64)
    previous_qpos = np.asarray(previous_qpos, dtype=np.float64)
    joint_count = candidate.shape[0] - 7
    if candidate.shape != previous_qpos.shape or candidate.ndim != 1:
        raise ReferenceUnavailable("GMR qpos candidates have incompatible shapes")
    if previous_joint_velocity.shape != (joint_count,) or joint_ranges.shape != (
        joint_count,
        2,
    ):
        raise ReferenceUnavailable("GMR temporal-limiter layout is incompatible")
    velocity_step = maximum_acceleration * dt
    desired_velocity = (candidate[7:] - previous_qpos[7:]) / dt
    lower_distance = np.maximum(previous_qpos[7:] - joint_ranges[:, 0], 0.0)
    upper_distance = np.maximum(joint_ranges[:, 1] - previous_qpos[7:], 0.0)
    lower_braking_velocity = velocity_step - np.sqrt(
        velocity_step**2 + 2.0 * maximum_acceleration * lower_distance
    )
    upper_braking_velocity = -velocity_step + np.sqrt(
        velocity_step**2 + 2.0 * maximum_acceleration * upper_distance
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
        raise ReferenceUnavailable("GMR joint limiter has no feasible solution")
    joint_velocity = np.clip(desired_velocity, lower_velocity, upper_velocity)
    limited = candidate.copy()
    limited[7:] = previous_qpos[7:] + joint_velocity * dt
    if np.any(limited[7:] < joint_ranges[:, 0] - 1.0e-10) or np.any(
        limited[7:] > joint_ranges[:, 1] + 1.0e-10
    ):
        raise ReferenceUnavailable("GMR output violates Bello joint limits")
    return limited, joint_velocity


def _copy_targets(
    targets: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        name: (position.copy(), quaternion.copy())
        for name, (position, quaternion) in targets.items()
    }


def _git_output(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"git {' '.join(arguments)} failed" + (f": {detail}" if detail else "")
        )
    return result.stdout.strip()
