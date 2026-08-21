"""Direct calibrated SMPL-X arm mapping for locomotion teleoperation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from typing import Any

import mujoco
import numpy as np

from bello_mujoco.hardware.smplx_retarget import SmplxTargetFrame
from bello_mujoco.hardware.twist2 import (
    ReferenceDiscontinuity,
    ReferenceUnavailable,
    Twist2ReferenceFrame,
)


SMPLX_ARM_TARGET_NAMES = (
    "spine3",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
)

_ARM_JOINT_SUFFIXES = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow_pitch",
    "elbow_yaw",
    "wrist_pitch",
)


@dataclass(frozen=True)
class _ArmCalibration:
    upper_frame_offset: np.ndarray
    wrist_frame_offset: np.ndarray
    standing_elbow_flexion: float


class DirectSmplxArmMapper:
    """Project calibrated SMPL-X arm frames directly onto Bello's six arm axes."""

    def __init__(
        self,
        env: Any,
        *,
        validate_reference: Callable[[Twist2ReferenceFrame], None] | None = None,
        max_source_interval_seconds: float = 0.2,
    ) -> None:
        if (
            not math.isfinite(max_source_interval_seconds)
            or max_source_interval_seconds <= 0.0
        ):
            raise ValueError("direct arm maximum source interval must be positive")
        self.env = env
        self.model = env.model.mj_model
        self.validate_reference = validate_reference
        self.max_source_interval_seconds = max_source_interval_seconds
        self._index_by_name = {
            name: index for index, name in enumerate(env.model.actuator_joint_names)
        }
        expected = {
            f"{side}_{suffix}_joint"
            for side in ("left", "right")
            for suffix in _ARM_JOINT_SUFFIXES
        }
        missing = sorted(expected - set(self._index_by_name))
        if missing:
            raise ValueError(
                "Bello direct arm mapping is missing: " + ", ".join(missing)
            )
        self._validate_robot_axes()
        self._default_joint_target = np.asarray(
            env.model.default_joint_target, dtype=np.float64
        )
        self._target_low = np.asarray(env.model.joint_target_low, dtype=np.float64)
        self._target_high = np.asarray(env.model.joint_target_high, dtype=np.float64)
        self._default_qpos = np.asarray(env.model.default_qpos, dtype=np.float64)
        self._qpos_indices = np.asarray(env.model.actuator_qpos_indices, dtype=np.int32)
        self._calibration: dict[str, _ArmCalibration] | None = None
        self._previous_qpos: np.ndarray | None = None
        self._previous_timestamp_ns: int | None = None
        self._last_sequence: int | None = None
        self._cached: Twist2ReferenceFrame | None = None

    def standing(self, frame: SmplxTargetFrame) -> Twist2ReferenceFrame:
        """Calibrate the source standing frames to Bello's canonical arm pose."""

        targets = _validate_smplx_arm_frame(frame, live=False)
        data = mujoco.MjData(self.model)
        data.qpos[:] = self._default_qpos
        mujoco.mj_forward(self.model, data)
        torso_rotation = _body_rotation(self.model, data, "torso_link")
        calibration: dict[str, _ArmCalibration] = {}
        for side in ("left", "right"):
            source_upper = _source_body_rotation(targets, f"{side}_shoulder")
            source_wrist = _source_body_rotation(targets, f"{side}_wrist")
            robot_upper = torso_rotation.T @ _body_rotation(
                self.model, data, f"{side[0]}_upper_arm_link"
            )
            robot_wrist = torso_rotation.T @ _body_rotation(
                self.model, data, f"{side[0]}_wrist_link"
            )
            calibration[side] = _ArmCalibration(
                upper_frame_offset=source_upper.T @ robot_upper,
                wrist_frame_offset=source_wrist.T @ robot_wrist,
                standing_elbow_flexion=_elbow_flexion(targets, side),
            )
        self._calibration = calibration
        self.reset_live()
        return Twist2ReferenceFrame(
            qpos=self._default_qpos.copy(),
            qvel=np.zeros(self.model.nv, dtype=np.float64),
            sequence=frame.sequence,
            source_age_ms=None,
            received_at=frame.received_at,
        )

    def retarget(self, frame: SmplxTargetFrame) -> Twist2ReferenceFrame:
        """Map one source frame without iterative IK or GMR optimization."""

        if self._calibration is None:
            raise RuntimeError(
                "direct arm standing calibration has not been initialized"
            )
        targets = _validate_smplx_arm_frame(frame, live=True)
        if self._last_sequence is not None:
            if frame.sequence == self._last_sequence:
                assert self._cached is not None
                return self._cached
            if frame.sequence < self._last_sequence:
                raise ReferenceDiscontinuity(
                    "SMPL-X source sequence moved backwards: "
                    f"{frame.sequence} < {self._last_sequence}"
                )
        dt: float | None = None
        if self._previous_timestamp_ns is not None:
            if frame.source_timestamp_ns <= self._previous_timestamp_ns:
                raise ReferenceDiscontinuity("SMPL-X source timestamp did not advance")
            dt = (frame.source_timestamp_ns - self._previous_timestamp_ns) / 1.0e9
            if not 0.0 < dt <= self.max_source_interval_seconds:
                raise ReferenceDiscontinuity(
                    f"SMPL-X source interval is invalid: {dt:.6f} seconds"
                )

        candidate = self._default_qpos.copy()
        joint_target = self._direct_joint_target(targets)
        candidate[self._qpos_indices] = joint_target
        qvel = np.zeros(self.model.nv, dtype=np.float64)
        if self._previous_qpos is not None:
            assert dt is not None
            mujoco.mj_differentiatePos(
                self.model,
                qvel,
                dt,
                self._previous_qpos,
                candidate,
            )
        reference = Twist2ReferenceFrame(
            qpos=candidate,
            qvel=qvel,
            sequence=frame.sequence,
            source_age_ms=frame.source_age_ms,
            received_at=frame.received_at,
        )
        if self.validate_reference is not None:
            self.validate_reference(reference)
        self._previous_qpos = candidate
        self._previous_timestamp_ns = frame.source_timestamp_ns
        self._last_sequence = frame.sequence
        self._cached = reference
        return reference

    def reset_live(self) -> None:
        self._previous_qpos = None
        self._previous_timestamp_ns = None
        self._last_sequence = None
        self._cached = None

    def _direct_joint_target(
        self,
        targets: Mapping[str, tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        assert self._calibration is not None
        result = self._default_joint_target.copy()
        for side in ("left", "right"):
            calibration = self._calibration[side]
            upper_target = (
                _source_body_rotation(targets, f"{side}_shoulder")
                @ calibration.upper_frame_offset
            )
            shoulder_x, shoulder_y, shoulder_z = _decompose_xyz(upper_target, side)
            shoulder_sign_x = -1.0 if side == "left" else 1.0
            shoulder_sign_y = 1.0 if side == "left" else -1.0
            shoulder = (
                shoulder_sign_x * shoulder_x,
                shoulder_sign_y * shoulder_y,
                shoulder_z,
            )
            for suffix, value in zip(_ARM_JOINT_SUFFIXES[:3], shoulder, strict=True):
                self._set_clipped(result, f"{side}_{suffix}_joint", value)

            elbow_name = f"{side}_elbow_pitch_joint"
            elbow_index = self._index_by_name[elbow_name]
            elbow_pitch = self._default_joint_target[elbow_index] + (
                _elbow_flexion(targets, side) - calibration.standing_elbow_flexion
            )
            self._set_clipped(result, elbow_name, elbow_pitch)

            shoulder_rotation = (
                _rotation_x(
                    shoulder_sign_x
                    * result[self._index_by_name[f"{side}_shoulder_pitch_joint"]]
                )
                @ _rotation_y(
                    shoulder_sign_y
                    * result[self._index_by_name[f"{side}_shoulder_roll_joint"]]
                )
                @ _rotation_z(result[self._index_by_name[f"{side}_shoulder_yaw_joint"]])
            )
            elbow_sign = -1.0 if side == "left" else 1.0
            wrist_base = shoulder_rotation @ _rotation_y(
                elbow_sign * result[elbow_index]
            )
            wrist_target = (
                _source_body_rotation(targets, f"{side}_wrist")
                @ calibration.wrist_frame_offset
            )
            residual = wrist_base.T @ wrist_target
            elbow_yaw, wrist_y = _decompose_zy(residual)
            self._set_clipped(result, f"{side}_elbow_yaw_joint", elbow_yaw)
            wrist_sign = 1.0 if side == "left" else -1.0
            self._set_clipped(
                result,
                f"{side}_wrist_pitch_joint",
                wrist_sign * wrist_y,
            )
        return result

    def _set_clipped(self, target: np.ndarray, name: str, value: float) -> None:
        index = self._index_by_name[name]
        target[index] = np.clip(
            value, self._target_low[index], self._target_high[index]
        )

    def _validate_robot_axes(self) -> None:
        expected = {
            "right_shoulder_pitch_joint": (1.0, 0.0, 0.0),
            "right_shoulder_roll_joint": (0.0, -1.0, 0.0),
            "right_shoulder_yaw_joint": (0.0, 0.0, 1.0),
            "right_elbow_pitch_joint": (0.0, 1.0, 0.0),
            "right_elbow_yaw_joint": (0.0, 0.0, 1.0),
            "right_wrist_pitch_joint": (0.0, -1.0, 0.0),
            "left_shoulder_pitch_joint": (-1.0, 0.0, 0.0),
            "left_shoulder_roll_joint": (0.0, 1.0, 0.0),
            "left_shoulder_yaw_joint": (0.0, 0.0, 1.0),
            "left_elbow_pitch_joint": (0.0, -1.0, 0.0),
            "left_elbow_yaw_joint": (0.0, 0.0, 1.0),
            "left_wrist_pitch_joint": (0.0, 1.0, 0.0),
        }
        for name, axis in expected.items():
            actuator_index = self._index_by_name[name]
            joint_id = int(self.env.model.actuator_joint_ids[actuator_index])
            if not np.allclose(self.model.jnt_axis[joint_id], axis, atol=1.0e-3):
                raise ValueError(f"Bello direct arm axis changed for {name!r}")
            if not self.model.jnt_limited[joint_id]:
                raise ValueError(
                    f"Bello direct arm joint {name!r} requires hard limits"
                )


def _validate_smplx_arm_frame(
    frame: SmplxTargetFrame,
    *,
    live: bool,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    missing = sorted(set(SMPLX_ARM_TARGET_NAMES) - set(frame.transforms))
    if missing:
        raise ReferenceUnavailable(
            "SMPL-X frame is missing direct arm joints: " + ", ".join(missing)
        )
    if frame.sequence < 0 or frame.source_age_ms < 0:
        raise ReferenceUnavailable("SMPL-X sequence and source age must be nonnegative")
    if live and frame.source_timestamp_ns <= 0:
        raise ReferenceUnavailable("live SMPL-X source timestamp must be positive")
    result = {}
    for name in SMPLX_ARM_TARGET_NAMES:
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
        result[name] = (position.copy(), quaternion / norm)
    return result


def _source_body_rotation(
    targets: Mapping[str, tuple[np.ndarray, np.ndarray]],
    name: str,
) -> np.ndarray:
    torso = _quaternion_matrix(targets["spine3"][1])
    return torso.T @ _quaternion_matrix(targets[name][1])


def _elbow_flexion(
    targets: Mapping[str, tuple[np.ndarray, np.ndarray]],
    side: str,
) -> float:
    shoulder = targets[f"{side}_shoulder"][0]
    elbow = targets[f"{side}_elbow"][0]
    wrist = targets[f"{side}_wrist"][0]
    upper = elbow - shoulder
    forearm = wrist - elbow
    upper_length = float(np.linalg.norm(upper))
    forearm_length = float(np.linalg.norm(forearm))
    if upper_length < 1.0e-4 or forearm_length < 1.0e-4:
        raise ReferenceUnavailable(f"SMPL-X {side} arm bone is degenerate")
    cosine = float(np.dot(upper, forearm) / (upper_length * forearm_length))
    return math.acos(float(np.clip(cosine, -1.0, 1.0)))


def _body_rotation(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"Bello direct arm body {name!r} is missing")
    return np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3).copy()


def _decompose_xyz(rotation: np.ndarray, side: str) -> tuple[float, float, float]:
    y = math.asin(float(np.clip(rotation[0, 2], -1.0, 1.0)))
    if abs(math.cos(y)) < 0.05:
        raise ReferenceUnavailable(f"SMPL-X {side} shoulder is near a singular pose")
    x = math.atan2(-float(rotation[1, 2]), float(rotation[2, 2]))
    z = math.atan2(-float(rotation[0, 1]), float(rotation[0, 0]))
    return x, y, z


def _decompose_zy(rotation: np.ndarray) -> tuple[float, float]:
    z = math.atan2(-float(rotation[0, 1]), float(rotation[1, 1]))
    y = math.atan2(-float(rotation[2, 0]), float(rotation[2, 2]))
    return z, y


def _rotation_x(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray(((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine)))


def _rotation_y(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray(((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)))


def _rotation_z(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))


def _quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        )
    )
