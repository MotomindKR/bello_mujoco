"""Versioned, runtime-independent motion clips for whole-body tracking."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np

from bello_mujoco.actuators import ActuatorConfig
from bello_mujoco.assets_pipeline.mesh import JointAvailabilityConfig
from bello_mujoco.assets_pipeline.mjcf import CollisionMode
from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.sim import BelloModel

MOTION_FORMAT_VERSION = 1

FIXED_TRACKING_JOINT_NAMES = (
    "waist_roll_joint",
    "waist_pitch_joint",
    "neck_yaw_joint",
    "head_pitch_joint",
)

FIXED_TRACKING_JOINT_AVAILABILITY = JointAvailabilityConfig(
    waist_yaw=True,
)

GMR_BELLO_JOINT_NAMES = (
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
    "neck_yaw_joint",
    "head_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_yaw_joint",
    "right_wrist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_yaw_joint",
    "left_wrist_pitch_joint",
)

DEFAULT_TRACKED_BODY_NAMES = (
    "pelvis_link",
    "torso_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "l_wrist_link",
    "r_wrist_link",
    "left_knee_link",
    "right_knee_link",
    "l_elbow_link",
    "r_elbow_link",
)

BEYONDMIMIC_TRACKED_BODY_NAMES = (
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


def motion_tracking_actuator_config() -> ActuatorConfig:
    """Return the all-policy logical actuator contract for motion tracking."""

    defaults = ActuatorConfig()
    return ActuatorConfig(
        joints={
            name: replace(config, mode="policy")
            for name, config in defaults.joints.items()
        }
    )


def motion_tracking_asset_config(
    *,
    collision_mode: CollisionMode = "box",
    output_root: Path | None = None,
) -> AssetPrepConfig:
    """Return the fixed-head, sphere-hand asset used by motion workflows."""

    return AssetPrepConfig(
        asset="full_body",
        output_root=output_root,
        collision_mode=collision_mode,
        target_total_mass_kg=None,
        enforce_policy_effort_limits=False,
        joint_availability=FIXED_TRACKING_JOINT_AVAILABILITY,
        actuator_config=motion_tracking_actuator_config(),
    )


@chex.dataclass(frozen=True)
class MotionSample:
    qpos: jax.Array
    qvel: jax.Array
    body_pos: jax.Array
    body_quat: jax.Array
    body_lin_vel: jax.Array
    body_ang_vel: jax.Array
    foot_contacts: jax.Array


@dataclass(frozen=True)
class MotionClip:
    fps: float
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    source_sha256: str
    qpos: jax.Array
    qvel: jax.Array
    body_pos: jax.Array
    body_quat: jax.Array
    body_lin_vel: jax.Array
    body_ang_vel: jax.Array
    foot_contacts: jax.Array

    @property
    def num_frames(self) -> int:
        return int(self.qpos.shape[0])

    @property
    def duration(self) -> float:
        return (self.num_frames - 1) / self.fps


def load_motion_clip(path: Path | str, model: BelloModel) -> MotionClip:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "format_version",
            "fps",
            "joint_names",
            "body_names",
            "source_sha256",
            "qpos",
            "qvel",
            "body_pos",
            "body_quat_wxyz",
            "body_lin_vel",
            "body_ang_vel",
            "foot_contacts",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"motion clip is missing fields: {missing}")
        version = int(_scalar(archive["format_version"]))
        if version != MOTION_FORMAT_VERSION:
            raise ValueError(
                f"unsupported motion format version {version}; "
                f"expected {MOTION_FORMAT_VERSION}"
            )
        fps = float(_scalar(archive["fps"]))
        joint_names = tuple(str(name) for name in archive["joint_names"].tolist())
        body_names = tuple(str(name) for name in archive["body_names"].tolist())
        source_sha256 = str(_scalar(archive["source_sha256"]))
        arrays = {
            "qpos": np.asarray(archive["qpos"], dtype=np.float32),
            "qvel": np.asarray(archive["qvel"], dtype=np.float32),
            "body_pos": np.asarray(archive["body_pos"], dtype=np.float32),
            "body_quat": np.asarray(archive["body_quat_wxyz"], dtype=np.float32),
            "body_lin_vel": np.asarray(archive["body_lin_vel"], dtype=np.float32),
            "body_ang_vel": np.asarray(archive["body_ang_vel"], dtype=np.float32),
            "foot_contacts": np.asarray(archive["foot_contacts"], dtype=np.float32),
        }
    _validate_motion(fps, joint_names, body_names, arrays, model)
    return MotionClip(
        fps=fps,
        joint_names=joint_names,
        body_names=body_names,
        source_sha256=source_sha256,
        **{name: jnp.asarray(value) for name, value in arrays.items()},
    )


def sample_motion(clip: MotionClip, time: jax.Array) -> MotionSample:
    frame = jnp.clip(time * clip.fps, 0.0, clip.num_frames - 1)
    lower = jnp.floor(frame).astype(jnp.int32)
    upper = jnp.minimum(lower + 1, clip.num_frames - 1)
    alpha = frame - lower.astype(frame.dtype)

    qpos = _lerp(clip.qpos[lower], clip.qpos[upper], alpha)
    qpos = qpos.at[3:7].set(
        quat_slerp(clip.qpos[lower, 3:7], clip.qpos[upper, 3:7], alpha)
    )
    body_quat = jax.vmap(quat_slerp, in_axes=(0, 0, None))(
        clip.body_quat[lower], clip.body_quat[upper], alpha
    )
    contacts = jnp.where(
        alpha < 0.5, clip.foot_contacts[lower], clip.foot_contacts[upper]
    )
    return MotionSample(
        qpos=qpos,
        qvel=_lerp(clip.qvel[lower], clip.qvel[upper], alpha),
        body_pos=_lerp(clip.body_pos[lower], clip.body_pos[upper], alpha),
        body_quat=body_quat,
        body_lin_vel=_lerp(clip.body_lin_vel[lower], clip.body_lin_vel[upper], alpha),
        body_ang_vel=_lerp(clip.body_ang_vel[lower], clip.body_ang_vel[upper], alpha),
        foot_contacts=contacts,
    )


def quat_slerp(q0: jax.Array, q1: jax.Array, alpha: jax.Array) -> jax.Array:
    q0 = q0 / jnp.maximum(jnp.linalg.norm(q0), 1.0e-8)
    q1 = q1 / jnp.maximum(jnp.linalg.norm(q1), 1.0e-8)
    dot = jnp.dot(q0, q1)
    q1 = jnp.where(dot < 0.0, -q1, q1)
    dot = jnp.clip(jnp.abs(dot), 0.0, 1.0)
    theta = jnp.arccos(dot)
    sin_theta = jnp.sin(theta)
    slerped = (
        jnp.sin((1.0 - alpha) * theta) * q0 + jnp.sin(alpha * theta) * q1
    ) / jnp.maximum(sin_theta, 1.0e-8)
    linear = (1.0 - alpha) * q0 + alpha * q1
    result = jnp.where(dot > 0.9995, linear, slerped)
    return result / jnp.maximum(jnp.linalg.norm(result), 1.0e-8)


def _lerp(a: jax.Array, b: jax.Array, alpha: jax.Array) -> jax.Array:
    return a + alpha * (b - a)


def _scalar(value: np.ndarray):
    if np.asarray(value).size != 1:
        raise ValueError(
            f"expected scalar metadata, got shape {np.asarray(value).shape}"
        )
    return np.asarray(value).reshape(()).item()


def _validate_motion(
    fps: float,
    joint_names: tuple[str, ...],
    body_names: tuple[str, ...],
    arrays: dict[str, np.ndarray],
    model: BelloModel,
) -> None:
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be positive and finite, got {fps}")
    if joint_names != model.actuator_joint_names:
        raise ValueError(
            "motion joint order does not match model actuators:\n"
            f"motion={joint_names}\nmodel={model.actuator_joint_names}"
        )
    if len(set(body_names)) != len(body_names):
        raise ValueError("motion body_names contains duplicates")
    frame_count = arrays["qpos"].shape[0]
    body_count = len(body_names)
    expected_shapes = {
        "qpos": (frame_count, model.mj_model.nq),
        "qvel": (frame_count, model.mj_model.nv),
        "body_pos": (frame_count, body_count, 3),
        "body_quat": (frame_count, body_count, 4),
        "body_lin_vel": (frame_count, body_count, 3),
        "body_ang_vel": (frame_count, body_count, 3),
        "foot_contacts": (frame_count, 2),
    }
    if frame_count < 2:
        raise ValueError(f"motion must contain at least two frames, got {frame_count}")
    for name, expected in expected_shapes.items():
        value = arrays[name]
        if value.shape != expected:
            raise ValueError(f"{name} has shape {value.shape}; expected {expected}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains nonfinite values")
    root_norm = np.linalg.norm(arrays["qpos"][:, 3:7], axis=-1)
    body_norm = np.linalg.norm(arrays["body_quat"], axis=-1)
    if not np.allclose(root_norm, 1.0, atol=1.0e-4):
        raise ValueError("root quaternions are not normalized")
    if not np.allclose(body_norm, 1.0, atol=1.0e-4):
        raise ValueError("body quaternions are not normalized")
