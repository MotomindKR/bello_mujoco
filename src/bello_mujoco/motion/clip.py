"""Runtime-independent reference arrays for whole-body tracking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import chex
import jax
import jax.numpy as jnp
import numpy as np

from bello_mujoco.actuators import ActuatorConfig
from bello_mujoco.models.compiler import ModelConfig
from bello_mujoco.robot import AssetName
from bello_mujoco.sim import BelloModel


def motion_tracking_actuator_config(asset: AssetName) -> ActuatorConfig:
    """Return the all-policy logical actuator contract for motion tracking."""

    return ActuatorConfig.for_asset(asset, mode="policy")


def motion_tracking_asset_config(
    *,
    asset: AssetName,
    end_effector_payload_mass_kg: float = 0.0,
) -> ModelConfig:
    """Return the fixed-head, sphere-hand asset used by motion workflows."""

    return ModelConfig(
        asset=asset,
        target_total_mass_kg=None,
        end_effector_payload_mass_kg=end_effector_payload_mass_kg,
        enforce_policy_effort_limits=False,
        actuator_mode="policy",
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
    qpos: jax.Array
    qvel: jax.Array
    body_pos: jax.Array
    body_quat: jax.Array
    body_lin_vel: jax.Array
    body_ang_vel: jax.Array
    foot_contacts: jax.Array

    object_mass_kg: float | None = None
    screen_passed: bool = False
    free_joint_qpos: tuple[int, ...] = (0,)
    joint_qpos_indices: tuple[int, ...] = ()
    joint_qvel_indices: tuple[int, ...] = ()

    @property
    def num_frames(self) -> int:
        return int(self.qpos.shape[0])

    @property
    def duration(self) -> float:
        return (self.num_frames - 1) / self.fps


def load_motion_clip(
    path: Path | str, model: BelloModel, *, allow_unscreened_reference: bool = False
) -> MotionClip:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "fps",
            "joint_names",
            "body_names",
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
        fps = float(_scalar(archive["fps"]))
        joint_names = tuple(str(name) for name in archive["joint_names"].tolist())
        body_names = tuple(str(name) for name in archive["body_names"].tolist())
        arrays = {
            "qpos": np.asarray(archive["qpos"], dtype=np.float32),
            "qvel": np.asarray(archive["qvel"], dtype=np.float32),
            "body_pos": np.asarray(archive["body_pos"], dtype=np.float32),
            "body_quat": np.asarray(archive["body_quat_wxyz"], dtype=np.float32),
            "body_lin_vel": np.asarray(archive["body_lin_vel"], dtype=np.float32),
            "body_ang_vel": np.asarray(archive["body_ang_vel"], dtype=np.float32),
            "foot_contacts": np.asarray(archive["foot_contacts"], dtype=np.float32),
        }
        object_mass = None
        screen_passed = False
        if model.bar_body_id >= 0:
            object_mass = float(_scalar(archive["bar_mass_kg"]))
            screen_passed = bool(_scalar(archive["screen_passed"]))
            if (
                not np.isfinite(object_mass)
                or object_mass <= 0.0
                or (not screen_passed and not allow_unscreened_reference)
                or not np.isclose(
                    model.mj_model.body_mass[model.bar_body_id],
                    object_mass,
                    rtol=0,
                    atol=1e-8,
                )
            ):
                raise ValueError(
                    "barbell motion requires a passed reference screen matching the configured bar mass"
                )
        free_qpos = tuple(q for q, _ in model.free_joint_coordinates)
        if "free_joint_qpos" in archive:
            if tuple(archive["free_joint_qpos"].tolist()) != free_qpos:
                raise ValueError(
                    "motion free-body coordinate mapping differs from scene"
                )
            if tuple(archive["joint_qpos_indices"].tolist()) != tuple(
                model.actuator_qpos_indices
            ) or tuple(archive["joint_qvel_indices"].tolist()) != tuple(
                model.actuator_qvel_indices
            ):
                raise ValueError("motion robot coordinate mapping differs from scene")
        elif len(free_qpos) != 1:
            raise ValueError("object motion requires explicit coordinate mapping")
    _validate_motion(fps, joint_names, body_names, arrays, model)
    return MotionClip(
        fps=fps,
        joint_names=joint_names,
        body_names=body_names,
        object_mass_kg=object_mass,
        screen_passed=screen_passed,
        free_joint_qpos=free_qpos,
        joint_qpos_indices=tuple(model.actuator_qpos_indices.tolist()),
        joint_qvel_indices=tuple(model.actuator_qvel_indices.tolist()),
        **{name: jnp.asarray(value) for name, value in arrays.items()},
    )


def save_motion_clip(path: Path | BinaryIO, clip: MotionClip) -> None:
    """Store the in-memory reference actually used by training, not its source path."""
    np.savez_compressed(
        path,
        **(
            dict(
                free_joint_qpos=np.asarray(clip.free_joint_qpos),
                joint_qpos_indices=np.asarray(clip.joint_qpos_indices),
                joint_qvel_indices=np.asarray(clip.joint_qvel_indices),
            )
            if clip.joint_qpos_indices
            else {}
        ),
        **(
            dict(bar_mass_kg=clip.object_mass_kg, screen_passed=clip.screen_passed)
            if clip.object_mass_kg is not None
            else {}
        ),
        fps=clip.fps,
        joint_names=np.asarray(clip.joint_names),
        body_names=np.asarray(clip.body_names),
        qpos=np.asarray(clip.qpos),
        qvel=np.asarray(clip.qvel),
        body_pos=np.asarray(clip.body_pos),
        body_quat_wxyz=np.asarray(clip.body_quat),
        body_lin_vel=np.asarray(clip.body_lin_vel),
        body_ang_vel=np.asarray(clip.body_ang_vel),
        foot_contacts=np.asarray(clip.foot_contacts),
    )


def sample_motion(clip: MotionClip, time: jax.Array) -> MotionSample:
    frame = jnp.clip(time * clip.fps, 0.0, clip.num_frames - 1)
    lower = jnp.floor(frame).astype(jnp.int32)
    upper = jnp.minimum(lower + 1, clip.num_frames - 1)
    alpha = frame - lower.astype(frame.dtype)

    qpos = _lerp(clip.qpos[lower], clip.qpos[upper], alpha)
    for address in clip.free_joint_qpos:
        qpos = qpos.at[address + 3 : address + 7].set(
            quat_slerp(
                clip.qpos[lower, address + 3 : address + 7],
                clip.qpos[upper, address + 3 : address + 7],
                alpha,
            )
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
    root_norm = np.stack(
        [
            np.linalg.norm(arrays["qpos"][:, q + 3 : q + 7], axis=-1)
            for q, _ in model.free_joint_coordinates
        ]
    )
    body_norm = np.linalg.norm(arrays["body_quat"], axis=-1)
    if not np.allclose(root_norm, 1.0, atol=1.0e-4):
        raise ValueError("root quaternions are not normalized")
    if not np.allclose(body_norm, 1.0, atol=1.0e-4):
        raise ValueError("body quaternions are not normalized")
