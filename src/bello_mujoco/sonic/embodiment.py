"""Name-keyed kinematic boundary between pretrained G1 SONIC and Bello Mini."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


# The released checkpoint is trained in this Isaac Lab articulation order.
# Keep it local rather than importing the reference implementation at runtime.
G1_JOINT_NAMES = (
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


def _source_name(target_name: str) -> str:
    return target_name.replace("_elbow_pitch_joint", "_elbow_joint")


def g1_neutral_positions() -> np.ndarray:
    """Return the release robot's named neutral pose in checkpoint order."""

    values = np.zeros(len(G1_JOINT_NAMES), dtype=np.float32)
    for side in ("left", "right"):
        values[G1_JOINT_NAMES.index(f"{side}_hip_pitch_joint")] = -0.312
        values[G1_JOINT_NAMES.index(f"{side}_knee_joint")] = 0.669
        values[G1_JOINT_NAMES.index(f"{side}_ankle_pitch_joint")] = -0.363
        values[G1_JOINT_NAMES.index(f"{side}_elbow_joint")] = 0.6
        values[G1_JOINT_NAMES.index(f"{side}_shoulder_pitch_joint")] = 0.2
    values[G1_JOINT_NAMES.index("left_shoulder_roll_joint")] = 0.2
    values[G1_JOINT_NAMES.index("right_shoulder_roll_joint")] = -0.2
    return values


@dataclass(frozen=True)
class MiniEmbodimentAdapter:
    """Validated analytic codec around the frozen G1 network.

    All transforms are name keyed.  ``signs`` expresses target-positive motion
    in the source joint convention and is deliberately explicit even when all
    currently validated axes have the same sign.
    """

    target_joint_names: tuple[str, ...]
    target_neutral: np.ndarray
    target_to_source_indices: np.ndarray
    source_to_target_indices: np.ndarray
    signs: np.ndarray
    source_neutral: np.ndarray
    motor_from_joint: np.ndarray

    def __hash__(self) -> int:
        arrays = (
            self.target_neutral,
            self.target_to_source_indices,
            self.source_to_target_indices,
            self.signs,
            self.source_neutral,
            self.motor_from_joint,
        )
        return hash(
            (
                self.target_joint_names,
                *((array.dtype.str, array.shape, array.tobytes()) for array in arrays),
            )
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, MiniEmbodimentAdapter):
            return NotImplemented
        return self.target_joint_names == other.target_joint_names and all(
            np.array_equal(left, right)
            for left, right in (
                (self.target_neutral, other.target_neutral),
                (self.target_to_source_indices, other.target_to_source_indices),
                (self.source_to_target_indices, other.source_to_target_indices),
                (self.signs, other.signs),
                (self.source_neutral, other.source_neutral),
                (self.motor_from_joint, other.motor_from_joint),
            )
        )

    @classmethod
    def from_model(cls, model) -> MiniEmbodimentAdapter:
        target_names = tuple(model.actuator_joint_names)
        source_names = tuple(_source_name(name) for name in target_names)
        unknown = sorted(set(source_names) - set(G1_JOINT_NAMES))
        missing = sorted(set(G1_JOINT_NAMES) - set(source_names))
        if unknown:
            raise ValueError(f"Bello Mini has joints with no SONIC semantic: {unknown}")
        if missing != ["waist_pitch_joint"]:
            raise ValueError(
                "SONIC Mini mapping must differ from G1 only by waist_pitch_joint; "
                f"got missing={missing}"
            )
        if len(set(source_names)) != len(source_names):
            raise ValueError("Bello Mini SONIC joint mapping contains duplicates")
        target_to_source = np.asarray(
            [G1_JOINT_NAMES.index(name) for name in source_names], dtype=np.int32
        )
        source_to_target = np.full(len(G1_JOINT_NAMES), -1, dtype=np.int32)
        source_to_target[target_to_source] = np.arange(
            len(target_names), dtype=np.int32
        )
        transform = np.asarray(model.actuator_coordinate_transform, dtype=np.float32)
        if transform.shape != (len(target_names), len(target_names)):
            raise ValueError(
                "Mini motor transmission must be square in logical joint coordinates; "
                f"got {transform.shape}"
            )
        if not np.all(np.isfinite(transform)) or np.linalg.matrix_rank(
            transform
        ) != len(target_names):
            raise ValueError("Mini motor transmission must be finite and invertible")
        return cls(
            target_joint_names=target_names,
            target_neutral=np.asarray(model.default_joint_target, dtype=np.float32),
            target_to_source_indices=target_to_source,
            source_to_target_indices=source_to_target,
            signs=np.ones(len(target_names), dtype=np.float32),
            source_neutral=g1_neutral_positions(),
            motor_from_joint=transform,
        )

    @property
    def target_size(self) -> int:
        return len(self.target_joint_names)

    @property
    def source_size(self) -> int:
        return len(G1_JOINT_NAMES)

    def target_position_to_source(self, value: jax.Array) -> jax.Array:
        value = self._require_last_dim(value, self.target_size, "target position")
        relative = (value - jnp.asarray(self.target_neutral)) * jnp.asarray(self.signs)
        output = jnp.asarray(self.source_neutral, dtype=value.dtype)
        return output.at[jnp.asarray(self.target_to_source_indices)].set(
            relative + output[jnp.asarray(self.target_to_source_indices)]
        )

    def target_vector_to_source(self, value: jax.Array) -> jax.Array:
        value = self._require_last_dim(value, self.target_size, "target vector")
        output = jnp.zeros(value.shape[:-1] + (self.source_size,), dtype=value.dtype)
        return output.at[..., jnp.asarray(self.target_to_source_indices)].set(
            value * jnp.asarray(self.signs)
        )

    def source_action_to_target(self, value: jax.Array) -> jax.Array:
        value = self._require_last_dim(value, self.source_size, "source action")
        return value[..., jnp.asarray(self.target_to_source_indices)] * jnp.asarray(
            self.signs
        )

    def source_scale_to_target(self, value: jax.Array) -> jax.Array:
        """Gather unsigned per-joint scales such as Gaussian log standard deviations."""

        value = self._require_last_dim(value, self.source_size, "source scale")
        return value[..., jnp.asarray(self.target_to_source_indices)]

    def motor_feedback_to_logical(self, value: jax.Array) -> jax.Array:
        value = self._require_last_dim(value, self.target_size, "motor feedback")
        return jnp.matmul(
            value,
            jnp.asarray(np.linalg.inv(self.motor_from_joint)).T,
            precision=jax.lax.Precision.HIGHEST,
        )

    def logical_target_to_motor(self, value: jax.Array) -> jax.Array:
        value = self._require_last_dim(value, self.target_size, "logical target")
        return jnp.matmul(
            value,
            jnp.asarray(self.motor_from_joint).T,
            precision=jax.lax.Precision.HIGHEST,
        )

    @staticmethod
    def _require_last_dim(value: jax.Array, size: int, name: str) -> jax.Array:
        value = jnp.asarray(value)
        if value.ndim == 0 or value.shape[-1] != size:
            raise ValueError(f"{name} must end in {size} values, got {value.shape}")
        return value
