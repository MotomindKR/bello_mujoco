"""Policy targets in physical actuator coordinates, shared by both tasks."""

import jax
import jax.numpy as jnp


def bounded_motor_target(
    action: jax.Array,
    default: jax.Array,
    limits: jax.Array,
    policy_indices: jax.Array,
    action_clip: float,
) -> jax.Array:
    default = jnp.clip(default, limits[:, 0], limits[:, 1])
    normalized = jnp.clip(action / action_clip, -1.0, 1.0)
    center = default[policy_indices]
    span = jnp.where(
        normalized >= 0.0,
        limits[policy_indices, 1] - center,
        center - limits[policy_indices, 0],
    )
    target = default.at[policy_indices].set(center + span * normalized)
    return jnp.clip(target, limits[:, 0], limits[:, 1])


def motor_default(model, joint_offset: jax.Array) -> jax.Array:
    """Transform calibration offsets, never policy actions, through the transmission."""
    return jnp.asarray(model.default_ctrl, dtype=jnp.float32) + jnp.matmul(
        jnp.asarray(model.actuator_coordinate_transform, dtype=jnp.float32),
        joint_offset,
        precision=jax.lax.Precision.HIGHEST,
    )
