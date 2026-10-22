"""Shared logical action scaling, motor projection, and command slew limits."""

import math

import jax
import jax.numpy as jnp


def logical_action_scale(model, actuator_config) -> jax.Array:
    """Scale logical joint proposals using their configured effort and stiffness."""
    scales = []
    for name in model.actuator_joint_names:
        joint = actuator_config.by_name[name]
        effort, stiffness = joint.effort_limit, joint.gains[0]
        if (
            effort is None
            or not math.isfinite(effort)
            or effort <= 0
            or not math.isfinite(stiffness)
            or stiffness <= 0
        ):
            raise ValueError(
                f"action mapping requires positive effort and stiffness for {name!r}"
            )
        scales.append(0.25 * effort / stiffness)
    return jnp.asarray(scales, dtype=jnp.float32)


def project_motor_target(target, reference, limits):
    """Shorten each connected motor group's move to the feasible boundary.

    This is a ray limit, not a nearest-point projection or a low-pass filter.
    The reference must satisfy both motor and joint limits.
    """
    matrix = jnp.asarray(limits.matrix, dtype=target.dtype)
    bounds = jnp.asarray(limits.bounds, dtype=target.dtype)
    start = matrix @ reference
    delta = target - reference
    change = matrix @ delta
    available = jnp.where(change >= 0, bounds[:, 1] - start, start - bounds[:, 0])
    ratios = jnp.where(
        change != 0, available / jnp.where(change != 0, jnp.abs(change), 1), 1
    )
    scale = jnp.min(
        jnp.where(jnp.asarray(limits.membership), ratios[None, :], 1), axis=-1
    )
    scale = jnp.clip(scale, 0, 1)
    # Stay just inside an active boundary after float32 serialization.
    scale = jnp.where(scale < 1, scale * (1 - 1e-6), scale)
    return jnp.where(scale < 1, reference + scale * delta, target)


def limit_motor_step(target, previous, limits):
    delta = target - previous
    ratios = jnp.asarray(limits.position_step) / jnp.where(
        delta != 0, jnp.abs(delta), 1
    )
    ratios = jnp.where(delta != 0, ratios, 1)
    scale = jnp.min(jnp.where(jnp.asarray(limits.groups), ratios[None, :], 1), axis=-1)
    scale = jnp.clip(scale, 0, 1)
    scale = jnp.where(scale < 1, scale * (1 - 1e-6), scale)
    return jnp.where(scale < 1, previous + scale * delta, target)


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
