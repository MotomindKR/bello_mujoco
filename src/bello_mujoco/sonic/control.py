"""SONIC logical action semantics and canonical Mini motor projection."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from bello_mujoco.envs.actions import (
    limit_motor_step,
    logical_action_scale,
    project_motor_target,
)


def sonic_action_target(
    action: jax.Array,
    previous_motor_target: jax.Array,
    joint_default_offset: jax.Array,
    *,
    model,
    actuator_config,
) -> jax.Array:
    """Map a Gaussian SONIC action through logical joints and transmissions."""

    action = jnp.asarray(action, dtype=jnp.float32)
    joint_count = len(model.actuator_joint_names)
    if action.shape != (joint_count,):
        raise ValueError(
            f"SONIC action must have shape ({joint_count},), got {action.shape}"
        )
    logical_target = (
        jnp.asarray(model.default_joint_target)
        + jnp.asarray(joint_default_offset)
        + logical_action_scale(model, actuator_config) * action
    )
    requested_motor = jnp.matmul(
        jnp.asarray(model.actuator_coordinate_transform),
        logical_target,
        precision=jax.lax.Precision.HIGHEST,
    )
    constrained = project_motor_target(
        requested_motor,
        jnp.asarray(model.default_ctrl),
        model.target_limits,
    )
    return limit_motor_step(constrained, previous_motor_target, model.target_limits)
