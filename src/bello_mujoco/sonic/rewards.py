"""Any2Any SONIC reward terms expressed as side-effect-free JAX functions."""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax.numpy as jnp


@dataclass(frozen=True)
class SonicRewardConfig:
    anchor_position_weight: float = 0.5
    anchor_orientation_weight: float = 0.5
    relative_body_position_weight: float = 1.0
    relative_body_orientation_weight: float = 1.0
    body_linear_velocity_weight: float = 1.0
    body_angular_velocity_weight: float = 1.0
    action_rate_weight: float = -0.1
    joint_limit_weight: float = -10.0
    undesired_contact_weight: float = -0.1
    anti_shake_weight: float = -5.0e-3
    five_point_weight: float = 2.0
    foot_acceleration_weight: float = -2.5e-6
    anchor_position_sigma: float = 0.3
    anchor_orientation_sigma: float = 0.4
    relative_body_position_sigma: float = 0.3
    relative_body_orientation_sigma: float = 0.4
    body_linear_velocity_sigma: float = 1.0
    body_angular_velocity_sigma: float = math.pi
    five_point_sigma: float = 0.1
    anti_shake_threshold: float = 1.5


def gaussian_mean_squared(error, sigma: float):
    error = jnp.asarray(error)
    return jnp.exp(-jnp.mean(jnp.square(error), axis=(-2, -1)) / sigma**2)


def local_point_tracking(
    robot_position,
    robot_anchor_position,
    robot_from_world_rotation,
    reference_position,
    reference_anchor_position,
    reference_from_world_rotation,
    *,
    sigma: float,
):
    """Reference five-point reward in each root's own local frame."""

    robot_local = jnp.einsum(
        "...ij,...pj->...pi",
        robot_from_world_rotation,
        robot_position - robot_anchor_position[..., None, :],
    )
    reference_local = jnp.einsum(
        "...ij,...pj->...pi",
        reference_from_world_rotation,
        reference_position - reference_anchor_position[..., None, :],
    )
    squared_distance = jnp.sum(jnp.square(robot_local - reference_local), axis=-1)
    return jnp.exp(-jnp.mean(squared_distance, axis=-1) / sigma**2)


def anti_shake_angular_velocity(angular_velocity, threshold: float = 1.5):
    speed = jnp.linalg.norm(jnp.asarray(angular_velocity), axis=-1)
    return jnp.mean(jnp.square(jnp.maximum(speed - threshold, 0.0)), axis=-1)


def joint_acceleration(current_velocity, previous_velocity, timestep: float):
    if timestep <= 0.0:
        raise ValueError("timestep must be positive")
    acceleration = (jnp.asarray(current_velocity) - previous_velocity) / timestep
    return jnp.sum(jnp.square(acceleration), axis=-1)
