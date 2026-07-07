"""Pure-JAX reward terms for Bello velocity-tracking locomotion."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class CommandConfig:
    """Uniform velocity command sampler.

    Commands are `[lin_vel_x, lin_vel_y, yaw_vel]`. The limit ranges are kept for
    later curriculum code; the v1 environment samples from the initial ranges.
    """

    lin_vel_x: tuple[float, float] = (-0.1, 0.1)
    lin_vel_y: tuple[float, float] = (-0.1, 0.1)
    ang_vel_yaw: tuple[float, float] = (-0.1, 0.1)
    limit_lin_vel_x: tuple[float, float] = (-0.5, 1.0)
    limit_lin_vel_y: tuple[float, float] = (-0.3, 0.3)
    limit_ang_vel_yaw: tuple[float, float] = (-0.2, 0.2)
    standing_probability: float = 0.02
    resample_seconds: float = 10.0

    def sample(self, rng: jax.Array, shape: tuple[int, ...] = ()) -> jax.Array:
        low = jnp.asarray(
            [self.lin_vel_x[0], self.lin_vel_y[0], self.ang_vel_yaw[0]],
            dtype=jnp.float32,
        )
        high = jnp.asarray(
            [self.lin_vel_x[1], self.lin_vel_y[1], self.ang_vel_yaw[1]],
            dtype=jnp.float32,
        )
        command_key, stand_key = jax.random.split(rng)
        command = jax.random.uniform(
            command_key, shape + (3,), minval=low, maxval=high
        )
        standing = jax.random.bernoulli(
            stand_key, self.standing_probability, shape
        )
        return jnp.where(standing[..., None], jnp.zeros_like(command), command)


@dataclass(frozen=True)
class RewardConfig:
    """Weights for the first Bello velocity-tracking baseline."""

    lin_vel_tracking_weight: float = 1.0
    yaw_vel_tracking_weight: float = 0.5
    alive_weight: float = 0.15
    lin_vel_z_weight: float = -2.0
    ang_vel_xy_weight: float = -0.05
    flat_orientation_weight: float = -5.0
    base_height_weight: float = -10.0
    joint_vel_weight: float = -0.001
    action_rate_weight: float = -0.05
    energy_weight: float = -2.0e-5
    joint_limit_weight: float = -5.0
    fixed_joint_deviation_weight: float = -0.1
    termination_weight: float = -1.0
    lin_vel_tracking_sigma: float = 0.25
    yaw_vel_tracking_sigma: float = 0.25
    base_height_target: float = 0.91


def compute_velocity_reward(
    *,
    config: RewardConfig,
    command: jax.Array,
    base_lin_vel: jax.Array,
    base_ang_vel: jax.Array,
    projected_gravity: jax.Array,
    base_height: jax.Array,
    joint_pos: jax.Array,
    joint_pos_rel: jax.Array,
    joint_vel: jax.Array,
    action: jax.Array,
    last_action: jax.Array,
    actuator_force: jax.Array,
    joint_limit_low: jax.Array,
    joint_limit_high: jax.Array,
    fixed_joint_indices: jax.Array,
    done: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Return weighted scalar reward and weighted component terms."""

    lin_vel_error = jnp.sum((base_lin_vel[:2] - command[:2]) ** 2)
    yaw_vel_error = (base_ang_vel[2] - command[2]) ** 2

    track_lin_vel_xy = jnp.exp(-lin_vel_error / config.lin_vel_tracking_sigma)
    track_ang_vel_z = jnp.exp(-yaw_vel_error / config.yaw_vel_tracking_sigma)
    alive = jnp.asarray(1.0, dtype=jnp.float32)

    lin_vel_z = base_lin_vel[2] ** 2
    ang_vel_xy = jnp.sum(base_ang_vel[:2] ** 2)
    flat_orientation = jnp.sum(projected_gravity[:2] ** 2)
    base_height_error = (base_height - config.base_height_target) ** 2
    joint_vel_l2 = jnp.mean(joint_vel**2)
    action_rate_l2 = jnp.mean((action - last_action) ** 2)
    energy = jnp.mean(jnp.abs(actuator_force * joint_vel))

    lower_violation = jnp.maximum(joint_limit_low - joint_pos, 0.0)
    upper_violation = jnp.maximum(joint_pos - joint_limit_high, 0.0)
    joint_limit = jnp.mean(lower_violation**2 + upper_violation**2)
    fixed_joint_deviation = jnp.mean(jnp.abs(joint_pos_rel[fixed_joint_indices]))
    termination = done.astype(jnp.float32)

    terms = {
        "reward/track_lin_vel_xy": (
            config.lin_vel_tracking_weight * track_lin_vel_xy
        ),
        "reward/track_ang_vel_z": (
            config.yaw_vel_tracking_weight * track_ang_vel_z
        ),
        "reward/alive": config.alive_weight * alive,
        "reward/lin_vel_z": config.lin_vel_z_weight * lin_vel_z,
        "reward/ang_vel_xy": config.ang_vel_xy_weight * ang_vel_xy,
        "reward/flat_orientation": (
            config.flat_orientation_weight * flat_orientation
        ),
        "reward/base_height": config.base_height_weight * base_height_error,
        "reward/joint_vel": config.joint_vel_weight * joint_vel_l2,
        "reward/action_rate": config.action_rate_weight * action_rate_l2,
        "reward/energy": config.energy_weight * energy,
        "reward/joint_limit": config.joint_limit_weight * joint_limit,
        "reward/fixed_joint_deviation": (
            config.fixed_joint_deviation_weight * fixed_joint_deviation
        ),
        "reward/termination": config.termination_weight * termination,
    }
    reward = sum(terms.values(), jnp.asarray(0.0, dtype=jnp.float32))
    return reward, terms
