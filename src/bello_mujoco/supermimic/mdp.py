"""Analytic tracking reward, termination, and value features for both physics paths."""

from dataclasses import dataclass, field
import math

import jax
import jax.numpy as jnp

from bello_mujoco.supermimic import rotations as rot
from bello_mujoco.supermimic.kinematics import Kinematics
from bello_mujoco.supermimic.model import invariant_features


@dataclass(frozen=True)
class TrackingConfig:
    body_position_sigma: float = 0.3
    body_orientation_sigma: float = 0.4
    body_linear_velocity_sigma: float = 1.0
    body_angular_velocity_sigma: float = math.pi
    root_position_sigma: float = 0.3
    root_orientation_sigma: float = 0.4
    action_rate_weight: float = -0.1
    joint_limit_weight: float = -2.0
    soft_limit_factor: float = 0.9
    height_error: float = 0.25
    tilt_error: float = 0.8
    soft_height_width: float = 0.02
    soft_tilt_width: float = 0.04

    def __post_init__(self):
        for name, value in vars(self).items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name.endswith(("sigma", "error", "width")) and value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.soft_limit_factor <= 1:
            raise ValueError("soft_limit_factor must be in (0, 1]")
        if self.action_rate_weight > 0 or self.joint_limit_weight > 0:
            raise ValueError("regularization weights must be nonpositive")


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TrackingMDP:
    kinematics: Kinematics
    joint_low: jax.Array
    joint_high: jax.Array
    timestep: float = field(metadata={"static": True})
    config: TrackingConfig = field(metadata={"static": True})

    def reward(self, state, reference, action, previous_action):
        cfg = self.config
        bodies, targets = (
            self.kinematics.forward(state),
            self.kinematics.forward(reference),
        )
        alignment = rot.heading(
            rot.multiply(state.quaternion, rot.conjugate(reference.quaternion))
        )
        origin = state.position.at[2].set(reference.position[2])
        target_position = origin + rot.rotate(
            alignment, targets.position - reference.position
        )
        target_quaternion = rot.multiply(alignment, targets.quaternion)
        errors = (
            jnp.mean(jnp.sum((bodies.position - target_position) ** 2, axis=-1)),
            jnp.mean(
                jnp.sum(
                    rot.log(
                        rot.multiply(
                            rot.conjugate(bodies.quaternion), target_quaternion
                        )
                    )
                    ** 2,
                    axis=-1,
                )
            ),
            jnp.mean(
                jnp.sum(
                    (bodies.linear_velocity - targets.linear_velocity) ** 2, axis=-1
                )
            ),
            jnp.mean(
                jnp.sum(
                    (bodies.angular_velocity - targets.angular_velocity) ** 2, axis=-1
                )
            ),
            jnp.sum((state.position - reference.position) ** 2),
            jnp.sum(
                rot.log(
                    rot.multiply(rot.conjugate(state.quaternion), reference.quaternion)
                )
                ** 2
            ),
        )
        sigmas = (
            cfg.body_position_sigma,
            cfg.body_orientation_sigma,
            cfg.body_linear_velocity_sigma,
            cfg.body_angular_velocity_sigma,
            cfg.root_position_sigma,
            cfg.root_orientation_sigma,
        )
        tracking = sum(
            weight * jnp.exp(-error / sigma**2)
            for weight, error, sigma in zip(
                (1.0, 1.0, 1.0, 1.0, 0.5, 0.5), errors, sigmas, strict=True
            )
        )
        mid = (self.joint_low + self.joint_high) / 2
        half = (self.joint_high - self.joint_low) / 2 * cfg.soft_limit_factor
        limit = jnp.sum(
            jax.nn.relu(state.joint_position - mid - half)
            + jax.nn.relu(mid - half - state.joint_position)
        )
        rate = jnp.sum((action - previous_action) ** 2)
        reward = (
            tracking + cfg.joint_limit_weight * limit + cfg.action_rate_weight * rate
        ) * self.timestep
        return reward, {
            "tracking": tracking * self.timestep,
            "joint_limit": limit,
            "action_rate": rate,
            "body_position_squared": errors[0],
            "root_position_squared": errors[4],
        }

    def errors(self, state, reference):
        gravity = jnp.asarray([0.0, 0.0, -1.0])
        return (
            jnp.abs(state.position[2] - reference.position[2]),
            jnp.abs(
                rot.rotate(rot.conjugate(state.quaternion), gravity)[2]
                - rot.rotate(rot.conjugate(reference.quaternion), gravity)[2]
            ),
        )

    def terminal(self, state, reference):
        height, tilt = self.errors(state, reference)
        return (
            (height > self.config.height_error)
            | (tilt > self.config.tilt_error)
            | ~jnp.all(jnp.isfinite(state.flatten()))
        )

    def soft_termination(self, state, reference):
        height, tilt = self.errors(state, reference)
        alive = jax.nn.sigmoid(
            (self.config.height_error - height) / self.config.soft_height_width
        ) * jax.nn.sigmoid(
            (self.config.tilt_error - tilt) / self.config.soft_tilt_width
        )
        return 1.0 - alive


def critic_features(state, reference, goal, context, previous_motor, history):
    heading_inverse = rot.conjugate(rot.heading(state.quaternion))
    return jnp.concatenate(
        (
            invariant_features(state),
            rot.rotate(heading_inverse, reference.position - state.position),
            rot.rotation_6d(
                rot.multiply(rot.conjugate(state.quaternion), reference.quaternion)
            ),
            reference.joint_position - state.joint_position,
            previous_motor,
            context,
            goal,
            history.reshape(-1),
        )
    )


def lambda_returns(rewards, terminations, values, gamma, lam):
    """Time-major imagined targets; values contains the initial and final state."""

    def backward(carry, inputs):
        reward, done, next_value = inputs
        target = reward + gamma * (1 - done) * ((1 - lam) * next_value + lam * carry)
        return target, target

    _, targets = jax.lax.scan(
        backward, values[-1], (rewards, terminations, values[1:]), reverse=True
    )
    return targets


def imagined_return(rewards, terminations, bootstrap, gamma):
    alive_after = jnp.cumprod(1.0 - terminations, axis=0)
    alive_before = jnp.concatenate(
        (jnp.ones_like(alive_after[:1]), alive_after[:-1]), axis=0
    )
    discounts = gamma ** jnp.arange(rewards.shape[0])
    discounts = discounts.reshape((rewards.shape[0],) + (1,) * (rewards.ndim - 1))
    return (discounts * alive_before * rewards).sum(axis=0) + gamma ** rewards.shape[
        0
    ] * alive_after[-1] * bootstrap
