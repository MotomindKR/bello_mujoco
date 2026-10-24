"""The same causal feedback recurrence is used by collection and imagination."""

from dataclasses import replace

import jax
import jax.numpy as jnp

from bello_mujoco.envs.feedback import feedback_filter_alpha
from bello_mujoco.supermimic import rotations as rot


def measured_feedback(state, key, noise, filtering):
    if key is None or not noise.enabled or not filtering.noise_before_filter:
        return state
    keys = jax.random.split(key, 4)

    def uniform(k, shape, magnitude):
        return jax.random.uniform(k, shape, minval=-magnitude, maxval=magnitude)

    angular = rot.rotate(rot.conjugate(state.quaternion), state.angular_velocity)
    return replace(
        state,
        linear_velocity=state.linear_velocity
        + rot.rotate(
            state.quaternion, uniform(keys[0], (3,), noise.base_linear_velocity)
        ),
        angular_velocity=rot.rotate(
            state.quaternion,
            angular + uniform(keys[1], (3,), noise.base_angular_velocity),
        ),
        joint_position=state.joint_position
        + uniform(keys[2], state.joint_position.shape, noise.joint_position),
        joint_velocity=state.joint_velocity
        + uniform(keys[3], state.joint_velocity.shape, noise.joint_velocity),
    )


def advance_feedback(history, state, key, *, noise, filtering, timestep):
    measured = measured_feedback(state, key, noise, filtering)
    if filtering.enabled:
        old = jax.tree.map(lambda x: x[0], history)
        alpha = feedback_filter_alpha(timestep, filtering.cutoff_hz)
        delta = rot.log(
            rot.multiply(rot.conjugate(old.quaternion), measured.quaternion)
        )
        orientation = rot.normalize(
            rot.multiply(old.quaternion, rot.exp(alpha * delta))
        )
        old_angular = rot.rotate(rot.conjugate(old.quaternion), old.angular_velocity)
        angular = rot.rotate(
            rot.conjugate(measured.quaternion), measured.angular_velocity
        )
        measured = replace(
            measured,
            quaternion=orientation,
            linear_velocity=old.linear_velocity
            + alpha * (measured.linear_velocity - old.linear_velocity),
            angular_velocity=rot.rotate(
                orientation, old_angular + alpha * (angular - old_angular)
            ),
            joint_velocity=old.joint_velocity
            + alpha * (measured.joint_velocity - old.joint_velocity),
        )
    return jax.tree.map(
        lambda new, queue: jnp.concatenate((new[None], queue[:-1])), measured, history
    )


def actor_sample(
    codec,
    feedback,
    previous_action,
    reference_quaternion,
    calibration,
    key,
    *,
    noise,
    filtering,
):
    sample = codec.sample(feedback, previous_action, reference_quaternion, calibration)
    return add_actor_noise(
        sample,
        key,
        joint_count=previous_action.shape[0],
        reference_orientation=codec.config.reference_orientation,
        noise=noise,
        filtering=filtering,
    )


def add_actor_noise(
    sample, key, *, joint_count, reference_orientation, noise, filtering
):
    if key is None or not noise.enabled:
        return sample
    scale = filtering.residual_noise_scale if filtering.noise_before_filter else 1.0
    n = joint_count
    magnitude = jnp.concatenate(
        (
            jnp.full(n, noise.joint_position * scale),
            jnp.full(n, noise.joint_velocity * scale),
            jnp.full(3, noise.base_angular_velocity * scale),
            jnp.full(3, noise.anchor_orientation),
            jnp.zeros(n),
            jnp.full(6 if reference_orientation else 0, noise.anchor_orientation),
        )
    )
    return sample + jax.random.uniform(
        key, sample.shape, minval=-magnitude, maxval=magnitude
    )
