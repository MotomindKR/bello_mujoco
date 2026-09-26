"""Observation assembly and sensor filtering, independent of the simulation backend."""

import jax
import jax.numpy as jnp

from bello_mujoco.envs.locomotion_config import (
    ObservationConfig,
    ObservationNoiseConfig,
)
from bello_mujoco.envs.feedback import (
    ObservationLowPassFilterConfig,
    feedback_filter_alpha,
)


def unpack_feedback(feedback: jax.Array) -> tuple[jax.Array, ...]:
    joints = (feedback.shape[-1] - 9) // 2
    return tuple(jnp.split(feedback, (3, 6, 9, 9 + joints), axis=-1))


def add_uniform_noise(value, bounds, rng):
    return value + jax.random.uniform(
        rng, value.shape, minval=bounds[0], maxval=bounds[1], dtype=value.dtype
    )


def noisy_feedback(measured, noise: ObservationNoiseConfig, rng):
    if rng is None or not noise.enabled:
        return measured
    bounds = (
        noise.base_lin_acc,
        noise.base_ang_vel,
        noise.projected_gravity,
        noise.joint_pos,
        noise.joint_vel,
    )
    return jnp.concatenate(
        [
            add_uniform_noise(value, interval, key)
            for value, interval, key in zip(
                unpack_feedback(measured), bounds, jax.random.split(rng, 5), strict=True
            )
        ]
    )


def filter_feedback(
    previous, measured, timestep, config: ObservationLowPassFilterConfig
):
    if not config.enabled:
        return measured
    alpha = feedback_filter_alpha(timestep, config.cutoff_hz)
    filtered = list(unpack_feedback(previous + alpha * (measured - previous)))
    raw = unpack_feedback(measured)
    norm = jnp.linalg.norm(filtered[2])
    filtered[2] = jnp.where(
        norm > jnp.finfo(measured.dtype).eps, filtered[2] / norm, raw[2]
    )
    filtered[3] = raw[3]
    return jnp.concatenate(filtered)


def advance_feedback_history(history, measured, rng, timestep, noise, filtering):
    """Pre-filter noise and retain newest-first feedback, without delaying commands."""
    if filtering.noise_before_filter:
        measured = noisy_feedback(measured, noise, rng)
    feedback = filter_feedback(history[0], measured, timestep, filtering)
    return jnp.concatenate((feedback[None, :], history[:-1]))


def sample_feedback_delay(config, rng):
    if not config.enabled or rng is None:
        return jnp.asarray(0, dtype=jnp.int32)
    return jax.random.randint(
        rng,
        (),
        minval=config.delay_steps[0],
        maxval=config.delay_steps[1] + 1,
        dtype=jnp.int32,
    )


def build_observation(
    terms,
    gait_features,
    config: ObservationConfig,
    rng,
    feedback=None,
    *,
    torso_command,
    noise_before_filter=False,
    residual_noise_scale=0.0,
):
    (
        linear_velocity,
        linear_acceleration,
        angular_velocity,
        gravity,
        height,
        command,
        position,
        velocity,
        force,
        action,
    ) = terms
    if feedback is not None:
        linear_acceleration, angular_velocity, gravity, position, velocity = (
            unpack_feedback(feedback)
        )
    filtered = feedback is not None and noise_before_filter
    feedback_noise = residual_noise_scale if filtered else 1.0
    components = [
        (linear_acceleration, "base_lin_acc", 0, feedback_noise),
        (angular_velocity, "base_ang_vel", 1, feedback_noise),
        (gravity, "projected_gravity", 2, feedback_noise),
    ]
    if gait_features is not None:
        components.extend(
            [
                (command, "command", 3, 1.0),
                (torso_command, "command", 3, 0.0),
                (gait_features, "gait_phase", 4, 1.0),
            ]
        )
    components.extend(
        [
            (position, "joint_pos", 5, 0.0 if filtered else 1.0),
            (velocity, "joint_vel", 6, feedback_noise),
            (action, "last_action", 7, 1.0),
        ]
    )
    if config.privileged:
        components.extend(
            [
                (linear_velocity, "base_lin_vel", 8, 1.0),
                (height, "base_height", 9, 1.0),
                (force, "actuator_force", 10, 1.0),
            ]
        )
    keys = (
        jax.random.split(rng, 11)
        if rng is not None and config.noise.enabled
        else (None,) * 11
    )
    values = []
    for value, name, key_index, amplitude in components:
        if keys[key_index] is not None and amplitude > 0.0:
            interval = tuple(amplitude * x for x in getattr(config.noise, name))
            value = add_uniform_noise(value, interval, keys[key_index])
        values.append(
            value * jnp.asarray(getattr(config.scale, name), dtype=value.dtype)
        )
    return jnp.concatenate(values)
