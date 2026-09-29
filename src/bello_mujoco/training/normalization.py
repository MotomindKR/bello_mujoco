from __future__ import annotations
import jax
import jax.numpy as jnp
from bello_mujoco.training.ppo_config import PPOConfig
from bello_mujoco.training.ppo_types import ObservationNormState, ValueNormState


def init_value_norm_state() -> ValueNormState:
    return ValueNormState(
        mean=jnp.asarray(0.0, dtype=jnp.float32),
        var=jnp.asarray(1.0, dtype=jnp.float32),
        count=jnp.asarray(0.0, dtype=jnp.float32),
    )


def init_observation_norm_state(size: int) -> ObservationNormState:
    return ObservationNormState(
        mean=jnp.zeros((size,), dtype=jnp.float32),
        var=jnp.ones((size,), dtype=jnp.float32),
        count=jnp.asarray(0.0, dtype=jnp.float32),
    )


def normalize_observation(
    observation: jax.Array,
    state: ObservationNormState,
    config: PPOConfig,
) -> jax.Array:
    if not config.normalize_observations:
        return observation
    if config.observation_norm_epsilon_mode == "std_offset":
        denominator = jnp.sqrt(jnp.maximum(state.var, 1.0e-8))
        denominator = denominator + config.observation_norm_eps
    else:
        denominator = jnp.sqrt(jnp.maximum(state.var, config.observation_norm_eps))
    normalized = (observation - state.mean) / denominator
    clipped = jnp.clip(
        normalized,
        -config.observation_norm_clip,
        config.observation_norm_clip,
    )
    valid = array_is_plausible(observation, config.rollout_observation_max_abs)
    return jnp.where(valid, clipped, observation)


def normalize_critic_observation(
    observation: jax.Array,
    state: ObservationNormState,
    config: PPOConfig,
) -> jax.Array:
    if not config.normalize_critic_observations:
        return observation
    if config.critic_normalization_mode == "any2any":
        return jnp.clip(
            (observation - state.mean) / jnp.sqrt(state.var + 1.0e-5), -5.0, 5.0
        )
    return normalize_observation(observation, state, config)


def update_observation_norm_state(
    state: ObservationNormState,
    observation: jax.Array,
    config: PPOConfig,
    *,
    max_count: float | None = None,
) -> ObservationNormState:
    if not config.normalize_observations:
        return state
    valid = array_is_plausible(observation, config.rollout_observation_max_abs)
    safe_observation = sanitize_array(observation, config.rollout_observation_max_abs)
    values = safe_observation.reshape((-1, safe_observation.shape[-1]))
    updated = _merge_moments(
        state,
        jnp.asarray(values.shape[0], dtype=jnp.float32),
        jnp.mean(values, axis=0),
        jnp.var(values, axis=0),
        valid,
    )
    if max_count is None:
        return updated
    should_update = state.count < jnp.asarray(max_count, dtype=state.count.dtype)
    return jax.tree.map(
        lambda new, old: jnp.where(should_update, new, old), updated, state
    )


def update_critic_observation_norm_state(
    state: ObservationNormState,
    observation: jax.Array,
    config: PPOConfig,
    *,
    during_optimization: bool = False,
) -> ObservationNormState:
    if not config.normalize_critic_observations:
        return state
    if config.critic_normalization_mode == "any2any":
        if not during_optimization:
            return state
        values = observation.reshape((-1, observation.shape[-1]))
        return _merge_moments(
            state,
            jnp.asarray(values.shape[0], dtype=jnp.float32),
            values.mean(axis=0),
            values.var(axis=0, ddof=1),
            array_is_plausible(values, config.rollout_observation_max_abs),
        )
    return update_observation_norm_state(state, observation, config)


def value_norm_std(state: ValueNormState, config: PPOConfig) -> jax.Array:
    return jnp.sqrt(jnp.maximum(state.var, config.value_norm_eps))


def normalize_value(
    value: jax.Array,
    state: ValueNormState,
    config: PPOConfig,
) -> jax.Array:
    if not config.normalize_value:
        return value
    return (value - state.mean) / value_norm_std(state, config)


def denormalize_value(
    value: jax.Array,
    state: ValueNormState,
    config: PPOConfig,
) -> jax.Array:
    if not config.normalize_value:
        return value
    return value * value_norm_std(state, config) + state.mean


def update_value_norm_state(
    state: ValueNormState,
    targets: jax.Array,
    config: PPOConfig,
) -> ValueNormState:
    if not config.normalize_value:
        return state
    valid = array_is_plausible(targets, config.target_max_abs)
    safe_targets = sanitize_array(targets, config.target_max_abs)
    return _merge_moments(
        state,
        jnp.asarray(safe_targets.size, dtype=jnp.float32),
        jnp.mean(safe_targets),
        jnp.var(safe_targets),
        valid,
    )


def _merge_moments(state, batch_count, batch_mean, batch_var, valid):
    """Parallel population-moment update; invalid batches leave state unchanged."""
    delta = batch_mean - state.mean
    total_count = state.count + batch_count
    mean = state.mean + delta * batch_count / total_count
    state_m2 = state.var * state.count
    batch_m2 = batch_var * batch_count
    correction = delta**2 * state.count * batch_count / total_count
    var = (state_m2 + batch_m2 + correction) / total_count
    return state._replace(
        mean=jnp.where(valid, mean, state.mean),
        var=jnp.where(valid, var, state.var),
        count=jnp.where(valid, total_count, state.count),
    )


def array_is_plausible(value: jax.Array, max_abs: float) -> jax.Array:
    return jnp.all(jnp.isfinite(value)) & jnp.all(jnp.abs(value) <= max_abs)


def sanitize_array(value: jax.Array, max_abs: float) -> jax.Array:
    return jnp.clip(
        jnp.nan_to_num(value, nan=0.0, posinf=max_abs, neginf=-max_abs),
        -max_abs,
        max_abs,
    )
