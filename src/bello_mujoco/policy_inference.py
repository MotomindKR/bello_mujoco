"""Shared checkpoint-compatible policy inference for evaluation and hardware."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from bello_mujoco.policy import apply_actor, sample_actor_action
from bello_mujoco.train.ppo import (
    ObservationNormState,
    PPOConfig,
    normalize_observation,
)


class PolicyInferenceStep(NamedTuple):
    action_mean: jax.Array
    requested_action: jax.Array
    applied_action: jax.Array


def infer_policy_action(
    policy: Any,
    observation: jax.Array,
    normalization: ObservationNormState,
    config: PPOConfig,
    action_clip: float,
    *,
    rng: jax.Array | None = None,
) -> PolicyInferenceStep:
    """Evaluate the actor with the same input and action contract used by PPO."""

    policy_observation = normalize_observation(observation, normalization, config)
    action_mean = apply_actor(policy, policy_observation)
    requested_action = (
        action_mean
        if rng is None
        else sample_actor_action(policy, policy_observation, rng)
    )
    applied_action = jnp.clip(requested_action, -action_clip, action_clip)
    return PolicyInferenceStep(action_mean, requested_action, applied_action)
