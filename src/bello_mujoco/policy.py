"""Feed-forward Gaussian actor-critic policy utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp

LOG_2PI = jnp.log(2.0 * jnp.pi)


@dataclass(frozen=True)
class ActorCriticConfig:
    hidden_size: int = 256
    hidden_depth: int = 2
    activation: Literal["tanh", "relu"] = "tanh"
    init_log_std: float = -0.5


class ActorCritic(eqx.Module):
    actor: eqx.nn.MLP
    critic: eqx.nn.MLP
    log_std: jax.Array

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        config: ActorCriticConfig = ActorCriticConfig(),
        *,
        key: jax.Array,
    ):
        actor_key, critic_key = jax.random.split(key)
        activation_fn = activation_function(config.activation)
        self.actor = eqx.nn.MLP(
            obs_size,
            action_size,
            width_size=config.hidden_size,
            depth=config.hidden_depth,
            activation=activation_fn,
            final_activation=lambda x: x,
            key=actor_key,
        )
        self.critic = eqx.nn.MLP(
            obs_size,
            1,
            width_size=config.hidden_size,
            depth=config.hidden_depth,
            activation=activation_fn,
            final_activation=lambda x: x,
            key=critic_key,
        )
        self.log_std = jnp.full((action_size,), config.init_log_std, dtype=jnp.float32)

    def __call__(self, obs: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        mean = self.actor(obs)
        value = jnp.squeeze(self.critic(obs), axis=-1)
        return mean, self.log_std, value


def init_actor_critic(
    rng: jax.Array,
    obs_size: int,
    action_size: int,
    config: ActorCriticConfig = ActorCriticConfig(),
) -> ActorCritic:
    return ActorCritic(obs_size, action_size, config, key=rng)


def apply_policy(
    policy: ActorCritic, obs: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    if obs.ndim == 1:
        return policy(obs)
    return jax.vmap(policy)(obs)


def sample_action(
    policy: ActorCritic,
    obs: jax.Array,
    rng: jax.Array,
    action_clip: float | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    mean, log_std, value = apply_policy(policy, obs)
    noise = jax.random.normal(rng, mean.shape)
    action = mean + jnp.exp(log_std) * noise
    if action_clip is not None:
        action = jnp.clip(action, -action_clip, action_clip)
    log_prob = gaussian_log_prob(mean, log_std, action)
    return action, log_prob, value


def evaluate_actions(
    policy: ActorCritic, obs: jax.Array, action: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    mean, log_std, value = apply_policy(policy, obs)
    log_prob = gaussian_log_prob(mean, log_std, action)
    entropy = gaussian_entropy(log_std)
    return log_prob, entropy, value


def gaussian_log_prob(
    mean: jax.Array, log_std: jax.Array, action: jax.Array
) -> jax.Array:
    inv_std = jnp.exp(-log_std)
    normalized = (action - mean) * inv_std
    log_probs = -0.5 * (normalized**2 + 2.0 * log_std + LOG_2PI)
    return jnp.sum(log_probs, axis=-1)


def gaussian_entropy(log_std: jax.Array) -> jax.Array:
    entropy = 0.5 * (1.0 + LOG_2PI) + log_std
    return jnp.sum(entropy, axis=-1)


def activation_function(name: str):
    if name == "tanh":
        return jnp.tanh
    if name == "relu":
        return jax.nn.relu
    raise ValueError(f"unsupported activation: {name!r}")
