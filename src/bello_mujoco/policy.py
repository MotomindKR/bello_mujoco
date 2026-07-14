"""Feed-forward Gaussian actor-critic policy utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp

LOG_2PI = jnp.log(2.0 * jnp.pi)


def identity_activation(x: jax.Array) -> jax.Array:
    return x


@dataclass(frozen=True)
class ActorCriticConfig:
    hidden_size: int = 256
    hidden_depth: int = 2
    hidden_dims: tuple[int, ...] | None = None
    activation: Literal["tanh", "relu", "elu"] = "tanh"
    init_log_std: float = 0.0
    min_log_std: float = -5.0
    max_log_std: float = 1.0
    obs_stack_size: int = 10
    action_distribution: Literal["gaussian", "tanh_gaussian"] = "gaussian"
    action_limit: float = 1.0


class LayeredMLP(eqx.Module):
    layers: tuple[eqx.nn.Linear, ...]
    activation: Callable[[jax.Array], jax.Array] = eqx.field(static=True)
    final_activation: Callable[[jax.Array], jax.Array] = eqx.field(static=True)

    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_dims: tuple[int, ...],
        *,
        activation: Callable[[jax.Array], jax.Array],
        final_activation: Callable[[jax.Array], jax.Array],
        key: jax.Array,
    ):
        dims = (in_size, *hidden_dims, out_size)
        keys = jax.random.split(key, len(dims) - 1)
        self.layers = tuple(
            eqx.nn.Linear(input_size, output_size, key=layer_key)
            for input_size, output_size, layer_key in zip(
                dims[:-1], dims[1:], keys, strict=True
            )
        )
        self.activation = activation
        self.final_activation = final_activation

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.final_activation(self.layers[-1](x))


class ActorCritic(eqx.Module):
    actor: eqx.Module
    critic: eqx.Module
    log_std: jax.Array
    min_log_std: float = eqx.field(static=True)
    max_log_std: float = eqx.field(static=True)
    action_distribution: str = eqx.field(static=True)
    action_limit: float = eqx.field(static=True)

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        config: ActorCriticConfig = ActorCriticConfig(),
        *,
        critic_obs_size: int | None = None,
        key: jax.Array,
    ):
        actor_key, critic_key = jax.random.split(key)
        critic_obs_size = obs_size if critic_obs_size is None else critic_obs_size
        activation_fn = activation_function(config.activation)
        self.actor = make_mlp(
            obs_size=obs_size,
            output_size=action_size,
            config=config,
            activation=activation_fn,
            key=actor_key,
        )
        self.critic = make_mlp(
            obs_size=critic_obs_size,
            output_size=1,
            config=config,
            activation=activation_fn,
            key=critic_key,
        )
        self.log_std = jnp.full((action_size,), config.init_log_std, dtype=jnp.float32)
        self.min_log_std = config.min_log_std
        self.max_log_std = config.max_log_std
        self.action_distribution = config.action_distribution
        self.action_limit = config.action_limit

    def __call__(
        self, obs: jax.Array, critic_obs: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        critic_obs = obs if critic_obs is None else critic_obs
        mean = self.actor(obs)
        value = jnp.squeeze(self.critic(critic_obs), axis=-1)
        return mean, policy_log_std(self), value

    def action_mean(self, obs: jax.Array) -> jax.Array:
        return transform_action(self, self.actor(obs))


def init_actor_critic(
    rng: jax.Array,
    obs_size: int,
    action_size: int,
    config: ActorCriticConfig = ActorCriticConfig(),
    *,
    critic_obs_size: int | None = None,
) -> ActorCritic:
    return ActorCritic(
        obs_size, action_size, config, critic_obs_size=critic_obs_size, key=rng
    )


def make_mlp(
    *,
    obs_size: int,
    output_size: int,
    config: ActorCriticConfig,
    activation: Callable[[jax.Array], jax.Array],
    key: jax.Array,
) -> eqx.Module:
    if config.hidden_dims is None:
        return eqx.nn.MLP(
            obs_size,
            output_size,
            width_size=config.hidden_size,
            depth=config.hidden_depth,
            activation=activation,
            final_activation=identity_activation,
            key=key,
        )
    return LayeredMLP(
        obs_size,
        output_size,
        tuple(config.hidden_dims),
        activation=activation,
        final_activation=identity_activation,
        key=key,
    )


def policy_log_std(policy: ActorCritic) -> jax.Array:
    min_log_std = getattr(policy, "min_log_std", -jnp.inf)
    max_log_std = getattr(policy, "max_log_std", jnp.inf)
    return jnp.clip(policy.log_std, min_log_std, max_log_std)


def clamp_policy_log_std(policy: ActorCritic) -> ActorCritic:
    return eqx.tree_at(lambda model: model.log_std, policy, policy_log_std(policy))


def policy_observation_size(obs_size: int, config: ActorCriticConfig) -> int:
    return obs_size * config.obs_stack_size


def init_observation_stack(obs: jax.Array, stack_size: int) -> jax.Array:
    obs = jnp.asarray(obs)
    return jnp.repeat(obs[..., None, :], stack_size, axis=-2)


def stacked_observation(obs_stack: jax.Array) -> jax.Array:
    return obs_stack.reshape(obs_stack.shape[:-2] + (-1,))


def update_observation_stack(
    obs_stack: jax.Array,
    obs: jax.Array,
    done: jax.Array,
) -> jax.Array:
    next_stack = jnp.concatenate((obs_stack[..., 1:, :], obs[..., None, :]), axis=-2)
    reset_stack = init_observation_stack(obs, obs_stack.shape[-2])
    done = done.astype(bool)
    while done.ndim < obs_stack.ndim:
        done = done[..., None]
    return jnp.where(done, reset_stack, next_stack)


def apply_policy(
    policy: ActorCritic,
    obs: jax.Array,
    critic_obs: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    if critic_obs is None:
        critic_obs = obs
    if obs.ndim == 1:
        return policy(obs, critic_obs)
    return jax.vmap(policy)(obs, critic_obs)


def apply_actor(policy: ActorCritic, obs: jax.Array) -> jax.Array:
    if obs.ndim == 1:
        return policy.action_mean(obs)
    return jax.vmap(policy.action_mean)(obs)


def sample_action(
    policy: ActorCritic,
    obs: jax.Array,
    rng: jax.Array,
    action_clip: float | None = None,
    critic_obs: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    action, log_prob, value, _ = sample_action_with_mean(
        policy, obs, rng, action_clip=action_clip, critic_obs=critic_obs
    )
    return action, log_prob, value


def sample_action_with_mean(
    policy: ActorCritic,
    obs: jax.Array,
    rng: jax.Array,
    action_clip: float | None = None,
    critic_obs: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    latent_mean, log_std, value = apply_policy(policy, obs, critic_obs)
    noise = jax.random.normal(rng, latent_mean.shape)
    latent_action = latent_mean + jnp.exp(log_std) * noise
    action = transform_action(policy, latent_action)
    if action_clip is not None:
        action = jnp.clip(action, -action_clip, action_clip)
    log_prob = policy_log_prob(
        policy,
        latent_mean,
        log_std,
        action,
        latent_action if action_clip is None else None,
    )
    return action, log_prob, value, transform_action(policy, latent_mean)


def evaluate_actions(
    policy: ActorCritic,
    obs: jax.Array,
    action: jax.Array,
    critic_obs: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    latent_mean, log_std, value = apply_policy(policy, obs, critic_obs)
    log_prob = policy_log_prob(policy, latent_mean, log_std, action)
    entropy = gaussian_entropy(log_std)
    return log_prob, entropy, value


def transform_action(policy: ActorCritic, latent_action: jax.Array) -> jax.Array:
    if getattr(policy, "action_distribution", "gaussian") == "tanh_gaussian":
        return getattr(policy, "action_limit", 1.0) * jnp.tanh(latent_action)
    return latent_action


def policy_log_prob(
    policy: ActorCritic,
    latent_mean: jax.Array,
    log_std: jax.Array,
    action: jax.Array,
    latent_action: jax.Array | None = None,
) -> jax.Array:
    if getattr(policy, "action_distribution", "gaussian") != "tanh_gaussian":
        return gaussian_log_prob(latent_mean, log_std, action)
    limit = jnp.asarray(getattr(policy, "action_limit", 1.0), dtype=action.dtype)
    normalized_action = jnp.clip(action / limit, -1.0 + 1.0e-6, 1.0 - 1.0e-6)
    if latent_action is None:
        latent_action = jnp.arctanh(normalized_action)
    base_log_prob = gaussian_log_prob(latent_mean, log_std, latent_action)
    log_det = jnp.log(limit) + 2.0 * (
        jnp.log(2.0) - latent_action - jax.nn.softplus(-2.0 * latent_action)
    )
    return base_log_prob - jnp.sum(log_det, axis=-1)


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
    if name == "elu":
        return jax.nn.elu
    raise ValueError(f"unsupported activation: {name!r}")
