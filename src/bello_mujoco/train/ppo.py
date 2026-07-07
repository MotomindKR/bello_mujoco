"""PureJaxRL-style PPO for the Bello MJX environment.

This module keeps the training loop end-to-end JAX: rollout collection uses
`lax.scan`, batching uses `vmap`, and optimization uses Optax. It intentionally
does not depend on PureJaxRL as a package; the structure follows the same
single-file implementation style while using Equinox for the actor-critic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from bello_mujoco.envs.locomotion import BelloLocomotionEnv, BelloState

LOG_2PI = jnp.log(2.0 * jnp.pi)


@dataclass(frozen=True)
class PPOConfig:
    seed: int = 0
    total_timesteps: int = 1_000_000
    num_envs: int = 64
    num_steps: int = 128
    update_epochs: int = 4
    num_minibatches: int = 4
    learning_rate: float = 3.0e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_size: int = 256
    hidden_depth: int = 2
    activation: str = "tanh"
    init_log_std: float = -0.5

    @property
    def num_updates(self) -> int:
        return self.total_timesteps // (self.num_envs * self.num_steps)

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches


class ActorCritic(eqx.Module):
    actor: eqx.nn.MLP
    critic: eqx.nn.MLP
    log_std: jax.Array

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden_size: int,
        hidden_depth: int,
        activation: str,
        init_log_std: float,
        *,
        key: jax.Array,
    ):
        actor_key, critic_key = jax.random.split(key)
        activation_fn = jax.nn.relu if activation == "relu" else jnp.tanh
        self.actor = eqx.nn.MLP(
            obs_size,
            action_size,
            width_size=hidden_size,
            depth=hidden_depth,
            activation=activation_fn,
            final_activation=lambda x: x,
            key=actor_key,
        )
        self.critic = eqx.nn.MLP(
            obs_size,
            1,
            width_size=hidden_size,
            depth=hidden_depth,
            activation=activation_fn,
            final_activation=lambda x: x,
            key=critic_key,
        )
        self.log_std = jnp.full((action_size,), init_log_std, dtype=jnp.float32)

    def __call__(self, obs: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        mean = self.actor(obs)
        value = jnp.squeeze(self.critic(obs), axis=-1)
        return mean, self.log_std, value


@chex.dataclass(frozen=True)
class TrainState:
    model: ActorCritic
    opt_state: optax.OptState
    update: jax.Array


class Transition(NamedTuple):
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    info: dict[str, jax.Array]


def init_train_state(
    rng: jax.Array,
    obs_size: int,
    action_size: int,
    config: PPOConfig,
) -> tuple[TrainState, optax.GradientTransformation]:
    model = ActorCritic(
        obs_size,
        action_size,
        config.hidden_size,
        config.hidden_depth,
        config.activation,
        config.init_log_std,
        key=rng,
    )
    optimizer = make_optimizer(config)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    return TrainState(
        model=model, opt_state=opt_state, update=jnp.asarray(0, dtype=jnp.int32)
    ), optimizer


def make_optimizer(config: PPOConfig) -> optax.GradientTransformation:
    if config.anneal_lr:

        def schedule(count: jax.Array) -> jax.Array:
            updates = jnp.maximum(
                config.num_updates * config.update_epochs * config.num_minibatches, 1
            )
            frac = 1.0 - count / updates
            return config.learning_rate * jnp.maximum(frac, 0.0)

        learning_rate = schedule
    else:
        learning_rate = config.learning_rate

    return optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=learning_rate, eps=1.0e-5),
    )


def apply_model(
    model: ActorCritic, obs: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    if obs.ndim == 1:
        return model(obs)
    return jax.vmap(model)(obs)


def sample_action(
    model: ActorCritic, obs: jax.Array, rng: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    mean, log_std, value = apply_model(model, obs)
    noise = jax.random.normal(rng, mean.shape)
    action = mean + jnp.exp(log_std) * noise
    log_prob = gaussian_log_prob(mean, log_std, action)
    return action, log_prob, value


def evaluate_actions(
    model: ActorCritic, obs: jax.Array, action: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    mean, log_std, value = apply_model(model, obs)
    log_prob = gaussian_log_prob(mean, log_std, action)
    entropy = jnp.ones_like(log_prob) * gaussian_entropy(log_std)
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
    return jnp.sum(entropy)


def compute_gae(
    transitions: Transition,
    last_value: jax.Array,
    config: PPOConfig,
) -> tuple[jax.Array, jax.Array]:
    def scan_step(
        carry: tuple[jax.Array, jax.Array], transition: Transition
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        gae, next_value = carry
        not_done = 1.0 - transition.done.astype(jnp.float32)
        delta = (
            transition.reward + config.gamma * next_value * not_done - transition.value
        )
        gae = delta + config.gamma * config.gae_lambda * not_done * gae
        return (gae, transition.value), gae

    _, advantages = jax.lax.scan(
        scan_step,
        (jnp.zeros_like(last_value), last_value),
        transitions,
        reverse=True,
    )
    return advantages, advantages + transitions.value


def ppo_loss(
    model: ActorCritic,
    transitions: Transition,
    advantages: jax.Array,
    targets: jax.Array,
    config: PPOConfig,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    log_prob, entropy, value = evaluate_actions(
        model, transitions.obs, transitions.action
    )
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)

    ratio = jnp.exp(log_prob - transitions.log_prob)
    actor_loss_1 = ratio * advantages
    actor_loss_2 = (
        jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * advantages
    )
    actor_loss = -jnp.minimum(actor_loss_1, actor_loss_2).mean()

    value_clipped = transitions.value + jnp.clip(
        value - transitions.value, -config.clip_eps, config.clip_eps
    )
    value_loss = (
        0.5 * jnp.maximum((value - targets) ** 2, (value_clipped - targets) ** 2).mean()
    )
    entropy_loss = entropy.mean()
    total_loss = (
        actor_loss + config.vf_coef * value_loss - config.ent_coef * entropy_loss
    )
    metrics = {
        "loss": total_loss,
        "actor_loss": actor_loss,
        "value_loss": value_loss,
        "entropy": entropy_loss,
        "approx_kl": (transitions.log_prob - log_prob).mean(),
    }
    return total_loss, metrics


def update_ppo(
    train_state: TrainState,
    optimizer: optax.GradientTransformation,
    transitions: Transition,
    advantages: jax.Array,
    targets: jax.Array,
    rng: jax.Array,
    config: PPOConfig,
) -> tuple[TrainState, dict[str, jax.Array]]:
    batch = flatten_batch((transitions, advantages, targets))
    init_params, static_model = eqx.partition(train_state.model, eqx.is_array)

    def update_epoch(carry, _: None):
        params, opt_state, update_count, key = carry
        key, perm_key = jax.random.split(key)
        permutation = jax.random.permutation(perm_key, config.batch_size)
        shuffled = jax.tree_util.tree_map(lambda x: x[permutation], batch)
        minibatches = jax.tree_util.tree_map(
            lambda x: x.reshape(
                (config.num_minibatches, config.minibatch_size) + x.shape[1:]
            ),
            shuffled,
        )

        def update_minibatch(carry, minibatch: tuple[Transition, jax.Array, jax.Array]):
            params, opt_state, update_count = carry
            mb_transitions, mb_advantages, mb_targets = minibatch

            def loss_fn(params):
                model = eqx.combine(params, static_model)
                return ppo_loss(
                    model, mb_transitions, mb_advantages, mb_targets, config
                )

            (loss_value, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params
            )
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            metrics = {**metrics, "loss": loss_value}
            return (params, opt_state, update_count + 1), metrics

        (params, opt_state, update_count), metrics = jax.lax.scan(
            update_minibatch,
            (params, opt_state, update_count),
            minibatches,
        )
        return (params, opt_state, update_count, key), metrics

    (params, opt_state, update_count, _), metrics = jax.lax.scan(
        update_epoch,
        (init_params, train_state.opt_state, train_state.update, rng),
        None,
        config.update_epochs,
    )
    metrics = jax.tree_util.tree_map(lambda x: x.mean(), metrics)
    train_state = TrainState(
        model=eqx.combine(params, static_model),
        opt_state=opt_state,
        update=update_count,
    )
    return train_state, metrics


def flatten_batch(batch):
    return jax.tree_util.tree_map(
        lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:]), batch
    )


def safe_returned_mean(info: dict[str, jax.Array]) -> jax.Array:
    episode_return = info.get("episode_return")
    returned_episode = info.get("returned_episode")
    if episode_return is None or returned_episode is None:
        return jnp.asarray(0.0, dtype=jnp.float32)
    mask = returned_episode.astype(jnp.float32)
    return jnp.sum(episode_return * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def collect_rollout(
    env: BelloLocomotionEnv,
    model: ActorCritic,
    env_state: BelloState,
    obs: jax.Array,
    rng: jax.Array,
    config: PPOConfig,
) -> tuple[tuple[BelloState, jax.Array, jax.Array], Transition]:
    def env_step(
        carry: tuple[BelloState, jax.Array, jax.Array], _: None
    ) -> tuple[tuple[BelloState, jax.Array, jax.Array], Transition]:
        state, last_obs, key = carry
        key, action_key = jax.random.split(key)
        action, log_prob, value = sample_action(model, last_obs, action_key)
        next_state, next_obs, reward, done, info = env.step_batch(state, action)
        transition = Transition(done, action, value, reward, log_prob, last_obs, info)
        return (next_state, next_obs, key), transition

    return jax.lax.scan(env_step, (env_state, obs, rng), None, config.num_steps)


def make_train(config: PPOConfig, env: BelloLocomotionEnv | None = None):
    if config.batch_size % config.num_minibatches != 0:
        raise ValueError("num_envs * num_steps must be divisible by num_minibatches")
    env = BelloLocomotionEnv.load(batch_size=config.num_envs) if env is None else env

    def train(rng: jax.Array):
        rng, init_key = jax.random.split(rng)
        train_state, optimizer = init_train_state(
            init_key, env.observation_size, env.action_size, config
        )
        init_params, static_model = eqx.partition(train_state.model, eqx.is_array)
        env_state, obs = env.reset_batch(config.num_envs)

        def update_step(carry, _: None):
            params, opt_state, update_count, e_state, last_obs, key = carry
            key, rollout_key, update_key = jax.random.split(key, 3)
            model = eqx.combine(params, static_model)
            state = TrainState(model=model, opt_state=opt_state, update=update_count)
            (e_state, last_obs, _), transitions = collect_rollout(
                env, model, e_state, last_obs, rollout_key, config
            )
            _, _, last_value = apply_model(model, last_obs)
            advantages, targets = compute_gae(transitions, last_value, config)
            state, metrics = update_ppo(
                state, optimizer, transitions, advantages, targets, update_key, config
            )
            params, _ = eqx.partition(state.model, eqx.is_array)
            metrics = {
                **metrics,
                "mean_reward": transitions.reward.mean(),
                "mean_done": transitions.done.astype(jnp.float32).mean(),
                "mean_episode_return": safe_returned_mean(transitions.info),
                "max_episode_return": transitions.info.get(
                    "episode_return", jnp.zeros_like(transitions.reward)
                ).max(),
            }
            return (
                params,
                state.opt_state,
                state.update,
                e_state,
                last_obs,
                key,
            ), metrics

        runner_state = (
            init_params,
            train_state.opt_state,
            train_state.update,
            env_state,
            obs,
            rng,
        )
        runner_state, metrics = jax.lax.scan(
            update_step, runner_state, None, config.num_updates
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train
