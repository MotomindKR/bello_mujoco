"""PureJaxRL-style PPO for the Bello MJX environment.

This module keeps the training loop end-to-end JAX: rollout collection uses
`lax.scan`, batching uses `vmap`, and optimization uses Optax. It intentionally
does not depend on PureJaxRL as a package; the structure follows the same
single-file implementation style while using Equinox for the actor-critic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from bello_mujoco.envs.locomotion import BelloLocomotionEnv, BelloState
from bello_mujoco.policy import (
    ActorCritic,
    ActorCriticConfig,
    apply_policy,
    evaluate_actions,
    init_observation_stack,
    init_actor_critic,
    policy_observation_size,
    sample_action_with_mean,
    stacked_observation,
    update_observation_stack,
)


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
    policy: ActorCriticConfig = ActorCriticConfig()

    @property
    def num_updates(self) -> int:
        return self.total_timesteps // (self.num_envs * self.num_steps)

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches


@chex.dataclass(frozen=True)
class TrainState:
    policy: ActorCritic
    opt_state: optax.OptState
    update: jax.Array


class Transition(NamedTuple):
    done: jax.Array
    action: jax.Array
    action_mean: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    info: dict[str, jax.Array]


class RunnerState(NamedTuple):
    params: eqx.Module
    opt_state: optax.OptState
    update: jax.Array
    env_state: BelloState
    obs_stack: jax.Array
    rng: jax.Array


def init_train_state(
    rng: jax.Array,
    obs_size: int,
    action_size: int,
    config: PPOConfig,
) -> tuple[TrainState, optax.GradientTransformation]:
    model = init_actor_critic(rng, obs_size, action_size, config.policy)
    optimizer = make_optimizer(config)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    return TrainState(
        policy=model, opt_state=opt_state, update=jnp.asarray(0, dtype=jnp.int32)
    ), optimizer


def make_optimizer(config: PPOConfig) -> optax.GradientTransformation:
    validate_ppo_config(config)
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
    init_params, static_policy = eqx.partition(train_state.policy, eqx.is_array)

    def update_epoch(carry, _: None):
        params, opt_state, key = carry
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
            params, opt_state = carry
            mb_transitions, mb_advantages, mb_targets = minibatch

            def loss_fn(params):
                model = eqx.combine(params, static_policy)
                return ppo_loss(
                    model, mb_transitions, mb_advantages, mb_targets, config
                )

            (loss_value, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params
            )
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            metrics = {**metrics, "loss": loss_value}
            return (params, opt_state), metrics

        (params, opt_state), metrics = jax.lax.scan(
            update_minibatch,
            (params, opt_state),
            minibatches,
        )
        return (params, opt_state, key), metrics

    (params, opt_state, _), metrics = jax.lax.scan(
        update_epoch,
        (init_params, train_state.opt_state, rng),
        None,
        config.update_epochs,
    )
    metrics = jax.tree_util.tree_map(lambda x: x.mean(), metrics)
    train_state = TrainState(
        policy=eqx.combine(params, static_policy),
        opt_state=opt_state,
        update=train_state.update + 1,
    )
    return train_state, metrics


def flatten_batch(batch):
    return jax.tree_util.tree_map(
        lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:]), batch
    )


def safe_returned_mean(
    info: dict[str, jax.Array], key: str = "episode_return"
) -> jax.Array:
    value = info.get(key)
    returned_episode = info.get("returned_episode")
    if value is None or returned_episode is None:
        return jnp.asarray(0.0, dtype=jnp.float32)
    mask = returned_episode.astype(jnp.float32)
    return jnp.sum(value.astype(jnp.float32) * mask) / jnp.maximum(
        jnp.sum(mask), 1.0
    )


def safe_returned_max(
    info: dict[str, jax.Array], key: str = "episode_return"
) -> jax.Array:
    value = info.get(key)
    returned_episode = info.get("returned_episode")
    if value is None or returned_episode is None:
        return jnp.asarray(jnp.nan, dtype=jnp.float32)
    mask = returned_episode.astype(bool)
    masked_value = jnp.where(mask, value.astype(jnp.float32), -jnp.inf)
    return jnp.where(jnp.any(mask), jnp.max(masked_value), jnp.nan)


def reward_component_means(info: dict[str, jax.Array]) -> dict[str, jax.Array]:
    return {
        f"mean_{name}": value.mean()
        for name, value in info.items()
        if name.startswith("reward/")
    }


def done_reason_means(info: dict[str, jax.Array]) -> dict[str, jax.Array]:
    return {
        f"mean_{name}": value.astype(jnp.float32).mean()
        for name, value in info.items()
        if name.startswith("done/")
    }


def action_diagnostics(
    transitions: Transition,
    env: BelloLocomotionEnv,
    policy: ActorCritic,
) -> dict[str, jax.Array]:
    action = transitions.action
    action_mean = transitions.action_mean
    metrics = {
        "action_rms": jnp.sqrt(jnp.mean(action**2)),
        "action_abs_max": jnp.max(jnp.abs(action)),
        "action_mean_rms": jnp.sqrt(jnp.mean(action_mean**2)),
        "action_mean_abs_max": jnp.max(jnp.abs(action_mean)),
        "action_std_mean": jnp.mean(jnp.exp(policy.log_std)),
        "action_std_min": jnp.min(jnp.exp(policy.log_std)),
        "action_std_max": jnp.max(jnp.exp(policy.log_std)),
    }
    if not hasattr(env, "_action_to_ctrl"):
        return metrics

    flat_action = action.reshape((-1, env.action_size))
    ctrl = jax.vmap(env._action_to_ctrl)(flat_action)
    policy_ctrl_indices = jnp.asarray(env.model.policy_ctrl_indices)
    policy_ctrl = ctrl[:, policy_ctrl_indices]
    ctrlrange = jnp.asarray(env.model.mj_model.actuator_ctrlrange)[
        policy_ctrl_indices
    ]
    ctrl_low = ctrlrange[:, 0]
    ctrl_high = ctrlrange[:, 1]
    ctrl_margin = jnp.minimum(policy_ctrl - ctrl_low, ctrl_high - policy_ctrl)
    saturated = (policy_ctrl <= ctrl_low + 1.0e-6) | (
        policy_ctrl >= ctrl_high - 1.0e-6
    )
    return {
        **metrics,
        "ctrl_saturation_fraction": saturated.astype(jnp.float32).mean(),
        "ctrl_margin_min": jnp.min(ctrl_margin),
    }


def collect_rollout(
    env: BelloLocomotionEnv,
    policy: ActorCritic,
    env_state: BelloState,
    obs_stack: jax.Array,
    rng: jax.Array,
    config: PPOConfig,
) -> tuple[tuple[BelloState, jax.Array, jax.Array], Transition]:
    def env_step(
        carry: tuple[BelloState, jax.Array, jax.Array], _: None
    ) -> tuple[tuple[BelloState, jax.Array, jax.Array], Transition]:
        state, last_obs_stack, key = carry
        key, action_key, env_key = jax.random.split(key, 3)
        policy_obs = stacked_observation(last_obs_stack)
        action, log_prob, value, action_mean = sample_action_with_mean(
            policy,
            policy_obs,
            action_key,
            action_clip=getattr(env, "action_clip", None),
        )
        next_state, next_obs, reward, done, info = env.step_batch(
            state, action, rng=env_key
        )
        next_obs_stack = update_observation_stack(last_obs_stack, next_obs, done)
        transition = Transition(
            done, action, action_mean, value, reward, log_prob, policy_obs, info
        )
        return (next_state, next_obs_stack, key), transition

    return jax.lax.scan(env_step, (env_state, obs_stack, rng), None, config.num_steps)


def init_runner_state(
    rng: jax.Array,
    env: BelloLocomotionEnv,
    config: PPOConfig,
) -> tuple[RunnerState, eqx.Module, optax.GradientTransformation]:
    rng, init_key, env_key = jax.random.split(rng, 3)
    train_state, optimizer = init_train_state(
        init_key,
        policy_observation_size(env.observation_size, config.policy),
        env.action_size,
        config,
    )
    params, static_policy = eqx.partition(train_state.policy, eqx.is_array)
    env_state, obs = env.reset_batch(config.num_envs, rng=env_key)
    obs_stack = init_observation_stack(obs, config.policy.obs_stack_size)
    return (
        RunnerState(
            params=params,
            opt_state=train_state.opt_state,
            update=train_state.update,
            env_state=env_state,
            obs_stack=obs_stack,
            rng=rng,
        ),
        static_policy,
        optimizer,
    )


def runner_train_state(
    runner_state: RunnerState,
    static_policy: eqx.Module,
) -> TrainState:
    return TrainState(
        policy=eqx.combine(runner_state.params, static_policy),
        opt_state=runner_state.opt_state,
        update=runner_state.update,
    )


def update_runner_state(
    runner_state: RunnerState,
    static_policy: eqx.Module,
    optimizer: optax.GradientTransformation,
    env: BelloLocomotionEnv,
    config: PPOConfig,
) -> tuple[RunnerState, dict[str, jax.Array]]:
    key, rollout_key, update_key = jax.random.split(runner_state.rng, 3)
    policy = eqx.combine(runner_state.params, static_policy)
    train_state = TrainState(
        policy=policy,
        opt_state=runner_state.opt_state,
        update=runner_state.update,
    )
    (env_state, obs_stack, _), transitions = collect_rollout(
        env,
        policy,
        runner_state.env_state,
        runner_state.obs_stack,
        rollout_key,
        config,
    )
    _, _, last_value = apply_policy(policy, stacked_observation(obs_stack))
    advantages, targets = compute_gae(transitions, last_value, config)
    train_state, metrics = update_ppo(
        train_state, optimizer, transitions, advantages, targets, update_key, config
    )
    params, _ = eqx.partition(train_state.policy, eqx.is_array)
    metrics = {
        **metrics,
        "mean_reward": transitions.reward.mean(),
        "mean_done": transitions.done.astype(jnp.float32).mean(),
        "mean_episode_return": safe_returned_mean(transitions.info),
        "max_episode_return": safe_returned_max(transitions.info),
        "mean_episode_length": safe_returned_mean(
            transitions.info, key="episode_length"
        ),
        "max_episode_length": safe_returned_max(
            transitions.info, key="episode_length"
        ),
        **action_diagnostics(transitions, env, policy),
        **done_reason_means(transitions.info),
        **reward_component_means(transitions.info),
    }
    return (
        RunnerState(
            params=params,
            opt_state=train_state.opt_state,
            update=train_state.update,
            env_state=env_state,
            obs_stack=obs_stack,
            rng=key,
        ),
        metrics,
    )


def make_train(config: PPOConfig, env: BelloLocomotionEnv | None = None):
    validate_ppo_config(config)
    env = BelloLocomotionEnv.load(batch_size=config.num_envs) if env is None else env

    def train(rng: jax.Array):
        runner_state, static_policy, optimizer = init_runner_state(rng, env, config)

        def update_step(carry: RunnerState, _: None):
            return update_runner_state(
                carry,
                static_policy,
                optimizer,
                env,
                config,
            )

        runner_state, metrics = jax.lax.scan(
            update_step, runner_state, None, config.num_updates
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


def validate_ppo_config(config: PPOConfig) -> None:
    positive_ints = {
        "total_timesteps": config.total_timesteps,
        "num_envs": config.num_envs,
        "num_steps": config.num_steps,
        "update_epochs": config.update_epochs,
        "num_minibatches": config.num_minibatches,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if config.num_updates <= 0:
        raise ValueError(
            "total_timesteps must cover at least one rollout batch, got "
            f"total_timesteps={config.total_timesteps}, "
            f"num_envs={config.num_envs}, num_steps={config.num_steps}"
        )
    if config.batch_size % config.num_minibatches != 0:
        raise ValueError("num_envs * num_steps must be divisible by num_minibatches")
    if config.learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {config.learning_rate}")
    if not math.isfinite(config.learning_rate):
        raise ValueError(f"learning_rate must be finite, got {config.learning_rate}")
    if not 0.0 <= config.gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {config.gamma}")
    if not 0.0 <= config.gae_lambda <= 1.0:
        raise ValueError(f"gae_lambda must be in [0, 1], got {config.gae_lambda}")
    if not math.isfinite(config.clip_eps) or config.clip_eps <= 0:
        raise ValueError(f"clip_eps must be positive, got {config.clip_eps}")
    if not math.isfinite(config.ent_coef) or config.ent_coef < 0:
        raise ValueError(
            f"ent_coef must be nonnegative and finite, got {config.ent_coef}"
        )
    if not math.isfinite(config.vf_coef) or config.vf_coef < 0:
        raise ValueError(
            f"vf_coef must be nonnegative and finite, got {config.vf_coef}"
        )
    if not math.isfinite(config.max_grad_norm) or config.max_grad_norm <= 0:
        raise ValueError(f"max_grad_norm must be positive, got {config.max_grad_norm}")
    if config.policy.hidden_size <= 0:
        raise ValueError(
            f"policy.hidden_size must be positive, got {config.policy.hidden_size}"
        )
    if config.policy.hidden_depth < 0:
        raise ValueError(
            f"policy.hidden_depth must be nonnegative, got {config.policy.hidden_depth}"
        )
    if config.policy.obs_stack_size <= 0:
        raise ValueError(
            "policy.obs_stack_size must be positive, got "
            f"{config.policy.obs_stack_size}"
        )
    if config.policy.activation not in {"tanh", "relu"}:
        raise ValueError(
            "policy.activation must be 'tanh' or 'relu', got "
            f"{config.policy.activation!r}"
        )
    if not math.isfinite(config.policy.init_log_std):
        raise ValueError(
            f"policy.init_log_std must be finite, got {config.policy.init_log_std}"
        )
