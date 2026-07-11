"""PureJaxRL-style PPO for the Bello MJX environment.

This module keeps the training loop end-to-end JAX: rollout collection uses
`lax.scan`, batching uses `vmap`, and optimization uses Optax. It intentionally
does not depend on PureJaxRL as a package; the structure follows the same
single-file implementation style while using Equinox for the actor-critic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, NamedTuple

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
    clamp_policy_log_std,
    evaluate_actions,
    init_observation_stack,
    init_actor_critic,
    policy_observation_size,
    policy_log_std,
    sample_action_with_mean,
    stacked_observation,
    update_observation_stack,
)

LearningRateSchedule = Literal["constant", "linear", "adaptive"]
LOG_RATIO_CLIP = 20.0


@dataclass(frozen=True)
class PPOConfig:
    seed: int = 0
    total_timesteps: int = 1_000_000
    num_envs: int = 64
    num_steps: int = 24
    update_epochs: int = 5
    num_minibatches: int = 4
    learning_rate: float = 1.0e-3
    anneal_lr: bool = False
    schedule: LearningRateSchedule | None = "adaptive"
    desired_kl: float = 0.01
    target_kl: float | None = None
    min_learning_rate: float = 1.0e-5
    max_learning_rate: float = 1.0e-2
    adaptive_lr_factor: float = 1.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 1.0
    max_grad_norm: float = 1.0
    normalize_value: bool = True
    value_norm_eps: float = 1.0e-8
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
    terminal: jax.Array
    bootstrap_value: jax.Array
    # Raw Gaussian sample used for PPO log-prob/ratio calculations.
    action: jax.Array
    # Action actually sent to the env after optional env-level clipping.
    env_action: jax.Array
    action_mean: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    critic_obs: jax.Array
    info: dict[str, jax.Array]


class ValueNormState(NamedTuple):
    mean: jax.Array
    var: jax.Array
    count: jax.Array


class RunnerState(NamedTuple):
    params: eqx.Module
    opt_state: optax.OptState
    update: jax.Array
    env_state: BelloState
    obs_stack: jax.Array
    critic_obs_stack: jax.Array
    command_ranges: jax.Array
    rng: jax.Array
    learning_rate: jax.Array
    value_norm_state: ValueNormState


def init_train_state(
    rng: jax.Array,
    obs_size: int,
    action_size: int,
    config: PPOConfig,
    critic_obs_size: int | None = None,
) -> tuple[TrainState, optax.GradientTransformation]:
    model = init_actor_critic(
        rng, obs_size, action_size, config.policy, critic_obs_size=critic_obs_size
    )
    optimizer = make_optimizer(config)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    return TrainState(
        policy=model, opt_state=opt_state, update=jnp.asarray(0, dtype=jnp.int32)
    ), optimizer


def make_optimizer(config: PPOConfig) -> optax.GradientTransformation:
    validate_ppo_config(config)
    return optax.apply_if_finite(
        optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.scale_by_adam(eps=1.0e-5),
        ),
        max_consecutive_errors=1,
    )


def effective_learning_rate_schedule(config: PPOConfig) -> LearningRateSchedule:
    if config.schedule is not None:
        return config.schedule
    return "linear" if config.anneal_lr else "constant"


def learning_rate_for_update(
    config: PPOConfig,
    update: jax.Array,
    current_learning_rate: jax.Array,
) -> jax.Array:
    schedule = effective_learning_rate_schedule(config)
    if schedule == "linear":
        frac = 1.0 - update.astype(jnp.float32) / jnp.maximum(config.num_updates, 1)
        return jnp.asarray(config.learning_rate, dtype=jnp.float32) * jnp.maximum(
            frac, 0.0
        )
    if schedule == "constant":
        return jnp.asarray(config.learning_rate, dtype=jnp.float32)
    return current_learning_rate


def adapt_learning_rate(
    learning_rate: jax.Array,
    approx_kl: jax.Array,
    config: PPOConfig,
) -> jax.Array:
    if effective_learning_rate_schedule(config) != "adaptive":
        return learning_rate
    finite_kl = jnp.where(jnp.isfinite(approx_kl), approx_kl, jnp.inf)
    low_kl = (finite_kl > 0.0) & (finite_kl < config.desired_kl / 2.0)
    high_kl = finite_kl > config.desired_kl * 2.0
    decreased = jnp.maximum(
        learning_rate / config.adaptive_lr_factor,
        jnp.asarray(config.min_learning_rate, dtype=jnp.float32),
    )
    increased = jnp.minimum(
        learning_rate * config.adaptive_lr_factor,
        jnp.asarray(config.max_learning_rate, dtype=jnp.float32),
    )
    return jnp.where(high_kl, decreased, jnp.where(low_kl, increased, learning_rate))


def init_value_norm_state() -> ValueNormState:
    return ValueNormState(
        mean=jnp.asarray(0.0, dtype=jnp.float32),
        var=jnp.asarray(1.0, dtype=jnp.float32),
        count=jnp.asarray(0.0, dtype=jnp.float32),
    )


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
    batch_count = jnp.asarray(targets.size, dtype=jnp.float32)
    batch_mean = jnp.mean(targets)
    batch_var = jnp.var(targets)
    delta = batch_mean - state.mean
    total_count = state.count + batch_count
    mean = state.mean + delta * batch_count / total_count
    state_m2 = state.var * state.count
    batch_m2 = batch_var * batch_count
    correction = delta**2 * state.count * batch_count / total_count
    var = (state_m2 + batch_m2 + correction) / total_count
    return ValueNormState(mean=mean, var=var, count=total_count)


def compute_gae(
    transitions: Transition,
    last_value: jax.Array,
    config: PPOConfig,
) -> tuple[jax.Array, jax.Array]:
    def scan_step(
        carry: tuple[jax.Array, jax.Array], transition: Transition
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        gae, next_value = carry
        not_terminal = 1.0 - transition.terminal.astype(jnp.float32)
        not_boundary = 1.0 - transition.done.astype(jnp.float32)
        next_value = jnp.where(
            transition.done & ~transition.terminal,
            transition.bootstrap_value,
            next_value,
        )
        delta = (
            transition.reward
            + config.gamma * next_value * not_terminal
            - transition.value
        )
        gae = delta + config.gamma * config.gae_lambda * not_boundary * gae
        return (gae, transition.value), gae

    _, advantages = jax.lax.scan(
        scan_step,
        (jnp.zeros_like(last_value), last_value),
        transitions,
        reverse=True,
    )
    return advantages, advantages + transitions.value


def normalize_advantages(advantages: jax.Array) -> jax.Array:
    return (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)


def ppo_loss(
    model: ActorCritic,
    transitions: Transition,
    advantages: jax.Array,
    targets: jax.Array,
    config: PPOConfig,
    value_norm_state: ValueNormState | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    log_prob, entropy, value = evaluate_actions(
        model, transitions.obs, transitions.action, transitions.critic_obs
    )

    log_ratio = log_prob - transitions.log_prob
    safe_log_ratio = jnp.clip(log_ratio, -LOG_RATIO_CLIP, LOG_RATIO_CLIP)
    ratio = jnp.exp(safe_log_ratio)
    actor_loss_1 = ratio * advantages
    actor_loss_2 = (
        jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * advantages
    )
    actor_loss = -jnp.minimum(actor_loss_1, actor_loss_2).mean()

    if value_norm_state is None:
        value_norm_state = init_value_norm_state()
    raw_value = denormalize_value(value, value_norm_state, config)
    value_for_loss = value
    old_value_for_loss = normalize_value(transitions.value, value_norm_state, config)
    targets_for_loss = normalize_value(targets, value_norm_state, config)
    value_clipped = old_value_for_loss + jnp.clip(
        value_for_loss - old_value_for_loss, -config.clip_eps, config.clip_eps
    )
    value_loss = (
        0.5
        * jnp.maximum(
            (value_for_loss - targets_for_loss) ** 2,
            (value_clipped - targets_for_loss) ** 2,
        ).mean()
    )
    value_loss_raw = 0.5 * ((raw_value - targets) ** 2).mean()
    entropy_loss = entropy.mean()
    total_loss = (
        actor_loss + config.vf_coef * value_loss - config.ent_coef * entropy_loss
    )
    metrics = {
        "loss": total_loss,
        "actor_loss": actor_loss,
        "value_loss": value_loss,
        "value_loss_raw": value_loss_raw,
        "entropy": entropy_loss,
        "approx_kl": ((ratio - 1.0) - safe_log_ratio).mean(),
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
    learning_rate: jax.Array | float | None = None,
    value_norm_state: ValueNormState | None = None,
) -> tuple[TrainState, dict[str, jax.Array]]:
    advantages = normalize_advantages(advantages)
    batch = flatten_batch((transitions, advantages, targets))
    init_params, static_policy = eqx.partition(train_state.policy, eqx.is_array)
    if learning_rate is None:
        learning_rate = jnp.asarray(config.learning_rate, dtype=jnp.float32)
    else:
        learning_rate = jnp.asarray(learning_rate, dtype=jnp.float32)
    target_kl = math.inf if config.target_kl is None else float(config.target_kl)

    def tree_all_finite(tree) -> jax.Array:
        leaves = jax.tree_util.tree_leaves(tree)
        if not leaves:
            return jnp.asarray(True)
        return jnp.all(
            jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves])
        )

    def update_epoch(carry, _: None):
        params, opt_state, key, stopped = carry
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
            params, opt_state, stopped = carry
            old_opt_state = opt_state
            mb_transitions, mb_advantages, mb_targets = minibatch

            def loss_fn(params):
                model = eqx.combine(params, static_policy)
                return ppo_loss(
                    model,
                    mb_transitions,
                    mb_advantages,
                    mb_targets,
                    config,
                    value_norm_state,
                )

            (loss_value, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params
            )
            updates, opt_state = optimizer.update(grads, opt_state, params)
            updates = jax.tree_util.tree_map(
                lambda update: -learning_rate * update, updates
            )
            next_params = optax.apply_updates(params, updates)
            policy = clamp_policy_log_std(eqx.combine(next_params, static_policy))
            next_params, _ = eqx.partition(policy, eqx.is_array)

            finite_update = (
                jnp.isfinite(loss_value)
                & tree_all_finite(metrics)
                & tree_all_finite(grads)
                & tree_all_finite(updates)
                & tree_all_finite(next_params)
            )
            approx_kl = jnp.where(
                jnp.isfinite(metrics["approx_kl"]),
                metrics["approx_kl"],
                jnp.asarray(jnp.inf, dtype=jnp.float32),
            )
            exceeds_target_kl = approx_kl > target_kl
            should_update = (~stopped) & finite_update & (~exceeds_target_kl)
            params = jax.tree_util.tree_map(
                lambda new, old: jnp.where(should_update, new, old),
                next_params,
                params,
            )
            opt_state = jax.tree_util.tree_map(
                lambda new, old: jnp.where(should_update, new, old),
                opt_state,
                old_opt_state,
            )
            stopped = stopped | exceeds_target_kl | (~finite_update)
            safe_metrics = jax.tree_util.tree_map(
                lambda value: jnp.where(
                    jnp.isfinite(value), value, jnp.zeros_like(value)
                ),
                metrics,
            )
            metrics = {
                **safe_metrics,
                "loss": jnp.where(
                    jnp.isfinite(loss_value), loss_value, jnp.zeros_like(loss_value)
                ),
                "approx_kl": approx_kl,
                "nonfinite_update": (~finite_update).astype(jnp.float32),
                "target_kl_exceeded": exceeds_target_kl.astype(jnp.float32),
                "ppo_update_fraction": should_update.astype(jnp.float32),
            }
            return (params, opt_state, stopped), metrics

        (params, opt_state, stopped), metrics = jax.lax.scan(
            update_minibatch,
            (params, opt_state, stopped),
            minibatches,
        )
        return (params, opt_state, key, stopped), metrics

    (params, opt_state, _, _), metrics = jax.lax.scan(
        update_epoch,
        (
            init_params,
            train_state.opt_state,
            rng,
            jnp.asarray(False),
        ),
        None,
        config.update_epochs,
    )
    metrics = jax.tree_util.tree_map(lambda x: x.mean(), metrics)
    train_state = TrainState(
        policy=clamp_policy_log_std(eqx.combine(params, static_policy)),
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
    return jnp.sum(value.astype(jnp.float32) * mask) / jnp.maximum(jnp.sum(mask), 1.0)


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


def random_push_means(info: dict[str, jax.Array]) -> dict[str, jax.Array]:
    return {
        f"mean_{name}": value.astype(jnp.float32).mean()
        for name, value in info.items()
        if name.startswith("random_push/")
    }


def foot_contact_means(info: dict[str, jax.Array]) -> dict[str, jax.Array]:
    return {
        f"mean_{name}": value.astype(jnp.float32).mean()
        for name, value in info.items()
        if name.startswith("foot_contact/")
    }


def action_diagnostics(
    transitions: Transition,
    env: BelloLocomotionEnv,
    policy: ActorCritic,
) -> dict[str, jax.Array]:
    action = transitions.action
    env_action = transitions.env_action
    action_mean = transitions.action_mean
    log_std = policy_log_std(policy)
    metrics = {
        "action_rms": jnp.sqrt(jnp.mean(action**2)),
        "action_abs_max": jnp.max(jnp.abs(action)),
        "action_mean_rms": jnp.sqrt(jnp.mean(action_mean**2)),
        "action_mean_abs_max": jnp.max(jnp.abs(action_mean)),
        "action_std_mean": jnp.mean(jnp.exp(log_std)),
        "action_std_min": jnp.min(jnp.exp(log_std)),
        "action_std_max": jnp.max(jnp.exp(log_std)),
    }
    if not hasattr(env, "_action_to_ctrl"):
        return metrics

    metrics = {**metrics, "env_action_abs_max": jnp.max(jnp.abs(env_action))}
    flat_action = env_action.reshape((-1, env.action_size))
    ctrl = jax.vmap(env._action_to_ctrl)(flat_action)
    policy_ctrl_indices = jnp.asarray(env.model.policy_ctrl_indices)
    policy_ctrl = ctrl[:, policy_ctrl_indices]
    ctrlrange = jnp.asarray(env.model.mj_model.actuator_ctrlrange)[policy_ctrl_indices]
    ctrl_low = ctrlrange[:, 0]
    ctrl_high = ctrlrange[:, 1]
    ctrl_margin = jnp.minimum(policy_ctrl - ctrl_low, ctrl_high - policy_ctrl)
    saturated = (policy_ctrl <= ctrl_low + 1.0e-6) | (policy_ctrl >= ctrl_high - 1.0e-6)
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
    critic_obs_stack: jax.Array,
    command_ranges: jax.Array,
    rng: jax.Array,
    config: PPOConfig,
    value_norm_state: ValueNormState,
) -> tuple[tuple[BelloState, jax.Array, jax.Array, jax.Array], Transition]:
    def env_step(
        carry: tuple[BelloState, jax.Array, jax.Array, jax.Array], _: None
    ) -> tuple[tuple[BelloState, jax.Array, jax.Array, jax.Array], Transition]:
        state, last_obs_stack, last_critic_obs_stack, key = carry
        key, action_key, env_key = jax.random.split(key, 3)
        policy_obs = stacked_observation(last_obs_stack)
        critic_policy_obs = stacked_observation(last_critic_obs_stack)
        action_clip = getattr(env, "action_clip", None)
        action, log_prob, value, action_mean = sample_action_with_mean(
            policy,
            policy_obs,
            action_key,
            critic_obs=critic_policy_obs,
        )
        env_action = (
            action
            if action_clip is None
            else jnp.clip(action, -float(action_clip), float(action_clip))
        )
        value = denormalize_value(value, value_norm_state, config)
        next_state, next_obs, reward, done, info = step_env_batch(
            env, state, env_action, env_key, command_ranges
        )
        next_critic_obs = critic_observation_batch(
            env, next_state, fallback_obs=next_obs
        )
        next_obs_stack = update_observation_stack(last_obs_stack, next_obs, done)
        next_critic_obs_stack = update_observation_stack(
            last_critic_obs_stack, next_critic_obs, done
        )
        terminal = info.get("done/terminal", done)
        bootstrap_obs = info.get("bootstrap_observation", next_obs)
        bootstrap_obs = jnp.where(terminal[:, None], next_obs, bootstrap_obs)
        bootstrap_critic_obs = info.get("bootstrap_critic_observation", next_critic_obs)
        bootstrap_critic_obs = jnp.where(
            terminal[:, None], next_critic_obs, bootstrap_critic_obs
        )
        bootstrap_obs_stack = update_observation_stack(
            last_obs_stack, bootstrap_obs, terminal
        )
        bootstrap_critic_obs_stack = update_observation_stack(
            last_critic_obs_stack, bootstrap_critic_obs, terminal
        )
        _, _, bootstrap_value = apply_policy(
            policy,
            stacked_observation(bootstrap_obs_stack),
            stacked_observation(bootstrap_critic_obs_stack),
        )
        bootstrap_value = denormalize_value(bootstrap_value, value_norm_state, config)
        transition_info = {
            name: value
            for name, value in info.items()
            if name not in {"bootstrap_observation", "bootstrap_critic_observation"}
        }
        transition = Transition(
            done,
            terminal,
            bootstrap_value,
            action,
            env_action,
            action_mean,
            value,
            reward,
            log_prob,
            policy_obs,
            critic_policy_obs,
            transition_info,
        )
        return (next_state, next_obs_stack, next_critic_obs_stack, key), transition

    return jax.lax.scan(
        env_step, (env_state, obs_stack, critic_obs_stack, rng), None, config.num_steps
    )


def init_runner_state(
    rng: jax.Array,
    env: BelloLocomotionEnv,
    config: PPOConfig,
) -> tuple[RunnerState, eqx.Module, optax.GradientTransformation]:
    rng, init_key, env_key, step_count_key = jax.random.split(rng, 4)
    train_state, optimizer = init_train_state(
        init_key,
        policy_observation_size(env.observation_size, config.policy),
        env.action_size,
        config,
        critic_obs_size=policy_observation_size(
            critic_observation_size(env), config.policy
        ),
    )
    params, static_policy = eqx.partition(train_state.policy, eqx.is_array)
    command_ranges = initial_command_ranges(env)
    env_state, obs = reset_env_batch(env, config.num_envs, env_key, command_ranges)
    critic_obs = critic_observation_batch(env, env_state, fallback_obs=obs)
    step_count = initial_step_counts(env, config.num_envs, step_count_key)
    env_state = env_state.replace(step_count=step_count, episode_length=step_count)
    obs_stack = init_observation_stack(obs, config.policy.obs_stack_size)
    critic_obs_stack = init_observation_stack(critic_obs, config.policy.obs_stack_size)
    return (
        RunnerState(
            params=params,
            opt_state=train_state.opt_state,
            update=train_state.update,
            env_state=env_state,
            obs_stack=obs_stack,
            critic_obs_stack=critic_obs_stack,
            command_ranges=command_ranges,
            rng=rng,
            learning_rate=jnp.asarray(config.learning_rate, dtype=jnp.float32),
            value_norm_state=init_value_norm_state(),
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
    (env_state, obs_stack, critic_obs_stack, _), transitions = collect_rollout(
        env,
        policy,
        runner_state.env_state,
        runner_state.obs_stack,
        runner_state.critic_obs_stack,
        runner_state.command_ranges,
        rollout_key,
        config,
        runner_state.value_norm_state,
    )
    _, _, last_value = apply_policy(
        policy, stacked_observation(obs_stack), stacked_observation(critic_obs_stack)
    )
    last_value = denormalize_value(last_value, runner_state.value_norm_state, config)
    advantages, targets = compute_gae(transitions, last_value, config)
    value_norm_state = update_value_norm_state(
        runner_state.value_norm_state, targets, config
    )
    learning_rate = learning_rate_for_update(
        config, runner_state.update, runner_state.learning_rate
    )
    train_state, metrics = update_ppo(
        train_state,
        optimizer,
        transitions,
        advantages,
        targets,
        update_key,
        config,
        learning_rate=learning_rate,
        value_norm_state=value_norm_state,
    )
    next_learning_rate = adapt_learning_rate(
        learning_rate, metrics["approx_kl"], config
    )
    params, _ = eqx.partition(train_state.policy, eqx.is_array)
    reward_means = reward_component_means(transitions.info)
    command_ranges, command_curriculum_state = update_command_curriculum(
        runner_state.command_ranges,
        transitions.info,
        runner_state.update,
        env,
        config,
    )
    metrics = {
        **metrics,
        "learning_rate": learning_rate,
        "next_learning_rate": next_learning_rate,
        "value_norm_mean": value_norm_state.mean,
        "value_norm_std": value_norm_std(value_norm_state, config),
        "value_norm_count": value_norm_state.count,
        "mean_reward": transitions.reward.mean(),
        "mean_done": transitions.done.astype(jnp.float32).mean(),
        "mean_episode_return": safe_returned_mean(transitions.info),
        "max_episode_return": safe_returned_max(transitions.info),
        "mean_episode_length": safe_returned_mean(
            transitions.info, key="episode_length"
        ),
        "max_episode_length": safe_returned_max(transitions.info, key="episode_length"),
        **action_diagnostics(transitions, env, policy),
        **done_reason_means(transitions.info),
        **random_push_means(transitions.info),
        **foot_contact_means(transitions.info),
        **reward_means,
        **command_curriculum_metrics(command_ranges),
        **command_curriculum_state,
    }
    return (
        RunnerState(
            params=params,
            opt_state=train_state.opt_state,
            update=train_state.update,
            env_state=env_state,
            obs_stack=obs_stack,
            critic_obs_stack=critic_obs_stack,
            command_ranges=command_ranges,
            rng=key,
            learning_rate=next_learning_rate,
            value_norm_state=value_norm_state,
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


def update_command_curriculum_ranges(
    ranges: jax.Array,
    info: dict[str, jax.Array],
    update: jax.Array,
    env: BelloLocomotionEnv,
    config: PPOConfig,
) -> jax.Array:
    ranges, _ = update_command_curriculum(ranges, info, update, env, config)
    return ranges


def update_command_curriculum(
    ranges: jax.Array,
    info: dict[str, jax.Array],
    update: jax.Array,
    env: BelloLocomotionEnv,
    config: PPOConfig,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    command_config = getattr(env, "command_config", None)
    if command_config is None:
        return ranges, command_curriculum_state_metrics(enabled=False)

    curriculum = command_config.curriculum
    reward_threshold = (
        env.reward_config.track_lin_vel_xy.weight * curriculum.success_threshold
    )
    old_period = (update * config.num_steps) // env.max_episode_steps
    new_period = ((update + 1) * config.num_steps) // env.max_episode_steps
    period_boundary = new_period > old_period

    if not command_config.enabled or not curriculum.enabled:
        return ranges, command_curriculum_state_metrics(
            enabled=False,
            reward_threshold=reward_threshold,
            new_period=new_period,
            period_boundary=period_boundary,
        )

    reward_key = "episode_reward/track_lin_vel_xy"
    returned_episode = info.get("returned_episode")
    if reward_key not in info or returned_episode is None:
        return ranges, command_curriculum_state_metrics(
            enabled=True,
            reward_threshold=reward_threshold,
            new_period=new_period,
            period_boundary=period_boundary,
        )

    episode_reward = safe_returned_mean(info, key=reward_key)
    returned_count = jnp.sum(returned_episode.astype(jnp.float32))
    normalized_reward = episode_reward / (env.max_episode_steps * env.control_timestep)
    success = normalized_reward > reward_threshold
    should_expand = (
        period_boundary
        & (returned_count > 0)
        & success
    )
    should_expand_yaw = should_expand & curriculum.expand_ang_vel_yaw

    x_range = expand_command_range(
        ranges[0],
        curriculum.limit_lin_vel_x,
        curriculum.increment,
        should_expand,
    )
    y_range = expand_command_range(
        ranges[1],
        curriculum.limit_lin_vel_y,
        curriculum.increment,
        should_expand,
    )
    yaw_range = expand_command_range(
        ranges[2],
        curriculum.limit_ang_vel_yaw,
        curriculum.increment,
        should_expand_yaw,
    )
    metrics = command_curriculum_state_metrics(
        enabled=True,
        returned_count=returned_count,
        episode_reward=episode_reward,
        normalized_reward=normalized_reward,
        reward_threshold=reward_threshold,
        new_period=new_period,
        period_boundary=period_boundary,
        success=success,
        should_expand=should_expand,
        should_expand_yaw=should_expand_yaw,
    )
    return jnp.stack([x_range, y_range, yaw_range]), metrics


def command_curriculum_state_metrics(
    *,
    enabled: bool | jax.Array,
    returned_count: jax.Array | float = 0.0,
    episode_reward: jax.Array | float = 0.0,
    normalized_reward: jax.Array | float = 0.0,
    reward_threshold: jax.Array | float = 0.0,
    new_period: jax.Array | float = 0.0,
    period_boundary: bool | jax.Array = False,
    success: bool | jax.Array = False,
    should_expand: bool | jax.Array = False,
    should_expand_yaw: bool | jax.Array = False,
) -> dict[str, jax.Array]:
    return {
        "command_curriculum/enabled": jnp.asarray(enabled, dtype=jnp.float32),
        "command_curriculum/returned_count": jnp.asarray(
            returned_count, dtype=jnp.float32
        ),
        "command_curriculum/episode_track_lin_vel_xy": jnp.asarray(
            episode_reward, dtype=jnp.float32
        ),
        "command_curriculum/normalized_track_lin_vel_xy": jnp.asarray(
            normalized_reward, dtype=jnp.float32
        ),
        "command_curriculum/reward_threshold": jnp.asarray(
            reward_threshold, dtype=jnp.float32
        ),
        "command_curriculum/period": jnp.asarray(new_period, dtype=jnp.float32),
        "command_curriculum/period_boundary": jnp.asarray(
            period_boundary, dtype=jnp.float32
        ),
        "command_curriculum/success": jnp.asarray(success, dtype=jnp.float32),
        "command_curriculum/should_expand": jnp.asarray(
            should_expand, dtype=jnp.float32
        ),
        "command_curriculum/should_expand_yaw": jnp.asarray(
            should_expand_yaw, dtype=jnp.float32
        ),
    }


def expand_command_range(
    command_range: jax.Array,
    limits: tuple[float, float],
    increment: float,
    should_expand: jax.Array,
) -> jax.Array:
    limit = jnp.asarray(limits, dtype=jnp.float32)
    expanded = jnp.clip(
        command_range + jnp.asarray([-increment, increment], dtype=jnp.float32),
        limit[0],
        limit[1],
    )
    return jnp.where(should_expand, expanded, command_range)


def command_curriculum_metrics(ranges: jax.Array) -> dict[str, jax.Array]:
    return {
        "command_range/lin_vel_x_min": ranges[0, 0],
        "command_range/lin_vel_x_max": ranges[0, 1],
        "command_range/lin_vel_y_min": ranges[1, 0],
        "command_range/lin_vel_y_max": ranges[1, 1],
        "command_range/ang_vel_yaw_min": ranges[2, 0],
        "command_range/ang_vel_yaw_max": ranges[2, 1],
    }


def initial_command_ranges(env) -> jax.Array:
    command_config = getattr(env, "command_config", None)
    if command_config is None:
        return jnp.zeros((3, 2), dtype=jnp.float32)
    return command_config.range_array()


def reset_env_batch(
    env,
    batch_size: int,
    rng: jax.Array,
    command_ranges: jax.Array,
):
    if hasattr(env, "command_config"):
        return env.reset_batch(batch_size, rng=rng, command_ranges=command_ranges)
    return env.reset_batch(batch_size, rng=rng)


def critic_observation_size(env) -> int:
    return int(getattr(env, "critic_observation_size", env.observation_size))


def critic_observation_batch(env, state, fallback_obs: jax.Array | None = None) -> jax.Array:
    method = getattr(env, "critic_observation_batch", None)
    if method is not None:
        return method(state)
    if fallback_obs is not None:
        return fallback_obs
    return getattr(state, "obs")


def step_env_batch(
    env,
    state,
    action: jax.Array,
    rng: jax.Array,
    command_ranges: jax.Array,
):
    if hasattr(env, "command_config"):
        return env.step_batch(state, action, rng=rng, command_ranges=command_ranges)
    return env.step_batch(state, action, rng=rng)


def initial_step_counts(env, batch_size: int, rng: jax.Array) -> jax.Array:
    method = getattr(env, "initial_step_counts", None)
    if method is None:
        return jnp.zeros((batch_size,), dtype=jnp.int32)
    return method(batch_size, rng)


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
    if effective_learning_rate_schedule(config) not in {
        "constant",
        "linear",
        "adaptive",
    }:
        raise ValueError(
            "schedule must be one of 'constant', 'linear', or 'adaptive', got "
            f"{config.schedule!r}"
        )
    if not math.isfinite(config.desired_kl) or config.desired_kl <= 0:
        raise ValueError(
            f"desired_kl must be positive and finite, got {config.desired_kl}"
        )
    if config.target_kl is not None and (
        not math.isfinite(config.target_kl) or config.target_kl <= 0
    ):
        raise ValueError(
            f"target_kl must be null or positive and finite, got {config.target_kl}"
        )
    if not math.isfinite(config.min_learning_rate) or config.min_learning_rate <= 0:
        raise ValueError(
            "min_learning_rate must be positive and finite, got "
            f"{config.min_learning_rate}"
        )
    if not math.isfinite(config.max_learning_rate) or config.max_learning_rate <= 0:
        raise ValueError(
            "max_learning_rate must be positive and finite, got "
            f"{config.max_learning_rate}"
        )
    if config.min_learning_rate > config.max_learning_rate:
        raise ValueError(
            "min_learning_rate must be <= max_learning_rate, got "
            f"{config.min_learning_rate} > {config.max_learning_rate}"
        )
    if not math.isfinite(config.adaptive_lr_factor) or config.adaptive_lr_factor <= 1:
        raise ValueError(
            "adaptive_lr_factor must be finite and > 1, got "
            f"{config.adaptive_lr_factor}"
        )
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
    if not math.isfinite(config.value_norm_eps) or config.value_norm_eps <= 0:
        raise ValueError(
            f"value_norm_eps must be positive and finite, got {config.value_norm_eps}"
        )
    if config.policy.hidden_size <= 0:
        raise ValueError(
            f"policy.hidden_size must be positive, got {config.policy.hidden_size}"
        )
    if config.policy.hidden_depth < 0:
        raise ValueError(
            f"policy.hidden_depth must be nonnegative, got {config.policy.hidden_depth}"
        )
    if config.policy.hidden_dims is not None and any(
        hidden_dim <= 0 for hidden_dim in config.policy.hidden_dims
    ):
        raise ValueError(
            "policy.hidden_dims entries must be positive, got "
            f"{config.policy.hidden_dims}"
        )
    if config.policy.obs_stack_size <= 0:
        raise ValueError(
            "policy.obs_stack_size must be positive, got "
            f"{config.policy.obs_stack_size}"
        )
    if config.policy.activation not in {"tanh", "relu", "elu"}:
        raise ValueError(
            "policy.activation must be 'tanh', 'relu', or 'elu', got "
            f"{config.policy.activation!r}"
        )
    if not math.isfinite(config.policy.init_log_std):
        raise ValueError(
            f"policy.init_log_std must be finite, got {config.policy.init_log_std}"
        )
    if not math.isfinite(config.policy.min_log_std):
        raise ValueError(
            f"policy.min_log_std must be finite, got {config.policy.min_log_std}"
        )
    if not math.isfinite(config.policy.max_log_std):
        raise ValueError(
            f"policy.max_log_std must be finite, got {config.policy.max_log_std}"
        )
    if config.policy.min_log_std > config.policy.max_log_std:
        raise ValueError(
            "policy.min_log_std must be <= policy.max_log_std, got "
            f"{config.policy.min_log_std} > {config.policy.max_log_std}"
        )
