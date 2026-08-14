"""PureJaxRL-style PPO for the Bello MJX environment.

This module keeps the training loop end-to-end JAX: rollout collection uses
`lax.scan`, batching uses `vmap`, and optimization uses Optax. It intentionally
does not depend on PureJaxRL as a package; the structure follows the same
single-file implementation style while using Equinox for the actor-critic.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import math
from typing import Any, Callable, Literal, NamedTuple, Protocol, get_args

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

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
    sample_action_with_latent_and_mean,
    stacked_observation,
    update_observation_stack,
)

LearningRateSchedule = Literal["constant", "linear", "adaptive"]
AdaptiveKLStatistic = Literal["mean", "max"]
LOG_RATIO_CLIP = 20.0
_VALIDATOR_METADATA_KEY = "validator"


FieldValidator = Callable[[str, Any], None]


def _validated(default: Any, validator: FieldValidator):
    return field(
        default=default,
        metadata={_VALIDATOR_METADATA_KEY: validator},
    )


def _no_scalar_constraint(_name: str, _value: Any) -> None:
    pass


def _positive_integer(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _positive_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite, got {value}")


def _optional_positive_finite(name: str, value: float | None) -> None:
    if value is not None:
        _positive_finite(name, value)


def _nonnegative_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be nonnegative and finite, got {value}")


def _unit_interval(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def _greater_than_one_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 1:
        raise ValueError(f"{name} must be finite and > 1, got {value}")


def _one_of(*choices: str) -> FieldValidator:
    allowed = frozenset(choices)

    def validate(name: str, value: str) -> None:
        if value not in allowed:
            options = ", ".join(repr(choice) for choice in choices)
            raise ValueError(f"{name} must be one of {options}, got {value!r}")

    return validate


class PPOEnvironment(Protocol):
    """Explicit environment surface consumed by the shared PPO runner."""

    @property
    def observation_size(self) -> int: ...

    @property
    def critic_observation_size(self) -> int: ...

    @property
    def action_size(self) -> int: ...

    @property
    def action_clip(self) -> float: ...

    @property
    def control_timestep(self) -> float: ...

    @property
    def ppo_stagger_initial_step_count(self) -> bool: ...

    def reset_batch(
        self, batch_size: int, rng: jax.Array | None = None
    ) -> tuple[Any, jax.Array]: ...

    def initial_step_counts(
        self, batch_size: int, rng: jax.Array | None = None
    ) -> jax.Array: ...

    def observation_batch(
        self, state: Any, rng: jax.Array | None = None
    ) -> jax.Array: ...

    def critic_observation_batch(
        self, state: Any, rng: jax.Array | None = None
    ) -> jax.Array: ...

    def step_batch(
        self, state: Any, action: jax.Array, rng: jax.Array | None = None
    ) -> tuple[Any, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]: ...

    def ppo_action_diagnostics(self, env_action: jax.Array) -> dict[str, jax.Array]: ...


@dataclass(frozen=True)
class PPOConfig:
    seed: int = _validated(0, _no_scalar_constraint)
    total_timesteps: int = _validated(1_000_000, _positive_integer)
    num_envs: int = _validated(64, _positive_integer)
    num_steps: int = _validated(24, _positive_integer)
    update_epochs: int = _validated(5, _positive_integer)
    num_minibatches: int = _validated(4, _positive_integer)
    learning_rate: float = _validated(1.0e-3, _positive_finite)
    schedule: LearningRateSchedule = _validated(
        "adaptive",
        _one_of(*get_args(LearningRateSchedule)),
    )
    desired_kl: float = _validated(0.01, _positive_finite)
    target_kl: float | None = _validated(None, _optional_positive_finite)
    min_learning_rate: float = _validated(1.0e-5, _positive_finite)
    max_learning_rate: float = _validated(1.0e-2, _positive_finite)
    adaptive_lr_factor: float = _validated(1.5, _greater_than_one_finite)
    adaptive_kl_statistic: AdaptiveKLStatistic = _validated(
        "max",
        _one_of(*get_args(AdaptiveKLStatistic)),
    )
    gamma: float = _validated(0.99, _unit_interval)
    gae_lambda: float = _validated(0.95, _unit_interval)
    clip_eps: float = _validated(0.2, _positive_finite)
    ent_coef: float = _validated(0.01, _nonnegative_finite)
    vf_coef: float = _validated(1.0, _nonnegative_finite)
    max_grad_norm: float = _validated(1.0, _positive_finite)
    normalize_value: bool = _validated(False, _no_scalar_constraint)
    value_norm_eps: float = _validated(1.0e-8, _positive_finite)
    normalize_observations: bool = _validated(False, _no_scalar_constraint)
    observation_norm_eps: float = _validated(1.0e-5, _positive_finite)
    observation_norm_clip: float = _validated(10.0, _positive_finite)
    rollout_observation_max_abs: float = _validated(1.0e4, _positive_finite)
    rollout_reward_max_abs: float = _validated(1.0e3, _positive_finite)
    rollout_value_max_abs: float = _validated(1.0e4, _positive_finite)
    target_max_abs: float = _validated(1.0e4, _positive_finite)
    gradient_norm_max: float = _validated(1.0e4, _positive_finite)
    policy: ActorCriticConfig = _validated(
        ActorCriticConfig(),
        _no_scalar_constraint,
    )

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
    # Action sampled from the policy distribution for PPO log-prob/ratio calculations.
    action: jax.Array
    # Exact pre-transform Gaussian sample; tanh saturation is not invertible in float32.
    latent_action: jax.Array
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


class ObservationNormState(NamedTuple):
    mean: jax.Array
    var: jax.Array
    count: jax.Array


class RunnerState(NamedTuple):
    params: eqx.Module
    opt_state: optax.OptState
    update: jax.Array
    env_state: Any
    obs_stack: jax.Array
    critic_obs_stack: jax.Array
    rng: jax.Array
    learning_rate: jax.Array
    value_norm_state: ValueNormState
    observation_norm_state: ObservationNormState
    critic_observation_norm_state: ObservationNormState


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
    return config.schedule


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
    *,
    target_kl_exceeded: jax.Array | bool = False,
) -> jax.Array:
    if effective_learning_rate_schedule(config) != "adaptive":
        return learning_rate
    min_learning_rate = jnp.asarray(config.min_learning_rate, dtype=jnp.float32)
    max_learning_rate = jnp.asarray(config.max_learning_rate, dtype=jnp.float32)
    learning_rate = jnp.clip(
        learning_rate,
        min_learning_rate,
        max_learning_rate,
    )
    finite_kl = jnp.where(jnp.isfinite(approx_kl), approx_kl, jnp.inf)
    low_kl = (finite_kl > 0.0) & (finite_kl < config.desired_kl / 2.0)
    high_kl = (finite_kl > config.desired_kl * 2.0) | jnp.asarray(
        target_kl_exceeded, dtype=bool
    )
    decreased = jnp.maximum(
        learning_rate / config.adaptive_lr_factor,
        min_learning_rate,
    )
    increased = jnp.minimum(
        learning_rate * config.adaptive_lr_factor,
        max_learning_rate,
    )
    return jnp.where(high_kl, decreased, jnp.where(low_kl, increased, learning_rate))


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
    normalized = (observation - state.mean) / jnp.sqrt(
        jnp.maximum(state.var, config.observation_norm_eps)
    )
    clipped = jnp.clip(
        normalized,
        -config.observation_norm_clip,
        config.observation_norm_clip,
    )
    valid = array_is_plausible(observation, config.rollout_observation_max_abs)
    return jnp.where(valid, clipped, observation)


def update_observation_norm_state(
    state: ObservationNormState,
    observation: jax.Array,
    config: PPOConfig,
) -> ObservationNormState:
    if not config.normalize_observations:
        return state
    valid = array_is_plausible(observation, config.rollout_observation_max_abs)
    safe_observation = sanitize_array(observation, config.rollout_observation_max_abs)
    values = safe_observation.reshape((-1, safe_observation.shape[-1]))
    batch_count = jnp.asarray(values.shape[0], dtype=jnp.float32)
    batch_mean = jnp.mean(values, axis=0)
    batch_var = jnp.var(values, axis=0)
    delta = batch_mean - state.mean
    total_count = state.count + batch_count
    mean = state.mean + delta * batch_count / total_count
    state_m2 = state.var * state.count
    batch_m2 = batch_var * batch_count
    correction = delta**2 * state.count * batch_count / total_count
    var = (state_m2 + batch_m2 + correction) / total_count
    return ObservationNormState(
        mean=jnp.where(valid, mean, state.mean),
        var=jnp.where(valid, var, state.var),
        count=jnp.where(valid, total_count, state.count),
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
    valid = array_is_plausible(targets, config.target_max_abs)
    safe_targets = sanitize_array(targets, config.target_max_abs)
    batch_count = jnp.asarray(safe_targets.size, dtype=jnp.float32)
    batch_mean = jnp.mean(safe_targets)
    batch_var = jnp.var(safe_targets)
    delta = batch_mean - state.mean
    total_count = state.count + batch_count
    mean = state.mean + delta * batch_count / total_count
    state_m2 = state.var * state.count
    batch_m2 = batch_var * batch_count
    correction = delta**2 * state.count * batch_count / total_count
    var = (state_m2 + batch_m2 + correction) / total_count
    return ValueNormState(
        mean=jnp.where(valid, mean, state.mean),
        var=jnp.where(valid, var, state.var),
        count=jnp.where(valid, total_count, state.count),
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
    entropy_rng: jax.Array | None = None,
    value_norm_state: ValueNormState | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    log_prob, entropy, value = evaluate_actions(
        model,
        transitions.obs,
        transitions.action,
        transitions.critic_obs,
        latent_action=transitions.latent_action,
        entropy_rng=entropy_rng,
    )

    log_ratio = log_prob - transitions.log_prob
    current_log_prob_finite = jnp.all(jnp.isfinite(log_prob))
    log_ratio_finite = jnp.all(jnp.isfinite(log_ratio))
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
    old_value_for_loss = normalize_value(transitions.value, value_norm_state, config)
    targets_for_loss = normalize_value(targets, value_norm_state, config)
    value_clipped = old_value_for_loss + jnp.clip(
        value - old_value_for_loss, -config.clip_eps, config.clip_eps
    )
    value_loss = (
        0.5
        * jnp.maximum(
            (value - targets_for_loss) ** 2,
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
        "nonfinite/current_log_probability": (~current_log_prob_finite).astype(
            jnp.float32
        ),
        "nonfinite/log_ratio": (~log_ratio_finite).astype(jnp.float32),
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
    bounded_policy = config.policy.action_distribution == "tanh_gaussian"
    policy_action_max_abs = (
        config.policy.action_limit + 1.0e-5
        if bounded_policy
        else config.rollout_observation_max_abs
    )
    applied_action_max_abs = (
        config.policy.action_limit + 1.0e-5
        if bounded_policy
        else config.rollout_observation_max_abs
    )
    input_validity = {
        "invalid/rollout_observation": ~(
            array_is_plausible(transitions.obs, config.rollout_observation_max_abs)
            & array_is_plausible(
                transitions.critic_obs, config.rollout_observation_max_abs
            )
        ),
        "invalid/rollout_reward": ~array_is_plausible(
            transitions.reward, config.rollout_reward_max_abs
        ),
        "invalid/rollout_value": ~(
            array_is_plausible(transitions.value, config.rollout_value_max_abs)
            & array_is_plausible(
                transitions.bootstrap_value, config.rollout_value_max_abs
            )
        ),
        "invalid/target": ~(
            array_is_plausible(advantages, config.target_max_abs)
            & array_is_plausible(targets, config.target_max_abs)
        ),
        "invalid/policy_sample": ~(
            array_is_plausible(transitions.action, policy_action_max_abs)
            & array_is_plausible(transitions.env_action, applied_action_max_abs)
            & array_is_plausible(transitions.action_mean, policy_action_max_abs)
            & array_is_plausible(
                transitions.latent_action, config.rollout_observation_max_abs
            )
            & array_is_plausible(transitions.log_prob, config.rollout_value_max_abs)
        ),
    }
    batch_valid = ~jnp.any(jnp.stack(tuple(input_validity.values())))
    safe_transitions = transitions._replace(
        bootstrap_value=sanitize_array(
            transitions.bootstrap_value, config.rollout_value_max_abs
        ),
        action=sanitize_array(transitions.action, policy_action_max_abs),
        latent_action=sanitize_array(
            transitions.latent_action, config.rollout_observation_max_abs
        ),
        env_action=sanitize_array(transitions.env_action, applied_action_max_abs),
        action_mean=sanitize_array(transitions.action_mean, policy_action_max_abs),
        value=sanitize_array(transitions.value, config.rollout_value_max_abs),
        reward=sanitize_array(transitions.reward, config.rollout_reward_max_abs),
        log_prob=sanitize_array(transitions.log_prob, config.rollout_value_max_abs),
        obs=sanitize_array(transitions.obs, config.rollout_observation_max_abs),
        critic_obs=sanitize_array(
            transitions.critic_obs, config.rollout_observation_max_abs
        ),
    )
    advantages = normalize_advantages(sanitize_array(advantages, config.target_max_abs))
    targets = sanitize_array(targets, config.target_max_abs)
    batch = flatten_batch((safe_transitions, advantages, targets))
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
        return jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))

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
            params, opt_state, stopped, key = carry
            key, entropy_key = jax.random.split(key)
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
                    entropy_rng=entropy_key,
                    value_norm_state=value_norm_state,
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

            finite_loss = jnp.isfinite(loss_value)
            finite_metrics = tree_all_finite(metrics)
            finite_gradients = tree_all_finite(grads)
            finite_updates = tree_all_finite(updates)
            finite_params = tree_all_finite(next_params)
            finite_optimizer = tree_all_finite(opt_state)
            finite_policy_probability = (
                metrics["nonfinite/current_log_probability"] == 0
            ) & (metrics["nonfinite/log_ratio"] == 0)
            gradient_norm = optax.tree.norm(grads)
            plausible_gradient = jnp.isfinite(gradient_norm) & (
                gradient_norm <= config.gradient_norm_max
            )
            finite_update = (
                finite_loss
                & finite_metrics
                & finite_gradients
                & finite_updates
                & finite_params
                & finite_optimizer
                & finite_policy_probability
            )
            approx_kl = jnp.where(
                jnp.isfinite(metrics["approx_kl"]),
                metrics["approx_kl"],
                jnp.asarray(jnp.inf, dtype=jnp.float32),
            )
            exceeds_target_kl = approx_kl > target_kl
            should_update = (
                (~stopped)
                & batch_valid
                & finite_update
                & plausible_gradient
                & (~exceeds_target_kl)
            )
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
            stopped = (
                stopped
                | (~batch_valid)
                | exceeds_target_kl
                | (~finite_update)
                | (~plausible_gradient)
            )
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
                "nonfinite/loss": (~finite_loss).astype(jnp.float32),
                "nonfinite/metrics": (~finite_metrics).astype(jnp.float32),
                "nonfinite/gradients": (~finite_gradients).astype(jnp.float32),
                "nonfinite/optimizer_update": (~finite_updates).astype(jnp.float32),
                "nonfinite/parameters": (~finite_params).astype(jnp.float32),
                "nonfinite/optimizer_state": (~finite_optimizer).astype(jnp.float32),
                "invalid/gradient_norm": (~plausible_gradient).astype(jnp.float32),
                "gradient_norm": gradient_norm,
                "target_kl_exceeded": exceeds_target_kl.astype(jnp.float32),
                "ppo_update_fraction": should_update.astype(jnp.float32),
            }
            return (params, opt_state, stopped, key), metrics

        (params, opt_state, stopped, key), metrics = jax.lax.scan(
            update_minibatch,
            (params, opt_state, stopped, key),
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
    metrics = aggregate_ppo_metrics(metrics)
    metrics.update(
        {name: invalid.astype(jnp.float32) for name, invalid in input_validity.items()}
    )
    metrics["invalid_update"] = (
        (~batch_valid) | (metrics["invalid/gradient_norm"] > 0)
    ).astype(jnp.float32)
    train_state = TrainState(
        policy=clamp_policy_log_std(eqx.combine(params, static_policy)),
        opt_state=opt_state,
        update=train_state.update + 1,
    )
    return train_state, metrics


def array_is_plausible(value: jax.Array, max_abs: float) -> jax.Array:
    return jnp.all(jnp.isfinite(value)) & jnp.all(jnp.abs(value) <= max_abs)


def sanitize_array(value: jax.Array, max_abs: float) -> jax.Array:
    return jnp.clip(
        jnp.nan_to_num(value, nan=0.0, posinf=max_abs, neginf=-max_abs),
        -max_abs,
        max_abs,
    )


def aggregate_ppo_metrics(
    metrics: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    max_metric_names = {
        "approx_kl",
        "gradient_norm",
        "nonfinite_update",
        "target_kl_exceeded",
        *(name for name in metrics if name.startswith(("invalid/", "nonfinite/"))),
    }
    mean_approx_kl = metrics["approx_kl"].mean()
    aggregated = {
        name: value.max() if name in max_metric_names else value.mean()
        for name, value in metrics.items()
    }
    aggregated["mean_approx_kl"] = mean_approx_kl
    aggregated["max_approx_kl"] = aggregated["approx_kl"]
    aggregated["approx_kl"] = mean_approx_kl
    return aggregated


def adaptive_learning_rate_kl(
    metrics: dict[str, jax.Array], config: PPOConfig
) -> jax.Array:
    if config.adaptive_kl_statistic == "mean":
        return metrics["mean_approx_kl"]
    return metrics["max_approx_kl"]


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


def sampling_diagnostic_means(info: dict[str, jax.Array]) -> dict[str, jax.Array]:
    return {
        name: value.astype(jnp.float32).mean()
        for name, value in info.items()
        if name.startswith("sampling/")
    }


def command_diagnostic_means(info: dict[str, jax.Array]) -> dict[str, jax.Array]:
    return {
        f"{name}_fraction": value.astype(jnp.float32).mean()
        for name, value in info.items()
        if name.startswith("command/")
    }


def command_conditioned_metrics(
    info: dict[str, jax.Array], reward: jax.Array
) -> dict[str, jax.Array]:
    """Report reward and completed-episode outcomes for each command mode."""

    standing = info.get("command/standing")
    moving = info.get("command/moving")
    if standing is None or moving is None:
        return {}

    metrics = {
        "command/standing_mean_reward": masked_mean_or_zero(reward, standing),
        "command/moving_mean_reward": masked_mean_or_zero(reward, moving),
    }
    returned = info.get("returned_episode")
    episode_return = info.get("episode_return")
    episode_length = info.get("episode_length")
    if returned is None or episode_return is None or episode_length is None:
        return metrics

    for name, mode in (("standing", standing), ("moving", moving)):
        completed = returned.astype(bool) & mode.astype(bool)
        metrics[f"episode/{name}_count"] = jnp.sum(completed.astype(jnp.float32))
        metrics[f"episode/{name}_return"] = masked_mean_or_zero(
            episode_return, completed
        )
        metrics[f"episode/{name}_length"] = masked_mean_or_zero(
            episode_length, completed
        )
    return metrics


def foot_contact_means(info: dict[str, jax.Array]) -> dict[str, jax.Array]:
    return {
        f"mean_{name}": value.astype(jnp.float32).mean()
        for name, value in info.items()
        if name.startswith("foot_contact/")
    }


def target_saturation_means(info: dict[str, jax.Array]) -> dict[str, jax.Array]:
    return {
        f"mean_{name}": value.astype(jnp.float32).mean()
        for name, value in info.items()
        if name.startswith("target/")
    }


def gait_diagnostics(
    info: dict[str, jax.Array], control_timestep: float
) -> dict[str, jax.Array]:
    metrics = {
        f"mean_{name}": value.astype(jnp.float32).mean()
        for name, value in info.items()
        if name.startswith("gait/")
    }
    standing = info.get("command/standing")
    moving = info.get("command/moving")
    if standing is not None and moving is not None:
        for name, value in info.items():
            if not name.startswith("gait/"):
                continue
            suffix = name.removeprefix("gait/")
            metrics[f"gait/standing_{suffix}"] = masked_mean(value, standing)
            metrics[f"gait/moving_{suffix}"] = masked_mean(value, moving)

    touchdown = info.get("gait/touchdown")
    liftoff = info.get("gait/liftoff")
    if touchdown is not None:
        metrics["gait/footfalls_per_second"] = (
            touchdown.astype(jnp.float32).mean() / control_timestep
        )
        if standing is not None and moving is not None:
            metrics["gait/standing_footfalls_per_second"] = (
                masked_mean(touchdown, standing) / control_timestep
            )
            metrics["gait/moving_footfalls_per_second"] = (
                masked_mean(touchdown, moving) / control_timestep
            )
    if liftoff is not None:
        metrics["gait/liftoffs_per_second"] = (
            liftoff.astype(jnp.float32).mean() / control_timestep
        )
    return metrics


def masked_mean(value: jax.Array, mask: jax.Array) -> jax.Array:
    value = value.astype(jnp.float32)
    mask = mask.astype(jnp.float32)
    count = jnp.sum(mask)
    return jnp.where(count > 0, jnp.sum(value * mask) / count, jnp.nan)


def masked_mean_or_zero(value: jax.Array, mask: jax.Array) -> jax.Array:
    value = value.astype(jnp.float32)
    mask = mask.astype(jnp.float32)
    return jnp.sum(value * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def action_diagnostics(
    transitions: Transition,
    env: PPOEnvironment,
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
        "action_clip_fraction": jnp.mean(
            (jnp.abs(action - env_action) > 1.0e-6).astype(jnp.float32)
        ),
    }
    if policy.action_distribution == "tanh_gaussian":
        action_limit = jnp.asarray(policy.action_limit, dtype=action_mean.dtype)
        metrics["action_bound_saturation_fraction"] = jnp.mean(
            (jnp.abs(action_mean) >= 0.98 * action_limit).astype(jnp.float32)
        )
    return {
        **metrics,
        "env_action_abs_max": jnp.max(jnp.abs(env_action)),
        **env.ppo_action_diagnostics(env_action),
    }


def collect_rollout(
    env: PPOEnvironment,
    policy: ActorCritic,
    env_state: Any,
    obs_stack: jax.Array,
    critic_obs_stack: jax.Array,
    rng: jax.Array,
    config: PPOConfig,
    value_norm_state: ValueNormState | None = None,
    observation_norm_state: ObservationNormState | None = None,
    critic_observation_norm_state: ObservationNormState | None = None,
) -> tuple[
    tuple[
        Any,
        jax.Array,
        jax.Array,
        jax.Array,
        ObservationNormState,
        ObservationNormState,
    ],
    Transition,
]:
    if value_norm_state is None:
        value_norm_state = init_value_norm_state()
    if observation_norm_state is None:
        observation_norm_state = init_observation_norm_state(
            int(obs_stack.shape[-2] * obs_stack.shape[-1])
        )
    if critic_observation_norm_state is None:
        critic_observation_norm_state = init_observation_norm_state(
            int(critic_obs_stack.shape[-2] * critic_obs_stack.shape[-1])
        )

    def env_step(
        carry: tuple[
            Any,
            jax.Array,
            jax.Array,
            jax.Array,
            ObservationNormState,
            ObservationNormState,
        ],
        _: None,
    ):
        (
            state,
            last_obs_stack,
            last_critic_obs_stack,
            key,
            actor_norm,
            critic_norm,
        ) = carry
        key, action_key, env_key = jax.random.split(key, 3)
        raw_policy_obs = stacked_observation(last_obs_stack)
        raw_critic_policy_obs = stacked_observation(last_critic_obs_stack)
        actor_norm = update_observation_norm_state(actor_norm, raw_policy_obs, config)
        critic_norm = update_observation_norm_state(
            critic_norm, raw_critic_policy_obs, config
        )
        policy_obs = normalize_observation(raw_policy_obs, actor_norm, config)
        critic_policy_obs = normalize_observation(
            raw_critic_policy_obs, critic_norm, config
        )
        action, latent_action, log_prob, value, action_mean = (
            sample_action_with_latent_and_mean(
                policy,
                policy_obs,
                action_key,
                critic_obs=critic_policy_obs,
            )
        )
        env_action = jnp.clip(action, -env.action_clip, env.action_clip)
        value = denormalize_value(value, value_norm_state, config)
        next_state, next_obs, reward, done, info = env.step_batch(
            state, action, rng=env_key
        )
        next_critic_obs = env.critic_observation_batch(next_state)
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
            normalize_observation(
                stacked_observation(bootstrap_obs_stack), actor_norm, config
            ),
            normalize_observation(
                stacked_observation(bootstrap_critic_obs_stack), critic_norm, config
            ),
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
            latent_action,
            env_action,
            action_mean,
            value,
            reward,
            log_prob,
            policy_obs,
            critic_policy_obs,
            transition_info,
        )
        return (
            next_state,
            next_obs_stack,
            next_critic_obs_stack,
            key,
            actor_norm,
            critic_norm,
        ), transition

    return jax.lax.scan(
        env_step,
        (
            env_state,
            obs_stack,
            critic_obs_stack,
            rng,
            observation_norm_state,
            critic_observation_norm_state,
        ),
        None,
        config.num_steps,
    )


def init_runner_state(
    rng: jax.Array,
    env: PPOEnvironment,
    config: PPOConfig,
) -> tuple[RunnerState, eqx.Module, optax.GradientTransformation]:
    validate_action_limit_contract(config, env.action_clip)
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
    env_state, obs = env.reset_batch(config.num_envs, rng=env_key)
    step_count = env.initial_step_counts(config.num_envs, step_count_key)
    env_state = env_state.replace(step_count=step_count, episode_length=step_count)
    if env.ppo_stagger_initial_step_count:
        rng, staggered_obs_key = jax.random.split(rng)
        obs = env.observation_batch(env_state, staggered_obs_key)
    critic_obs = env.critic_observation_batch(env_state)
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
            rng=rng,
            learning_rate=jnp.asarray(config.learning_rate, dtype=jnp.float32),
            value_norm_state=init_value_norm_state(),
            observation_norm_state=init_observation_norm_state(
                policy_observation_size(env.observation_size, config.policy)
            ),
            critic_observation_norm_state=init_observation_norm_state(
                policy_observation_size(critic_observation_size(env), config.policy)
            ),
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
    env: PPOEnvironment,
    config: PPOConfig,
) -> tuple[RunnerState, dict[str, jax.Array]]:
    validate_action_limit_contract(config, env.action_clip)
    key, rollout_key, update_key = jax.random.split(runner_state.rng, 3)
    policy = eqx.combine(runner_state.params, static_policy)
    train_state = TrainState(
        policy=policy,
        opt_state=runner_state.opt_state,
        update=runner_state.update,
    )
    (
        (
            env_state,
            obs_stack,
            critic_obs_stack,
            _,
            observation_norm_state,
            critic_observation_norm_state,
        ),
        transitions,
    ) = collect_rollout(
        env,
        policy,
        runner_state.env_state,
        runner_state.obs_stack,
        runner_state.critic_obs_stack,
        rollout_key,
        config,
        runner_state.value_norm_state,
        runner_state.observation_norm_state,
        runner_state.critic_observation_norm_state,
    )
    _, _, last_value = apply_policy(
        policy,
        normalize_observation(
            stacked_observation(obs_stack), observation_norm_state, config
        ),
        normalize_observation(
            stacked_observation(critic_obs_stack),
            critic_observation_norm_state,
            config,
        ),
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
        learning_rate,
        adaptive_learning_rate_kl(metrics, config),
        config,
        target_kl_exceeded=metrics["target_kl_exceeded"] > 0,
    )
    params, _ = eqx.partition(train_state.policy, eqx.is_array)
    reward_means = reward_component_means(transitions.info)
    metrics = {
        **metrics,
        "learning_rate": learning_rate,
        "next_learning_rate": next_learning_rate,
        "value_norm_mean": value_norm_state.mean,
        "value_norm_std": value_norm_std(value_norm_state, config),
        "value_norm_count": value_norm_state.count,
        "observation_norm_count": observation_norm_state.count,
        "critic_observation_norm_count": critic_observation_norm_state.count,
        "mean_reward": transitions.reward.mean(),
        "mean_done": transitions.done.astype(jnp.float32).mean(),
        "nonfinite/rollout_observation": (
            ~jnp.all(jnp.isfinite(transitions.obs))
        ).astype(jnp.float32),
        "nonfinite/rollout_critic_observation": (
            ~jnp.all(jnp.isfinite(transitions.critic_obs))
        ).astype(jnp.float32),
        "nonfinite/rollout_reward": (~jnp.all(jnp.isfinite(transitions.reward))).astype(
            jnp.float32
        ),
        "nonfinite/rollout_action": (
            ~jnp.all(jnp.isfinite(transitions.env_action))
        ).astype(jnp.float32),
        "mean_episode_return": safe_returned_mean(transitions.info),
        "max_episode_return": safe_returned_max(transitions.info),
        "mean_episode_length": safe_returned_mean(
            transitions.info, key="episode_length"
        ),
        "max_episode_length": safe_returned_max(transitions.info, key="episode_length"),
        **action_diagnostics(transitions, env, policy),
        **done_reason_means(transitions.info),
        **random_push_means(transitions.info),
        **sampling_diagnostic_means(transitions.info),
        **command_diagnostic_means(transitions.info),
        **command_conditioned_metrics(transitions.info, transitions.reward),
        **foot_contact_means(transitions.info),
        **target_saturation_means(transitions.info),
        **gait_diagnostics(transitions.info, env.control_timestep),
        **reward_means,
    }
    return (
        RunnerState(
            params=params,
            opt_state=train_state.opt_state,
            update=train_state.update,
            env_state=env_state,
            obs_stack=obs_stack,
            critic_obs_stack=critic_obs_stack,
            rng=key,
            learning_rate=next_learning_rate,
            value_norm_state=value_norm_state,
            observation_norm_state=observation_norm_state,
            critic_observation_norm_state=critic_observation_norm_state,
        ),
        metrics,
    )


def validate_action_limit_contract(config: PPOConfig, env_action_clip: float) -> None:
    if config.policy.action_distribution == "gaussian":
        return
    if config.policy.action_limit != env_action_clip:
        raise ValueError(
            "config.policy.action_limit must equal env.action_clip so PPO "
            "log-probabilities describe the actions applied by the environment, got "
            f"{config.policy.action_limit} != {env_action_clip}"
        )


def critic_observation_size(env: PPOEnvironment) -> int:
    return env.critic_observation_size


def _validate_declared_fields(config: PPOConfig) -> None:
    for config_field in fields(config):
        if _VALIDATOR_METADATA_KEY not in config_field.metadata:
            raise RuntimeError(
                f"PPOConfig.{config_field.name} has no declared validation"
            )
        validator = config_field.metadata[_VALIDATOR_METADATA_KEY]
        if not callable(validator):
            raise RuntimeError(
                f"PPOConfig.{config_field.name} has an invalid validator"
            )
        validator(config_field.name, getattr(config, config_field.name))


def validate_ppo_config(config: PPOConfig) -> None:
    _validate_declared_fields(config)
    if config.num_updates <= 0:
        raise ValueError(
            "total_timesteps must cover at least one rollout batch, got "
            f"total_timesteps={config.total_timesteps}, "
            f"num_envs={config.num_envs}, num_steps={config.num_steps}"
        )
    if config.batch_size % config.num_minibatches != 0:
        raise ValueError("num_envs * num_steps must be divisible by num_minibatches")
    if config.min_learning_rate > config.max_learning_rate:
        raise ValueError(
            "min_learning_rate must be <= max_learning_rate, got "
            f"{config.min_learning_rate} > {config.max_learning_rate}"
        )
    if effective_learning_rate_schedule(config) == "adaptive" and not (
        config.min_learning_rate <= config.learning_rate <= config.max_learning_rate
    ):
        raise ValueError(
            "adaptive learning_rate must be within "
            "[min_learning_rate, max_learning_rate], got "
            f"{config.learning_rate} outside "
            f"[{config.min_learning_rate}, {config.max_learning_rate}]"
        )
    if any(hidden_dim <= 0 for hidden_dim in config.policy.hidden_dims):
        raise ValueError(
            "policy.hidden_dims entries must be positive, got "
            f"{config.policy.hidden_dims}"
        )
    if config.policy.obs_stack_size <= 0:
        raise ValueError(
            "policy.obs_stack_size must be positive, got "
            f"{config.policy.obs_stack_size}"
        )
    if config.policy.activation not in {"tanh", "relu", "elu", "swish"}:
        raise ValueError(
            "policy.activation must be 'tanh', 'relu', 'elu', or 'swish', got "
            f"{config.policy.activation!r}"
        )
    if config.policy.action_distribution not in {"gaussian", "tanh_gaussian"}:
        raise ValueError(
            "policy.action_distribution must be 'gaussian' or 'tanh_gaussian', got "
            f"{config.policy.action_distribution!r}"
        )
    if not math.isfinite(config.policy.action_limit) or config.policy.action_limit <= 0:
        raise ValueError(
            "policy.action_limit must be positive and finite, got "
            f"{config.policy.action_limit}"
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
    if not (
        config.policy.min_log_std
        <= config.policy.init_log_std
        <= config.policy.max_log_std
    ):
        raise ValueError(
            "policy.init_log_std must be within "
            "[policy.min_log_std, policy.max_log_std], got "
            f"{config.policy.init_log_std} outside "
            f"[{config.policy.min_log_std}, {config.policy.max_log_std}]"
        )
