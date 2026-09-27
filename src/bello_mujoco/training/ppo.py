"""JAX PPO optimization and rollout collection."""

from __future__ import annotations

import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from bello_mujoco.training.ppo_metrics import (
    safe_returned_mean,
    safe_returned_max,
    environment_metric_means,
    command_conditioned_metrics,
    physics_diagnostics,
    gait_diagnostics,
    action_diagnostics,
)
from bello_mujoco.training.normalization import (
    init_value_norm_state,
    init_observation_norm_state,
    normalize_observation,
    normalize_critic_observation,
    update_observation_norm_state,
    update_critic_observation_norm_state,
    value_norm_std,
    normalize_value,
    denormalize_value,
    update_value_norm_state,
    array_is_plausible,
    sanitize_array,
)
from bello_mujoco.training.ppo_types import (
    PPOEnvironment,
    TrainState,
    Transition,
    ValueNormState,
    ObservationNormState,
    RunnerState,
)
from bello_mujoco.training.ppo_config import (
    PPOConfig,
    parameter_learning_rate,
    validate_action_limit_contract,
    validate_ppo_config,
)

from bello_mujoco.policy import (
    ActorCritic,
    apply_policy,
    clamp_policy_log_std,
    evaluate_actions,
    init_observation_stack,
    init_actor_critic,
    policy_observation_size,
    sample_action_with_latent_and_mean,
    stacked_observation,
    update_observation_stack,
)

LOG_RATIO_CLIP = 20.0


def policy_trainable_filter(policy):
    """Return a model's explicit trainability mask or the default array mask."""

    return (
        policy.trainable_filter()
        if hasattr(policy, "trainable_filter")
        else eqx.is_array
    )


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
            optax.scale_by_adam(eps=config.adam_epsilon),
        ),
        max_consecutive_errors=1,
    )


def learning_rate_for_update(
    config: PPOConfig,
    update: jax.Array,
    current_learning_rate: jax.Array,
) -> jax.Array:
    schedule = config.schedule
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
    if config.schedule != "adaptive":
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
    value_loss = config.value_loss_scale * (
        jnp.maximum(
            (value - targets_for_loss) ** 2,
            (value_clipped - targets_for_loss) ** 2,
        ).mean()
    )
    value_loss_raw = config.value_loss_scale * ((raw_value - targets) ** 2).mean()
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
        "policy_clip_fraction": jnp.mean(
            (jnp.abs(ratio - 1.0) > config.clip_eps).astype(jnp.float32)
        ),
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
    trainable_filter = policy_trainable_filter(train_state.policy)
    init_params, static_policy = eqx.partition(train_state.policy, trainable_filter)
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
            updates = jax.tree_util.tree_map_with_path(
                lambda path, update: (
                    -parameter_learning_rate(config, learning_rate, path) * update
                ),
                updates,
            )
            next_params = optax.apply_updates(params, updates)
            policy = clamp_policy_log_std(eqx.combine(next_params, static_policy))
            next_params, _ = eqx.partition(policy, trainable_filter)

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


def collect_rollout(
    env: PPOEnvironment,
    policy: ActorCritic,
    env_state: Any,
    obs_stack: jax.Array,
    critic_obs_stack: jax.Array,
    rng: jax.Array,
    config: PPOConfig,
    value_norm_state: ValueNormState,
    observation_norm_state: ObservationNormState,
    critic_observation_norm_state: ObservationNormState,
    task_state: Any,
) -> tuple[
    tuple[
        Any,
        jax.Array,
        jax.Array,
        jax.Array,
        ObservationNormState,
        ObservationNormState,
        Any,
    ],
    Transition,
]:
    observation_norm_max_count = (
        None
        if config.observation_norm_update_updates is None
        else float(
            config.observation_norm_update_updates * config.num_envs * config.num_steps
        )
    )
    rollout_normalization = config.observation_norm_update_mode == "rollout"

    def env_step(
        carry: tuple[
            Any,
            jax.Array,
            jax.Array,
            jax.Array,
            ObservationNormState,
            ObservationNormState,
            Any,
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
            current_task_state,
        ) = carry
        key, action_key, env_key = jax.random.split(key, 3)
        raw_policy_obs = stacked_observation(last_obs_stack)
        raw_critic_policy_obs = stacked_observation(last_critic_obs_stack)
        if rollout_normalization:
            policy_actor_norm = observation_norm_state
        else:
            actor_norm = update_observation_norm_state(
                actor_norm,
                raw_policy_obs,
                config,
                max_count=observation_norm_max_count,
            )
            policy_actor_norm = actor_norm
        critic_norm = update_critic_observation_norm_state(
            critic_norm, raw_critic_policy_obs, config
        )
        policy_obs = normalize_observation(raw_policy_obs, policy_actor_norm, config)
        critic_policy_obs = normalize_critic_observation(
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
        current_env = env.with_task_state(current_task_state)
        env_action = jnp.clip(action, -current_env.action_clip, current_env.action_clip)
        value = denormalize_value(value, value_norm_state, config)
        next_state, next_obs, reward, done, info = current_env.step_batch(
            state, action, rng=env_key
        )
        next_task_state = current_env.update_task_state(current_task_state, info)
        next_critic_obs = current_env.critic_observation_batch(next_state)
        next_obs_stack = update_observation_stack(last_obs_stack, next_obs, done)
        next_critic_obs_stack = update_observation_stack(
            last_critic_obs_stack, next_critic_obs, done
        )
        if rollout_normalization:
            actor_norm = update_observation_norm_state(
                actor_norm,
                stacked_observation(next_obs_stack),
                config,
                max_count=observation_norm_max_count,
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
                stacked_observation(bootstrap_obs_stack), policy_actor_norm, config
            ),
            normalize_critic_observation(
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
            next_task_state,
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
            task_state,
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
            env.critic_observation_size, config.policy
        ),
    )
    params, static_policy = eqx.partition(train_state.policy, eqx.is_array)
    task_state = env.initial_task_state()
    initial_env = env.with_task_state(task_state)
    env_state, obs = initial_env.reset_batch(config.num_envs, rng=env_key)
    step_count = env.initial_step_counts(config.num_envs, step_count_key)
    env_state = initial_env.with_initial_step_counts(env_state, step_count)
    if env.ppo_stagger_initial_step_count:
        rng, staggered_obs_key = jax.random.split(rng)
        obs = initial_env.observation_batch(env_state, staggered_obs_key)
    critic_obs = initial_env.critic_observation_batch(env_state)
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
                policy_observation_size(env.critic_observation_size, config.policy)
            ),
            task_state=task_state,
        ),
        static_policy,
        optimizer,
    )


def init_runner_state_with_policy(
    rng: jax.Array,
    env: PPOEnvironment,
    config: PPOConfig,
    policy: eqx.Module,
) -> tuple[RunnerState, eqx.Module, optax.GradientTransformation]:
    """Initialize the shared PPO runner from an imported, non-random policy."""

    validate_action_limit_contract(config, env.action_clip)
    rng, env_key, step_count_key = jax.random.split(rng, 3)
    optimizer = make_optimizer(config)
    filter_spec = policy_trainable_filter(policy)
    params, static_policy = eqx.partition(policy, filter_spec)
    opt_state = optimizer.init(params)
    task_state = env.initial_task_state()
    initial_env = env.with_task_state(task_state)
    env_state, obs = initial_env.reset_batch(config.num_envs, rng=env_key)
    step_count = env.initial_step_counts(config.num_envs, step_count_key)
    env_state = initial_env.with_initial_step_counts(env_state, step_count)
    if env.ppo_stagger_initial_step_count:
        rng, staggered_obs_key = jax.random.split(rng)
        obs = initial_env.observation_batch(env_state, staggered_obs_key)
    critic_obs = initial_env.critic_observation_batch(env_state)
    obs_stack = init_observation_stack(obs, config.policy.obs_stack_size)
    critic_obs_stack = init_observation_stack(critic_obs, config.policy.obs_stack_size)
    expected_actor = policy_observation_size(env.observation_size, config.policy)
    expected_critic = policy_observation_size(
        env.critic_observation_size, config.policy
    )
    if getattr(policy.actor, "observation_size", expected_actor) != expected_actor:
        raise ValueError("imported actor observation layout does not match environment")
    if getattr(policy.critic, "observation_size", expected_critic) != expected_critic:
        raise ValueError(
            "imported critic observation layout does not match environment"
        )
    return (
        RunnerState(
            params=params,
            opt_state=opt_state,
            update=jnp.asarray(0, dtype=jnp.int32),
            env_state=env_state,
            obs_stack=obs_stack,
            critic_obs_stack=critic_obs_stack,
            rng=rng,
            learning_rate=jnp.asarray(config.learning_rate, dtype=jnp.float32),
            value_norm_state=init_value_norm_state(),
            observation_norm_state=init_observation_norm_state(expected_actor),
            critic_observation_norm_state=init_observation_norm_state(expected_critic),
            task_state=task_state,
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
            task_state,
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
        runner_state.task_state,
    )
    _, _, last_value = apply_policy(
        policy,
        normalize_observation(
            stacked_observation(obs_stack), observation_norm_state, config
        ),
        normalize_critic_observation(
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
    params, _ = eqx.partition(
        train_state.policy, policy_trainable_filter(train_state.policy)
    )
    task_metrics = env.task_state_metrics(task_state)
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
        **environment_metric_means(transitions.info),
        **command_conditioned_metrics(transitions.info, transitions.reward),
        **physics_diagnostics(transitions.info),
        **gait_diagnostics(transitions.info, env.control_timestep),
        **task_metrics,
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
            task_state=task_state,
        ),
        metrics,
    )
