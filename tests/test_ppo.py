import jax
import jax.numpy as jnp
import pytest

from bello_mujoco.policy import (
    ActorCriticConfig,
    apply_actor,
    apply_policy,
    evaluate_actions,
    init_observation_stack,
    policy_log_prob,
    sample_actor_action,
    sample_action,
    stacked_observation,
    update_observation_stack,
    update_policy_observation_normalization,
)
from bello_mujoco.train.ppo import (
    PPOConfig,
    Transition,
    adapt_learning_rate,
    compute_gae,
    denormalize_value,
    init_train_state,
    init_value_norm_state,
    learning_rate_for_update,
    normalize_advantages,
    normalize_value,
    update_ppo,
    update_value_norm_state,
    validate_ppo_config,
)


def small_config(**updates) -> PPOConfig:
    values = dict(
        total_timesteps=32,
        num_envs=4,
        num_steps=8,
        update_epochs=2,
        num_minibatches=2,
        learning_rate=3.0e-4,
        min_learning_rate=1.0e-5,
        max_learning_rate=1.0e-3,
        policy=ActorCriticConfig(
            hidden_size=16,
            hidden_depth=1,
            obs_stack_size=1,
            init_log_std=-1.0,
            action_limit=1.0,
        ),
    )
    values.update(updates)
    return PPOConfig(**values)


def transitions(
    config: PPOConfig,
    policy,
    *,
    obs_size: int = 3,
    critic_obs_size: int | None = None,
    reward: float = 1.0,
) -> Transition:
    critic_obs_size = obs_size if critic_obs_size is None else critic_obs_size
    shape = (config.num_steps, config.num_envs)
    obs = jnp.zeros(shape + (obs_size,), dtype=jnp.float32)
    critic_obs = jnp.zeros(shape + (critic_obs_size,), dtype=jnp.float32)
    action = jnp.zeros(shape + (policy.log_std.size,), dtype=jnp.float32)
    flat_obs = obs.reshape((-1, obs_size))
    flat_critic_obs = critic_obs.reshape((-1, critic_obs_size))
    latent_mean, log_std, value = apply_policy(policy, flat_obs, flat_critic_obs)
    latent_mean = latent_mean.reshape(action.shape)
    log_std = log_std.reshape(action.shape)
    value = value.reshape(shape)
    log_prob = policy_log_prob(
        policy, latent_mean, log_std, action, latent_action=action
    )
    return Transition(
        done=jnp.zeros(shape, dtype=bool),
        terminal=jnp.zeros(shape, dtype=bool),
        bootstrap_value=jnp.zeros(shape, dtype=jnp.float32),
        action=action,
        latent_action=action,
        env_action=action,
        action_mean=apply_actor(policy, flat_obs).reshape(action.shape),
        value=value,
        reward=jnp.full(shape, reward, dtype=jnp.float32),
        log_prob=log_prob,
        obs=obs,
        critic_obs=critic_obs,
        info={},
    )


def test_actor_critic_has_explicit_asymmetric_shapes_and_bounded_actions() -> None:
    config = small_config()
    state, _ = init_train_state(jax.random.PRNGKey(0), 5, 2, config, critic_obs_size=7)
    actor_obs = jnp.zeros((config.num_envs, 5), dtype=jnp.float32)
    critic_obs = jnp.zeros((config.num_envs, 7), dtype=jnp.float32)
    action, log_prob, value = sample_action(
        state.policy,
        actor_obs,
        jax.random.PRNGKey(1),
        critic_obs=critic_obs,
    )
    assert action.shape == (4, 2)
    assert log_prob.shape == (4,)
    assert value.shape == (4,)
    assert jnp.all(jnp.abs(action) < 1.0)


def test_sampled_action_log_probability_matches_evaluation() -> None:
    config = small_config(
        policy=ActorCriticConfig(
            hidden_size=8,
            hidden_depth=1,
            obs_stack_size=1,
            init_log_std=-3.0,
            action_limit=0.75,
        )
    )
    state, _ = init_train_state(jax.random.PRNGKey(0), 4, 2, config)
    obs = jnp.zeros((4, 4), dtype=jnp.float32)
    action, sampled_log_prob, _ = sample_action(
        state.policy, obs, jax.random.PRNGKey(1)
    )
    evaluated_log_prob, _, _ = evaluate_actions(state.policy, obs, action)
    assert jnp.allclose(sampled_log_prob, evaluated_log_prob, atol=1.0e-5)


def test_actor_only_sampling_is_bounded_with_asymmetric_critic() -> None:
    config = small_config(
        policy=ActorCriticConfig(
            hidden_size=8,
            hidden_depth=1,
            obs_stack_size=1,
            init_log_std=1.0,
            action_limit=0.6,
        )
    )
    state, _ = init_train_state(jax.random.PRNGKey(0), 3, 2, config, critic_obs_size=5)
    action = sample_actor_action(state.policy, jnp.zeros((4, 3)), jax.random.PRNGKey(1))

    assert action.shape == (4, 2)
    assert jnp.all(jnp.abs(action) <= 0.6)


def test_exact_latent_keeps_saturated_tanh_ratio_finite() -> None:
    config = small_config(num_steps=1, total_timesteps=4, num_minibatches=1)
    state, _ = init_train_state(jax.random.PRNGKey(0), 2, 1, config)
    obs = jnp.zeros((4, 2), dtype=jnp.float32)
    mean, log_std, _ = apply_policy(state.policy, obs)
    latent = jnp.full((4, 1), 20.0, dtype=jnp.float32)
    action = jnp.tanh(latent)
    exact = policy_log_prob(state.policy, mean, log_std, action, latent_action=latent)
    reconstructed = policy_log_prob(state.policy, mean, log_std, action)
    assert jnp.all(jnp.isfinite(exact))
    assert not jnp.allclose(exact, reconstructed)


def test_observation_stack_shifts_and_resets_per_environment() -> None:
    initial = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    stack = init_observation_stack(initial, 3)
    updated = update_observation_stack(
        stack,
        jnp.asarray([[5.0, 6.0], [7.0, 8.0]]),
        jnp.asarray([False, True]),
    )
    assert stacked_observation(updated[0]) == pytest.approx(
        jnp.asarray([1.0, 2.0, 1.0, 2.0, 5.0, 6.0])
    )
    assert updated[1] == pytest.approx(jnp.tile(jnp.asarray([7.0, 8.0]), (3, 1)))


def test_actor_and_critic_normalization_are_independent() -> None:
    config = small_config(
        policy=ActorCriticConfig(
            hidden_size=8,
            hidden_depth=1,
            obs_stack_size=1,
            normalize_observations=True,
        )
    )
    state, _ = init_train_state(jax.random.PRNGKey(0), 2, 1, config, critic_obs_size=3)
    policy = update_policy_observation_normalization(
        state.policy,
        jnp.asarray([[1.0, 2.0], [3.0, 6.0]]),
        jnp.asarray([[2.0, 4.0, 8.0], [6.0, 8.0, 10.0]]),
    )
    assert policy.actor_obs_mean == pytest.approx(jnp.asarray([2.0, 4.0]))
    assert policy.critic_obs_mean == pytest.approx(jnp.asarray([4.0, 6.0, 9.0]))
    assert policy.actor_obs_count == 2
    assert policy.critic_obs_count == 2


def test_gae_bootstraps_timeout_but_not_terminal_transition() -> None:
    config = small_config(num_steps=1, total_timesteps=4, gamma=0.9)
    state, _ = init_train_state(jax.random.PRNGKey(0), 3, 1, config)
    base = transitions(config, state.policy)
    timeout = base._replace(
        done=jnp.ones((1, 4), dtype=bool),
        terminal=jnp.zeros((1, 4), dtype=bool),
        bootstrap_value=jnp.full((1, 4), 2.0),
        reward=jnp.ones((1, 4)),
        value=jnp.zeros((1, 4)),
    )
    terminal = timeout._replace(terminal=jnp.ones((1, 4), dtype=bool))
    timeout_advantage, _ = compute_gae(timeout, jnp.zeros(4), config)
    terminal_advantage, _ = compute_gae(terminal, jnp.zeros(4), config)
    assert timeout_advantage == pytest.approx(jnp.full((1, 4), 2.8))
    assert terminal_advantage == pytest.approx(jnp.ones((1, 4)))


def test_value_normalization_round_trip_and_statistics() -> None:
    config = small_config(normalize_value=True)
    state = update_value_norm_state(
        init_value_norm_state(), jnp.asarray([1.0, 3.0, 5.0]), config
    )
    values = jnp.asarray([-2.0, 0.0, 7.0])
    assert denormalize_value(normalize_value(values, state, config), state, config) == (
        pytest.approx(values)
    )
    assert state.mean == pytest.approx(3.0)
    assert state.count == pytest.approx(3.0)


def test_advantage_normalization_is_zero_mean_and_unit_scale() -> None:
    normalized = normalize_advantages(jnp.asarray([1.0, 2.0, 4.0, 8.0]))
    assert normalized.mean() == pytest.approx(0.0, abs=1.0e-6)
    assert normalized.std() == pytest.approx(1.0, abs=1.0e-6)


def test_learning_rate_schedules_have_no_legacy_fallback() -> None:
    linear = small_config(schedule="linear", total_timesteps=64)
    constant = small_config(schedule="constant")
    current = jnp.asarray(2.0e-4)
    assert (
        learning_rate_for_update(linear, jnp.asarray(1), current) < linear.learning_rate
    )
    assert learning_rate_for_update(constant, jnp.asarray(1), current) == pytest.approx(
        constant.learning_rate
    )


def test_adaptive_learning_rate_responds_to_kl_and_clamps() -> None:
    config = small_config(schedule="adaptive", desired_kl=0.01)
    current = jnp.asarray(3.0e-4)
    assert adapt_learning_rate(current, jnp.asarray(0.001), config) > current
    assert adapt_learning_rate(current, jnp.asarray(0.1), config) < current
    assert adapt_learning_rate(current, jnp.asarray(0.01), config) == current


def test_finite_ppo_update_changes_parameters() -> None:
    config = small_config()
    train_state, optimizer = init_train_state(jax.random.PRNGKey(0), 3, 2, config)
    batch = transitions(config, train_state.policy)
    advantages = jnp.ones((config.num_steps, config.num_envs))
    targets = jnp.ones_like(advantages)
    next_state, metrics = update_ppo(
        train_state,
        optimizer,
        batch,
        advantages,
        targets,
        jax.random.PRNGKey(1),
        config,
    )
    assert next_state.update == 1
    assert metrics["nonfinite_update"] == 0
    assert metrics["ppo_update_fraction"] > 0
    assert jnp.isfinite(metrics["loss"])


def test_nonfinite_gradient_is_identified_and_skipped() -> None:
    config = small_config()
    train_state, optimizer = init_train_state(jax.random.PRNGKey(0), 3, 2, config)
    batch = transitions(config, train_state.policy)
    bad = jnp.full((config.num_steps, config.num_envs), jnp.nan)
    _, metrics = update_ppo(
        train_state,
        optimizer,
        batch,
        bad,
        jnp.ones_like(bad),
        jax.random.PRNGKey(1),
        config,
    )
    assert metrics["nonfinite_update"] > 0
    assert metrics["nonfinite/loss"] > 0
    assert metrics["ppo_update_fraction"] == 0


@pytest.mark.parametrize(
    "config, message",
    [
        (small_config(num_minibatches=3), "divisible"),
        (small_config(schedule="unknown"), "schedule"),
        (
            small_config(policy=ActorCriticConfig(min_log_std=1.0, max_log_std=-1.0)),
            "min_log_std",
        ),
    ],
)
def test_ppo_validation_rejects_invalid_contracts(
    config: PPOConfig, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_ppo_config(config)
