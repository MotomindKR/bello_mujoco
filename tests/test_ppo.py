from dataclasses import replace
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from bello_mujoco.assets_pipeline.prepare import PreparedAsset
from bello_mujoco.envs.locomotion import BelloEnvConfig, BelloLocomotionEnv, SpawnConfig
from bello_mujoco.policy import (
    ActorCriticConfig,
    apply_actor,
    apply_policy,
    evaluate_actions,
    gaussian_entropy,
    init_observation_stack,
    policy_log_prob,
    reparameterized_policy_entropy,
    sample_actor_action,
    sample_action,
    stacked_observation,
    update_observation_stack,
)
from bello_mujoco.training.ppo import (
    PPOConfig,
    Transition,
    action_diagnostics,
    adapt_learning_rate,
    adaptive_learning_rate_kl,
    aggregate_ppo_metrics,
    command_diagnostic_means,
    compute_gae,
    denormalize_value,
    init_observation_norm_state,
    init_runner_state,
    init_train_state,
    init_value_norm_state,
    gait_diagnostics,
    learning_rate_for_update,
    make_train,
    normalize_advantages,
    normalize_observation,
    normalize_value,
    update_observation_norm_state,
    update_ppo,
    update_value_norm_state,
    validate_action_limit_contract,
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
            hidden_dims=(16,),
            obs_stack_size=1,
            init_log_std=-1.0,
            action_limit=1.0,
        ),
    )
    values.update(updates)
    return PPOConfig(**values)


def test_command_and_gait_metrics_split_standing_from_moving() -> None:
    info = {
        "command/standing": jnp.asarray([[1.0, 0.0, 1.0, 0.0]]),
        "command/moving": jnp.asarray([[0.0, 1.0, 0.0, 1.0]]),
        "gait/touchdown": jnp.asarray([[2.0, 1.0, 0.0, 1.0]]),
        "gait/double_support": jnp.asarray([[1.0, 0.0, 1.0, 0.0]]),
    }

    command_metrics = command_diagnostic_means(info)
    gait_metrics = gait_diagnostics(info, control_timestep=0.02)

    assert command_metrics["command/standing_fraction"] == pytest.approx(0.5)
    assert command_metrics["command/moving_fraction"] == pytest.approx(0.5)
    assert gait_metrics["gait/standing_touchdown"] == pytest.approx(1.0)
    assert gait_metrics["gait/moving_touchdown"] == pytest.approx(1.0)
    assert gait_metrics["gait/standing_double_support"] == pytest.approx(1.0)
    assert gait_metrics["gait/moving_double_support"] == pytest.approx(0.0)
    assert gait_metrics["gait/standing_footfalls_per_second"] == pytest.approx(50.0)
    assert gait_metrics["gait/moving_footfalls_per_second"] == pytest.approx(50.0)


def test_library_training_rejects_policy_environment_action_limit_mismatch() -> None:
    env = SimpleNamespace(action_clip=0.5)

    with pytest.raises(ValueError, match="action_limit must equal.*action_clip"):
        make_train(small_config(), env)


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
            hidden_dims=(8,),
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
            hidden_dims=(8,),
            obs_stack_size=1,
            init_log_std=1.0,
            action_limit=0.6,
        )
    )
    state, _ = init_train_state(jax.random.PRNGKey(0), 3, 2, config, critic_obs_size=5)
    action = sample_actor_action(state.policy, jnp.zeros((4, 3)), jax.random.PRNGKey(1))

    assert action.shape == (4, 2)
    assert jnp.all(jnp.abs(action) <= 0.6)


def test_gaussian_actions_are_unsquashed_with_exact_log_prob_and_entropy() -> None:
    config = small_config(
        policy=ActorCriticConfig(
            hidden_dims=(8,),
            obs_stack_size=1,
            init_log_std=0.0,
            action_distribution="gaussian",
            action_limit=0.1,
        )
    )
    state, _ = init_train_state(jax.random.PRNGKey(0), 3, 2, config)
    obs = jnp.zeros((2048, 3), dtype=jnp.float32)
    action, sampled_log_prob, _ = sample_action(
        state.policy, obs, jax.random.PRNGKey(1)
    )
    evaluated_log_prob, entropy, _ = evaluate_actions(
        state.policy,
        obs,
        action,
        entropy_rng=jax.random.PRNGKey(2),
    )
    _, log_std, _ = apply_policy(state.policy, obs)

    assert jnp.any(jnp.abs(action) > config.policy.action_limit)
    assert sampled_log_prob == pytest.approx(evaluated_log_prob, abs=1.0e-5)
    assert entropy == pytest.approx(gaussian_entropy(log_std))
    validate_action_limit_contract(config, env_action_clip=100.0)


def test_gaussian_samples_above_tanh_limit_are_not_rejected() -> None:
    config = small_config(
        policy=ActorCriticConfig(
            hidden_dims=(8,),
            obs_stack_size=1,
            init_log_std=-1.0,
            action_distribution="gaussian",
            action_limit=1.0,
        )
    )
    train_state, optimizer = init_train_state(jax.random.PRNGKey(0), 3, 2, config)
    batch = transitions(config, train_state.policy)
    action = jnp.full_like(batch.action, 2.0)
    flat_obs = batch.obs.reshape((-1, batch.obs.shape[-1]))
    mean, log_std, _ = apply_policy(train_state.policy, flat_obs)
    log_prob = policy_log_prob(
        train_state.policy,
        mean.reshape(action.shape),
        log_std.reshape(action.shape),
        action,
        latent_action=action,
    )
    batch = batch._replace(
        action=action,
        latent_action=action,
        env_action=action,
        log_prob=log_prob,
    )

    _, metrics = update_ppo(
        train_state,
        optimizer,
        batch,
        jnp.ones((config.num_steps, config.num_envs)),
        jnp.ones((config.num_steps, config.num_envs)),
        jax.random.PRNGKey(1),
        config,
    )

    assert metrics["invalid/policy_sample"] == 0
    assert metrics["ppo_update_fraction"] > 0


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


def test_observation_normalization_updates_round_trips_and_rejects_poison() -> None:
    config = small_config(normalize_observations=True)
    state = init_observation_norm_state(2)
    observation = jnp.asarray([[1.0, 3.0], [3.0, 7.0]], dtype=jnp.float32)
    updated = update_observation_norm_state(state, observation, config)

    assert updated.mean == pytest.approx(jnp.asarray([2.0, 5.0]))
    assert updated.var == pytest.approx(jnp.asarray([1.0, 4.0]))
    assert updated.count == 2
    assert normalize_observation(observation, updated, config) == pytest.approx(
        jnp.asarray([[-1.0, -1.0], [1.0, 1.0]])
    )

    poisoned = update_observation_norm_state(
        updated, observation.at[0, 0].set(jnp.nan), config
    )
    assert all(
        jnp.array_equal(before, after)
        for before, after in zip(updated, poisoned, strict=True)
    )


def test_value_normalization_round_trips_and_rejects_poison() -> None:
    config = small_config(normalize_value=True)
    initial = init_value_norm_state()
    targets = jnp.asarray([1.0, 3.0], dtype=jnp.float32)
    updated = update_value_norm_state(initial, targets, config)
    normalized = normalize_value(targets, updated, config)

    assert updated.mean == pytest.approx(2.0)
    assert updated.var == pytest.approx(1.0)
    assert updated.count == 2
    assert denormalize_value(normalized, updated, config) == pytest.approx(targets)

    poisoned = update_value_norm_state(
        updated, targets.at[0].set(config.target_max_abs * 2.0), config
    )
    assert poisoned == updated


def test_disabled_normalization_preserves_locomotion_values() -> None:
    config = small_config()
    observation_state = init_observation_norm_state(2)
    value_state = init_value_norm_state()
    observation = jnp.asarray([[2.0, -3.0]], dtype=jnp.float32)
    values = jnp.asarray([2.0, -3.0], dtype=jnp.float32)

    assert normalize_observation(observation, observation_state, config) is observation
    assert (
        update_observation_norm_state(observation_state, observation, config)
        is observation_state
    )
    assert normalize_value(values, value_state, config) is values
    assert denormalize_value(values, value_state, config) is values
    assert update_value_norm_state(value_state, values, config) is value_state


def test_staggered_initial_step_counts_match_actor_and_critic_observations(
    full_body_asset: PreparedAsset,
) -> None:
    num_envs = 4
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        batch_size=num_envs,
        env_config=replace(
            BelloEnvConfig(),
            spawn=SpawnConfig(stagger_initial_step_count=True),
        ),
    )
    config = small_config(
        num_envs=num_envs,
        policy=ActorCriticConfig(
            hidden_dims=(8,),
            obs_stack_size=2,
            init_log_std=-1.0,
        ),
    )

    runner, _, _ = init_runner_state(jax.random.PRNGKey(19), env, config)
    expected_actor = env.observation_batch(runner.env_state)
    expected_critic = env.critic_observation_batch(runner.env_state)

    assert jnp.any(runner.env_state.step_count > 0)
    assert runner.obs_stack[:, 0] == pytest.approx(expected_actor)
    assert runner.obs_stack[:, 1] == pytest.approx(expected_actor)
    assert runner.critic_obs_stack[:, 0] == pytest.approx(expected_critic)
    assert runner.critic_obs_stack[:, 1] == pytest.approx(expected_critic)


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


def test_squashed_entropy_gradient_moves_saturated_mean_toward_center() -> None:
    config = small_config()
    state, _ = init_train_state(jax.random.PRNGKey(0), 2, 1, config)
    log_std = jnp.full((4096, 1), -1.0)
    key = jax.random.PRNGKey(11)

    def entropy(mean):
        latent_mean = jnp.full((4096, 1), mean)
        return reparameterized_policy_entropy(
            state.policy, latent_mean, log_std, key
        ).mean()

    assert jax.grad(entropy)(jnp.asarray(2.0)) < 0.0
    assert jax.grad(entropy)(jnp.asarray(-2.0)) > 0.0


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
    assert (
        adapt_learning_rate(
            current,
            jnp.asarray(0.01),
            config,
            target_kl_exceeded=True,
        )
        < current
    )
    assert adapt_learning_rate(jnp.asarray(1.0), jnp.asarray(0.01), config) == (
        pytest.approx(config.max_learning_rate)
    )
    assert adapt_learning_rate(jnp.asarray(1.0e-9), jnp.asarray(0.01), config) == (
        pytest.approx(config.min_learning_rate)
    )


def test_ppo_metric_aggregation_preserves_maximum_kl_and_any_guard() -> None:
    metrics = aggregate_ppo_metrics(
        {
            "approx_kl": jnp.asarray([[0.001, 0.002], [0.003, 0.08]]),
            "target_kl_exceeded": jnp.asarray([[0.0, 0.0], [0.0, 1.0]]),
            "loss": jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
        }
    )

    assert metrics["approx_kl"] == pytest.approx(0.0215)
    assert metrics["max_approx_kl"] == pytest.approx(0.08)
    assert metrics["target_kl_exceeded"] == 1.0
    assert metrics["loss"] == pytest.approx(2.5)

    assert adaptive_learning_rate_kl(metrics, small_config()) == pytest.approx(0.08)
    mean_config = small_config(adaptive_kl_statistic="mean")
    assert adaptive_learning_rate_kl(metrics, mean_config) == pytest.approx(0.0215)


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


def test_nonfinite_training_input_is_identified_and_skipped() -> None:
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
    assert metrics["invalid/target"] > 0
    assert metrics["invalid_update"] > 0
    assert metrics["ppo_update_fraction"] == 0


def test_nonfinite_current_policy_log_probability_is_identified_and_skipped() -> None:
    config = small_config(target_kl=None)
    train_state, optimizer = init_train_state(jax.random.PRNGKey(0), 3, 2, config)
    batch = transitions(config, train_state.policy)
    poisoned_policy = eqx.tree_at(
        lambda policy: policy.actor.layers[-1].bias,
        train_state.policy,
        jnp.full_like(train_state.policy.actor.layers[-1].bias, 1.0e20),
    )
    poisoned_state = train_state.replace(policy=poisoned_policy)

    next_state, metrics = update_ppo(
        poisoned_state,
        optimizer,
        batch,
        jnp.ones((config.num_steps, config.num_envs)),
        jnp.ones((config.num_steps, config.num_envs)),
        jax.random.PRNGKey(1),
        config,
    )

    assert metrics["nonfinite/current_log_probability"] == 1
    assert metrics["nonfinite/log_ratio"] == 1
    assert metrics["ppo_update_fraction"] == 0
    assert all(
        jnp.array_equal(before, after)
        for before, after in zip(
            jax.tree.leaves(eqx.filter(poisoned_state.policy, eqx.is_array)),
            jax.tree.leaves(eqx.filter(next_state.policy, eqx.is_array)),
            strict=True,
        )
    )
    assert all(
        jnp.array_equal(before, after)
        for before, after in zip(
            jax.tree.leaves(poisoned_state.opt_state),
            jax.tree.leaves(next_state.opt_state),
            strict=True,
        )
    )


@pytest.mark.parametrize(
    "field, value, metric",
    [
        ("obs", 2.0e4, "invalid/rollout_observation"),
        ("reward", 2.0e3, "invalid/rollout_reward"),
        ("value", 2.0e4, "invalid/rollout_value"),
    ],
)
def test_implausible_finite_rollout_is_skipped(field, value, metric) -> None:
    config = small_config()
    train_state, optimizer = init_train_state(jax.random.PRNGKey(0), 3, 2, config)
    batch = transitions(config, train_state.policy)
    contaminated = getattr(batch, field).at[0, 0].set(value)
    batch = batch._replace(**{field: contaminated})
    next_state, metrics = update_ppo(
        train_state,
        optimizer,
        batch,
        jnp.ones((config.num_steps, config.num_envs)),
        jnp.ones((config.num_steps, config.num_envs)),
        jax.random.PRNGKey(1),
        config,
    )

    assert metrics[metric] == 1
    assert metrics["invalid_update"] == 1
    assert metrics["ppo_update_fraction"] == 0
    assert all(
        jnp.array_equal(before, after)
        for before, after in zip(
            jax.tree.leaves(eqx.filter(train_state.policy, eqx.is_array)),
            jax.tree.leaves(eqx.filter(next_state.policy, eqx.is_array)),
            strict=True,
        )
    )


def test_implausible_target_and_gradient_norm_are_skipped() -> None:
    config = small_config(gradient_norm_max=1.0e-12)
    train_state, optimizer = init_train_state(jax.random.PRNGKey(0), 3, 2, config)
    batch = transitions(config, train_state.policy)
    targets = jnp.ones((config.num_steps, config.num_envs))
    _, gradient_metrics = update_ppo(
        train_state,
        optimizer,
        batch,
        jnp.arange(config.batch_size, dtype=jnp.float32).reshape(targets.shape),
        targets,
        jax.random.PRNGKey(1),
        config,
    )
    _, target_metrics = update_ppo(
        train_state,
        optimizer,
        batch,
        jnp.ones_like(targets),
        targets.at[0, 0].set(config.target_max_abs * 2.0),
        jax.random.PRNGKey(1),
        small_config(),
    )

    assert gradient_metrics["invalid/gradient_norm"] == 1
    assert gradient_metrics["ppo_update_fraction"] == 0
    assert target_metrics["invalid/target"] == 1
    assert target_metrics["ppo_update_fraction"] == 0


def test_action_diagnostics_use_each_worlds_joint_angle_offset() -> None:
    config = small_config(num_steps=1, total_timesteps=4)
    train_state, _ = init_train_state(jax.random.PRNGKey(0), 3, 1, config)
    batch = transitions(config, train_state.policy)
    joint_angle_offset = jnp.asarray([[0.0], [0.25], [0.5], [1.0]])

    class OffsetEnv:
        action_size = 1
        control_timestep = 0.02
        model = SimpleNamespace(
            joint_angle_offset=joint_angle_offset,
            policy_ctrl_indices=jnp.asarray([0]),
            mj_model=SimpleNamespace(actuator_ctrlrange=jnp.asarray([[-1.0, 1.0]])),
        )

        def _joint_angle_offset_batch(self, batch_size):
            assert batch_size == 4
            return self.model.joint_angle_offset

        def action_to_ctrl(self, action, offset):
            return action + offset

        def ppo_action_diagnostics(self, env_action):
            flat_action = env_action.reshape((-1, self.action_size))
            num_steps, num_envs = env_action.shape[:2]
            offset = self._joint_angle_offset_batch(num_envs)
            flat_offset = jnp.broadcast_to(
                offset[None, ...], (num_steps, *offset.shape)
            ).reshape((-1, offset.shape[-1]))
            ctrl = jax.vmap(self.action_to_ctrl)(flat_action, flat_offset)
            ctrl_low, ctrl_high = self.model.mj_model.actuator_ctrlrange[0]
            margin = jnp.minimum(ctrl - ctrl_low, ctrl_high - ctrl)
            saturated = (ctrl <= ctrl_low + 1.0e-6) | (ctrl >= ctrl_high - 1.0e-6)
            return {
                "ctrl_saturation_fraction": saturated.astype(jnp.float32).mean(),
                "ctrl_margin_min": jnp.min(margin),
            }

    metrics = action_diagnostics(batch, OffsetEnv(), train_state.policy)

    assert metrics["ctrl_saturation_fraction"] == pytest.approx(0.25)
    assert metrics["ctrl_margin_min"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "config, message",
    [
        (small_config(num_minibatches=3), "divisible"),
        (small_config(schedule="unknown"), "schedule"),
        (small_config(adaptive_kl_statistic="median"), "adaptive_kl_statistic"),
        (
            small_config(learning_rate=1.0e-6),
            "adaptive learning_rate",
        ),
        (
            small_config(learning_rate=1.0e-2),
            "adaptive learning_rate",
        ),
        (
            small_config(policy=ActorCriticConfig(min_log_std=1.0, max_log_std=-1.0)),
            "min_log_std",
        ),
        (
            small_config(policy=ActorCriticConfig(init_log_std=-6.0, min_log_std=-5.0)),
            "init_log_std",
        ),
        (
            small_config(policy=ActorCriticConfig(init_log_std=2.0, max_log_std=1.0)),
            "init_log_std",
        ),
        (
            small_config(policy=ActorCriticConfig(action_distribution="categorical")),
            "action_distribution",
        ),
        (small_config(value_norm_eps=0.0), "value_norm_eps"),
        (small_config(observation_norm_clip=jnp.inf), "observation_norm_clip"),
    ],
)
def test_ppo_validation_rejects_invalid_contracts(
    config: PPOConfig, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_ppo_config(config)
