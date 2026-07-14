from types import SimpleNamespace
from typing import NamedTuple

import equinox as eqx
import pytest
import jax
import jax.numpy as jnp

from bello_mujoco.envs.rewards import (
    CommandConfig,
    CommandCurriculumConfig,
    RewardConfig,
)
from bello_mujoco.policy import (
    ActorCriticConfig,
    apply_actor,
    evaluate_actions,
    gaussian_entropy,
    init_observation_stack,
    policy_observation_size,
    sample_action,
    stacked_observation,
    update_observation_stack,
)
from bello_mujoco.train.ppo import (
    PPOConfig,
    Transition,
    action_diagnostics,
    adapt_learning_rate,
    collect_rollout,
    compute_gae,
    denormalize_value,
    done_reason_means,
    effective_learning_rate_schedule,
    gait_diagnostics,
    init_value_norm_state,
    init_train_state,
    learning_rate_for_update,
    make_train,
    normalize_advantages,
    normalize_value,
    ppo_loss,
    safe_returned_mean,
    safe_returned_max,
    update_command_curriculum,
    update_command_curriculum_ranges,
    update_value_norm_state,
    update_ppo,
)


class DummyRolloutState(NamedTuple):
    obs: jax.Array


class ConstantLargeActionPolicy:
    log_std = jnp.asarray([0.0], dtype=jnp.float32)

    def __call__(self, obs, critic_obs=None):
        del obs, critic_obs
        return (
            jnp.asarray([100.0], dtype=jnp.float32),
            self.log_std,
            jnp.asarray(0.0, dtype=jnp.float32),
        )


class ClippedActionEnv:
    observation_size = 1
    action_size = 1
    action_clip = 1.0

    def step_batch(self, state, action, *, rng):
        del rng
        next_obs = state.obs
        clipped_action = (
            action
            if self.action_clip is None
            else jnp.clip(action, -self.action_clip, self.action_clip)
        )
        reward = jnp.squeeze(clipped_action, axis=-1)
        done = jnp.zeros(action.shape[0], dtype=bool)
        info = {
            "returned_episode": done,
            "bootstrap_observation": next_obs,
            "requested_action_abs_max": jnp.max(jnp.abs(action), axis=-1),
        }
        return DummyRolloutState(next_obs), next_obs, reward, done, info


class UnclippedActionEnv(ClippedActionEnv):
    action_clip = None


def small_config() -> PPOConfig:
    return PPOConfig(
        total_timesteps=32,
        num_envs=4,
        num_steps=8,
        update_epochs=2,
        num_minibatches=2,
        policy=ActorCriticConfig(hidden_size=16, hidden_depth=1),
        anneal_lr=False,
    )


def dummy_transitions(config: PPOConfig, obs_size: int, action_size: int) -> Transition:
    shape = (config.num_steps, config.num_envs)
    obs = jnp.zeros(shape + (obs_size,), dtype=jnp.float32)
    action = jnp.zeros(shape + (action_size,), dtype=jnp.float32)
    return Transition(
        done=jnp.zeros(shape, dtype=bool),
        terminal=jnp.zeros(shape, dtype=bool),
        bootstrap_value=jnp.zeros(shape, dtype=jnp.float32),
        action=action,
        env_action=action,
        action_mean=action,
        value=jnp.zeros(shape, dtype=jnp.float32),
        reward=jnp.ones(shape, dtype=jnp.float32),
        log_prob=jnp.zeros(shape, dtype=jnp.float32),
        obs=obs,
        critic_obs=obs,
        info={},
    )


def test_actor_critic_sample_shapes() -> None:
    config = small_config()
    obs_size = 149
    action_size = 35
    state, _ = init_train_state(jax.random.PRNGKey(0), obs_size, action_size, config)
    obs = jnp.zeros((config.num_envs, obs_size), dtype=jnp.float32)

    action, log_prob, value = sample_action(state.policy, obs, jax.random.PRNGKey(1))

    assert action.shape == (config.num_envs, action_size)
    assert log_prob.shape == (config.num_envs,)
    assert value.shape == (config.num_envs,)


def test_actor_critic_supports_nonuniform_hidden_dims_and_elu() -> None:
    config = PPOConfig(
        total_timesteps=8,
        num_envs=2,
        num_steps=4,
        policy=ActorCriticConfig(
            hidden_dims=(8, 4),
            activation="elu",
            obs_stack_size=1,
        ),
    )
    obs_size = 6
    action_size = 3
    state, _ = init_train_state(jax.random.PRNGKey(0), obs_size, action_size, config)
    obs = jnp.zeros((config.num_envs, obs_size), dtype=jnp.float32)

    action, log_prob, value = sample_action(state.policy, obs, jax.random.PRNGKey(1))

    assert action.shape == (config.num_envs, action_size)
    assert log_prob.shape == (config.num_envs,)
    assert value.shape == (config.num_envs,)


def test_actor_critic_supports_privileged_critic_observation_size() -> None:
    config = small_config()
    action_size = 3
    state, _ = init_train_state(
        jax.random.PRNGKey(0),
        4,
        action_size,
        config,
        critic_obs_size=7,
    )
    actor_obs = jnp.zeros((config.num_envs, 4), dtype=jnp.float32)
    critic_obs = jnp.zeros((config.num_envs, 7), dtype=jnp.float32)

    action, log_prob, value = sample_action(
        state.policy, actor_obs, jax.random.PRNGKey(1), critic_obs=critic_obs
    )

    assert action.shape == (config.num_envs, action_size)
    assert log_prob.shape == (config.num_envs,)
    assert value.shape == (config.num_envs,)


def test_sample_action_applies_configured_clip() -> None:
    config = small_config()
    state, _ = init_train_state(jax.random.PRNGKey(0), 4, 2, config)
    obs = jnp.zeros((config.num_envs, 4), dtype=jnp.float32)

    action, _, _ = sample_action(
        state.policy, obs, jax.random.PRNGKey(1), action_clip=0.01
    )

    assert jnp.all(action <= 0.01)
    assert jnp.all(action >= -0.01)


def test_tanh_gaussian_actions_are_bounded_and_have_consistent_log_prob() -> None:
    config = PPOConfig(
        total_timesteps=8,
        num_envs=2,
        num_steps=4,
        policy=ActorCriticConfig(
            hidden_size=8,
            hidden_depth=1,
            obs_stack_size=1,
            init_log_std=1.0,
            action_distribution="tanh_gaussian",
            action_limit=0.75,
        ),
    )
    state, _ = init_train_state(jax.random.PRNGKey(0), 4, 2, config)
    obs = jnp.zeros((config.num_envs, 4), dtype=jnp.float32)

    action, sampled_log_prob, _ = sample_action(
        state.policy, obs, jax.random.PRNGKey(1)
    )
    evaluated_log_prob, _, _ = evaluate_actions(state.policy, obs, action)
    deterministic_action = apply_actor(state.policy, obs)

    assert jnp.all(jnp.abs(action) < 0.75)
    assert jnp.all(jnp.abs(deterministic_action) < 0.75)
    assert jnp.allclose(sampled_log_prob, evaluated_log_prob, atol=1.0e-5)


def test_collect_rollout_stores_raw_actions_and_steps_clipped_env_actions() -> None:
    config = PPOConfig(
        total_timesteps=4,
        num_envs=2,
        num_steps=2,
        policy=ActorCriticConfig(obs_stack_size=1),
    )
    env = ClippedActionEnv()
    obs = jnp.zeros((config.num_envs, env.observation_size), dtype=jnp.float32)
    obs_stack = init_observation_stack(obs, config.policy.obs_stack_size)

    _, transitions = collect_rollout(
        env,
        ConstantLargeActionPolicy(),
        DummyRolloutState(obs),
        obs_stack,
        obs_stack,
        jnp.zeros((3, 2), dtype=jnp.float32),
        jax.random.PRNGKey(0),
        config,
        init_value_norm_state(),
    )

    assert bool(jnp.any(jnp.abs(transitions.action) > env.action_clip))
    assert jnp.all(transitions.env_action <= env.action_clip)
    assert jnp.all(transitions.env_action >= -env.action_clip)
    assert float(jnp.max(jnp.abs(transitions.action_mean))) == pytest.approx(100.0)
    assert float(jnp.max(jnp.abs(transitions.reward))) == pytest.approx(env.action_clip)
    assert float(jnp.max(transitions.info["requested_action_abs_max"])) > 1.0


def test_collect_rollout_does_not_clip_env_actions_without_env_action_clip() -> None:
    config = PPOConfig(
        total_timesteps=4,
        num_envs=2,
        num_steps=2,
        policy=ActorCriticConfig(obs_stack_size=1),
    )
    env = UnclippedActionEnv()
    obs = jnp.zeros((config.num_envs, env.observation_size), dtype=jnp.float32)
    obs_stack = init_observation_stack(obs, config.policy.obs_stack_size)

    _, transitions = collect_rollout(
        env,
        ConstantLargeActionPolicy(),
        DummyRolloutState(obs),
        obs_stack,
        obs_stack,
        jnp.zeros((3, 2), dtype=jnp.float32),
        jax.random.PRNGKey(0),
        config,
        init_value_norm_state(),
    )

    assert jnp.allclose(transitions.env_action, transitions.action)
    assert float(jnp.max(jnp.abs(transitions.reward))) > 1.0


def test_observation_stack_repeats_shifts_and_resets_done_envs() -> None:
    obs = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    obs_stack = init_observation_stack(obs, 3)

    assert obs_stack.shape == (2, 3, 2)
    assert stacked_observation(obs_stack).shape == (2, 6)

    next_obs = jnp.asarray([[5.0, 6.0], [7.0, 8.0]])
    next_stack = update_observation_stack(
        obs_stack,
        next_obs,
        jnp.asarray([False, True]),
    )

    assert jnp.allclose(
        next_stack[0], jnp.asarray([[1.0, 2.0], [1.0, 2.0], [5.0, 6.0]])
    )
    assert jnp.allclose(
        next_stack[1], jnp.asarray([[7.0, 8.0], [7.0, 8.0], [7.0, 8.0]])
    )


def test_command_curriculum_expands_linear_ranges_after_success_interval() -> None:
    env = SimpleNamespace(
        command_config=CommandConfig(
            curriculum=CommandCurriculumConfig(enabled=True),
        ),
        reward_config=RewardConfig(),
        control_timestep=0.02,
        max_episode_steps=1000,
    )
    config = PPOConfig(total_timesteps=10000, num_envs=4, num_steps=24)
    ranges = env.command_config.range_array()
    info = {
        "returned_episode": jnp.asarray([True]),
        "episode_reward/track_lin_vel_xy": jnp.asarray(
            [
                0.81
                * env.reward_config.track_lin_vel_xy.weight
                * env.max_episode_steps
                * env.control_timestep
            ]
        ),
    }

    next_ranges = update_command_curriculum_ranges(
        ranges, info, jnp.asarray(41, dtype=jnp.int32), env, config
    )
    _, metrics = update_command_curriculum(
        ranges, info, jnp.asarray(41, dtype=jnp.int32), env, config
    )

    assert next_ranges[0, 0] == pytest.approx(-0.2)
    assert next_ranges[0, 1] == pytest.approx(0.2)
    assert next_ranges[1, 0] == pytest.approx(-0.2)
    assert next_ranges[1, 1] == pytest.approx(0.2)
    assert jnp.all(next_ranges[2] == ranges[2])
    assert metrics["command_curriculum/enabled"] == pytest.approx(1.0)
    assert metrics["command_curriculum/returned_count"] == pytest.approx(1.0)
    assert metrics["command_curriculum/normalized_track_lin_vel_xy"] == pytest.approx(
        0.81 * env.reward_config.track_lin_vel_xy.weight
    )
    assert metrics["command_curriculum/reward_threshold"] == pytest.approx(
        env.reward_config.track_lin_vel_xy.weight
        * env.command_config.curriculum.success_threshold
    )
    assert metrics["command_curriculum/period_boundary"] == pytest.approx(1.0)
    assert metrics["command_curriculum/success"] == pytest.approx(1.0)
    assert metrics["command_curriculum/should_expand"] == pytest.approx(1.0)
    assert metrics["command_curriculum/should_expand_yaw"] == pytest.approx(0.0)


def test_command_curriculum_does_not_expand_without_completed_episode() -> None:
    env = SimpleNamespace(
        command_config=CommandConfig(
            curriculum=CommandCurriculumConfig(enabled=True),
        ),
        reward_config=RewardConfig(),
        control_timestep=0.02,
        max_episode_steps=1000,
    )
    config = PPOConfig(total_timesteps=10000, num_envs=4, num_steps=24)
    ranges = env.command_config.range_array()
    info = {
        "returned_episode": jnp.asarray([False]),
        "episode_reward/track_lin_vel_xy": jnp.asarray(
            [
                0.99
                * env.reward_config.track_lin_vel_xy.weight
                * env.max_episode_steps
                * env.control_timestep
            ]
        ),
    }

    next_ranges = update_command_curriculum_ranges(
        ranges, info, jnp.asarray(41, dtype=jnp.int32), env, config
    )
    _, metrics = update_command_curriculum(
        ranges, info, jnp.asarray(41, dtype=jnp.int32), env, config
    )

    assert jnp.all(next_ranges == ranges)
    assert metrics["command_curriculum/enabled"] == pytest.approx(1.0)
    assert metrics["command_curriculum/returned_count"] == pytest.approx(0.0)
    assert metrics["command_curriculum/success"] == pytest.approx(0.0)
    assert metrics["command_curriculum/should_expand"] == pytest.approx(0.0)


def test_gaussian_entropy_sums_only_action_axis() -> None:
    entropy = gaussian_entropy(jnp.zeros((3, 2), dtype=jnp.float32))
    expected = jnp.full((3,), 2 * 0.5 * (1.0 + jnp.log(2.0 * jnp.pi)))

    assert entropy.shape == (3,)
    assert jnp.allclose(entropy, expected)


def test_gae_and_loss_shapes() -> None:
    config = small_config()
    obs_size = 149
    action_size = 35
    state, _ = init_train_state(jax.random.PRNGKey(0), obs_size, action_size, config)
    transitions = dummy_transitions(config, obs_size, action_size)
    advantages, targets = compute_gae(
        transitions, jnp.zeros((config.num_envs,)), config
    )

    loss, metrics = ppo_loss(
        state.policy,
        Transition(
            done=transitions.done.reshape((-1,)),
            terminal=transitions.terminal.reshape((-1,)),
            bootstrap_value=transitions.bootstrap_value.reshape((-1,)),
            action=transitions.action.reshape((-1, action_size)),
            env_action=transitions.env_action.reshape((-1, action_size)),
            action_mean=transitions.action_mean.reshape((-1, action_size)),
            value=transitions.value.reshape((-1,)),
            reward=transitions.reward.reshape((-1,)),
            log_prob=transitions.log_prob.reshape((-1,)),
            obs=transitions.obs.reshape((-1, obs_size)),
            critic_obs=transitions.critic_obs.reshape((-1, obs_size)),
            info={},
        ),
        advantages.reshape((-1,)),
        targets.reshape((-1,)),
        config,
    )

    assert loss.shape == ()
    assert metrics["actor_loss"].shape == ()
    assert advantages.shape == (config.num_steps, config.num_envs)
    assert targets.shape == (config.num_steps, config.num_envs)


def test_advantage_normalization_is_rollout_global() -> None:
    advantages = jnp.asarray([[1.0, 2.0], [3.0, 9.0]], dtype=jnp.float32)
    normalized = normalize_advantages(advantages)

    assert normalized.shape == advantages.shape
    assert float(normalized.mean()) == pytest.approx(0.0, abs=1.0e-6)
    assert float(normalized.std()) == pytest.approx(1.0)
    assert not jnp.allclose(normalized[0], jnp.zeros(2, dtype=jnp.float32))


def test_gae_bootstraps_timeout_without_continuing_across_reset() -> None:
    config = PPOConfig(
        total_timesteps=2,
        num_envs=1,
        num_steps=2,
        gamma=1.0,
        gae_lambda=1.0,
    )
    transitions = Transition(
        done=jnp.asarray([[False], [True]]),
        terminal=jnp.asarray([[False], [False]]),
        bootstrap_value=jnp.asarray([[0.0], [10.0]], dtype=jnp.float32),
        action=jnp.zeros((2, 1, 1), dtype=jnp.float32),
        env_action=jnp.zeros((2, 1, 1), dtype=jnp.float32),
        action_mean=jnp.zeros((2, 1, 1), dtype=jnp.float32),
        value=jnp.zeros((2, 1), dtype=jnp.float32),
        reward=jnp.zeros((2, 1), dtype=jnp.float32),
        log_prob=jnp.zeros((2, 1), dtype=jnp.float32),
        obs=jnp.zeros((2, 1, 1), dtype=jnp.float32),
        critic_obs=jnp.zeros((2, 1, 1), dtype=jnp.float32),
        info={},
    )

    advantages, targets = compute_gae(
        transitions, jnp.zeros((config.num_envs,), dtype=jnp.float32), config
    )

    assert jnp.allclose(advantages, jnp.asarray([[10.0], [10.0]]))
    assert jnp.allclose(targets, advantages)


def test_gae_does_not_bootstrap_terminal_failure() -> None:
    config = PPOConfig(
        total_timesteps=1,
        num_envs=1,
        num_steps=1,
        gamma=1.0,
        gae_lambda=1.0,
    )
    transitions = Transition(
        done=jnp.asarray([[True]]),
        terminal=jnp.asarray([[True]]),
        bootstrap_value=jnp.asarray([[10.0]], dtype=jnp.float32),
        action=jnp.zeros((1, 1, 1), dtype=jnp.float32),
        env_action=jnp.zeros((1, 1, 1), dtype=jnp.float32),
        action_mean=jnp.zeros((1, 1, 1), dtype=jnp.float32),
        value=jnp.zeros((1, 1), dtype=jnp.float32),
        reward=jnp.zeros((1, 1), dtype=jnp.float32),
        log_prob=jnp.zeros((1, 1), dtype=jnp.float32),
        obs=jnp.zeros((1, 1, 1), dtype=jnp.float32),
        critic_obs=jnp.zeros((1, 1, 1), dtype=jnp.float32),
        info={},
    )

    advantages, targets = compute_gae(
        transitions, jnp.zeros((config.num_envs,), dtype=jnp.float32), config
    )

    assert jnp.allclose(advantages, 0.0)
    assert jnp.allclose(targets, 0.0)


def test_update_ppo_runs_on_dummy_batch() -> None:
    config = small_config()
    obs_size = 149
    action_size = 35
    train_state, optimizer = init_train_state(
        jax.random.PRNGKey(0), obs_size, action_size, config
    )
    transitions = dummy_transitions(config, obs_size, action_size)
    advantages, targets = compute_gae(
        transitions, jnp.zeros((config.num_envs,)), config
    )

    next_state, metrics = update_ppo(
        train_state,
        optimizer,
        transitions,
        advantages,
        targets,
        jax.random.PRNGKey(1),
        config,
    )

    assert int(next_state.update) == 1
    assert metrics["loss"].shape == ()


def test_update_ppo_skips_nonfinite_minibatch_without_poisoning_policy() -> None:
    config = small_config()
    obs_size = 149
    action_size = 35
    train_state, optimizer = init_train_state(
        jax.random.PRNGKey(0), obs_size, action_size, config
    )
    transitions = dummy_transitions(config, obs_size, action_size)
    advantages = jnp.full(
        (config.num_steps, config.num_envs), jnp.nan, dtype=jnp.float32
    )
    targets = jnp.zeros((config.num_steps, config.num_envs), dtype=jnp.float32)
    initial_params, _ = eqx.partition(train_state.policy, eqx.is_array)

    next_state, metrics = update_ppo(
        train_state,
        optimizer,
        transitions,
        advantages,
        targets,
        jax.random.PRNGKey(1),
        config,
    )
    next_params, _ = eqx.partition(next_state.policy, eqx.is_array)

    assert int(next_state.update) == 1
    assert float(metrics["nonfinite_update"]) > 0.0
    assert float(metrics["ppo_update_fraction"]) == 0.0
    for before, after in zip(
        jax.tree_util.tree_leaves(initial_params),
        jax.tree_util.tree_leaves(next_params),
        strict=True,
    ):
        assert jnp.allclose(before, after)


def test_value_normalization_round_trips_and_updates_stats() -> None:
    config = PPOConfig(normalize_value=True)
    state = update_value_norm_state(
        init_value_norm_state(),
        jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32),
        config,
    )

    assert float(state.mean) == pytest.approx(2.5)
    assert float(state.var) == pytest.approx(1.25)
    assert float(state.count) == pytest.approx(4.0)

    values = jnp.asarray([1.0, 2.5, 4.0], dtype=jnp.float32)
    normalized = normalize_value(values, state, config)
    denormalized = denormalize_value(normalized, state, config)

    assert jnp.allclose(denormalized, values)


def test_update_ppo_supports_normalized_value_loss() -> None:
    config = PPOConfig(
        total_timesteps=32,
        num_envs=4,
        num_steps=8,
        update_epochs=1,
        num_minibatches=2,
        policy=ActorCriticConfig(hidden_size=16, hidden_depth=1),
        anneal_lr=False,
        normalize_value=True,
    )
    obs_size = 149
    action_size = 35
    train_state, optimizer = init_train_state(
        jax.random.PRNGKey(0), obs_size, action_size, config
    )
    transitions = dummy_transitions(config, obs_size, action_size)
    advantages, targets = compute_gae(
        transitions, jnp.zeros((config.num_envs,)), config
    )
    value_norm_state = update_value_norm_state(init_value_norm_state(), targets, config)

    next_state, metrics = update_ppo(
        train_state,
        optimizer,
        transitions,
        advantages,
        targets,
        jax.random.PRNGKey(1),
        config,
        value_norm_state=value_norm_state,
    )

    assert int(next_state.update) == 1
    assert metrics["value_loss"].shape == ()
    assert metrics["value_loss_raw"].shape == ()
    assert jnp.isfinite(metrics["value_loss"])
    assert jnp.isfinite(metrics["value_loss_raw"])


def test_update_ppo_clamps_policy_log_std_after_update() -> None:
    config = PPOConfig(
        total_timesteps=32,
        num_envs=4,
        num_steps=8,
        update_epochs=1,
        num_minibatches=1,
        policy=ActorCriticConfig(
            hidden_size=16,
            hidden_depth=1,
            init_log_std=0.0,
            min_log_std=-2.0,
            max_log_std=0.0,
        ),
        anneal_lr=False,
    )
    obs_size = 8
    action_size = 3
    train_state, optimizer = init_train_state(
        jax.random.PRNGKey(0), obs_size, action_size, config
    )
    policy = eqx.tree_at(
        lambda model: model.log_std,
        train_state.policy,
        jnp.full((action_size,), 5.0, dtype=jnp.float32),
    )
    train_state = train_state.__class__(
        policy=policy,
        opt_state=train_state.opt_state,
        update=train_state.update,
    )
    transitions = dummy_transitions(config, obs_size, action_size)
    advantages, targets = compute_gae(
        transitions, jnp.zeros((config.num_envs,)), config
    )

    next_state, _ = update_ppo(
        train_state,
        optimizer,
        transitions,
        advantages,
        targets,
        jax.random.PRNGKey(1),
        config,
    )

    assert jnp.all(next_state.policy.log_std <= 0.0)
    assert jnp.all(next_state.policy.log_std >= -2.0)


def test_learning_rate_schedule_preserves_anneal_lr_compatibility() -> None:
    linear = PPOConfig(
        total_timesteps=40,
        num_envs=2,
        num_steps=2,
        anneal_lr=True,
        schedule=None,
    )
    constant = PPOConfig(
        total_timesteps=40,
        num_envs=2,
        num_steps=2,
        anneal_lr=False,
        schedule=None,
    )

    assert effective_learning_rate_schedule(linear) == "linear"
    assert effective_learning_rate_schedule(constant) == "constant"
    assert float(
        learning_rate_for_update(
            linear,
            jnp.asarray(5, dtype=jnp.int32),
            jnp.asarray(123.0, dtype=jnp.float32),
        )
    ) == pytest.approx(linear.learning_rate * 0.5)
    assert float(
        learning_rate_for_update(
            constant,
            jnp.asarray(5, dtype=jnp.int32),
            jnp.asarray(123.0, dtype=jnp.float32),
        )
    ) == pytest.approx(constant.learning_rate)


def test_adaptive_learning_rate_reacts_to_kl_thresholds() -> None:
    config = PPOConfig(
        total_timesteps=40,
        num_envs=2,
        num_steps=2,
        schedule="adaptive",
        learning_rate=1.0e-3,
        desired_kl=0.01,
        min_learning_rate=1.0e-5,
        max_learning_rate=1.0e-2,
        adaptive_lr_factor=1.5,
    )
    learning_rate = jnp.asarray(1.0e-3, dtype=jnp.float32)

    assert float(adapt_learning_rate(learning_rate, jnp.asarray(0.03), config)) == (
        pytest.approx(1.0e-3 / 1.5)
    )
    assert float(adapt_learning_rate(learning_rate, jnp.asarray(0.001), config)) == (
        pytest.approx(1.0e-3 * 1.5)
    )
    assert float(adapt_learning_rate(learning_rate, jnp.asarray(0.01), config)) == (
        pytest.approx(1.0e-3)
    )


def test_action_diagnostics_reports_action_and_control_stats() -> None:
    action = jnp.asarray(
        [
            [[0.0, 0.0], [1.0, -1.0]],
            [[2.0, -2.0], [0.5, -0.5]],
        ],
        dtype=jnp.float32,
    )
    transitions = Transition(
        done=jnp.zeros((2, 2), dtype=bool),
        terminal=jnp.zeros((2, 2), dtype=bool),
        bootstrap_value=jnp.zeros((2, 2), dtype=jnp.float32),
        action=action,
        env_action=jnp.clip(action, -1.0, 1.0),
        action_mean=action * 0.5,
        value=jnp.zeros((2, 2), dtype=jnp.float32),
        reward=jnp.ones((2, 2), dtype=jnp.float32),
        log_prob=jnp.zeros((2, 2), dtype=jnp.float32),
        obs=jnp.zeros((2, 2, 4), dtype=jnp.float32),
        critic_obs=jnp.zeros((2, 2, 4), dtype=jnp.float32),
        info={},
    )
    env = SimpleNamespace(
        action_size=2,
        action_clip=1.0,
        model=SimpleNamespace(
            policy_ctrl_indices=jnp.asarray([0, 1]),
            mj_model=SimpleNamespace(
                actuator_ctrlrange=jnp.asarray(
                    [[-1.0, 1.0], [-2.0, 2.0]], dtype=jnp.float32
                )
            ),
        ),
        _action_to_ctrl=lambda act: jnp.clip(
            act,
            jnp.asarray([-1.0, -2.0], dtype=jnp.float32),
            jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        ),
    )
    policy = SimpleNamespace(log_std=jnp.asarray([-2.0, -1.0], dtype=jnp.float32))

    metrics = action_diagnostics(transitions, env, policy)

    assert float(metrics["action_rms"]) == pytest.approx(
        float(jnp.sqrt(jnp.mean(action**2)))
    )
    assert float(metrics["action_abs_max"]) == pytest.approx(2.0)
    assert float(metrics["env_action_abs_max"]) == pytest.approx(1.0)
    assert float(metrics["action_mean_abs_max"]) == pytest.approx(1.0)
    assert float(metrics["action_clip_fraction"]) == pytest.approx(0.25)
    assert float(metrics["action_std_min"]) == pytest.approx(float(jnp.exp(-2.0)))
    assert float(metrics["action_std_max"]) == pytest.approx(float(jnp.exp(-1.0)))
    assert float(metrics["ctrl_saturation_fraction"]) == pytest.approx(0.25)
    assert float(metrics["ctrl_margin_min"]) == pytest.approx(0.0)


def test_action_diagnostics_omits_control_stats_for_generic_envs() -> None:
    action = jnp.asarray([[[0.0], [1.0]]], dtype=jnp.float32)
    transitions = Transition(
        done=jnp.zeros((1, 2), dtype=bool),
        terminal=jnp.zeros((1, 2), dtype=bool),
        bootstrap_value=jnp.zeros((1, 2), dtype=jnp.float32),
        action=action,
        env_action=action,
        action_mean=action,
        value=jnp.zeros((1, 2), dtype=jnp.float32),
        reward=jnp.ones((1, 2), dtype=jnp.float32),
        log_prob=jnp.zeros((1, 2), dtype=jnp.float32),
        obs=jnp.zeros((1, 2, 4), dtype=jnp.float32),
        critic_obs=jnp.zeros((1, 2, 4), dtype=jnp.float32),
        info={},
    )
    env = SimpleNamespace(action_size=1)
    policy = SimpleNamespace(log_std=jnp.asarray([-1.0], dtype=jnp.float32))

    metrics = action_diagnostics(transitions, env, policy)

    assert "action_rms" in metrics
    assert "ctrl_saturation_fraction" not in metrics
    assert "ctrl_margin_min" not in metrics


def test_safe_returned_max_ignores_non_returned_zero_placeholders() -> None:
    info = {
        "episode_return": jnp.asarray([[-5.0, 0.0], [-2.0, 0.0]]),
        "returned_episode": jnp.asarray([[True, False], [True, False]]),
    }

    assert float(safe_returned_max(info)) == -2.0


def test_safe_returned_max_is_nan_without_completed_episode() -> None:
    info = {
        "episode_return": jnp.zeros((2, 2), dtype=jnp.float32),
        "returned_episode": jnp.zeros((2, 2), dtype=bool),
    }

    assert bool(jnp.isnan(safe_returned_max(info)))


def test_safe_returned_mean_and_max_accept_episode_length_key() -> None:
    info = {
        "episode_length": jnp.asarray([[100, 0], [150, 0]], dtype=jnp.int32),
        "returned_episode": jnp.asarray([[True, False], [True, False]]),
    }

    assert float(safe_returned_mean(info, key="episode_length")) == pytest.approx(125.0)
    assert float(safe_returned_max(info, key="episode_length")) == pytest.approx(150.0)


def test_done_reason_means_reports_done_component_rates() -> None:
    info = {
        "done/any": jnp.asarray([[True, False], [False, True]]),
        "done/timeout": jnp.asarray([[True, False], [False, False]]),
        "reward/alive": jnp.ones((2, 2), dtype=jnp.float32),
    }

    metrics = done_reason_means(info)

    assert metrics.keys() == {"mean_done/any", "mean_done/timeout"}
    assert float(metrics["mean_done/any"]) == 0.5
    assert float(metrics["mean_done/timeout"]) == 0.25


def test_gait_diagnostics_converts_touchdowns_to_footfalls_per_second() -> None:
    info = {
        "gait/stance_width": jnp.asarray([[0.28, 0.30]], dtype=jnp.float32),
        "gait/touchdown": jnp.asarray([[0.0, 1.0]], dtype=jnp.float32),
        "gait/liftoff": jnp.asarray([[1.0, 0.0]], dtype=jnp.float32),
    }

    metrics = gait_diagnostics(info, control_timestep=0.02)

    assert float(metrics["mean_gait/stance_width"]) == pytest.approx(0.29)
    assert float(metrics["gait/footfalls_per_second"]) == pytest.approx(25.0)
    assert float(metrics["gait/liftoffs_per_second"]) == pytest.approx(25.0)


def test_ppo_config_rejects_zero_update_runs() -> None:
    config = PPOConfig(total_timesteps=7, num_envs=4, num_steps=2)

    with pytest.raises(ValueError, match="at least one rollout batch"):
        make_train(config, env=object())


def test_ppo_config_rejects_invalid_minibatch_count() -> None:
    config = PPOConfig(total_timesteps=8, num_envs=4, num_steps=2, num_minibatches=0)

    with pytest.raises(ValueError, match="num_minibatches must be positive"):
        make_train(config, env=object())


def test_ppo_config_rejects_invalid_target_kl() -> None:
    config = PPOConfig(target_kl=0.0)

    with pytest.raises(ValueError, match="target_kl must be null or positive"):
        make_train(config, env=object())


def test_ppo_config_rejects_invalid_action_distribution() -> None:
    config = PPOConfig(
        policy=ActorCriticConfig(action_distribution="not-a-distribution")
    )

    with pytest.raises(ValueError, match="policy.action_distribution"):
        make_train(config, env=object())


def test_ppo_config_defaults_to_no_target_kl_stop() -> None:
    assert PPOConfig().target_kl is None


def test_ppo_config_rejects_invalid_policy_architecture() -> None:
    config = PPOConfig(policy=ActorCriticConfig(hidden_size=0))

    with pytest.raises(ValueError, match="policy.hidden_size must be positive"):
        make_train(config, env=object())


def test_ppo_config_rejects_invalid_hidden_dims() -> None:
    config = PPOConfig(policy=ActorCriticConfig(hidden_dims=(16, 0)))

    with pytest.raises(ValueError, match="policy.hidden_dims entries must be positive"):
        make_train(config, env=object())


def test_ppo_config_rejects_invalid_observation_stack_size() -> None:
    config = PPOConfig(policy=ActorCriticConfig(obs_stack_size=0))

    with pytest.raises(ValueError, match="policy.obs_stack_size must be positive"):
        make_train(config, env=object())


def test_policy_observation_size_scales_with_stack_size() -> None:
    assert policy_observation_size(12, ActorCriticConfig(obs_stack_size=4)) == 48
