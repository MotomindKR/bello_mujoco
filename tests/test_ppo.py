from types import SimpleNamespace

import pytest
import jax
import jax.numpy as jnp

from bello_mujoco.policy import (
    ActorCriticConfig,
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
    compute_gae,
    done_reason_means,
    init_train_state,
    make_train,
    ppo_loss,
    safe_returned_mean,
    safe_returned_max,
    update_ppo,
)


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
        action=action,
        action_mean=action,
        value=jnp.zeros(shape, dtype=jnp.float32),
        reward=jnp.ones(shape, dtype=jnp.float32),
        log_prob=jnp.zeros(shape, dtype=jnp.float32),
        obs=obs,
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


def test_sample_action_applies_configured_clip() -> None:
    config = small_config()
    state, _ = init_train_state(jax.random.PRNGKey(0), 4, 2, config)
    obs = jnp.zeros((config.num_envs, 4), dtype=jnp.float32)

    action, _, _ = sample_action(
        state.policy, obs, jax.random.PRNGKey(1), action_clip=0.01
    )

    assert jnp.all(action <= 0.01)
    assert jnp.all(action >= -0.01)


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
            action=transitions.action.reshape((-1, action_size)),
            action_mean=transitions.action_mean.reshape((-1, action_size)),
            value=transitions.value.reshape((-1,)),
            reward=transitions.reward.reshape((-1,)),
            log_prob=transitions.log_prob.reshape((-1,)),
            obs=transitions.obs.reshape((-1, obs_size)),
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
        action=action,
        action_mean=action * 0.5,
        value=jnp.zeros((2, 2), dtype=jnp.float32),
        reward=jnp.ones((2, 2), dtype=jnp.float32),
        log_prob=jnp.zeros((2, 2), dtype=jnp.float32),
        obs=jnp.zeros((2, 2, 4), dtype=jnp.float32),
        info={},
    )
    env = SimpleNamespace(
        action_size=2,
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
    assert float(metrics["action_mean_abs_max"]) == pytest.approx(1.0)
    assert float(metrics["action_std_min"]) == pytest.approx(float(jnp.exp(-2.0)))
    assert float(metrics["action_std_max"]) == pytest.approx(float(jnp.exp(-1.0)))
    assert float(metrics["ctrl_saturation_fraction"]) == pytest.approx(0.375)
    assert float(metrics["ctrl_margin_min"]) == pytest.approx(0.0)


def test_action_diagnostics_omits_control_stats_for_generic_envs() -> None:
    action = jnp.asarray([[[0.0], [1.0]]], dtype=jnp.float32)
    transitions = Transition(
        done=jnp.zeros((1, 2), dtype=bool),
        action=action,
        action_mean=action,
        value=jnp.zeros((1, 2), dtype=jnp.float32),
        reward=jnp.ones((1, 2), dtype=jnp.float32),
        log_prob=jnp.zeros((1, 2), dtype=jnp.float32),
        obs=jnp.zeros((1, 2, 4), dtype=jnp.float32),
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

    assert float(safe_returned_mean(info, key="episode_length")) == pytest.approx(
        125.0
    )
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


def test_ppo_config_rejects_zero_update_runs() -> None:
    config = PPOConfig(total_timesteps=7, num_envs=4, num_steps=2)

    with pytest.raises(ValueError, match="at least one rollout batch"):
        make_train(config, env=object())


def test_ppo_config_rejects_invalid_minibatch_count() -> None:
    config = PPOConfig(total_timesteps=8, num_envs=4, num_steps=2, num_minibatches=0)

    with pytest.raises(ValueError, match="num_minibatches must be positive"):
        make_train(config, env=object())


def test_ppo_config_rejects_invalid_policy_architecture() -> None:
    config = PPOConfig(policy=ActorCriticConfig(hidden_size=0))

    with pytest.raises(ValueError, match="policy.hidden_size must be positive"):
        make_train(config, env=object())


def test_ppo_config_rejects_invalid_observation_stack_size() -> None:
    config = PPOConfig(policy=ActorCriticConfig(obs_stack_size=0))

    with pytest.raises(ValueError, match="policy.obs_stack_size must be positive"):
        make_train(config, env=object())


def test_policy_observation_size_scales_with_stack_size() -> None:
    assert policy_observation_size(12, ActorCriticConfig(obs_stack_size=4)) == 48
