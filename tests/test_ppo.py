import pytest
import jax
import jax.numpy as jnp

from bello_mujoco.policy import ActorCriticConfig, gaussian_entropy, sample_action
from bello_mujoco.train.ppo import (
    PPOConfig,
    Transition,
    compute_gae,
    init_train_state,
    make_train,
    ppo_loss,
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
