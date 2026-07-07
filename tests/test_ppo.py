import jax
import jax.numpy as jnp

from bello_mujoco.train.ppo import (
    PPOConfig,
    Transition,
    compute_gae,
    init_train_state,
    ppo_loss,
    sample_action,
    update_ppo,
)


def small_config() -> PPOConfig:
    return PPOConfig(
        total_timesteps=32,
        num_envs=4,
        num_steps=8,
        update_epochs=2,
        num_minibatches=2,
        hidden_size=16,
        hidden_depth=1,
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
    obs_size = 114
    action_size = 35
    state, _ = init_train_state(jax.random.PRNGKey(0), obs_size, action_size, config)
    obs = jnp.zeros((config.num_envs, obs_size), dtype=jnp.float32)

    action, log_prob, value = sample_action(state.model, obs, jax.random.PRNGKey(1))

    assert action.shape == (config.num_envs, action_size)
    assert log_prob.shape == (config.num_envs,)
    assert value.shape == (config.num_envs,)


def test_gae_and_loss_shapes() -> None:
    config = small_config()
    obs_size = 114
    action_size = 35
    state, _ = init_train_state(jax.random.PRNGKey(0), obs_size, action_size, config)
    transitions = dummy_transitions(config, obs_size, action_size)
    advantages, targets = compute_gae(
        transitions, jnp.zeros((config.num_envs,)), config
    )

    loss, metrics = ppo_loss(
        state.model,
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
    obs_size = 114
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

    assert int(next_state.update) == config.update_epochs * config.num_minibatches
    assert metrics["loss"].shape == ()
