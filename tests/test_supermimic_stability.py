from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bello_mujoco.supermimic.imagination import ImaginedRollout, actor_loss
from bello_mujoco.supermimic.mdp import imagined_return, lambda_returns
from bello_mujoco.training.supermimic import compile_updates, require_valid_metrics
from test_supermimic_runner import ToyEnv, toy_runner as make_toy_runner


@pytest.fixture
def toy_runner():
    return make_toy_runner.__wrapped__()


def test_finite_joint_speed_spike_is_rejected_before_normalization(
    toy_runner, monkeypatch
):
    env, config, runner = toy_runner
    original = ToyEnv.step_batch

    def spike(self, *args, **kwargs):
        state, *outputs = original(self, *args, **kwargs)
        state = replace(
            state, data=replace(state.data, qvel=state.data.qvel.at[0, 6].set(1e7))
        )
        return state, *outputs

    monkeypatch.setattr(ToyEnv, "step_batch", spike)
    collect, _, _ = compile_updates(env, config)
    result, metrics = collect(runner, env.reference_bank)
    require_valid_metrics(metrics)
    assert metrics["diagnostic/invalid/joint_velocity"] == 1
    assert metrics["diagnostic/invalid/physics"] == 0
    np.testing.assert_array_equal(result.replay.valid[0], [False, True])
    np.testing.assert_array_equal(result.collector.environment.episode_length, [0, 1])
    for name in ("input_norm", "output_norm", "observation_norm", "goal_norm"):
        moments = getattr(result.learner, name)
        assert moments.count == 1
        assert np.max(np.abs(moments.mean)) < 1e3


def test_terminated_return_never_uses_unreachable_rewards_or_bootstrap():
    rewards = jnp.array([[0.1], [jnp.nan], [jnp.inf]])
    terms = jnp.ones_like(rewards)
    result = jax.jit(imagined_return)(rewards, terms, jnp.array([jnp.inf]), 0.99)
    np.testing.assert_allclose(result, 0.1)
    targets = jax.jit(lambda_returns)(
        jnp.array([[0.1], [0.2]]),
        jnp.ones((2, 1)),
        jnp.array([[0.0], [jnp.inf], [jnp.nan]]),
        0.99,
        0.95,
    )
    np.testing.assert_allclose(targets[:, 0], [0.1, 0.2])


@pytest.mark.parametrize("gamma", [0.9, 1.0])
def test_actor_bootstrap_ceiling_blocks_exploitation_but_keeps_negative_values(
    toy_runner, monkeypatch, gamma
):
    from bello_mujoco.supermimic import imagination

    env, _, _ = toy_runner
    env = replace(env, mdp=replace(env.mdp, timestep=0.03))
    bound = env.mdp.maximum_return(gamma, env.max_episode_steps)

    def rollout(actor, *args, **kwargs):
        features = jnp.zeros((3, 2, 1)).at[-1, :, 0].set(actor)
        return ImaginedRollout(jnp.zeros((2, 2)), jnp.zeros((2, 2)), features)

    monkeypatch.setattr(imagination, "imagine", rollout)
    evaluate = jax.jit(
        jax.value_and_grad(
            lambda value: actor_loss(
                value,
                None,
                lambda features, _: features[..., 0],
                None,
                env,
                None,
                None,
                key=jax.random.key(0),
                horizon=2,
                gamma=gamma,
            )[0]
        )
    )
    loss, gradient = evaluate(jnp.asarray(1000.0))
    assert loss == pytest.approx(-(gamma**2) * bound)
    assert gradient == 0
    for value in (bound / 2, -1000.0):
        loss, gradient = evaluate(jnp.asarray(value))
        assert loss == pytest.approx(-(gamma**2) * value)
        assert gradient == pytest.approx(-(gamma**2))


@pytest.mark.parametrize("bias", [15.0, -100.0, 1000.0])
def test_critic_fits_raw_predictions_and_rolls_back_finite_divergence(
    toy_runner, monkeypatch, bias
):
    from bello_mujoco.supermimic import optimization

    env, config, runner = toy_runner
    leaves = eqx.filter(runner.learner.critic, eqx.is_array)
    critic = jax.tree.map(jnp.zeros_like, leaves)
    critic = eqx.tree_at(
        lambda value: value.network.layers[-1].bias, critic, jnp.array([bias])
    )
    runner = replace(runner, learner=replace(runner.learner, critic=critic))
    collect, _, improve = compile_updates(env, config)
    for _ in range(2):
        runner, _ = collect(runner, env.reference_bank)
    count, width = config.imagined_batch_size, runner.learner.critic_norm.mean.size

    def controlled(actor, *args, **kwargs):
        loss = sum(jnp.sum(x) * 0 for x in jax.tree.leaves(actor))
        rollout = ImaginedRollout(
            jnp.zeros((2, count)),
            jnp.zeros((2, count)),
            jnp.zeros((3, count, width)),
        )
        return loss, rollout

    monkeypatch.setattr(optimization, "actor_loss", controlled)
    result, metrics = improve(runner, env.reference_bank)
    bound = env.mdp.maximum_return(config.gamma, env.max_episode_steps)
    assert metrics["diagnostic/critic_target_max"] <= bound
    assert metrics["diagnostic/critic_value_abs_max"] == abs(bias)
    if bias == 1000:
        assert np.isfinite(metrics["critic/loss"])
        assert metrics["invalid/critic_loss"] == 1
        with pytest.raises(RuntimeError, match="invalid/actor_critic_update"):
            require_valid_metrics(metrics)
        for actual, expected in zip(
            jax.tree.leaves(result.learner),
            jax.tree.leaves(runner.learner),
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)
    else:
        require_valid_metrics(metrics)
        fitted = result.learner.critic.network.layers[-1].bias[0]
        if bias > 0:
            assert bound < fitted < bias
        else:
            assert bias < fitted < -bound
            assert metrics["diagnostic/critic_target_max"] < -bound


def test_reward_ceiling_violation_rejects_the_entire_update(toy_runner, monkeypatch):
    from bello_mujoco.supermimic import optimization

    env, config, runner = toy_runner
    collect, _, improve = compile_updates(env, config)
    for _ in range(2):
        runner, _ = collect(runner, env.reference_bank)
    count, width = config.imagined_batch_size, runner.learner.critic_norm.mean.size

    def controlled(actor, *args, **kwargs):
        loss = sum(jnp.sum(x) * 0 for x in jax.tree.leaves(actor))
        return loss, ImaginedRollout(
            jnp.full((2, count), 2 * env.mdp.maximum_step_reward),
            jnp.zeros((2, count)),
            jnp.zeros((3, count, width)),
        )

    monkeypatch.setattr(optimization, "actor_loss", controlled)
    result, metrics = improve(runner, env.reference_bank)
    assert metrics["invalid/return_bound"] == 1
    for actual, expected in zip(
        jax.tree.leaves(result.learner), jax.tree.leaves(runner.learner), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


def test_imagination_stops_extrapolating_after_full_termination(toy_runner):
    env, config, runner = toy_runner
    collect, _, _ = compile_updates(env, config)
    runner, _ = collect(runner, env.reference_bank)
    starts = runner.replay.sample_starts(
        jax.random.key(3),
        2,
        history_frames=2,
        feedback_frames=1,
        horizon=8,
    )

    class ExplodingAfterFall:
        def step(self, state, action, *args):
            return replace(
                state,
                position=state.position.at[2].add(-1),
                joint_position=state.joint_position * 1e20 + 1 + action,
                joint_velocity=state.joint_position * 1e30,
            )

    learner = runner.learner
    norms = tuple(
        getattr(learner, name)
        for name in (
            "input_norm",
            "output_norm",
            "observation_norm",
            "goal_norm",
        )
    )
    evaluate = eqx.filter_jit(
        eqx.filter_value_and_grad(
            lambda actor: actor_loss(
                actor,
                ExplodingAfterFall(),
                learner.critic,
                starts,
                env,
                norms,
                learner.critic_norm,
                key=jax.random.key(4),
                horizon=8,
                gamma=config.gamma,
            ),
            has_aux=True,
        )
    )
    (loss, rollout), gradient = evaluate(learner.actor)
    assert np.isfinite(loss)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves((rollout, gradient)))
    np.testing.assert_array_equal(rollout.terminations, 1)
    np.testing.assert_array_equal(rollout.rewards[1:], 0)
    np.testing.assert_array_equal(rollout.critic_inputs[1:], 0)


@pytest.mark.parametrize("terminated", [True, False])
def test_critic_excludes_only_unreachable_tail(toy_runner, monkeypatch, terminated):
    from bello_mujoco.supermimic import optimization

    env, config, runner = toy_runner
    collect, _, improve = compile_updates(env, config)
    for _ in range(2):
        runner, _ = collect(runner, env.reference_bank)
    count = config.imagined_batch_size
    width = runner.learner.critic_norm.mean.size

    def controlled_rollout(actor, *args, **kwargs):
        features = jnp.zeros((3, count, width)).at[1:].set(1e20)
        rewards = jnp.full((2, count), -1e20).at[0].set(0.1)
        terms = jnp.full((2, count), float(terminated))
        loss = sum(jnp.sum(x) * 0 for x in jax.tree.leaves(actor))
        return loss, ImaginedRollout(rewards, terms, features)

    monkeypatch.setattr(optimization, "actor_loss", controlled_rollout)
    result, metrics = improve(runner, env.reference_bank)
    if terminated:
        require_valid_metrics(metrics)
        assert result.learner.critic_norm.count == count
        np.testing.assert_array_equal(result.learner.critic_norm.mean, 0)
        assert metrics["critic/loss"] < 1
        assert metrics["diagnostic/critic_target_abs_max"] < 1
    else:
        with pytest.raises(
            RuntimeError, match="critic/loss|invalid/actor_critic_update"
        ):
            require_valid_metrics(metrics)
        for actual, previous in zip(
            jax.tree.leaves(result.learner),
            jax.tree.leaves(runner.learner),
            strict=True,
        ):
            np.testing.assert_array_equal(actual, previous)
