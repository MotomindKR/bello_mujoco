from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from bello_mujoco.supermimic import rotations as rot
from bello_mujoco.supermimic.model import (
    Actor,
    Critic,
    Moments,
    NetworkConfig,
    WorldModel,
    world_model_loss,
)
from bello_mujoco.supermimic.replay import ReplayBuffer, ReplayWindow, Transition
from bello_mujoco.supermimic.state import PhysicalState


def transition(time, *, terminal=False, episode=0, step=None):
    state = PhysicalState(
        jnp.array([[time, 0.0, 1.0], [time, 1.0, 1.0]]),
        jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (2, 1)),
        jnp.zeros((2, 3)),
        jnp.zeros((2, 3)),
        jnp.full((2, 1), time / 10),
        jnp.zeros((2, 1)),
    )
    return Transition(
        state,
        state,
        state,
        jnp.zeros((2, 1)),
        jnp.zeros((2, 1)),
        jnp.zeros((2, 1)),
        jnp.zeros((2, 1)),
        jnp.tile(jnp.array([float(episode), float(time)]), (2, 1)),
        jnp.zeros((2, 6)),
        jnp.zeros((2, 2)),
        jnp.zeros(2),
        jnp.full(2, time),
        jnp.zeros(2, jnp.int32),
        jnp.full(2, episode),
        jnp.full(2, time if step is None else step),
        jnp.full(2, 20),
        jnp.zeros(2),
        jnp.full(2, terminal),
        jnp.zeros(2, bool),
    )


def test_replay_requires_complete_windows_and_excludes_internal_resets():
    replay = ReplayBuffer.empty(transition(0), 5)
    for time in range(3):
        replay = jax.jit(ReplayBuffer.insert)(
            replay, transition(time, terminal=time == 2)
        )
    assert not np.any(replay.sample_windows(jax.random.key(1), 8, 5).mask)
    window = replay.sample_windows(jax.random.key(1), 8, 3)
    assert window.transitions.action.shape == (8, 2, 1)
    np.testing.assert_array_equal(window.mask, 1)
    np.testing.assert_array_equal(
        window.transitions.state.position[:, :, 0], [[0, 1]] * 8
    )


def test_rejected_replay_rows_never_train_windows_or_start_imagination():
    replay = ReplayBuffer.empty(transition(0), 4)
    for time in range(3):
        sample = transition(time)
        valid = jnp.array([True, time != 1])
        if time == 1:
            sample = replace(sample, action=sample.action.at[1].set(jnp.nan))
        replay = replay.insert(sample, valid)
    window = replay.sample_windows(jax.random.key(3), 64, 3)
    np.testing.assert_array_equal(window.mask, 1)
    np.testing.assert_array_equal(window.transitions.state.position[..., 1], 0)
    assert np.isfinite(window.transitions.action).all()
    starts = replay.sample_starts(
        jax.random.key(4), 128, history_frames=1, feedback_frames=1
    )
    assert starts.valid
    assert not np.any(
        (np.asarray(starts.transition.reference_frame) == 1)
        & (np.asarray(starts.transition.state.position[:, 1]) == 1)
    )


def test_masked_world_model_tail_cannot_overflow_loss_or_gradients():
    class UnstableModel(eqx.Module):
        factor: jax.Array

        def step(self, state, action, context, input_norm):
            return replace(
                state, joint_position=self.factor * (state.joint_position + 1)
            )

    samples = jax.tree.map(lambda x: jnp.repeat(x[:, None], 8, axis=1), transition(0))
    window = ReplayWindow(samples, jnp.tile(jnp.array([1.0] + [0.0] * 7), (2, 1)))
    loss, gradient = eqx.filter_jit(eqx.filter_value_and_grad(world_model_loss))(
        UnstableModel(jnp.array(1e4)),
        window,
        Moments.empty(1),
        (1.0, 1.0, 1.0, 1.0),
    )
    assert float(loss) == pytest.approx(1e8 / 14, rel=1e-6)
    assert float(gradient.factor) == pytest.approx(2e4 / 14, rel=1e-6)


def test_replay_wrap_and_history_do_not_cross_episodes_or_evicted_records():
    replay = ReplayBuffer.empty(transition(0), 5)
    for time in range(9):
        replay = replay.insert(
            transition(
                time, episode=int(time >= 6), step=time - 6 if time >= 6 else time
            )
        )
    starts = jax.jit(
        lambda b, k: b.sample_starts(k, 64, history_frames=3, feedback_frames=2)
    )(replay, jax.random.key(3))
    assert starts.valid
    times = np.asarray(starts.transition.reference_frame)
    assert np.all(times >= 4)
    boundary = np.where(times >= 6, 6, 4)
    expected = np.maximum(times[:, None] - np.arange(2, -1, -1), boundary[:, None])
    np.testing.assert_array_equal(starts.history[..., 1], expected)
    np.testing.assert_array_equal(
        starts.history[..., 0], np.broadcast_to((times >= 6)[:, None], expected.shape)
    )
    np.testing.assert_array_equal(
        starts.feedback_history.position[..., 0],
        np.maximum(times[:, None] - np.arange(2), boundary[:, None]),
    )
    exhausted = replace(
        replay,
        data=replace(
            replay.data, remaining_steps=jnp.zeros_like(replay.data.remaining_steps)
        ),
    )
    assert not exhausted.sample_starts(
        jax.random.key(4), 2, history_frames=2, feedback_frames=1
    ).valid


def test_replay_excludes_context_changes_even_without_terminal_flags():
    replay = ReplayBuffer.empty(transition(0), 4)
    for time in range(4):
        tr = transition(time)
        if time >= 2:
            tr = replace(tr, context=jnp.ones_like(tr.context))
        replay = replay.insert(tr)
    assert not np.any(replay.sample_windows(jax.random.key(5), 3, 4).mask)


def test_weighted_moments_and_local_deltas_are_consistent():
    values = jnp.array([[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]])
    state = Moments.empty(2).update(values, jnp.array([1.0, 1.0, 0.0]))
    np.testing.assert_allclose(state.mean, [2.0, 3.0])
    np.testing.assert_allclose(state.variance, [1.0, 1.0])
    updated = state.update(jnp.array([[5.0, 6.0]]))
    np.testing.assert_allclose(updated.mean, [3.0, 4.0])
    np.testing.assert_allclose(updated.variance, [8 / 3, 8 / 3])
    np.testing.assert_allclose(updated.std(), np.sqrt(np.array([8 / 3, 8 / 3])) + 1e-2)


def test_dynamics_fits_logged_motor_transitions_with_finite_gradients():
    keys = jax.random.split(jax.random.key(7), 3)
    config = NetworkConfig(
        dynamics_widths=(32, 32), actor_widths=(16,), critic_widths=(16,), latent_size=4
    )
    from test_supermimic_runner import toy_runner

    env, _, _ = toy_runner.__wrapped__()
    model = WorldModel(env, config=config, key=keys[0])
    tr = transition(0)
    tr = jax.tree.map(lambda x: jnp.repeat(x, 16, axis=0), tr)
    commands = jax.random.uniform(keys[1], (32, 1), minval=-1, maxval=1)
    successor = replace(
        tr.state,
        joint_position=tr.state.joint_position
        + model.substep_coefficient * 0.02 * commands,
        joint_velocity=commands,
    )
    tr = replace(
        tr,
        successor=successor,
        action=commands,
        previous_motor=jnp.full_like(commands, 100),
        motor=jnp.full_like(commands, 200),
    )
    inputs = model.features(tr.state, tr.action)
    input_norm = Moments.empty(inputs.shape[-1]).update(inputs)
    window = ReplayWindow(jax.tree.map(lambda x: x[:, None], tr), jnp.ones((32, 1)))
    optimizer = optax.adam(0.01)
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(net, opt_state):
        loss, gradient = eqx.filter_value_and_grad(world_model_loss)(
            net, window, input_norm, (0.01, 0.02, 0.1, 3.14)
        )
        updates, opt_state = optimizer.update(
            gradient, opt_state, eqx.filter(net, eqx.is_inexact_array)
        )
        return eqx.apply_updates(net, updates), opt_state, loss

    initial = float(
        world_model_loss(model, window, input_norm, (0.01, 0.02, 0.1, 3.14))
    )
    for _ in range(40):
        model, optimizer_state, loss = step(model, optimizer_state)
    assert np.isfinite(loss)
    assert float(loss) < initial * 0.1
    actor = Actor(5, 2, 7, 1, config=config, key=keys[1])
    actions = actor(
        jnp.zeros((3, 2, 5)), jnp.zeros((3, 7)), Moments.empty(5), Moments.empty(7)
    )
    assert actions.shape == (3, 1)
    critic = Critic(7, config=config, key=keys[2])
    assert critic(jnp.ones((3, 7))).shape == (3,)


def test_dynamics_conditions_on_processed_action_and_excludes_sensor_latency():
    from test_supermimic_runner import toy_runner

    env, _, _ = toy_runner.__wrapped__()
    env = replace(
        env,
        mapping=replace(
            env.mapping, scale=jnp.array([0.3]), neutral_joint=jnp.array([0.2])
        ),
    )
    model = WorldModel(
        env, config=NetworkConfig(dynamics_widths=(8,)), key=jax.random.key(0)
    )
    tr = transition(0)
    action = jnp.full((2, 1), 25.0)
    features = model.features(tr.state, action)
    assert features.shape == (2, model.input_size)
    np.testing.assert_allclose(features[:, -1], 7.7)
    jac = jax.jacfwd(
        lambda a: model.features(jax.tree.map(lambda x: x[0], tr.state), a)
    )(action[0])
    assert jac[-1, 0] == pytest.approx(0.3)
    np.testing.assert_array_equal(
        model.context_features(tr.context),
        model.context_features(tr.context.at[:, -1].set(2.0)),
    )


def test_world_model_sums_time_errors_and_masks_resets_without_reweighting_windows():
    class IdentityModel:
        def step(self, state, *args):
            return state

    tr = transition(0)
    tr = replace(tr, successor=replace(tr.state, joint_position=jnp.ones((2, 1))))
    transitions = jax.tree.map(lambda x: jnp.repeat(x[:, None], 3, axis=1), tr)
    window = ReplayWindow(transitions, jnp.array([[1, 1, 1], [1, 0, 0]], dtype=float))
    loss = world_model_loss(
        IdentityModel(), window, Moments.empty(1), (1.0, 1.0, 1.0, 1.0)
    )
    # Average over four valid transitions and fourteen weighted physical channels.
    assert float(loss) == pytest.approx(1 / 14, rel=1e-6)


def test_actor_starts_include_the_last_valid_transition_of_short_motion_tails():
    replay = ReplayBuffer.empty(transition(0), 4)
    tail = replace(
        transition(0), remaining_steps=jnp.ones(2, jnp.int32), timeout=jnp.ones(2, bool)
    )
    replay = replay.insert(tail)
    starts = replay.sample_starts(
        jax.random.key(5), 8, history_frames=3, feedback_frames=1
    )
    assert starts.valid
    np.testing.assert_array_equal(starts.transition.remaining_steps, 1)
    np.testing.assert_array_equal(starts.history[..., 1], 0)


def test_model_loss_uses_fixed_physical_scales_and_matrix_rotation_error():
    class IdentityModel:
        def step(self, state, *args):
            return state

    tr = transition(0)
    target = replace(
        tr.state,
        position=tr.state.position.at[:, 2].add(0.02),
        quaternion=jnp.broadcast_to(rot.exp(jnp.array([0.0, 0.0, jnp.pi / 2])), (2, 4)),
        joint_position=tr.state.joint_position + 0.1,
        joint_velocity=tr.state.joint_velocity + 3.14,
        linear_velocity=tr.state.linear_velocity + 100.0,
    )
    tr = replace(tr, successor=target)
    window = ReplayWindow(jax.tree.map(lambda x: x[:, None], tr), jnp.ones((2, 1)))
    value = world_model_loss(
        IdentityModel(), window, Moments.empty(1), (0.01, 0.02, 0.1, 3.14)
    )
    expected = (4 + 2 / 0.02**2 + 1 + 1) / 14
    assert float(value) == pytest.approx(expected, rel=1e-6)
