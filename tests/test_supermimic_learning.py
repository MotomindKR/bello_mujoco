from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from bello_mujoco.supermimic import rotations as rot
from bello_mujoco.supermimic.model import (
    Actor,
    Critic,
    Moments,
    NetworkConfig,
    WorldModel,
    apply_delta,
    state_delta,
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
        jnp.zeros((2, 1)),
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


def test_replay_masks_terminal_successors_and_unfilled_storage():
    replay = ReplayBuffer.empty(transition(0), 5)
    for time in range(3):
        replay = jax.jit(ReplayBuffer.insert)(
            replay, transition(time, terminal=time == 2)
        )
    window = replay.sample_windows(jax.random.key(1), 8, 5)
    np.testing.assert_array_equal(
        window.mask, jnp.tile(jnp.array([1.0, 1.0, 1.0, 0.0, 0.0]), (8, 1))
    )
    np.testing.assert_array_equal(
        window.transitions.state.position[:, :3, 0], jnp.tile(jnp.arange(3), (8, 1))
    )
    assert np.all(window.transitions.terminal[:, 2])


def test_replay_wrap_and_history_do_not_cross_episodes_or_evicted_records():
    replay = ReplayBuffer.empty(transition(0), 5)
    for time in range(9):
        replay = replay.insert(
            transition(
                time, episode=int(time >= 6), step=time - 6 if time >= 6 else time
            )
        )
    starts = jax.jit(
        lambda b, k: b.sample_starts(
            k, 64, history_frames=3, feedback_frames=2, horizon=2
        )
    )(replay, jax.random.key(3))
    assert starts.valid
    times = np.asarray(starts.transition.reference_frame)
    assert np.all(times >= 6)
    expected = np.maximum(times[:, None] - np.arange(2, -1, -1), 6)
    np.testing.assert_array_equal(starts.history[..., 1], expected)
    np.testing.assert_array_equal(starts.history[..., 0], np.ones_like(expected))
    np.testing.assert_array_equal(
        starts.feedback_history.position[..., 0],
        np.maximum(times[:, None] - np.arange(2), 6),
    )
    exhausted = replace(
        replay,
        data=replace(
            replay.data, remaining_steps=jnp.zeros_like(replay.data.remaining_steps)
        ),
    )
    assert not exhausted.sample_starts(
        jax.random.key(4), 2, history_frames=2, feedback_frames=1, horizon=2
    ).valid


def test_replay_masks_context_changes_even_without_a_terminal_flag():
    replay = ReplayBuffer.empty(transition(0), 4)
    for time in range(4):
        tr = transition(time)
        if time >= 2:
            tr = replace(tr, context=jnp.ones((2, 1)))
        replay = replay.insert(tr)
    windows = replay.sample_windows(jax.random.key(5), 3, 4)
    np.testing.assert_array_equal(windows.mask, [[1, 1, 0, 0]] * 3)


def test_weighted_moments_and_local_deltas_are_consistent():
    values = jnp.array([[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]])
    state = Moments.empty(2).update(values, jnp.array([1.0, 1.0, 0.0]))
    np.testing.assert_allclose(state.mean, [2.0, 3.0])
    np.testing.assert_allclose(state.variance, [1.0, 1.0])
    updated = state.update(jnp.array([[5.0, 6.0]]))
    np.testing.assert_allclose(updated.mean, [3.0, 4.0])
    np.testing.assert_allclose(updated.variance, [8 / 3, 8 / 3])
    original = jax.tree.map(lambda x: x[0], transition(0).state)
    original = replace(original, quaternion=rot.exp(jnp.array([0.1, -0.2, 0.4])))
    delta = jnp.linspace(-0.02, 0.02, 14)
    result = apply_delta(original, delta)
    np.testing.assert_allclose(state_delta(original, result), delta, atol=1e-6)
    np.testing.assert_allclose(
        jax.jacfwd(lambda d: state_delta(original, apply_delta(original, d)))(
            jnp.zeros(14)
        ),
        np.eye(14),
        atol=1e-5,
    )


def test_dynamics_fits_logged_motor_transitions_with_finite_gradients():
    keys = jax.random.split(jax.random.key(7), 3)
    config = NetworkConfig(
        dynamics_widths=(32, 32), actor_widths=(16,), critic_widths=(16,), latent_size=4
    )
    model = WorldModel(1, 1, config=config, key=keys[0])
    tr = transition(0)
    tr = jax.tree.map(lambda x: jnp.repeat(x, 16, axis=0), tr)
    commands = jax.random.uniform(keys[1], (32, 1), minval=-1, maxval=1)
    successor = replace(
        tr.state,
        joint_position=tr.state.joint_position + 0.02 * commands,
        joint_velocity=commands,
    )
    tr = replace(tr, successor=successor, motor=commands)
    inputs = model.features(tr.state, tr.previous_motor, tr.motor, tr.context)
    targets = state_delta(tr.state, tr.successor)
    input_norm = Moments.empty(inputs.shape[-1]).update(inputs)
    output_norm = Moments.empty(targets.shape[-1]).update(targets)
    window = ReplayWindow(jax.tree.map(lambda x: x[:, None], tr), jnp.ones((32, 1)))
    optimizer = optax.adam(0.01)
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(net, opt_state):
        loss, gradient = eqx.filter_value_and_grad(world_model_loss)(
            net, window, input_norm, output_norm, error_floor=1e-3
        )
        updates, opt_state = optimizer.update(
            gradient, opt_state, eqx.filter(net, eqx.is_inexact_array)
        )
        return eqx.apply_updates(net, updates), opt_state, loss

    initial = float(
        world_model_loss(model, window, input_norm, output_norm, error_floor=1e-3)
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
    assert critic(jnp.ones((3, 7)), Moments.empty(7)).shape == (3,)
