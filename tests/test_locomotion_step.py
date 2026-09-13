"""Synthetic scalar/batch transition tests; no robot assets or GPU required."""

from dataclasses import fields
from functools import partial
from bello_mujoco.envs.locomotion_step import (
    step_physics_with_impacts,
    step_outcome,
    advance_step_state,
)
from types import SimpleNamespace

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bello_mujoco.envs.locomotion import BelloLocomotionEnv
from bello_mujoco.envs.locomotion_state import BelloState


@chex.dataclass(frozen=True)
class PhysicsData:
    ctrl: jax.Array
    tick: jax.Array


class ToyEnv(BelloLocomotionEnv):
    @property
    def decimation(self):
        return 3

    @property
    def control_timestep(self):
        return 0.03

    def _contact_terms(self, state, world_id):
        contact = state.data.tick >= jnp.asarray([1, 2])
        force = contact * jnp.asarray([10.0, 20.0])
        velocity = self._foot_world_linear_velocity(state)
        return (
            contact,
            force,
            velocity,
            jnp.zeros(2),
            jnp.asarray(0.0),
            jnp.asarray(0.0),
        )

    def _foot_world_linear_velocity(self, state):
        return jnp.full((2, 3), state.data.tick, dtype=jnp.float32)

    def requested_actuator_force(self, data):
        return data.ctrl

    def done_flags(self, state, world_id):
        done = state.step_count >= 3
        return {
            "done/any": done,
            "done/terminal": done,
            "done/nonfinite": jnp.asarray(False),
        }

    def reward(self, state, next_state, action, requested_action, *args):
        return jnp.sum(action), {"reward/track_lin_vel_xy": jnp.asarray(0.25)}

    def sanitize_nonfinite_reward(self, reward, info, flags):
        return reward, info

    def _locomotion_right_axis(self):
        return jnp.asarray([0.0, -1.0, 0.0])


def toy_env():
    env = object.__new__(ToyEnv)
    for field in fields(BelloLocomotionEnv):
        object.__setattr__(env, field.name, field.default)
    object.__setattr__(
        env,
        "model",
        SimpleNamespace(
            foot_body_ids=np.asarray([0, 1]),
            mj_model=SimpleNamespace(nu=1, opt=SimpleNamespace(timestep=0.01)),
            mjx_model=object(),
        ),
    )
    return env


def toy_state():
    values = {field.name: jnp.asarray(0.0) for field in fields(BelloState)}
    values.update(
        data=PhysicsData(ctrl=jnp.asarray([0.0]), tick=jnp.asarray(0)),
        last_action=jnp.asarray([0.0]),
        last_action_delta=jnp.asarray([0.0]),
        actor_feedback_history=jnp.zeros((1, 1)),
        actor_feedback_delay_steps=jnp.asarray(0),
        last_foot_contact=jnp.zeros(2, dtype=bool),
        foot_contact_debounce_time=jnp.zeros(2),
        command=jnp.asarray([0.5, 0.0, 0.0]),
        command_target=jnp.asarray([0.5, 0.0, 0.0]),
        standing_transition_pending=jnp.asarray(0),
        next_start_left=jnp.asarray(True),
        step_count=jnp.asarray(1),
        episode_length=jnp.asarray(1),
        last_episode_length=jnp.asarray(0),
        command_resample_step_count=jnp.asarray(1),
    )
    return BelloState(**values)


def batched(tree):
    return jax.tree.map(lambda x: jnp.stack([x, x]), tree)


def assert_batched_equal(batch, scalar):
    for actual, expected in zip(
        jax.tree.leaves(batch), jax.tree.leaves(scalar), strict=True
    ):
        np.testing.assert_allclose(actual, jnp.stack([expected, expected]), atol=1e-6)


def test_impact_accumulation_matches_scalar_and_batch(monkeypatch):
    monkeypatch.setattr(
        "bello_mujoco.envs.locomotion_step.mjx.step",
        lambda model, data: data.replace(tick=data.tick + 1),
    )
    env, state = toy_env(), toy_state()
    requested = state.data.replace(ctrl=jnp.asarray([2.0]))
    scalar = jax.jit(partial(step_physics_with_impacts, env))(state, requested)
    batch = jax.jit(lambda state, data: step_physics_with_impacts(env, state, data, 2))(
        batched(state), batched(requested)
    )
    assert_batched_equal(batch, scalar)
    _, touchdown, impulse, peak, velocity, force = scalar
    np.testing.assert_array_equal(touchdown, [True, True])
    np.testing.assert_allclose(impulse, [0.3, 0.4], atol=1e-6)
    np.testing.assert_allclose(peak, [10.0, 20.0])
    np.testing.assert_allclose(velocity, [[0.0] * 3, [1.0] * 3])
    np.testing.assert_allclose(force, [2.0])


@pytest.mark.parametrize("done", [False, True])
def test_post_physics_episode_accounting_matches_vmap(monkeypatch, done):
    monkeypatch.setattr(
        "bello_mujoco.envs.locomotion_step.gait_diagnostics", lambda *args: {}
    )
    env, state = toy_env(), toy_state()
    state = state.replace(
        episode_return=jnp.asarray(5.0), last_episode_return=jnp.asarray(9.0)
    )
    reward_state = advance_step_state(
        env,
        state,
        state,
        jnp.asarray([2.0]),
        state.actor_feedback_history,
        jnp.asarray(0.0),
    ).replace(step_count=jnp.asarray(3 if done else 2))
    impacts = (
        jnp.zeros(2),
        jnp.zeros(2),
        jnp.zeros(2),
        jnp.zeros((2, 3)),
        jnp.zeros(1),
    )
    args = (state, reward_state, jnp.asarray([2.0]), impacts, jnp.asarray(0.0))
    scalar = jax.jit(partial(step_outcome, env))(*args)
    batch = jax.jit(jax.vmap(partial(step_outcome, env)))(*batched(args))
    assert_batched_equal(batch, scalar)
    next_state, reward, terminal, info = scalar
    assert float(reward) == 2.0
    assert bool(terminal) == done
    assert float(next_state.episode_return) == 7.0
    assert float(next_state.last_episode_return) == (7.0 if done else 9.0)
    assert float(info["episode_return"]) == (7.0 if done else 0.0)
    assert int(next_state.episode_length) == 2
