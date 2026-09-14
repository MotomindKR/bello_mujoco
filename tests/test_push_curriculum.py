"""Persistent disturbance scheduling with synthetic state, no robot or GPU."""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bello_mujoco.envs.locomotion import BelloLocomotionEnv
from bello_mujoco.envs.locomotion_config import (
    BelloEnvConfig,
    RandomPushConfig,
    TorsoComCurriculumConfig,
    DoneConfig,
    validate_locomotion_config,
)
from bello_mujoco.envs.rewards import RewardConfig, CommandConfig
from bello_mujoco.envs.randomization import apply_push_batch


class CurriculumEnv(BelloLocomotionEnv):
    def __post_init__(self):
        pass  # Only the curriculum and additive push algorithms are under test.


@dataclass(frozen=True)
class Model:
    mj_model: object
    mjx_model: object
    torso_com_offset: object
    root_qvel_adr: int = 0


@chex.dataclass(frozen=True)
class PhysicsModel:
    body_ipos: jax.Array


@chex.dataclass(frozen=True)
class Data:
    qvel: jax.Array


@pytest.mark.parametrize("torso_enabled", [False, True])
def test_shared_persistent_clock_scales_pushes_under_jit(monkeypatch, torso_enabled):
    monkeypatch.setattr("bello_mujoco.envs.locomotion.mujoco.mj_name2id", lambda *_: 0)
    env = CurriculumEnv(
        Model(
            SimpleNamespace(opt=SimpleNamespace(timestep=0.005)),
            PhysicsModel(body_ipos=jnp.zeros((2, 1, 3))),
            jnp.ones((2, 3)),
        ),
        config=BelloEnvConfig(
            random_push=RandomPushConfig(
                enabled=True,
                curriculum_steps=100,
                initial_scale=0.2,
                interval_seconds=(0.02, 0.02),
                lin_vel_x=(0.5, 0.5),
                lin_vel_y=(0.0, 0.0),
            ),
            torso_com_curriculum=TorsoComCurriculumConfig(
                enabled=torso_enabled, ramp_steps=50
            ),
        ),
    )
    initial = jnp.asarray([[0.25, 0.5], [-0.75, 1.0]])

    @jax.jit
    def pushed(clock):
        current = env.with_task_state(clock)
        data, remaining, applied, velocity = current._maybe_apply_random_push_batch(
            Data(qvel=initial), jnp.zeros(2, dtype=jnp.int32), jax.random.PRNGKey(0)
        )
        return data.qvel, applied, velocity

    for clock, scale in ((0.0, 0.2), (50.0, 0.6), (100.0, 1.0), (200.0, 1.0)):
        qvel, applied, velocity = pushed(jnp.asarray([clock]))
        np.testing.assert_array_equal(applied, [True, True])
        np.testing.assert_allclose(velocity, [[0.5 * scale, 0.0]] * 2)
        np.testing.assert_allclose(qvel, initial + velocity)
    np.testing.assert_array_equal(env.initial_task_state(), [0.0])
    np.testing.assert_array_equal(
        env.update_task_state(jnp.asarray([99.0]), {}), [100.0]
    )
    np.testing.assert_array_equal(
        env.update_task_state(jnp.asarray([100.0]), {}), [100.0]
    )
    assert env.task_state_metrics(jnp.asarray([0.0]))[
        "curriculum/push_scale"
    ] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="shape"):
        env.with_task_state(jnp.zeros(2))


@pytest.mark.parametrize(
    "changes",
    [
        {"curriculum_steps": -1},
        {"curriculum_steps": True},
        {"initial_scale": 0.0},
        {"initial_scale": float("nan")},
        {"initial_scale": 1.1},
        {"initial_scale": 0.2},
        {"curriculum_steps": 10},
    ],
)
def test_invalid_push_curriculum_is_rejected_at_initialization(changes):
    config = BelloEnvConfig(random_push=replace(RandomPushConfig(), **changes))
    with pytest.raises(ValueError, match="random_push"):
        validate_locomotion_config(
            config, RewardConfig(), CommandConfig(), DoneConfig(), physics_dt=0.005
        )


def test_push_intervals_are_independent_bounded_and_replayable():
    config = RandomPushConfig(
        enabled=True,
        interval_seconds=(0.06, 0.14),
        lin_vel_x=(0.25, 0.25),
        lin_vel_y=(-0.5, -0.5),
    )
    model = SimpleNamespace(root_qvel_adr=1)
    initial_data = Data(qvel=jnp.ones((16, 5)))
    initial_timer = jnp.full(16, -1, dtype=jnp.int32)
    keys = jax.random.split(jax.random.PRNGKey(73), 200)

    @jax.jit
    def rollout(data, timer, keys):
        def step(carry, key):
            data, timer, applied, velocity = apply_push_batch(
                model, config, 0.02, *carry, key
            )
            return (data, timer), (applied, velocity)

        return jax.lax.scan(step, (data, timer), keys)

    (data, timer), (applied, velocity) = rollout(initial_data, initial_timer, keys)
    events = np.asarray(applied)
    first_steps = []
    for world in range(16):
        times = np.flatnonzero(events[:, world])
        intervals = np.diff(np.r_[0, times])
        assert np.all((intervals >= 3) & (intervals <= 7))
        assert np.unique(intervals).size > 1
        first_steps.append(times[0])
    assert len(set(first_steps)) > 1
    np.testing.assert_array_equal(data.qvel[:, [0, 3, 4]], 1)
    np.testing.assert_allclose(data.qvel[:, 1:3], 1 + velocity.sum(axis=0))

    # A saved timer plus the rollout PRNG sequence reproduces continuation.
    (saved_data, saved_timer), _ = rollout(initial_data, initial_timer, keys[:81])
    (resumed_data, resumed_timer), (resumed_events, _) = rollout(
        saved_data, saved_timer, keys[81:]
    )
    np.testing.assert_array_equal(resumed_events, events[81:])
    np.testing.assert_array_equal(resumed_timer, timer)
    np.testing.assert_array_equal(resumed_data.qvel, data.qvel)

    # Resetting one world must not restart the other worlds' countdowns.
    (_, reset_timer), (reset_events, _) = rollout(
        saved_data, saved_timer.at[0].set(-1), keys[81:82]
    )
    (_, intact_timer), (intact_events, _) = rollout(
        saved_data, saved_timer, keys[81:82]
    )
    assert not bool(reset_events[0, 0])
    np.testing.assert_array_equal(reset_events[:, 1:], intact_events[:, 1:])
    np.testing.assert_array_equal(reset_timer[1:], intact_timer[1:])


@pytest.mark.parametrize("enabled,with_rng", [(False, True), (True, False)])
def test_disabled_pushes_leave_data_and_timer_unchanged(enabled, with_rng):
    data = Data(qvel=jnp.ones((3, 4)))
    timer = jnp.asarray([-1, 0, 2], dtype=jnp.int32)
    result, remaining, applied, velocity = apply_push_batch(
        SimpleNamespace(root_qvel_adr=0),
        RandomPushConfig(enabled=enabled),
        0.02,
        data,
        timer,
        jax.random.PRNGKey(0) if with_rng else None,
    )
    np.testing.assert_array_equal(result.qvel, data.qvel)
    np.testing.assert_array_equal(remaining, timer)
    assert not np.asarray(applied).any()
    assert not np.asarray(velocity).any()


@pytest.mark.parametrize(
    "interval",
    [
        (0.0, 1.0),
        (0.01, 0.1),
        (0.021, 0.039),
        (2.0, 1.0),
        (1.0, float("nan")),
    ],
)
def test_invalid_push_timing_is_rejected_before_rollout(interval):
    config = BelloEnvConfig(random_push=RandomPushConfig(interval_seconds=interval))
    with pytest.raises(ValueError, match="random_push.interval_seconds"):
        validate_locomotion_config(
            config, RewardConfig(), CommandConfig(), DoneConfig(), physics_dt=0.005
        )
