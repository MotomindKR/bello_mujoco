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
                interval_seconds=0.02,
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
        data, applied, velocity = current._maybe_apply_random_push_batch(
            Data(qvel=initial), jnp.ones(2, dtype=jnp.int32), jax.random.PRNGKey(0)
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
