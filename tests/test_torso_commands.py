"""Synthetic torso frames, commands and observation contracts; CPU only."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from bello_mujoco.envs.gait import GaitConfig, gait_is_active
from bello_mujoco.envs.locomotion_config import (
    BelloEnvConfig,
    DoneConfig,
    ObservationConfig,
    ObservationNoiseConfig,
    validate_locomotion_config,
)
from bello_mujoco.envs.rewards import (
    CommandConfig,
    RewardConfig,
    RewardWeights,
    TorsoCommandConfig,
)
from bello_mujoco.envs.torso import (
    slew_torso_command,
    torso_orientation_error,
    torso_command_from_quaternions,
)
from bello_mujoco.envs.observations import build_observation


def wxyz(rotation):
    return jnp.asarray(np.roll(rotation.as_quat(), 1), dtype=jnp.float32)


@pytest.mark.parametrize("forward", [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
def test_orientation_tracks_lean_twist_not_world_heading_or_pelvis_tilt(forward):
    # A nontrivial CAD-frame rotation must not change anatomical command signs.
    neutral = Rotation.from_euler("xyz", [0.1, -0.2, 1.1])
    basis = Rotation.from_euler("z", np.arctan2(forward[1], forward[0]))
    command = jnp.asarray([0.15, -0.2, 0.4])
    heading = Rotation.from_euler("z", 2.9)
    pelvis = heading * Rotation.from_euler("xyz", [-0.25, 0.3, 0.0])
    target = (
        heading * basis * Rotation.from_euler("xyz", command) * basis.inv() * neutral
    )
    error = jax.jit(torso_orientation_error)
    args = (wxyz(pelvis), command, wxyz(neutral), jnp.asarray(forward))
    assert float(error(wxyz(target), *args)) < 1e-6
    np.testing.assert_allclose(
        torso_command_from_quaternions(
            wxyz(target), wxyz(pelvis), wxyz(neutral), forward
        ),
        command,
        atol=1e-6,
    )
    assert float(error(wxyz(heading * neutral), *args)) > 0.3
    assert float(error(-wxyz(target), *args)) < 1e-6
    # A known local rotation error is reported in radians, including near pi.
    for delta in (0.1, 3.13):
        actual = target * Rotation.from_rotvec([delta, 0.0, 0.0])
        assert float(error(wxyz(actual), *args)) == pytest.approx(delta, abs=1e-6)


def test_sampling_slew_and_standing_remain_independent_under_jit_vmap():
    config = TorsoCommandConfig(
        roll=(-0.2, 0.2),
        pitch=(-0.3, 0.3),
        yaw=(-0.4, 0.4),
        slew_rate=(0.2, 0.3, 0.5),
        neutral_probability=0.25,
    )
    targets = jax.jit(lambda key: config.sample(key, (128,)))(jax.random.PRNGKey(4))
    bounds = np.asarray(config.ranges)
    assert np.all(targets >= bounds[:, 0]) and np.all(targets <= bounds[:, 1])
    neutral = np.all(targets == 0, axis=-1)
    assert 0 < neutral.sum() < len(targets)
    current = jnp.zeros_like(targets)
    step = jax.jit(lambda q: slew_torso_command(q, targets, config.slew_rate, 0.02))
    result = step(current)
    np.testing.assert_allclose(
        result,
        jax.vmap(lambda q, t: slew_torso_command(q, t, config.slew_rate, 0.02))(
            current, targets
        ),
    )
    assert np.all(np.abs(result) <= np.asarray(config.slew_rate) * 0.02 + 1e-8)
    for _ in range(60):
        current = step(current)
    np.testing.assert_allclose(current, targets, atol=1e-6)
    assert not bool(
        gait_is_active(
            jnp.zeros(3),
            CommandConfig(torso=config),
            gait_config=GaitConfig(command_slew_enabled=True),
        )
    )


def test_nonfinite_out_of_range_and_conflicting_rewards_are_rejected():
    torso = TorsoCommandConfig(roll=(-0.2, 0.2))
    for value in ([np.nan, 0, 0], [0, 0, 0.1], [0, 0]):
        with pytest.raises(ValueError):
            torso.validate_target(value)
    for kwargs in (
        dict(pitch=(0.1, 0.2)),
        dict(yaw=(-2.0, 2.0)),
        dict(slew_rate=(0.0, 1.0, 1.0)),
        dict(neutral_probability=np.nan),
    ):
        with pytest.raises(ValueError):
            TorsoCommandConfig(**kwargs)
    command = CommandConfig(torso=torso)
    reward = RewardConfig(weights=RewardWeights(track_torso_orientation=3.0))
    with pytest.raises(ValueError, match="upright"):
        validate_locomotion_config(
            BelloEnvConfig(), reward, command, DoneConfig(), physics_dt=0.005
        )
    reward = replace(
        reward, weights=replace(reward.weights, flat_orientation=0.0, stand_upright=0.0)
    )
    validate_locomotion_config(
        BelloEnvConfig(), reward, command, DoneConfig(), physics_dt=0.005
    )


def test_torso_observation_is_clean_even_with_feedback_latency_and_noise():
    terms = tuple(jnp.zeros(n) for n in (3, 3, 3, 3, 1, 3, 2, 2, 2, 2))
    target = jnp.asarray([0.1, -0.2, 0.3])
    config = ObservationConfig(
        noise=ObservationNoiseConfig(
            enabled=True,
            command=(-1.0, 1.0),
            joint_pos=(-1.0, 1.0),
        )
    )
    # Delayed feedback contains sensors/joints only; torso intent stays current.
    observation = build_observation(
        terms,
        jnp.zeros(3),
        config,
        jax.random.PRNGKey(2),
        feedback=jnp.zeros(13),
        torso_command=target,
    )
    np.testing.assert_array_equal(observation[12:15], target)
    np.testing.assert_array_equal(observation[15:18], jnp.zeros(3))
