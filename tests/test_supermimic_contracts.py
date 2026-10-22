from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bello_mujoco.envs.feedback import delayed_ctrl
from bello_mujoco.models.target_limits import TargetLimits
from bello_mujoco.supermimic import rotations as rot
from bello_mujoco.supermimic.config import ControlConfig, ObservationConfig
from bello_mujoco.supermimic.control import ActionMapping
from bello_mujoco.supermimic.observations import ObservationCodec
from bello_mujoco.supermimic.state import PhysicalState


def test_coupled_commands_respect_limits_and_differentiate_the_executed_target():
    transform = np.array([[1.0, -1.0], [1.0, 1.0]])
    limits = TargetLimits(
        np.array([[-2.0, 2.0]] * 2),
        np.array([[-0.5, 0.5]] * 2),
        np.linalg.inv(transform),
        np.array([0.1, 0.1]),
    )
    model = SimpleNamespace(
        actuator_joint_names=("a", "b"),
        actuator_coordinate_transform=transform,
        policy_ctrl_indices=np.arange(2),
        target_limits=limits,
        default_joint_target=np.zeros(2),
        default_ctrl=np.zeros(2),
    )
    actuators = SimpleNamespace(
        by_name={
            name: SimpleNamespace(effort_limit=4.0, gains=(1.0, 0.1))
            for name in model.actuator_joint_names
        }
    )
    mapping = ActionMapping.from_model(model, actuators)
    command = jax.jit(mapping.command)(
        jnp.array([1.0, 1.0]), jnp.zeros(2), jnp.zeros(2)
    )
    np.testing.assert_allclose(command.requested, [0.0, 2.0])
    assert np.all(np.abs(command.accepted) <= 0.1)
    values = limits.matrix @ command.accepted
    assert np.all(values >= limits.bounds[:, 0]) and np.all(
        values <= limits.bounds[:, 1]
    )

    def target(a):
        return mapping.command(a, jnp.zeros(2), jnp.zeros(2)).accepted

    a = jnp.array([0.01, 0.02])
    np.testing.assert_allclose(jax.jacfwd(target)(a), transform, atol=1e-6)
    jac = jax.jacfwd(target)(jnp.array([0.3, 0.8]))
    h = 1e-3
    finite = jnp.stack(
        [
            (
                target(jnp.array([0.3, 0.8]) + h * d)
                - target(jnp.array([0.3, 0.8]) - h * d)
            )
            / (2 * h)
            for d in jnp.eye(2)
        ],
        axis=1,
    )
    np.testing.assert_allclose(jac, finite, atol=1e-5)
    bad = replace(ControlConfig(), action_clip=0.5)
    clipped = ActionMapping.from_model(model, actuators, bad).command(
        jnp.ones(2), jnp.zeros(2), jnp.zeros(2)
    )
    np.testing.assert_array_equal(clipped.action, [0.5, 0.5])


def test_observation_and_goal_are_mini_native_and_heading_invariant():
    codec = ObservationCodec(
        ObservationConfig(future_frames=3, history_frames=2), jnp.zeros(2)
    )
    state = PhysicalState(
        jnp.array([1.0, 2.0, 1.0]),
        rot.exp(jnp.array([0.1, 0.2, 0.3])),
        jnp.array([0.3, 0.2, 0.1]),
        jnp.array([0.1, 0.2, 0.3]),
        jnp.array([0.2, -0.3]),
        jnp.array([1.0, 2.0]),
    )
    future = jax.tree.map(lambda value: jnp.stack((value, value, value)), state)
    goal = codec.goal(state, future)
    assert goal.shape == (codec.goal_size,)
    sample = codec.sample(state, jnp.zeros(2), state.quaternion, jnp.array([0.1, 0.1]))
    np.testing.assert_allclose(sample[:2], [0.1, -0.4], atol=1e-7)
    assert sample.shape == (codec.sample_size,)
    history = codec.advance_history(codec.initial_history(sample), sample + 1.0)
    np.testing.assert_array_equal(history, jnp.stack((sample, sample + 1.0)))
    yaw = rot.exp(jnp.array([0.0, 0.0, 0.8]))

    def move(s):
        return replace(
            s,
            position=rot.rotate(yaw, s.position) + jnp.array([4.0, -2.0, 0.0]),
            quaternion=rot.multiply(yaw, s.quaternion),
            linear_velocity=rot.rotate(yaw, s.linear_velocity),
            angular_velocity=rot.rotate(yaw, s.angular_velocity),
        )

    np.testing.assert_allclose(codec.goal(move(state), move(future)), goal, atol=1e-6)
    np.testing.assert_allclose(
        codec.sample(
            move(state), jnp.zeros(2), move(state).quaternion, jnp.array([0.1, 0.1])
        ),
        sample,
        atol=1e-6,
    )
    assert np.isfinite(
        jax.jacfwd(
            lambda q: codec.sample(
                replace(state, quaternion=q),
                jnp.zeros(2),
                state.quaternion,
                jnp.zeros(2),
            )
        )(state.quaternion)
    ).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(decimation=0),
        dict(decimation=True),
        dict(physics_timestep=float("nan")),
        dict(command_delay_seconds=0.003),
        dict(command_delay_seconds=0.02),
        dict(action_clip=0.0),
    ],
)
def test_control_rejects_invalid_timing(kwargs):
    with pytest.raises(ValueError):
        ControlConfig(**kwargs)


def test_command_delay_selects_whole_physics_substeps():
    config = ControlConfig(command_delay_seconds=0.01)
    actual = jax.jit(
        jax.vmap(
            lambda i: delayed_ctrl(
                jnp.array([1.0]),
                jnp.array([2.0]),
                i,
                delay_seconds=config.command_delay_seconds,
                physics_dt=config.physics_timestep,
            )
        )
    )(jnp.arange(config.decimation))
    np.testing.assert_array_equal(actual[:, 0], [1.0, 1.0, 2.0, 2.0])


def test_corpus_frame_clock_rejects_rounding_and_fractional_drift():
    from types import SimpleNamespace
    from bello_mujoco.envs.supermimic import validate_frame_clock

    for fps in (25.0, 50.0, 100.0):
        validate_frame_clock(SimpleNamespace(fps=fps, clock_end=1000), 0.02)
    with pytest.raises(ValueError, match="power-of-two"):
        validate_frame_clock(SimpleNamespace(fps=60.0, clock_end=1000), 0.02)
    with pytest.raises(ValueError, match="capacity"):
        validate_frame_clock(SimpleNamespace(fps=25.0, clock_end=2**23), 0.02)
