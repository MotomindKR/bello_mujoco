from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bello_mujoco.envs.locomotion import (
    GaitConfig,
    advance_gait_phase,
    advance_standing_blend,
    desired_gait_foot_contact,
    gait_frequency,
    gait_phase_features,
    next_start_preference,
    resolve_gait_command_transition,
    slew_command,
)
from bello_mujoco.envs.rewards import CommandConfig


GAIT = GaitConfig(
    speed_dependent=True,
    stop_extra_steps=1,
    command_slew_enabled=True,
    directional_start=True,
    standing_reward_ramp_seconds=0.2,
)
COMMAND = CommandConfig(standing_transitions_enabled=True)


def test_cadence_counts_individual_footsteps():
    commands = jnp.asarray(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]
    )
    actual = jax.jit(lambda c: gait_frequency(c, GAIT))(commands)
    np.testing.assert_allclose(
        actual, (1.6 + 0.35 * np.linalg.norm(commands[:, :2], axis=1)) / 2
    )
    assert float(actual[1]) == pytest.approx(0.8875)


@pytest.mark.parametrize("request_phase", [0.0, 0.1, 0.49, 0.5, 0.8, 0.99])
def test_stop_decelerates_across_two_boundaries(request_phase):
    command = jnp.asarray([0.4, -0.3, 0.2])
    phase = jnp.asarray(request_phase)
    pending = jnp.asarray(0)
    resolve = jax.jit(
        lambda c, p, n, pending: resolve_gait_command_transition(
            c, jnp.zeros(3), p, n, pending, COMMAND, GAIT, elapsed_seconds=0.02
        )
    )
    command, phase, pending = resolve(command, phase, phase, pending)
    assert int(pending) == 2
    boundaries = 0
    previous_norm = float(jnp.linalg.norm(command))
    for _step in range(100):
        next_phase = advance_gait_phase(phase, command, COMMAND, GAIT, 0.02, pending)
        crossed = (float(phase) < 0.5 <= float(next_phase)) or float(
            next_phase
        ) < float(phase)
        boundaries += int(crossed)
        command, phase, pending = resolve(command, phase, next_phase, pending)
        norm = float(jnp.linalg.norm(command))
        assert norm <= previous_norm + 1e-6
        previous_norm = norm
        if int(pending) == 0:
            assert boundaries == 2
            np.testing.assert_array_equal(command, np.zeros(3))
            break
        assert (
            float(gait_phase_features(phase, command, COMMAND, pending, GAIT)[2]) == 1.0
        )
    else:
        pytest.fail("stop transition failed to finish within two seconds")


@pytest.mark.parametrize("left_axis", [(0.0, 1.0), (-1.0, 0.0)])
@pytest.mark.parametrize("direction,first_swing", [(1, 0), (-1, 1)])
def test_lateral_start_uses_robot_left_axis(left_axis, direction, first_swing):
    command = jnp.asarray([*np.asarray(left_axis) * direction * 0.4, 0.0])
    _, phase, pending = resolve_gait_command_transition(
        jnp.zeros(3),
        command,
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        jnp.asarray(0),
        COMMAND,
        GAIT,
        jnp.asarray(True),
        jnp.asarray(left_axis),
    )
    assert not int(pending)
    contacts = desired_gait_foot_contact(jnp.mod(phase + 0.11, 1.0), GAIT)
    assert float(contacts[first_swing]) == 0.0
    assert float(contacts[1 - first_swing]) == 1.0


def test_restart_cancels_stop_without_resetting_swing():
    command, phase, pending = resolve_gait_command_transition(
        jnp.asarray([0.1, 0.1, 0.0]),
        jnp.asarray([0.0, -0.5, 0.0]),
        jnp.asarray(0.7),
        jnp.asarray(0.72),
        jnp.asarray(1),
        COMMAND,
        GAIT,
        elapsed_seconds=0.02,
    )
    assert float(phase) == pytest.approx(0.72)
    assert int(pending) == 0
    assert np.linalg.norm(command[:2] - np.array([0.1, 0.1])) == pytest.approx(0.02)


def test_forward_start_preference_alternates_and_blend_is_bounded():
    preference = jnp.asarray(True)
    phases = []
    for _ in range(4):
        command, phase, _ = resolve_gait_command_transition(
            jnp.zeros(3),
            jnp.asarray([0.5, 0.0, 0.0]),
            jnp.asarray(0.0),
            jnp.asarray(0.0),
            jnp.asarray(0),
            COMMAND,
            GAIT,
            preference,
            elapsed_seconds=0.02,
        )
        phases.append(float(phase))
        preference = next_start_preference(
            jnp.zeros(3), command, 0, preference, COMMAND, GAIT
        )
    assert phases == [0.5, 0.0, 0.5, 0.0]
    blend = jnp.asarray(1.0)
    blend = advance_standing_blend(blend, jnp.zeros(3), 1, COMMAND, GAIT, 0.02)
    assert float(blend) == 0.0
    for _ in range(10):
        blend = advance_standing_blend(blend, jnp.zeros(3), 0, COMMAND, GAIT, 0.02)
    assert float(blend) == pytest.approx(1.0)


def test_scheduler_batches_have_independent_stop_counters():
    resolve = jax.jit(
        lambda c, r, p, n, pending: resolve_gait_command_transition(
            c, r, p, n, pending, COMMAND, GAIT, elapsed_seconds=0.02
        )
    )
    command, phase, pending = resolve(
        jnp.asarray([[0.4, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        jnp.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, -0.4, 0.0]]),
        jnp.asarray([0.49, 0.99, 0.0]),
        jnp.asarray([0.51, 0.01, 0.0]),
        jnp.asarray([2, 1, 0]),
    )
    np.testing.assert_array_equal(pending, [1, -1, 0])
    np.testing.assert_allclose(command[1], [0.38, 0.0, 0.0])
    assert float(phase[1]) == 0.0
    assert float(phase[2]) == 0.0


def test_shared_slew_limits_start_reverse_and_stop_without_same_tick_update():
    command = jnp.zeros((2, 3))
    for target in (
        jnp.array([[0.5, 0.5, 1.0], [-0.5, 0.5, -1.0]]),
        jnp.array([[-0.5, -0.5, -1.0], [0.5, -0.5, 1.0]]),
        jnp.zeros((2, 3)),
    ):
        for _ in range(110):
            np.testing.assert_array_equal(
                slew_command(command, target, GAIT, 0.0), command
            )
            updated = jax.jit(slew_command, static_argnums=2)(
                command, target, GAIT, 0.02
            )
            delta = np.asarray(updated - command)
            assert np.all(np.linalg.norm(delta[:, :2], axis=-1) <= 0.020001)
            assert np.all(np.abs(delta[:, 2]) <= 0.020001)
            command = updated
        np.testing.assert_array_equal(command, target)


def test_completed_buffer_freezes_phase_until_slew_finishes():
    phase, pending, command = (
        jnp.asarray(0.5),
        jnp.asarray(-1),
        jnp.asarray([0.1, 0.0, 0.0]),
    )
    for _ in range(10):
        advanced = advance_gait_phase(phase, command, COMMAND, GAIT, 0.02, pending)
        assert float(advanced) == float(phase)
        command, phase, pending = resolve_gait_command_transition(
            command,
            jnp.zeros(3),
            phase,
            advanced,
            pending,
            COMMAND,
            GAIT,
            elapsed_seconds=0.02,
        )
        if int(pending) == 0:
            break
    assert int(pending) == 0
    np.testing.assert_array_equal(command, jnp.zeros(3))


@pytest.mark.parametrize(
    "updates",
    [
        {"stop_extra_steps": -1},
        {"step_frequency_base": 0.0},
        {"standing_reward_ramp_seconds": float("nan")},
    ],
)
def test_invalid_gait_settings_fail(updates):
    with pytest.raises(ValueError):
        replace(GAIT, **updates)
