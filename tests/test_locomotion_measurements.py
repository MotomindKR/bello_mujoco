"""Synthetic CPU/JIT checks for shared measurements, without robot assets."""

from dataclasses import dataclass, replace
from functools import partial
from types import SimpleNamespace

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bello_mujoco.envs.contacts import contact_forces, debounce_contact
from bello_mujoco.envs.foot_geometry import heightfield_surface_normal
from bello_mujoco.envs.locomotion_config import (
    BelloEnvConfig,
    ContactConfig,
    ObservationConfig,
    ObservationLatencyConfig,
    ObservationLowPassFilterConfig,
    ObservationNoiseConfig,
)
from bello_mujoco.envs.locomotion import BelloLocomotionEnv
from bello_mujoco.envs.rewards import CommandConfig
from bello_mujoco.envs.observations import (
    advance_feedback_history,
    filter_feedback,
    noisy_feedback,
    sample_feedback_delay,
)


def test_contact_classification_masks_worlds_padding_and_support_geometry():
    model = SimpleNamespace(
        mj_model=SimpleNamespace(ngeom=4, opt=SimpleNamespace(cone=0)),
        geom_body_ids=np.asarray([0, 10, 20, 10]),
        foot_body_ids=np.asarray([10, 20]),
        foot_geom_ids=np.asarray([1, 2]),
        terrain_geom_ids=np.asarray([0]),
        robot_geom_ids=np.asarray([1, 2, 3]),
        nonfoot_robot_geom_ids=np.asarray([3]),
        nonfoot_robot_body_ids=np.asarray([10]),
    )

    def evaluate(world):
        data = SimpleNamespace(
            _impl=SimpleNamespace(
                contact__dim=jnp.ones(5, dtype=jnp.int32),
                contact__geom=jnp.asarray([[0, 1], [2, 0], [0, 3], [1, 2], [0, 1]]),
                nacon=jnp.asarray([4]),
                contact__worldid=jnp.asarray([0, 1, 0, 0, 0]),
                contact__efc_address=jnp.asarray([[0], [1], [2], [3], [-1]]),
                efc__force=jnp.asarray([2.0, 4.0, 8.0, 5.0]),
            )
        )
        return contact_forces(model, ContactConfig(), data, world)

    contact, force, undesired, collision = jax.jit(jax.vmap(evaluate))(jnp.arange(2))
    np.testing.assert_array_equal(contact, [[1, 0], [0, 1]])
    np.testing.assert_array_equal(force, [[2, 0], [0, 4]])
    np.testing.assert_array_equal(undesired, [1, 0])
    np.testing.assert_array_equal(collision, [1, 0])


def test_debounce_uses_separate_on_off_thresholds_and_clears_at_done():
    config = ContactConfig(foot_contact_on_seconds=0.04, foot_contact_off_seconds=0.06)
    step = jax.jit(
        lambda raw, previous, elapsed, done: debounce_contact(
            raw,
            previous,
            elapsed,
            done,
            config,
            0.02,
        )
    )
    raw = jnp.asarray([1.0, 0.0])
    previous = 1.0 - raw
    elapsed = jnp.zeros(2)
    previous, elapsed = step(raw, previous, elapsed, False)
    np.testing.assert_array_equal(previous, [0, 1])
    previous, elapsed = step(raw, previous, elapsed, False)
    np.testing.assert_array_equal(previous, [1, 1])
    previous, elapsed = step(raw, previous, elapsed, True)
    np.testing.assert_array_equal(previous, [1, 0])
    np.testing.assert_array_equal(elapsed, [0, 0])


@pytest.mark.parametrize("enabled", [False, True])
def test_feedback_history_preserves_noise_filter_order_and_raw_joint_position(enabled):
    previous = jnp.asarray([0.0] * 8 + [-1.0] + [0.0] * 4)
    measured = previous + 0.1
    history = jnp.stack([previous, previous * 2, previous * 3])
    noise = ObservationNoiseConfig(enabled=True)
    filtering = ObservationLowPassFilterConfig(
        enabled=enabled, noise_before_filter=enabled
    )
    key = jax.random.PRNGKey(19)
    advance = jax.jit(
        lambda h, m, k: advance_feedback_history(h, m, k, 0.02, noise, filtering)
    )
    result = advance(history, measured, key)
    noisy = noisy_feedback(measured, noise, key) if enabled else measured
    expected = filter_feedback(previous, noisy, 0.02, filtering)
    np.testing.assert_allclose(result[0], expected, atol=1e-6)
    np.testing.assert_array_equal(result[1:], history[:-1])
    np.testing.assert_allclose(result[0, 9:11], noisy[9:11], atol=1e-6)


def test_feedback_delay_is_seeded_inclusive_and_optional():
    config = ObservationLatencyConfig(enabled=True, delay_steps=(1, 3))
    keys = jax.random.split(jax.random.PRNGKey(7), 64)
    sample = jax.jit(jax.vmap(lambda key: sample_feedback_delay(config, key)))
    values = sample(keys)
    np.testing.assert_array_equal(values, sample(keys))
    assert set(np.asarray(values).tolist()) == {1, 2, 3}
    assert sample_feedback_delay(config, None) == 0
    assert sample_feedback_delay(ObservationLatencyConfig(), keys[0]) == 0


@partial(jax.tree_util.register_dataclass, data_fields=[], meta_fields=["nworld"])
@dataclass(frozen=True)
class ResetWorld:
    nworld: int

    replace = replace


@chex.dataclass(frozen=True)
class ResetData:
    qpos: jax.Array
    qvel: jax.Array
    ctrl: jax.Array
    _impl: ResetWorld


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("noise_before_filter", [False, True])
@pytest.mark.parametrize("seeded", [False, True])
def test_reset_initializes_noisy_feedback_history(
    monkeypatch, batched, noise_before_filter, seeded
):
    for name in ("kinematics", "forward"):
        monkeypatch.setattr(
            f"bello_mujoco.envs.locomotion.mjx.{name}", lambda model, data: data
        )
    measured = jnp.arange(13, dtype=jnp.float32)
    noise = ObservationNoiseConfig(enabled=True)
    env = SimpleNamespace(
        config=BelloEnvConfig(
            observation=ObservationConfig(noise=noise),
            observation_low_pass_filter=ObservationLowPassFilterConfig(
                noise_before_filter=noise_before_filter
            ),
        ),
        command_config=CommandConfig(),
        model=SimpleNamespace(
            make_data=lambda: ResetData(
                qpos=jnp.zeros(3),
                qvel=jnp.zeros(2),
                ctrl=jnp.zeros(2),
                _impl=ResetWorld(nworld=1),
            ),
            default_qpos=jnp.zeros(3),
            default_qvel=jnp.zeros(2),
            foot_body_ids=np.asarray([0, 1]),
            root_qpos_adr=0,
            mjx_model=None,
        ),
        action_size=2,
        actor_feedback_size=13,
        maximum_observation_delay_steps=2,
        _resolve_command=lambda command, key: command,
        _resolve_batch_command=lambda n, command, key: jnp.tile(command, (n, 1)),
        _locomotion_right_axis=lambda: jnp.asarray([0.0, -1.0, 0.0]),
        action_to_ctrl=lambda action, offset=None: action,
        _apply_reset_randomization=lambda data, key, world: data,
        _sample_observation_delay=lambda key: jnp.asarray(0),
        _raw_tracking_ang_vel_z=lambda state: jnp.asarray(0.0),
        _actor_feedback=lambda state: measured,
        observation=lambda state, key: state.actor_feedback_history[0],
        observation_batch=lambda state, key: state.actor_feedback_history[:, 0],
        _joint_angle_offset_batch=lambda n: jnp.zeros((n, 2)),
        _reset_foot_clearance_adjustment=lambda state, world: jnp.asarray(0.0),
        initial_command_resample_step_counts=lambda n, key: jnp.zeros(n),
    )
    env._reset = partial(BelloLocomotionEnv._reset, env)
    key = jax.random.PRNGKey(13) if seeded else None

    def reset(key):
        if batched:
            return BelloLocomotionEnv.reset_batch(env, 2, jnp.zeros(3), key)
        return env._reset(jnp.zeros(3), key, 0, forward=False, align_clearance=False)

    state, _ = jax.jit(reset)(key)
    expected = measured
    if seeded and noise_before_filter:
        obs_key = jax.random.split(key, 3 if batched else 5)[2]
        feedback_key = jax.random.fold_in(obs_key, 1)
        if batched:
            expected = jax.vmap(lambda k: noisy_feedback(measured, noise, k))(
                jax.random.split(feedback_key, 2)
            )
        else:
            expected = noisy_feedback(measured, noise, feedback_key)
    expected = jnp.broadcast_to(expected, (2, 13)) if batched else expected
    expected = jnp.repeat(expected[..., None, :], 3, axis=-2)
    np.testing.assert_allclose(state.actor_feedback_history, expected, atol=1e-6)


def test_surface_normals_match_numpy_jit_and_vmap():
    xy = np.asarray([[0.2, -0.3], [0.1, 0.4]], dtype=np.float32)

    def height(p):
        return 0.2 * p[..., 0] ** 2 - 0.3 * p[..., 1]

    native = heightfield_surface_normal(xy, height, array_module=np)

    def normal(p):
        return heightfield_surface_normal(p, height, array_module=jnp)

    compiled = jax.jit(normal)(jnp.asarray(xy))
    batched = jax.jit(jax.vmap(normal))(jnp.asarray(xy))
    expected = np.stack([-0.4 * xy[:, 0], np.full(2, 0.3), np.ones(2)], axis=-1)
    expected /= np.linalg.norm(expected, axis=-1, keepdims=True)
    for value in (native, compiled, batched):
        np.testing.assert_allclose(value, expected, atol=1e-5)
