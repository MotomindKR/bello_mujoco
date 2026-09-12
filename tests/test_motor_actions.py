import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from bello_mujoco.envs.actions import bounded_motor_target, motor_default
from bello_mujoco.envs.locomotion import BelloLocomotionEnv
from bello_mujoco.evaluation.render import mujoco_action_to_ctrl


def test_asymmetric_motor_mapping_is_batched_and_preserves_held_controls():
    limits = jnp.array([[-2.0, 3.0], [-4.0, 1.0], [-1.0, 1.0]])
    default = jnp.array([1.0, -1.0, 0.25])
    policy = jnp.array([0, 1])
    actions = jnp.array([[-2.0, -1.0], [0.0, 0.0], [1.0, 2.0]])
    targets = jax.jit(
        jax.vmap(lambda a: bounded_motor_target(a, default, limits, policy, 1.0))
    )(actions)
    np.testing.assert_allclose(
        targets, [[-2.0, -4.0, 0.25], [1.0, -1.0, 0.25], [3.0, 1.0, 0.25]]
    )


@pytest.mark.parametrize("fixture", ["bello_asset", "bello_mini_asset"])
def test_policy_controls_motors_independently_and_keeps_physical_tendons(
    request, fixture
):
    env = BelloLocomotionEnv.from_compiled(
        request.getfixturevalue(fixture), load_mjx=False
    )
    model = env.model
    limits = model.mj_model.actuator_ctrlrange
    for side in ("left", "right"):
        motor_name = f"{side}_ankle_motor_1"
        action_id = env.action_layout.names.index(motor_name)
        motor_id = model.policy_ctrl_indices[action_id]
        action = jnp.zeros(env.action_size).at[action_id].set(0.5)
        ctrl = np.asarray(env.action_to_ctrl(action))
        changed = np.flatnonzero(np.abs(ctrl - model.default_ctrl) > 1e-6)
        np.testing.assert_array_equal(changed, [motor_id])
        assert ctrl[motor_id] == pytest.approx(
            0.5 * (model.default_ctrl[motor_id] + limits[motor_id, 1])
        )
        np.testing.assert_allclose(mujoco_action_to_ctrl(env, action), ctrl)

    data = mujoco.MjData(model.mj_model)
    data.qpos[:] = model.default_qpos
    data.qpos[model.actuator_qpos_indices] += np.linspace(
        -0.02, 0.02, model.mj_model.nu
    )
    mujoco.mj_forward(model.mj_model, data)
    np.testing.assert_allclose(
        data.actuator_length,
        model.actuator_coordinate_transform @ data.qpos[model.actuator_qpos_indices],
        atol=1e-7,
    )


def test_randomized_motor_neutral_preserves_motor_endpoints(bello_mini_asset):
    env = BelloLocomotionEnv.from_compiled(bello_mini_asset, load_mjx=False)
    offsets = jnp.asarray(
        np.random.default_rng(3).uniform(-0.02, 0.02, (4, env.model.mj_model.nu)),
        dtype=jnp.float32,
    )
    limits = env.model.mj_model.actuator_ctrlrange
    policy = env.model.policy_ctrl_indices
    for sign, column in ((-1.0, 0), (1.0, 1)):
        targets = jax.jit(jax.vmap(env.action_to_ctrl))(
            jnp.full((4, env.action_size), sign), offsets
        )
        np.testing.assert_allclose(
            targets[:, policy],
            np.broadcast_to(limits[policy, column], (4, env.action_size)),
            atol=1e-6,
        )
    targets = jax.vmap(env.action_to_ctrl)(jnp.zeros((4, env.action_size)), offsets)
    expected = jax.vmap(lambda offset: motor_default(env.model, offset))(offsets)
    np.testing.assert_allclose(
        targets, np.clip(expected, limits[:, 0], limits[:, 1]), atol=1e-6
    )


def test_motor_mapping_endpoints_do_not_round_outside_bounds():
    limits = jnp.asarray([[-1.41372, 1.41372]], dtype=jnp.float32)
    default = jnp.asarray([-0.325], dtype=jnp.float32)
    for action in (-1.0, 1.0):
        target = bounded_motor_target(
            jnp.asarray([action]), default, limits, jnp.asarray([0]), 1.0
        )
        assert jnp.all(target >= limits[:, 0])
        assert jnp.all(target <= limits[:, 1])
