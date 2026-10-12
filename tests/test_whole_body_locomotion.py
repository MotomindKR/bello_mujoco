from dataclasses import replace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from bello_mujoco.configuration import locomotion_training_profile
from bello_mujoco.envs.locomotion import BelloLocomotionEnv
from bello_mujoco.evaluation.render import mujoco_eval_state, reset_mujoco_data
from bello_mujoco.envs.whole_body import (
    centroidal_angular_momentum,
    swing_cross_track_velocity,
)


def test_centroidal_momentum_matches_native_mujoco_under_jit_and_vmap():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><body pos="0 0 1"><freejoint/>
      <geom type="box" size=".2 .1 .3"/>
      <body pos=".4 0 .2"><joint axis="0 1 0"/>
        <geom type="capsule" size=".1 .3"/>
      </body></body></worldbody></mujoco>""")
    rng = np.random.default_rng(73)
    inertias, velocities, expected = [], [], []
    for _ in range(8):
        model.body_mass[1:] *= rng.uniform(0.8, 1.2)
        model.body_inertia[1:] *= rng.uniform(0.8, 1.2)
        data = mujoco.MjData(model)
        mujoco.mj_setConst(model, data)
        data.qvel[:] = rng.normal(size=model.nv)
        mujoco.mj_integratePos(model, data.qpos, data.qvel, 0.2)
        mujoco.mj_forward(model, data)
        mujoco.mj_subtreeVel(model, data)
        inertias.append(data.cinert.copy())
        velocities.append(data.cvel.copy())
        expected.append(data.subtree_angmom[1].copy())
    actual = jax.jit(jax.vmap(centroidal_angular_momentum))(
        jnp.asarray(inertias), jnp.asarray(velocities)
    )
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize(
    "command", [[0.5, 0, 0], [0, 0.3, 0], [0, 0, 0.5], [0.3, 0.2, 0.4]]
)
def test_swing_penalty_allows_commanded_strafe_and_turn_tangents(command):
    command = jnp.asarray(command)
    positions = jnp.asarray([[0.1, 0.15, 0], [-0.1, -0.15, 0]])
    tangent = command[:2] + command[2] * jnp.stack(
        (-positions[:, 1], positions[:, 0]), axis=-1
    )
    compute = jax.jit(swing_cross_track_velocity)
    np.testing.assert_allclose(compute(command, positions, tangent * 2), 0, atol=1e-7)
    transverse = jnp.stack((-tangent[:, 1], tangent[:, 0]), axis=-1)
    assert np.all(np.asarray(compute(command, positions, tangent + transverse)) > 0)
    np.testing.assert_allclose(compute(jnp.zeros(3), positions, transverse), 0)


def test_ablation_profiles_have_equal_budgets_and_intentional_differences():
    baseline, tracking, gait, whole = [
        locomotion_training_profile(f"bello_mini_ablation_{name}")
        for name in ("baseline", "tracking", "gait", "whole_body")
    ]
    original = locomotion_training_profile("bello_mini_locomotion")
    assert baseline.reward_config == original.reward_config
    for profile in (tracking, gait, whole):
        assert profile.config == baseline.config
        assert profile.command_config == baseline.command_config
        assert profile.env_config == baseline.env_config
    assert not tracking.reward_config.tracking.respect_command_deadband
    assert (
        tracking.reward_config.weights.track_lin_vel_xy
        > baseline.reward_config.weights.track_lin_vel_xy
    )
    assert gait.reward_config.gait.unbounded_touchdown
    assert gait.asset_config == baseline.asset_config
    assert whole.reward_config == replace(
        gait.reward_config, limits_pose=whole.reward_config.limits_pose
    )
    actuators = whole.asset_config.actuator_config.joints
    assert all(a.mode == "policy" for a in actuators.values())
    assert set(dict(whole.reward_config.limits_pose.joint_pose_weights)) == set(
        actuators
    )


def test_whole_body_native_observation_action_and_reward_contract():
    profile = locomotion_training_profile("bello_mini_ablation_whole_body")
    env = BelloLocomotionEnv.load(
        profile.asset_config,
        env_config=profile.env_config,
        reward_config=profile.reward_config,
        command_config=profile.command_config,
        done_config=profile.done_config,
        load_mjx=False,
    )
    data = mujoco.MjData(env.model.mj_model)
    reset_mujoco_data(env, data)
    state = mujoco_eval_state(
        env,
        data,
        command=np.asarray([0.5, 0, 0]),
        last_action=np.zeros(env.action_size),
        foot_contact_debounce_time=jnp.zeros(2),
        last_foot_contact=jnp.ones(2),
        step_count=1,
        episode_return=jnp.asarray(0.0),
        episode_track_lin_vel_xy=jnp.asarray(0.0),
        episode_length=jnp.asarray(1),
    )
    assert env.action_size == env.model.mj_model.nu
    components = dict(env.observation_layout.components)
    assert components["joint_position"] == env.action_size
    assert components["joint_velocity"] == env.action_size
    assert components["last_action"] == env.action_size
    assert env.observation(state).shape == (env.observation_layout.size,)
    reward, terms = env.reward(
        state,
        state,
        jnp.zeros(env.action_size),
        jnp.asarray(False),
        jnp.ones(2),
        jnp.zeros(2, dtype=bool),
        jnp.zeros(2),
        jnp.zeros(2),
        jnp.zeros((2, 3)),
        jnp.zeros(env.model.mj_model.nu),
        jnp.zeros((2, 3)),
        jnp.zeros(2),
    )
    assert np.isfinite(reward)
    assert all(np.isfinite(value) for value in terms.values())
