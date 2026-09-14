from dataclasses import replace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from bello_mujoco.configuration import locomotion_training_profile
from bello_mujoco.envs.locomotion import BelloLocomotionEnv
from bello_mujoco.envs.rewards import StandingRewardConfig, compute_locomotion_reward
from bello_mujoco.robstride import ratings
from test_rewards import reward_config, reward_inputs


def test_standing_motor_penalty_covers_held_motors_and_half_rated_thresholds():
    config = reward_config(
        stand_motor_continuous_torque_excess=-8.0,
        standing=StandingRewardConfig(recovery_relaxation=True),
    )
    inputs = reward_inputs(
        policy_joint_indices=jnp.asarray([0]),
        motor_continuous_torque=jnp.asarray([40.0, 5.0]),
        actuator_force=jnp.asarray([20.0, -2.5]),
    )
    compute = jax.jit(lambda x: compute_locomotion_reward(config, x)[1])
    key = "reward/stand_motor_continuous_torque_excess"
    assert compute(inputs)[key] == 0.0
    overloaded = inputs.replace(actuator_force=jnp.asarray([20.0, -5.0]))
    assert compute(overloaded)[key] == pytest.approx(-0.08)
    assert compute(overloaded.replace(standing_reward_blend=jnp.asarray(0.5)))[
        key
    ] == pytest.approx(-0.04)
    assert compute(overloaded.replace(command=jnp.asarray([0.5, 0.0, 0.0])))[key] == 0.0
    assert compute(
        overloaded.replace(
            lin_vel_xy=jnp.asarray([1.0, 0.0]),
            projected_gravity=jnp.asarray([0.5, 0.0, -0.866]),
            foot_contact=jnp.asarray([True, False]),
        )
    )[key] == pytest.approx(-0.08)
    with pytest.raises(ValueError, match="continuous ratings"):
        compute(overloaded.replace(motor_continuous_torque=None))
    with pytest.raises(ValueError, match="cover all actuator"):
        compute(overloaded.replace(motor_continuous_torque=jnp.ones(1)))
    with pytest.raises(ValueError, match="not both"):
        reward_config(
            stand_motor_continuous_torque_excess=-8.0,
            stand_hip_roll_continuous_torque_excess=-8.0,
        )


def test_mini_ratings_bind_all_28_motors_including_differential_ankles(
    bello_mini_asset, monkeypatch
):
    profile = locomotion_training_profile("bello_mini_locomotion")
    env = BelloLocomotionEnv.load_prepared(
        bello_mini_asset, reward_config=profile.reward_config, load_mjx=False
    )
    mapping = env._motor_torque_ratings
    assert len(mapping.names) == env.model.mj_model.nu == 28
    assert len(set(mapping.names)) == 28
    thresholds = dict(
        zip(mapping.names, np.asarray(mapping.continuous_torque_nm) / 2, strict=True)
    )
    assert thresholds["left_hip_pitch_motor"] == 20.0
    assert thresholds["left_hip_roll_motor"] == 10.0
    assert thresholds["left_ankle_motor_1"] == thresholds["left_ankle_motor_2"] == 3.0
    assert thresholds["left_shoulder_yaw_motor"] == 5.5
    assert thresholds["right_wrist_roll_motor"] == 2.5
    for name, threshold in thresholds.items():
        if name.startswith("left_"):
            assert thresholds[name.replace("left_", "right_", 1)] == threshold
    for side in ("left", "right"):
        ids = [mapping.names.index(f"{side}_ankle_motor_{i}") for i in (1, 2)]
        assert np.all(
            env.model.mj_model.actuator_trntype[ids] == mujoco.mjtTrn.mjTRN_TENDON
        )
        # Already in motor coordinates: pure pitch can combine two 3 Nm motor
        # outputs into 6 Nm joint torque without exceeding either threshold.
        force = np.zeros(28)
        force[ids] = [-3.0, 3.0]
        joint_force = env.model.actuator_coordinate_transform.T @ force
        assert np.max(np.abs(joint_force)) == pytest.approx(6.0)
        normalized_excess = np.maximum(
            np.abs(force) / (0.5 * np.asarray(mapping.continuous_torque_nm)) - 1, 0
        )
        assert np.sum(normalized_excess) == 0.0
    model = env.model.mj_model
    model.actuator_gear[0, 0] = 2.0
    try:
        with pytest.raises(ValueError, match="unit actuator gear"):
            ratings.motor_torque_ratings(model, env.model.robot)
    finally:
        model.actuator_gear[0, 0] = 1.0
    target = int(model.actuator_trnid[0, 0])
    model.actuator_trnid[0, 0] = model.actuator_trnid[1, 0]
    try:
        with pytest.raises(ValueError, match="one actuator per physical motor"):
            ratings.motor_torque_ratings(model, env.model.robot)
    finally:
        model.actuator_trnid[0, 0] = target
    unknown = dict(ratings.RATED_OUTPUT_TORQUE_NM)
    del unknown["ROBSTRIDE_00"]
    monkeypatch.setattr(ratings, "RATED_OUTPUT_TORQUE_NM", unknown)
    with pytest.raises(ValueError, match="no verified continuous torque"):
        replace(env)
