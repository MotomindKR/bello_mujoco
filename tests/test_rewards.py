import jax.numpy as jnp
import pytest

from bello_mujoco.envs.rewards import (
    CommandConfig,
    FeetAirTimeRewardConfig,
    FootClearanceRewardConfig,
    FootContactRewardConfig,
    FootDistanceRewardConfig,
    FootSlipRewardConfig,
    GaitContactRewardConfig,
    PolicyJointDeviationRewardConfig,
    RewardConfig,
    TerminationRewardConfig,
    TorqueRewardConfig,
    TrackLinVelXYRewardConfig,
    UndesiredContactRewardConfig,
    compute_velocity_reward,
)


def test_base_height_reward_default_matches_nominal_stance() -> None:
    assert RewardConfig().base_height.target == pytest.approx(0.86)


def test_velocity_reward_prefers_matching_command() -> None:
    config = RewardConfig()
    common = {
        "config": config,
        "command": jnp.asarray([0.1, 0.0, 0.05], dtype=jnp.float32),
        "yaw_vel_z": jnp.asarray(0.05, dtype=jnp.float32),
        "lin_vel_z": jnp.asarray(0.0, dtype=jnp.float32),
        "ang_vel_xy": jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        "projected_gravity": jnp.asarray([0.0, 0.0, -1.0], dtype=jnp.float32),
        "base_height": jnp.asarray(config.base_height.target, dtype=jnp.float32),
        "joint_pos": jnp.zeros(4, dtype=jnp.float32),
        "joint_pos_rel": jnp.zeros(4, dtype=jnp.float32),
        "joint_vel": jnp.zeros(4, dtype=jnp.float32),
        "action": jnp.zeros(4, dtype=jnp.float32),
        "last_action": jnp.zeros(4, dtype=jnp.float32),
        "actuator_force": jnp.zeros(4, dtype=jnp.float32),
        "joint_limit_low": -jnp.ones(4, dtype=jnp.float32),
        "joint_limit_high": jnp.ones(4, dtype=jnp.float32),
        "hold_joint_indices": jnp.asarray([2, 3], dtype=jnp.int32),
        "policy_joint_indices": jnp.asarray([0, 1], dtype=jnp.int32),
        "foot_contact": jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        "foot_height": jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        "foot_lateral_speed_sq": jnp.zeros(2, dtype=jnp.float32),
        "foot_distance": jnp.asarray(0.3, dtype=jnp.float32),
        "gait_contact_mask": jnp.asarray([True, True]),
        "foot_air_time": jnp.zeros(2, dtype=jnp.float32),
        "last_foot_contact": jnp.asarray([True, True]),
        "undesired_contact": jnp.asarray(0.0, dtype=jnp.float32),
        "control_timestep": 0.02,
        "done": jnp.asarray(False),
    }

    matching_reward, matching_terms = compute_velocity_reward(
        lin_vel_xy=jnp.asarray([0.1, 0.0], dtype=jnp.float32), **common
    )
    mismatched_reward, _ = compute_velocity_reward(
        lin_vel_xy=jnp.asarray([-0.5, 0.5], dtype=jnp.float32), **common
    )

    assert matching_reward > mismatched_reward
    assert matching_terms["reward/track_lin_vel_xy"].shape == ()
    assert jnp.all(jnp.isfinite(matching_reward))


def test_reward_config_rejects_nonpositive_tracking_sigma() -> None:
    with pytest.raises(ValueError, match="track_lin_vel_xy.sigma"):
        RewardConfig(track_lin_vel_xy=TrackLinVelXYRewardConfig(sigma=0.0))


def test_command_config_rejects_invalid_resample_seconds() -> None:
    with pytest.raises(ValueError, match="resample_seconds must be positive"):
        CommandConfig(resample_seconds=0.0)


def test_velocity_reward_penalizes_termination() -> None:
    config = RewardConfig(termination=TerminationRewardConfig(weight=-3.0))
    common = {
        "config": config,
        "command": jnp.zeros(3, dtype=jnp.float32),
        "lin_vel_xy": jnp.zeros(2, dtype=jnp.float32),
        "yaw_vel_z": jnp.asarray(0.0, dtype=jnp.float32),
        "lin_vel_z": jnp.asarray(0.0, dtype=jnp.float32),
        "ang_vel_xy": jnp.zeros(2, dtype=jnp.float32),
        "projected_gravity": jnp.asarray([0.0, 0.0, -1.0], dtype=jnp.float32),
        "base_height": jnp.asarray(config.base_height.target, dtype=jnp.float32),
        "joint_pos": jnp.zeros(2, dtype=jnp.float32),
        "joint_pos_rel": jnp.zeros(2, dtype=jnp.float32),
        "joint_vel": jnp.zeros(2, dtype=jnp.float32),
        "action": jnp.zeros(2, dtype=jnp.float32),
        "last_action": jnp.zeros(2, dtype=jnp.float32),
        "actuator_force": jnp.zeros(2, dtype=jnp.float32),
        "joint_limit_low": -jnp.ones(2, dtype=jnp.float32),
        "joint_limit_high": jnp.ones(2, dtype=jnp.float32),
        "hold_joint_indices": jnp.asarray([1], dtype=jnp.int32),
        "policy_joint_indices": jnp.asarray([0], dtype=jnp.int32),
        "foot_contact": jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        "foot_height": jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        "foot_lateral_speed_sq": jnp.zeros(2, dtype=jnp.float32),
        "foot_distance": jnp.asarray(0.3, dtype=jnp.float32),
        "gait_contact_mask": jnp.asarray([True, True]),
        "foot_air_time": jnp.zeros(2, dtype=jnp.float32),
        "last_foot_contact": jnp.asarray([True, True]),
        "undesired_contact": jnp.asarray(0.0, dtype=jnp.float32),
        "control_timestep": 0.02,
    }

    alive_reward, _ = compute_velocity_reward(done=jnp.asarray(False), **common)
    done_reward, terms = compute_velocity_reward(done=jnp.asarray(True), **common)

    assert done_reward < alive_reward
    assert terms["reward/termination"] == -3.0


def test_contact_rewards_penalize_slip_and_undesired_contacts() -> None:
    config = RewardConfig(
        foot_slip=FootSlipRewardConfig(weight=-1.0),
        undesired_contact=UndesiredContactRewardConfig(weight=-2.0),
        foot_contact=FootContactRewardConfig(weight=0.0),
        foot_clearance=FootClearanceRewardConfig(weight=0.0),
    )
    common = {
        "config": config,
        "command": jnp.zeros(3, dtype=jnp.float32),
        "lin_vel_xy": jnp.zeros(2, dtype=jnp.float32),
        "yaw_vel_z": jnp.asarray(0.0, dtype=jnp.float32),
        "lin_vel_z": jnp.asarray(0.0, dtype=jnp.float32),
        "ang_vel_xy": jnp.zeros(2, dtype=jnp.float32),
        "projected_gravity": jnp.asarray([0.0, 0.0, -1.0], dtype=jnp.float32),
        "base_height": jnp.asarray(config.base_height.target, dtype=jnp.float32),
        "joint_pos": jnp.zeros(2, dtype=jnp.float32),
        "joint_pos_rel": jnp.zeros(2, dtype=jnp.float32),
        "joint_vel": jnp.zeros(2, dtype=jnp.float32),
        "action": jnp.zeros(2, dtype=jnp.float32),
        "last_action": jnp.zeros(2, dtype=jnp.float32),
        "actuator_force": jnp.zeros(2, dtype=jnp.float32),
        "joint_limit_low": -jnp.ones(2, dtype=jnp.float32),
        "joint_limit_high": jnp.ones(2, dtype=jnp.float32),
        "hold_joint_indices": jnp.asarray([], dtype=jnp.int32),
        "policy_joint_indices": jnp.asarray([0, 1], dtype=jnp.int32),
        "foot_contact": jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        "foot_height": jnp.asarray([0.0, 0.2], dtype=jnp.float32),
        "foot_distance": jnp.asarray(0.3, dtype=jnp.float32),
        "gait_contact_mask": jnp.asarray([True, False]),
        "foot_air_time": jnp.zeros(2, dtype=jnp.float32),
        "last_foot_contact": jnp.asarray([True, True]),
        "control_timestep": 0.02,
        "done": jnp.asarray(False),
    }

    clean_reward, _ = compute_velocity_reward(
        foot_lateral_speed_sq=jnp.zeros(2, dtype=jnp.float32),
        undesired_contact=jnp.asarray(0.0, dtype=jnp.float32),
        **common,
    )
    bad_reward, terms = compute_velocity_reward(
        foot_lateral_speed_sq=jnp.asarray([0.5, 0.0], dtype=jnp.float32),
        undesired_contact=jnp.asarray(1.0, dtype=jnp.float32),
        **common,
    )

    assert bad_reward < clean_reward
    assert terms["reward/foot_slip"] < 0.0
    assert terms["reward/undesired_contact"] == pytest.approx(-0.04)


def test_reward_terms_can_be_disabled_individually() -> None:
    config = RewardConfig(
        foot_slip=FootSlipRewardConfig(enabled=False, weight=-10.0),
        termination=TerminationRewardConfig(enabled=False, weight=-10.0),
    )

    _, terms = compute_velocity_reward(
        config=config,
        command=jnp.zeros(3, dtype=jnp.float32),
        lin_vel_xy=jnp.zeros(2, dtype=jnp.float32),
        yaw_vel_z=jnp.asarray(0.0, dtype=jnp.float32),
        lin_vel_z=jnp.asarray(0.0, dtype=jnp.float32),
        ang_vel_xy=jnp.zeros(2, dtype=jnp.float32),
        projected_gravity=jnp.asarray([0.0, 0.0, -1.0], dtype=jnp.float32),
        base_height=jnp.asarray(config.base_height.target, dtype=jnp.float32),
        joint_pos=jnp.zeros(2, dtype=jnp.float32),
        joint_pos_rel=jnp.zeros(2, dtype=jnp.float32),
        joint_vel=jnp.zeros(2, dtype=jnp.float32),
        action=jnp.zeros(2, dtype=jnp.float32),
        last_action=jnp.zeros(2, dtype=jnp.float32),
        actuator_force=jnp.zeros(2, dtype=jnp.float32),
        joint_limit_low=-jnp.ones(2, dtype=jnp.float32),
        joint_limit_high=jnp.ones(2, dtype=jnp.float32),
        hold_joint_indices=jnp.asarray([], dtype=jnp.int32),
        policy_joint_indices=jnp.asarray([], dtype=jnp.int32),
        foot_contact=jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        foot_height=jnp.zeros(2, dtype=jnp.float32),
        foot_lateral_speed_sq=jnp.asarray([5.0, 0.0], dtype=jnp.float32),
        foot_distance=jnp.asarray(0.3, dtype=jnp.float32),
        gait_contact_mask=jnp.asarray([True, False]),
        foot_air_time=jnp.zeros(2, dtype=jnp.float32),
        last_foot_contact=jnp.asarray([True, True]),
        undesired_contact=jnp.asarray(0.0, dtype=jnp.float32),
        control_timestep=0.02,
        done=jnp.asarray(True),
    )

    assert terms["reward/foot_slip"] == 0.0
    assert terms["reward/termination"] == 0.0


def test_torque_reward_penalizes_actuator_force() -> None:
    config = RewardConfig(torque=TorqueRewardConfig(enabled=True, weight=-2.0))

    _, terms = compute_velocity_reward(
        config=config,
        command=jnp.zeros(3, dtype=jnp.float32),
        lin_vel_xy=jnp.zeros(2, dtype=jnp.float32),
        yaw_vel_z=jnp.asarray(0.0, dtype=jnp.float32),
        lin_vel_z=jnp.asarray(0.0, dtype=jnp.float32),
        ang_vel_xy=jnp.zeros(2, dtype=jnp.float32),
        projected_gravity=jnp.asarray([0.0, 0.0, -1.0], dtype=jnp.float32),
        base_height=jnp.asarray(config.base_height.target, dtype=jnp.float32),
        joint_pos=jnp.zeros(2, dtype=jnp.float32),
        joint_pos_rel=jnp.zeros(2, dtype=jnp.float32),
        joint_vel=jnp.zeros(2, dtype=jnp.float32),
        action=jnp.zeros(2, dtype=jnp.float32),
        last_action=jnp.zeros(2, dtype=jnp.float32),
        actuator_force=jnp.asarray([2.0, -1.0], dtype=jnp.float32),
        joint_limit_low=-jnp.ones(2, dtype=jnp.float32),
        joint_limit_high=jnp.ones(2, dtype=jnp.float32),
        hold_joint_indices=jnp.asarray([], dtype=jnp.int32),
        policy_joint_indices=jnp.asarray([], dtype=jnp.int32),
        foot_contact=jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        foot_height=jnp.zeros(2, dtype=jnp.float32),
        foot_lateral_speed_sq=jnp.zeros(2, dtype=jnp.float32),
        foot_distance=jnp.asarray(0.3, dtype=jnp.float32),
        gait_contact_mask=jnp.asarray([True, True]),
        foot_air_time=jnp.zeros(2, dtype=jnp.float32),
        last_foot_contact=jnp.asarray([True, True]),
        undesired_contact=jnp.asarray(0.0, dtype=jnp.float32),
        control_timestep=0.02,
        done=jnp.asarray(False),
    )

    assert terms["reward/torque"] == pytest.approx(-0.1)


def test_foot_distance_reward_prefers_farther_feet() -> None:
    config = RewardConfig(
        foot_distance=FootDistanceRewardConfig(enabled=True, weight=1.0)
    )
    common = {
        "config": config,
        "command": jnp.zeros(3, dtype=jnp.float32),
        "lin_vel_xy": jnp.zeros(2, dtype=jnp.float32),
        "yaw_vel_z": jnp.asarray(0.0, dtype=jnp.float32),
        "lin_vel_z": jnp.asarray(0.0, dtype=jnp.float32),
        "ang_vel_xy": jnp.zeros(2, dtype=jnp.float32),
        "projected_gravity": jnp.asarray([0.0, 0.0, -1.0], dtype=jnp.float32),
        "base_height": jnp.asarray(config.base_height.target, dtype=jnp.float32),
        "joint_pos": jnp.zeros(2, dtype=jnp.float32),
        "joint_pos_rel": jnp.zeros(2, dtype=jnp.float32),
        "joint_vel": jnp.zeros(2, dtype=jnp.float32),
        "action": jnp.zeros(2, dtype=jnp.float32),
        "last_action": jnp.zeros(2, dtype=jnp.float32),
        "actuator_force": jnp.zeros(2, dtype=jnp.float32),
        "joint_limit_low": -jnp.ones(2, dtype=jnp.float32),
        "joint_limit_high": jnp.ones(2, dtype=jnp.float32),
        "hold_joint_indices": jnp.asarray([], dtype=jnp.int32),
        "policy_joint_indices": jnp.asarray([], dtype=jnp.int32),
        "foot_contact": jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        "foot_height": jnp.zeros(2, dtype=jnp.float32),
        "foot_lateral_speed_sq": jnp.zeros(2, dtype=jnp.float32),
        "gait_contact_mask": jnp.asarray([True, True]),
        "foot_air_time": jnp.zeros(2, dtype=jnp.float32),
        "last_foot_contact": jnp.asarray([True, True]),
        "undesired_contact": jnp.asarray(0.0, dtype=jnp.float32),
        "control_timestep": 0.02,
        "done": jnp.asarray(False),
    }

    near_reward, _ = compute_velocity_reward(
        foot_distance=jnp.asarray(0.2, dtype=jnp.float32), **common
    )
    far_reward, terms = compute_velocity_reward(
        foot_distance=jnp.asarray(0.6, dtype=jnp.float32), **common
    )

    assert far_reward > near_reward
    assert terms["reward/foot_distance"] == pytest.approx(0.012)


def test_policy_joint_deviation_penalizes_policy_pose_error() -> None:
    config = RewardConfig(
        policy_joint_deviation=PolicyJointDeviationRewardConfig(
            enabled=True, weight=-2.0
        )
    )
    common = {
        "config": config,
        "command": jnp.zeros(3, dtype=jnp.float32),
        "lin_vel_xy": jnp.zeros(2, dtype=jnp.float32),
        "yaw_vel_z": jnp.asarray(0.0, dtype=jnp.float32),
        "lin_vel_z": jnp.asarray(0.0, dtype=jnp.float32),
        "ang_vel_xy": jnp.zeros(2, dtype=jnp.float32),
        "projected_gravity": jnp.asarray([0.0, 0.0, -1.0], dtype=jnp.float32),
        "base_height": jnp.asarray(config.base_height.target, dtype=jnp.float32),
        "joint_pos": jnp.zeros(3, dtype=jnp.float32),
        "joint_vel": jnp.zeros(3, dtype=jnp.float32),
        "action": jnp.zeros(3, dtype=jnp.float32),
        "last_action": jnp.zeros(3, dtype=jnp.float32),
        "actuator_force": jnp.zeros(3, dtype=jnp.float32),
        "joint_limit_low": -jnp.ones(3, dtype=jnp.float32),
        "joint_limit_high": jnp.ones(3, dtype=jnp.float32),
        "hold_joint_indices": jnp.asarray([], dtype=jnp.int32),
        "policy_joint_indices": jnp.asarray([0, 1], dtype=jnp.int32),
        "foot_contact": jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        "foot_height": jnp.zeros(2, dtype=jnp.float32),
        "foot_lateral_speed_sq": jnp.zeros(2, dtype=jnp.float32),
        "foot_distance": jnp.asarray(0.3, dtype=jnp.float32),
        "gait_contact_mask": jnp.asarray([True, True]),
        "foot_air_time": jnp.zeros(2, dtype=jnp.float32),
        "last_foot_contact": jnp.asarray([True, True]),
        "undesired_contact": jnp.asarray(0.0, dtype=jnp.float32),
        "control_timestep": 0.02,
        "done": jnp.asarray(False),
    }

    neutral_reward, _ = compute_velocity_reward(
        joint_pos_rel=jnp.zeros(3, dtype=jnp.float32), **common
    )
    deviated_reward, terms = compute_velocity_reward(
        joint_pos_rel=jnp.asarray([0.4, -0.2, 0.0], dtype=jnp.float32), **common
    )

    assert deviated_reward < neutral_reward
    assert terms["reward/policy_joint_deviation"] == pytest.approx(-0.012)


def test_gait_contact_penalizes_mismatched_contact_schedule() -> None:
    config = RewardConfig(
        gait_contact=GaitContactRewardConfig(weight=-1.0),
        foot_contact=FootContactRewardConfig(weight=0.0),
        foot_slip=FootSlipRewardConfig(weight=0.0),
        foot_clearance=FootClearanceRewardConfig(weight=0.0),
        feet_air_time=FeetAirTimeRewardConfig(weight=0.0),
    )

    _, terms = compute_velocity_reward(
        config=config,
        command=jnp.asarray([0.2, 0.0, 0.0], dtype=jnp.float32),
        lin_vel_xy=jnp.zeros(2, dtype=jnp.float32),
        yaw_vel_z=jnp.asarray(0.0, dtype=jnp.float32),
        lin_vel_z=jnp.asarray(0.0, dtype=jnp.float32),
        ang_vel_xy=jnp.zeros(2, dtype=jnp.float32),
        projected_gravity=jnp.asarray([0.0, 0.0, -1.0], dtype=jnp.float32),
        base_height=jnp.asarray(config.base_height.target, dtype=jnp.float32),
        joint_pos=jnp.zeros(2, dtype=jnp.float32),
        joint_pos_rel=jnp.zeros(2, dtype=jnp.float32),
        joint_vel=jnp.zeros(2, dtype=jnp.float32),
        action=jnp.zeros(2, dtype=jnp.float32),
        last_action=jnp.zeros(2, dtype=jnp.float32),
        actuator_force=jnp.zeros(2, dtype=jnp.float32),
        joint_limit_low=-jnp.ones(2, dtype=jnp.float32),
        joint_limit_high=jnp.ones(2, dtype=jnp.float32),
        hold_joint_indices=jnp.asarray([], dtype=jnp.int32),
        policy_joint_indices=jnp.asarray([], dtype=jnp.int32),
        foot_contact=jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        foot_height=jnp.zeros(2, dtype=jnp.float32),
        foot_lateral_speed_sq=jnp.zeros(2, dtype=jnp.float32),
        foot_distance=jnp.asarray(0.3, dtype=jnp.float32),
        gait_contact_mask=jnp.asarray([True, False]),
        foot_air_time=jnp.zeros(2, dtype=jnp.float32),
        last_foot_contact=jnp.asarray([True, True]),
        undesired_contact=jnp.asarray(0.0, dtype=jnp.float32),
        control_timestep=0.02,
        done=jnp.asarray(False),
    )

    assert terms["reward/gait_contact"] == pytest.approx(-0.02)


def test_feet_air_time_rewards_first_contact_after_swing() -> None:
    config = RewardConfig(
        feet_air_time=FeetAirTimeRewardConfig(
            weight=1.0,
            min_air_time=0.2,
            max_air_time=0.6,
            command_speed_threshold=0.05,
        )
    )

    _, terms = compute_velocity_reward(
        config=config,
        command=jnp.asarray([0.2, 0.0, 0.0], dtype=jnp.float32),
        lin_vel_xy=jnp.zeros(2, dtype=jnp.float32),
        yaw_vel_z=jnp.asarray(0.0, dtype=jnp.float32),
        lin_vel_z=jnp.asarray(0.0, dtype=jnp.float32),
        ang_vel_xy=jnp.zeros(2, dtype=jnp.float32),
        projected_gravity=jnp.asarray([0.0, 0.0, -1.0], dtype=jnp.float32),
        base_height=jnp.asarray(config.base_height.target, dtype=jnp.float32),
        joint_pos=jnp.zeros(2, dtype=jnp.float32),
        joint_pos_rel=jnp.zeros(2, dtype=jnp.float32),
        joint_vel=jnp.zeros(2, dtype=jnp.float32),
        action=jnp.zeros(2, dtype=jnp.float32),
        last_action=jnp.zeros(2, dtype=jnp.float32),
        actuator_force=jnp.zeros(2, dtype=jnp.float32),
        joint_limit_low=-jnp.ones(2, dtype=jnp.float32),
        joint_limit_high=jnp.ones(2, dtype=jnp.float32),
        hold_joint_indices=jnp.asarray([], dtype=jnp.int32),
        policy_joint_indices=jnp.asarray([], dtype=jnp.int32),
        foot_contact=jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        foot_height=jnp.zeros(2, dtype=jnp.float32),
        foot_lateral_speed_sq=jnp.zeros(2, dtype=jnp.float32),
        foot_distance=jnp.asarray(0.3, dtype=jnp.float32),
        gait_contact_mask=jnp.asarray([True, False]),
        foot_air_time=jnp.asarray([0.28, 0.0], dtype=jnp.float32),
        last_foot_contact=jnp.asarray([False, False]),
        undesired_contact=jnp.asarray(0.0, dtype=jnp.float32),
        control_timestep=0.02,
        done=jnp.asarray(False),
    )

    assert terms["reward/feet_air_time"] == pytest.approx(0.05)
