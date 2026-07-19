import jax.numpy as jnp
import pytest

from bello_mujoco.envs.rewards import (
    ActionLimitRewardConfig,
    CommandConfig,
    ContactTimingRewardConfig,
    FeetAirTimeRewardConfig,
    FootClearanceRewardConfig,
    FootContactRewardConfig,
    FootDistanceRewardConfig,
    FootSlipRewardConfig,
    GaitContactRewardConfig,
    InvalidActionRewardConfig,
    JointAccRewardConfig,
    JointLimitRewardConfig,
    PolicyJointDeviationRewardConfig,
    RewardConfig,
    SelfCollisionRewardConfig,
    StanceWidthRewardConfig,
    StandingFootAirTimeRewardConfig,
    StandingFootVelRewardConfig,
    SwingFootVelocityRewardConfig,
    TerminationRewardConfig,
    TorqueRewardConfig,
    TrackLinVelXYRewardConfig,
    UndesiredContactRewardConfig,
    compute_velocity_reward,
)


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
        "last_joint_vel": jnp.zeros(4, dtype=jnp.float32),
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


def test_reward_config_enabled_links_include_tracking_frame_link() -> None:
    config = RewardConfig(
        track_lin_vel_xy=TrackLinVelXYRewardConfig(
            link="pelvis_link", frame_link="torso_link"
        )
    )

    assert "pelvis_link" in config.enabled_link_names
    assert "torso_link" in config.enabled_link_names


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
        "last_joint_vel": jnp.zeros(2, dtype=jnp.float32),
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
    assert terms["reward/alive"] == 0.0
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
        "last_joint_vel": jnp.zeros(2, dtype=jnp.float32),
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
    assert terms["reward/foot_slip"] == pytest.approx(-0.014142135)
    assert terms["reward/undesired_contact"] == pytest.approx(-0.04)


def test_self_collision_penalizes_robot_robot_contacts() -> None:
    config = RewardConfig(
        self_collision=SelfCollisionRewardConfig(weight=-2.0),
        foot_contact=FootContactRewardConfig(weight=0.0),
        foot_clearance=FootClearanceRewardConfig(weight=0.0),
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
        last_joint_vel=jnp.zeros(2, dtype=jnp.float32),
        action=jnp.zeros(2, dtype=jnp.float32),
        last_action=jnp.zeros(2, dtype=jnp.float32),
        actuator_force=jnp.zeros(2, dtype=jnp.float32),
        joint_limit_low=-jnp.ones(2, dtype=jnp.float32),
        joint_limit_high=jnp.ones(2, dtype=jnp.float32),
        hold_joint_indices=jnp.asarray([], dtype=jnp.int32),
        policy_joint_indices=jnp.asarray([0, 1], dtype=jnp.int32),
        foot_contact=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        foot_height=jnp.zeros(2, dtype=jnp.float32),
        foot_lateral_speed_sq=jnp.zeros(2, dtype=jnp.float32),
        foot_distance=jnp.asarray(0.3, dtype=jnp.float32),
        gait_contact_mask=jnp.asarray([False, False]),
        foot_air_time=jnp.zeros(2, dtype=jnp.float32),
        last_foot_contact=jnp.asarray([False, False]),
        undesired_contact=jnp.asarray(0.0, dtype=jnp.float32),
        self_collision=jnp.asarray(3.0, dtype=jnp.float32),
        control_timestep=0.02,
        done=jnp.asarray(False),
    )

    assert terms["reward/self_collision"] == pytest.approx(-0.12)


def test_foot_slip_penalizes_contacted_foot_rotation() -> None:
    config = RewardConfig(
        foot_slip=FootSlipRewardConfig(weight=-1.0, angular_speed_scale=0.05),
        foot_contact=FootContactRewardConfig(weight=0.0),
        foot_clearance=FootClearanceRewardConfig(weight=0.0),
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
        last_joint_vel=jnp.zeros(2, dtype=jnp.float32),
        action=jnp.zeros(2, dtype=jnp.float32),
        last_action=jnp.zeros(2, dtype=jnp.float32),
        actuator_force=jnp.zeros(2, dtype=jnp.float32),
        joint_limit_low=-jnp.ones(2, dtype=jnp.float32),
        joint_limit_high=jnp.ones(2, dtype=jnp.float32),
        hold_joint_indices=jnp.asarray([], dtype=jnp.int32),
        policy_joint_indices=jnp.asarray([0, 1], dtype=jnp.int32),
        foot_contact=jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        foot_height=jnp.zeros(2, dtype=jnp.float32),
        foot_lateral_speed_sq=jnp.zeros(2, dtype=jnp.float32),
        foot_angular_speed=jnp.asarray([4.0, 4.0], dtype=jnp.float32),
        foot_distance=jnp.asarray(0.3, dtype=jnp.float32),
        gait_contact_mask=jnp.asarray([True, False]),
        foot_air_time=jnp.zeros(2, dtype=jnp.float32),
        last_foot_contact=jnp.asarray([True, True]),
        undesired_contact=jnp.asarray(0.0, dtype=jnp.float32),
        control_timestep=0.02,
        done=jnp.asarray(False),
    )

    assert terms["reward/foot_slip"] == pytest.approx(-0.004)


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
        last_joint_vel=jnp.zeros(2, dtype=jnp.float32),
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
        last_joint_vel=jnp.zeros(2, dtype=jnp.float32),
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

    assert terms["reward/torque"] == pytest.approx(-0.2)


def test_joint_limit_penalty_uses_softened_hard_range() -> None:
    config = RewardConfig(
        joint_limit=JointLimitRewardConfig(
            enabled=True, weight=-1.0, soft_limit_factor=0.9
        )
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
        joint_pos=jnp.asarray([-1.5, 1.25, 0.0], dtype=jnp.float32),
        joint_pos_rel=jnp.zeros(3, dtype=jnp.float32),
        joint_vel=jnp.zeros(3, dtype=jnp.float32),
        last_joint_vel=jnp.zeros(3, dtype=jnp.float32),
        action=jnp.zeros(3, dtype=jnp.float32),
        last_action=jnp.zeros(3, dtype=jnp.float32),
        actuator_force=jnp.zeros(3, dtype=jnp.float32),
        joint_limit_low=-jnp.ones(3, dtype=jnp.float32),
        joint_limit_high=jnp.ones(3, dtype=jnp.float32),
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

    assert terms["reward/joint_limit"] == pytest.approx(-0.019)


def test_action_limit_penalty_is_zero_below_margin_and_quadratic_above() -> None:
    config = RewardConfig(
        action_limit=ActionLimitRewardConfig(
            enabled=True, weight=-1.0, soft_limit_ratio=0.8
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
        "joint_pos": jnp.zeros(2, dtype=jnp.float32),
        "joint_pos_rel": jnp.zeros(2, dtype=jnp.float32),
        "joint_vel": jnp.zeros(2, dtype=jnp.float32),
        "last_joint_vel": jnp.zeros(2, dtype=jnp.float32),
        "last_action": jnp.zeros(2, dtype=jnp.float32),
        "actuator_force": jnp.zeros(2, dtype=jnp.float32),
        "joint_limit_low": -jnp.ones(2, dtype=jnp.float32),
        "joint_limit_high": jnp.ones(2, dtype=jnp.float32),
        "hold_joint_indices": jnp.asarray([], dtype=jnp.int32),
        "policy_joint_indices": jnp.asarray([], dtype=jnp.int32),
        "foot_contact": jnp.ones(2, dtype=jnp.float32),
        "foot_height": jnp.zeros(2, dtype=jnp.float32),
        "foot_lateral_speed_sq": jnp.zeros(2, dtype=jnp.float32),
        "foot_distance": jnp.asarray(0.3, dtype=jnp.float32),
        "gait_contact_mask": jnp.asarray([True, True]),
        "foot_air_time": jnp.zeros(2, dtype=jnp.float32),
        "last_foot_contact": jnp.asarray([True, True]),
        "undesired_contact": jnp.asarray(0.0, dtype=jnp.float32),
        "control_timestep": 0.02,
        "done": jnp.asarray(False),
        "action_clip_limit": 1.0,
    }

    _, inside_terms = compute_velocity_reward(
        action=jnp.asarray([0.8, -0.5], dtype=jnp.float32), **common
    )
    _, boundary_terms = compute_velocity_reward(
        action=jnp.asarray([1.0, -0.8], dtype=jnp.float32), **common
    )

    assert inside_terms["reward/action_limit"] == pytest.approx(0.0)
    assert boundary_terms["reward/action_limit"] == pytest.approx(-0.01)


@pytest.mark.parametrize("soft_limit_factor", [0.0, 1.01, float("nan")])
def test_joint_limit_rejects_invalid_soft_factor(soft_limit_factor: float) -> None:
    with pytest.raises(ValueError, match="joint_limit.soft_limit_factor"):
        RewardConfig(
            joint_limit=JointLimitRewardConfig(
                enabled=True, soft_limit_factor=soft_limit_factor
            )
        )


@pytest.mark.parametrize("soft_limit_ratio", [-0.01, 1.0, float("nan")])
def test_action_limit_rejects_invalid_soft_ratio(soft_limit_ratio: float) -> None:
    with pytest.raises(ValueError, match="action_limit.soft_limit_ratio"):
        RewardConfig(
            action_limit=ActionLimitRewardConfig(
                enabled=True, soft_limit_ratio=soft_limit_ratio
            )
        )


def test_invalid_action_penalty_uses_normalized_squared_target_violation() -> None:
    config = RewardConfig(
        invalid_action=InvalidActionRewardConfig(enabled=True, weight=-2.0)
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
        last_joint_vel=jnp.zeros(2, dtype=jnp.float32),
        action=jnp.zeros(2, dtype=jnp.float32),
        requested_action=jnp.asarray([-4.0, 5.0], dtype=jnp.float32),
        last_action=jnp.zeros(2, dtype=jnp.float32),
        actuator_force=jnp.zeros(2, dtype=jnp.float32),
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
        action_target_default=jnp.zeros(2, dtype=jnp.float32),
        action_target_scale=jnp.asarray([0.25, 0.5], dtype=jnp.float32),
        action_target_low=jnp.asarray([-0.5, -1.0], dtype=jnp.float32),
        action_target_high=jnp.asarray([0.5, 1.0], dtype=jnp.float32),
    )

    assert terms["reward/invalid_action"] == pytest.approx(-0.52)


def test_joint_acc_reward_penalizes_joint_velocity_changes() -> None:
    config = RewardConfig(joint_acc=JointAccRewardConfig(enabled=True, weight=-1.0))

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
        joint_vel=jnp.asarray([1.0, -1.0], dtype=jnp.float32),
        last_joint_vel=jnp.zeros(2, dtype=jnp.float32),
        action=jnp.zeros(2, dtype=jnp.float32),
        last_action=jnp.zeros(2, dtype=jnp.float32),
        actuator_force=jnp.zeros(2, dtype=jnp.float32),
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

    assert terms["reward/joint_acc"] == pytest.approx(-100.0)


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
        "last_joint_vel": jnp.zeros(2, dtype=jnp.float32),
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

    stance_config = RewardConfig(
        foot_distance=FootDistanceRewardConfig(enabled=False),
        stance_width=StanceWidthRewardConfig(
            enabled=True,
            weight=-2.0,
            min_width=0.12,
            max_width=0.24,
        ),
    )
    in_band_reward, in_band_terms = compute_velocity_reward(
        config=stance_config,
        stance_width=jnp.asarray(0.18, dtype=jnp.float32),
        foot_distance=jnp.asarray(0.3, dtype=jnp.float32),
        **{key: value for key, value in common.items() if key != "config"},
    )
    wide_reward, wide_terms = compute_velocity_reward(
        config=stance_config,
        stance_width=jnp.asarray(0.44, dtype=jnp.float32),
        foot_distance=jnp.asarray(0.3, dtype=jnp.float32),
        **{key: value for key, value in common.items() if key != "config"},
    )

    assert in_band_reward > wide_reward
    assert in_band_terms["reward/stance_width"] == pytest.approx(0.0)
    assert wide_terms["reward/stance_width"] == pytest.approx(-0.0016)


def test_stance_width_reward_validates_band() -> None:
    with pytest.raises(ValueError, match="stance_width.min_width"):
        RewardConfig(
            stance_width=StanceWidthRewardConfig(
                enabled=True, min_width=-0.1, max_width=0.2
            )
        )
    with pytest.raises(ValueError, match="stance_width.max_width"):
        RewardConfig(
            stance_width=StanceWidthRewardConfig(
                enabled=True, min_width=0.2, max_width=0.2
            )
        )


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
        "last_joint_vel": jnp.zeros(3, dtype=jnp.float32),
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
    assert terms["reward/policy_joint_deviation"] == pytest.approx(-0.024)


def test_foot_clearance_rewards_moving_feet_near_target() -> None:
    config = RewardConfig(
        foot_clearance=FootClearanceRewardConfig(
            weight=1.0, target=0.10, std=0.05, tanh_mult=2.0
        ),
        gait_contact=GaitContactRewardConfig(enabled=False),
        foot_contact=FootContactRewardConfig(weight=0.0),
        foot_slip=FootSlipRewardConfig(weight=0.0),
        feet_air_time=FeetAirTimeRewardConfig(enabled=False),
    )
    common = {
        "config": config,
        "command": jnp.asarray([0.2, 0.0, 0.0], dtype=jnp.float32),
        "lin_vel_xy": jnp.zeros(2, dtype=jnp.float32),
        "yaw_vel_z": jnp.asarray(0.0, dtype=jnp.float32),
        "lin_vel_z": jnp.asarray(0.0, dtype=jnp.float32),
        "ang_vel_xy": jnp.zeros(2, dtype=jnp.float32),
        "projected_gravity": jnp.asarray([0.0, 0.0, -1.0], dtype=jnp.float32),
        "base_height": jnp.asarray(config.base_height.target, dtype=jnp.float32),
        "joint_pos": jnp.zeros(2, dtype=jnp.float32),
        "joint_pos_rel": jnp.zeros(2, dtype=jnp.float32),
        "joint_vel": jnp.zeros(2, dtype=jnp.float32),
        "last_joint_vel": jnp.zeros(2, dtype=jnp.float32),
        "action": jnp.zeros(2, dtype=jnp.float32),
        "last_action": jnp.zeros(2, dtype=jnp.float32),
        "actuator_force": jnp.zeros(2, dtype=jnp.float32),
        "joint_limit_low": -jnp.ones(2, dtype=jnp.float32),
        "joint_limit_high": jnp.ones(2, dtype=jnp.float32),
        "hold_joint_indices": jnp.asarray([], dtype=jnp.int32),
        "policy_joint_indices": jnp.asarray([], dtype=jnp.int32),
        "foot_contact": jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        "foot_lateral_speed_sq": jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        "foot_distance": jnp.asarray(0.3, dtype=jnp.float32),
        "gait_contact_mask": jnp.asarray([False, False]),
        "foot_air_time": jnp.zeros(2, dtype=jnp.float32),
        "last_foot_contact": jnp.asarray([False, False]),
        "undesired_contact": jnp.asarray(0.0, dtype=jnp.float32),
        "control_timestep": 0.02,
        "done": jnp.asarray(False),
    }

    _, target_terms = compute_velocity_reward(
        foot_height=jnp.asarray([0.10, 0.10], dtype=jnp.float32), **common
    )
    _, low_terms = compute_velocity_reward(
        foot_height=jnp.asarray([0.0, 0.0], dtype=jnp.float32), **common
    )

    assert target_terms["reward/foot_clearance"] > low_terms["reward/foot_clearance"]
    assert target_terms["reward/foot_clearance"] == pytest.approx(0.02)


def test_foot_clearance_ignores_contacted_feet() -> None:
    config = RewardConfig(
        foot_clearance=FootClearanceRewardConfig(
            weight=1.0,
            target=0.10,
            std=0.05,
            tanh_mult=2.0,
        ),
        gait_contact=GaitContactRewardConfig(enabled=False),
        foot_contact=FootContactRewardConfig(weight=0.0),
        foot_slip=FootSlipRewardConfig(weight=0.0),
        feet_air_time=FeetAirTimeRewardConfig(enabled=False),
    )
    common = {
        "config": config,
        "command": jnp.asarray([0.2, 0.0, 0.0], dtype=jnp.float32),
        "lin_vel_xy": jnp.zeros(2, dtype=jnp.float32),
        "yaw_vel_z": jnp.asarray(0.0, dtype=jnp.float32),
        "lin_vel_z": jnp.asarray(0.0, dtype=jnp.float32),
        "ang_vel_xy": jnp.zeros(2, dtype=jnp.float32),
        "projected_gravity": jnp.asarray([0.0, 0.0, -1.0], dtype=jnp.float32),
        "base_height": jnp.asarray(config.base_height.target, dtype=jnp.float32),
        "joint_pos": jnp.zeros(2, dtype=jnp.float32),
        "joint_pos_rel": jnp.zeros(2, dtype=jnp.float32),
        "joint_vel": jnp.zeros(2, dtype=jnp.float32),
        "last_joint_vel": jnp.zeros(2, dtype=jnp.float32),
        "action": jnp.zeros(2, dtype=jnp.float32),
        "last_action": jnp.zeros(2, dtype=jnp.float32),
        "actuator_force": jnp.zeros(2, dtype=jnp.float32),
        "joint_limit_low": -jnp.ones(2, dtype=jnp.float32),
        "joint_limit_high": jnp.ones(2, dtype=jnp.float32),
        "hold_joint_indices": jnp.asarray([], dtype=jnp.int32),
        "policy_joint_indices": jnp.asarray([], dtype=jnp.int32),
        "foot_height": jnp.asarray([0.10, 0.10], dtype=jnp.float32),
        "foot_lateral_speed_sq": jnp.ones(2, dtype=jnp.float32),
        "foot_distance": jnp.asarray(0.3, dtype=jnp.float32),
        "gait_contact_mask": jnp.asarray([False, False]),
        "foot_air_time": jnp.zeros(2, dtype=jnp.float32),
        "last_foot_contact": jnp.asarray([False, False]),
        "undesired_contact": jnp.asarray(0.0, dtype=jnp.float32),
        "control_timestep": 0.02,
        "done": jnp.asarray(False),
    }

    _, contacted_terms = compute_velocity_reward(
        foot_contact=jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        **common,
    )
    _, swing_terms = compute_velocity_reward(
        foot_contact=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        **common,
    )

    assert contacted_terms["reward/foot_clearance"] == pytest.approx(0.0)
    assert swing_terms["reward/foot_clearance"] == pytest.approx(0.02)


def test_foot_clearance_ignores_standing_commands() -> None:
    config = RewardConfig(
        foot_clearance=FootClearanceRewardConfig(
            weight=1.0,
            target=0.10,
            std=0.05,
            tanh_mult=2.0,
            command_speed_threshold=0.05,
        ),
        gait_contact=GaitContactRewardConfig(enabled=False),
        foot_contact=FootContactRewardConfig(weight=0.0),
        foot_slip=FootSlipRewardConfig(weight=0.0),
        feet_air_time=FeetAirTimeRewardConfig(enabled=False),
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
        last_joint_vel=jnp.zeros(2, dtype=jnp.float32),
        action=jnp.zeros(2, dtype=jnp.float32),
        last_action=jnp.zeros(2, dtype=jnp.float32),
        actuator_force=jnp.zeros(2, dtype=jnp.float32),
        joint_limit_low=-jnp.ones(2, dtype=jnp.float32),
        joint_limit_high=jnp.ones(2, dtype=jnp.float32),
        hold_joint_indices=jnp.asarray([], dtype=jnp.int32),
        policy_joint_indices=jnp.asarray([], dtype=jnp.int32),
        foot_contact=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        foot_height=jnp.asarray([0.10, 0.10], dtype=jnp.float32),
        foot_lateral_speed_sq=jnp.ones(2, dtype=jnp.float32),
        foot_distance=jnp.asarray(0.3, dtype=jnp.float32),
        gait_contact_mask=jnp.asarray([False, False]),
        foot_air_time=jnp.zeros(2, dtype=jnp.float32),
        last_foot_contact=jnp.asarray([False, False]),
        undesired_contact=jnp.asarray(0.0, dtype=jnp.float32),
        control_timestep=0.02,
        done=jnp.asarray(False),
    )

    assert terms["reward/foot_clearance"] == pytest.approx(0.0)


def test_gait_contact_rewards_matching_commanded_contact_schedule() -> None:
    config = RewardConfig(
        gait_contact=GaitContactRewardConfig(weight=1.0, command_speed_threshold=0.10),
        foot_contact=FootContactRewardConfig(weight=0.0),
        foot_slip=FootSlipRewardConfig(weight=0.0),
        foot_clearance=FootClearanceRewardConfig(enabled=False),
        feet_air_time=FeetAirTimeRewardConfig(enabled=False),
    )

    common = {
        "config": config,
        "command": jnp.asarray([0.2, 0.0, 0.0], dtype=jnp.float32),
        "lin_vel_xy": jnp.zeros(2, dtype=jnp.float32),
        "yaw_vel_z": jnp.asarray(0.0, dtype=jnp.float32),
        "lin_vel_z": jnp.asarray(0.0, dtype=jnp.float32),
        "ang_vel_xy": jnp.zeros(2, dtype=jnp.float32),
        "projected_gravity": jnp.asarray([0.0, 0.0, -1.0], dtype=jnp.float32),
        "base_height": jnp.asarray(config.base_height.target, dtype=jnp.float32),
        "joint_pos": jnp.zeros(2, dtype=jnp.float32),
        "joint_pos_rel": jnp.zeros(2, dtype=jnp.float32),
        "joint_vel": jnp.zeros(2, dtype=jnp.float32),
        "last_joint_vel": jnp.zeros(2, dtype=jnp.float32),
        "action": jnp.zeros(2, dtype=jnp.float32),
        "last_action": jnp.zeros(2, dtype=jnp.float32),
        "actuator_force": jnp.zeros(2, dtype=jnp.float32),
        "joint_limit_low": -jnp.ones(2, dtype=jnp.float32),
        "joint_limit_high": jnp.ones(2, dtype=jnp.float32),
        "hold_joint_indices": jnp.asarray([], dtype=jnp.int32),
        "policy_joint_indices": jnp.asarray([], dtype=jnp.int32),
        "foot_height": jnp.zeros(2, dtype=jnp.float32),
        "foot_lateral_speed_sq": jnp.zeros(2, dtype=jnp.float32),
        "foot_distance": jnp.asarray(0.3, dtype=jnp.float32),
        "gait_contact_mask": jnp.asarray([True, False]),
        "foot_air_time": jnp.zeros(2, dtype=jnp.float32),
        "last_foot_contact": jnp.asarray([True, True]),
        "undesired_contact": jnp.asarray(0.0, dtype=jnp.float32),
        "control_timestep": 0.02,
        "done": jnp.asarray(False),
    }

    _, matched_terms = compute_velocity_reward(
        foot_contact=jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        **common,
    )
    _, mismatched_terms = compute_velocity_reward(
        foot_contact=jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        **common,
    )
    _, standing_terms = compute_velocity_reward(
        command=jnp.zeros(3, dtype=jnp.float32),
        foot_contact=jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        **{key: value for key, value in common.items() if key != "command"},
    )

    assert matched_terms["reward/gait_contact"] == pytest.approx(0.04)
    assert mismatched_terms["reward/gait_contact"] == 0.0
    assert standing_terms["reward/gait_contact"] == 0.0


def test_contact_timing_penalizes_only_premature_moving_transitions() -> None:
    config = RewardConfig(
        contact_timing=ContactTimingRewardConfig(
            enabled=True,
            weight=-1.0,
            min_air_time=0.20,
            min_contact_time=0.30,
            command_speed_threshold=0.10,
        )
    )
    common = {
        "config": config,
        "lin_vel_xy": jnp.zeros(2, dtype=jnp.float32),
        "yaw_vel_z": jnp.asarray(0.0, dtype=jnp.float32),
        "lin_vel_z": jnp.asarray(0.0, dtype=jnp.float32),
        "ang_vel_xy": jnp.zeros(2, dtype=jnp.float32),
        "projected_gravity": jnp.asarray([0.0, 0.0, -1.0], dtype=jnp.float32),
        "base_height": jnp.asarray(config.base_height.target, dtype=jnp.float32),
        "joint_pos": jnp.zeros(2, dtype=jnp.float32),
        "joint_pos_rel": jnp.zeros(2, dtype=jnp.float32),
        "joint_vel": jnp.zeros(2, dtype=jnp.float32),
        "last_joint_vel": jnp.zeros(2, dtype=jnp.float32),
        "action": jnp.zeros(2, dtype=jnp.float32),
        "last_action": jnp.zeros(2, dtype=jnp.float32),
        "actuator_force": jnp.zeros(2, dtype=jnp.float32),
        "joint_limit_low": -jnp.ones(2, dtype=jnp.float32),
        "joint_limit_high": jnp.ones(2, dtype=jnp.float32),
        "hold_joint_indices": jnp.asarray([], dtype=jnp.int32),
        "policy_joint_indices": jnp.asarray([], dtype=jnp.int32),
        "foot_height": jnp.zeros(2, dtype=jnp.float32),
        "foot_lateral_speed_sq": jnp.zeros(2, dtype=jnp.float32),
        "foot_distance": jnp.asarray(0.3, dtype=jnp.float32),
        "gait_contact_mask": jnp.asarray([True, False]),
        "last_foot_contact": jnp.asarray([False, True]),
        "undesired_contact": jnp.asarray(0.0, dtype=jnp.float32),
        "control_timestep": 0.02,
        "done": jnp.asarray(False),
    }
    transitions = {
        "foot_contact": jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        "foot_air_time": jnp.asarray([0.10, 0.0], dtype=jnp.float32),
        "foot_contact_time": jnp.asarray([0.0, 0.15], dtype=jnp.float32),
    }

    _, premature_terms = compute_velocity_reward(
        command=jnp.asarray([0.2, 0.0, 0.0], dtype=jnp.float32),
        **transitions,
        **common,
    )
    _, mature_terms = compute_velocity_reward(
        command=jnp.asarray([0.2, 0.0, 0.0], dtype=jnp.float32),
        **(
            transitions
            | {
                "foot_air_time": jnp.asarray([0.20, 0.0], dtype=jnp.float32),
                "foot_contact_time": jnp.asarray([0.0, 0.30], dtype=jnp.float32),
            }
        ),
        **common,
    )
    _, standing_terms = compute_velocity_reward(
        command=jnp.zeros(3, dtype=jnp.float32),
        **transitions,
        **common,
    )

    assert premature_terms["reward/contact_timing"] == pytest.approx(-0.5)
    assert mature_terms["reward/contact_timing"] == pytest.approx(0.0)
    assert standing_terms["reward/contact_timing"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_air_time", 0.0),
        ("min_contact_time", 0.0),
        ("command_speed_threshold", -0.1),
    ],
)
def test_contact_timing_config_validates_durations_and_gate(field, value) -> None:
    kwargs = {field: value}
    with pytest.raises(ValueError, match=f"contact_timing.{field}"):
        RewardConfig(contact_timing=ContactTimingRewardConfig(enabled=True, **kwargs))


def test_swing_foot_velocity_tracks_capped_directional_target() -> None:
    config = RewardConfig(
        swing_foot_velocity=SwingFootVelocityRewardConfig(
            enabled=True,
            weight=1.0,
            std=0.25,
            max_target_speed=0.5,
            command_speed_threshold=0.1,
        )
    )
    common = {
        "config": config,
        "lin_vel_xy": jnp.zeros(2, dtype=jnp.float32),
        "yaw_vel_z": jnp.asarray(0.0, dtype=jnp.float32),
        "lin_vel_z": jnp.asarray(0.0, dtype=jnp.float32),
        "ang_vel_xy": jnp.zeros(2, dtype=jnp.float32),
        "projected_gravity": jnp.asarray([0.0, 0.0, -1.0], dtype=jnp.float32),
        "base_height": jnp.asarray(config.base_height.target, dtype=jnp.float32),
        "joint_pos": jnp.zeros(2, dtype=jnp.float32),
        "joint_pos_rel": jnp.zeros(2, dtype=jnp.float32),
        "joint_vel": jnp.zeros(2, dtype=jnp.float32),
        "last_joint_vel": jnp.zeros(2, dtype=jnp.float32),
        "action": jnp.zeros(2, dtype=jnp.float32),
        "last_action": jnp.zeros(2, dtype=jnp.float32),
        "actuator_force": jnp.zeros(2, dtype=jnp.float32),
        "joint_limit_low": -jnp.ones(2, dtype=jnp.float32),
        "joint_limit_high": jnp.ones(2, dtype=jnp.float32),
        "hold_joint_indices": jnp.asarray([], dtype=jnp.int32),
        "policy_joint_indices": jnp.asarray([], dtype=jnp.int32),
        "foot_contact": jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        "foot_height": jnp.zeros(2, dtype=jnp.float32),
        "foot_lateral_speed_sq": jnp.zeros(2, dtype=jnp.float32),
        "foot_distance": jnp.asarray(0.3, dtype=jnp.float32),
        "gait_contact_mask": jnp.asarray([True, False]),
        "foot_air_time": jnp.zeros(2, dtype=jnp.float32),
        "last_foot_contact": jnp.asarray([True, False]),
        "undesired_contact": jnp.asarray(0.0, dtype=jnp.float32),
        "control_timestep": 0.02,
        "done": jnp.asarray(False),
        "swing_velocity_target_scale": 2.0,
    }
    command = jnp.asarray([0.0, 0.5, 0.0], dtype=jnp.float32)

    _, matched_terms = compute_velocity_reward(
        command=command,
        foot_relative_lin_vel_xy=jnp.asarray(
            [[10.0, 10.0], [0.0, 0.5]], dtype=jnp.float32
        ),
        **common,
    )
    _, reversed_terms = compute_velocity_reward(
        command=command,
        foot_relative_lin_vel_xy=jnp.asarray(
            [[0.0, 0.5], [0.0, -0.5]], dtype=jnp.float32
        ),
        **common,
    )
    _, standing_terms = compute_velocity_reward(
        command=jnp.zeros(3, dtype=jnp.float32),
        foot_relative_lin_vel_xy=jnp.zeros((2, 2), dtype=jnp.float32),
        **common,
    )

    assert matched_terms["reward/swing_foot_velocity"] == pytest.approx(0.02)
    assert reversed_terms["reward/swing_foot_velocity"] < 0.001
    assert standing_terms["reward/swing_foot_velocity"] == pytest.approx(0.0)


def test_swing_foot_velocity_config_validates_std() -> None:
    with pytest.raises(ValueError, match="swing_foot_velocity.std"):
        RewardConfig(
            swing_foot_velocity=SwingFootVelocityRewardConfig(enabled=True, std=0.0)
        )


def test_feet_air_time_rewards_capped_biped_single_stance() -> None:
    config = RewardConfig(
        feet_air_time=FeetAirTimeRewardConfig(
            weight=1.0,
            max_duration=0.4,
            command_speed_threshold=0.1,
        )
    )

    common = dict(
        config=config,
        lin_vel_xy=jnp.zeros(2, dtype=jnp.float32),
        yaw_vel_z=jnp.asarray(0.0, dtype=jnp.float32),
        lin_vel_z=jnp.asarray(0.0, dtype=jnp.float32),
        ang_vel_xy=jnp.zeros(2, dtype=jnp.float32),
        projected_gravity=jnp.asarray([0.0, 0.0, -1.0], dtype=jnp.float32),
        base_height=jnp.asarray(config.base_height.target, dtype=jnp.float32),
        joint_pos=jnp.zeros(2, dtype=jnp.float32),
        joint_pos_rel=jnp.zeros(2, dtype=jnp.float32),
        joint_vel=jnp.zeros(2, dtype=jnp.float32),
        last_joint_vel=jnp.zeros(2, dtype=jnp.float32),
        action=jnp.zeros(2, dtype=jnp.float32),
        last_action=jnp.zeros(2, dtype=jnp.float32),
        actuator_force=jnp.zeros(2, dtype=jnp.float32),
        joint_limit_low=-jnp.ones(2, dtype=jnp.float32),
        joint_limit_high=jnp.ones(2, dtype=jnp.float32),
        hold_joint_indices=jnp.asarray([], dtype=jnp.int32),
        policy_joint_indices=jnp.asarray([], dtype=jnp.int32),
        foot_height=jnp.zeros(2, dtype=jnp.float32),
        foot_lateral_speed_sq=jnp.zeros(2, dtype=jnp.float32),
        foot_distance=jnp.asarray(0.3, dtype=jnp.float32),
        gait_contact_mask=jnp.asarray([True, False]),
        foot_air_time=jnp.asarray([0.0, 0.28], dtype=jnp.float32),
        foot_contact_time=jnp.asarray([0.18, 0.0], dtype=jnp.float32),
        last_foot_contact=jnp.asarray([True, False]),
        undesired_contact=jnp.asarray(0.0, dtype=jnp.float32),
        control_timestep=0.02,
        done=jnp.asarray(False),
    )

    _, single_stance_terms = compute_velocity_reward(
        command=jnp.asarray([0.2, 0.0, 0.0], dtype=jnp.float32),
        foot_contact=jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        **common,
    )
    _, double_stance_terms = compute_velocity_reward(
        command=jnp.asarray([0.2, 0.0, 0.0], dtype=jnp.float32),
        foot_contact=jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        **common,
    )
    _, standing_terms = compute_velocity_reward(
        command=jnp.zeros(3, dtype=jnp.float32),
        foot_contact=jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        **common,
    )
    _, capped_terms = compute_velocity_reward(
        command=jnp.asarray([0.2, 0.0, 0.0], dtype=jnp.float32),
        foot_contact=jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        **(
            common
            | {
                "foot_air_time": jnp.asarray([0.0, 0.58], dtype=jnp.float32),
                "foot_contact_time": jnp.asarray([0.48, 0.0], dtype=jnp.float32),
            }
        ),
    )

    assert single_stance_terms["reward/feet_air_time"] == pytest.approx(0.004)
    assert double_stance_terms["reward/feet_air_time"] == pytest.approx(0.0)
    assert standing_terms["reward/feet_air_time"] == pytest.approx(0.0)
    assert capped_terms["reward/feet_air_time"] == pytest.approx(0.008)


def test_feet_air_time_validates_max_duration() -> None:
    with pytest.raises(ValueError, match="feet_air_time.max_duration"):
        RewardConfig(
            feet_air_time=FeetAirTimeRewardConfig(enabled=True, max_duration=0.0)
        )


def test_standing_foot_penalties_apply_only_to_low_commands() -> None:
    config = RewardConfig(
        standing_foot_vel=StandingFootVelRewardConfig(
            enabled=True, weight=-2.0, command_speed_threshold=0.05
        ),
        standing_foot_air_time=StandingFootAirTimeRewardConfig(
            enabled=True, weight=-3.0, command_speed_threshold=0.05
        ),
        foot_contact=FootContactRewardConfig(weight=0.0),
        foot_slip=FootSlipRewardConfig(weight=0.0),
        foot_clearance=FootClearanceRewardConfig(enabled=False),
        gait_contact=GaitContactRewardConfig(enabled=False),
        feet_air_time=FeetAirTimeRewardConfig(enabled=False),
    )
    common = {
        "config": config,
        "lin_vel_xy": jnp.zeros(2, dtype=jnp.float32),
        "yaw_vel_z": jnp.asarray(0.0, dtype=jnp.float32),
        "lin_vel_z": jnp.asarray(0.0, dtype=jnp.float32),
        "ang_vel_xy": jnp.zeros(2, dtype=jnp.float32),
        "projected_gravity": jnp.asarray([0.0, 0.0, -1.0], dtype=jnp.float32),
        "base_height": jnp.asarray(config.base_height.target, dtype=jnp.float32),
        "joint_pos": jnp.zeros(2, dtype=jnp.float32),
        "joint_pos_rel": jnp.zeros(2, dtype=jnp.float32),
        "joint_vel": jnp.zeros(2, dtype=jnp.float32),
        "last_joint_vel": jnp.zeros(2, dtype=jnp.float32),
        "action": jnp.zeros(2, dtype=jnp.float32),
        "last_action": jnp.zeros(2, dtype=jnp.float32),
        "actuator_force": jnp.zeros(2, dtype=jnp.float32),
        "joint_limit_low": -jnp.ones(2, dtype=jnp.float32),
        "joint_limit_high": jnp.ones(2, dtype=jnp.float32),
        "hold_joint_indices": jnp.asarray([], dtype=jnp.int32),
        "policy_joint_indices": jnp.asarray([], dtype=jnp.int32),
        "foot_contact": jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        "foot_height": jnp.zeros(2, dtype=jnp.float32),
        "foot_lateral_speed_sq": jnp.asarray([0.25, 1.0], dtype=jnp.float32),
        "foot_distance": jnp.asarray(0.3, dtype=jnp.float32),
        "gait_contact_mask": jnp.asarray([True, True]),
        "foot_air_time": jnp.asarray([0.1, 0.3], dtype=jnp.float32),
        "last_foot_contact": jnp.asarray([False, True]),
        "undesired_contact": jnp.asarray(0.0, dtype=jnp.float32),
        "control_timestep": 0.02,
        "done": jnp.asarray(False),
    }

    _, standing_terms = compute_velocity_reward(
        command=jnp.zeros(3, dtype=jnp.float32),
        **common,
    )
    _, moving_terms = compute_velocity_reward(
        command=jnp.asarray([0.2, 0.0, 0.0], dtype=jnp.float32),
        **common,
    )

    assert standing_terms["reward/standing_foot_vel"] == pytest.approx(-0.025)
    assert standing_terms["reward/standing_foot_air_time"] == pytest.approx(-0.012)
    assert moving_terms["reward/standing_foot_vel"] == pytest.approx(0.0)
    assert moving_terms["reward/standing_foot_air_time"] == pytest.approx(0.0)
