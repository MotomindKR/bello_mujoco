import jax.numpy as jnp

from bello_mujoco.envs.rewards import (
    FootClearanceRewardConfig,
    FootContactRewardConfig,
    FootSlipRewardConfig,
    RewardConfig,
    TerminationRewardConfig,
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
        "action": jnp.zeros(4, dtype=jnp.float32),
        "last_action": jnp.zeros(4, dtype=jnp.float32),
        "actuator_force": jnp.zeros(4, dtype=jnp.float32),
        "joint_limit_low": -jnp.ones(4, dtype=jnp.float32),
        "joint_limit_high": jnp.ones(4, dtype=jnp.float32),
        "hold_joint_indices": jnp.asarray([2, 3], dtype=jnp.int32),
        "foot_contact": jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        "foot_height": jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        "foot_lateral_speed_sq": jnp.zeros(2, dtype=jnp.float32),
        "undesired_contact": jnp.asarray(0.0, dtype=jnp.float32),
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
        "foot_contact": jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        "foot_height": jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        "foot_lateral_speed_sq": jnp.zeros(2, dtype=jnp.float32),
        "undesired_contact": jnp.asarray(0.0, dtype=jnp.float32),
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
        "foot_contact": jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        "foot_height": jnp.asarray([0.0, 0.2], dtype=jnp.float32),
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
    assert terms["reward/undesired_contact"] == -2.0


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
        foot_contact=jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        foot_height=jnp.zeros(2, dtype=jnp.float32),
        foot_lateral_speed_sq=jnp.asarray([5.0, 0.0], dtype=jnp.float32),
        undesired_contact=jnp.asarray(0.0, dtype=jnp.float32),
        done=jnp.asarray(True),
    )

    assert terms["reward/foot_slip"] == 0.0
    assert terms["reward/termination"] == 0.0
