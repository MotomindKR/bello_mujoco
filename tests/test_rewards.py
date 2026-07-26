from dataclasses import fields

import jax
import jax.numpy as jnp
import pytest

from bello_mujoco.envs.rewards import (
    CommandConfig,
    GaitRewardConfig,
    LimitsPoseRewardConfig,
    RewardConfig,
    RewardInputs,
    RewardWeights,
    StandingRewardConfig,
    TrackingRewardConfig,
    command_is_moving,
    command_is_standing,
    compute_locomotion_reward,
)


def reward_config(
    *,
    tracking: TrackingRewardConfig = TrackingRewardConfig(),
    standing: StandingRewardConfig = StandingRewardConfig(),
    gait: GaitRewardConfig = GaitRewardConfig(),
    limits_pose: LimitsPoseRewardConfig = LimitsPoseRewardConfig(),
    **weights: float,
) -> RewardConfig:
    disabled = {field.name: 0.0 for field in fields(RewardWeights)}
    disabled.update(weights)
    return RewardConfig(
        weights=RewardWeights(**disabled),
        tracking=tracking,
        standing=standing,
        gait=gait,
        limits_pose=limits_pose,
    )


def reward_inputs(**updates) -> RewardInputs:
    inputs = RewardInputs(
        command=jnp.zeros(3, dtype=jnp.float32),
        lin_vel_xy=jnp.zeros(2, dtype=jnp.float32),
        yaw_vel_z=jnp.asarray(0.0, dtype=jnp.float32),
        lin_vel_z=jnp.asarray(0.0, dtype=jnp.float32),
        ang_vel_xy=jnp.zeros(2, dtype=jnp.float32),
        projected_gravity=jnp.asarray([0.0, 0.0, -1.0], dtype=jnp.float32),
        base_height=jnp.asarray(0.86, dtype=jnp.float32),
        joint_pos=jnp.zeros(2, dtype=jnp.float32),
        joint_pos_rel=jnp.zeros(2, dtype=jnp.float32),
        action=jnp.zeros(2, dtype=jnp.float32),
        requested_action=jnp.zeros(2, dtype=jnp.float32),
        last_action=jnp.zeros(2, dtype=jnp.float32),
        actuator_force=jnp.zeros(2, dtype=jnp.float32),
        actuator_peak_force=jnp.ones(2, dtype=jnp.float32),
        joint_limit_low=jnp.asarray([-1.0, -1.0], dtype=jnp.float32),
        joint_limit_high=jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        policy_joint_indices=jnp.asarray([0, 1], dtype=jnp.int32),
        action_target_default=jnp.zeros(2, dtype=jnp.float32),
        action_target_low=jnp.asarray([-1.0, -1.0], dtype=jnp.float32),
        action_target_high=jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        foot_contact=jnp.ones(2, dtype=jnp.float32),
        foot_linear_velocity=jnp.zeros((2, 3), dtype=jnp.float32),
        foot_angular_speed=jnp.zeros(2, dtype=jnp.float32),
        foot_air_time=jnp.zeros(2, dtype=jnp.float32),
        foot_contact_time=jnp.ones(2, dtype=jnp.float32),
        last_foot_contact=jnp.ones(2, dtype=bool),
        joint_pose_weights=jnp.ones(2, dtype=jnp.float32),
        hip_joint_indices=jnp.asarray([0], dtype=jnp.int32),
        knee_joint_indices=jnp.asarray([1], dtype=jnp.int32),
        undesired_contact=jnp.asarray(0.0, dtype=jnp.float32),
        self_collision=jnp.asarray(0.0, dtype=jnp.float32),
        done=jnp.asarray(False),
        control_timestep=jnp.asarray(0.02, dtype=jnp.float32),
        action_clip_limit=jnp.asarray(1.0, dtype=jnp.float32),
    )
    return inputs.replace(**updates)


def test_command_sampler_respects_ranges_and_exact_standing_probability() -> None:
    moving = CommandConfig(
        lin_vel_x=(0.2, 0.2),
        lin_vel_y=(-0.3, -0.3),
        ang_vel_yaw=(0.4, 0.4),
        standing_probability=0.0,
    ).sample(jax.random.PRNGKey(0), (4,))
    standing = CommandConfig(standing_probability=1.0).sample(
        jax.random.PRNGKey(1), (4,)
    )
    assert moving == pytest.approx(jnp.tile(jnp.asarray([0.2, -0.3, 0.4]), (4, 1)))
    assert standing == pytest.approx(jnp.zeros((4, 3)))


def test_command_sampler_produces_only_explicit_standing_commands() -> None:
    config = CommandConfig(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_yaw=(-1.0, 1.0),
        standing_probability=0.10,
        linear_deadband=0.10,
        yaw_deadband=0.10,
        minimum_moving_linear_speed=0.20,
        minimum_moving_yaw_speed=0.20,
    )
    commands = config.sample(jax.random.PRNGKey(2), (100_000,))
    standing = command_is_standing(commands, config)
    exact_zero = jnp.all(commands == 0.0, axis=1)

    assert jnp.mean(standing.astype(jnp.float32)) == pytest.approx(0.10, abs=0.005)
    assert jnp.array_equal(standing, exact_zero)
    assert jnp.all(command_is_moving(commands[~standing], config))


def test_command_semantics_use_per_axis_deadbands() -> None:
    config = CommandConfig()

    assert command_is_standing(jnp.asarray([0.08, 0.08, 0.08]), config)
    assert command_is_moving(jnp.asarray([0.20, 0.0, 0.0]), config)
    assert command_is_moving(jnp.asarray([0.0, 0.0, -0.20]), config)


def test_tracking_reward_is_maximal_at_command_and_scales_with_timestep() -> None:
    config = reward_config(
        track_lin_vel_xy=2.0,
        tracking=TrackingRewardConfig(
            lin_vel_xy_moving_std=0.75,
            lin_vel_xy_still_std=0.25,
        ),
    )
    inputs = reward_inputs(
        command=jnp.asarray([0.5, -0.25, 0.0]),
        lin_vel_xy=jnp.asarray([0.5, -0.25]),
    )
    reward, terms = compute_locomotion_reward(config, inputs)
    assert terms["reward/track_lin_vel_xy"] == pytest.approx(0.04)
    assert reward == pytest.approx(0.04)
    worse, _ = compute_locomotion_reward(
        config, inputs.replace(lin_vel_xy=jnp.asarray([-0.5, 0.25]))
    )
    assert worse < reward


def test_tracking_reward_uses_tighter_tolerance_on_each_zero_command_axis() -> None:
    config = reward_config(
        track_lin_vel_xy=1.0,
        track_ang_vel_z=1.0,
        tracking=TrackingRewardConfig(
            lin_vel_xy_moving_std=0.75,
            lin_vel_xy_still_std=0.25,
            ang_vel_z_moving_std=0.5,
            ang_vel_z_still_std=0.2,
        ),
    )
    inputs = reward_inputs(command=jnp.asarray([0.5, 0.0, 0.0]))
    _, perfect = compute_locomotion_reward(
        config,
        inputs.replace(lin_vel_xy=jnp.asarray([0.5, 0.0])),
    )
    _, moving_axis_error = compute_locomotion_reward(
        config,
        inputs.replace(lin_vel_xy=jnp.asarray([0.7, 0.0])),
    )
    _, still_axis_error = compute_locomotion_reward(
        config,
        inputs.replace(lin_vel_xy=jnp.asarray([0.5, 0.2])),
    )
    _, tiny_command = compute_locomotion_reward(
        config,
        inputs.replace(command=jnp.asarray([0.05, 0.0, 0.05])),
    )

    assert perfect["reward/track_lin_vel_xy"] == pytest.approx(0.02)
    assert moving_axis_error["reward/track_lin_vel_xy"] == pytest.approx(
        0.02 * jnp.exp(-((0.2 / 0.75) ** 2))
    )
    assert still_axis_error["reward/track_lin_vel_xy"] == pytest.approx(
        0.02 * jnp.exp(-((0.2 / 0.25) ** 2))
    )
    assert (
        still_axis_error["reward/track_lin_vel_xy"]
        < moving_axis_error["reward/track_lin_vel_xy"]
    )
    assert tiny_command["reward/track_lin_vel_xy"] == pytest.approx(0.02)
    assert tiny_command["reward/track_ang_vel_z"] == pytest.approx(0.02)


def test_joint_limit_penalty_increases_with_distance_past_soft_limit() -> None:
    config = reward_config(
        joint_limit=-1.0,
        limits_pose=LimitsPoseRewardConfig(joint_soft_limit_factor=0.9),
    )
    _, near = compute_locomotion_reward(
        config, reward_inputs(joint_pos=jnp.asarray([0.95, 0.0]))
    )
    _, far = compute_locomotion_reward(
        config, reward_inputs(joint_pos=jnp.asarray([1.20, 0.0]))
    )
    assert near["reward/joint_limit"] == pytest.approx(-0.001)
    assert far["reward/joint_limit"] == pytest.approx(-0.006)


def test_negative_total_reward_is_not_discarded() -> None:
    config = reward_config(
        joint_limit=-1.0,
        limits_pose=LimitsPoseRewardConfig(joint_soft_limit_factor=0.9),
    )
    reward, terms = compute_locomotion_reward(
        config, reward_inputs(joint_pos=jnp.asarray([1.2, 0.0]))
    )

    assert reward == pytest.approx(terms["reward/joint_limit"])
    assert reward < 0.0


def test_termination_penalty_scales_with_timestep() -> None:
    config = reward_config(termination=-2.0)

    first, first_terms = compute_locomotion_reward(
        config,
        reward_inputs(done=jnp.asarray(True), control_timestep=jnp.asarray(0.02)),
    )
    second, second_terms = compute_locomotion_reward(
        config,
        reward_inputs(done=jnp.asarray(True), control_timestep=jnp.asarray(0.005)),
    )

    assert first_terms["reward/termination"] == pytest.approx(-0.04)
    assert second_terms["reward/termination"] == pytest.approx(-0.01)
    assert first == pytest.approx(4.0 * second)


def test_hip_deviation_uses_lateral_command_axis() -> None:
    config = reward_config(
        hip_deviation=-1.0,
        limits_pose=LimitsPoseRewardConfig(hip_lateral_command_threshold=0.1),
    )
    inputs = reward_inputs(joint_pos_rel=jnp.asarray([0.5, 0.0]))

    _, forward = compute_locomotion_reward(
        config, inputs.replace(command=jnp.asarray([0.5, 0.0, 0.0]))
    )
    _, lateral = compute_locomotion_reward(
        config, inputs.replace(command=jnp.asarray([0.0, 0.5, 0.0]))
    )

    assert forward["reward/hip_deviation"] == pytest.approx(0.0)
    assert lateral["reward/hip_deviation"] == pytest.approx(-0.01)


def test_action_limit_penalty_is_quadratic_outside_soft_band() -> None:
    config = reward_config(
        action_limit=-1.0,
        limits_pose=LimitsPoseRewardConfig(action_soft_limit_ratio=0.8),
    )
    _, boundary = compute_locomotion_reward(
        config, reward_inputs(action=jnp.asarray([0.8, 0.0]))
    )
    _, outside = compute_locomotion_reward(
        config, reward_inputs(action=jnp.asarray([0.9, 0.0]))
    )
    assert boundary["reward/action_limit"] == pytest.approx(0.0)
    assert outside["reward/action_limit"] == pytest.approx(-0.0025)


def test_torque_penalty_normalizes_each_actuator_and_emphasizes_over_peak() -> None:
    config = reward_config(
        torque_l1=-1.0,
        torque_peak=-10.0,
    )
    inputs = reward_inputs(
        actuator_force=jnp.asarray([1.0, 4.0]),
        actuator_peak_force=jnp.asarray([2.0, 4.0]),
    )
    _, at_or_below_peak = compute_locomotion_reward(config, inputs)
    _, over_peak = compute_locomotion_reward(
        config,
        inputs.replace(actuator_force=jnp.asarray([4.0, 4.0])),
    )

    assert at_or_below_peak["reward/torque_l1"] == pytest.approx(-0.015)
    assert at_or_below_peak["reward/torque_peak"] == pytest.approx(0.0)
    assert over_peak["reward/torque_l1"] == pytest.approx(-0.03)
    assert over_peak["reward/torque_peak"] == pytest.approx(-0.1)


def test_invalid_action_checks_requested_target_and_policy_clip() -> None:
    config = reward_config(invalid_action=-1.0)
    _, terms = compute_locomotion_reward(
        config,
        reward_inputs(
            requested_action=jnp.asarray([1.5, 0.0]),
            action=jnp.asarray([1.0, 0.0]),
        ),
    )
    assert terms["reward/invalid_action"] == pytest.approx(-0.01)


def test_contact_timing_event_penalty_scales_with_timestep() -> None:
    config = reward_config(
        contact_timing=-5.0,
        gait=GaitRewardConfig(
            contact_min_air_time=0.2,
            contact_min_contact_time=0.3,
        ),
    )
    _, terms = compute_locomotion_reward(
        config,
        reward_inputs(
            command=jnp.asarray([0.5, 0.0, 0.0]),
            foot_contact=jnp.asarray([1.0, 1.0]),
            last_foot_contact=jnp.asarray([False, True]),
            foot_air_time=jnp.asarray([0.1, 0.0]),
        ),
    )
    assert terms["reward/contact_timing"] == pytest.approx(-0.025)

    _, finer_timestep = compute_locomotion_reward(
        config,
        reward_inputs(
            command=jnp.asarray([0.5, 0.0, 0.0]),
            foot_contact=jnp.asarray([1.0, 1.0]),
            last_foot_contact=jnp.asarray([False, True]),
            foot_air_time=jnp.asarray([0.1, 0.0]),
            control_timestep=jnp.asarray(0.005),
        ),
    )
    assert finer_timestep["reward/contact_timing"] == pytest.approx(-0.00625)

    _, stopping = compute_locomotion_reward(
        config,
        reward_inputs(
            lin_vel_xy=jnp.asarray([0.5, 0.0]),
            foot_contact=jnp.asarray([1.0, 1.0]),
            last_foot_contact=jnp.asarray([False, True]),
            foot_air_time=jnp.asarray([0.1, 0.0]),
        ),
    )
    assert stopping["reward/contact_timing"] == pytest.approx(0.0)


def test_flight_penalty_requires_motion_and_both_feet_airborne() -> None:
    config = reward_config(
        flight=-2.0,
        gait=GaitRewardConfig(
            flight_body_speed_threshold=0.1,
            flight_yaw_speed_threshold=0.1,
        ),
    )
    airborne = reward_inputs(
        foot_contact=jnp.zeros(2),
        last_foot_contact=jnp.zeros(2, dtype=bool),
    )

    _, standing = compute_locomotion_reward(config, airborne)
    _, commanded = compute_locomotion_reward(
        config, airborne.replace(command=jnp.asarray([0.5, 0.0, 0.0]))
    )
    _, stopping = compute_locomotion_reward(
        config, airborne.replace(lin_vel_xy=jnp.asarray([0.5, 0.0]))
    )
    _, one_foot_contact = compute_locomotion_reward(
        config,
        airborne.replace(
            command=jnp.asarray([0.5, 0.0, 0.0]),
            foot_contact=jnp.asarray([1.0, 0.0]),
        ),
    )

    assert standing["reward/flight"] == pytest.approx(0.0)
    assert commanded["reward/flight"] == pytest.approx(-0.04)
    assert stopping["reward/flight"] == pytest.approx(-0.04)
    assert one_foot_contact["reward/flight"] == pytest.approx(0.0)


def test_pure_yaw_command_enables_stepping_event_rewards() -> None:
    config = reward_config(
        contact_timing=-0.1,
        touchdown_air_time=2.0,
        gait=GaitRewardConfig(
            contact_min_air_time=0.2,
            contact_min_contact_time=0.3,
            touchdown_min_air_time=0.2,
            touchdown_max_air_time=0.5,
        ),
    )
    _, terms = compute_locomotion_reward(
        config,
        reward_inputs(
            command=jnp.asarray([0.0, 0.0, 0.5]),
            foot_contact=jnp.asarray([0.0, 1.0]),
            last_foot_contact=jnp.asarray([True, False]),
            foot_air_time=jnp.asarray([0.0, 0.3]),
            foot_contact_time=jnp.asarray([0.1, 0.0]),
        ),
    )

    assert terms["reward/contact_timing"] < 0.0
    assert terms["reward/touchdown_air_time"] > 0.0


def test_touchdown_air_time_requires_a_moving_command() -> None:
    config = reward_config(
        touchdown_air_time=2.0,
        gait=GaitRewardConfig(
            touchdown_min_air_time=0.2,
            touchdown_max_air_time=0.5,
        ),
    )
    inputs = reward_inputs(
        foot_contact=jnp.asarray([1.0, 1.0]),
        last_foot_contact=jnp.asarray([False, True]),
        foot_air_time=jnp.asarray([0.3, 0.0]),
    )
    _, standing = compute_locomotion_reward(config, inputs)
    _, moving = compute_locomotion_reward(
        config, inputs.replace(command=jnp.asarray([0.5, 0.0, 0.0]))
    )
    _, stopping = compute_locomotion_reward(
        config, inputs.replace(lin_vel_xy=jnp.asarray([0.5, 0.0]))
    )
    _, finer_timestep = compute_locomotion_reward(
        config,
        inputs.replace(
            command=jnp.asarray([0.5, 0.0, 0.0]),
            control_timestep=jnp.asarray(0.005),
        ),
    )
    assert standing["reward/touchdown_air_time"] == pytest.approx(0.0)
    assert moving["reward/touchdown_air_time"] == pytest.approx(0.004)
    assert stopping["reward/touchdown_air_time"] == pytest.approx(0.0)
    assert finer_timestep["reward/touchdown_air_time"] == pytest.approx(0.001)


def test_stand_foot_motion_is_bounded_command_gated_and_contact_free() -> None:
    config = reward_config(
        stand_foot_motion=-0.25,
        standing=StandingRewardConfig(
            foot_linear_speed_scale=0.15,
            foot_angular_speed_scale=1.0,
            foot_angular_weight=0.25,
        ),
    )
    inputs = reward_inputs()
    foot_motion = jnp.asarray([[0.15, 0.0, 0.0], [0.0, 0.0, 0.0]])
    _, still = compute_locomotion_reward(config, inputs)
    _, moving_feet = compute_locomotion_reward(
        config,
        inputs.replace(foot_linear_velocity=foot_motion),
    )
    _, no_contact = compute_locomotion_reward(
        config,
        inputs.replace(
            foot_contact=jnp.zeros(2),
            foot_linear_velocity=foot_motion,
        ),
    )
    _, commanded = compute_locomotion_reward(
        config,
        inputs.replace(
            command=jnp.asarray([0.2, 0.0, 0.0]),
            foot_linear_velocity=foot_motion,
        ),
    )
    _, stopping = compute_locomotion_reward(
        config,
        inputs.replace(
            lin_vel_xy=jnp.asarray([0.2, 0.0]),
            foot_linear_velocity=foot_motion,
        ),
    )
    _, saturated = compute_locomotion_reward(
        config,
        inputs.replace(
            foot_linear_velocity=jnp.full((2, 3), 100.0),
            foot_angular_speed=jnp.full(2, 100.0),
        ),
    )

    expected = -0.0025 * jnp.tanh(1.0)
    assert still["reward/stand_foot_motion"] == pytest.approx(0.0)
    assert moving_feet["reward/stand_foot_motion"] == pytest.approx(expected)
    assert no_contact["reward/stand_foot_motion"] == pytest.approx(expected)
    assert commanded["reward/stand_foot_motion"] == pytest.approx(0.0)
    assert stopping["reward/stand_foot_motion"] == pytest.approx(expected)
    assert saturated["reward/stand_foot_motion"] == pytest.approx(-0.00625)


def test_default_reward_matches_strong_stand_numerical_baseline() -> None:
    inputs = reward_inputs(
        command=jnp.asarray([0.35, -0.22, 0.28]),
        lin_vel_xy=jnp.asarray([0.18, -0.31]),
        yaw_vel_z=jnp.asarray(0.11),
        lin_vel_z=jnp.asarray(-0.07),
        ang_vel_xy=jnp.asarray([0.04, -0.09]),
        projected_gravity=jnp.asarray([0.08, -0.03, -0.99]),
        base_height=jnp.asarray(0.82),
        joint_pos=jnp.asarray([0.95, -0.93]),
        joint_pos_rel=jnp.asarray([0.12, -0.19]),
        action=jnp.asarray([0.91, -0.84]),
        requested_action=jnp.asarray([1.15, -0.92]),
        last_action=jnp.asarray([0.64, -0.55]),
        actuator_force=jnp.asarray([2.4, -3.2]),
        actuator_peak_force=jnp.asarray([2.0, 4.0]),
        foot_contact=jnp.asarray([1.0, 0.0]),
        last_foot_contact=jnp.asarray([False, True]),
        foot_air_time=jnp.asarray([0.12, 0.34]),
        foot_contact_time=jnp.asarray([0.0, 0.18]),
        foot_linear_velocity=jnp.asarray([[0.17, 0.03, -0.02], [0.09, -0.11, 0.04]]),
        foot_angular_speed=jnp.asarray([0.6, 0.9]),
        undesired_contact=jnp.asarray(1.0),
        self_collision=jnp.asarray(0.5),
        done=jnp.asarray(True),
    )

    reward, terms = compute_locomotion_reward(RewardConfig(), inputs)
    assert reward == pytest.approx(-2.031494140625)
    assert terms["reward/track_lin_vel_xy"] == pytest.approx(0.0187267773)
    assert terms["reward/track_ang_vel_z"] == pytest.approx(0.0178166311)
    assert terms["reward/action_limit"] == pytest.approx(-0.003425)
    assert terms["reward/torque_peak"] == pytest.approx(-0.0048000021)
    assert terms["reward/contact_timing"] == pytest.approx(-0.0320000052)
    assert terms["reward/touchdown_air_time"] == pytest.approx(0.0005999999)
    assert terms["reward/termination"] == pytest.approx(-2.0)

    standing_reward, standing_terms = compute_locomotion_reward(
        RewardConfig(),
        inputs.replace(
            command=jnp.zeros(3),
            lin_vel_xy=jnp.asarray([0.05, -0.02]),
            yaw_vel_z=jnp.asarray(0.03),
            foot_contact=jnp.ones(2),
            last_foot_contact=jnp.ones(2, dtype=bool),
            done=jnp.asarray(False),
        ),
    )
    assert standing_reward == pytest.approx(-0.0168412589)
    assert standing_terms["reward/stand_foot_motion"] == pytest.approx(-0.0187412165)
    assert standing_terms["reward/contact_timing"] == pytest.approx(0.0)
    assert standing_terms["reward/touchdown_air_time"] == pytest.approx(0.0)


def test_full_reward_is_jittable_finite_and_uniformly_timestep_scaled() -> None:
    config = RewardConfig(
        weights=RewardWeights(**{field.name: 1.0 for field in fields(RewardWeights)})
    )
    moving_inputs = reward_inputs(
        command=jnp.asarray([0.5, 0.5, 0.5]),
        lin_vel_z=jnp.asarray(1.0),
        ang_vel_xy=jnp.asarray([1.0, 1.0]),
        projected_gravity=jnp.asarray([0.2, 0.1, -0.9]),
        base_height=jnp.asarray(0.5),
        joint_pos=jnp.asarray([1.2, -1.2]),
        joint_pos_rel=jnp.asarray([0.5, 0.5]),
        action=jnp.asarray([0.9, 0.5]),
        requested_action=jnp.asarray([1.5, 0.0]),
        actuator_force=jnp.asarray([1.0, 2.0]),
        foot_contact=jnp.asarray([1.0, 0.0]),
        last_foot_contact=jnp.asarray([False, True]),
        foot_air_time=jnp.asarray([0.3, 0.1]),
        foot_contact_time=jnp.asarray([0.0, 0.1]),
        foot_linear_velocity=jnp.asarray([[0.3, 0.1, 0.2], [0.2, 0.4, 0.1]]),
        foot_angular_speed=jnp.asarray([1.0, 1.0]),
        undesired_contact=jnp.asarray(1.0),
        self_collision=jnp.asarray(1.0),
        done=jnp.asarray(True),
    )
    standing_inputs = moving_inputs.replace(
        command=jnp.zeros(3),
        foot_contact=jnp.ones(2),
    )
    compute = jax.jit(compute_locomotion_reward, static_argnums=0)
    reward, moving_terms = compute(config, moving_inputs)
    _, moving_fine = compute(
        config, moving_inputs.replace(control_timestep=jnp.asarray(0.005))
    )
    _, standing_terms = compute(config, standing_inputs)
    _, standing_fine = compute(
        config, standing_inputs.replace(control_timestep=jnp.asarray(0.005))
    )
    _, airborne_terms = compute(
        config,
        moving_inputs.replace(
            foot_contact=jnp.zeros(2),
            last_foot_contact=jnp.zeros(2, dtype=bool),
        ),
    )
    _, airborne_fine = compute(
        config,
        moving_inputs.replace(
            foot_contact=jnp.zeros(2),
            last_foot_contact=jnp.zeros(2, dtype=bool),
            control_timestep=jnp.asarray(0.005),
        ),
    )

    assert jnp.isfinite(reward)
    assert all(bool(jnp.isfinite(term)) for term in moving_terms.values())
    assert set(moving_terms) == {
        f"reward/{field.name}" for field in fields(RewardWeights)
    }
    for name in moving_terms:
        coarse = moving_terms[name]
        fine = moving_fine[name]
        if coarse == 0.0:
            coarse = standing_terms[name]
            fine = standing_fine[name]
        if coarse == 0.0:
            coarse = airborne_terms[name]
            fine = airborne_fine[name]
        assert coarse != 0.0, name
        assert coarse == pytest.approx(4.0 * fine), name


def test_reward_validation_rejects_invalid_safety_ranges() -> None:
    with pytest.raises(ValueError, match="soft_limit_ratio"):
        RewardConfig(limits_pose=LimitsPoseRewardConfig(action_soft_limit_ratio=1.0))
    with pytest.raises(ValueError, match="soft_limit_factor"):
        RewardConfig(limits_pose=LimitsPoseRewardConfig(joint_soft_limit_factor=0.0))
    with pytest.raises(ValueError, match="base_height_target"):
        RewardConfig(limits_pose=LimitsPoseRewardConfig(base_height_target=0.0))
    with pytest.raises(ValueError, match="foot_linear_speed_scale"):
        RewardConfig(standing=StandingRewardConfig(foot_linear_speed_scale=0.0))
    with pytest.raises(ValueError, match="foot_angular_speed_scale"):
        RewardConfig(
            standing=StandingRewardConfig(foot_angular_speed_scale=float("nan"))
        )
    with pytest.raises(ValueError, match="foot_angular_weight"):
        RewardConfig(standing=StandingRewardConfig(foot_angular_weight=-1.0))


def test_command_validation_requires_a_separate_moving_range() -> None:
    with pytest.raises(ValueError, match="minimum_moving_linear_speed"):
        CommandConfig(
            linear_deadband=0.1,
            minimum_moving_linear_speed=0.1,
        )
    with pytest.raises(ValueError, match="at least one command range"):
        CommandConfig(
            lin_vel_x=(-0.1, 0.1),
            lin_vel_y=(-0.1, 0.1),
            ang_vel_yaw=(-0.1, 0.1),
        )


@pytest.mark.parametrize(
    "gait",
    [
        GaitRewardConfig(contact_min_air_time=float("nan")),
        GaitRewardConfig(contact_min_air_time=float("inf")),
        GaitRewardConfig(contact_min_contact_time=float("nan")),
        GaitRewardConfig(contact_min_contact_time=float("inf")),
    ],
)
def test_contact_timing_validation_rejects_nonfinite_bounds(
    gait: GaitRewardConfig,
) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        RewardConfig(gait=gait)


@pytest.mark.parametrize(
    "gait",
    [
        GaitRewardConfig(touchdown_min_air_time=float("nan")),
        GaitRewardConfig(touchdown_min_air_time=float("inf")),
        GaitRewardConfig(touchdown_max_air_time=float("nan")),
        GaitRewardConfig(touchdown_max_air_time=float("inf")),
    ],
)
def test_touchdown_air_time_validation_rejects_nonfinite_bounds(
    gait: GaitRewardConfig,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        RewardConfig(gait=gait)
