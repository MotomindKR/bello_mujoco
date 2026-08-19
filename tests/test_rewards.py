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
    SmoothnessRewardConfig,
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
    smoothness: SmoothnessRewardConfig = SmoothnessRewardConfig(),
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
        smoothness=smoothness,
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
        last_action_delta=jnp.zeros(2, dtype=jnp.float32),
        action_smoothness_weights=jnp.ones(2, dtype=jnp.float32),
        actuator_force=jnp.zeros(2, dtype=jnp.float32),
        requested_actuator_force=jnp.zeros(2, dtype=jnp.float32),
        actuator_peak_force=jnp.ones(2, dtype=jnp.float32),
        joint_force=jnp.zeros(2, dtype=jnp.float32),
        robot_weight=jnp.asarray(1.0, dtype=jnp.float32),
        joint_limit_low=jnp.asarray([-1.0, -1.0], dtype=jnp.float32),
        joint_limit_high=jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        policy_joint_indices=jnp.asarray([0, 1], dtype=jnp.int32),
        action_target_default=jnp.zeros(2, dtype=jnp.float32),
        action_target_low=jnp.asarray([-1.0, -1.0], dtype=jnp.float32),
        action_target_high=jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        foot_contact=jnp.ones(2, dtype=jnp.float32),
        foot_touchdown=jnp.zeros(2, dtype=bool),
        foot_normal_impulse=jnp.zeros(2, dtype=jnp.float32),
        foot_peak_normal_force=jnp.zeros(2, dtype=jnp.float32),
        touchdown_foot_linear_velocity=jnp.zeros((2, 3), dtype=jnp.float32),
        foot_linear_velocity=jnp.zeros((2, 3), dtype=jnp.float32),
        foot_angular_speed=jnp.zeros(2, dtype=jnp.float32),
        foot_heading_error=jnp.zeros(2, dtype=jnp.float32),
        foot_height=jnp.zeros(2, dtype=jnp.float32),
        gait_active=jnp.asarray(False),
        desired_foot_contact=jnp.ones(2, dtype=jnp.float32),
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


def test_stratified_command_batch_has_exact_standing_cohort() -> None:
    config = CommandConfig(standing_probability=0.10)
    sample = jax.jit(config.sample_stratified_batch, static_argnums=1)
    commands = sample(jax.random.PRNGKey(3), 100)
    standing = command_is_standing(commands, config)

    assert jnp.sum(standing) == 10
    assert jnp.all(commands[standing] == 0.0)
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


def test_healthy_survival_reward_excludes_terminal_step() -> None:
    config = reward_config(healthy_survival=1.5)

    healthy, healthy_terms = compute_locomotion_reward(config, reward_inputs())
    terminal, terminal_terms = compute_locomotion_reward(
        config, reward_inputs(done=jnp.asarray(True))
    )

    assert healthy_terms["reward/healthy_survival"] == pytest.approx(0.03)
    assert terminal_terms["reward/healthy_survival"] == pytest.approx(0.0)
    assert healthy == pytest.approx(0.03)
    assert terminal == pytest.approx(0.0)


def test_hip_deviation_uses_lateral_command_axis() -> None:
    config = reward_config(
        hip_deviation=-1.0,
        limits_pose=LimitsPoseRewardConfig(hip_lateral_command_threshold=0.1),
    )
    inputs = reward_inputs(joint_pos_rel=jnp.asarray([0.5, 0.0]))

    _, lateral = compute_locomotion_reward(
        config, inputs.replace(command=jnp.asarray([0.5, 0.0, 0.0]))
    )
    _, forward = compute_locomotion_reward(
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


def test_action_smoothness_uses_joint_weights_and_previous_delta() -> None:
    config = reward_config(action_rate=-1.0, action_acceleration=-1.0)
    _, terms = compute_locomotion_reward(
        config,
        reward_inputs(
            action=jnp.asarray([1.0, 1.0]),
            last_action=jnp.zeros(2),
            last_action_delta=jnp.asarray([0.5, 0.0]),
            action_smoothness_weights=jnp.asarray([2.0, 1.0]),
        ),
    )

    assert terms["reward/action_rate"] == pytest.approx(-0.06)
    assert terms["reward/action_acceleration"] == pytest.approx(-0.03)


def test_torque_penalty_normalizes_each_actuator_and_emphasizes_over_peak() -> None:
    config = reward_config(
        torque_l1=-1.0,
        torque_peak=-10.0,
    )
    inputs = reward_inputs(
        actuator_force=jnp.asarray([1.0, 4.0]),
        requested_actuator_force=jnp.asarray([1.0, 4.0]),
        actuator_peak_force=jnp.asarray([2.0, 4.0]),
    )
    _, at_or_below_peak = compute_locomotion_reward(config, inputs)
    _, over_peak = compute_locomotion_reward(
        config,
        inputs.replace(
            actuator_force=jnp.asarray([2.0, 4.0]),
            requested_actuator_force=jnp.asarray([4.0, 4.0]),
        ),
    )

    assert at_or_below_peak["reward/torque_l1"] == pytest.approx(-0.015)
    assert at_or_below_peak["reward/torque_peak"] == pytest.approx(0.0)
    assert over_peak["reward/torque_l1"] == pytest.approx(-0.02)
    assert over_peak["reward/torque_peak"] == pytest.approx(-0.1)


def test_joint_force_peak_uses_top_loads_only_during_commanded_gait() -> None:
    config = reward_config(
        joint_force_peak=-2.0,
        gait=GaitRewardConfig(joint_force_soft_limit_body_weights=1.0),
    )
    inputs = reward_inputs(
        command=jnp.asarray([0.5, 0.0, 0.0]),
        gait_active=jnp.asarray(True),
        actuator_force=jnp.asarray([100.0, 100.0]),
        joint_force=jnp.asarray([0.0, 500.0, 1_000.0, 1_500.0]),
        robot_weight=jnp.asarray(500.0),
    )

    _, terms = compute_locomotion_reward(config, inputs)
    _, below_limit = compute_locomotion_reward(
        config,
        inputs.replace(joint_force=jnp.asarray([250.0, 500.0, 250.0, 500.0])),
    )
    _, standing = compute_locomotion_reward(
        config,
        inputs.replace(command=jnp.zeros(3), gait_active=jnp.asarray(False)),
    )

    assert terms["reward/joint_force_peak"] == pytest.approx(-0.1)
    assert terms["reward/torque_l1"] == pytest.approx(0.0)
    assert terms["reward/torque_peak"] == pytest.approx(0.0)
    assert below_limit["reward/joint_force_peak"] == pytest.approx(0.0)
    assert standing["reward/joint_force_peak"] == pytest.approx(0.0)


def test_touchdown_impact_uses_normal_impulse_and_preimpact_vertical_speed() -> None:
    config = reward_config(
        touchdown_impact=-4.0,
        gait=GaitRewardConfig(
            touchdown_force_soft_limit_body_weights=1.0,
            touchdown_vertical_speed_scale=0.3,
        ),
    )
    inputs = reward_inputs(
        foot_touchdown=jnp.asarray([True, False]),
        foot_normal_impulse=jnp.asarray([0.04, 0.0]),
        foot_peak_normal_force=jnp.asarray([1.2, 0.0]),
        touchdown_foot_linear_velocity=jnp.asarray([[0.0, 0.0, -0.3], [0.0, 0.0, 0.0]]),
    )

    _, terms = compute_locomotion_reward(config, inputs)
    _, no_touchdown = compute_locomotion_reward(
        config, inputs.replace(foot_touchdown=jnp.zeros(2, dtype=bool))
    )

    assert terms["reward/touchdown_impact"] == pytest.approx(-0.08)
    assert no_touchdown["reward/touchdown_impact"] == pytest.approx(0.0)


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


def test_gait_contact_penalizes_phase_mismatch_only_while_active() -> None:
    config = reward_config(gait_contact=-1.0)
    inputs = reward_inputs(
        command=jnp.asarray([0.5, 0.0, 0.0]),
        gait_active=jnp.asarray(True),
        desired_foot_contact=jnp.asarray([1.0, 0.0]),
    )

    _, matched = compute_locomotion_reward(
        config, inputs.replace(foot_contact=jnp.asarray([1.0, 0.0]))
    )
    _, mismatched = compute_locomotion_reward(
        config, inputs.replace(foot_contact=jnp.asarray([0.0, 1.0]))
    )
    _, standing = compute_locomotion_reward(
        config,
        inputs.replace(command=jnp.zeros(3), gait_active=jnp.asarray(False)),
    )

    assert matched["reward/gait_contact"] == pytest.approx(0.0)
    assert mismatched["reward/gait_contact"] == pytest.approx(-0.02)
    assert standing["reward/gait_contact"] == pytest.approx(0.0)


def test_stance_slip_and_swing_clearance_follow_desired_foot_phase() -> None:
    config = reward_config(
        gait_stance_slip=-1.0,
        gait_swing_clearance=-1.0,
        gait=GaitRewardConfig(swing_clearance_target=0.08),
    )
    inputs = reward_inputs(
        command=jnp.asarray([0.5, 0.0, 0.0]),
        gait_active=jnp.asarray(True),
        desired_foot_contact=jnp.asarray([1.0, 0.0]),
        foot_linear_velocity=jnp.asarray([[0.2, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        foot_height=jnp.asarray([0.0, 0.04]),
    )

    _, terms = compute_locomotion_reward(config, inputs)
    _, standing = compute_locomotion_reward(
        config,
        inputs.replace(command=jnp.zeros(3), gait_active=jnp.asarray(False)),
    )

    assert terms["reward/gait_stance_slip"] == pytest.approx(-0.004)
    assert terms["reward/gait_swing_clearance"] == pytest.approx(-0.0025)
    assert standing["reward/gait_stance_slip"] == pytest.approx(0.0)
    assert standing["reward/gait_swing_clearance"] == pytest.approx(0.0)


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


def test_stand_foot_heading_is_bounded_and_requires_healthy_double_support() -> None:
    config = reward_config(
        stand_foot_heading=-0.5,
        standing=StandingRewardConfig(
            foot_heading_deadband=0.1,
            foot_heading_scale=0.2,
        ),
    )
    inputs = reward_inputs(
        foot_heading_error=jnp.asarray([0.3, -0.2], dtype=jnp.float32)
    )

    _, standing = compute_locomotion_reward(config, inputs)
    _, inside_deadband = compute_locomotion_reward(
        config,
        inputs.replace(foot_heading_error=jnp.asarray([0.1, -0.1])),
    )
    _, single_support = compute_locomotion_reward(
        config,
        inputs.replace(foot_contact=jnp.asarray([1.0, 0.0])),
    )
    _, terminal = compute_locomotion_reward(
        config,
        inputs.replace(done=jnp.asarray(True)),
    )
    _, moving = compute_locomotion_reward(
        config,
        inputs.replace(command=jnp.asarray([0.5, 0.0, 0.0])),
    )
    _, saturated = compute_locomotion_reward(
        config,
        inputs.replace(foot_heading_error=jnp.asarray([1.0e6, -1.0e6])),
    )

    assert standing["reward/stand_foot_heading"] == pytest.approx(-1.0 / 240.0)
    assert inside_deadband["reward/stand_foot_heading"] == pytest.approx(0.0)
    assert single_support["reward/stand_foot_heading"] == pytest.approx(0.0)
    assert terminal["reward/stand_foot_heading"] == pytest.approx(0.0)
    assert moving["reward/stand_foot_heading"] == pytest.approx(0.0)
    assert saturated["reward/stand_foot_heading"] == pytest.approx(-0.01, abs=1.0e-8)


def test_stand_contact_penalizes_missing_double_support_only_while_standing() -> None:
    config = reward_config(stand_contact=-1.0)
    inputs = reward_inputs(foot_contact=jnp.asarray([1.0, 0.0]))

    _, standing = compute_locomotion_reward(config, inputs)
    _, supported = compute_locomotion_reward(
        config, inputs.replace(foot_contact=jnp.ones(2))
    )
    _, moving = compute_locomotion_reward(
        config, inputs.replace(command=jnp.asarray([0.5, 0.0, 0.0]))
    )

    assert standing["reward/stand_contact"] == pytest.approx(-0.01)
    assert supported["reward/stand_contact"] == pytest.approx(0.0)
    assert moving["reward/stand_contact"] == pytest.approx(0.0)


def test_stand_still_rewards_healthy_quiet_double_support() -> None:
    config = reward_config(
        stand_still=1.0,
        standing=StandingRewardConfig(
            foot_linear_speed_scale=0.15,
            foot_angular_speed_scale=1.0,
            foot_angular_weight=0.25,
        ),
    )
    inputs = reward_inputs()

    quiet_reward, quiet = compute_locomotion_reward(config, inputs)
    _, one_foot = compute_locomotion_reward(
        config, inputs.replace(foot_contact=jnp.asarray([1.0, 0.0]))
    )
    _, moving_feet = compute_locomotion_reward(
        config,
        inputs.replace(
            foot_linear_velocity=jnp.asarray([[0.15, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ),
    )
    _, moving_command = compute_locomotion_reward(
        config, inputs.replace(command=jnp.asarray([0.5, 0.0, 0.0]))
    )
    terminal_reward, terminal = compute_locomotion_reward(
        config, inputs.replace(done=jnp.asarray(True))
    )

    assert quiet["reward/stand_still"] == pytest.approx(0.02)
    assert quiet_reward > 0.0
    assert one_foot["reward/stand_still"] == pytest.approx(0.01)
    assert 0.0 < moving_feet["reward/stand_still"] < 0.02
    assert moving_command["reward/stand_still"] == pytest.approx(0.0)
    assert terminal["reward/stand_still"] == pytest.approx(0.0)
    assert terminal_reward == pytest.approx(0.0)


def test_default_healthy_standing_return_is_positive_despite_joint_support_load() -> (
    None
):
    config = RewardConfig()
    healthy_reward, healthy = compute_locomotion_reward(
        config,
        reward_inputs(
            joint_force=jnp.asarray([1.5, 1.5]),
            robot_weight=jnp.asarray(1.0),
        ),
    )
    terminal_reward, terminal = compute_locomotion_reward(
        config,
        reward_inputs(
            joint_force=jnp.asarray([1.5, 1.5]),
            robot_weight=jnp.asarray(1.0),
            done=jnp.asarray(True),
        ),
    )

    assert healthy["reward/joint_force_peak"] == pytest.approx(0.0)
    assert healthy["reward/stand_still"] == pytest.approx(0.02)
    assert healthy_reward > 0.0
    assert terminal["reward/stand_still"] == pytest.approx(0.0)
    assert terminal_reward < 0.0


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
        requested_actuator_force=jnp.asarray([1.0, 2.0]),
        joint_force=jnp.asarray([1.0, 2.0]),
        foot_contact=jnp.asarray([1.0, 0.0]),
        foot_touchdown=jnp.asarray([True, False]),
        foot_normal_impulse=jnp.asarray([0.04, 0.0]),
        foot_peak_normal_force=jnp.asarray([2.0, 0.0]),
        touchdown_foot_linear_velocity=jnp.asarray([[0.0, 0.0, -0.3], [0.0, 0.0, 0.0]]),
        gait_active=jnp.asarray(True),
        desired_foot_contact=jnp.asarray([0.0, 1.0]),
        foot_height=jnp.asarray([0.0, 0.0]),
        foot_linear_velocity=jnp.asarray([[0.3, 0.1, 0.2], [0.2, 0.4, 0.1]]),
        foot_angular_speed=jnp.asarray([1.0, 1.0]),
        undesired_contact=jnp.asarray(1.0),
        self_collision=jnp.asarray(1.0),
        done=jnp.asarray(True),
    )
    standing_inputs = moving_inputs.replace(
        command=jnp.zeros(3),
        gait_active=jnp.asarray(False),
        foot_contact=jnp.asarray([1.0, 0.0]),
        foot_heading_error=jnp.asarray([0.2, -0.1]),
        done=jnp.asarray(False),
    )
    compute = jax.jit(compute_locomotion_reward, static_argnums=0)
    reward, moving_terms = compute(config, moving_inputs)
    _, moving_fine = compute(
        config,
        moving_inputs.replace(
            foot_normal_impulse=jnp.asarray([0.01, 0.0]),
            control_timestep=jnp.asarray(0.005),
        ),
    )
    _, standing_terms = compute(config, standing_inputs)
    _, standing_fine = compute(
        config,
        standing_inputs.replace(
            foot_normal_impulse=jnp.asarray([0.01, 0.0]),
            control_timestep=jnp.asarray(0.005),
        ),
    )
    heading_inputs = standing_inputs.replace(foot_contact=jnp.ones(2))
    _, heading_terms = compute(config, heading_inputs)
    _, heading_fine = compute(
        config,
        heading_inputs.replace(
            foot_normal_impulse=jnp.asarray([0.01, 0.0]),
            control_timestep=jnp.asarray(0.005),
        ),
    )
    _, airborne_terms = compute(
        config,
        moving_inputs.replace(
            foot_contact=jnp.zeros(2),
        ),
    )
    _, airborne_fine = compute(
        config,
        moving_inputs.replace(
            foot_contact=jnp.zeros(2),
            foot_normal_impulse=jnp.asarray([0.01, 0.0]),
            control_timestep=jnp.asarray(0.005),
        ),
    )
    _, healthy_terms = compute(config, moving_inputs.replace(done=jnp.asarray(False)))
    _, healthy_fine = compute(
        config,
        moving_inputs.replace(
            done=jnp.asarray(False),
            foot_normal_impulse=jnp.asarray([0.01, 0.0]),
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
            coarse = heading_terms[name]
            fine = heading_fine[name]
        if coarse == 0.0:
            coarse = airborne_terms[name]
            fine = airborne_fine[name]
        if coarse == 0.0:
            coarse = healthy_terms[name]
            fine = healthy_fine[name]
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
    with pytest.raises(ValueError, match="ang_vel_z_lpf_cutoff_hz"):
        RewardConfig(tracking=TrackingRewardConfig(ang_vel_z_lpf_cutoff_hz=0.0))
    with pytest.raises(ValueError, match="foot_angular_speed_scale"):
        RewardConfig(
            standing=StandingRewardConfig(foot_angular_speed_scale=float("nan"))
        )
    with pytest.raises(ValueError, match="foot_angular_weight"):
        RewardConfig(standing=StandingRewardConfig(foot_angular_weight=-1.0))
    with pytest.raises(ValueError, match="foot_heading_deadband"):
        RewardConfig(standing=StandingRewardConfig(foot_heading_deadband=-1.0))
    with pytest.raises(ValueError, match="foot_heading_scale"):
        RewardConfig(standing=StandingRewardConfig(foot_heading_scale=0.0))
    with pytest.raises(ValueError, match="joint_force_soft_limit_body_weights"):
        RewardConfig(gait=GaitRewardConfig(joint_force_soft_limit_body_weights=0.0))
    with pytest.raises(ValueError, match="joint_force_top_k"):
        RewardConfig(gait=GaitRewardConfig(joint_force_top_k=0))
    with pytest.raises(ValueError, match="touchdown_force_soft_limit_body_weights"):
        RewardConfig(gait=GaitRewardConfig(touchdown_force_soft_limit_body_weights=0.0))
    with pytest.raises(ValueError, match="touchdown_vertical_speed_scale"):
        RewardConfig(gait=GaitRewardConfig(touchdown_vertical_speed_scale=0.0))
    with pytest.raises(ValueError, match="swing_clearance_target"):
        RewardConfig(gait=GaitRewardConfig(swing_clearance_target=0.0))
    with pytest.raises(ValueError, match="stance_slip_angular_speed_scale"):
        RewardConfig(gait=GaitRewardConfig(stance_slip_angular_speed_scale=-1.0))


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
