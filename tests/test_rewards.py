from dataclasses import fields, replace

import jax
import jax.numpy as jnp
import pytest

from bello_mujoco.envs.rewards import (
    ActionLimitRewardConfig,
    CommandConfig,
    ContactTimingRewardConfig,
    FootPhaseRewardConfig,
    HipDeviationRewardConfig,
    InvalidActionRewardConfig,
    JointLimitRewardConfig,
    RewardConfig,
    RewardInputs,
    StandingFootAirTimeRewardConfig,
    StandingFootVelRewardConfig,
    TouchdownAirTimeRewardConfig,
    TrackLinVelXYRewardConfig,
    compute_locomotion_reward,
)


def reward_config(**terms) -> RewardConfig:
    base = RewardConfig()
    disabled = {
        field.name: replace(getattr(base, field.name), weight=0.0)
        for field in fields(base)
    }
    disabled.update(terms)
    return RewardConfig(**disabled)


def reward_inputs(**updates) -> RewardInputs:
    inputs = RewardInputs(
        command=jnp.zeros(3, dtype=jnp.float32),
        lin_vel_xy=jnp.zeros(2, dtype=jnp.float32),
        yaw_vel_z=jnp.asarray(0.0, dtype=jnp.float32),
        lin_vel_z=jnp.asarray(0.0, dtype=jnp.float32),
        ang_vel_xy=jnp.zeros(2, dtype=jnp.float32),
        projected_gravity=jnp.asarray([0.0, 0.0, -1.0], dtype=jnp.float32),
        joint_pos=jnp.zeros(2, dtype=jnp.float32),
        joint_pos_rel=jnp.zeros(2, dtype=jnp.float32),
        action=jnp.zeros(2, dtype=jnp.float32),
        requested_action=jnp.zeros(2, dtype=jnp.float32),
        last_action=jnp.zeros(2, dtype=jnp.float32),
        actuator_force=jnp.zeros(2, dtype=jnp.float32),
        joint_limit_low=jnp.asarray([-1.0, -1.0], dtype=jnp.float32),
        joint_limit_high=jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        policy_joint_indices=jnp.asarray([0, 1], dtype=jnp.int32),
        action_target_default=jnp.zeros(2, dtype=jnp.float32),
        action_target_low=jnp.asarray([-1.0, -1.0], dtype=jnp.float32),
        action_target_high=jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        foot_contact=jnp.ones(2, dtype=jnp.float32),
        foot_height=jnp.zeros(2, dtype=jnp.float32),
        foot_lateral_speed_sq=jnp.zeros(2, dtype=jnp.float32),
        foot_angular_speed=jnp.zeros(2, dtype=jnp.float32),
        foot_air_time=jnp.zeros(2, dtype=jnp.float32),
        foot_contact_time=jnp.ones(2, dtype=jnp.float32),
        last_foot_contact=jnp.ones(2, dtype=bool),
        foot_phase_height_target=jnp.zeros(2, dtype=jnp.float32),
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


def test_tracking_reward_is_maximal_at_command_and_scales_with_timestep() -> None:
    config = reward_config(
        track_lin_vel_xy=TrackLinVelXYRewardConfig(weight=2.0, sigma=0.5)
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


def test_joint_limit_penalty_increases_with_distance_past_soft_limit() -> None:
    config = reward_config(
        joint_limit=JointLimitRewardConfig(weight=-1.0, soft_limit_factor=0.9)
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
        joint_limit=JointLimitRewardConfig(weight=-1.0, soft_limit_factor=0.9)
    )
    reward, terms = compute_locomotion_reward(
        config, reward_inputs(joint_pos=jnp.asarray([1.2, 0.0]))
    )

    assert reward == pytest.approx(terms["reward/joint_limit"])
    assert reward < 0.0


def test_termination_is_a_one_shot_unscaled_penalty() -> None:
    base = RewardConfig()
    config = reward_config(termination=replace(base.termination, weight=-2.0))

    first, first_terms = compute_locomotion_reward(
        config,
        reward_inputs(done=jnp.asarray(True), control_timestep=jnp.asarray(0.02)),
    )
    second, second_terms = compute_locomotion_reward(
        config,
        reward_inputs(done=jnp.asarray(True), control_timestep=jnp.asarray(0.005)),
    )

    assert first_terms["reward/termination"] == pytest.approx(-2.0)
    assert second_terms["reward/termination"] == pytest.approx(-2.0)
    assert first == pytest.approx(second)


def test_hip_deviation_uses_lateral_command_axis() -> None:
    config = reward_config(
        hip_deviation=HipDeviationRewardConfig(
            weight=-1.0, lateral_command_threshold=0.1
        )
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
        action_limit=ActionLimitRewardConfig(weight=-1.0, soft_limit_ratio=0.8)
    )
    _, boundary = compute_locomotion_reward(
        config, reward_inputs(action=jnp.asarray([0.8, 0.0]))
    )
    _, outside = compute_locomotion_reward(
        config, reward_inputs(action=jnp.asarray([0.9, 0.0]))
    )
    assert boundary["reward/action_limit"] == pytest.approx(0.0)
    assert outside["reward/action_limit"] == pytest.approx(-0.0025)


def test_invalid_action_checks_requested_target_and_policy_clip() -> None:
    config = reward_config(invalid_action=InvalidActionRewardConfig(weight=-1.0))
    _, terms = compute_locomotion_reward(
        config,
        reward_inputs(
            requested_action=jnp.asarray([1.5, 0.0]),
            action=jnp.asarray([1.0, 0.0]),
        ),
    )
    assert terms["reward/invalid_action"] == pytest.approx(-0.01)


def test_contact_timing_is_an_unscaled_event_penalty() -> None:
    config = reward_config(
        contact_timing=ContactTimingRewardConfig(
            weight=-0.1,
            min_air_time=0.2,
            min_contact_time=0.3,
            command_speed_threshold=0.1,
        )
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


def test_pure_yaw_command_enables_stepping_event_rewards() -> None:
    config = reward_config(
        contact_timing=ContactTimingRewardConfig(
            weight=-0.1,
            min_air_time=0.2,
            min_contact_time=0.3,
            command_speed_threshold=0.1,
        ),
        touchdown_air_time=TouchdownAirTimeRewardConfig(
            weight=2.0,
            min_air_time=0.2,
            max_air_time=0.5,
            command_speed_threshold=0.1,
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


def test_touchdown_air_time_is_gated_for_standing_commands() -> None:
    config = reward_config(
        touchdown_air_time=TouchdownAirTimeRewardConfig(
            weight=2.0,
            min_air_time=0.2,
            max_air_time=0.5,
            command_speed_threshold=0.1,
        )
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
    _, finer_timestep = compute_locomotion_reward(
        config,
        inputs.replace(
            command=jnp.asarray([0.5, 0.0, 0.0]),
            control_timestep=jnp.asarray(0.005),
        ),
    )
    assert standing["reward/touchdown_air_time"] == pytest.approx(0.0)
    assert moving["reward/touchdown_air_time"] == pytest.approx(0.2)
    assert finer_timestep["reward/touchdown_air_time"] == pytest.approx(
        moving["reward/touchdown_air_time"]
    )


def test_phase_and_standing_penalties_use_mutually_relevant_features() -> None:
    config = reward_config(
        foot_phase=FootPhaseRewardConfig(weight=1.0, sigma=0.01),
        standing_foot_vel=StandingFootVelRewardConfig(
            weight=-2.0, command_speed_threshold=0.1
        ),
        standing_foot_air_time=StandingFootAirTimeRewardConfig(
            weight=-3.0, command_speed_threshold=0.1
        ),
    )
    _, terms = compute_locomotion_reward(
        config,
        reward_inputs(
            foot_lateral_speed_sq=jnp.asarray([0.04, 0.0]),
            foot_air_time=jnp.asarray([0.2, 0.0]),
        ),
    )
    assert terms["reward/foot_phase"] == pytest.approx(0.02)
    assert terms["reward/standing_foot_vel"] == pytest.approx(-0.0008)
    assert terms["reward/standing_foot_air_time"] == pytest.approx(-0.006)


def test_full_reward_is_jittable_and_finite() -> None:
    reward, terms = jax.jit(compute_locomotion_reward, static_argnums=0)(
        RewardConfig(), reward_inputs()
    )
    assert jnp.isfinite(reward)
    assert all(bool(jnp.isfinite(term)) for term in terms.values())
    assert set(terms) == {f"reward/{field.name}" for field in fields(RewardConfig())}


def test_reward_validation_rejects_invalid_safety_ranges() -> None:
    with pytest.raises(ValueError, match="soft_limit_ratio"):
        reward_config(
            action_limit=ActionLimitRewardConfig(weight=-1.0, soft_limit_ratio=1.0)
        )
    with pytest.raises(ValueError, match="soft_limit_factor"):
        reward_config(
            joint_limit=JointLimitRewardConfig(weight=-1.0, soft_limit_factor=0.0)
        )


@pytest.mark.parametrize(
    "term",
    [
        ContactTimingRewardConfig(min_air_time=float("nan")),
        ContactTimingRewardConfig(min_air_time=float("inf")),
        ContactTimingRewardConfig(min_contact_time=float("nan")),
        ContactTimingRewardConfig(min_contact_time=float("inf")),
    ],
)
def test_contact_timing_validation_rejects_nonfinite_bounds(term) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        reward_config(contact_timing=term)


@pytest.mark.parametrize(
    "term",
    [
        TouchdownAirTimeRewardConfig(min_air_time=float("nan")),
        TouchdownAirTimeRewardConfig(min_air_time=float("inf")),
        TouchdownAirTimeRewardConfig(max_air_time=float("nan")),
        TouchdownAirTimeRewardConfig(max_air_time=float("inf")),
    ],
)
def test_touchdown_air_time_validation_rejects_nonfinite_bounds(term) -> None:
    with pytest.raises(ValueError, match="finite"):
        reward_config(touchdown_air_time=term)
