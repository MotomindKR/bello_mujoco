from dataclasses import fields
import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import mujoco
import pytest
import numpy as np

from bello_mujoco.envs.locomotion import BelloLocomotionEnv
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
    validate_reward_config,
)


def test_standing_alignment_deadband_support_and_blend():
    config = reward_config(
        stand_foot_foreaft=-4.0,
        stand_foot_width=-2.0,
        standing=StandingRewardConfig(),
    )
    inputs = reward_inputs(
        foot_separation=jnp.asarray(0.22), nominal_stance_width=jnp.asarray(0.22)
    )
    compute = jax.jit(lambda inputs: compute_locomotion_reward(config, inputs)[0])
    assert float(compute(inputs)) == 0.0
    assert float(compute(inputs.replace(foot_foreaft=jnp.asarray(0.019)))) == 0.0
    misaligned = inputs.replace(foot_foreaft=jnp.asarray(0.08))
    assert float(compute(misaligned)) < 0.0
    assert (
        float(compute(misaligned.replace(standing_reward_blend=jnp.asarray(0.0))))
        == 0.0
    )
    assert (
        float(compute(misaligned.replace(foot_contact=jnp.asarray([1.0, 0.0])))) == 0.0
    )
    assert float(
        compute(misaligned.replace(lin_vel_xy=jnp.asarray([0.6, 0.0])))
    ) == pytest.approx(float(compute(misaligned)))


def test_crossing_uses_signed_separation_in_every_mode():
    config = reward_config(foot_crossing=-8.0)
    for moving in (False, True):
        inputs = reward_inputs(
            foot_lateral_clearance=jnp.asarray(0.1),
            foot_separation=jnp.asarray(0.22),
            gait_active=jnp.asarray(moving),
        )
        normal = compute_locomotion_reward(config, inputs)[0]
        crossed = compute_locomotion_reward(
            config, inputs.replace(foot_lateral_clearance=jnp.asarray(-0.34))
        )[0]
        assert float(normal) == 0.0
    assert float(crossed) < -1.0


def test_current_sole_geometry_reaches_crossing_reward_under_jit_and_vmap():
    config = reward_config(foot_crossing=-8.0)
    positions = jnp.asarray([[0, 0.07, 0], [0, -0.07, 0]])
    env = SimpleNamespace(
        config=SimpleNamespace(gait=SimpleNamespace(stance_fraction=0.6)),
        reward_config=config,
        model=SimpleNamespace(
            foot_body_ids=np.asarray([0, 1]),
            foot_geom_ids=np.asarray([0, 1]),
            geom_body_ids=np.asarray([0, 1]),
            robot=SimpleNamespace(locomotion_forward_axis=(1.0, 0.0, 0.0)),
            mj_model=SimpleNamespace(
                geom_size=np.asarray([[0.12, 0.04, 0.015]] * 2),
                geom_type=np.full(2, int(mujoco.mjtGeom.mjGEOM_BOX)),
            ),
        ),
        _sole_reward_geometry=SimpleNamespace(
            centers=np.zeros((2, 3)), corners=np.zeros((2, 4, 3)), nominal_width=0.14
        ),
        _reward_frame_link=lambda _: "root",
        _link_quat=lambda *_: jnp.asarray([1.0, 0, 0, 0]),
        _locomotion_right_axis=lambda: jnp.asarray([0.0, -1.0, 0.0]),
        _terrain_height=lambda *_: 0.0,
    )

    def evaluate(angle):
        c, s = jnp.cos(angle), jnp.sin(angle)
        rotation = jnp.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        quats = jnp.asarray(
            [
                [jnp.cos(angle / 2), 0, 0, jnp.sin(angle / 2)],
                [jnp.cos(angle / 2), 0, 0, -jnp.sin(angle / 2)],
            ]
        )
        state = SimpleNamespace(
            gait_phase=jnp.asarray(0.0),
            data=SimpleNamespace(
                xpos=positions,
                xquat=quats,
                geom_xpos=positions,
                geom_xmat=jnp.stack([rotation, rotation.T]),
            ),
        )
        placement = BelloLocomotionEnv._foot_placement_inputs(env, state)
        reward, _ = compute_locomotion_reward(config, reward_inputs(**placement))
        return placement["foot_separation"], placement["foot_lateral_clearance"], reward

    separation, gap, reward = jax.jit(jax.vmap(evaluate))(
        jnp.asarray([0.0, jnp.pi / 6])
    )
    np.testing.assert_allclose(separation, [0.14, 0.14], atol=1e-6)
    assert gap[0] > 0 and gap[1] < 0
    assert reward[0] == 0 and reward[1] < 0


def test_continuous_alignment_has_no_touchdown_cliff_and_cannot_be_disabled():
    config = reward_config(
        stand_foot_foreaft=-0.25,
        flat_orientation=-5.0,
        standing=StandingRewardConfig(
            position_scale=0.05, contact_independent_alignment=True
        ),
    )
    inputs = reward_inputs(
        base_height=jnp.asarray(0.87),
        foot_foreaft=jnp.asarray(0.18),
        foot_separation=jnp.asarray(0.22),
        nominal_stance_width=jnp.asarray(0.22),
        foot_lateral_clearance=jnp.asarray(0.1),
        swing_progress=jnp.zeros(2),
        foot_clearance_samples=jnp.zeros((2, 4)),
    )
    compute = jax.jit(lambda x: compute_locomotion_reward(config, x))
    two, terms = compute(inputs)
    one, one_terms = compute(inputs.replace(foot_contact=jnp.asarray([True, False])))
    assert float(two) == pytest.approx(float(one))
    assert float(terms["reward/stand_foot_foreaft"]) == pytest.approx(
        float(one_terms["reward/stand_foot_foreaft"])
    )
    assert float(terms["reward/stand_foot_foreaft"]) == pytest.approx(-0.0135)
    tilt = math.radians(18)
    leaned, lean_terms = compute(
        inputs.replace(
            projected_gravity=jnp.asarray([math.sin(tilt), 0, -math.cos(tilt)])
        )
    )
    assert float(leaned) < float(two)
    assert float(lean_terms["reward/stand_foot_foreaft"]) < 0
    _, fast_terms = compute(inputs.replace(lin_vel_xy=jnp.asarray([0.6, 0])))
    assert float(fast_terms["reward/stand_foot_foreaft"]) < 0
    aligned, _ = compute(inputs.replace(foot_foreaft=jnp.asarray(0.0)))
    assert float(aligned) > float(two)
    # Returning a recovery foot progressively reduces cost, including before contact.
    costs = []
    for stagger in (0.18, 0.10, 0.05, 0.02):
        _, result = compute(
            inputs.replace(
                foot_contact=jnp.asarray([True, False]),
                foot_foreaft=jnp.asarray(stagger),
            )
        )
        costs.append(float(result["reward/stand_foot_foreaft"]))
    assert costs == sorted(costs)
    assert costs[-1] == 0


def test_clearance_peaks_mid_swing_and_checks_each_sole_corner():
    config = reward_config(
        gait_swing_clearance=-1.0, gait=GaitRewardConfig(phase_shaped_clearance=True)
    )
    inputs = reward_inputs(
        gait_active=jnp.asarray(True),
        desired_foot_contact=jnp.asarray([0.0, 1.0]),
        swing_progress=jnp.asarray([0.5, 0.0]),
        foot_clearance_samples=jnp.full((2, 4), 0.08),
    )
    assert float(compute_locomotion_reward(config, inputs)[0]) == 0.0
    toe_low = inputs.replace(
        foot_clearance_samples=inputs.foot_clearance_samples.at[0, 0].set(0.0)
    )
    assert float(compute_locomotion_reward(config, toe_low)[0]) < 0.0
    landing = toe_low.replace(swing_progress=jnp.asarray([1.0, 0.0]))
    assert float(compute_locomotion_reward(config, landing)[0]) == pytest.approx(
        0.0, abs=1e-8
    )


@pytest.mark.parametrize("moving", [False, True])
def test_shared_heading_and_contact_slip_ignore_gait_and_support_gates(moving):
    config = reward_config(foot_heading=-1.0, foot_slip=-0.5)
    inputs = reward_inputs(
        gait_active=jnp.asarray(moving),
        command=jnp.asarray([0.5 if moving else 0.0, 0.0, 0.0]),
        standing_reward_blend=jnp.asarray(0.0),
        lin_vel_xy=jnp.asarray([0.6, 0.0]),
        projected_gravity=jnp.asarray([0.5, 0.0, -0.866]),
        foot_contact=jnp.asarray([True, False]),
        desired_foot_contact=jnp.asarray([False, True]),
        foot_heading_error=jnp.asarray([0.3, -0.3]),
        foot_linear_velocity=jnp.asarray([[0.2, 0.0, 3.0], [2.0, 0.0, 0.0]]),
    )
    compute = jax.jit(lambda x: compute_locomotion_reward(config, x)[1])
    terms = compute(inputs)
    assert float(terms["reward/foot_heading"]) == pytest.approx(-0.0018)
    assert float(terms["reward/foot_slip"]) == pytest.approx(-0.0004)
    airborne = compute(inputs.replace(foot_contact=jnp.zeros(2, dtype=bool)))
    assert float(airborne["reward/foot_slip"]) == 0
    assert airborne["reward/foot_heading"] == terms["reward/foot_heading"]


def test_alignment_keeps_linear_tail_instead_of_saturating():
    config = reward_config(
        stand_foot_foreaft=-0.25,
        standing=StandingRewardConfig(
            position_scale=0.05, contact_independent_alignment=True
        ),
    )
    compute = jax.jit(lambda x: compute_locomotion_reward(config, x)[0])
    costs = [
        float(
            compute(
                reward_inputs(
                    foot_foreaft=jnp.asarray(distance),
                    foot_contact=jnp.asarray([True, False]),
                )
            )
        )
        for distance in (0.1, 0.2, 0.3)
    ]
    assert costs[1] - costs[0] == pytest.approx(-0.01)
    assert costs[2] - costs[1] == pytest.approx(-0.01)


@pytest.mark.parametrize(
    "weights",
    [
        (("a", -1.0),),
        (("a", float("nan")),),
        (("a", 1.0), ("a", 2.0)),
        (("", 1.0),),
    ],
)
def test_invalid_explicit_posture_weights_are_rejected(weights):
    with pytest.raises(ValueError, match="joint_pose_weights"):
        validate_reward_config(
            reward_config(
                limits_pose=LimitsPoseRewardConfig(joint_pose_weights=weights)
            )
        )


@pytest.mark.parametrize("names", [("b", "a"), ("a",), ("a", "b", "held")])
def test_explicit_pose_weights_map_by_joint_name_and_require_policy_coverage(names):
    weights = tuple((name, float(index + 1)) for index, name in enumerate(names))
    env = SimpleNamespace(
        model=SimpleNamespace(
            mj_model=SimpleNamespace(nu=3),
            actuator_joint_names=("a", "held", "b"),
            policy_ctrl_indices=np.asarray([2, 0]),
        ),
        reward_config=reward_config(
            limits_pose=LimitsPoseRewardConfig(joint_pose_weights=weights)
        ),
    )
    if set(names) != {"a", "b"}:
        with pytest.raises(ValueError, match="every policy joint exactly once"):
            BelloLocomotionEnv._joint_pose_weights(env)
    else:
        np.testing.assert_array_equal(
            BelloLocomotionEnv._joint_pose_weights(env), [2.0, 0.0, 1.0]
        )


def test_stumble_grace_windows_and_contact_load():
    config = reward_config(foot_stumble=-2.0)
    inputs = reward_inputs(
        gait_active=jnp.asarray(True),
        desired_foot_contact=jnp.asarray([0.0, 1.0]),
        swing_progress=jnp.asarray([0.5, 0.0]),
        foot_peak_normal_force=jnp.asarray([0.3, 1.0]),
    )
    assert float(compute_locomotion_reward(config, inputs)[0]) < 0.0
    for phase in (0.05, 0.95):
        assert (
            float(
                compute_locomotion_reward(
                    config, inputs.replace(swing_progress=jnp.asarray([phase, 0.0]))
                )[0]
            )
            == 0.0
        )
    assert (
        float(
            compute_locomotion_reward(
                config, inputs.replace(foot_peak_normal_force=jnp.zeros(2))
            )[0]
        )
        == 0.0
    )


def test_standing_recovery_penalizes_loaded_sliding_not_unloaded_steps():
    config = reward_config(foot_stumble=-2.0, standing=StandingRewardConfig())
    inputs = reward_inputs(
        swing_progress=jnp.zeros(2),
        lin_vel_xy=jnp.asarray([0.6, 0.0]),
        foot_linear_velocity=jnp.asarray([[0.3, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        foot_peak_normal_force=jnp.asarray([0.3, 1.0]),
    )
    assert float(compute_locomotion_reward(config, inputs)[0]) < 0.0
    unloaded = inputs.replace(foot_peak_normal_force=jnp.asarray([0.0, 1.0]))
    assert float(compute_locomotion_reward(config, unloaded)[0]) == 0.0


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
        torso_orientation_error=jnp.asarray(0.0),
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
        last_action=jnp.zeros(2, dtype=jnp.float32),
        last_action_delta=jnp.zeros(2, dtype=jnp.float32),
        action_smoothness_weights=jnp.ones(2, dtype=jnp.float32),
        actuator_force=jnp.zeros(2, dtype=jnp.float32),
        requested_actuator_force=jnp.zeros(2, dtype=jnp.float32),
        actuator_peak_force=jnp.ones(2, dtype=jnp.float32),
        motor_continuous_torque=jnp.full(2, 20.0, dtype=jnp.float32),
        joint_force=jnp.zeros(2, dtype=jnp.float32),
        robot_weight=jnp.asarray(1.0, dtype=jnp.float32),
        joint_limit_low=jnp.asarray([-1.0, -1.0], dtype=jnp.float32),
        joint_limit_high=jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        policy_joint_indices=jnp.asarray([0, 1], dtype=jnp.int32),
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
        done=jnp.asarray(False),
        control_timestep=jnp.asarray(0.02, dtype=jnp.float32),
        action_clip_limit=jnp.asarray(1.0, dtype=jnp.float32),
    )
    return inputs.replace(**updates)


@pytest.mark.parametrize("error", [0.0, 0.25, 0.5])
def test_torso_reward_tracks_full_orientation_error(error):
    config = reward_config(track_torso_orientation=3.0)
    inputs = reward_inputs(torso_orientation_error=jnp.asarray(error))
    total, terms = compute_locomotion_reward(config, inputs)
    expected = (
        3.0
        * float(inputs.control_timestep)
        * np.exp(-((error / config.tracking.torso_orientation_std) ** 2))
    )
    assert float(terms["reward/track_torso_orientation"]) == pytest.approx(expected)
    assert float(total) == pytest.approx(expected)


def test_command_sampler_respects_ranges_and_exact_standing_probability() -> None:
    moving = CommandConfig(
        lin_vel_x=(0.2, 0.2),
        lin_vel_y=(-0.3, -0.3),
        ang_vel_yaw=(0.4, 0.4),
        standing_probability=0.0,
    ).sample(jax.random.PRNGKey(0), (4,))
    standing_config = CommandConfig(standing_probability=1.0)
    standing = standing_config.sample(jax.random.PRNGKey(1), (4,))
    assert moving == pytest.approx(jnp.tile(jnp.asarray([0.2, -0.3, 0.4]), (4, 1)))
    assert standing == pytest.approx(jnp.zeros((4, 3)))
    assert standing_config.sample_moving(jax.random.PRNGKey(2), (4,)) == pytest.approx(
        jnp.zeros((4, 3))
    )
    assert standing_config.sample_stratified_batch(
        jax.random.PRNGKey(3), 4
    ) == pytest.approx(jnp.zeros((4, 3)))


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


def test_moving_velocity_error_has_a_non_saturating_signed_gradient() -> None:
    config = reward_config(
        moving_lin_vel_xy_error=-1.0,
        tracking=TrackingRewardConfig(lin_vel_xy_error_delta=0.25),
    )

    def velocity_error_reward(
        velocity_x: jax.Array, command_x: float = 0.5
    ) -> jax.Array:
        _, terms = compute_locomotion_reward(
            config,
            reward_inputs(
                command=jnp.asarray([command_x, 0.0, 0.0]),
                lin_vel_xy=jnp.asarray([velocity_x, 0.0]),
            ),
        )
        return terms["reward/moving_lin_vel_xy_error"]

    assert velocity_error_reward(jnp.asarray(0.5)) == pytest.approx(0.0)
    assert velocity_error_reward(jnp.asarray(0.0)) == pytest.approx(-0.0075)
    assert velocity_error_reward(jnp.asarray(-0.5)) == pytest.approx(-0.0175)
    assert jax.grad(velocity_error_reward)(jnp.asarray(-0.5)) == pytest.approx(0.02)
    assert jax.grad(velocity_error_reward)(jnp.asarray(-5.0)) == pytest.approx(0.02)
    assert jax.grad(lambda velocity: velocity_error_reward(velocity, -0.5))(
        jnp.asarray(0.5)
    ) == pytest.approx(-0.02)

    _, standing_terms = compute_locomotion_reward(
        config,
        reward_inputs(
            command=jnp.zeros(3),
            lin_vel_xy=jnp.asarray([1.0, -1.0]),
        ),
    )
    assert standing_terms["reward/moving_lin_vel_xy_error"] == pytest.approx(0.0)


def test_moving_angular_velocity_error_has_a_non_saturating_signed_gradient() -> None:
    config = reward_config(
        moving_ang_vel_z_error=-1.0,
        tracking=TrackingRewardConfig(ang_vel_z_error_delta=0.25),
    )

    def angular_velocity_error_reward(
        velocity_z: jax.Array, command_z: float = 0.5
    ) -> jax.Array:
        _, terms = compute_locomotion_reward(
            config,
            reward_inputs(
                command=jnp.asarray([0.0, 0.0, command_z]),
                yaw_vel_z=velocity_z,
            ),
        )
        return terms["reward/moving_ang_vel_z_error"]

    assert angular_velocity_error_reward(jnp.asarray(0.5)) == pytest.approx(0.0)
    assert angular_velocity_error_reward(jnp.asarray(0.0)) == pytest.approx(-0.0075)
    assert angular_velocity_error_reward(jnp.asarray(-0.5)) == pytest.approx(-0.0175)
    assert jax.grad(angular_velocity_error_reward)(jnp.asarray(-0.5)) == pytest.approx(
        0.02
    )
    assert jax.grad(angular_velocity_error_reward)(jnp.asarray(-5.0)) == pytest.approx(
        0.02
    )
    assert jax.grad(lambda velocity: angular_velocity_error_reward(velocity, -0.5))(
        jnp.asarray(0.5)
    ) == pytest.approx(-0.02)

    _, standing_terms = compute_locomotion_reward(
        config,
        reward_inputs(
            command=jnp.zeros(3),
            yaw_vel_z=jnp.asarray(1.0),
        ),
    )
    assert standing_terms["reward/moving_ang_vel_z_error"] == pytest.approx(0.0)


def test_default_tracking_weights_are_doubled() -> None:
    config = RewardConfig()

    assert config.weights.track_lin_vel_xy == pytest.approx(2.0)
    assert config.weights.track_ang_vel_z == pytest.approx(2.0)


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


@pytest.mark.parametrize("error", [0.0, 0.25, 0.5])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_tracking_retains_reward_during_single_support_recovery(error, axis):
    config = reward_config(
        track_lin_vel_xy=2.0,
        track_ang_vel_z=2.0,
        tracking=TrackingRewardConfig(
            lin_vel_xy_still_std=0.5, ang_vel_z_still_std=0.5
        ),
    )
    velocity = jnp.zeros(3).at[axis].set(error)
    inputs = reward_inputs(
        lin_vel_xy=velocity[:2],
        yaw_vel_z=velocity[2],
        foot_contact=jnp.asarray([1.0, 0.0]),
        foot_clearance_samples=jnp.zeros((2, 4)),
        swing_progress=jnp.zeros(2),
    )
    compute = jax.jit(lambda value: compute_locomotion_reward(config, value)[1])
    terms = compute(inputs)
    key = "reward/track_lin_vel_xy" if axis < 2 else "reward/track_ang_vel_z"
    assert float(terms[key]) == pytest.approx(0.04 * math.exp(-(error**2) / 0.25))
    assert float(compute(inputs.replace(foot_contact=jnp.zeros(2)))[key]) == 0.0


def test_standing_tracking_remains_available_during_recovery_steps() -> None:
    config = reward_config(track_lin_vel_xy=2.0, track_ang_vel_z=2.0)
    inputs = reward_inputs(command=jnp.zeros(3))

    _, supported = compute_locomotion_reward(config, inputs)
    _, single_support = compute_locomotion_reward(
        config,
        inputs.replace(foot_contact=jnp.asarray([1.0, 0.0])),
    )
    _, moving = compute_locomotion_reward(
        config,
        inputs.replace(
            command=jnp.asarray([0.5, 0.0, 0.0]),
            foot_contact=jnp.asarray([1.0, 0.0]),
        ),
    )

    assert supported["reward/track_lin_vel_xy"] == pytest.approx(0.04)
    assert supported["reward/track_ang_vel_z"] == pytest.approx(0.04)
    assert single_support["reward/track_lin_vel_xy"] == pytest.approx(0.04)
    assert single_support["reward/track_ang_vel_z"] == pytest.approx(0.04)
    _, flight = compute_locomotion_reward(
        config, inputs.replace(foot_contact=jnp.zeros(2))
    )
    assert flight["reward/track_lin_vel_xy"] == 0.0
    assert flight["reward/track_ang_vel_z"] == 0.0
    assert moving["reward/track_lin_vel_xy"] > 0.0
    assert moving["reward/track_ang_vel_z"] > 0.0


@pytest.mark.parametrize("speed", [0.0, 0.375, 0.5])
def test_speed_cannot_relax_posture_thermal_or_safety_penalties(speed):
    config = reward_config(
        stand_motor_continuous_torque_excess=-8.0,
        stand_contact=-0.5,
        hip_deviation=-0.25,
        knee_deviation=-0.1,
        torque_peak=-12.0,
        termination=-100.0,
        standing=StandingRewardConfig(),
    )
    inputs = reward_inputs(
        actuator_force=jnp.asarray([15.0, 15.0]),
        requested_actuator_force=jnp.asarray([100.0, 100.0]),
        actuator_peak_force=jnp.asarray([20.0, 20.0]),
        joint_pos_rel=jnp.asarray([0.2, 0.2]),
        foot_contact=jnp.asarray([1.0, 0.0]),
    )
    compute = jax.jit(lambda value: compute_locomotion_reward(config, value)[1])
    normal = compute(inputs)
    recovery = compute(inputs.replace(lin_vel_xy=jnp.asarray([speed, 0.0])))
    for name in (
        "stand_motor_continuous_torque_excess",
        "stand_contact",
        "hip_deviation",
        "knee_deviation",
    ):
        assert recovery[f"reward/{name}"] == pytest.approx(
            float(normal[f"reward/{name}"])
        )
    assert recovery["reward/torque_peak"] == normal["reward/torque_peak"]
    assert compute(
        inputs.replace(done=jnp.asarray(True), lin_vel_xy=jnp.asarray([speed, 0.0]))
    )["reward/termination"] == pytest.approx(-2.0)


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


def test_hip_deviation_uses_configured_command_axes() -> None:
    config = reward_config(
        hip_deviation=-1.0,
        limits_pose=LimitsPoseRewardConfig(
            hip_deviation_command_indices=(0, 2),
            hip_deviation_command_threshold=0.1,
        ),
    )
    inputs = reward_inputs(joint_pos_rel=jnp.asarray([0.5, 0.0]))

    _, lateral = compute_locomotion_reward(
        config, inputs.replace(command=jnp.asarray([0.5, 0.0, 0.0]))
    )
    _, forward = compute_locomotion_reward(
        config, inputs.replace(command=jnp.asarray([0.0, 0.5, 0.0]))
    )
    _, yaw = compute_locomotion_reward(
        config, inputs.replace(command=jnp.asarray([0.0, 0.0, 0.5]))
    )
    _, standing = compute_locomotion_reward(config, inputs)

    assert forward["reward/hip_deviation"] == pytest.approx(0.0)
    assert lateral["reward/hip_deviation"] == pytest.approx(-0.01)
    assert yaw["reward/hip_deviation"] == pytest.approx(-0.01)
    assert standing["reward/hip_deviation"] == pytest.approx(-0.01)


@pytest.mark.parametrize("apply_while_moving", (False, True))
def test_knee_deviation_moving_gate_is_explicit(apply_while_moving: bool) -> None:
    config = reward_config(
        knee_deviation=-1.0,
        limits_pose=LimitsPoseRewardConfig(
            knee_deviation_apply_while_moving=apply_while_moving,
        ),
    )
    inputs = reward_inputs(
        joint_pos_rel=jnp.asarray([0.2, -0.3]),
        knee_joint_indices=jnp.asarray([0, 1], dtype=jnp.int32),
    )

    _, standing = compute_locomotion_reward(config, inputs)
    _, moving = compute_locomotion_reward(
        config,
        inputs.replace(
            command=jnp.asarray([0.0, 0.5, 0.0]),
            gait_active=jnp.asarray(True),
        ),
    )

    assert standing["reward/knee_deviation"] == pytest.approx(-0.01)
    assert moving["reward/knee_deviation"] == pytest.approx(
        -0.01 if apply_while_moving else 0.0
    )


def test_stand_upright_penalty_is_gated_by_standing_command() -> None:
    config = reward_config(stand_upright=-8.0)
    inputs = reward_inputs(
        projected_gravity=jnp.asarray([0.5, 0.0, -0.8660254], dtype=jnp.float32)
    )

    _, standing = compute_locomotion_reward(config, inputs)
    _, moving = compute_locomotion_reward(
        config, inputs.replace(command=jnp.asarray([0.0, 0.5, 0.0]))
    )

    assert standing["reward/stand_upright"] == pytest.approx(-0.04)
    assert moving["reward/stand_upright"] == pytest.approx(0.0)


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

    expected = -4.0 * 0.02 * math.tanh(1.0)
    assert terms["reward/touchdown_impact"] == pytest.approx(expected)
    assert no_touchdown["reward/touchdown_impact"] == pytest.approx(0.0)


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


@pytest.mark.parametrize("forward_speed", [-0.5, 0.0, 0.25, 0.5])
@pytest.mark.parametrize("tilt", [0.0, 0.4])
@pytest.mark.parametrize("requested_speed", [0.0, 0.5])
def test_recovery_cannot_remove_active_gait_contact_penalty(
    forward_speed: float, tilt: float, requested_speed: float
) -> None:
    config = reward_config(
        gait_contact=-2.5,
        standing=StandingRewardConfig(),
    )
    inputs = reward_inputs(
        command=jnp.asarray([requested_speed, 0.0, 0.0]),
        lin_vel_xy=jnp.asarray([forward_speed, 0.0]),
        projected_gravity=jnp.asarray([math.sin(tilt), 0.0, -math.cos(tilt)]),
        gait_active=jnp.asarray(True),
        desired_foot_contact=jnp.asarray([1.0, 0.0]),
        foot_contact=jnp.asarray([1.0, 1.0]),
    )
    compute = jax.jit(lambda value: compute_locomotion_reward(config, value)[1])
    # One incorrectly planted foot costs 2.5 * 0.5 * 0.02, even during a
    # zero-command extra-step stop buffer or large tracking error/tilt.
    assert compute(inputs)["reward/gait_contact"] == pytest.approx(-0.025)
    assert compute(inputs.replace(foot_contact=jnp.asarray([1.0, 0.0])))[
        "reward/gait_contact"
    ] == pytest.approx(0.0)
    assert compute(inputs.replace(gait_active=jnp.asarray(False)))[
        "reward/gait_contact"
    ] == pytest.approx(0.0)


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
    assert one_foot["reward/stand_still"] == pytest.approx(0.0)
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
        swing_progress=jnp.asarray([0.5, 0.0]),
        foot_foreaft=jnp.asarray(0.08),
        foot_separation=jnp.asarray(-0.2),
        nominal_stance_width=jnp.asarray(0.2),
        foot_lateral_clearance=jnp.asarray(-0.32),
        command=jnp.asarray([0.5, 0.5, 0.5]),
        lin_vel_z=jnp.asarray(1.0),
        ang_vel_xy=jnp.asarray([1.0, 1.0]),
        projected_gravity=jnp.asarray([0.2, 0.1, -0.9]),
        base_height=jnp.asarray(0.5),
        joint_pos=jnp.asarray([1.2, -1.2]),
        joint_pos_rel=jnp.asarray([0.5, 0.5]),
        action=jnp.asarray([0.9, 0.5]),
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
        done=jnp.asarray(True),
    )
    standing_inputs = moving_inputs.replace(
        command=jnp.zeros(3),
        actuator_force=jnp.asarray([30.0, 2.0]),
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
    heading_inputs = standing_inputs.replace(foot_contact=jnp.ones(2))
    _, heading_terms = compute(config, heading_inputs)
    _, heading_fine = compute(
        config, heading_inputs.replace(control_timestep=jnp.asarray(0.005))
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

    quiet = reward_inputs(swing_progress=jnp.zeros(2))
    _, quiet_terms = compute(config, quiet)
    _, quiet_fine = compute(config, quiet.replace(control_timestep=jnp.asarray(0.005)))
    contact = quiet.replace(self_collision=jnp.asarray(1.0))
    _, contact_terms = compute(config, contact)
    _, contact_fine = compute(
        config, contact.replace(control_timestep=jnp.asarray(0.005))
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
        if coarse == 0.0:
            coarse, fine = quiet_terms[name], quiet_fine[name]
        if coarse == 0.0:
            coarse, fine = contact_terms[name], contact_fine[name]
        assert coarse != 0.0, name
        assert coarse == pytest.approx(4.0 * fine), name


def test_reward_validation_rejects_invalid_safety_ranges() -> None:
    with pytest.raises(ValueError, match="soft_limit_ratio"):
        RewardConfig(limits_pose=LimitsPoseRewardConfig(action_soft_limit_ratio=1.0))
    with pytest.raises(ValueError, match="soft_limit_factor"):
        RewardConfig(limits_pose=LimitsPoseRewardConfig(joint_soft_limit_factor=0.0))
    with pytest.raises(ValueError, match="base_height_target"):
        RewardConfig(limits_pose=LimitsPoseRewardConfig(base_height_target=0.0))
    with pytest.raises(ValueError, match="hip_deviation_command_indices"):
        RewardConfig(
            limits_pose=LimitsPoseRewardConfig(
                hip_deviation_command_indices=(3,),
            )
        )
    with pytest.raises(TypeError, match="knee_deviation_apply_while_moving"):
        RewardConfig(
            limits_pose=LimitsPoseRewardConfig(
                knee_deviation_apply_while_moving=1,  # type: ignore[arg-type]
            )
        )
    with pytest.raises(ValueError, match="foot_linear_speed_scale"):
        RewardConfig(standing=StandingRewardConfig(foot_linear_speed_scale=0.0))
    with pytest.raises(ValueError, match="ang_vel_z_lpf_cutoff_hz"):
        RewardConfig(tracking=TrackingRewardConfig(ang_vel_z_lpf_cutoff_hz=0.0))
    with pytest.raises(ValueError, match="lin_vel_xy_error_delta"):
        RewardConfig(tracking=TrackingRewardConfig(lin_vel_xy_error_delta=0.0))
    with pytest.raises(ValueError, match="ang_vel_z_error_delta"):
        RewardConfig(tracking=TrackingRewardConfig(ang_vel_z_error_delta=0.0))
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


def test_standing_motor_penalty_covers_held_motors_and_half_rated_thresholds():
    config = reward_config(
        stand_motor_continuous_torque_excess=-8.0,
        standing=StandingRewardConfig(),
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


def test_body_stillness_rejects_rocking_with_stationary_feet_and_self_support():
    config = reward_config(stand_body_still=2.0, stand_self_contact=-4.0)
    quiet = reward_inputs()
    compute = jax.jit(lambda inputs: compute_locomotion_reward(config, inputs))
    quiet_reward, _ = compute(quiet)
    assert float(quiet_reward) == pytest.approx(0.04)
    rocking = quiet.replace(ang_vel_xy=jnp.asarray([0.0, 0.5]))
    moving = quiet.replace(lin_vel_xy=jnp.asarray([0.1, 0.0]))
    assert float(compute(rocking)[0]) < 0.001
    assert float(compute(moving)[0]) < 0.001
    supported, terms = compute(quiet.replace(self_collision=jnp.asarray(2.0)))
    assert float(terms["reward/stand_body_still"]) == 0
    assert float(supported) == pytest.approx(-0.16)
    walking = quiet.replace(
        command=jnp.asarray([0.3, 0, 0]), gait_active=jnp.asarray(True)
    )
    assert float(compute(walking)[0]) == 0
    batched = jax.tree.map(lambda *xs: jnp.stack(xs), quiet, rocking, moving)
    rewards = jax.jit(jax.vmap(lambda x: compute_locomotion_reward(config, x)[0]))(
        batched
    )
    np.testing.assert_allclose(
        rewards, [float(compute(x)[0]) for x in (quiet, rocking, moving)], atol=1e-7
    )


@pytest.mark.parametrize(
    "field", ["body_linear_speed_scale", "body_angular_speed_scale"]
)
def test_body_stillness_scales_must_be_positive(field):
    with pytest.raises(ValueError, match=field):
        validate_reward_config(
            reward_config(standing=StandingRewardConfig(**{field: 0.0}))
        )
