from dataclasses import replace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from bello_mujoco.assets_pipeline.prepare import PreparedAsset
from bello_mujoco.envs.locomotion import (
    BadOrientationDoneConfig,
    BaseHeightDoneConfig,
    BelloEnvConfig,
    BelloLocomotionEnv,
    ContactConfig,
    DoneConfig,
    GaitConfig,
    ObservationConfig,
    ObservationNoiseConfig,
    ObservationScaleConfig,
    RandomPushConfig,
    SpawnConfig,
    TimeoutDoneConfig,
)
from bello_mujoco.envs.rewards import (
    AngVelXYRewardConfig,
    BaseHeightRewardConfig,
    CommandCurriculumConfig,
    CommandConfig,
    FeetAirTimeRewardConfig,
    FlatOrientationRewardConfig,
    FootClearanceRewardConfig,
    FootContactRewardConfig,
    FootDistanceRewardConfig,
    FootSlipRewardConfig,
    GaitContactRewardConfig,
    LinVelZRewardConfig,
    RewardConfig,
    TerminationRewardConfig,
    TrackAngVelZRewardConfig,
    TrackLinVelXYRewardConfig,
)
from bello_mujoco.sim import WarpConfig
from bello_mujoco.terrain import TerrainConfig


def root_quat_from_roll_pitch(roll: float, pitch: float) -> jnp.ndarray:
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    return jnp.asarray([cr * cp, sr * cp, cr * sp, -sr * sp], dtype=jnp.float32)


def test_reset_shapes(full_body_asset: PreparedAsset) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)
    state, obs = env.reset()

    assert env.model.mj_model.nu == 35
    assert env.action_size == 35
    assert (
        env.observation_size
        == 6
        + env.command_observation_size
        + 2 * env.model.mj_model.nu
        + env.model.mj_model.nu
        + env.action_size
    )
    assert obs.shape == (env.observation_size,)
    assert state.last_action.shape == (env.action_size,)
    assert state.foot_air_time.shape == (2,)
    assert state.foot_contact_debounce_time.shape == (2,)
    assert state.last_foot_contact.shape == (2,)
    assert int(state.step_count) == 0


def test_model_exposes_contact_reward_sets(full_body_asset: PreparedAsset) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)

    assert env.model.foot_body_ids.shape == (2,)
    assert env.model.foot_geom_ids.shape == (10,)
    assert env.model.terrain_geom_ids.shape == (1,)
    assert env.model.robot_geom_ids.size >= env.model.foot_geom_ids.size
    assert env.model.nonfoot_robot_geom_ids.size > env.model.foot_geom_ids.size


def test_locomotion_env_rejects_missing_reward_body_reference(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)
    bad_model = replace(
        env.model,
        foot_body_ids=np.asarray([1, env.model.mj_model.nbody], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="foot_body_ids contains out-of-range"):
        BelloLocomotionEnv(bad_model, config=env.config)


def test_locomotion_env_rejects_policy_actions_for_held_actuators(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)
    bad_model = replace(
        env.model,
        policy_ctrl_indices=np.asarray([0, 1], dtype=np.int32),
        hold_ctrl_indices=np.asarray([1], dtype=np.int32),
        policy_qpos_indices=env.model.policy_qpos_indices[:2],
        policy_qvel_indices=env.model.policy_qvel_indices[:2],
        policy_action_scale=env.model.policy_action_scale[:2],
    )

    with pytest.raises(ValueError, match="must not reference held actuator controls"):
        BelloLocomotionEnv(bad_model, config=env.config)


def test_locomotion_env_rejects_enabled_missing_reward_link(
    full_body_asset: PreparedAsset,
) -> None:
    reward_config = RewardConfig(
        track_lin_vel_xy=TrackLinVelXYRewardConfig(link="missing_link")
    )

    with pytest.raises(ValueError, match="configured link/body not found"):
        BelloLocomotionEnv.load_prepared(full_body_asset, reward_config=reward_config)


def test_locomotion_env_rejects_enabled_missing_reward_frame_link(
    full_body_asset: PreparedAsset,
) -> None:
    reward_config = RewardConfig(
        track_lin_vel_xy=TrackLinVelXYRewardConfig(
            link="pelvis_link", frame_link="missing_link"
        )
    )

    with pytest.raises(ValueError, match="configured link/body not found"):
        BelloLocomotionEnv.load_prepared(full_body_asset, reward_config=reward_config)


def test_locomotion_env_rejects_missing_root_link(
    full_body_asset: PreparedAsset,
) -> None:
    with pytest.raises(ValueError, match="configured link/body not found"):
        BelloLocomotionEnv.load_prepared(
            full_body_asset,
            env_config=BelloEnvConfig(root_link="missing_link"),
        )


def test_locomotion_env_rejects_negative_contact_debounce(
    full_body_asset: PreparedAsset,
) -> None:
    with pytest.raises(
        ValueError, match="contact.foot_contact_on_seconds must be finite"
    ):
        BelloLocomotionEnv.load_prepared(
            full_body_asset,
            env_config=BelloEnvConfig(
                contact=ContactConfig(foot_contact_on_seconds=-0.01)
            ),
        )


def test_locomotion_env_rejects_negative_foot_contact_force_threshold(
    full_body_asset: PreparedAsset,
) -> None:
    with pytest.raises(
        ValueError, match="contact.foot_contact_force_threshold must be finite"
    ):
        BelloLocomotionEnv.load_prepared(
            full_body_asset,
            env_config=BelloEnvConfig(
                contact=ContactConfig(foot_contact_force_threshold=-0.01)
            ),
        )


def test_locomotion_env_ignores_disabled_missing_reward_link(
    full_body_asset: PreparedAsset,
) -> None:
    reward_config = RewardConfig(
        track_lin_vel_xy=TrackLinVelXYRewardConfig(enabled=False, link="missing_link")
    )

    env = BelloLocomotionEnv.load_prepared(full_body_asset, reward_config=reward_config)

    assert env.reward_config.track_lin_vel_xy.link == "missing_link"


def test_locomotion_env_adds_flat_runtime_terrain(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        env_config=BelloEnvConfig(terrain=TerrainConfig(mode="plane")),
    )
    plane_count = sum(
        1
        for geom_type in env.model.mj_model.geom_type
        if geom_type == mujoco.mjtGeom.mjGEOM_PLANE
    )

    assert plane_count == 1


def test_observation_is_finite_after_reset(full_body_asset: PreparedAsset) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)
    _, obs = env.reset()

    assert obs.shape == (env.observation_size,)
    assert jnp.all(jnp.isfinite(obs))


def test_reset_batch_shapes(full_body_asset: PreparedAsset) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)
    batch_size = 4
    state, obs = env.reset_batch(batch_size)

    assert obs.shape == (batch_size, env.observation_size)
    assert state.last_action.shape == (batch_size, env.action_size)
    assert state.command.shape == (batch_size, 3)
    assert state.foot_contact_debounce_time.shape == (batch_size, 2)
    assert state.step_count.shape == (batch_size,)


def test_foot_contact_debounce_delays_contact_flips(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        env_config=BelloEnvConfig(
            contact=ContactConfig(
                foot_contact_on_seconds=0.04,
                foot_contact_off_seconds=0.04,
            )
        ),
    )
    state, _ = env.reset()
    state = state.replace(
        foot_contact_debounce_time=jnp.zeros(2, dtype=jnp.float32),
        last_foot_contact=jnp.asarray([False, True]),
    )
    raw_contact = jnp.asarray([1.0, 0.0], dtype=jnp.float32)

    filtered, timers = env._debounce_foot_contact(
        state, raw_contact, jnp.asarray(False)
    )
    np.testing.assert_array_equal(np.asarray(filtered), np.asarray([0.0, 1.0]))
    np.testing.assert_allclose(np.asarray(timers), np.asarray([0.02, 0.02]))

    filtered, timers = env._debounce_foot_contact(
        state.replace(foot_contact_debounce_time=timers),
        raw_contact,
        jnp.asarray(False),
    )
    np.testing.assert_array_equal(np.asarray(filtered), np.asarray([1.0, 0.0]))
    np.testing.assert_allclose(np.asarray(timers), np.zeros(2))


def test_foot_contact_debounce_default_flips_immediately(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)
    state, _ = env.reset()
    state = state.replace(
        foot_contact_debounce_time=jnp.zeros(2, dtype=jnp.float32),
        last_foot_contact=jnp.asarray([False, True]),
    )

    filtered, timers = env._debounce_foot_contact(
        state, jnp.asarray([1.0, 0.0], dtype=jnp.float32), jnp.asarray(False)
    )

    np.testing.assert_array_equal(np.asarray(filtered), np.asarray([1.0, 0.0]))
    np.testing.assert_allclose(np.asarray(timers), np.zeros(2))


def test_foot_contact_debounce_resets_timer_on_done(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        env_config=BelloEnvConfig(contact=ContactConfig(foot_contact_off_seconds=0.04)),
    )
    state, _ = env.reset()
    state = state.replace(
        foot_contact_debounce_time=jnp.asarray([0.02, 0.02], dtype=jnp.float32),
        last_foot_contact=jnp.asarray([True, True]),
    )

    _, timers = env._debounce_foot_contact(
        state, jnp.zeros(2, dtype=jnp.float32), jnp.asarray(True)
    )

    np.testing.assert_allclose(np.asarray(timers), np.zeros(2))


def test_foot_memory_keeps_current_contact_on_done(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)
    state, _ = env.reset()
    state = state.replace(
        foot_air_time=jnp.asarray([0.1, 0.2], dtype=jnp.float32),
        last_foot_contact=jnp.asarray([False, False]),
    )

    air_time, last_contact = env._next_foot_memory(
        state,
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        jnp.asarray(True),
    )

    np.testing.assert_allclose(np.asarray(air_time), np.zeros(2))
    np.testing.assert_array_equal(np.asarray(last_contact), np.asarray([True, False]))


def test_reset_initializes_last_foot_contact_without_synthetic_contact(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)

    state, _ = env.reset()

    np.testing.assert_array_equal(np.asarray(state.last_foot_contact), np.zeros(2))


def test_policy_action_controls_all_default_actuators(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)
    action = jnp.ones(env.action_size, dtype=jnp.float32)
    ctrl = env._action_to_ctrl(action)
    default_ctrl = jnp.asarray(env.model.default_ctrl)

    assert ctrl.shape == (env.model.mj_model.nu,)
    assert len(env.model.hold_ctrl_indices) == 0
    assert jnp.all(
        ctrl[env.model.policy_ctrl_indices]
        != default_ctrl[env.model.policy_ctrl_indices]
    )


def test_policy_action_uses_reference_style_bounded_affine_targets(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)
    action = jnp.full((env.action_size,), 10.0, dtype=jnp.float32)
    ctrl = env._action_to_ctrl(action)
    policy_indices = jnp.asarray(env.model.policy_ctrl_indices)
    policy_ctrl = ctrl[policy_indices]
    expected = (
        jnp.asarray(env.model.default_ctrl)[policy_indices]
        + jnp.asarray(env.model.policy_action_scale) * 5.0
    )

    assert env.action_clip == 5.0
    assert bool(jnp.allclose(policy_ctrl, expected))
    assert not bool(jnp.any(jnp.asarray(env.model.mj_model.actuator_ctrllimited)))


def test_reset_batch_broadcasts_command(full_body_asset: PreparedAsset) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)
    command = jnp.asarray([0.1, 0.2, 0.3], dtype=jnp.float32)
    state, obs = env.reset_batch(3, command=command)

    assert obs.shape == (3, env.observation_size)
    assert jnp.all(state.command == command)


def test_reset_rejects_invalid_command_shape(full_body_asset: PreparedAsset) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)

    with pytest.raises(ValueError, match="command must have shape"):
        env.reset(command=jnp.asarray([0.1, 0.2], dtype=jnp.float32))


def test_reset_batch_rejects_invalid_command_shape(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)

    with pytest.raises(ValueError, match="batch command must have shape"):
        env.reset_batch(3, command=jnp.ones((2, 3), dtype=jnp.float32))


def test_reset_batch_samples_commands_with_rng(
    full_body_asset: PreparedAsset,
) -> None:
    env = replace(
        BelloLocomotionEnv.load_prepared(full_body_asset),
        command_config=CommandConfig(
            lin_vel_x=(0.2, 0.2),
            lin_vel_y=(0.1, 0.1),
            ang_vel_yaw=(-0.2, -0.2),
            standing_probability=0.0,
        ),
    )
    state, obs = env.reset_batch(4, rng=jax.random.PRNGKey(0))

    assert obs.shape == (4, env.observation_size)
    assert jnp.all(state.command == jnp.asarray([0.2, 0.1, -0.2]))


def test_reset_batch_can_randomize_spawn_state(full_body_asset: PreparedAsset) -> None:
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        env_config=BelloEnvConfig(
            spawn=SpawnConfig(
                root_xy_range=(0.1, 0.1),
                root_yaw_range=(0.2, 0.2),
                joint_pos_noise=(0.01, 0.01),
                joint_vel_range=(0.3, 0.3),
            )
        ),
    )
    state, _ = env.reset_batch(2, rng=jax.random.PRNGKey(7))
    root_qpos = int(env.model.root_qpos_adr)
    root_qvel = int(env.model.root_qvel_adr)

    np.testing.assert_allclose(
        np.asarray(state.data.qpos[:, root_qpos : root_qpos + 2]),
        np.tile(
            np.asarray(env.model.default_qpos[root_qpos : root_qpos + 2]) + 0.1,
            (2, 1),
        ),
    )
    assert not jnp.allclose(
        state.data.qpos[:, root_qpos + 3 : root_qpos + 7],
        jnp.asarray(env.model.default_qpos[root_qpos + 3 : root_qpos + 7]),
    )
    np.testing.assert_allclose(
        np.asarray(state.data.qvel[:, env.model.actuator_qvel_indices]),
        np.full((2, env.model.mj_model.nu), 0.3, dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(state.data.qvel[:, root_qvel : root_qvel + 2]),
        np.zeros((2, 2), dtype=np.float32),
    )


def test_initial_step_count_staggering_is_explicit(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        env_config=BelloEnvConfig(
            max_episode_steps=8,
            spawn=SpawnConfig(stagger_initial_step_count=True),
        ),
    )
    step_count = env.initial_step_counts(16, jax.random.PRNGKey(0))

    assert step_count.shape == (16,)
    assert jnp.any(step_count != 0)
    assert jnp.all(step_count < 8)


def test_reset_batch_samples_commands_from_override_ranges(
    full_body_asset: PreparedAsset,
) -> None:
    env = replace(
        BelloLocomotionEnv.load_prepared(full_body_asset),
        command_config=CommandConfig(
            lin_vel_x=(-0.1, 0.1),
            lin_vel_y=(-0.1, 0.1),
            ang_vel_yaw=(-0.1, 0.1),
            standing_probability=0.0,
            curriculum=CommandCurriculumConfig(enabled=True),
        ),
    )
    ranges = jnp.asarray([[0.2, 0.2], [0.1, 0.1], [-0.2, -0.2]], dtype=jnp.float32)
    state, obs = env.reset_batch(4, rng=jax.random.PRNGKey(0), command_ranges=ranges)

    assert obs.shape == (4, env.observation_size)
    assert jnp.all(state.command == jnp.asarray([0.2, 0.1, -0.2]))


def test_command_config_can_disable_command_observation_and_sampling(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        command_config=CommandConfig(
            enabled=False,
            lin_vel_x=(0.2, 0.2),
            lin_vel_y=(0.1, 0.1),
            ang_vel_yaw=(-0.2, -0.2),
            standing_probability=0.0,
        ),
    )
    explicit_command = jnp.asarray([0.1, 0.2, 0.3], dtype=jnp.float32)
    state, obs = env.reset(command=explicit_command, rng=jax.random.PRNGKey(0))
    batch_state, batch_obs = env.reset_batch(4, rng=jax.random.PRNGKey(1))

    assert env.command_observation_size == 0
    assert env.observation_size == 6 + 3 * env.model.mj_model.nu + env.action_size
    assert obs.shape == (env.observation_size,)
    assert batch_obs.shape == (4, env.observation_size)
    assert jnp.all(state.command == jnp.zeros(3, dtype=jnp.float32))
    assert jnp.all(batch_state.command == jnp.zeros((4, 3), dtype=jnp.float32))


def test_observation_config_can_hide_command_without_disabling_sampling(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        env_config=BelloEnvConfig(
            observation=ObservationConfig(command=False),
        ),
        command_config=CommandConfig(
            lin_vel_x=(0.2, 0.2),
            lin_vel_y=(0.1, 0.1),
            ang_vel_yaw=(-0.2, -0.2),
            standing_probability=0.0,
        ),
    )
    state, obs = env.reset(rng=jax.random.PRNGKey(0))

    assert env.command_observation_size == 0
    assert env.observation_size == 6 + 3 * env.model.mj_model.nu + env.action_size
    assert obs.shape == (env.observation_size,)
    assert jnp.all(state.command == jnp.asarray([0.2, 0.1, -0.2]))


def test_observation_config_can_select_components(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        env_config=BelloEnvConfig(
            observation=ObservationConfig(
                base_ang_vel=False,
                projected_gravity=True,
                command=False,
                joint_pos=True,
                joint_vel=False,
                actuator_force=True,
                last_action=False,
            ),
        ),
    )
    _, obs = env.reset()

    assert env.observation_size == 3 + 2 * env.model.mj_model.nu
    assert obs.shape == (env.observation_size,)


def test_observation_noise_is_applied_when_rng_is_passed(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        env_config=BelloEnvConfig(
            observation=ObservationConfig(
                base_ang_vel=True,
                projected_gravity=False,
                command=False,
                joint_pos=False,
                joint_vel=False,
                actuator_force=False,
                last_action=False,
                noise=ObservationNoiseConfig(
                    enabled=True,
                    base_ang_vel=(0.5, 0.5),
                ),
            ),
        ),
    )
    state, noisy_obs = env.reset_batch(2, rng=jax.random.PRNGKey(0))
    clean_obs = jax.vmap(env.observation)(state)

    assert noisy_obs.shape == clean_obs.shape
    assert jnp.allclose(noisy_obs - clean_obs, 0.5)


def test_random_push_sets_root_xy_velocity_on_interval(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        batch_size=2,
        env_config=BelloEnvConfig(
            random_push=RandomPushConfig(
                enabled=True,
                interval_seconds=0.02,
                lin_vel_x=(0.3, 0.3),
                lin_vel_y=(-0.2, -0.2),
            ),
        ),
    )
    state, _ = env.reset_batch(2)
    state = state.replace(step_count=jnp.ones((2,), dtype=jnp.int32))

    data, push_applied, push_velocity = env._maybe_apply_random_push_batch(
        state.data, state.step_count, jax.random.PRNGKey(0)
    )

    root_qvel = int(env.model.root_qvel_adr)
    assert jnp.all(push_applied)
    assert jnp.allclose(push_velocity, jnp.asarray([[0.3, -0.2], [0.3, -0.2]]))
    assert jnp.allclose(
        data.qvel[:, root_qvel : root_qvel + 2],
        jnp.asarray([[0.3, -0.2], [0.3, -0.2]]),
    )


def test_random_push_overwrites_existing_root_xy_velocity(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        batch_size=1,
        env_config=BelloEnvConfig(
            random_push=RandomPushConfig(
                enabled=True,
                interval_seconds=0.02,
                lin_vel_x=(0.3, 0.3),
                lin_vel_y=(-0.2, -0.2),
            ),
        ),
    )
    state, _ = env.reset_batch(1)
    root_qvel = int(env.model.root_qvel_adr)
    qvel = state.data.qvel.at[:, root_qvel : root_qvel + 2].set(
        jnp.asarray([[1.0, 1.0]], dtype=jnp.float32)
    )
    state = state.replace(
        data=state.data.replace(qvel=qvel),
        step_count=jnp.ones((1,), dtype=jnp.int32),
    )

    data, push_applied, _ = env._maybe_apply_random_push_batch(
        state.data, state.step_count, jax.random.PRNGKey(0)
    )

    assert bool(push_applied[0])
    assert jnp.allclose(
        data.qvel[:, root_qvel : root_qvel + 2],
        jnp.asarray([[0.3, -0.2]], dtype=jnp.float32),
    )


def test_observation_config_can_include_gait_phase(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        env_config=BelloEnvConfig(
            observation=ObservationConfig(
                base_ang_vel=False,
                projected_gravity=False,
                gait_phase=True,
                command=False,
                joint_pos=False,
                joint_vel=False,
                actuator_force=False,
                last_action=False,
            ),
        ),
    )
    _, obs = env.reset()

    assert env.observation_size == 2
    assert obs.shape == (2,)
    assert jnp.allclose(obs, jnp.asarray([0.0, 1.0], dtype=jnp.float32))


def test_gait_frequency_tracks_command_speed(full_body_asset: PreparedAsset) -> None:
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        env_config=BelloEnvConfig(
            gait=GaitConfig(
                cycle_time=0.80,
                command_speed_frequency_gain=0.55,
                min_frequency=1.20,
                max_frequency=2.20,
            )
        ),
    )

    assert float(env._gait_frequency(jnp.asarray([0.0, 0.0, 0.0]))) == pytest.approx(
        1.25
    )
    assert float(env._gait_frequency(jnp.asarray([0.0, 0.5, 0.0]))) == pytest.approx(
        1.525
    )
    assert float(env._gait_frequency(jnp.asarray([0.0, 0.5, 1.0]))) == pytest.approx(
        1.525
    )
    assert float(env._gait_frequency(jnp.asarray([0.0, 10.0, 0.0]))) == pytest.approx(
        2.20
    )


def test_gait_phase_freezes_for_standing_commands(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        env_config=BelloEnvConfig(
            gait=GaitConfig(
                cycle_time=0.80,
                command_speed_frequency_gain=0.55,
                min_frequency=1.20,
                max_frequency=2.20,
                standing_command_threshold=0.08,
            )
        ),
    )
    step_count = jnp.asarray(10, dtype=jnp.int32)

    assert float(
        env._gait_phase_from_step_count(
            step_count, jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32)
        )
    ) == pytest.approx(0.0)
    assert float(
        env._gait_phase_from_step_count(
            step_count, jnp.asarray([0.0, 0.0, 0.08], dtype=jnp.float32)
        )
    ) == pytest.approx(0.0)
    assert float(
        env._gait_phase_from_step_count(
            step_count, jnp.asarray([0.0, 0.5, 0.0], dtype=jnp.float32)
        )
    ) == pytest.approx(0.305)


def test_observation_config_can_include_base_state(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        env_config=BelloEnvConfig(
            observation=ObservationConfig(
                base_lin_vel=True,
                base_ang_vel=False,
                projected_gravity=False,
                base_height=True,
                command=False,
                joint_pos=False,
                joint_vel=False,
                actuator_force=False,
                last_action=False,
            ),
        ),
    )
    state, obs = env.reset()

    assert env.observation_size == 4
    assert obs.shape == (4,)
    assert jnp.allclose(obs[:3], jnp.zeros(3, dtype=jnp.float32))
    assert obs[3] == pytest.approx(
        float(state.data.qpos[int(env.model.root_qpos_adr) + 2])
    )


def test_root_link_controls_base_observation_and_default_measurement_link(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        env_config=BelloEnvConfig(
            root_link="torso_link",
            observation=ObservationConfig(
                base_lin_vel=True,
                base_ang_vel=False,
                projected_gravity=False,
                base_height=True,
                command=False,
                joint_pos=False,
                joint_vel=False,
                actuator_force=False,
                last_action=False,
            ),
        ),
    )
    state, obs = env.reset()
    torso_id = mujoco.mj_name2id(
        env.model.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
    )

    assert torso_id >= 0
    assert env.observation_size == 4
    assert obs.shape == (4,)
    assert obs[3] == pytest.approx(float(state.data.xpos[torso_id, 2]))
    assert obs[3] > 0.5
    assert float(env._base_height(state)) == pytest.approx(
        float(state.data.xpos[torso_id, 2])
    )
    assert env._done_link(env.done_config.base_height) == "torso_link"


def test_observation_config_actuator_force_follows_joint_mode(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        env_config=BelloEnvConfig(
            observation=ObservationConfig(
                joint_mode="policy",
                base_ang_vel=False,
                projected_gravity=False,
                command=False,
                joint_pos=False,
                joint_vel=False,
                actuator_force=True,
                last_action=False,
            ),
        ),
    )
    _, obs = env.reset()

    assert env.observation_size == env.action_size
    assert obs.shape == (env.action_size,)


def test_observation_config_allows_empty_observation(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        env_config=BelloEnvConfig(
            observation=ObservationConfig(
                base_ang_vel=False,
                projected_gravity=False,
                command=False,
                joint_pos=False,
                joint_vel=False,
                actuator_force=False,
                last_action=False,
            ),
        ),
    )
    _, obs = env.reset()

    assert env.observation_size == 0
    assert obs.shape == (0,)


def test_critic_observation_config_can_be_privileged(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        env_config=BelloEnvConfig(
            observation=ObservationConfig(
                joint_mode="policy",
                base_ang_vel=True,
                projected_gravity=True,
                command=True,
                joint_pos=True,
                joint_vel=True,
                actuator_force=False,
                last_action=True,
            ),
            critic_observation=ObservationConfig(
                joint_mode="policy",
                base_lin_vel=True,
                base_ang_vel=True,
                projected_gravity=True,
                base_height=False,
                gait_phase=False,
                command=True,
                joint_pos=True,
                joint_vel=True,
                actuator_force=False,
                last_action=True,
                scale=ObservationScaleConfig(base_ang_vel=0.2, joint_vel=0.05),
                noise=ObservationNoiseConfig(enabled=False),
            ),
        ),
    )
    state, obs = env.reset()
    critic_obs = env.critic_observation(state, jax.random.PRNGKey(0))

    assert env.observation_size == 9 + 3 * env.action_size
    assert env.critic_observation_size == 12 + 3 * env.action_size
    assert obs.shape == (env.observation_size,)
    assert critic_obs.shape == (env.critic_observation_size,)
    assert jnp.allclose(
        critic_obs,
        env.critic_observation(state, jax.random.PRNGKey(1)),
    )


def test_step_batch_reports_reward_and_auto_resets_done_envs(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        env_config=BelloEnvConfig(max_episode_steps=1),
    )
    state, _ = env.reset_batch(2)
    next_state, next_obs, reward, done, info = env.step_batch(
        state,
        jnp.zeros((2, env.action_size), dtype=jnp.float32),
        rng=jax.random.PRNGKey(1),
    )

    assert next_obs.shape == (2, env.observation_size)
    assert reward.shape == (2,)
    assert done.shape == (2,)
    assert jnp.all(done)
    assert jnp.all(next_state.step_count == 0)
    assert jnp.all(next_state.last_episode_return == info["episode_return"])
    assert jnp.all(next_state.last_episode_length == info["episode_length"])
    assert jnp.all(info["returned_episode"])
    assert jnp.all(info["done/any"])
    assert jnp.all(info["done/timeout"])
    assert not bool(jnp.any(info["done/terminal"]))
    assert not bool(jnp.any(info["done/base_height"]))
    assert not bool(jnp.any(info["done/bad_orientation"]))
    assert not bool(jnp.any(info["done/nonfinite"]))
    assert jnp.all(info["reward/termination"] == 0.0)
    assert info["episode_return"].shape == (2,)
    assert info["reward/track_lin_vel_xy"].shape == (2,)
    assert info["reward/foot_contact"].shape == (2,)
    assert info["reward/foot_slip"].shape == (2,)
    assert info["reward/foot_clearance"].shape == (2,)
    assert info["reward/foot_distance"].shape == (2,)
    assert info["reward/undesired_contact"].shape == (2,)
    assert jnp.all(jnp.isfinite(reward))


def test_step_batch_sanitizes_nonfinite_terminal_rewards(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        reward_config=RewardConfig(
            termination=TerminationRewardConfig(weight=-3.0),
        ),
    )
    state, _ = env.reset_batch(2)
    state = state.replace(
        data=state.data.replace(qpos=state.data.qpos.at[0, 0].set(jnp.nan))
    )

    next_state, next_obs, reward, done, info = env.step_batch(
        state,
        jnp.zeros((2, env.action_size), dtype=jnp.float32),
        rng=jax.random.PRNGKey(1),
    )

    assert bool(done[0])
    assert bool(info["done/nonfinite"][0])
    assert int(next_state.step_count[0]) == 0
    assert jnp.all(jnp.isfinite(next_obs[0]))
    assert jnp.all(jnp.isfinite(reward))
    for name, value in info.items():
        if name.startswith("reward/"):
            assert jnp.all(jnp.isfinite(value)), name
    assert float(reward[0]) == pytest.approx(-3.0)
    assert float(info["reward/termination"][0]) == pytest.approx(-3.0)
    assert float(info["reward/base_height"][0]) == pytest.approx(0.0)
    assert float(info["reward/joint_acc"][0]) == pytest.approx(0.0)
    assert float(info["reward/energy"][0]) == pytest.approx(0.0)


def test_bad_orientation_done_uses_projected_gravity_tilt(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        done_config=DoneConfig(
            base_height=BaseHeightDoneConfig(enabled=False),
            bad_orientation=BadOrientationDoneConfig(
                limit_angle=0.8,
                link="bello_root",
            ),
        ),
    )
    state, _ = env.reset()
    root_qpos = int(env.model.root_qpos_adr)

    qpos = state.data.qpos.at[root_qpos + 3 : root_qpos + 7].set(
        root_quat_from_roll_pitch(0.4, 0.4)
    )
    below_limit = state.replace(data=state.data.replace(qpos=qpos))
    assert not bool(env._done_flags(below_limit)["done/bad_orientation"])

    qpos = state.data.qpos.at[root_qpos + 3 : root_qpos + 7].set(
        root_quat_from_roll_pitch(0.7, 0.7)
    )
    above_limit = state.replace(data=state.data.replace(qpos=qpos))
    assert bool(env._done_flags(above_limit)["done/bad_orientation"])


def test_default_base_height_threshold_is_below_nominal_reset(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)
    state, _ = env.reset()
    base_height = float(env._base_height(state))

    assert env.reward_config.base_height.target < base_height
    assert base_height - env.reward_config.base_height.target < 0.05
    assert base_height > env.done_config.base_height.min_height


def test_step_batch_contact_rewards_are_world_local(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        warp=WarpConfig(contact_budget_per_env=64),
        batch_size=2,
    )
    state, _ = env.reset_batch(2)
    root_z = env.model.root_qpos_adr + 2
    qpos = state.data.qpos.at[0, root_z].set(0.5).at[1, root_z].set(2.0)
    state = state.replace(data=state.data.replace(qpos=qpos))

    _, _, _, _, info = env.step_batch(
        state,
        jnp.zeros((2, env.action_size), dtype=jnp.float32),
        rng=jax.random.PRNGKey(0),
    )

    assert info["reward/foot_contact"][0] > info["reward/foot_contact"][1]


def test_undesired_contact_uses_force_threshold(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        warp=WarpConfig(contact_budget_per_env=64),
        batch_size=1,
        env_config=BelloEnvConfig(
            contact=ContactConfig(undesired_contact_force_threshold=1.0e9)
        ),
    )
    state, _ = env.reset_batch(1)
    root_z = env.model.root_qpos_adr + 2
    state = state.replace(
        data=state.data.replace(qpos=state.data.qpos.at[:, root_z].set(0.2))
    )

    _, _, _, _, info = env.step_batch(
        state,
        jnp.zeros((1, env.action_size), dtype=jnp.float32),
        rng=jax.random.PRNGKey(0),
    )

    assert jnp.all(info["reward/undesired_contact"] == 0.0)


def test_locomotion_env_rejects_negative_self_collision_force_threshold(
    full_body_asset: PreparedAsset,
) -> None:
    with pytest.raises(ValueError, match="contact.self_collision_force_threshold"):
        BelloLocomotionEnv.load_prepared(
            full_body_asset,
            env_config=BelloEnvConfig(
                contact=ContactConfig(self_collision_force_threshold=-0.01)
            ),
        )


def test_foot_contact_uses_force_threshold(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        warp=WarpConfig(contact_budget_per_env=64),
        batch_size=1,
        env_config=BelloEnvConfig(
            contact=ContactConfig(foot_contact_force_threshold=1.0e9)
        ),
    )
    state, _ = env.reset_batch(1)
    root_z = env.model.root_qpos_adr + 2
    state = state.replace(
        data=state.data.replace(qpos=state.data.qpos.at[:, root_z].set(0.2))
    )

    _, _, _, _, info = env.step_batch(
        state,
        jnp.zeros((1, env.action_size), dtype=jnp.float32),
        rng=jax.random.PRNGKey(0),
    )

    assert jnp.all(info["reward/foot_contact"] == 0.0)


def test_step_reports_enabled_foot_distance_reward(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        reward_config=RewardConfig(
            foot_distance=FootDistanceRewardConfig(enabled=True, weight=1.0)
        ),
    )
    state, _ = env.reset()
    foot_body_ids = jnp.asarray(env.model.foot_body_ids)
    xpos = (
        state.data.xpos.at[foot_body_ids[0], :2]
        .set(jnp.asarray([0.0, 0.0], dtype=jnp.float32))
        .at[foot_body_ids[1], :2]
        .set(jnp.asarray([0.3, 0.4], dtype=jnp.float32))
    )
    state = state.replace(data=state.data.replace(xpos=xpos))

    _, info = env._reward(
        state,
        state,
        jnp.zeros(env.action_size, dtype=jnp.float32),
        jnp.asarray(False),
        jnp.ones(2, dtype=jnp.float32),
        env._foot_height(state),
        jnp.zeros(2, dtype=jnp.float32),
        jnp.zeros(2, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
    )

    assert info["reward/foot_distance"] == pytest.approx(0.01)


def test_step_batch_resamples_commands_periodically(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        batch_size=2,
        command_config=CommandConfig(
            lin_vel_x=(0.2, 0.2),
            lin_vel_y=(0.1, 0.1),
            ang_vel_yaw=(-0.2, -0.2),
            standing_probability=0.0,
            resample_seconds=1.0e-9,
        ),
    )
    state, _ = env.reset_batch(2, command=jnp.zeros(3, dtype=jnp.float32))
    next_state, _, _, done, _ = env.step_batch(
        state,
        jnp.zeros((2, env.action_size), dtype=jnp.float32),
        rng=jax.random.PRNGKey(0),
    )

    assert not bool(jnp.any(done))
    assert jnp.all(next_state.command == jnp.asarray([0.2, 0.1, -0.2]))


def test_done_config_can_disable_timeout(
    full_body_asset: PreparedAsset,
) -> None:
    env = replace(
        BelloLocomotionEnv.load_prepared(full_body_asset),
        config=BelloEnvConfig(max_episode_steps=1),
        done_config=DoneConfig(timeout=TimeoutDoneConfig(enabled=False)),
    )
    state, _ = env.reset_batch(2)
    next_state, _, _, done, info = env.step_batch(
        state,
        jnp.zeros((2, env.action_size), dtype=jnp.float32),
    )

    assert not bool(jnp.any(done))
    assert jnp.all(next_state.step_count == 1)
    assert not bool(jnp.any(info["returned_episode"]))
    assert not bool(jnp.any(info["done/any"]))
    assert not bool(jnp.any(info["done/timeout"]))


def test_observation_size_tracks_loaded_model_action_size(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)

    assert (
        env.observation_size
        == 6
        + env.command_observation_size
        + 2 * env.model.mj_model.nu
        + env.model.mj_model.nu
        + env.action_size
    )


def test_locomotion_env_loads_lower_body_with_default_locomotion_terms(
    generated_test_assets: dict[str, PreparedAsset],
) -> None:
    env = BelloLocomotionEnv.load_prepared(generated_test_assets["lower_body"])

    assert env.model.mj_model.nu == 12
    assert env.model.foot_body_ids.shape == (2,)
    assert env.action_size == 12


def test_locomotion_env_rejects_upper_body_when_foot_rewards_are_enabled(
    generated_test_assets: dict[str, PreparedAsset],
) -> None:
    with pytest.raises(ValueError, match="enabled foot reward terms require"):
        BelloLocomotionEnv.load_prepared(generated_test_assets["upper_body"])


def test_locomotion_env_loads_upper_body_when_foot_rewards_are_disabled(
    generated_test_assets: dict[str, PreparedAsset],
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        generated_test_assets["upper_body"],
        reward_config=upper_body_reward_config(),
        done_config=DoneConfig(
            base_height=BaseHeightDoneConfig(enabled=False),
            bad_orientation=BadOrientationDoneConfig(enabled=False),
        ),
    )
    state, obs = env.reset()

    assert env.model.mj_model.nu == 22
    assert env.model.foot_body_ids.size == 0
    assert env.action_size == 22
    assert obs.shape == (env.observation_size,)
    assert state.last_action.shape == (22,)


def upper_body_reward_config() -> RewardConfig:
    return RewardConfig(
        track_lin_vel_xy=TrackLinVelXYRewardConfig(enabled=False),
        track_ang_vel_z=TrackAngVelZRewardConfig(enabled=False),
        lin_vel_z=LinVelZRewardConfig(enabled=False),
        ang_vel_xy=AngVelXYRewardConfig(enabled=False),
        flat_orientation=FlatOrientationRewardConfig(enabled=False),
        base_height=BaseHeightRewardConfig(enabled=False),
        foot_contact=FootContactRewardConfig(enabled=False),
        foot_slip=FootSlipRewardConfig(enabled=False),
        foot_clearance=FootClearanceRewardConfig(enabled=False),
        gait_contact=GaitContactRewardConfig(enabled=False),
        feet_air_time=FeetAirTimeRewardConfig(enabled=False),
    )
