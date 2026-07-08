from dataclasses import replace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from bello_mujoco.assets_pipeline.prepare import PreparedAsset
from bello_mujoco.envs.locomotion import (
    BaseHeightDoneConfig,
    BelloEnvConfig,
    BelloLocomotionEnv,
    DoneConfig,
    ObservationConfig,
    RollPitchDoneConfig,
    TimeoutDoneConfig,
)
from bello_mujoco.envs.rewards import (
    AngVelXYRewardConfig,
    BaseHeightRewardConfig,
    CommandConfig,
    FlatOrientationRewardConfig,
    FootClearanceRewardConfig,
    FootContactRewardConfig,
    FootSlipRewardConfig,
    FeetAirTimeRewardConfig,
    GaitContactRewardConfig,
    LinVelZRewardConfig,
    RewardConfig,
    TrackAngVelZRewardConfig,
    TrackLinVelXYRewardConfig,
)
from bello_mujoco.sim import WarpConfig


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
    assert state.last_foot_contact.shape == (2,)
    assert int(state.step_count) == 0


def test_model_exposes_contact_reward_sets(full_body_asset: PreparedAsset) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)

    assert env.model.foot_body_ids.shape == (2,)
    assert env.model.foot_geom_ids.shape == (2,)
    assert env.model.terrain_geom_ids.shape == (1,)
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


def test_locomotion_env_ignores_disabled_missing_reward_link(
    full_body_asset: PreparedAsset,
) -> None:
    reward_config = RewardConfig(
        track_lin_vel_xy=TrackLinVelXYRewardConfig(enabled=False, link="missing_link")
    )

    env = BelloLocomotionEnv.load_prepared(full_body_asset, reward_config=reward_config)

    assert env.reward_config.track_lin_vel_xy.link == "missing_link"


def test_locomotion_env_uses_flat_runtime_terrain_by_default(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)
    plane_count = sum(
        1
        for geom_type in env.model.mj_model.geom_type
        if geom_type == mujoco.mjtGeom.mjGEOM_PLANE
    )

    assert env.terrain_config.mode == "plane"
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
    assert state.step_count.shape == (batch_size,)


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
    assert not bool(jnp.any(info["done/base_height"]))
    assert not bool(jnp.any(info["done/roll_pitch"]))
    assert not bool(jnp.any(info["done/nonfinite"]))
    assert info["episode_return"].shape == (2,)
    assert info["reward/track_lin_vel_xy"].shape == (2,)
    assert info["reward/foot_contact"].shape == (2,)
    assert info["reward/foot_slip"].shape == (2,)
    assert info["reward/foot_clearance"].shape == (2,)
    assert info["reward/undesired_contact"].shape == (2,)
    assert jnp.all(jnp.isfinite(reward))


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
    qpos = state.data.qpos.at[0, root_z].set(0.2).at[1, root_z].set(2.0)
    state = state.replace(data=state.data.replace(qpos=qpos))

    _, _, _, _, info = env.step_batch(
        state,
        jnp.zeros((2, env.action_size), dtype=jnp.float32),
        rng=jax.random.PRNGKey(0),
    )

    assert info["reward/foot_contact"][0] > info["reward/foot_contact"][1]
    assert info["reward/undesired_contact"][0] < info["reward/undesired_contact"][1]


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
            roll_pitch=RollPitchDoneConfig(enabled=False),
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
