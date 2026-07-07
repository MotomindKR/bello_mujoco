from dataclasses import replace

import jax
import jax.numpy as jnp
import mujoco
import pytest

from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig, PreparedAsset
from bello_mujoco.envs.locomotion import BelloEnvConfig, BelloLocomotionEnv, DoneConfig
from bello_mujoco.envs.rewards import CommandConfig


def test_reset_shapes(full_body_asset: PreparedAsset) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)
    state, obs = env.reset()

    assert env.model.mj_model.nu == 35
    assert env.action_size == 12
    assert env.observation_size == 9 + 2 * env.model.mj_model.nu + env.action_size
    assert obs.shape == (env.observation_size,)
    assert state.last_action.shape == (env.action_size,)
    assert int(state.step_count) == 0


def test_model_exposes_contact_reward_sets(full_body_asset: PreparedAsset) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)

    assert env.model.foot_body_ids.shape == (2,)
    assert env.model.foot_geom_ids.shape == (2,)
    assert env.model.terrain_geom_ids.shape == (1,)
    assert env.model.nonfoot_robot_geom_ids.size > env.model.foot_geom_ids.size


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


def test_policy_action_only_controls_policy_actuators(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)
    action = jnp.ones(env.action_size, dtype=jnp.float32)
    ctrl = env._action_to_ctrl(action)
    default_ctrl = jnp.asarray(env.model.default_ctrl)

    assert ctrl.shape == (env.model.mj_model.nu,)
    assert jnp.any(
        ctrl[env.model.policy_ctrl_indices]
        != default_ctrl[env.model.policy_ctrl_indices]
    )
    assert jnp.all(
        ctrl[env.model.hold_ctrl_indices] == default_ctrl[env.model.hold_ctrl_indices]
    )


def test_reset_batch_broadcasts_command(full_body_asset: PreparedAsset) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)
    command = jnp.asarray([0.1, 0.2, 0.3], dtype=jnp.float32)
    state, obs = env.reset_batch(3, command=command)

    assert obs.shape == (3, env.observation_size)
    assert jnp.all(state.command == command)


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
    assert jnp.all(info["returned_episode"])
    assert info["episode_return"].shape == (2,)
    assert info["reward/track_lin_vel_xy"].shape == (2,)
    assert info["reward/foot_contact"].shape == (2,)
    assert info["reward/foot_slip"].shape == (2,)
    assert info["reward/foot_clearance"].shape == (2,)
    assert info["reward/undesired_contact"].shape == (2,)
    assert jnp.all(jnp.isfinite(reward))


def test_done_config_can_disable_timeout(
    full_body_asset: PreparedAsset,
) -> None:
    env = replace(
        BelloLocomotionEnv.load_prepared(full_body_asset),
        config=BelloEnvConfig(max_episode_steps=1),
        done_config=DoneConfig(terminate_on_timeout=False),
    )
    state, _ = env.reset_batch(2)
    next_state, _, _, done, info = env.step_batch(
        state,
        jnp.zeros((2, env.action_size), dtype=jnp.float32),
    )

    assert not bool(jnp.any(done))
    assert jnp.all(next_state.step_count == 1)
    assert not bool(jnp.any(info["returned_episode"]))


def test_observation_size_tracks_loaded_model_action_size(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)

    assert env.observation_size == 9 + 2 * env.model.mj_model.nu + env.action_size


def test_locomotion_env_rejects_non_full_body_config() -> None:
    with pytest.raises(ValueError, match="requires asset='full_body'"):
        BelloLocomotionEnv.load(AssetPrepConfig(asset="lower_body"))
