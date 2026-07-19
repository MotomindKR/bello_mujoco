from dataclasses import replace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from bello_mujoco.assets_pipeline.prepare import PreparedAsset
from bello_mujoco.contracts import ActionLayout, ObservationLayout
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
    RandomPushConfig,
    SpawnConfig,
)
from bello_mujoco.envs.rewards import CommandConfig, RewardConfig
from bello_mujoco.render import (
    mujoco_eval_state,
    mujoco_foot_height,
    mujoco_link_ang_vel,
    mujoco_link_lin_acc,
    mujoco_link_lin_vel,
    mujoco_policy_observation,
)
from bello_mujoco.terrain import TerrainConfig, terrain_surface_height


def test_reset_exposes_explicit_actor_critic_and_action_contracts(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)
    state, observation = env.reset()

    assert env.action_layout.size == env.action_size == 35
    assert env.action_layout.names == env.model.actuator_joint_names
    assert env.observation_layout.components == (
        ("imu_linear_acceleration", 3),
        ("imu_angular_velocity", 3),
        ("projected_gravity", 3),
        ("foot_phase", 4),
        ("command", 3),
        ("joint_position", 35),
        ("joint_velocity", 35),
        ("last_action", 35),
    )
    assert env.critic_observation_layout.components[-3:] == (
        ("base_linear_velocity", 3),
        ("base_height", 1),
        ("actuator_force", 35),
    )
    assert observation.shape == (env.observation_size,)
    assert env.critic_observation(state).shape == (env.critic_observation_size,)
    assert state.last_action.shape == (env.action_size,)


def test_actor_observation_contains_only_deployable_signals(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)
    actor_names = {name for name, _ in env.observation_layout.components}
    critic_names = {name for name, _ in env.critic_observation_layout.components}

    assert "base_linear_velocity" not in actor_names
    assert "base_height" not in actor_names
    assert "actuator_force" not in actor_names
    assert {"imu_linear_acceleration", "imu_angular_velocity"} <= actor_names
    assert actor_names < critic_names


def test_native_data_adapter_uses_shared_observation_function(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)
    mj_data = mujoco.MjData(env.model.mj_model)
    mj_data.qpos[:] = env.model.default_qpos
    mj_data.qvel[:] = env.model.default_qvel
    mujoco.mj_forward(env.model.mj_model, mj_data)
    command = np.asarray([0.3, -0.1, 0.2], dtype=np.float32)
    last_action = np.linspace(-0.2, 0.2, env.action_size, dtype=np.float32)
    zeros = jnp.zeros(2, dtype=jnp.float32)
    state = mujoco_eval_state(
        env,
        mj_data,
        command=command,
        last_action=last_action,
        foot_air_time=zeros,
        foot_contact_time=zeros,
        foot_contact_debounce_time=zeros,
        last_foot_contact=zeros,
        step_count=7,
        episode_return=jnp.asarray(0.0),
        episode_track_lin_vel_xy=jnp.asarray(0.0),
        episode_length=jnp.asarray(7),
    )

    direct = env.observation(state)
    adapted = mujoco_policy_observation(
        env,
        mj_data,
        command=command,
        last_action=last_action,
        step_count=7,
    )
    assert direct == pytest.approx(adapted, abs=1.0e-6)


def test_mjx_link_kinematics_match_native_mujoco(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)
    mj_data = mujoco.MjData(env.model.mj_model)
    mj_data.qpos[:] = env.model.default_qpos
    mj_data.qvel[:] = np.linspace(-0.2, 0.3, env.model.mj_model.nv)
    mujoco.mj_forward(env.model.mj_model, mj_data)
    state, _ = env.reset()
    state = state.replace(
        data=state.data.replace(
            qpos=jnp.asarray(mj_data.qpos),
            qvel=jnp.asarray(mj_data.qvel),
            xpos=jnp.asarray(mj_data.xpos),
            xquat=jnp.asarray(mj_data.xquat),
            cvel=jnp.asarray(mj_data.cvel),
            subtree_com=jnp.asarray(mj_data.subtree_com),
            _impl=state.data._impl.replace(cacc=jnp.asarray(mj_data.cacc)),
        )
    )

    for link in ("bello_root", "pelvis_link", "torso_link"):
        assert env._link_body_lin_vel(state, link) == pytest.approx(
            mujoco_link_lin_vel(env, mj_data, link), abs=1.0e-5
        )
        assert env._link_body_ang_vel(state, link) == pytest.approx(
            mujoco_link_ang_vel(env, mj_data, link), abs=1.0e-5
        )
        assert env._link_body_lin_acc(state, link) == pytest.approx(
            mujoco_link_lin_acc(env, mj_data, link), abs=1.0e-4
        )


def test_reset_batch_and_step_have_finite_fixed_shapes(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset, batch_size=2)
    state, observation = env.reset_batch(2, rng=jax.random.PRNGKey(0))
    next_state, next_observation, reward, done, info = env.step_batch(
        state,
        jnp.zeros((2, env.action_size), dtype=jnp.float32),
        rng=jax.random.PRNGKey(1),
    )

    assert observation.shape == (2, env.observation_size)
    assert next_observation.shape == observation.shape
    assert reward.shape == done.shape == (2,)
    assert jnp.all(jnp.isfinite(next_observation))
    assert jnp.all(jnp.isfinite(reward))
    assert next_state.data.ctrl.shape == (2, env.model.mj_model.nu)
    assert "reward/track_lin_vel_xy" in info
    assert "done/nonfinite" in info


def test_timeout_auto_reset_preserves_terminal_diagnostics(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        batch_size=1,
        env_config=BelloEnvConfig(max_episode_steps=1),
    )
    state, _ = env.reset_batch(1)
    next_state, _, _, done, info = env.step_batch(
        state, jnp.zeros((1, env.action_size), dtype=jnp.float32)
    )

    assert bool(done[0])
    assert bool(info["done/timeout"][0])
    assert not bool(info["done/terminal"][0])
    assert int(next_state.step_count[0]) == 0
    assert int(info["episode_length"][0]) == 1


def test_action_target_reports_joint_and_motor_saturation_separately(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)
    ctrl, joint_saturation, motor_saturation = env._action_target_details(
        jnp.full(env.action_size, env.action_clip)
    )
    ctrl_range = env.model.mj_model.actuator_ctrlrange

    assert ctrl.shape == (env.model.mj_model.nu,)
    assert jnp.all(ctrl >= ctrl_range[:, 0])
    assert jnp.all(ctrl <= ctrl_range[:, 1])
    assert 0.0 <= joint_saturation <= 1.0
    assert 0.0 <= motor_saturation <= 1.0


def test_contact_debounce_and_memory_have_explicit_transitions(
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
    raw = jnp.asarray([True, False])
    first, elapsed = env.debounce_foot_contact(state, raw, jnp.asarray(False))
    second, _ = env.debounce_foot_contact(
        state.replace(foot_contact_debounce_time=elapsed), raw, jnp.asarray(False)
    )
    air, contact, previous = env.next_foot_memory(
        state,
        jnp.asarray([False, True]),
        jnp.asarray(False),
    )

    assert jnp.array_equal(first.astype(bool), jnp.asarray([False, False]))
    assert jnp.array_equal(second.astype(bool), jnp.asarray([True, False]))
    assert air == pytest.approx(jnp.asarray([env.control_timestep, 0.0]))
    assert contact == pytest.approx(jnp.asarray([0.0, env.control_timestep]))
    assert previous.tolist() == [False, True]


def test_standing_command_freezes_randomized_gait_phase(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        env_config=BelloEnvConfig(
            gait=GaitConfig(randomize_frequency=True, frequency_range=(1.2, 1.5))
        ),
    )
    state, _ = env.reset(rng=jax.random.PRNGKey(0))
    state = state.replace(step_count=jnp.asarray(20))
    standing = state.replace(command=jnp.zeros(3))
    moving = state.replace(command=jnp.asarray([0.2, 0.0, 0.0]))

    assert env._gait_phase(standing) == pytest.approx(0.0)
    assert env._gait_phase(moving) > 0.0
    assert env._foot_phase_height_target(standing) == pytest.approx(jnp.zeros(2))


def test_observation_noise_changes_only_noisy_evaluation(
    full_body_asset: PreparedAsset,
) -> None:
    noise = ObservationNoiseConfig(enabled=True, joint_pos=(-0.1, 0.1))
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        env_config=BelloEnvConfig(observation=ObservationConfig(noise=noise)),
    )
    state, clean = env.reset()
    noisy = env.observation(state, jax.random.PRNGKey(0))

    assert clean.shape == noisy.shape
    assert not jnp.allclose(clean, noisy)
    assert env.observation(state) == pytest.approx(clean)


def test_command_sampling_and_periodic_resampling_are_bounded(
    full_body_asset: PreparedAsset,
) -> None:
    command_config = CommandConfig(
        lin_vel_x=(0.2, 0.4),
        lin_vel_y=(-0.1, 0.1),
        ang_vel_yaw=(-0.2, 0.2),
        standing_probability=0.0,
        resample_seconds=0.02,
    )
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset, batch_size=4, command_config=command_config
    )
    state, _ = env.reset_batch(4, rng=jax.random.PRNGKey(0))
    next_state, *_ = env.step_batch(
        state,
        jnp.zeros((4, env.action_size)),
        rng=jax.random.PRNGKey(1),
    )

    assert jnp.all((state.command[:, 0] >= 0.2) & (state.command[:, 0] <= 0.4))
    assert not jnp.allclose(next_state.command, state.command)


def test_random_push_overwrites_root_xy_velocity_on_interval(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        batch_size=2,
        env_config=BelloEnvConfig(
            random_push=RandomPushConfig(
                enabled=True,
                interval_seconds=0.02,
                lin_vel_x=(0.5, 0.5),
                lin_vel_y=(-0.25, -0.25),
            )
        ),
    )
    state, _ = env.reset_batch(2)
    data, applied, velocity = env._maybe_apply_random_push_batch(
        state.data,
        jnp.ones(2, dtype=jnp.int32),
        jax.random.PRNGKey(0),
    )
    root_qvel = env.model.root_qvel_adr

    assert applied.tolist() == [True, True]
    assert velocity == pytest.approx(jnp.asarray([[0.5, -0.25]] * 2))
    assert data.qvel[:, root_qvel : root_qvel + 2] == pytest.approx(velocity)


def test_rough_terrain_foot_height_queries_local_elevation(
    full_body_asset: PreparedAsset,
) -> None:
    terrain = TerrainConfig(
        mode="rough",
        hfield_resolution=17,
        hfield_amplitude=0.02,
        hfield_frequency=3.0,
    )
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        env_config=BelloEnvConfig(terrain=terrain, spawn=SpawnConfig(clearance=0.01)),
    )
    state, _ = env.reset()
    foot_position = np.asarray(state.data.xpos[env.model.foot_body_ids])
    expected_ground = terrain_surface_height(
        env.model.mj_model,
        int(env.model.terrain_geom_ids[0]),
        foot_position[:, :2],
        geom_quat=env.model.terrain_geom_quat[0],
    )
    expected_height = foot_position[:, 2] - expected_ground

    assert env._foot_height(state, 0) == pytest.approx(expected_height, abs=1.0e-5)
    mj_data = mujoco.MjData(env.model.mj_model)
    mj_data.qpos[:] = np.asarray(state.data.qpos)
    mj_data.qvel[:] = np.asarray(state.data.qvel)
    mujoco.mj_forward(env.model.mj_model, mj_data)
    assert mujoco_foot_height(env, mj_data) == pytest.approx(
        expected_height, abs=1.0e-5
    )


def test_done_flags_share_height_orientation_timeout_and_nonfinite_contract(
    full_body_asset: PreparedAsset,
) -> None:
    done_config = DoneConfig(
        base_height=BaseHeightDoneConfig(min_height=0.8, link="bello_root"),
        bad_orientation=BadOrientationDoneConfig(limit_angle=0.5, link="bello_root"),
    )
    env = BelloLocomotionEnv.load_prepared(full_body_asset, done_config=done_config)
    state, _ = env.reset()
    root = env.model.root_qpos_adr
    low = state.replace(
        data=state.data.replace(qpos=state.data.qpos.at[root + 2].set(0.7))
    )
    nonfinite = state.replace(
        data=state.data.replace(qvel=state.data.qvel.at[0].set(jnp.nan))
    )

    assert bool(env.done_flags(low)["done/base_height"])
    assert bool(env.done_flags(nonfinite)["done/nonfinite"])
    assert not bool(env.done_flags(state)["done/any"])


@pytest.mark.parametrize(
    "env_config, message",
    [
        (BelloEnvConfig(action_clip=0.0), "action_clip"),
        (BelloEnvConfig(root_link=""), "root_link"),
        (
            BelloEnvConfig(contact=ContactConfig(foot_contact_on_seconds=-0.1)),
            "foot_contact_on_seconds",
        ),
        (
            BelloEnvConfig(observation=ObservationConfig(privileged=True)),
            "actor observation must remain deployable",
        ),
    ],
)
def test_environment_rejects_invalid_static_contracts(
    full_body_asset: PreparedAsset, env_config: BelloEnvConfig, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        BelloLocomotionEnv.load_prepared(full_body_asset, env_config=env_config)


def test_environment_rejects_missing_reward_link(
    full_body_asset: PreparedAsset,
) -> None:
    reward = RewardConfig(
        track_lin_vel_xy=replace(RewardConfig().track_lin_vel_xy, link="missing_link")
    )
    with pytest.raises(ValueError, match="missing_link"):
        BelloLocomotionEnv.load_prepared(full_body_asset, reward_config=reward)


def test_observation_and_action_layouts_reject_ambiguous_contracts() -> None:
    with pytest.raises(ValueError, match="component names must be unique"):
        ObservationLayout((("joint", 1), ("joint", 2)))
    with pytest.raises(ValueError, match="sizes must be positive"):
        ObservationLayout((("joint", 0),))
    with pytest.raises(ValueError, match="names must be unique"):
        ActionLayout(("left", "left"))
