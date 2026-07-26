from dataclasses import replace

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
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
    ObservationConfig,
    ObservationNoiseConfig,
    RandomPushConfig,
    SpawnConfig,
)
from bello_mujoco.envs.rewards import CommandConfig, RewardConfig, RewardLinks
from bello_mujoco.render import (
    mujoco_done_flags,
    mujoco_eval_info,
    mujoco_eval_state,
    mujoco_foot_height,
    mujoco_link_ang_vel,
    mujoco_link_lin_acc,
    mujoco_link_lin_vel,
    mujoco_policy_observation,
    reset_mujoco_data,
)
from bello_mujoco.sim import ModelRandomizationConfig
from bello_mujoco.terrain import TerrainConfig, terrain_surface_height


def observation_component_slice(
    env: BelloLocomotionEnv, component_name: str, *, critic: bool = False
) -> slice:
    offset = 0
    layout = env.critic_observation_layout if critic else env.observation_layout
    for name, size in layout.components:
        if name == component_name:
            return slice(offset, offset + size)
        offset += size
    raise AssertionError(f"observation component not found: {component_name}")


def test_reset_exposes_explicit_actor_critic_and_action_contracts(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)
    state, observation = env.reset()

    assert env.action_layout.size == env.action_size == 13
    assert set(env.action_layout.names) == {
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
    }
    assert env.observation_layout.components == (
        ("imu_linear_acceleration", 3),
        ("imu_angular_velocity", 3),
        ("projected_gravity", 3),
        ("command", 3),
        ("joint_position", 13),
        ("joint_velocity", 13),
        ("last_action", 13),
    )
    assert env.critic_observation_layout.components[-3:] == (
        ("base_linear_velocity", 3),
        ("base_height", 1),
        ("actuator_force", 13),
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
    assert "foot_phase" not in actor_names
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


def test_native_data_adapter_preserves_constraint_forces(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset, load_mjx=False)
    mj_data = mujoco.MjData(env.model.mj_model)
    reset_mujoco_data(env, mj_data)
    mj_data.qpos[env.model.root_qpos_adr + 2] -= 0.02
    mujoco.mj_forward(env.model.mj_model, mj_data)
    assert mj_data.efc_force.size > 0

    expected = np.asarray(mj_data.efc_force, dtype=np.float32).copy()
    state = mujoco_eval_state(
        env,
        mj_data,
        command=np.zeros(3, dtype=np.float32),
        last_action=np.zeros(env.action_size, dtype=np.float32),
        foot_air_time=jnp.zeros(2, dtype=jnp.float32),
        foot_contact_time=jnp.zeros(2, dtype=jnp.float32),
        foot_contact_debounce_time=jnp.zeros(2, dtype=jnp.float32),
        last_foot_contact=jnp.zeros(2, dtype=jnp.float32),
        step_count=0,
        episode_return=jnp.asarray(0.0),
        episode_track_lin_vel_xy=jnp.asarray(0.0),
        episode_length=jnp.asarray(0),
    )

    assert state.data._impl.efc__force == pytest.approx(expected)

    mj_data.efc_force[0] = 2.0e6
    assert mujoco_done_flags(env, mj_data, 0)["done/unstable"]


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
    assert "done/unstable" in info


def test_step_batch_skips_full_reset_when_no_world_is_done(
    full_body_asset: PreparedAsset, monkeypatch
) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset, batch_size=2)
    state, _ = env.reset_batch(2)
    reset_callbacks = []
    original_reset_batch = BelloLocomotionEnv.reset_batch

    def instrumented_reset_batch(self, *args, **kwargs):
        jax.debug.callback(lambda _: reset_callbacks.append(True), jnp.asarray(0))
        return original_reset_batch(self, *args, **kwargs)

    monkeypatch.setattr(
        BelloLocomotionEnv,
        "reset_batch",
        instrumented_reset_batch,
    )
    step = jax.jit(
        lambda item: env.step_batch(
            item,
            jnp.zeros((2, env.action_size), dtype=jnp.float32),
        )
    )

    _, observation, _, done, _ = step(state)
    observation.block_until_ready()
    assert not jnp.any(done)
    assert reset_callbacks == []

    timed_out = state.replace(
        step_count=jnp.full(2, env.max_episode_steps, dtype=jnp.int32)
    )
    _, observation, _, done, _ = step(timed_out)
    observation.block_until_ready()
    assert jnp.all(done)
    assert reset_callbacks == [True]


def test_reset_forward_populates_velocity_derived_imu_observation(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        env_config=BelloEnvConfig(
            spawn=SpawnConfig(
                joint_vel_range=(0.1, 0.1),
                root_vel_range=(0.25, 0.25),
            )
        ),
    )

    state, observation = env.reset(rng=jax.random.PRNGKey(3))
    pelvis_id = env._link_body_id(env.model.imu_body_name)
    imu_ang_vel = observation[observation_component_slice(env, "imu_angular_velocity")]

    assert jnp.allclose(state.data.qvel[:6], 0.25)
    assert not jnp.allclose(state.data.cvel[pelvis_id], 0.0)
    assert imu_ang_vel == pytest.approx(env._imu_angular_velocity(state))
    assert not jnp.allclose(imu_ang_vel, 0.0)


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
    next_state, _, _, _, info = env.step_batch(
        state,
        jnp.zeros((4, env.action_size)),
        rng=jax.random.PRNGKey(1),
    )

    assert jnp.all((state.command[:, 0] >= 0.2) & (state.command[:, 0] <= 0.4))
    assert info["command/standing"] == pytest.approx(jnp.zeros(4))
    assert info["command/moving"] == pytest.approx(jnp.ones(4))
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
        size=(0.5, 0.5, 0.05),
        hfield_resolution=17,
        hfield_amplitude=0.03,
        hfield_frequency=8.0,
    )
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        env_config=BelloEnvConfig(terrain=terrain, spawn=SpawnConfig(clearance=0.01)),
    )
    box_geom_ids = [
        next(
            int(geom_id)
            for geom_id in env.model.foot_geom_ids
            if env.model.geom_body_ids[geom_id] == foot_body_id
            and env.model.mj_model.geom_type[geom_id] == int(mujoco.mjtGeom.mjGEOM_BOX)
        )
        for foot_body_id in env.model.foot_body_ids
    ]
    env = replace(
        env,
        model=replace(
            env.model,
            foot_geom_ids=np.asarray(box_geom_ids, dtype=np.int32),
        ),
    )
    state, _ = env.reset()
    terrain_geom_id = int(env.model.terrain_geom_ids[0])
    expected_height = []
    for geom_id in box_geom_ids:
        signs = np.asarray(
            [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
            dtype=np.float32,
        )
        corners = signs * np.asarray(env.model.mj_model.geom_size[geom_id])
        rotation = np.asarray(state.data.geom_xmat[geom_id])
        world_corners = np.asarray(state.data.geom_xpos[geom_id]) + corners @ rotation.T
        terrain_height = terrain_surface_height(
            env.model.mj_model,
            terrain_geom_id,
            world_corners[:, :2],
            geom_quat=env.model.terrain_geom_quat[0],
        )
        expected_height.append(np.min(world_corners[:, 2] - terrain_height))

    assert env._foot_height(state, 0) == pytest.approx(expected_height, abs=1.0e-5)
    mj_data = mujoco.MjData(env.model.mj_model)
    mj_data.qpos[:] = np.asarray(state.data.qpos)
    mj_data.qvel[:] = np.asarray(state.data.qvel)
    mujoco.mj_forward(env.model.mj_model, mj_data)
    assert mujoco_foot_height(env, mj_data) == pytest.approx(
        expected_height, abs=1.0e-5
    )


def test_foot_height_uses_physical_sole_instead_of_ankle_origin(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)
    state, _ = env.reset()

    sole_clearance = env._foot_height(state)
    ankle_height = state.data.xpos[jnp.asarray(env.model.foot_body_ids), 2]

    assert jnp.all(sole_clearance >= 0.0)
    assert jnp.all(ankle_height - sole_clearance > 0.05)


def test_sloped_plane_foot_height_uses_minimum_terrain_clearance(
    full_body_asset: PreparedAsset,
) -> None:
    terrain = TerrainConfig(
        mode="plane",
        slope_angle_range=(np.deg2rad(10.0), np.deg2rad(10.0)),
    )
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        env_config=BelloEnvConfig(terrain=terrain),
    )
    box_geom_ids = []
    for foot_body_id in env.model.foot_body_ids:
        box_geom_ids.append(
            next(
                int(geom_id)
                for geom_id in env.model.foot_geom_ids
                if env.model.geom_body_ids[geom_id] == foot_body_id
                and env.model.mj_model.geom_type[geom_id]
                == int(mujoco.mjtGeom.mjGEOM_BOX)
            )
        )
    env = replace(
        env,
        model=replace(
            env.model,
            foot_geom_ids=np.asarray(box_geom_ids, dtype=np.int32),
        ),
    )
    state, _ = env.reset()
    terrain_geom_id = int(env.model.terrain_geom_ids[0])
    expected = []
    for geom_id in box_geom_ids:
        signs = np.asarray(
            [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
            dtype=np.float32,
        )
        corners = signs * np.asarray(env.model.mj_model.geom_size[geom_id])
        rotation = np.asarray(state.data.geom_xmat[geom_id])
        world_corners = np.asarray(state.data.geom_xpos[geom_id]) + corners @ rotation.T
        terrain_height = terrain_surface_height(
            env.model.mj_model,
            terrain_geom_id,
            world_corners[:, :2],
            geom_quat=env.model.terrain_geom_quat[0],
        )
        expected.append(np.min(world_corners[:, 2] - terrain_height))

    mj_data = mujoco.MjData(env.model.mj_model)
    mj_data.qpos[:] = np.asarray(state.data.qpos)
    mj_data.qvel[:] = np.asarray(state.data.qvel)
    mujoco.mj_forward(env.model.mj_model, mj_data)

    assert env._foot_height(state) == pytest.approx(expected, abs=1.0e-5)
    assert mujoco_foot_height(env, mj_data) == pytest.approx(expected, abs=1.0e-5)


def test_base_height_and_critic_observation_are_terrain_relative(
    full_body_asset: PreparedAsset,
) -> None:
    terrain = TerrainConfig(
        mode="plane",
        slope_angle_range=(np.deg2rad(2.0), np.deg2rad(2.0)),
    )
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        env_config=BelloEnvConfig(terrain=terrain, root_link="pelvis_link"),
    )
    state, _ = env.reset()
    root_qpos = env.model.root_qpos_adr
    original_xy = state.data.qpos[root_qpos : root_qpos + 2]
    translated_xy = original_xy + jnp.asarray([0.5, -0.25])
    terrain_delta = env._terrain_height(translated_xy) - env._terrain_height(
        original_xy
    )
    translated_data = state.data.replace(
        qpos=(
            state.data.qpos.at[root_qpos : root_qpos + 2]
            .set(translated_xy)
            .at[root_qpos + 2]
            .add(terrain_delta)
        )
    )
    translated = state.replace(
        data=mjx.kinematics(env.model.mjx_model, translated_data)
    )
    base_height_slice = observation_component_slice(env, "base_height", critic=True)

    assert env._base_height(translated) == pytest.approx(
        env._base_height(state), abs=1.0e-5
    )
    assert env.critic_observation(translated)[base_height_slice] == pytest.approx(
        env.critic_observation(state)[base_height_slice], abs=1.0e-5
    )
    assert bool(env.done_flags(translated)["done/base_height"]) == bool(
        env.done_flags(state)["done/base_height"]
    )
    for sample in (state, translated):
        mj_data = mujoco.MjData(env.model.mj_model)
        mj_data.qpos[:] = np.asarray(sample.data.qpos)
        mj_data.qvel[:] = np.asarray(sample.data.qvel)
        mujoco.mj_forward(env.model.mj_model, mj_data)
        assert mujoco_eval_info(env, mj_data)["base_height"] == pytest.approx(
            env._base_height(sample), abs=1.0e-5
        )


def test_batched_slope_resets_to_identical_sole_clearance_without_rng(
    full_body_asset: PreparedAsset,
) -> None:
    terrain = TerrainConfig(
        mode="plane",
        pos=(1.5, -0.75, 0.2),
        seed=19,
        slope_angle_range=(-np.deg2rad(2.0), np.deg2rad(2.0)),
    )
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        batch_size=8,
        env_config=BelloEnvConfig(
            terrain=terrain,
            spawn=SpawnConfig(clearance=0.01),
            model_randomization=ModelRandomizationConfig(enabled=True),
        ),
    )

    state, _ = env.reset_batch(8, rng=None)
    minimum_clearance = jax.vmap(lambda item: jnp.min(env._foot_height(item)))(state)

    assert minimum_clearance == pytest.approx(jnp.full(8, 0.01), abs=2.0e-5)
    assert int(np.ravel(np.asarray(state.data._impl.nacon))[0]) == 0


def test_native_and_mjx_reset_share_slope_relative_sole_clearance(
    full_body_asset: PreparedAsset,
) -> None:
    terrain = TerrainConfig(
        mode="plane",
        pos=(1.5, -0.75, 0.2),
        slope_angle_range=(np.deg2rad(2.0), np.deg2rad(2.0)),
    )
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        env_config=BelloEnvConfig(
            terrain=terrain,
            spawn=SpawnConfig(clearance=0.01),
        ),
    )
    state, _ = env.reset()
    mj_data = mujoco.MjData(env.model.mj_model)
    reset_mujoco_data(env, mj_data)

    assert np.min(mujoco_foot_height(env, mj_data)) == pytest.approx(0.01, abs=1.0e-5)
    assert np.min(np.asarray(env._foot_height(state))) == pytest.approx(
        0.01, abs=1.0e-5
    )
    assert mj_data.qpos[env.model.root_qpos_adr + 2] == pytest.approx(
        np.asarray(state.data.qpos)[env.model.root_qpos_adr + 2], abs=1.0e-5
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


def test_large_finite_physics_state_terminates_and_sanitizes_reward(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)
    state, _ = env.reset()
    exploded = state.replace(
        data=state.data.replace(qvel=state.data.qvel.at[0].set(1.0e5))
    )
    flags = env.done_flags(exploded)
    reward, terms = env.sanitize_nonfinite_reward(
        jnp.asarray(-1.0e20),
        {
            "reward/joint_pose": jnp.asarray(-1.0e20),
            "reward/termination": jnp.asarray(-1.0),
        },
        flags,
    )

    assert bool(flags["done/unstable"])
    assert not bool(flags["done/nonfinite"])
    assert terms["reward/joint_pose"] == pytest.approx(0.0)
    assert terms["reward/termination"] == pytest.approx(-1.0)
    assert reward == pytest.approx(-1.0)


def test_large_or_nonfinite_constraint_force_terminates_episode(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)
    state, _ = env.reset()
    large_force = state.replace(
        data=state.data.replace(
            qfrc_constraint=state.data.qfrc_constraint.at[0].set(2.0e6)
        )
    )
    nonfinite_force = state.replace(
        data=state.data.replace(
            qfrc_constraint=state.data.qfrc_constraint.at[0].set(jnp.nan)
        )
    )
    large_contact_force = state.replace(
        data=state.data.replace(
            _impl=state.data._impl.replace(
                efc__force=state.data._impl.efc__force.at[0].set(2.0e6)
            )
        )
    )
    nonfinite_contact_force = state.replace(
        data=state.data.replace(
            _impl=state.data._impl.replace(
                efc__force=state.data._impl.efc__force.at[0].set(jnp.nan)
            )
        )
    )

    assert bool(env.done_flags(large_force)["done/unstable"])
    assert bool(env.done_flags(nonfinite_force)["done/nonfinite"])
    assert bool(env.done_flags(large_contact_force)["done/unstable"])
    assert bool(env.done_flags(nonfinite_contact_force)["done/nonfinite"])


def test_stance_width_uses_lateral_frame_axis(full_body_asset: PreparedAsset) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)
    state, _ = env.reset()
    left, right = env.model.foot_body_ids
    xpos = (
        state.data.xpos.at[left]
        .set(jnp.asarray([5.0, 0.2, 0.0]))
        .at[right]
        .set(jnp.asarray([-5.0, 0.0, 0.0]))
    )
    state = state.replace(data=state.data.replace(xpos=xpos))

    assert env._stance_width(state, env.config.root_link) == pytest.approx(0.2)


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
        (
            BelloEnvConfig(
                critic_observation=ObservationConfig(
                    privileged=True,
                    noise=ObservationNoiseConfig(enabled=True),
                )
            ),
            "critic_observation.noise.enabled must be false",
        ),
        (
            BelloEnvConfig(
                numerical_safety=replace(
                    BelloEnvConfig().numerical_safety, max_abs_velocity=0.0
                )
            ),
            "numerical_safety.max_abs_velocity",
        ),
        (
            BelloEnvConfig(
                numerical_safety=replace(
                    BelloEnvConfig().numerical_safety,
                    max_abs_constraint_force=0.0,
                )
            ),
            "numerical_safety.max_abs_constraint_force",
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
    reward = RewardConfig(links=RewardLinks(track_lin_vel_xy="missing_link"))
    with pytest.raises(ValueError, match="missing_link"):
        BelloLocomotionEnv.load_prepared(full_body_asset, reward_config=reward)


def test_observation_and_action_layouts_reject_ambiguous_contracts() -> None:
    with pytest.raises(ValueError, match="component names must be unique"):
        ObservationLayout((("joint", 1), ("joint", 2)))
    with pytest.raises(ValueError, match="sizes must be positive"):
        ObservationLayout((("joint", 0),))
    with pytest.raises(ValueError, match="names must be unique"):
        ActionLayout(("left", "left"))
