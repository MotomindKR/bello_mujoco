from dataclasses import replace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from bello_mujoco.actuators import ActuatorConfig
from bello_mujoco.assets_pipeline.prepare import PreparedAsset
from bello_mujoco.contracts import DEFAULT_SPAWN_CLEARANCE
from bello_mujoco.envs.locomotion import (
    BelloEnvConfig,
    BelloLocomotionEnv,
)
from bello_mujoco.render import mujoco_action_to_ctrl, mujoco_policy_observation
from bello_mujoco.sim import (
    ModelRandomizationConfig,
    SimulationPhysicsConfig,
    configured_actuator_joint_range,
    load_prepared_bello_model,
    robot_geom_min_z,
)
from bello_mujoco.terrain import (
    FLAT_TERRAIN,
    ROBOT_ONLY_TERRAIN,
    ROUGH_TERRAIN,
    TerrainConfig,
    random_slope_angles,
    random_slope_angles_batch,
    rough_height_data,
    slope_quat,
    terrain_surface_height,
)


def observation_component_slice(env: BelloLocomotionEnv, component_name: str) -> slice:
    offset = 0
    for name, size in env.observation_layout.components:
        if name == component_name:
            return slice(offset, offset + size)
        offset += size
    raise AssertionError(f"observation component not found: {component_name}")


def test_full_body_model_loads_with_all_actuators_in_policy_by_default(
    full_body_asset: PreparedAsset,
) -> None:
    model = load_prepared_bello_model(full_body_asset)

    assert model.mj_model.nq == 42
    assert model.mj_model.nv == 41
    assert model.mj_model.nu == 35
    assert model.policy_ctrl_indices.tolist() == list(range(35))
    assert len(model.hold_ctrl_indices) == 0
    expected_range = []
    for joint_name in model.actuator_joint_names:
        low, high = configured_actuator_joint_range(
            model.mj_model,
            joint_name,
            ActuatorConfig(),
        )
        expected_range.append((low, high))
    np.testing.assert_allclose(model.joint_target_low, np.asarray(expected_range)[:, 0])
    np.testing.assert_allclose(
        model.joint_target_high, np.asarray(expected_range)[:, 1]
    )


def test_policy_action_endpoints_reach_logical_soft_limits(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)
    policy_indices = env.model.policy_ctrl_indices

    low_target = np.asarray(env.action_to_joint_target(-np.ones(env.action_size)))
    neutral_target = np.asarray(env.action_to_joint_target(np.zeros(env.action_size)))
    high_target = np.asarray(env.action_to_joint_target(np.ones(env.action_size)))

    np.testing.assert_allclose(
        low_target[policy_indices], env.model.joint_target_low[policy_indices]
    )
    np.testing.assert_allclose(
        neutral_target[policy_indices],
        env.model.default_joint_target[policy_indices]
        + env.model.joint_angle_offset[0, policy_indices],
    )
    np.testing.assert_allclose(
        high_target[policy_indices], env.model.joint_target_high[policy_indices]
    )


def test_differential_ankle_joint_targets_map_to_both_motors(
    full_body_asset: PreparedAsset,
) -> None:
    model = load_prepared_bello_model(
        full_body_asset,
        terrain=FLAT_TERRAIN,
    )
    env = BelloLocomotionEnv(model)
    action = np.zeros(env.action_size, dtype=np.float32)
    motor_1_id = mujoco.mj_name2id(
        model.mj_model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "left_ankle_motor_1_pos",
    )
    motor_2_id = mujoco.mj_name2id(
        model.mj_model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "left_ankle_motor_2_pos",
    )
    pitch_action = env.action_layout.names.index("left_ankle_pitch_joint")
    roll_action = env.action_layout.names.index("left_ankle_roll_joint")
    action[pitch_action] = 1.0

    ctrl = np.asarray(env.action_to_ctrl(action))

    pitch_high = model.joint_target_high[model.policy_ctrl_indices[pitch_action]]
    roll_default = model.default_joint_target[model.policy_ctrl_indices[roll_action]]
    assert ctrl[motor_1_id] == pytest.approx(roll_default - pitch_high)
    assert ctrl[motor_2_id] == pytest.approx(roll_default + pitch_high)

    action[:] = 0.0
    action[roll_action] = 1.0
    ctrl = np.asarray(env.action_to_ctrl(action))
    roll_high = model.joint_target_high[model.policy_ctrl_indices[roll_action]]
    pitch_default = model.default_joint_target[model.policy_ctrl_indices[pitch_action]]
    assert ctrl[motor_1_id] == pytest.approx(roll_high - pitch_default)
    assert ctrl[motor_2_id] == pytest.approx(roll_high + pitch_default)


@pytest.mark.parametrize(
    ("pitch_action_value", "roll_action_value"),
    [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)],
)
def test_differential_ankle_soft_limit_corners_fit_motor_ranges(
    full_body_asset: PreparedAsset,
    pitch_action_value: float,
    roll_action_value: float,
) -> None:
    model = load_prepared_bello_model(full_body_asset, terrain=FLAT_TERRAIN)
    env = BelloLocomotionEnv(model)
    action = np.zeros(env.action_size, dtype=np.float32)
    action[env.action_layout.names.index("left_ankle_pitch_joint")] = pitch_action_value
    action[env.action_layout.names.index("left_ankle_roll_joint")] = roll_action_value

    ctrl, joint_saturation, motor_saturation = env._action_target_details(action)
    ctrl = np.asarray(ctrl)
    motor_ids = np.asarray(
        [
            mujoco.mj_name2id(
                model.mj_model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"left_ankle_motor_{index}_pos",
            )
            for index in (1, 2)
        ]
    )
    motor_ranges = model.mj_model.actuator_ctrlrange[motor_ids]

    assert float(joint_saturation) == pytest.approx(0.0)
    assert float(motor_saturation) == pytest.approx(0.0)
    assert np.all(ctrl[motor_ids] >= motor_ranges[:, 0] - 1.0e-6)
    assert np.all(ctrl[motor_ids] <= motor_ranges[:, 1] + 1.0e-6)


def test_differential_ankle_observations_use_joint_coordinates(
    full_body_asset: PreparedAsset,
) -> None:
    model = load_prepared_bello_model(
        full_body_asset,
        terrain=FLAT_TERRAIN,
    )
    env = BelloLocomotionEnv(model)
    state, _ = env.reset()
    motor_1_id = mujoco.mj_name2id(
        model.mj_model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "left_ankle_motor_1_pos",
    )
    motor_2_id = mujoco.mj_name2id(
        model.mj_model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "left_ankle_motor_2_pos",
    )
    pitch_qpos = int(model.actuator_qpos_indices[motor_1_id])
    roll_qpos = int(model.actuator_qpos_indices[motor_2_id])
    pitch_qvel = int(model.actuator_qvel_indices[motor_1_id])
    roll_qvel = int(model.actuator_qvel_indices[motor_2_id])
    qpos = state.data.qpos.at[pitch_qpos].add(0.10).at[roll_qpos].add(0.02)
    qvel = state.data.qvel.at[pitch_qvel].set(0.30).at[roll_qvel].set(-0.10)
    state = state.replace(data=state.data.replace(qpos=qpos, qvel=qvel))

    observation = np.asarray(env.observation(state))
    position = observation[observation_component_slice(env, "joint_position")]
    velocity = observation[observation_component_slice(env, "joint_velocity")]
    motor_1_action = env.action_layout.names.index("left_ankle_pitch_joint")
    motor_2_action = env.action_layout.names.index("left_ankle_roll_joint")

    assert position[motor_1_action] == pytest.approx(0.10)
    assert position[motor_2_action] == pytest.approx(0.02)
    assert velocity[motor_1_action] == pytest.approx(0.30)
    assert velocity[motor_2_action] == pytest.approx(-0.10)


def test_differential_control_matches_native_mujoco(
    full_body_asset: PreparedAsset,
) -> None:
    model = load_prepared_bello_model(
        full_body_asset,
        terrain=FLAT_TERRAIN,
    )
    env = BelloLocomotionEnv(model)
    action = np.linspace(-1.0, 1.0, env.action_size, dtype=np.float32)

    np.testing.assert_allclose(
        np.asarray(env.action_to_ctrl(action)),
        mujoco_action_to_ctrl(env, action),
        atol=1.0e-6,
    )


def test_differential_joint_observation_matches_native_mujoco(
    full_body_asset: PreparedAsset,
) -> None:
    model = load_prepared_bello_model(
        full_body_asset,
        terrain=FLAT_TERRAIN,
    )
    env = BelloLocomotionEnv(model)
    state, _ = env.reset()
    qpos = np.asarray(state.data.qpos).copy()
    qvel = np.asarray(state.data.qvel).copy()
    qpos[model.actuator_qpos_indices] += np.linspace(
        -0.05, 0.05, model.mj_model.nu, dtype=np.float32
    )
    qvel[model.actuator_qvel_indices] = np.linspace(
        -0.5, 0.5, model.mj_model.nu, dtype=np.float32
    )
    state = state.replace(
        data=state.data.replace(qpos=np.asarray(qpos), qvel=np.asarray(qvel))
    )
    mj_data = mujoco.MjData(model.mj_model)
    mj_data.qpos[:] = qpos
    mj_data.qvel[:] = qvel
    mujoco.mj_forward(model.mj_model, mj_data)

    mjx_observation = np.asarray(env.observation(state))
    native_observation = np.asarray(
        mujoco_policy_observation(
            env,
            mj_data,
            command=np.zeros(3, dtype=np.float32),
            last_action=np.zeros(env.action_size, dtype=np.float32),
            step_count=0,
        )
    )
    for component_name in ("joint_position", "joint_velocity"):
        component = observation_component_slice(env, component_name)
        np.testing.assert_allclose(
            mjx_observation[component],
            native_observation[component],
            atol=1.0e-6,
        )


def test_model_can_load_without_mjx_for_cpu_only_viewers(
    full_body_asset: PreparedAsset,
) -> None:
    model = load_prepared_bello_model(full_body_asset, load_mjx=False)

    assert model.mjx_model is None
    with pytest.raises(RuntimeError, match="loaded without MJX"):
        model.make_data()


def test_runtime_simulation_physics_are_applied_before_mjx_conversion(
    full_body_asset: PreparedAsset,
) -> None:
    physics = SimulationPhysicsConfig(
        timestep=0.002,
        foot_terrain_margin=0.005,
        foot_terrain_solref=(0.004, 1.0),
        foot_terrain_solimp=(0.98, 0.999, 0.001, 0.5, 2.0),
    )
    scene = load_prepared_bello_model(
        full_body_asset,
        terrain=FLAT_TERRAIN,
        simulation_physics=physics,
    )
    model = scene.mj_model
    contact_geom_ids = np.concatenate((scene.foot_geom_ids, scene.terrain_geom_ids))

    assert model.opt.timestep == pytest.approx(0.002)
    np.testing.assert_allclose(model.geom_margin[scene.terrain_geom_ids], 0.005)
    np.testing.assert_allclose(model.geom_margin[scene.foot_geom_ids], 0.0)
    np.testing.assert_allclose(
        model.geom_solref[contact_geom_ids],
        np.tile(np.asarray([0.004, 1.0]), (contact_geom_ids.size, 1)),
    )
    np.testing.assert_allclose(
        model.geom_solimp[contact_geom_ids],
        np.tile(
            np.asarray([0.98, 0.999, 0.001, 0.5, 2.0]),
            (contact_geom_ids.size, 1),
        ),
    )
    assert scene.mjx_model is not None
    assert float(np.asarray(scene.mjx_model.opt.timestep)) == pytest.approx(0.002)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"timestep": 0.0}, "timestep"),
        ({"foot_terrain_margin": -0.001}, "foot_terrain_margin"),
        (
            {"timestep": 0.002, "foot_terrain_solref": (0.003, 1.0)},
            "at least twice",
        ),
        ({"foot_terrain_solimp": (0.99, 0.95, 0.001, 0.5, 2.0)}, "impedance"),
    ],
)
def test_simulation_physics_rejects_invalid_values(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SimulationPhysicsConfig(**kwargs)


def test_model_randomization_batches_world_parameters_and_mirrors_first_sample(
    full_body_asset: PreparedAsset,
) -> None:
    nominal = load_prepared_bello_model(
        full_body_asset, terrain=FLAT_TERRAIN, load_mjx=False
    )
    randomized = load_prepared_bello_model(
        full_body_asset,
        terrain=FLAT_TERRAIN,
        batch_size=4,
        model_randomization=ModelRandomizationConfig(
            enabled=True,
            seed=7,
            joint_origin_range=(-0.001, 0.001),
            joint_angle_offset_range=(-0.02, 0.02),
            joint_inertia_scale_range=(0.95, 1.05),
        ),
    )
    assert randomized.mjx_model is not None
    mjx_model = randomized.mjx_model
    assert mjx_model.jnt_pos.shape == (4, randomized.mj_model.njnt, 3)
    assert mjx_model.body_inertia.shape == (4, randomized.mj_model.nbody, 3)
    assert randomized.joint_angle_offset.shape == (4, randomized.mj_model.nu)

    joint_ids = np.unique(randomized.mj_model.actuator_trnid[:, 0]).astype(np.int32)
    body_ids = np.unique(randomized.mj_model.jnt_bodyid[joint_ids]).astype(np.int32)
    jnt_pos = np.asarray(mjx_model.jnt_pos)
    body_inertia = np.asarray(mjx_model.body_inertia)
    joint_angle_offset = randomized.joint_angle_offset

    assert np.max(
        np.abs(jnt_pos[:, joint_ids] - nominal.mj_model.jnt_pos[joint_ids])
    ) <= (0.001 + 1.0e-7)
    assert not np.allclose(jnt_pos[0, joint_ids], jnt_pos[1, joint_ids])
    assert np.min(joint_angle_offset) >= -0.02 - 1.0e-7
    assert np.max(joint_angle_offset) <= 0.02 + 1.0e-7
    assert not np.allclose(joint_angle_offset[0], joint_angle_offset[1])
    np.testing.assert_allclose(randomized.mj_model.jnt_pos, jnt_pos[0], atol=1.0e-7)
    np.testing.assert_allclose(
        randomized.mj_model.body_inertia, body_inertia[0], atol=1.0e-7
    )

    nominal_inertia = nominal.mj_model.body_inertia[body_ids]
    nonzero = nominal_inertia > 0.0
    ratios = body_inertia[:, body_ids][..., nonzero] / nominal_inertia[nonzero]
    assert np.min(ratios) >= 0.95 - 1.0e-6
    assert np.max(ratios) <= 1.05 + 1.0e-6


def test_plane_slope_can_be_randomized_per_world(
    full_body_asset: PreparedAsset,
) -> None:
    terrain = TerrainConfig(
        mode="plane",
        seed=11,
        slope_angle_range=(-np.deg2rad(2.0), np.deg2rad(2.0)),
        randomize_slope_per_world=True,
    )
    model = load_prepared_bello_model(
        full_body_asset,
        terrain=terrain,
        batch_size=4,
    )
    expected = np.asarray(
        [slope_quat(*angles) for angles in random_slope_angles_batch(terrain, 4)]
    )

    np.testing.assert_allclose(model.terrain_geom_quat, expected, atol=1.0e-7)
    assert not np.allclose(model.terrain_geom_quat[0], model.terrain_geom_quat[1])
    terrain_geom_id = int(model.terrain_geom_ids[0])
    np.testing.assert_allclose(
        model.mj_model.geom_quat[terrain_geom_id], expected[0], atol=1.0e-7
    )
    assert model.mjx_model is not None
    np.testing.assert_allclose(
        np.asarray(model.mjx_model.geom_quat)[:, terrain_geom_id],
        expected,
        atol=1.0e-7,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"seed": -1}, "seed must be nonnegative"),
        ({"joint_origin_range": (float("nan"), 0.0)}, "joint_origin_range"),
        ({"joint_angle_offset_range": (0.1, -0.1)}, "joint_angle_offset_range"),
        ({"joint_angle_offset_scope": "bad"}, "joint_angle_offset_scope"),
        ({"joint_inertia_scale_range": (0.0, 1.0)}, "positive values"),
    ],
)
def test_model_randomization_config_rejects_invalid_ranges(
    kwargs: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ModelRandomizationConfig(**kwargs)


def test_joint_actuator_mode_controls_policy_action_partition(
    full_body_asset: PreparedAsset,
) -> None:
    default_config = ActuatorConfig()
    joints = {
        name: (replace(config, mode="hold") if name == "waist_yaw_joint" else config)
        for name, config in default_config.joints.items()
    }
    model = load_prepared_bello_model(
        full_body_asset,
        actuator_config=ActuatorConfig(joints=joints),
    )

    assert model.mj_model.nu == 35
    assert len(model.policy_ctrl_indices) == 34
    assert len(model.hold_ctrl_indices) == 1


def test_policy_joint_angle_offsets_leave_hold_targets_unchanged(
    full_body_asset: PreparedAsset,
) -> None:
    default_config = ActuatorConfig()
    joints = {
        name: replace(config, mode="hold") if name == "waist_yaw_joint" else config
        for name, config in default_config.joints.items()
    }
    model = load_prepared_bello_model(
        full_body_asset,
        batch_size=4,
        actuator_config=ActuatorConfig(joints=joints),
        model_randomization=ModelRandomizationConfig(
            enabled=True,
            seed=7,
            joint_angle_offset_range=(-0.005, 0.005),
            joint_angle_offset_scope="policy",
        ),
    )

    assert np.all(model.joint_angle_offset[:, model.hold_ctrl_indices] == 0.0)
    assert np.any(model.joint_angle_offset[:, model.policy_ctrl_indices] != 0.0)


def test_randomized_neutral_targets_preserve_action_soft_limit_endpoints(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        batch_size=4,
        env_config=BelloEnvConfig(
            model_randomization=ModelRandomizationConfig(
                enabled=True,
                seed=11,
                joint_angle_offset_range=(-0.02, 0.02),
            )
        ),
    )
    offsets = env._joint_angle_offset_batch(4)
    low_actions = -jnp.ones((4, env.action_size), dtype=jnp.float32)
    zero_actions = jnp.zeros((4, env.action_size), dtype=jnp.float32)
    high_actions = jnp.ones((4, env.action_size), dtype=jnp.float32)
    low_targets = jax.vmap(env.action_to_joint_target)(low_actions, offsets)
    neutral_targets = jax.vmap(env.action_to_joint_target)(zero_actions, offsets)
    high_targets = jax.vmap(env.action_to_joint_target)(high_actions, offsets)
    policy_indices = env.model.policy_ctrl_indices

    np.testing.assert_allclose(
        np.asarray(low_targets)[:, policy_indices],
        np.broadcast_to(
            env.model.joint_target_low[policy_indices], (4, env.action_size)
        ),
    )
    np.testing.assert_allclose(
        np.asarray(neutral_targets)[:, policy_indices],
        np.clip(
            env.model.default_joint_target[None, policy_indices]
            + env.model.joint_angle_offset[:, policy_indices],
            env.model.joint_target_low[None, policy_indices],
            env.model.joint_target_high[None, policy_indices],
        ),
    )
    np.testing.assert_allclose(
        np.asarray(high_targets)[:, policy_indices],
        np.broadcast_to(
            env.model.joint_target_high[policy_indices], (4, env.action_size)
        ),
    )


def test_model_partitions_actuators_by_joint_not_actuator_name(
    full_body_asset: PreparedAsset,
) -> None:
    xml = full_body_asset.mjcf_path.read_text()
    xml = xml.replace("_pos", "_motor")
    xml = xml.replace("_hold", "_target")
    path = full_body_asset.mjcf_path.with_name("renamed_actuators.xml")
    path.write_text(xml)

    model = load_prepared_bello_model(
        PreparedAsset(
            name="full_body",
            urdf_path=full_body_asset.urdf_path,
            mjcf_path=path,
            viewer_mjcf_path=full_body_asset.viewer_mjcf_path,
            collision_mesh_count=full_body_asset.collision_mesh_count,
            visual_mesh_count=full_body_asset.visual_mesh_count,
        )
    )

    assert len(model.policy_ctrl_indices) == 35
    assert len(model.hold_ctrl_indices) == 0


def test_training_model_uses_simplified_mesh_collision_profile(
    full_body_asset: PreparedAsset,
) -> None:
    model = load_prepared_bello_model(full_body_asset).mj_model
    mesh_geoms = [
        geom_id
        for geom_id, geom_type in enumerate(model.geom_type)
        if geom_type == mujoco.mjtGeom.mjGEOM_MESH
    ]
    collision_mesh_geoms = [
        geom_id
        for geom_id in mesh_geoms
        if model.geom_contype[geom_id] != 0 or model.geom_conaffinity[geom_id] != 0
    ]
    collision_mesh_faces = [
        int(model.mesh_facenum[int(model.geom_dataid[geom_id])])
        for geom_id in collision_mesh_geoms
    ]
    box_collision_count = sum(
        1
        for geom_id, geom_type in enumerate(model.geom_type)
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX
        and (model.geom_contype[geom_id] != 0 or model.geom_conaffinity[geom_id] != 0)
    )
    cylinder_collision_count = sum(
        1
        for geom_id, geom_type in enumerate(model.geom_type)
        if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER
        and (model.geom_contype[geom_id] != 0 or model.geom_conaffinity[geom_id] != 0)
    )
    plane_count = sum(
        1 for geom_type in model.geom_type if geom_type == mujoco.mjtGeom.mjGEOM_PLANE
    )

    assert model.ngeom == 48
    assert len(collision_mesh_geoms) == 38
    assert 100_000 < sum(collision_mesh_faces) < 150_000
    assert min(collision_mesh_faces) >= 1_000
    assert max(collision_mesh_faces) > 20_000
    assert box_collision_count == 6
    assert cylinder_collision_count == 4
    assert plane_count == 0
    assert model.nexclude == 104


def test_viewer_model_adds_non_colliding_visual_meshes(
    full_body_asset: PreparedAsset,
) -> None:
    model = mujoco.MjModel.from_xml_path(str(full_body_asset.viewer_mjcf_path))
    visual_mesh_geoms = [
        geom_id
        for geom_id, geom_type in enumerate(model.geom_type)
        if geom_type == mujoco.mjtGeom.mjGEOM_MESH
        and model.geom_contype[geom_id] == 0
        and model.geom_conaffinity[geom_id] == 0
    ]

    assert model.ngeom == 88
    assert len(visual_mesh_geoms) == 40


def test_loading_viewer_model_preserves_policy_contract(
    full_body_asset: PreparedAsset,
) -> None:
    sim_model = load_prepared_bello_model(full_body_asset)
    viewer_model = load_prepared_bello_model(full_body_asset, use_viewer_model=True)

    assert viewer_model.xml_path == full_body_asset.viewer_mjcf_path
    assert viewer_model.viewer_xml_path == full_body_asset.viewer_mjcf_path
    assert viewer_model.mj_model.ngeom == sim_model.mj_model.ngeom + 40
    assert viewer_model.mj_model.nq == sim_model.mj_model.nq
    assert viewer_model.mj_model.nv == sim_model.mj_model.nv
    assert viewer_model.mj_model.nu == sim_model.mj_model.nu
    assert (
        viewer_model.policy_ctrl_indices.tolist()
        == sim_model.policy_ctrl_indices.tolist()
    )
    assert (
        viewer_model.hold_ctrl_indices.tolist() == sim_model.hold_ctrl_indices.tolist()
    )
    np.testing.assert_allclose(
        viewer_model.joint_target_low, sim_model.joint_target_low
    )
    np.testing.assert_allclose(
        viewer_model.joint_target_high, sim_model.joint_target_high
    )
    assert np.allclose(viewer_model.default_qpos, sim_model.default_qpos)


def test_runtime_plane_terrain_adds_ground_without_changing_robot_asset(
    full_body_asset: PreparedAsset,
) -> None:
    robot_model = load_prepared_bello_model(
        full_body_asset, terrain=ROBOT_ONLY_TERRAIN
    ).mj_model
    scene = load_prepared_bello_model(full_body_asset, terrain=FLAT_TERRAIN)
    scene_model = scene.mj_model
    plane_geoms = [
        geom_id
        for geom_id, geom_type in enumerate(scene_model.geom_type)
        if geom_type == mujoco.mjtGeom.mjGEOM_PLANE
    ]

    assert scene.xml_path != full_body_asset.mjcf_path
    assert full_body_asset.mjcf_path.exists()
    assert scene_model.ngeom == robot_model.ngeom + 1
    assert len(plane_geoms) == 1
    assert (
        mujoco.mj_id2name(scene_model, mujoco.mjtObj.mjOBJ_GEOM, plane_geoms[0])
        == "ground"
    )


def test_runtime_rough_terrain_uses_heightfield(
    full_body_asset: PreparedAsset,
) -> None:
    scene = load_prepared_bello_model(full_body_asset, terrain=ROUGH_TERRAIN)
    model = scene.mj_model
    hfield_geoms = [
        geom_id
        for geom_id, geom_type in enumerate(model.geom_type)
        if geom_type == mujoco.mjtGeom.mjGEOM_HFIELD
    ]

    assert scene.xml_path != full_body_asset.mjcf_path
    assert model.nhfield == 1
    assert len(hfield_geoms) == 1
    assert (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, hfield_geoms[0]) == "ground"
    )
    assert np.ptp(model.hfield_data) == pytest.approx(1.0)
    hfield_id = int(model.geom_dataid[hfield_geoms[0]])
    assert model.hfield_size[hfield_id, 2] == pytest.approx(
        ROUGH_TERRAIN.hfield_amplitude
    )


def test_runtime_rough_terrain_seed_changes_scene(
    full_body_asset: PreparedAsset,
) -> None:
    terrain_a = TerrainConfig(mode="rough", seed=1, hfield_resolution=17)
    terrain_b = TerrainConfig(mode="rough", seed=2, hfield_resolution=17)
    scene_a = load_prepared_bello_model(full_body_asset, terrain=terrain_a)
    scene_b = load_prepared_bello_model(full_body_asset, terrain=terrain_b)

    assert scene_a.xml_path != scene_b.xml_path
    assert rough_height_data(terrain_a) == rough_height_data(terrain_a)
    assert rough_height_data(terrain_a) != rough_height_data(terrain_b)


def test_runtime_rough_terrain_applies_seeded_slight_slope(
    full_body_asset: PreparedAsset,
) -> None:
    terrain = TerrainConfig(
        mode="rough",
        seed=11,
        hfield_resolution=17,
        slope_angle_range=(-0.005, 0.005),
    )
    scene = load_prepared_bello_model(full_body_asset, terrain=terrain)
    ground_id = mujoco.mj_name2id(
        scene.mj_model, mujoco.mjtObj.mjOBJ_GEOM, terrain.name
    )
    roll, pitch = random_slope_angles(terrain)

    assert -0.005 <= roll <= 0.005
    assert -0.005 <= pitch <= 0.005
    assert random_slope_angles(terrain) == (roll, pitch)
    assert not np.allclose(scene.mj_model.geom_quat[ground_id], [1.0, 0.0, 0.0, 0.0])
    data = mujoco.MjData(scene.mj_model)
    mujoco.mj_forward(scene.mj_model, data)
    ground_normal = data.geom_xmat[ground_id].reshape(3, 3)[:, 2]
    assert not np.allclose(ground_normal, [0.0, 0.0, 1.0])

    data.qpos[:] = scene.default_qpos
    mujoco.mj_forward(scene.mj_model, data)
    for xy in (np.asarray([1.0, 1.0]), np.asarray([-1.0, 0.5])):
        distance = mujoco.mj_ray(
            scene.mj_model,
            data,
            np.asarray([xy[0], xy[1], 1.0]),
            np.asarray([0.0, 0.0, -1.0]),
            None,
            True,
            -1,
            None,
        )
        expected_height = float(terrain_surface_height(scene.mj_model, ground_id, xy))
        assert 1.0 - distance == pytest.approx(expected_height, abs=1.0e-6)


@pytest.mark.parametrize(
    "slope_range",
    [
        (0.1, -0.1),
        (float("nan"), 0.0),
        (-np.pi / 2.0, 0.0),
    ],
)
def test_terrain_rejects_invalid_slope_range(
    slope_range: tuple[float, float],
) -> None:
    with pytest.raises(ValueError, match="slope_angle_range"):
        TerrainConfig(slope_angle_range=slope_range)


def test_default_spawn_pose_starts_above_flat_terrain(
    full_body_asset: PreparedAsset,
) -> None:
    model = load_prepared_bello_model(full_body_asset, terrain=FLAT_TERRAIN)
    min_robot_z = robot_geom_min_z(model.mj_model, model.default_qpos)
    ground_id = int(model.terrain_geom_ids[0])
    root_xy = model.default_qpos[model.root_qpos_adr : model.root_qpos_adr + 2]
    target_min_z = (
        float(terrain_surface_height(model.mj_model, ground_id, root_xy))
        + DEFAULT_SPAWN_CLEARANCE
    )

    assert min_robot_z is not None
    assert abs(min_robot_z - target_min_z) < 1.0e-5


def test_spawn_pose_lifts_for_high_rough_terrain(
    full_body_asset: PreparedAsset,
) -> None:
    flat_model = load_prepared_bello_model(full_body_asset, terrain=FLAT_TERRAIN)
    rough_terrain = TerrainConfig(
        mode="rough",
        hfield_resolution=17,
        hfield_amplitude=0.25,
    )
    rough_model = load_prepared_bello_model(full_body_asset, terrain=rough_terrain)
    min_robot_z = robot_geom_min_z(rough_model.mj_model, rough_model.default_qpos)
    ground_id = int(rough_model.terrain_geom_ids[0])
    root_xy = rough_model.default_qpos[
        rough_model.root_qpos_adr : rough_model.root_qpos_adr + 2
    ]
    target_min_z = (
        float(terrain_surface_height(rough_model.mj_model, ground_id, root_xy))
        + DEFAULT_SPAWN_CLEARANCE
    )

    assert (
        rough_model.default_qpos[rough_model.root_qpos_adr + 2]
        > flat_model.default_qpos[flat_model.root_qpos_adr + 2]
    )
    assert min_robot_z is not None
    assert abs(min_robot_z - target_min_z) < 1.0e-5
