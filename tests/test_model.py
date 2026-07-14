from dataclasses import replace

import mujoco
import numpy as np
import pytest

from bello_mujoco.actuators import ActuatorConfig
from bello_mujoco.assets_pipeline.prepare import (
    AssetPrepConfig,
    PreparedAsset,
    prepare_asset,
)
from bello_mujoco.assets_pipeline.prepare import actuated_urdf_joint_names
from bello_mujoco.assets import source_assets_root
from bello_mujoco.contracts import DEFAULT_SPAWN_CLEARANCE
from bello_mujoco.sim import (
    load_bello_model,
    load_prepared_bello_model,
    robot_geom_min_z,
    terrain_surface_upper_bound,
)
from bello_mujoco.terrain import (
    FLAT_TERRAIN,
    ROBOT_ONLY_TERRAIN,
    ROUGH_TERRAIN,
    TerrainConfig,
    rough_height_data,
)


def test_full_body_model_loads_with_all_actuators_in_policy_by_default(
    full_body_asset: PreparedAsset,
) -> None:
    model = load_prepared_bello_model(full_body_asset)

    assert model.mj_model.nq == 42
    assert model.mj_model.nv == 41
    assert model.mj_model.nu == 35
    assert model.policy_ctrl_indices.tolist() == list(range(35))
    assert len(model.hold_ctrl_indices) == 0
    assert model.policy_action_scale.shape == (35,)
    ctrlrange = model.mj_model.actuator_ctrlrange[model.policy_ctrl_indices]
    expected_scale = 0.25 * (ctrlrange[:, 1] - ctrlrange[:, 0])
    assert np.allclose(model.policy_action_scale, expected_scale)


def test_model_can_load_without_mjx_for_cpu_only_viewers(
    full_body_asset: PreparedAsset,
) -> None:
    model = load_prepared_bello_model(full_body_asset, load_mjx=False)

    assert model.mjx_model is None
    with pytest.raises(RuntimeError, match="loaded without MJX"):
        model.make_data()


def test_joint_actuator_mode_controls_policy_action_partition(
    full_body_asset: PreparedAsset,
) -> None:
    default_config = ActuatorConfig()
    joints = {
        name: (
            replace(
                config,
                mode="hold",
                action_scale=0.1,
            )
            if name == "waist_yaw_joint"
            else config
        )
        for name, config in default_config.joints.items()
    }
    model = load_prepared_bello_model(
        full_body_asset,
        actuator_config=ActuatorConfig(joints=joints),
    )

    assert model.mj_model.nu == 35
    assert len(model.policy_ctrl_indices) == 34
    assert len(model.hold_ctrl_indices) == 1


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
    assert model.nexclude == 75


def test_ankle_pitch_zero_places_sole_primitives_horizontal(
    tmp_path,
) -> None:
    asset = prepare_asset(
        AssetPrepConfig(
            asset="full_body",
            output_root=tmp_path,
            collision_mode="box",
            visual_mesh_face_budget=5_000,
        )
    )
    model = mujoco.MjModel.from_xml_path(str(asset.mjcf_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    for side in ("left", "right"):
        joint = model.joint(f"{side}_ankle_pitch_joint")
        actuator = model.actuator(f"{side}_ankle_pitch_joint_pos")
        sole = model.geom(f"{side}_ankle_roll_link_collision_box_1")
        sole_normal = data.geom_xmat[sole.id].reshape(3, 3)[:, 2]

        assert model.qpos0[joint.qposadr[0]] == pytest.approx(0.0)
        assert sole_normal[2] == pytest.approx(1.0, abs=1.0e-8)
        assert actuator.ctrlrange == pytest.approx((-0.706859, 0.706861))
        assert model.actuator_user[actuator.id, 0] == pytest.approx(0.038201)


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
    assert np.allclose(viewer_model.policy_action_scale, sim_model.policy_action_scale)
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


def test_default_spawn_pose_starts_above_flat_terrain(
    full_body_asset: PreparedAsset,
) -> None:
    model = load_prepared_bello_model(full_body_asset, terrain=FLAT_TERRAIN)
    min_robot_z = robot_geom_min_z(model.mj_model, model.default_qpos)
    target_min_z = terrain_surface_upper_bound(FLAT_TERRAIN) + DEFAULT_SPAWN_CLEARANCE

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
    target_min_z = terrain_surface_upper_bound(rough_terrain) + DEFAULT_SPAWN_CLEARANCE

    assert (
        rough_model.default_qpos[rough_model.root_qpos_adr + 2]
        > flat_model.default_qpos[flat_model.root_qpos_adr + 2]
    )
    assert min_robot_z is not None
    assert abs(min_robot_z - target_min_z) < 1.0e-5


def test_split_body_models_load_with_asset_specific_actuators(
    generated_test_assets: dict[str, PreparedAsset],
) -> None:
    lower = mujoco.MjModel.from_xml_path(
        str(generated_test_assets["lower_body"].mjcf_path)
    )
    upper = mujoco.MjModel.from_xml_path(
        str(generated_test_assets["upper_body"].mjcf_path)
    )

    lower_actuated_joints = actuated_joint_names(lower)
    upper_actuated_joints = actuated_joint_names(upper)

    assert lower.nu == 12
    runtime_locomotion_joints = actuated_urdf_joint_names(
        source_assets_root() / "lower_body" / "urdf" / "bello_lower_body.urdf"
    )
    assert lower_actuated_joints == list(runtime_locomotion_joints)
    assert upper.nu == 22
    assert set(upper_actuated_joints).isdisjoint(runtime_locomotion_joints)
    assert "right_wrist_yaw_joint" in upper_actuated_joints
    assert "left_wrist_yaw_joint" in upper_actuated_joints


def test_config_loader_loads_upper_body_model_metadata() -> None:
    model = load_bello_model(AssetPrepConfig(asset="upper_body"))

    assert model.mj_model.nu == 22
    assert model.foot_body_ids.size == 0
    assert len(model.policy_ctrl_indices) == 22


def actuated_joint_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            int(model.actuator_trnid[actuator_id, 0]),
        )
        for actuator_id in range(model.nu)
    ]
