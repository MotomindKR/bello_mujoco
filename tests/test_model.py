import mujoco
import pytest

from bello_mujoco.actuators import ActuatorConfig, JointActuatorConfig
from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig, PreparedAsset
from bello_mujoco.assets_pipeline.prepare import actuated_urdf_joint_names
from bello_mujoco.assets import source_assets_root
from bello_mujoco.sim import load_bello_model, load_prepared_bello_model
from bello_mujoco.terrain import (
    FLAT_TERRAIN,
    ROBOT_ONLY_TERRAIN,
    ROUGH_TERRAIN,
    TerrainConfig,
    rough_height_data,
)


def test_full_body_model_loads_with_policy_and_hold_actuators(
    full_body_asset: PreparedAsset,
) -> None:
    model = load_prepared_bello_model(full_body_asset)

    assert model.mj_model.nq == 42
    assert model.mj_model.nv == 41
    assert model.mj_model.nu == 35
    assert model.policy_ctrl_indices.tolist() == list(range(12))
    assert len(model.hold_ctrl_indices) == 23
    assert model.policy_action_scale.shape == (12,)


def test_joint_actuator_mode_controls_policy_action_partition(
    full_body_asset: PreparedAsset,
) -> None:
    default_config = ActuatorConfig()
    joints = {
        name: (
            JointActuatorConfig(
                mode="policy",
                default_pos=config.default_pos,
                action_scale=0.1,
                soft_limit=config.soft_limit,
                gains=config.gains,
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
    assert len(model.policy_ctrl_indices) == 13
    assert len(model.hold_ctrl_indices) == 22


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

    assert len(model.policy_ctrl_indices) == 12
    assert len(model.hold_ctrl_indices) == 23


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
    plane_count = sum(
        1 for geom_type in model.geom_type if geom_type == mujoco.mjtGeom.mjGEOM_PLANE
    )

    assert model.ngeom == 40
    assert len(collision_mesh_geoms) == 40
    assert 100_000 < sum(collision_mesh_faces) < 150_000
    assert min(collision_mesh_faces) >= 1_000
    assert max(collision_mesh_faces) > 20_000
    assert box_collision_count == 0
    assert plane_count == 0
    assert model.nexclude == 75


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

    assert model.ngeom == 80
    assert len(visual_mesh_geoms) == 40


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


def test_config_loader_rejects_upper_body_policy_metadata() -> None:
    with pytest.raises(ValueError, match="requires full_body or lower_body"):
        load_bello_model(AssetPrepConfig(asset="upper_body"))


def actuated_joint_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            int(model.actuator_trnid[actuator_id, 0]),
        )
        for actuator_id in range(model.nu)
    ]
