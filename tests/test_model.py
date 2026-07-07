import mujoco

from bello_mujoco.assets_pipeline.prepare import PreparedAsset
from bello_mujoco.contracts import POLICY_JOINT_ORDER
from bello_mujoco.sim import load_bello_model


def test_full_body_model_loads_with_policy_actuators_first(
    full_body_asset: PreparedAsset,
) -> None:
    model = load_bello_model(full_body_asset.mjcf_path)

    assert model.mj_model.nq == 42
    assert model.mj_model.nv == 41
    assert model.mj_model.nu == 35
    assert model.policy_ctrl_indices.tolist() == list(range(12))
    assert len(model.fixed_ctrl_indices) == 23


def test_training_model_uses_simplified_mesh_collision_profile(
    full_body_asset: PreparedAsset,
) -> None:
    model = load_bello_model(full_body_asset.mjcf_path).mj_model
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
    assert lower_actuated_joints == list(POLICY_JOINT_ORDER)
    assert upper.nu == 22
    assert set(upper_actuated_joints).isdisjoint(POLICY_JOINT_ORDER)
    assert "right_wrist_yaw_joint" in upper_actuated_joints
    assert "left_wrist_yaw_joint" in upper_actuated_joints


def actuated_joint_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            int(model.actuator_trnid[actuator_id, 0]),
        )
        for actuator_id in range(model.nu)
    ]
