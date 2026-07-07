import mujoco

from bello_mujoco.sim import load_bello_model


def test_full_body_model_loads_with_policy_actuators_first() -> None:
    model = load_bello_model()

    assert model.mj_model.nq == 42
    assert model.mj_model.nv == 41
    assert model.mj_model.nu == 35
    assert model.policy_ctrl_indices.tolist() == list(range(12))
    assert len(model.fixed_ctrl_indices) == 23


def test_training_model_uses_foot_mesh_collision_profile() -> None:
    model = load_bello_model().mj_model
    mesh_geom_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        for geom_id, geom_type in enumerate(model.geom_type)
        if geom_type == mujoco.mjtGeom.mjGEOM_MESH
    ]

    assert model.ngeom == 42
    assert mesh_geom_names == ["left_ankle_roll_link_collision", "right_ankle_roll_link_collision"]
    assert int((model.geom_type == mujoco.mjtGeom.mjGEOM_BOX).sum()) == 39
    assert int((model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE).sum()) == 1
    assert model.nexclude == 75
