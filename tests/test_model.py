import mujoco

from bello_mujoco.sim import load_bello_model


def test_full_body_model_loads_with_policy_actuators_first() -> None:
    model = load_bello_model()

    assert model.mj_model.nq == 42
    assert model.mj_model.nv == 41
    assert model.mj_model.nu == 35
    assert model.policy_ctrl_indices.tolist() == list(range(12))
    assert len(model.fixed_ctrl_indices) == 23


def test_training_model_uses_simplified_mesh_collision() -> None:
    model = load_bello_model().mj_model
    geom_types = {int(value) for value in model.geom_type}

    assert mujoco.mjtGeom.mjGEOM_MESH in geom_types
    assert mujoco.mjtGeom.mjGEOM_PLANE in geom_types
