from dataclasses import replace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from bello_mujoco.actuators import ActuatorConfig, JointActuatorConfig
from bello_mujoco.models.compiler import (
    ModelConfig,
    compile_robot,
    validate_actuator_joint_contract,
)
from bello_mujoco.models.urdf import exclude_robot_self_collisions, import_urdf_model


@pytest.mark.parametrize("asset", ["bello", "bello_mini"])
@pytest.mark.parametrize("self_collisions", [True, False])
@pytest.mark.integration
def test_compilation_preserves_canonical_mass_and_dynamics_with_optional_visuals(
    asset, self_collisions
):
    compiled = compile_robot(ModelConfig(asset=asset, self_collisions=self_collisions))
    source = import_urdf_model(compiled.urdf_path)
    physics = mujoco.MjModel.from_xml_path(str(compiled.mjcf_path))
    visual = mujoco.MjModel.from_xml_path(str(compiled.viewer_mjcf_path))
    assert physics.body_mass.sum() == pytest.approx(source.body_mass.sum(), abs=1e-10)
    for field in (
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "actuator_gainprm",
        "actuator_forcerange",
    ):
        np.testing.assert_array_equal(getattr(physics, field), getattr(visual, field))
    assert visual.ngeom > physics.ngeom
    assert np.all(physics.actuator_forcelimited)


def test_self_collision_exclusion_preserves_robot_ground_contacts():
    root = ET.fromstring("""<mujoco><worldbody>
      <geom name="floor" type="plane" size="2 2 .1"/>
      <body name="root" pos="0 0 .08"><freejoint/><geom size=".02"/>
        <body name="left" pos="-.06 0 0"><joint axis="0 0 1"/><geom size=".1"/></body>
        <body name="right" pos=".06 0 0"><joint axis="0 0 1"/><geom size=".1"/></body>
      </body></worldbody></mujoco>""")

    def contacts(document):
        model = mujoco.MjModel.from_xml_string(
            ET.tostring(document, encoding="unicode")
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        bodies = model.geom_bodyid[data.contact.geom]
        return model, np.any(np.all(bodies > 0, axis=1)), np.any(bodies == 0)

    baseline, self_contact, ground_contact = contacts(root)
    assert self_contact and ground_contact
    exclude_robot_self_collisions(root)
    disabled, self_contact, ground_contact = contacts(root)
    assert not self_contact and ground_contact
    np.testing.assert_array_equal(disabled.body_mass, baseline.body_mass)
    np.testing.assert_array_equal(disabled.geom_size, baseline.geom_size)


@pytest.mark.integration
def test_payload_and_mass_override_change_only_requested_inertials():
    base = compile_robot(ModelConfig())
    baseline = mujoco.MjModel.from_xml_path(str(base.mjcf_path))
    config = ModelConfig(target_total_mass_kg=60.0, end_effector_payload_mass_kg=2.0)
    modified = compile_robot(config)
    model = mujoco.MjModel.from_xml_path(str(modified.mjcf_path))
    assert model.body_mass.sum() == pytest.approx(64.0, abs=1e-9)
    np.testing.assert_array_equal(model.jnt_pos, baseline.jnt_pos)
    np.testing.assert_array_equal(model.jnt_axis, baseline.jnt_axis)
    assert base.urdf_path == modified.urdf_path


def test_actuator_joint_coverage_is_validated():
    joint = JointActuatorConfig(effort_limit=1.0)
    with pytest.raises(ValueError, match="missing"):
        validate_actuator_joint_contract(ActuatorConfig(joints={}), ("left",))
    with pytest.raises(ValueError, match="unknown"):
        validate_actuator_joint_contract(
            ActuatorConfig(joints={"left": joint, "extra": joint}), ("left",)
        )


def test_actuator_mapping_is_immutable():
    config = ActuatorConfig(joints={"joint": JointActuatorConfig(default_pos=0.1)})
    copy = config.by_name
    copy["joint"] = replace(copy["joint"], default_pos=0.2)
    assert config.joints["joint"].default_pos == 0.1
    with pytest.raises(TypeError):
        config.joints["joint"] = copy["joint"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_invalid_payload_is_rejected(value):
    with pytest.raises(ValueError, match="payload"):
        ModelConfig(end_effector_payload_mass_kg=value)


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_invalid_self_collision_setting_is_rejected(value):
    with pytest.raises(ValueError, match="self_collisions"):
        ModelConfig(self_collisions=value)
