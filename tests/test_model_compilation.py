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
from bello_mujoco.models.urdf import add_end_effector_payloads
from bello_mujoco.robot import PayloadAttachment


@pytest.mark.parametrize("asset", ["bello", "bello_mini"])
@pytest.mark.integration
def test_compilation_preserves_canonical_mass_and_dynamics_with_optional_visuals(asset):
    compiled = compile_robot(ModelConfig(asset=asset))
    source = mujoco.MjModel.from_xml_path(str(compiled.urdf_path))
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


@pytest.mark.parametrize("offset", [(0, 0), (0, 0, float("nan")), (float("inf"), 0, 0)])
def test_payload_offset_rejects_invalid_coordinates(offset):
    with pytest.raises(ValueError, match="offset"):
        ModelConfig(
            end_effector_payload_mass_kg=1.0, end_effector_payload_offset_m=offset
        )


def test_payload_offset_requires_mass_and_is_immutable():
    with pytest.raises(ValueError, match="positive payload mass"):
        ModelConfig(end_effector_payload_offset_m=(0.01, 0.0, 0.0))
    offset = [0.01, 0.02, -0.03]
    config = ModelConfig(
        end_effector_payload_mass_kg=1.0, end_effector_payload_offset_m=offset
    )
    offset[0] = 99.0
    assert config.end_effector_payload_offset_m == (0.01, 0.02, -0.03)


@pytest.mark.parametrize("hand_mass", [0.0, 0.4])
def test_payload_offset_preserves_existing_mass_geometry_and_frame(hand_mass):
    root = ET.fromstring("""<mujoco><worldbody>
      <body name="arm"><joint name="hinge" axis="0 1 0"/>
        <geom type="sphere" size=".1" mass="1"/>
        <body name="hand" pos=".1 0 -.2">
          <geom name="visual" type="sphere" size=".03" mass="0"/>
        </body>
      </body>
    </worldbody></mujoco>""")
    if hand_mass:
        ET.SubElement(
            root.find(".//body[@name='hand']"),
            "inertial",
            mass=str(hand_mass),
            pos=".01 .02 -.03",
            diaginertia=".001 .001 .001",
        )
    attachment = PayloadAttachment("hand", 0.03, (0.01, 0.02, -0.03))
    add_end_effector_payloads(root, (attachment,), 2.0, (0.04, 0.0, -0.06))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    hand = model.body("hand").id
    payload = model.body("hand_payload").id
    assert model.body_mass.sum() == pytest.approx(3.0 + hand_mass)
    assert model.body_mass[hand] == pytest.approx(hand_mass)
    np.testing.assert_allclose(model.body_pos[hand], [0.1, 0, -0.2])
    assert model.body_parentid[payload] == hand
    np.testing.assert_allclose(model.body_pos[payload], [0.05, 0.02, -0.09])
    np.testing.assert_allclose(model.body_ipos[payload], [0, 0, 0])
    if hand_mass:
        np.testing.assert_allclose(model.body_ipos[hand], [0.01, 0.02, -0.03])
        np.testing.assert_allclose(model.body_inertia[hand], [0.001] * 3)
    np.testing.assert_allclose(model.geom("visual").pos, [0, 0, 0])
    np.testing.assert_allclose(model.body_inertia[payload], [0.4 * 2.0 * 0.03**2] * 3)
    with pytest.raises(ValueError, match="already exists"):
        add_end_effector_payloads(root, (attachment,), 2.0, (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="missing payload attachment"):
        add_end_effector_payloads(
            ET.fromstring("<mujoco/>"), (attachment,), 2.0, (0.0, 0.0, 0.0)
        )
    with pytest.raises(ValueError, match="no payload attachment"):
        add_end_effector_payloads(root, (), 2.0, (0.0, 0.0, 0.0))
