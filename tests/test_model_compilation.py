from dataclasses import replace

import mujoco
import numpy as np
import pytest

from bello_mujoco.actuators import ActuatorConfig, JointActuatorConfig
from bello_mujoco.models.compiler import (
    ModelConfig,
    compile_robot,
    validate_actuator_joint_contract,
)


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
