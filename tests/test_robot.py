from dataclasses import replace

import pytest

from bello_mujoco.actuators import ActuatorConfig
from bello_mujoco.models import robot_descriptions_root
from bello_mujoco.models.compiler import ModelConfig
from bello_mujoco.robot import (
    robot_definition,
    validate_robot_source_urdf,
    urdf_coupled_actuators,
)


def test_asset_config_resolves_actuators_from_selected_robot() -> None:
    config = ModelConfig()

    assert config.actuator_config == ActuatorConfig.for_asset("bello")


def test_asset_config_can_explicitly_promote_every_robot_joint_to_policy() -> None:
    config = ModelConfig(asset="bello", actuator_mode="policy")

    assert config.actuator_config == ActuatorConfig.for_asset("bello", mode="policy")
    assert not config.actuator_config.hold_joint_names


def test_robot_source_validation_rejects_unknown_configured_link() -> None:
    robot = robot_definition("bello")
    invalid = replace(robot, default_imu_body_name="missing_imu_link")

    with pytest.raises(ValueError, match="missing_imu_link"):
        validate_robot_source_urdf(
            invalid,
            robot.source_urdf_path(robot_descriptions_root()),
        )


def test_robot_config_rejects_unknown_asset_without_alias() -> None:
    with pytest.raises(ValueError, match="unsupported robot asset 'typo'"):
        robot_definition("typo")


def test_differential_rows_come_from_urdf_reductions(tmp_path):
    path = tmp_path / "robot.urdf"
    path.write_text("""<robot><transmission name="differential">
      <type>transmission_interface/DifferentialTransmission</type>
      <actuator name="a" role="actuator1"><mechanical_reduction>2</mechanical_reduction></actuator>
      <actuator name="b" role="actuator2"><mechanical_reduction>-3</mechanical_reduction></actuator>
      <joint name="roll" role="joint1"><mechanical_reduction>4</mechanical_reduction></joint>
      <joint name="pitch" role="joint2"><mechanical_reduction>-5</mechanical_reduction></joint>
    </transmission></robot>""")
    (group,) = urdf_coupled_actuators(path, ("unrelated", "pitch", "roll"))
    a, b = group.actuators
    assert a.logical_joint_name == "pitch"
    assert b.logical_joint_name == "roll"
    assert a.joint_coefficients == (("pitch", -10.0), ("roll", 8.0))
    assert b.joint_coefficients == (("pitch", -15.0), ("roll", -12.0))


def test_incomplete_transmission_fails_during_loading(tmp_path):
    path = tmp_path / "robot.urdf"
    path.write_text("""<robot><transmission name="broken">
      <type>transmission_interface/DifferentialTransmission</type>
    </transmission></robot>""")
    with pytest.raises(ValueError, match="missing actuator1"):
        urdf_coupled_actuators(path, ("pitch", "roll"))
