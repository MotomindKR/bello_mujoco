from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from bello_mujoco.actuators import ActuatorConfig, default_joint_configs
from bello_mujoco.assets_pipeline import robot_descriptions_root
from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.robot import robot_definition, validate_robot_source_urdf
from bello_mujoco.robot_tasks import (
    motion_task_definition,
    validate_motion_task_source_urdf,
)


def test_full_body_robot_config_declares_asset_topology() -> None:
    robot = robot_definition("full_body")

    assert robot.source_urdf == Path("bello/full_body/urdf/bello_full_body.urdf")
    assert robot.source_urdf_path(robot_descriptions_root()).is_file()
    assert robot.hardware_description_path(robot_descriptions_root()).is_file()
    assert robot.motor_model_catalog_path(robot_descriptions_root()).is_file()
    assert robot.root_body_name == "bello_root"
    assert robot.root_joint_name == "root_free"
    assert robot.default_imu_body_name == "pelvis_link"
    assert robot.randomization_torso_body_name == "torso_link"
    assert robot.foot_body_names == (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )
    assert tuple(default_joint_configs("full_body")) == tuple(
        actuator.joint_name for actuator in robot.actuators
    )
    assert len(robot.actuators) == 25
    assert robot.actuators[0].mode == "policy"
    assert robot.actuators[0].gains == (200.0, 5.0)
    assert robot.actuators[-1].mode == "hold"
    assert tuple(actuator.tendon_name for actuator in robot.coupled_actuators) == (
        "left_ankle_motor_1",
        "left_ankle_motor_2",
        "right_ankle_motor_1",
        "right_ankle_motor_2",
    )
    validate_robot_source_urdf(
        robot,
        robot.source_urdf_path(robot_descriptions_root()),
    )


def test_asset_config_resolves_actuators_from_selected_robot() -> None:
    config = AssetPrepConfig()

    assert config.actuator_config == ActuatorConfig.for_asset("full_body")


def test_bello_mini_robot_config_declares_reviewed_static_topology() -> None:
    robot = robot_definition("bello_mini")

    assert robot.source_urdf == Path("bello_mini/urdf/bello_mini.urdf")
    assert robot.source_urdf_path(robot_descriptions_root()).is_file()
    assert robot.hardware_description is None
    with pytest.raises(ValueError, match="has no hardware description"):
        robot.hardware_description_path(robot_descriptions_root())
    assert robot.motor_model_catalog_path(robot_descriptions_root()).is_file()
    assert robot.root_body_name == "bello_mini_root"
    assert robot.default_root_position == (0.0, 0.0, 0.8731232537850352)
    assert robot.foot_body_names == (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )
    assert robot.foot_forward_axes == (
        ("left_ankle_roll_link", (1.0, 0.0, 0.0)),
        ("right_ankle_roll_link", (1.0, 0.0, 0.0)),
    )
    assert len(robot.actuators) == 28
    assert sum(actuator.mode == "policy" for actuator in robot.actuators) == 13
    assert sum(actuator.mode == "hold" for actuator in robot.actuators) == 15
    assert not robot.coupled_actuator_groups
    assert tuple(item.link_name for item in robot.payload_attachments) == (
        "left_end_effector_sphere_link",
        "right_end_effector_sphere_link",
    )
    source_urdf = robot.source_urdf_path(robot_descriptions_root())
    source_root = ET.parse(source_urdf).getroot()
    movable_joint_names = {
        joint.attrib["name"]
        for joint in source_root.findall("joint")
        if joint.attrib["type"] != "fixed"
    }
    assert {actuator.joint_name for actuator in robot.actuators} == (
        movable_joint_names
    )
    validate_robot_source_urdf(
        robot,
        source_urdf,
    )


def test_bello_mini_motion_profile_is_explicitly_deferred() -> None:
    with pytest.raises(ValueError, match="has no motion task profile"):
        motion_task_definition("bello_mini")


def test_asset_config_can_explicitly_promote_every_robot_joint_to_policy() -> None:
    config = AssetPrepConfig(asset="full_body", actuator_mode="policy")

    assert config.actuator_config == ActuatorConfig.for_asset(
        "full_body", mode="policy"
    )
    assert not config.actuator_config.hold_joint_names


def test_full_body_motion_task_profile_matches_robot_source() -> None:
    robot = robot_definition("full_body")
    task = motion_task_definition("full_body")

    assert task.imported_actuator_joint_names == tuple(
        name
        for name in task.legacy_import_joint_names
        if name not in task.fixed_joint_names
    )
    assert set(task.imported_actuator_joint_names) == {
        actuator.joint_name for actuator in robot.actuators
    }
    assert task.twist_key_body_names[-1] == "head_link"
    assert task.gmr.source_human == "smplx"
    assert task.gmr.target_robot == "bello"
    assert task.gmr.task_profile == "universal"
    assert task.gmr.config_path(Path("deps/GMR")) == Path(
        "deps/GMR/general_motion_retargeting/ik_configs/smplx_to_bello.json"
    )
    assert task.gmr.source_foot_body_names == ("left_foot", "right_foot")
    assert task.gmr.output_body_names[-1] == "head_link"
    assert tuple(leg.name for leg in task.gmr.knee_pole.legs) == ("left", "right")
    assert task.gmr.knee_pole.legs[0].target_body_names == (
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
    )
    assert task.gmr.knee_pole.legs[0].target_knee_joint_name == "left_knee_joint"
    validate_motion_task_source_urdf(
        task,
        robot.source_urdf_path(robot_descriptions_root()),
    )


def test_robot_source_validation_rejects_unknown_configured_link() -> None:
    robot = robot_definition("full_body")
    invalid = replace(robot, default_imu_body_name="missing_imu_link")

    with pytest.raises(ValueError, match="missing_imu_link"):
        validate_robot_source_urdf(
            invalid,
            robot.source_urdf_path(robot_descriptions_root()),
        )


def test_robot_config_rejects_unknown_asset_without_alias() -> None:
    with pytest.raises(ValueError, match="unsupported robot asset 'typo'"):
        robot_definition("typo")
