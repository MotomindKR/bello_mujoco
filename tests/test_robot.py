from dataclasses import replace
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from omegaconf import OmegaConf
import pytest

from bello_mujoco.actuators import ActuatorConfig, default_joint_configs
from bello_mujoco.assets_pipeline import robot_descriptions_root
from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.robot import robot_definition, validate_robot_source_urdf
from bello_mujoco.robot_tasks import (
    motion_task_definition,
    validate_motion_task_source_urdf,
)


def test_bello_robot_config_declares_asset_topology() -> None:
    robot = robot_definition("bello")

    assert robot.source_urdf == Path("bello/urdf/bello.urdf")
    assert robot.source_urdf_path(robot_descriptions_root()).is_file()
    assert robot.hardware_description_path(robot_descriptions_root()).is_file()
    assert robot.motor_model_catalog_path(robot_descriptions_root()).is_file()
    assert robot.root_body_name == "bello_root"
    assert robot.root_joint_name == "root_free"
    assert robot.default_imu_body_name == "pelvis_link"
    assert robot.locomotion_command_frame_body_name == "pelvis_link"
    assert robot.locomotion_forward_axis == (0.0, 1.0, 0.0)
    mirror = {
        item.joint_name: (item.counterpart_joint_name, item.sign)
        for item in robot.locomotion_action_mirror
    }
    assert mirror["left_hip_pitch_joint"] == ("right_hip_pitch_joint", -1.0)
    assert mirror["left_hip_roll_joint"] == ("right_hip_roll_joint", 1.0)
    assert robot.randomization_torso_body_name == "torso_link"
    assert robot.foot_body_names == (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )
    assert tuple(default_joint_configs("bello")) == tuple(
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
    assert robot.physical_hardware_ready
    validate_robot_source_urdf(
        robot,
        robot.source_urdf_path(robot_descriptions_root()),
    )


def test_asset_config_resolves_actuators_from_selected_robot() -> None:
    config = AssetPrepConfig()

    assert config.actuator_config == ActuatorConfig.for_asset("bello")


def test_bello_mini_robot_config_declares_locomotion_topology() -> None:
    robot = robot_definition("bello_mini")

    assert robot.source_urdf == Path("bello_mini/urdf/bello_mini.urdf")
    assert robot.source_urdf_path(robot_descriptions_root()).is_file()
    assert robot.hardware_description == Path("bello_mini/hardware.yaml")
    assert robot.hardware_description_path(robot_descriptions_root()).is_file()
    assert not robot.physical_hardware_ready
    assert robot.motor_model_catalog_path(robot_descriptions_root()).is_file()
    assert robot.root_body_name == "bello_mini_root"
    assert robot.default_imu_body_name == "imu_link"
    assert robot.locomotion_command_frame_body_name == "pelvis_link"
    assert robot.locomotion_forward_axis == (1.0, 0.0, 0.0)
    mirror = {
        item.joint_name: (item.counterpart_joint_name, item.sign)
        for item in robot.locomotion_action_mirror
    }
    assert mirror["left_hip_pitch_joint"] == ("right_hip_pitch_joint", 1.0)
    assert mirror["left_hip_roll_joint"] == ("right_hip_roll_joint", -1.0)
    assert robot.default_root_position == (0.0, 0.0, 0.859)
    assert robot.foot_body_names == (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )
    assert robot.foot_forward_axes == (
        ("left_ankle_roll_link", (1.0, 0.0, 0.0)),
        ("right_ankle_roll_link", (1.0, 0.0, 0.0)),
    )
    assert len(robot.actuators) == 28
    assert sum(actuator.mode == "policy" for actuator in robot.actuators) == 14
    assert sum(actuator.mode == "hold" for actuator in robot.actuators) == 14
    default_pos = {
        actuator.joint_name: actuator.default_pos for actuator in robot.actuators
    }
    soft_limits = {
        actuator.joint_name: actuator.soft_limit for actuator in robot.actuators
    }
    for side in ("left", "right"):
        leg_pitch = (
            default_pos[f"{side}_hip_pitch_joint"],
            default_pos[f"{side}_knee_joint"],
            default_pos[f"{side}_ankle_pitch_joint"],
        )
        assert leg_pitch == pytest.approx((-0.15, 0.45, -0.30))
        assert sum(leg_pitch) == pytest.approx(0.0)
        assert soft_limits[f"{side}_knee_joint"] == pytest.approx((0.0, 1.2))
    assert default_pos["left_shoulder_roll_joint"] == pytest.approx(0.15)
    assert default_pos["right_shoulder_roll_joint"] == pytest.approx(-0.15)
    assert (
        next(
            actuator
            for actuator in robot.actuators
            if actuator.joint_name == "waist_roll_joint"
        ).mode
        == "policy"
    )
    assert tuple(group.name for group in robot.coupled_actuator_groups) == (
        "left_ankle_differential",
        "right_ankle_differential",
    )
    assert {
        actuator.tendon_name: actuator.joint_coefficients
        for actuator in robot.coupled_actuators
    } == {
        "left_ankle_motor_1": (
            ("left_ankle_pitch_joint", 1.0),
            ("left_ankle_roll_joint", 1.0),
        ),
        "left_ankle_motor_2": (
            ("left_ankle_pitch_joint", 1.0),
            ("left_ankle_roll_joint", -1.0),
        ),
        "right_ankle_motor_1": (
            ("right_ankle_pitch_joint", 1.0),
            ("right_ankle_roll_joint", 1.0),
        ),
        "right_ankle_motor_2": (
            ("right_ankle_pitch_joint", 1.0),
            ("right_ankle_roll_joint", -1.0),
        ),
    }
    assert tuple(item.link_name for item in robot.payload_attachments) == (
        "left_end_effector_sphere_link",
        "right_end_effector_sphere_link",
    )
    source_urdf = robot.source_urdf_path(robot_descriptions_root())
    source_root = ET.parse(source_urdf).getroot()
    imu_joint = source_root.find("./joint[@name='imu_joint']")
    assert imu_joint is not None and imu_joint.attrib["type"] == "fixed"
    assert imu_joint.find("parent").attrib["link"] == "pelvis_link"
    assert imu_joint.find("child").attrib["link"] == "imu_link"
    assert imu_joint.find("origin").attrib == {
        "xyz": "-0.0099999999999894115 0 -0.230000001143162",
        "rpy": "0 0 -1.5707963267948966",
    }
    movable_joint_names = {
        joint.attrib["name"]
        for joint in source_root.findall("joint")
        if joint.attrib["type"] != "fixed"
    }
    assert {actuator.joint_name for actuator in robot.actuators} == (
        movable_joint_names
    )
    assert "left_wrist_yaw_joint" in movable_joint_names
    assert "right_wrist_yaw_joint" in movable_joint_names
    assert "left_elbow_yaw_joint" not in movable_joint_names
    assert "right_elbow_yaw_joint" not in movable_joint_names

    expected_limit_degrees = {
        "waist_yaw_joint": (-90.0, 90.0),
        "waist_roll_joint": (-25.0, 25.0),
        "left_hip_pitch_joint": (-180.0, 90.0),
        "right_hip_pitch_joint": (-180.0, 90.0),
        "left_hip_roll_joint": (-31.0, 121.0),
        "right_hip_roll_joint": (-121.0, 31.0),
        "left_hip_yaw_joint": (-90.0, 90.0),
        "right_hip_yaw_joint": (-90.0, 90.0),
        "left_knee_joint": (0.0, 145.0),
        "right_knee_joint": (0.0, 145.0),
        "left_ankle_pitch_joint": (-45.0, 45.0),
        "right_ankle_pitch_joint": (-45.0, 45.0),
        "left_ankle_roll_joint": (-35.0, 35.0),
        "right_ankle_roll_joint": (-35.0, 35.0),
        "left_shoulder_pitch_joint": (-225.0, 45.0),
        "right_shoulder_pitch_joint": (-225.0, 45.0),
        "left_shoulder_roll_joint": (-15.0, 180.0),
        "right_shoulder_roll_joint": (-180.0, 15.0),
        "left_shoulder_yaw_joint": (-90.0, 90.0),
        "right_shoulder_yaw_joint": (-90.0, 90.0),
        "left_elbow_pitch_joint": (-130.0, 15.0),
        "right_elbow_pitch_joint": (-130.0, 15.0),
        "left_wrist_yaw_joint": (-116.8, 116.8),
        "right_wrist_yaw_joint": (-116.8, 116.8),
        "left_wrist_pitch_joint": (-90.0, 90.0),
        "right_wrist_pitch_joint": (-90.0, 90.0),
        "left_wrist_roll_joint": (-90.0, 90.0),
        "right_wrist_roll_joint": (-90.0, 90.0),
    }
    actual_limits = {}
    for joint in source_root.findall("joint"):
        limit = joint.find("limit")
        if limit is None:
            continue
        actual_limits[joint.attrib["name"]] = (
            float(limit.attrib["lower"]),
            float(limit.attrib["upper"]),
        )
    assert actual_limits.keys() == expected_limit_degrees.keys()
    for joint_name, degrees in expected_limit_degrees.items():
        assert actual_limits[joint_name] == pytest.approx(
            tuple(math.radians(value) for value in degrees), abs=1e-14
        )
    validate_robot_source_urdf(
        robot,
        source_urdf,
    )


def test_locomotion_action_mirror_must_cover_policy_joints_exactly() -> None:
    robot = robot_definition("bello_mini")

    with pytest.raises(ValueError, match="cover the policy joints exactly"):
        replace(
            robot,
            locomotion_action_mirror=robot.locomotion_action_mirror[:-1],
        )


def test_bello_mini_motion_task_profile_matches_robot_source() -> None:
    robot = robot_definition("bello_mini")
    task = motion_task_definition("bello_mini")
    expected_joint_names = (
        "waist_yaw_joint",
        "waist_roll_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_pitch_joint",
        "left_wrist_yaw_joint",
        "left_wrist_pitch_joint",
        "left_wrist_roll_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_pitch_joint",
        "right_wrist_yaw_joint",
        "right_wrist_pitch_joint",
        "right_wrist_roll_joint",
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    )

    assert task.legacy_import_joint_names == expected_joint_names
    assert task.fixed_joint_names == ()
    assert task.imported_actuator_joint_names == expected_joint_names
    assert set(task.imported_actuator_joint_names) == {
        actuator.joint_name for actuator in robot.actuators
    }
    assert task.default_end_effector_body_names == (
        "left_wrist_roll_link",
        "right_wrist_roll_link",
    )
    assert task.twist_key_body_names[-1] == "torso_link"
    assert task.gmr.source_human == "smplx"
    assert task.gmr.target_robot == "bello_mini"
    assert task.gmr.task_profile == "universal"
    assert task.gmr.config_path(Path("deps/GMR")) == Path(
        "deps/GMR/general_motion_retargeting/ik_configs/smplx_to_bello_mini.json"
    )
    assert task.gmr.output_body_names == (
        "pelvis_link",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_roll_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_roll_link",
    )
    assert len(task.gmr.standing_default_joint_names) == 12
    assert task.gmr.knee_pole == motion_task_definition("bello").gmr.knee_pole
    validate_motion_task_source_urdf(
        task,
        robot.source_urdf_path(robot_descriptions_root()),
    )


def test_bello_mini_can_topology_is_monotonic_and_calibration_gated() -> None:
    topology = OmegaConf.load(
        robot_descriptions_root() / "bello_mini" / "motor_topology.yaml"
    )
    expected = (
        ("waist_yaw_motor", "can0", 51, 1),
        ("waist_roll_motor", "can0", 52, 1),
        ("left_hip_pitch_motor", "can0", 61, 1),
        ("left_hip_roll_motor", "can0", 62, 1),
        ("left_hip_yaw_motor", "can0", 63, 1),
        ("left_knee_motor", "can0", 64, -1),
        ("left_ankle_motor_1", "can0", 65, -1),
        ("left_ankle_motor_2", "can0", 66, 1),
        ("right_hip_pitch_motor", "can0", 71, -1),
        ("right_hip_roll_motor", "can0", 72, 1),
        ("right_hip_yaw_motor", "can0", 73, 1),
        ("right_knee_motor", "can0", 74, -1),
        ("right_ankle_motor_1", "can0", 75, -1),
        ("right_ankle_motor_2", "can0", 76, 1),
        ("left_shoulder_pitch_motor", "can1", 31, 1),
        ("left_shoulder_roll_motor", "can1", 32, 1),
        ("left_shoulder_yaw_motor", "can1", 33, 1),
        ("left_elbow_pitch_motor", "can1", 34, -1),
        ("left_wrist_yaw_motor", "can1", 35, 1),
        ("left_wrist_pitch_motor", "can1", 36, 1),
        ("left_wrist_roll_motor", "can1", 37, -1),
        ("right_shoulder_pitch_motor", "can1", 41, -1),
        ("right_shoulder_roll_motor", "can1", 42, -1),
        ("right_shoulder_yaw_motor", "can1", 43, 1),
        ("right_elbow_pitch_motor", "can1", 44, -1),
        ("right_wrist_yaw_motor", "can1", 45, 1),
        ("right_wrist_pitch_motor", "can1", 46, -1),
        ("right_wrist_roll_motor", "can1", 47, 1),
    )

    assert topology.schema_version == 2
    assert topology.deployment_status == "calibration_required"
    actual = tuple(
        (
            motor.name,
            motor.hardware_binding.bus,
            motor.hardware_binding.can_id,
            motor.cad_direction_inference.axis_alignment_sign,
        )
        for motor in topology.motors
    )
    assert actual == expected
    assert len({(bus, can_id) for _, bus, can_id, _ in actual}) == len(actual)
    assert all(
        motor.hardware_binding.encoder_zero_offset_rad is None
        for motor in topology.motors
    )
    confirmed_directions = tuple(
        motor.hardware_binding.command_direction
        for motor in topology.motors
        if motor.hardware_binding.command_direction is not None
    )
    assert confirmed_directions
    assert len(confirmed_directions) < len(topology.motors)
    assert all(direction in (-1, 1) for direction in confirmed_directions)
    assert all(
        (
            motor.hardware_binding.command_direction is None
            and motor.hardware_binding.status
            == "can_assigned_direction_calibration_required"
        )
        or (
            motor.hardware_binding.command_direction in (-1, 1)
            and motor.hardware_binding.status == "direction_confirmed_by_operator"
        )
        for motor in topology.motors
    )
    assert all(
        group.motor_from_joint_coefficients is None
        for group in topology.differential_groups
    )


def test_asset_config_can_explicitly_promote_every_robot_joint_to_policy() -> None:
    config = AssetPrepConfig(asset="bello", actuator_mode="policy")

    assert config.actuator_config == ActuatorConfig.for_asset("bello", mode="policy")
    assert not config.actuator_config.hold_joint_names


def test_bello_motion_task_profile_matches_robot_source() -> None:
    robot = robot_definition("bello")
    task = motion_task_definition("bello")

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


def test_robot_config_rejects_removed_full_body_asset_name() -> None:
    with pytest.raises(ValueError, match="unsupported robot asset 'full_body'"):
        robot_definition("full_body")
