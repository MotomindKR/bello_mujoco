from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from bello_mujoco.robot_description import astra_stereo_s_u3 as astra


MODULE_PATH = (
    Path(__file__).parents[1]
    / "tools"
    / "robot_description"
    / "build_robot_description.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "build_robot_description", MODULE_PATH
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = MODULE
MODULE_SPEC.loader.exec_module(MODULE)


def foot_inertials() -> dict:
    inertia = {
        "ixx": 0.001,
        "ixy": 0.00001,
        "ixz": 0.00002,
        "iyy": 0.002,
        "iyz": 0.00003,
        "izz": 0.003,
    }
    return {
        "links": [
            {
                "name": f"{side}_ankle_pitch_link",
                "mass_kg": 0.0,
                "center_urdf_link_m": [0.0, 0.0, 0.0],
                "inertia_urdf_link_com_kg_m2": None,
            }
            for side in ("left", "right")
        ]
        + [
            {
                "name": f"{side}_ankle_roll_link",
                "mass_kg": 1.0,
                "center_urdf_link_m": [0.1, 0.0, -0.02],
                "inertia_urdf_link_com_kg_m2": inertia,
            }
            for side in ("left", "right")
        ]
    }


def inertia_matrix(values: dict[str, float]) -> np.ndarray:
    return np.asarray(
        [
            [values["ixx"], values["ixy"], values["ixz"]],
            [values["ixy"], values["iyy"], values["iyz"]],
            [values["ixz"], values["iyz"], values["izz"]],
        ]
    )


def test_numerical_ankle_split_preserves_combined_zero_pose_properties() -> None:
    source = foot_inertials()

    values, adjustments = MODULE.inertial_values_by_link(source)

    for side in ("left", "right"):
        pitch = values[f"{side}_ankle_pitch_link"]
        roll = values[f"{side}_ankle_roll_link"]
        assert pitch["mass_kg"] == MODULE.ANKLE_PITCH_NUMERICAL_MASS_KG
        assert pitch["mass_kg"] + roll["mass_kg"] == 1.0
        np.testing.assert_allclose(
            np.asarray(pitch["center_urdf_link_m"]),
            np.asarray(roll["center_urdf_link_m"]),
        )
        combined = inertia_matrix(pitch["inertia_urdf_link_com_kg_m2"])
        combined += inertia_matrix(roll["inertia_urdf_link_com_kg_m2"])
        np.testing.assert_allclose(
            combined,
            inertia_matrix(
                next(
                    link["inertia_urdf_link_com_kg_m2"]
                    for link in source["links"]
                    if link["name"] == f"{side}_ankle_roll_link"
                )
            ),
        )
    assert len(adjustments) == 2


def test_cad_to_urdf_axes_are_right_handed_and_canonical() -> None:
    assert np.linalg.det(MODULE.CAD_TO_URDF) == 1.0
    np.testing.assert_array_equal(
        MODULE.CAD_TO_URDF @ np.asarray([1.0, 0.0, 0.0]),
        [0.0, 1.0, 0.0],
    )
    np.testing.assert_array_equal(
        MODULE.CAD_TO_URDF @ np.asarray([0.0, 0.0, 1.0]),
        [1.0, 0.0, 0.0],
    )
    np.testing.assert_array_equal(
        MODULE.CAD_TO_URDF @ np.asarray([0.0, 1.0, 0.0]),
        [0.0, 0.0, 1.0],
    )


def test_bello_mini_can_ids_preserve_original_bello_station_ranges() -> None:
    assert MODULE.MOTOR_CAN_BINDINGS == {
        "waist_yaw_motor": ("can1", 51),
        "waist_roll_motor": ("can1", 52),
        "left_hip_pitch_motor": ("can0", 61),
        "left_hip_roll_motor": ("can0", 62),
        "left_hip_yaw_motor": ("can0", 63),
        "left_knee_motor": ("can0", 64),
        "left_ankle_motor_1": ("can0", 65),
        "left_ankle_motor_2": ("can0", 66),
        "right_hip_pitch_motor": ("can0", 71),
        "right_hip_roll_motor": ("can0", 72),
        "right_hip_yaw_motor": ("can0", 73),
        "right_knee_motor": ("can0", 74),
        "right_ankle_motor_1": ("can0", 75),
        "right_ankle_motor_2": ("can0", 76),
        "left_shoulder_pitch_motor": ("can1", 31),
        "left_shoulder_roll_motor": ("can1", 32),
        "left_shoulder_yaw_motor": ("can1", 33),
        "left_elbow_pitch_motor": ("can1", 34),
        "left_elbow_yaw_motor": ("can1", 35),
        "left_wrist_pitch_motor": ("can1", 36),
        "left_wrist_roll_motor": ("can1", 37),
        "right_shoulder_pitch_motor": ("can1", 41),
        "right_shoulder_roll_motor": ("can1", 42),
        "right_shoulder_yaw_motor": ("can1", 43),
        "right_elbow_pitch_motor": ("can1", 44),
        "right_elbow_yaw_motor": ("can1", 45),
        "right_wrist_pitch_motor": ("can1", 46),
        "right_wrist_roll_motor": ("can1", 47),
    }
    assert len(set(MODULE.MOTOR_CAN_BINDINGS.values())) == 28


def test_cad_direction_inference_never_claims_calibrated_protocol_polarity() -> None:
    direct = MODULE.cad_direction_inference(
        {
            "motor_name": "direct_motor",
            "joint_names": ["joint"],
            "coupling": "direct",
        },
        np.asarray([0.0, -1.0, 0.0]),
        {"joint": [0.0, 1.0, 0.0]},
    )
    differential = MODULE.cad_direction_inference(
        {
            "motor_name": "differential_motor",
            "joint_names": ["ankle_pitch_joint", "ankle_roll_joint"],
            "coupling": "differential",
        },
        np.asarray([0.0, 1.0, 0.0]),
        {
            "ankle_pitch_joint": [0.0, 1.0, 0.0],
            "ankle_roll_joint": [1.0, 0.0, 0.0],
        },
    )

    assert direct == {
        "reference_joint": "joint",
        "axis_dot": -1.0,
        "axis_alignment_sign": -1,
        "candidate_command_direction": -1,
        "status": "cad_geometry_only_protocol_sign_unverified",
    }
    assert differential == {
        "reference_joint": "ankle_pitch_joint",
        "axis_dot": 1.0,
        "axis_alignment_sign": 1,
        "candidate_command_direction": None,
        "status": "cad_pitch_projection_only_differential_unresolved",
    }


def test_collision_geometry_accepts_only_mujoco_compatible_primitives() -> None:
    for primitive in (
        {"type": "box", "size_m": [1.0, 2.0, 3.0]},
        {"type": "cylinder", "radius_m": 0.2, "length_m": 1.0},
        {"type": "sphere", "radius_m": 0.3},
    ):
        import xml.etree.ElementTree as ET

        root = ET.Element("geometry_parent")
        MODULE.add_primitive_geometry(root, primitive)
        assert root.find(f"./geometry/{primitive['type']}") is not None


def test_nominal_grounding_places_both_coplanar_soles_at_zero() -> None:
    inertials = {
        "links": [
            {
                "name": link_name,
                "frame_origin_urdf_world_m": [0.0, 0.0, -0.8271232537850352],
            }
            for link_name in MODULE.FOOT_LINK_NAMES
        ]
    }
    sole = {
        "type": "box",
        "role": MODULE.FOOT_CONTACT_ROLE,
        "name": "sole",
        "origin_xyz_m": [0.0, 0.0, -0.040],
        "origin_rpy_rad": [0.0, 0.0, 0.0],
        "size_m": [0.1, 0.1, 0.012],
    }
    geometry = {
        "links": [
            {"name": link_name, "collision_primitives": [sole]}
            for link_name in MODULE.FOOT_LINK_NAMES
        ]
    }

    result = MODULE.nominal_grounding(inertials, geometry)

    assert result["sole_surface_z_in_urdf_root_frame_m"] == -0.8731232537850352
    assert result["nominal_root_translation_m"] == [0.0, 0.0, 0.8731232537850352]
    assert set(result["grounded_sole_surface_z_m"].values()) == {0.0}


def camera_ownership() -> dict:
    angle = np.deg2rad(67.34999621943078)
    transform = np.eye(4)
    transform[:3, :3] = [
        [1.0, 0.0, 0.0],
        [0.0, np.cos(angle), -np.sin(angle)],
        [0.0, np.sin(angle), np.cos(angle)],
    ]
    transform[:3, 3] = [0.0, -136.8292600604, 53.72063336791]
    return {
        "groups": [
            {
                "id": "ownership-068",
                "kind": "catalog_component",
                "catalog_model": astra.CAMERA_CATALOG_MODEL,
                "reviewed_link": "pelvis_link",
                "occurrences": [
                    {
                        "source_name": astra.CAMERA_BODY_SOURCE_NAME,
                        "transform": transform.reshape(-1, order="F").tolist(),
                    }
                ],
            }
        ]
    }


def test_astra_mount_uses_cad_pose_without_moving_mass_from_pelvis() -> None:
    frame = next(
        frame
        for frame in MODULE.fixed_sensor_frames(camera_ownership())
        if frame["link_name"] == "astra_mount_frame"
    )
    angle = np.deg2rad(67.34999621943078)

    assert frame["parent_link"] == "pelvis_link"
    assert frame["link_name"] == "astra_mount_frame"
    np.testing.assert_allclose(
        frame["origin_xyz_parent_m"],
        [0.05372063336791, 0.0, -0.1368292600604],
        atol=1e-14,
    )
    np.testing.assert_allclose(
        frame["origin_rpy_parent_rad"],
        [0.0, angle, 0.0],
        atol=1e-14,
    )
    expected_view_axis = [np.cos(angle), 0.0, -np.sin(angle)]
    np.testing.assert_allclose(
        np.asarray(frame["rotation_parent_from_frame"])[:, 0],
        expected_view_axis,
        atol=1e-14,
    )
    assert "massless fixed frame" in frame["mass_policy"]

    root = ET.Element("robot")
    MODULE.add_fixed_sensor_frame(
        root,
        frame,
        geometry=None,
        mesh=None,
        urdf_path=Path("/tmp/bello_mini.urdf"),
    )
    link = root.find("./link[@name='astra_mount_frame']")
    assert link is not None
    assert link.find("inertial") is None


def test_camera_frame_rejects_mass_owned_by_camera_link() -> None:
    ownership = camera_ownership()
    ownership["groups"][0]["reviewed_link"] = "pelvis_camera_link"

    with pytest.raises(ValueError, match="mass must remain owned by pelvis_link"):
        MODULE.fixed_sensor_frames(ownership)


def test_astra_sensor_tree_uses_hard_coded_nominal_offsets() -> None:
    frames = MODULE.fixed_sensor_frames(camera_ownership())
    by_name = {frame["link_name"]: frame for frame in frames}

    assert len(frames) == 11
    assert by_name["camera_link"]["parent_link"] == "astra_mount_frame"
    assert by_name["camera_link"]["origin_xyz_parent_m"] == [0.0043, 0.02, 0.0]
    assert by_name["camera_depth_frame"]["origin_xyz_parent_m"] == [0.0] * 3
    assert by_name["camera_left_ir_frame"]["origin_xyz_parent_m"] == [0.0] * 3
    assert by_name["camera_right_ir_frame"]["origin_xyz_parent_m"] == [
        0.0,
        -0.04,
        0.0,
    ]
    assert by_name["camera_color_frame"]["origin_xyz_parent_m"] == [
        0.002,
        -0.01,
        0.0,
    ]


def test_nominal_imu_frame_matches_original_bello_pelvis_convention() -> None:
    frame = MODULE.nominal_imu_frame()

    assert frame["link_name"] == "imu_link"
    assert frame["joint_name"] == "imu_joint"
    assert frame["parent_link"] == "pelvis_link"
    assert frame["origin_xyz_parent_m"] == [0.0, 0.0, 0.0]
    assert frame["origin_rpy_parent_rad"] == [0.0, 0.0, 0.0]
    np.testing.assert_array_equal(frame["rotation_parent_from_frame"], np.eye(3))
    assert frame["origin_xyz_pelvis_m"] == [0.0, 0.0, 0.0]
    np.testing.assert_array_equal(frame["rotation_pelvis_from_frame"], np.eye(3))
    assert "no reviewed IMU component" in frame["datum"]

    root = ET.Element("robot")
    MODULE.add_fixed_sensor_frame(
        root,
        frame,
        geometry=None,
        mesh=None,
        urdf_path=Path("/tmp/bello_mini.urdf"),
    )
    link = root.find("./link[@name='imu_link']")
    joint = root.find("./joint[@name='imu_joint']")
    assert link is not None and link.find("inertial") is None
    assert joint is not None
    assert joint.find("parent").attrib["link"] == "pelvis_link"
    assert joint.find("child").attrib["link"] == "imu_link"
    assert joint.find("origin").attrib == {"xyz": "0 0 0", "rpy": "0 0 0"}


def test_astra_optical_frames_use_exact_rep_103_rotation() -> None:
    frames = MODULE.fixed_sensor_frames(camera_ownership())
    optical_frames = [
        frame for frame in frames if frame["link_name"].endswith("_optical_frame")
    ]

    assert len(optical_frames) == 4
    for frame in optical_frames:
        assert frame["origin_xyz_parent_m"] == [0.0] * 3
        assert frame["origin_rpy_parent_rad"] == [
            -1.5707963267948966,
            0.0,
            -1.5707963267948966,
        ]
        rotation = np.asarray(frame["rotation_parent_from_frame"])
        np.testing.assert_allclose(rotation[:, 2], [1.0, 0.0, 0.0], atol=1e-15)
        np.testing.assert_allclose(rotation[:, 0], [0.0, -1.0, 0.0], atol=1e-15)
        np.testing.assert_allclose(rotation[:, 1], [0.0, 0.0, -1.0], atol=1e-15)


def test_astra_fov_uses_published_depth_and_color_specs() -> None:
    report = astra.camera_cad_report()

    assert report["field_of_view"] == {
        "source": "https://store.orbbec.com/products/astra-stereo-s-u3",
        "depth": {
            "frame": "camera_depth_optical_frame",
            "horizontal_deg": 67.9,
            "vertical_deg": 45.3,
            "minimum_range_m": 0.25,
            "maximum_range_m": 2.5,
        },
        "color": {
            "frame": "camera_color_optical_frame",
            "horizontal_deg": 71.5,
            "vertical_deg": 56.7,
            "visualization_length_m": 2.5,
        },
    }
