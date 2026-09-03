from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


MODULE_PATH = (
    Path(__file__).parents[1]
    / "tools"
    / "robot_description"
    / "build_link_ownership_manifest.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "build_link_ownership_manifest", MODULE_PATH
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = MODULE
MODULE_SPEC.loader.exec_module(MODULE)


def test_arm_structure_uses_output_side_station_intervals() -> None:
    assert MODULE.arm_link("left", 260.0) == "torso_link"
    assert MODULE.arm_link("left", 240.0) == "left_shoulder_pitch_link"
    assert MODULE.arm_link("left", 180.0) == "left_shoulder_roll_link"
    assert MODULE.arm_link("left", 80.0) == "left_upper_arm_link"
    assert MODULE.arm_link("left", 0.0) == "left_elbow_link"
    assert MODULE.arm_link("left", -100.0) == "left_lower_arm_link"
    assert MODULE.arm_link("left", -180.0) == "left_wrist_link"
    assert MODULE.arm_link("left", -240.0) == "left_wrist_roll_link"


def test_leg_structure_never_uses_virtual_ankle_pitch_link() -> None:
    assert MODULE.leg_link("right", -100.0) == "pelvis_link"
    assert MODULE.leg_link("right", -190.0) == "right_hip_pitch_link"
    assert MODULE.leg_link("right", -250.0) == "right_hip_roll_link"
    assert MODULE.leg_link("right", -400.0) == "right_hip_yaw_link"
    assert MODULE.leg_link("right", -700.0) == "right_knee_link"
    assert MODULE.leg_link("right", -850.0) == "right_ankle_roll_link"


def test_occurrence_side_preserves_center_spanning_parts() -> None:
    assert MODULE.occurrence_side({"min": [2.0, 0, 0], "max": [12.0, 1, 1]}) == "left"
    assert (
        MODULE.occurrence_side({"min": [-12.0, 0, 0], "max": [-2.0, 1, 1]}) == "right"
    )
    assert (
        MODULE.occurrence_side({"min": [-60.0, 0, 0], "max": [60.0, 1, 1]}) == "center"
    )


def test_motor_parent_prefers_ankle_pitch_parent_for_remote_pair() -> None:
    frames = {
        "joints": [
            {
                "name": "left_ankle_pitch_joint",
                "station": "ankle_pitch",
                "parent_link": "left_knee_link",
                "motor_ids": ["motor-a", "motor-b"],
            },
            {
                "name": "left_ankle_roll_joint",
                "station": "ankle_roll",
                "parent_link": "left_ankle_pitch_link",
                "motor_ids": ["motor-a", "motor-b"],
            },
        ],
        "motors": [
            {
                "id": "motor-a",
                "role": "remote_transmission_actuator",
                "creates_joint_at_motor_center": False,
            },
            {
                "id": "motor-b",
                "role": "remote_transmission_actuator",
                "creates_joint_at_motor_center": False,
            },
        ],
    }

    assert MODULE.motor_parent_links(frames) == {
        "motor-a": "left_knee_link",
        "motor-b": "left_knee_link",
    }


def test_bilateral_spatial_residual_accepts_reused_arm_orientation() -> None:
    left = {
        "path_prefix": "/root/LinkGroup2988/left_part",
        "center_mm": [181.6773, 25.6, -5.0],
    }
    right = {
        "path_prefix": "/root/LinkGroup2926/right_part",
        "center_mm": [-230.8227, 25.6, -5.0],
    }

    assert MODULE.bilateral_spatial_residual_mm(left, right) < 1e-6
