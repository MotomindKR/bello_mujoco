from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "tools"
    / "robot_description"
    / "build_joint_review_manifest.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "build_joint_review_manifest", MODULE_PATH
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = MODULE
MODULE_SPEC.loader.exec_module(MODULE)


def test_bello_mini_motor_names_cover_28_unique_actuators() -> None:
    specs = MODULE.motor_specs()

    assert len(specs) == 28
    assert len({spec.motor_id for spec in specs}) == 28
    assert len({spec.motor_name for spec in specs}) == 28
    assert {spec.model for spec in specs} == {
        "ROBSTRIDE_00",
        "ROBSTRIDE_02",
        "ROBSTRIDE_03",
        "ROBSTRIDE_04",
        "ROBSTRIDE_06",
    }


def test_bello_mini_joint_tree_has_seven_dof_arms() -> None:
    joints = MODULE.joint_specs()
    movable = [joint for joint in joints if joint.joint_type == "revolute"]

    assert len(movable) == 28
    for side in ("left", "right"):
        arm = [
            joint
            for joint in movable
            if joint.side == side
            and joint.station
            in {
                "shoulder_pitch",
                "shoulder_roll",
                "shoulder_yaw",
                "elbow_pitch",
                "wrist_yaw",
                "wrist_pitch",
                "wrist_roll",
            }
        ]
        assert len(arm) == 7
        assert f"{side}_wrist_yaw_joint" in {joint.name for joint in arm}
        assert arm[-1].name == f"{side}_wrist_roll_joint"

    motor_names = {spec.motor_name for spec in MODULE.motor_specs()}
    assert {"left_wrist_yaw_motor", "right_wrist_yaw_motor"} <= motor_names
    assert not {"left_elbow_yaw_motor", "right_elbow_yaw_motor"} & motor_names


def test_ankle_motors_are_differential_bindings() -> None:
    specs = MODULE.motor_specs()
    ankles = [spec for spec in specs if "ankle_motor" in spec.motor_name]

    assert len(ankles) == 4
    assert all(spec.coupling == "differential" for spec in ankles)
    assert all(len(spec.joint_names) == 2 for spec in ankles)
    direct_joint_motor_ids = {
        joint.motor_ids[0]
        for joint in MODULE.joint_specs()
        if joint.coupling == "direct" and joint.joint_type == "revolute"
    }
    assert not direct_joint_motor_ids.intersection(spec.motor_id for spec in ankles)


def test_proposed_links_form_one_single_parent_tree() -> None:
    joints = MODULE.joint_specs()
    child_links = [joint.child_link for joint in joints]
    links = {"pelvis_link"}
    for joint in joints:
        links.add(joint.parent_link)
        links.add(joint.child_link)

    assert len(links) == 31
    assert len(child_links) == len(set(child_links))
    assert "pelvis_link" not in child_links


def test_ankle_pivot_is_orthogonal_bearing_intersection() -> None:
    def occurrence(center: tuple[float, float, float], shortest_axis: int) -> dict:
        size = [20.0, 20.0, 20.0]
        size[shortest_axis] = 10.0
        return {
            "motor_assembly": None,
            "measured_assembly": None,
            "anonymous_end_effector": None,
            "bounds_mm": {
                "min": [value - size[index] / 2 for index, value in enumerate(center)],
                "max": [value + size[index] / 2 for index, value in enumerate(center)],
            },
        }

    manifest = {
        "components": [
            {
                "source_label": "22SUS-6B2",
                "occurrences": [
                    occurrence((8.0, -20.0, 30.0), 0),
                    occurrence((12.0, -20.0, 30.0), 0),
                    occurrence((10.0, -20.0, 27.0), 2),
                    occurrence((10.0, -20.0, 33.0), 2),
                    occurrence((-8.0, -20.0, 30.0), 0),
                    occurrence((-12.0, -20.0, 30.0), 0),
                    occurrence((-10.0, -20.0, 27.0), 2),
                    occurrence((-10.0, -20.0, 33.0), 2),
                ],
            }
        ]
    }

    pivots = MODULE.ankle_pivot_origins(manifest)

    assert pivots["left"] == pytest.approx([10.0, -20.0, 30.0])
    assert pivots["right"] == pytest.approx([-10.0, -20.0, 30.0])
