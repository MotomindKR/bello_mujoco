from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "tools"
    / "robot_description"
    / "build_geometry_manifest.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "build_geometry_manifest", MODULE_PATH
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = MODULE
MODULE_SPEC.loader.exec_module(MODULE)


def test_segment_orientation_maps_cylinder_z_axis_to_segment() -> None:
    vector = np.asarray([0.25, -0.10, 0.40])

    roll, pitch, yaw = MODULE.vector_rpy_from_z(vector)
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    rotation = np.asarray(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ]
    )

    np.testing.assert_allclose(rotation[:, 2], vector / np.linalg.norm(vector))


@pytest.mark.parametrize(
    ("label", "mass", "maximum_size", "included"),
    [
        ("M4 bolt", 0.01, 40.0, False),
        ("RS03_Spacer", 0.10, 100.0, False),
        ("bearing", 0.02, 40.0, False),
        ("outer cover", 0.0005, 50.0, True),
        ("tiny detail", 0.0005, 20.0, False),
    ],
)
def test_visual_occurrence_policy(
    label: str, mass: float, maximum_size: float, included: bool
) -> None:
    occurrence = {
        "source_name": "part",
        "source_label": label,
        "estimated_mass_kg": mass,
        "bounds_mm": {"min": [0.0, 0.0, 0.0], "max": [maximum_size] * 3},
    }

    assert MODULE.visual_occurrence_decision(occurrence)[0] is included


def test_foot_sole_primitives_match_reviewed_bottom_plane() -> None:
    values = MODULE.foot_collision_primitives("left_ankle_roll_link")
    sole = [value for value in values if value["role"] == "foot_contact_envelope"]

    assert len(sole) == 5
    assert sum(value["type"] == "box" for value in sole) == 2
    assert sum(value["type"] == "cylinder" for value in sole) == 3
    for value in sole:
        half_height = (
            value["size_m"][2] * 0.5
            if value["type"] == "box"
            else value["length_m"] * 0.5
        )
        assert value["origin_xyz_m"][2] - half_height == pytest.approx(-0.046)
        assert value["origin_xyz_m"][2] + half_height == pytest.approx(-0.034)
    toe = next(value for value in sole if value["name"].endswith("_sole_toe"))
    assert toe["origin_xyz_m"][0] + toe["radius_m"] == pytest.approx(0.150)

    heel_center = next(
        value for value in sole if value["name"].endswith("_sole_heel_center")
    )
    heel_fillet_centers = sorted(
        value["origin_xyz_m"][:2]
        for value in sole
        if "_sole_heel_" in value["name"]
        and not value["name"].endswith("_sole_heel_center")
    )
    assert heel_center["origin_xyz_m"][0] - heel_center["size_m"][0] * 0.5 == pytest.approx(-0.070)
    assert heel_fillet_centers == [[-0.050, -0.025], [-0.050, 0.025]]


def test_anonymous_endpoint_uses_hand_mount_plane_not_hand_center() -> None:
    classification = {
        "end_effector_policy": {
            "source_assemblies": [
                {
                    "id": "left-hand",
                    "bounds_mm": {
                        "min": [170.0, -451.0, -65.0],
                        "max": [214.0, -258.9, 132.0],
                    },
                },
                {
                    "id": "right-hand",
                    "bounds_mm": {
                        "min": [-214.0, -451.0, -65.0],
                        "max": [-170.0, -258.9, 132.0],
                    },
                },
            ]
        }
    }
    ownership = {
        "joints": [
            {
                "side": side,
                "station": "wrist_roll",
                "origin_cad_mm": [sign * 206.25, -221.4, -14.0],
                "child_link": f"{side}_wrist_roll_link",
            }
            for side, sign in (("left", 1.0), ("right", -1.0))
        ]
    }

    endpoints = MODULE.anonymous_end_effectors(classification, ownership)

    for endpoint in endpoints:
        np.testing.assert_allclose(
            endpoint["origin_xyz_parent_m"], [0.0, 0.0, -0.0375]
        )
        assert endpoint["sphere_center_xyz_link_m"] == [0.0, 0.0, -0.025]
        assert endpoint["radius_m"] == 0.025
