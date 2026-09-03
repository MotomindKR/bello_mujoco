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
    / "build_inertial_manifest.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "build_inertial_manifest", MODULE_PATH
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = MODULE
MODULE_SPEC.loader.exec_module(MODULE)


def test_coordinate_conversion_is_right_handed_and_maps_axes() -> None:
    matrix = MODULE.CAD_TO_URDF

    np.testing.assert_allclose(matrix @ [0, 0, 1], [1, 0, 0])
    np.testing.assert_allclose(matrix @ [1, 0, 0], [0, 1, 0])
    np.testing.assert_allclose(matrix @ [0, 1, 0], [0, 0, 1])
    assert np.linalg.det(matrix) == pytest.approx(1.0)


def test_combined_mass_properties_apply_parallel_axis_theorem() -> None:
    unit = np.eye(3)

    mass, center, inertia = MODULE.combine_mass_properties(
        [
            (2.0, np.asarray([-1.0, 0.0, 0.0]), unit),
            (2.0, np.asarray([1.0, 0.0, 0.0]), unit),
        ]
    )

    assert mass == 4.0
    np.testing.assert_allclose(center, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(inertia, np.diag([2.0, 6.0, 6.0]))


def test_catalog_contains_every_bello_mini_motor_model() -> None:
    assert MODULE.MOTOR_MODELS == {
        "ROBSTRIDE_00": {"mass_kg": 0.310, "mass_tolerance_kg": 0.003},
        "ROBSTRIDE_02": {"mass_kg": 0.380, "mass_tolerance_kg": 0.003},
        "ROBSTRIDE_03": {"mass_kg": 0.900, "mass_tolerance_kg": 0.020},
        "ROBSTRIDE_04": {"mass_kg": 1.420, "mass_tolerance_kg": 0.020},
        "ROBSTRIDE_06": {"mass_kg": 0.621, "mass_tolerance_kg": None},
    }
