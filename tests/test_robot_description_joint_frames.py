from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "tools"
    / "robot_description"
    / "build_joint_frame_manifest.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "build_joint_frame_manifest", MODULE_PATH
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = MODULE
MODULE_SPEC.loader.exec_module(MODULE)


def test_dominant_axis_line_uses_concentric_cad_circle_cluster() -> None:
    candidates = [
        {"center_mm": [2.0, 3.0, z], "axis": [0.0, 0.0, 1.0], "radius_mm": r}
        for z, r in zip(range(12), [5.0, 6.0, 7.0, 8.0] * 3, strict=True)
    ]
    candidates.extend(
        {
            "center_mm": [20.0, 30.0, z],
            "axis": [0.0, 0.0, 1.0],
            "radius_mm": 6.0,
        }
        for z in range(4)
    )

    result = MODULE.dominant_axis_line(
        candidates,
        expected_axis=[0.0, 0.0, 1.0],
        reference_point=[2.1, 3.1, 50.0],
    )

    assert result["origin_mm"] == pytest.approx([2.0, 3.0, 50.0])
    assert result["cluster_circle_count"] == 12
    assert result["unique_radius_count"] == 4


def test_dominant_axis_line_rejects_motor_center_off_axis() -> None:
    candidates = [
        {"center_mm": [0.0, y, 0.0], "axis": [0.0, 1.0, 0.0], "radius_mm": r}
        for y, r in zip(range(12), [5.0, 6.0, 7.0, 8.0] * 3, strict=True)
    ]

    with pytest.raises(ValueError, match="unexpectedly far"):
        MODULE.dominant_axis_line(
            candidates,
            expected_axis=[0.0, 1.0, 0.0],
            reference_point=[4.0, 0.0, 0.0],
        )


def test_axis_origin_symmetry_ignores_only_along_axis_gauge() -> None:
    left, right, residual = MODULE.symmetrize_axis_origins(
        [120.0, -500.0, 20.0],
        [-135.0, -500.0, 20.0],
        left_axis=[1.0, 0.0, 0.0],
    )

    assert left == pytest.approx([127.5, -500.0, 20.0])
    assert right == pytest.approx([-127.5, -500.0, 20.0])
    assert residual == pytest.approx(0.0)


def test_axis_origin_symmetry_rejects_physical_line_mismatch() -> None:
    with pytest.raises(ValueError, match="not symmetric"):
        MODULE.symmetrize_axis_origins(
            [120.0, -500.0, 20.0],
            [-120.0, -501.0, 20.0],
            left_axis=[1.0, 0.0, 0.0],
        )
