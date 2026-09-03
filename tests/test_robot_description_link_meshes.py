from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


MODULE_PATH = (
    Path(__file__).parents[1] / "tools" / "robot_description" / "export_link_meshes.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location("export_link_meshes", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = MODULE
MODULE_SPEC.loader.exec_module(MODULE)


def test_transformed_instance_uses_link_local_urdf_coordinates() -> None:
    vertices = np.asarray([[1.0, 2.0, 3.0], [2.0, 2.0, 3.0], [1.0, 3.0, 3.0]])
    faces = np.asarray([[0, 1, 2]])
    transform = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 20, 30, 1]

    mesh = MODULE.transformed_instance(
        vertices,
        faces,
        transform,
        np.asarray([10.0, 20.0, 30.0]),
    )

    np.testing.assert_allclose(
        mesh.vertices,
        [[3.0, 1.0, 2.0], [3.0, 2.0, 2.0], [3.0, 1.0, 3.0]],
    )
    np.testing.assert_array_equal(mesh.faces, faces)


def test_reflected_instance_reverses_triangle_winding() -> None:
    vertices = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.asarray([[0, 1, 2]])
    transform = [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

    mesh = MODULE.transformed_instance(vertices, faces, transform, np.zeros(3))

    np.testing.assert_array_equal(mesh.faces, [[0, 2, 1]])


def test_transformed_instance_supports_rotated_fixed_camera_frame() -> None:
    vertices = np.asarray([[1.0, 2.0, 3.0], [2.0, 2.0, 3.0], [1.0, 3.0, 3.0]])
    faces = np.asarray([[0, 1, 2]])
    source_to_cad = np.asarray(
        [
            [0.0, -1.0, 0.0, 10.0],
            [1.0, 0.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    frame_to_source = np.asarray(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    )
    rotation_urdf_world_from_frame = (
        MODULE.CAD_TO_URDF @ source_to_cad[:3, :3] @ frame_to_source
    )

    mesh = MODULE.transformed_instance(
        vertices,
        faces,
        source_to_cad.reshape(-1, order="F").tolist(),
        source_to_cad[:3, 3],
        rotation_urdf_world_from_frame,
    )

    np.testing.assert_allclose(
        mesh.vertices,
        [[3.0, 1.0, 2.0], [3.0, 2.0, 2.0], [3.0, 1.0, 3.0]],
    )
