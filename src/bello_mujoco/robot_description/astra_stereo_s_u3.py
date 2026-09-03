"""Nominal CAD-backed frames for the Orbbec Astra Stereo S U3."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np


CAMERA_CATALOG_MODEL = "ORBBEC_ASTRA_STEREO_S_U3"
CAMERA_BODY_SOURCE_NAME = "Compound008"
CAMERA_PARENT_LINK = "pelvis_link"
ASTRA_MOUNT_FRAME = "astra_mount_frame"
OPTICAL_FRAME_RPY_RAD = (-math.pi / 2.0, 0.0, -math.pi / 2.0)

# Compound008's local axes are +Z through the lenses, +X camera-left, and +Y
# camera-up. Its exact IR aperture axes are at local X = +/-20 mm, Y = 0.
FRAME_TO_SOURCE = np.asarray(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

# The STEP contains no optical prescription or calibrated pinhole datum. Use the
# requested nominal mechanical model: the IR/depth center is 2 mm behind the
# source-local frontmost plane at Z=6.30 mm, while RGB is 2 mm farther forward.
# These are fixed values, not runtime calibration parameters.
ASTRA_OPTICAL_X_OFFSET_M = 0.00430
ASTRA_OPTICAL_Z_OFFSET_M = 0.0
ASTRA_RGB_X_OFFSET_M = 0.002
ASTRA_RGB_Z_OFFSET_M = 0.0
ASTRA_RIGHT_IR_X_OFFSET_M = 0.0
ASTRA_RIGHT_IR_Z_OFFSET_M = 0.0

# Exact Compound008 bounds, ordered in the normal camera frame: X depth,
# Y width, Z height. They intentionally retain the CAD dimensions rather than
# the rounded product-summary dimensions.
CAD_BOUNDS_SOURCE_MM = {
    "min": [-32.65000000115047, -11.350000000010121, -7.30104382781445],
    "max": [32.6500000004, 11.35000000000012, 6.30000000411],
}
CAD_SIZE_FRAME_M = [
    0.01360104383192445,
    0.06530000000155047,
    0.02270000001013024,
]
CAD_CENTER_FRAME_M = [-0.000500521911852225, -3.752349542640809e-13, 0.0]
FIELD_OF_VIEW = {
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

# No unambiguous pair of 2.30 mm mounting-hole cylinders exists in the imported
# camera solid. The CAD source datum is nevertheless mechanically meaningful:
# it is exactly on both IR aperture-axis midpoint planes and on the enclosure
# width/height symmetry planes. Its axial coordinate remains the source datum.
MOUNT_ORIGIN_SOURCE_MM = np.zeros(3, dtype=np.float64)
MOUNT_DATUM_DESCRIPTION = (
    "Compound008 local origin: exact midpoint of the CAD left/right IR aperture "
    "axes and the enclosure width/height symmetry planes; the imported solid "
    "does not expose an unambiguous 2.30 mm, 54.99 mm-spaced mounting-hole pair, "
    "so source-local Z=0 is retained as the axial mechanical datum"
)


@dataclass(frozen=True)
class AstraCadPlacement:
    """The single reviewed Astra occurrence and its source-bound placement."""

    group_id: str
    occurrence: dict[str, Any]
    source_to_cad: np.ndarray


def clean_vector(
    values: Iterable[float], *, tolerance: float = 1e-12
) -> np.ndarray:
    result = np.asarray(list(values), dtype=np.float64)
    result[np.abs(result) < tolerance] = 0.0
    return result


def rotation_to_rpy(rotation: np.ndarray) -> np.ndarray:
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9):
        raise ValueError("camera-frame rotation is not orthogonal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-9):
        raise ValueError("camera-frame rotation is not right-handed")
    pitch = math.asin(max(-1.0, min(1.0, -float(rotation[2, 0]))))
    if abs(math.cos(pitch)) > 1e-9:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = 0.0
        yaw = math.atan2(-rotation[0, 1], rotation[1, 1])
    return clean_vector((roll, pitch, yaw))


def rpy_to_rotation(rpy: Iterable[float]) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def find_camera_placement(ownership: dict[str, Any]) -> AstraCadPlacement:
    groups = [
        group
        for group in ownership["groups"]
        if group.get("kind") == "catalog_component"
        and group.get("catalog_model") == CAMERA_CATALOG_MODEL
    ]
    if len(groups) != 1:
        raise ValueError(
            f"expected one {CAMERA_CATALOG_MODEL} CAD group, got {len(groups)}"
        )
    group = groups[0]
    if group["reviewed_link"] != CAMERA_PARENT_LINK:
        raise ValueError(
            f"camera mass must remain owned by {CAMERA_PARENT_LINK}: {group['id']}"
        )
    occurrences = [
        occurrence
        for occurrence in group["occurrences"]
        if occurrence["source_name"] == CAMERA_BODY_SOURCE_NAME
    ]
    if len(occurrences) != 1:
        raise ValueError(
            f"expected one {CAMERA_BODY_SOURCE_NAME} camera body, got "
            f"{len(occurrences)}"
        )
    transform = np.asarray(
        occurrences[0]["transform"], dtype=np.float64
    ).reshape((4, 4), order="F")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-10):
        raise ValueError("camera CAD transform is not affine")
    return AstraCadPlacement(group["id"], occurrences[0], transform)


def mount_pose(
    ownership: dict[str, Any], cad_to_urdf: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    placement = find_camera_placement(ownership)
    source_point_cad_mm = (
        placement.source_to_cad[:3, :3] @ MOUNT_ORIGIN_SOURCE_MM
        + placement.source_to_cad[:3, 3]
    )
    origin_parent_m = cad_to_urdf @ (source_point_cad_mm * 1e-3)
    rotation_parent_from_mount = (
        cad_to_urdf @ placement.source_to_cad[:3, :3] @ FRAME_TO_SOURCE
    )
    return clean_vector(origin_parent_m), rotation_parent_from_mount


def _frame(
    *,
    link_name: str,
    joint_name: str,
    parent_link: str,
    xyz: Iterable[float],
    rpy: Iterable[float],
    convention: str,
) -> dict[str, Any]:
    rpy_values = clean_vector(rpy)
    return {
        "link_name": link_name,
        "joint_name": joint_name,
        "parent_link": parent_link,
        "origin_xyz_parent_m": clean_vector(xyz).tolist(),
        "origin_rpy_parent_rad": rpy_values.tolist(),
        "rotation_parent_from_frame": rpy_to_rotation(rpy_values).tolist(),
        "frame_convention": convention,
    }


def camera_frames(
    ownership: dict[str, Any], cad_to_urdf: np.ndarray
) -> list[dict[str, Any]]:
    """Return the complete nominal Orbbec-style fixed-frame tree."""

    placement = find_camera_placement(ownership)
    mount_xyz, mount_rotation = mount_pose(ownership, cad_to_urdf)
    body_convention = "+X forward through lenses, +Y camera-left, +Z up"
    optical_convention = "+X image-right, +Y image-down, +Z optical-forward"
    frames = [
        {
            **_frame(
                link_name=ASTRA_MOUNT_FRAME,
                joint_name="astra_mount_joint",
                parent_link=CAMERA_PARENT_LINK,
                xyz=mount_xyz,
                rpy=rotation_to_rpy(mount_rotation),
                convention=body_convention,
            ),
            "rotation_parent_from_frame": mount_rotation.tolist(),
            "catalog_model": CAMERA_CATALOG_MODEL,
            "source_group_id": placement.group_id,
            "source_occurrence": CAMERA_BODY_SOURCE_NAME,
            "datum": MOUNT_DATUM_DESCRIPTION,
            "mass_policy": (
                "massless fixed frame; the 30 g camera mass and transformed "
                "inertia remain combined into pelvis_link at the exact CAD pose"
            ),
        },
        _frame(
            link_name="camera_link",
            joint_name="astra_mount_to_camera_joint",
            parent_link=ASTRA_MOUNT_FRAME,
            xyz=(ASTRA_OPTICAL_X_OFFSET_M, 0.020, ASTRA_OPTICAL_Z_OFFSET_M),
            rpy=(0.0, 0.0, 0.0),
            convention=body_convention,
        ),
        _frame(
            link_name="camera_depth_frame",
            joint_name="camera_depth_joint",
            parent_link="camera_link",
            xyz=(0.0, 0.0, 0.0),
            rpy=(0.0, 0.0, 0.0),
            convention=body_convention,
        ),
        _frame(
            link_name="camera_depth_optical_frame",
            joint_name="camera_depth_optical_joint",
            parent_link="camera_depth_frame",
            xyz=(0.0, 0.0, 0.0),
            rpy=OPTICAL_FRAME_RPY_RAD,
            convention=optical_convention,
        ),
        _frame(
            link_name="camera_left_ir_frame",
            joint_name="camera_left_ir_joint",
            parent_link="camera_depth_frame",
            xyz=(0.0, 0.0, 0.0),
            rpy=(0.0, 0.0, 0.0),
            convention=body_convention,
        ),
        _frame(
            link_name="camera_left_ir_optical_frame",
            joint_name="camera_left_ir_optical_joint",
            parent_link="camera_left_ir_frame",
            xyz=(0.0, 0.0, 0.0),
            rpy=OPTICAL_FRAME_RPY_RAD,
            convention=optical_convention,
        ),
        _frame(
            link_name="camera_right_ir_frame",
            joint_name="camera_right_ir_joint",
            parent_link="camera_depth_frame",
            xyz=(
                ASTRA_RIGHT_IR_X_OFFSET_M,
                -0.040,
                ASTRA_RIGHT_IR_Z_OFFSET_M,
            ),
            rpy=(0.0, 0.0, 0.0),
            convention=body_convention,
        ),
        _frame(
            link_name="camera_right_ir_optical_frame",
            joint_name="camera_right_ir_optical_joint",
            parent_link="camera_right_ir_frame",
            xyz=(0.0, 0.0, 0.0),
            rpy=OPTICAL_FRAME_RPY_RAD,
            convention=optical_convention,
        ),
        _frame(
            link_name="camera_color_frame",
            joint_name="camera_color_joint",
            parent_link="camera_depth_frame",
            xyz=(ASTRA_RGB_X_OFFSET_M, -0.010, ASTRA_RGB_Z_OFFSET_M),
            rpy=(0.0, 0.0, 0.0),
            convention=body_convention,
        ),
        _frame(
            link_name="camera_color_optical_frame",
            joint_name="camera_color_optical_joint",
            parent_link="camera_color_frame",
            xyz=(0.0, 0.0, 0.0),
            rpy=OPTICAL_FRAME_RPY_RAD,
            convention=optical_convention,
        ),
    ]
    world_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {
        CAMERA_PARENT_LINK: (np.zeros(3), np.eye(3))
    }
    for frame in frames:
        parent_xyz, parent_rotation = world_poses[frame["parent_link"]]
        local_xyz = np.asarray(frame["origin_xyz_parent_m"])
        local_rotation = np.asarray(frame["rotation_parent_from_frame"])
        world_xyz = parent_xyz + parent_rotation @ local_xyz
        world_rotation = parent_rotation @ local_rotation
        frame["origin_xyz_pelvis_m"] = clean_vector(world_xyz).tolist()
        frame["rotation_pelvis_from_frame"] = world_rotation.tolist()
        world_poses[frame["link_name"]] = (world_xyz, world_rotation)
    return frames


def camera_collision_primitive() -> dict[str, Any]:
    return {
        "name": "astra_stereo_s_u3_enclosure",
        "type": "box",
        "role": "camera_enclosure",
        "origin_xyz_m": list(CAD_CENTER_FRAME_M),
        "origin_rpy_rad": [0.0, 0.0, 0.0],
        "size_m": list(CAD_SIZE_FRAME_M),
    }


def camera_cad_report() -> dict[str, Any]:
    return {
        "model": CAMERA_CATALOG_MODEL,
        "mechanical_frame": ASTRA_MOUNT_FRAME,
        "mount_datum": MOUNT_DATUM_DESCRIPTION,
        "mount_origin_source_mm": MOUNT_ORIGIN_SOURCE_MM.tolist(),
        "cad_source_axes": {
            "camera_forward": "+Z",
            "camera_left": "+X",
            "camera_up": "+Y",
        },
        "cad_bounds_source_mm": CAD_BOUNDS_SOURCE_MM,
        "cad_size_camera_frame_m": CAD_SIZE_FRAME_M,
        "published_nominal_size_camera_frame_m": [0.0123, 0.0653, 0.0225],
        "visible_aperture_axes_source_mm": {
            "left_ir": [20.0, 0.0, 4.95],
            "right_ir": [-20.0, 0.0, 4.95],
            "color": [10.0, 0.0, 4.95],
            "ldm_projector": [0.0, 0.0, 4.95],
        },
        "modeled_optical_origins_source_mm": {
            "left_ir_depth": [20.0, 0.0, 4.30],
            "right_ir": [-20.0, 0.0, 4.30],
            "color": [10.0, 0.0, 6.30],
        },
        "optical_origin_status": "hard_coded_nominal_mechanical_not_factory_calibrated",
        "optical_origin_derivation": (
            "requested nominal effective model: IR/depth is 2 mm behind the "
            "Compound008 frontmost source-local Z=6.30 mm plane; RGB is 2 mm "
            "forward of IR/depth; the STEP encodes no image-sensor plane, optical "
            "prescription, or calibrated pinhole datum"
        ),
        "hard_coded_nominal_offsets_m": {
            "mount_to_left_ir_depth": [
                ASTRA_OPTICAL_X_OFFSET_M,
                0.020,
                ASTRA_OPTICAL_Z_OFFSET_M,
            ],
            "depth_to_right_ir": [
                ASTRA_RIGHT_IR_X_OFFSET_M,
                -0.040,
                ASTRA_RIGHT_IR_Z_OFFSET_M,
            ],
            "depth_to_color": [
                ASTRA_RGB_X_OFFSET_M,
                -0.010,
                ASTRA_RGB_Z_OFFSET_M,
            ],
        },
        "nominal_lateral_geometry_m": {
            "left_ir_from_mount_y": 0.020,
            "right_ir_from_mount_y": -0.020,
            "color_from_mount_y": 0.010,
            "right_ir_from_camera_link_y": -0.040,
            "color_from_camera_link_y": -0.010,
        },
        "field_of_view": FIELD_OF_VIEW,
        "runtime_tf_policy": (
            "publish the sensor-frame subtree from either robot_state_publisher "
            "or the Orbbec driver, never both; prefer driver factory extrinsics "
            "when available and frame-compatible"
        ),
    }
