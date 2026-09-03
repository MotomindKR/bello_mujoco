"""Build the reviewed Bello Mini URDF and calibration-gated motor topology."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from bello_mujoco.robot_description.astra_stereo_s_u3 import (
    camera_cad_report,
    camera_frames,
)


OWNERSHIP_CONTRACT = "bello_mini_resolved_link_ownership_v1"
INERTIAL_CONTRACT = "bello_mini_inertial_manifest_v1"
GEOMETRY_CONTRACT = "bello_mini_geometry_manifest_v1"
MESH_CONTRACT = "bello_mini_visual_meshes_v1"
ANKLE_PITCH_NUMERICAL_MASS_KG = 1e-6
CAD_TO_URDF = np.asarray(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

# A STEP assembly does not encode usable joint travel. These ranges are an
# explicit, provisional starting point inherited from the prior Bello model.
# They must be checked against the Mini hardware before motion is enabled.
PROVISIONAL_LIMITS_RAD = {
    "waist_yaw": (-0.5, 0.5),
    "waist_roll": (-0.5, 0.5),
    "hip_pitch": (-1.5708, 1.5708),
    "hip_roll": (-1.5708, 1.5708),
    "hip_yaw": (-1.5708, 1.5708),
    "knee": (0.0, 2.0944),
    "ankle_pitch": (-0.5236, 1.0472),
    "ankle_roll": (-0.1745, 0.1745),
    "shoulder_pitch": (-1.5, 0.5),
    "shoulder_roll": (-0.2, 2.0),
    "shoulder_yaw": (-1.5, 2.0),
    "elbow_pitch": (-1.5, 2.4),
    "elbow_yaw": (-2.0943951, 0.523598776),
    "wrist_pitch": (-1.5, 1.5),
    "wrist_roll": (-math.pi, math.pi),
}
MOTOR_RATINGS = {
    "ROBSTRIDE_00": {"peak_torque_nm": 14.0, "maximum_speed_rad_s": 33.0},
    "ROBSTRIDE_02": {"peak_torque_nm": 17.0, "maximum_speed_rad_s": 44.0},
    "ROBSTRIDE_03": {"peak_torque_nm": 60.0, "maximum_speed_rad_s": 20.0},
    "ROBSTRIDE_04": {"peak_torque_nm": 120.0, "maximum_speed_rad_s": 15.0},
    "ROBSTRIDE_06": {"peak_torque_nm": 36.0, "maximum_speed_rad_s": 50.0},
}
MOTOR_CAN_BINDINGS = {
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
FOOT_LINK_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
FOOT_CONTACT_ROLE = "foot_contact_envelope"
LINK_COLORS = {
    "center": "0.32 0.36 0.42 1",
    "left": "0.17 0.46 0.73 1",
    "right": "0.91 0.36 0.18 1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ownership", type=Path, required=True)
    parser.add_argument("--inertials", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--meshes", type=Path, required=True)
    parser.add_argument("--urdf-output", type=Path, required=True)
    parser.add_argument("--motor-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def clean_vector(values: Iterable[float], *, tolerance: float = 1e-12) -> np.ndarray:
    result = np.asarray(list(values), dtype=np.float64)
    result[np.abs(result) < tolerance] = 0.0
    return result


def format_float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError(f"non-finite robot-description value: {value}")
    if abs(value) < 1e-15:
        return "0"
    return format(value, ".17g")


def format_vector(values: Iterable[float]) -> str:
    return " ".join(format_float(float(value)) for value in values)


def add_origin(
    parent: ET.Element,
    xyz: Iterable[float],
    rpy: Iterable[float] = (0.0, 0.0, 0.0),
) -> ET.Element:
    return ET.SubElement(
        parent,
        "origin",
        {"xyz": format_vector(xyz), "rpy": format_vector(rpy)},
    )


def nominal_imu_frame() -> dict[str, Any]:
    return {
        "link_name": "imu_link",
        "joint_name": "imu_joint",
        "parent_link": "pelvis_link",
        "origin_xyz_parent_m": [0.0, 0.0, 0.0],
        "origin_rpy_parent_rad": [0.0, 0.0, 0.0],
        "rotation_parent_from_frame": np.eye(3).tolist(),
        "origin_xyz_pelvis_m": [0.0, 0.0, 0.0],
        "rotation_pelvis_from_frame": np.eye(3).tolist(),
        "frame_convention": "+X forward, +Y left, +Z up",
        "datum": (
            "nominal identity pose matching the original Bello pelvis IMU frame; "
            "the STEP contains no reviewed IMU component or mounting datum"
        ),
        "mass_policy": "massless fixed sensor frame",
    }


def fixed_sensor_frames(ownership: dict[str, Any]) -> list[dict[str, Any]]:
    return [nominal_imu_frame(), *camera_frames(ownership, CAD_TO_URDF)]


def validate_inputs(
    ownership: dict[str, Any],
    inertials: dict[str, Any],
    geometry: dict[str, Any],
    meshes: dict[str, Any],
) -> None:
    contracts = (
        (ownership, OWNERSHIP_CONTRACT, "ownership"),
        (inertials, INERTIAL_CONTRACT, "inertials"),
        (geometry, GEOMETRY_CONTRACT, "geometry"),
        (meshes, MESH_CONTRACT, "meshes"),
    )
    for manifest, contract, label in contracts:
        if manifest.get("review_contract") != contract:
            raise ValueError(f"{label} input has not passed its review/build gate")
    source_hashes = {
        manifest.get("source", {}).get("sha256") for manifest, _, _ in contracts
    }
    if None in source_hashes or len(source_hashes) != 1:
        raise ValueError("robot-description inputs refer to different STEP sources")
    ownership_links = {link["name"] for link in ownership["links"]}
    inertial_links = {link["name"] for link in inertials["links"]}
    geometry_links = {link["name"] for link in geometry["links"]}
    if ownership_links != inertial_links or ownership_links != geometry_links:
        raise ValueError("ownership, inertial, and geometry link sets differ")


def validate_source_lineage(
    paths: dict[str, Path], manifests: dict[str, dict[str, Any]]
) -> dict[str, str]:
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    expected = (
        ("inertials", "resolved_link_ownership_sha256", "ownership"),
        ("geometry", "resolved_link_ownership_sha256", "ownership"),
        ("geometry", "inertial_manifest_sha256", "inertials"),
        ("meshes", "geometry_manifest_sha256", "geometry"),
        ("meshes", "inertial_manifest_sha256", "inertials"),
    )
    for manifest_name, field, source_name in expected:
        recorded = manifests[manifest_name].get("source_manifests", {}).get(field)
        if recorded != hashes[source_name]:
            raise ValueError(
                f"{manifest_name} input has stale {field}: "
                f"expected {hashes[source_name]}, got {recorded}"
            )
    for mesh in manifests["meshes"]["meshes"]:
        path = Path(mesh["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if file_sha256(path) != mesh["sha256"]:
            raise ValueError(f"visual mesh checksum mismatch: {path}")
    return hashes


def link_side(name: str) -> str:
    if name.startswith("left_"):
        return "left"
    if name.startswith("right_"):
        return "right"
    return "center"


def inertial_values_by_link(
    inertials: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    values = {link["name"]: dict(link) for link in inertials["links"]}
    adjustments = []
    for side in ("left", "right"):
        pitch_name = f"{side}_ankle_pitch_link"
        roll_name = f"{side}_ankle_roll_link"
        roll = values[roll_name]
        original_mass = float(roll["mass_kg"])
        if original_mass <= ANKLE_PITCH_NUMERICAL_MASS_KG:
            raise ValueError(f"{roll_name} cannot supply numerical ankle mass")
        fraction = ANKLE_PITCH_NUMERICAL_MASS_KG / original_mass
        original_inertia = roll["inertia_urdf_link_com_kg_m2"]
        pitch = values[pitch_name]
        pitch["mass_kg"] = ANKLE_PITCH_NUMERICAL_MASS_KG
        pitch["center_urdf_link_m"] = list(roll["center_urdf_link_m"])
        pitch["inertia_urdf_link_com_kg_m2"] = {
            key: float(value) * fraction for key, value in original_inertia.items()
        }
        roll["mass_kg"] = original_mass - ANKLE_PITCH_NUMERICAL_MASS_KG
        roll["inertia_urdf_link_com_kg_m2"] = {
            key: float(value) * (1.0 - fraction)
            for key, value in original_inertia.items()
        }
        adjustments.append(
            {
                "source_link": roll_name,
                "target_link": pitch_name,
                "transferred_mass_kg": ANKLE_PITCH_NUMERICAL_MASS_KG,
                "reason": "MuJoCo requires positive mass/inertia on serial moving bodies",
                "zero_pose_combined_mass_properties_preserved": True,
            }
        )
    return values, adjustments


def add_inertial(parent: ET.Element, values: dict[str, Any]) -> None:
    mass = float(values["mass_kg"])
    inertia_values = values["inertia_urdf_link_com_kg_m2"]
    if mass <= 0.0 or inertia_values is None:
        raise ValueError(f"dynamic link {values['name']} lacks positive inertial data")
    matrix = np.asarray(
        [
            [inertia_values["ixx"], inertia_values["ixy"], inertia_values["ixz"]],
            [inertia_values["ixy"], inertia_values["iyy"], inertia_values["iyz"]],
            [inertia_values["ixz"], inertia_values["iyz"], inertia_values["izz"]],
        ],
        dtype=np.float64,
    )
    if np.min(np.linalg.eigvalsh(matrix)) <= 0.0:
        raise ValueError(f"link {values['name']} inertia is not positive definite")
    inertial = ET.SubElement(parent, "inertial")
    add_origin(inertial, values["center_urdf_link_m"])
    ET.SubElement(inertial, "mass", {"value": format_float(mass)})
    ET.SubElement(
        inertial,
        "inertia",
        {key: format_float(float(value)) for key, value in inertia_values.items()},
    )


def add_primitive_geometry(parent: ET.Element, primitive: dict[str, Any]) -> None:
    geometry = ET.SubElement(parent, "geometry")
    kind = primitive["type"]
    if kind == "box":
        ET.SubElement(geometry, "box", {"size": format_vector(primitive["size_m"])})
    elif kind == "cylinder":
        ET.SubElement(
            geometry,
            "cylinder",
            {
                "radius": format_float(float(primitive["radius_m"])),
                "length": format_float(float(primitive["length_m"])),
            },
        )
    elif kind == "sphere":
        ET.SubElement(
            geometry,
            "sphere",
            {"radius": format_float(float(primitive["radius_m"]))},
        )
    else:
        raise ValueError(f"unsupported primitive collision type: {kind}")


def mesh_reference(urdf_path: Path, mesh_path: Path) -> str:
    return (
        Path("..")
        .joinpath(
            Path(mesh_path).resolve().relative_to(urdf_path.parent.parent.resolve())
        )
        .as_posix()
    )


def add_link_geometry(
    link: ET.Element,
    *,
    name: str,
    geometry: dict[str, Any],
    mesh: dict[str, Any] | None,
    urdf_path: Path,
) -> None:
    if mesh is not None:
        visual = ET.SubElement(link, "visual", {"name": f"{name}_visual"})
        add_origin(visual, (0.0, 0.0, 0.0))
        visual_geometry = ET.SubElement(visual, "geometry")
        ET.SubElement(
            visual_geometry,
            "mesh",
            {
                "filename": mesh_reference(urdf_path, Path(mesh["path"])),
                "scale": "0.001 0.001 0.001",
            },
        )
        ET.SubElement(visual, "material", {"name": f"{link_side(name)}_links"})
    for primitive_value in geometry["collision_primitives"]:
        collision = ET.SubElement(link, "collision", {"name": primitive_value["name"]})
        add_origin(
            collision,
            primitive_value["origin_xyz_m"],
            primitive_value["origin_rpy_rad"],
        )
        add_primitive_geometry(collision, primitive_value)


def add_link(
    root: ET.Element,
    *,
    name: str,
    inertial: dict[str, Any],
    geometry: dict[str, Any],
    mesh: dict[str, Any] | None,
    urdf_path: Path,
) -> None:
    link = ET.SubElement(root, "link", {"name": name})
    add_inertial(link, inertial)
    add_link_geometry(
        link,
        name=name,
        geometry=geometry,
        mesh=mesh,
        urdf_path=urdf_path,
    )


def joint_limit(
    joint: dict[str, Any], motors: dict[str, dict[str, Any]]
) -> dict[str, str]:
    lower, upper = PROVISIONAL_LIMITS_RAD[joint["station"]]
    models = {motors[motor_id]["model"] for motor_id in joint["motor_ids"]}
    if len(models) != 1:
        raise ValueError(f"joint {joint['name']} has inconsistent motor models")
    rating = MOTOR_RATINGS[models.pop()]
    return {
        "lower": format_float(lower),
        "upper": format_float(upper),
        "effort": format_float(rating["peak_torque_nm"]),
        "velocity": format_float(rating["maximum_speed_rad_s"]),
    }


def add_joint(
    root: ET.Element,
    joint: dict[str, Any],
    frame_origins: dict[str, np.ndarray],
    motors: dict[str, dict[str, Any]],
) -> None:
    element = ET.SubElement(root, "joint", {"name": joint["name"], "type": "revolute"})
    ET.SubElement(element, "parent", {"link": joint["parent_link"]})
    ET.SubElement(element, "child", {"link": joint["child_link"]})
    relative_origin = (
        frame_origins[joint["child_link"]] - frame_origins[joint["parent_link"]]
    )
    add_origin(element, clean_vector(relative_origin))
    axis = clean_vector(CAD_TO_URDF @ np.asarray(joint["axis_cad"], dtype=np.float64))
    ET.SubElement(element, "axis", {"xyz": format_vector(axis)})
    ET.SubElement(element, "limit", joint_limit(joint, motors))


def add_anonymous_endpoint(
    root: ET.Element,
    endpoint: dict[str, Any],
) -> None:
    link = ET.SubElement(root, "link", {"name": endpoint["link_name"]})
    visual = ET.SubElement(link, "visual", {"name": f"{endpoint['link_name']}_visual"})
    add_origin(visual, endpoint["sphere_center_xyz_link_m"])
    add_primitive_geometry(visual, {"type": "sphere", **endpoint})
    ET.SubElement(visual, "material", {"name": "anonymous_end_effector"})
    collision = ET.SubElement(
        link, "collision", {"name": f"{endpoint['link_name']}_collision"}
    )
    add_origin(collision, endpoint["sphere_center_xyz_link_m"])
    add_primitive_geometry(collision, {"type": "sphere", **endpoint})
    joint = ET.SubElement(
        root, "joint", {"name": endpoint["joint_name"], "type": "fixed"}
    )
    ET.SubElement(joint, "parent", {"link": endpoint["parent_link"]})
    ET.SubElement(joint, "child", {"link": endpoint["link_name"]})
    add_origin(joint, endpoint["origin_xyz_parent_m"])


def add_fixed_sensor_frame(
    root: ET.Element,
    frame: dict[str, Any],
    *,
    geometry: dict[str, Any] | None,
    mesh: dict[str, Any] | None,
    urdf_path: Path,
) -> None:
    link = ET.SubElement(root, "link", {"name": frame["link_name"]})
    if geometry is not None:
        add_link_geometry(
            link,
            name=frame["link_name"],
            geometry=geometry,
            mesh=mesh,
            urdf_path=urdf_path,
        )
    joint = ET.SubElement(
        root,
        "joint",
        {"name": frame["joint_name"], "type": "fixed"},
    )
    ET.SubElement(joint, "parent", {"link": frame["parent_link"]})
    ET.SubElement(joint, "child", {"link": frame["link_name"]})
    add_origin(
        joint,
        frame["origin_xyz_parent_m"],
        frame["origin_rpy_parent_rad"],
    )


def nominal_grounding(
    inertials: dict[str, Any], geometry: dict[str, Any]
) -> dict[str, Any]:
    inertial_by_link = {link["name"]: link for link in inertials["links"]}
    geometry_by_link = {link["name"]: link for link in geometry["links"]}
    sole_heights = {}
    for link_name in FOOT_LINK_NAMES:
        frame_z = float(inertial_by_link[link_name]["frame_origin_urdf_world_m"][2])
        primitives = [
            primitive
            for primitive in geometry_by_link[link_name]["collision_primitives"]
            if primitive["role"] == FOOT_CONTACT_ROLE
        ]
        if not primitives:
            raise ValueError(f"{link_name} has no foot contact primitives")
        bottoms = []
        for primitive in primitives:
            if any(abs(float(value)) > 1e-12 for value in primitive["origin_rpy_rad"]):
                raise ValueError(
                    f"{primitive['name']} must be horizontal for nominal grounding"
                )
            if primitive["type"] == "box":
                half_height = float(primitive["size_m"][2]) * 0.5
            elif primitive["type"] == "cylinder":
                half_height = float(primitive["length_m"]) * 0.5
            else:
                raise ValueError(
                    f"unsupported foot contact primitive {primitive['type']}"
                )
            bottoms.append(frame_z + float(primitive["origin_xyz_m"][2]) - half_height)
        if max(bottoms) - min(bottoms) > 1e-12:
            raise ValueError(f"{link_name} contact primitives are not coplanar")
        sole_heights[link_name] = min(bottoms)
    if max(sole_heights.values()) - min(sole_heights.values()) > 1e-12:
        raise ValueError(f"left and right sole planes differ: {sole_heights}")
    source_height = min(sole_heights.values())
    translation_z = -source_height
    return {
        "pose": "all movable joints at zero",
        "support_surface_z_m": 0.0,
        "urdf_root_link": "pelvis_link",
        "sole_surface_z_in_urdf_root_frame_m": source_height,
        "nominal_root_translation_m": [0.0, 0.0, translation_z],
        "grounded_sole_surface_z_m": {
            link_name: height + translation_z
            for link_name, height in sole_heights.items()
        },
        "integration_policy": (
            "preserve the pelvis URDF root and apply this translation as the "
            "robot definition default root position"
        ),
    }


def build_urdf(
    ownership: dict[str, Any],
    inertials: dict[str, Any],
    geometry: dict[str, Any],
    meshes: dict[str, Any],
    *,
    urdf_path: Path,
) -> tuple[ET.ElementTree, list[dict[str, Any]], list[dict[str, Any]]]:
    root = ET.Element("robot", {"name": "bello_mini"})
    root.append(
        ET.Comment(
            "Joint ranges are provisional and must be hardware-validated before motion."
        )
    )
    root.append(
        ET.Comment(
            "Astra Stereo S U3 sensor origins are nominal mechanical values: "
            "IR/depth is 2 mm behind the CAD front face; RGB is 2 mm forward "
            "of IR/depth. They are not factory-calibrated extrinsics."
        )
    )
    root.append(
        ET.Comment(
            "Publish camera sensor TFs from this URDF or from the Orbbec driver, "
            "never from both sources simultaneously."
        )
    )
    root.append(
        ET.Comment(
            "imu_link is a nominal identity child of pelvis_link, matching the "
            "original Bello pelvis-frame IMU convention; no IMU CAD datum was found."
        )
    )
    mujoco_extension = ET.SubElement(root, "mujoco")
    ET.SubElement(
        mujoco_extension,
        "compiler",
        {"fusestatic": "false", "inertiafromgeom": "false"},
    )
    for name, rgba in (
        ("center_links", LINK_COLORS["center"]),
        ("left_links", LINK_COLORS["left"]),
        ("right_links", LINK_COLORS["right"]),
        ("anonymous_end_effector", "0.85 0.85 0.88 1"),
    ):
        material = ET.SubElement(root, "material", {"name": name})
        ET.SubElement(material, "color", {"rgba": rgba})

    inertial_by_link, adjustments = inertial_values_by_link(inertials)
    geometry_by_link = {link["name"]: link for link in geometry["links"]}
    mesh_by_link = {mesh["link"]: mesh for mesh in meshes["meshes"]}
    for link in ownership["links"]:
        name = link["name"]
        add_link(
            root,
            name=name,
            inertial=inertial_by_link[name],
            geometry=geometry_by_link[name],
            mesh=mesh_by_link.get(name),
            urdf_path=urdf_path,
        )

    frame_origins = {
        name: CAD_TO_URDF
        @ np.asarray(values["frame_origin_cad_world_m"], dtype=np.float64)
        for name, values in inertial_by_link.items()
    }
    motors = {motor["id"]: motor for motor in ownership["motors"]}
    for joint in ownership["joints"]:
        add_joint(root, joint, frame_origins, motors)
    sensor_frames = fixed_sensor_frames(ownership)
    fixed_geometry_by_link = {
        fixed["name"]: fixed for fixed in geometry.get("fixed_geometry", [])
    }
    for frame in sensor_frames:
        fixed_geometry = fixed_geometry_by_link.get(frame["link_name"])
        add_fixed_sensor_frame(
            root,
            frame,
            geometry=fixed_geometry,
            mesh=mesh_by_link.get(frame["link_name"]),
            urdf_path=urdf_path,
        )
    for endpoint in geometry["anonymous_end_effectors"]:
        add_anonymous_endpoint(root, endpoint)
    return ET.ElementTree(root), adjustments, sensor_frames


def validate_motor_can_bindings(motors: list[dict[str, Any]]) -> None:
    motor_names = {motor["motor_name"] for motor in motors}
    binding_names = set(MOTOR_CAN_BINDINGS)
    if motor_names != binding_names:
        missing = sorted(motor_names - binding_names)
        extra = sorted(binding_names - motor_names)
        raise ValueError(
            f"motor CAN binding mismatch: missing={missing}, unexpected={extra}"
        )
    addresses = list(MOTOR_CAN_BINDINGS.values())
    if len(addresses) != len(set(addresses)):
        raise ValueError("motor CAN bindings contain duplicate bus/ID addresses")
    for motor_name, (bus, can_id) in MOTOR_CAN_BINDINGS.items():
        if bus not in {"can0", "can1"} or not 0 <= can_id <= 127:
            raise ValueError(f"invalid CAN binding for {motor_name}: {bus}/{can_id}")


def cad_direction_inference(
    motor: dict[str, Any],
    output_axis: np.ndarray,
    joint_axes: dict[str, list[float]],
) -> dict[str, Any]:
    reference_joint = motor["joint_names"][0]
    axis_dot = float(np.dot(output_axis, np.asarray(joint_axes[reference_joint])))
    if not math.isclose(abs(axis_dot), 1.0, abs_tol=1e-6):
        raise ValueError(
            f"motor {motor['motor_name']} CAD axis is not parallel to its "
            f"reference joint {reference_joint}: dot={axis_dot}"
        )
    alignment_sign = 1 if axis_dot > 0.0 else -1
    differential = motor["coupling"] == "differential"
    return {
        "reference_joint": reference_joint,
        "axis_dot": float(alignment_sign),
        "axis_alignment_sign": alignment_sign,
        "candidate_command_direction": None if differential else alignment_sign,
        "status": (
            "cad_pitch_projection_only_differential_unresolved"
            if differential
            else "cad_geometry_only_protocol_sign_unverified"
        ),
    }


def build_motor_topology(
    ownership: dict[str, Any], inertials: dict[str, Any]
) -> dict[str, Any]:
    joint_by_name = {joint["name"]: joint for joint in ownership["joints"]}
    catalog = inertials["motor_catalog"]
    validate_motor_can_bindings(ownership["motors"])
    motors = []
    for motor in ownership["motors"]:
        output_axis = clean_vector(
            CAD_TO_URDF @ np.asarray(motor["output_axis_cad"], dtype=np.float64)
        )
        joint_axes = {
            joint_name: clean_vector(
                CAD_TO_URDF
                @ np.asarray(joint_by_name[joint_name]["axis_cad"], dtype=np.float64)
            ).tolist()
            for joint_name in motor["joint_names"]
        }
        bus, can_id = MOTOR_CAN_BINDINGS[motor["motor_name"]]
        direction_inference = cad_direction_inference(motor, output_axis, joint_axes)
        center_urdf_world_m = clean_vector(
            CAD_TO_URDF @ np.asarray(motor["center_mm"], dtype=np.float64) / 1000.0
        )
        motors.append(
            {
                "name": motor["motor_name"],
                "cad_id": motor["id"],
                "model": motor["model"],
                "side": motor["side"],
                "station": motor["station"],
                "coupling": motor["coupling"],
                "logical_joints": motor["joint_names"],
                "cad_center_urdf_world_m": center_urdf_world_m.tolist(),
                "cad_motor_frame_output_axis_urdf": output_axis.tolist(),
                "logical_joint_axes_urdf": joint_axes,
                "cad_direction_inference": direction_inference,
                "hardware_binding": {
                    "bus": bus,
                    "can_id": can_id,
                    "command_direction": None,
                    "encoder_zero_offset_rad": None,
                    "status": "can_assigned_direction_calibration_required",
                },
            }
        )
    differentials = []
    for side in ("left", "right"):
        names = [
            motor["motor_name"]
            for motor in ownership["motors"]
            if motor["side"] == side and motor["coupling"] == "differential"
        ]
        differentials.append(
            {
                "name": f"{side}_ankle_differential",
                "motors": names,
                "logical_joints": [
                    f"{side}_ankle_pitch_joint",
                    f"{side}_ankle_roll_joint",
                ],
                "motor_from_joint_coefficients": None,
                "status": "calibration_required",
            }
        )
    return {
        "schema_version": 2,
        "robot": "bello_mini",
        "deployment_status": "calibration_required",
        "deployment_blockers": [
            "RobStride command directions require raw encoder/low-current bench validation",
            "encoder zero positions are uncalibrated",
            "ankle differential coefficients/signs are unvalidated",
            "joint travel limits are provisional and require physical stop checks",
        ],
        "can_topology": {
            "logical_buses": {
                "can0": "left and right legs",
                "can1": "waist and left/right arms",
            },
            "bitrate_bps": 1_000_000,
            "id_policy": (
                "original Bello body ranges with IDs increasing along each kinematic "
                "chain; waist yaw/roll use 51/52"
            ),
        },
        "direction_inference_policy": {
            "cad_positive_axis": (
                "motor occurrence local +Z transformed into the URDF frame"
            ),
            "axis_alignment_sign": (
                "sign(dot(CAD positive motor axis, reviewed logical joint positive axis))"
            ),
            "candidate_command_direction": (
                "direct-drive geometric placeholder only, conditional on RobStride raw "
                "positive matching CAD local +Z"
            ),
            "protocol_limitation": (
                "RobStride documentation does not bind raw-positive rotation to the STEP "
                "frame; hardware command_direction remains null until bench validation"
            ),
            "differential_limitation": (
                "ankle alignment is a pitch-axis projection, not a standalone motor "
                "direction or calibrated differential row"
            ),
        },
        "motor_model_catalog": {
            "source": catalog["source"],
            "models": {
                name: {**values, **MOTOR_RATINGS[name]}
                for name, values in catalog["models"].items()
            },
        },
        "motors": motors,
        "differential_groups": differentials,
    }


def build_report(
    ownership: dict[str, Any],
    inertials: dict[str, Any],
    geometry: dict[str, Any],
    meshes: dict[str, Any],
    adjustments: list[dict[str, Any]],
    sensor_frames: list[dict[str, Any]],
    motor_topology: dict[str, Any],
    *,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "robot": "bello_mini",
        "source": ownership["source"],
        "source_manifest_sha256": source_hashes,
        "kinematics": {
            "root_link": "pelvis_link",
            "movable_joint_count": len(ownership["joints"]),
            "fixed_anonymous_endpoint_count": len(geometry["anonymous_end_effectors"]),
            "fixed_sensor_frame_count": len(sensor_frames),
            "joint_limit_status": "provisional_hardware_validation_required",
        },
        "fixed_sensor_frames": sensor_frames,
        "astra_stereo_s_u3": camera_cad_report(),
        "grounding": nominal_grounding(inertials, geometry),
        "mass": inertials["summary"],
        "component_catalog": inertials["component_catalog"],
        "catalog_components": [
            {
                key: group[key]
                for key in (
                    "id",
                    "kind",
                    "link",
                    "catalog_model",
                    "catalog_dimensions_m",
                    "catalog_mass_tolerance_kg",
                    "catalog_source",
                    "center_cad_world_m",
                    "mass_kg",
                )
            }
            for group in inertials["groups"]
            if group["kind"] == "catalog_component"
        ],
        "visual_geometry": {
            **meshes["summary"],
            "policy": geometry["visual_policy"],
            "excluded_visual_instance_counts": geometry["summary"][
                "excluded_visual_instance_counts"
            ],
            "visually_omitted_structure_mass_kg": geometry["summary"][
                "visually_omitted_structure_mass_kg"
            ],
        },
        "collision_geometry": {
            "primitive_count": geometry["summary"]["collision_primitive_count"],
            "role_counts": geometry["summary"]["collision_primitive_role_counts"],
            "policy": geometry["collision_policy"],
        },
        "anonymous_end_effectors": geometry["anonymous_end_effectors"],
        "numerical_inertial_adjustments": adjustments,
        "motor_configuration_status": "calibration_required",
        "motor_topology": {
            key: motor_topology[key]
            for key in (
                "deployment_status",
                "deployment_blockers",
                "can_topology",
                "direction_inference_policy",
                "motors",
                "differential_groups",
            )
        },
    }


def main() -> None:
    args = parse_args()
    paths = {
        "ownership": args.ownership.expanduser().resolve(),
        "inertials": args.inertials.expanduser().resolve(),
        "geometry": args.geometry.expanduser().resolve(),
        "meshes": args.meshes.expanduser().resolve(),
    }
    manifests = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    validate_inputs(**manifests)
    source_hashes = validate_source_lineage(paths, manifests)
    urdf_path = args.urdf_output.expanduser().resolve()
    tree, adjustments, sensor_frames = build_urdf(**manifests, urdf_path=urdf_path)
    ET.indent(tree, space="  ")
    xml = ET.tostring(tree.getroot(), encoding="unicode", xml_declaration=True) + "\n"
    atomic_write_text(urdf_path, xml)

    topology = build_motor_topology(manifests["ownership"], manifests["inertials"])
    atomic_write_text(
        args.motor_output.expanduser().resolve(),
        yaml.safe_dump(topology, sort_keys=False),
    )
    report = build_report(
        **manifests,
        adjustments=adjustments,
        sensor_frames=sensor_frames,
        motor_topology=topology,
        source_hashes=source_hashes,
    )
    atomic_write_text(
        args.report_output.expanduser().resolve(),
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(f"wrote URDF: {urdf_path}")
    print(f"wrote motor topology: {args.motor_output}")
    print(f"wrote build report: {args.report_output}")
    print(
        json.dumps(
            {
                "mass_kg": report["mass"]["total_mass_kg"],
                **manifests["meshes"]["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
