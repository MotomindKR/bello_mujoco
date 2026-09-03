"""Select minimal Bello Mini visuals and fit primitive collision geometry."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np

from bello_mujoco.robot_description.astra_stereo_s_u3 import (
    ASTRA_MOUNT_FRAME,
    CAMERA_CATALOG_MODEL,
    camera_collision_primitive,
    find_camera_placement,
    mount_pose,
)


SCHEMA_VERSION = 1
OWNERSHIP_CONTRACT = "bello_mini_resolved_link_ownership_v1"
INERTIAL_CONTRACT = "bello_mini_inertial_manifest_v1"
OUTPUT_CONTRACT = "bello_mini_geometry_manifest_v1"
ANONYMOUS_SPHERE_RADIUS_M = 0.025
FOOT_SOLE_BOTTOM_M = -0.046
FOOT_SOLE_TOP_M = -0.034
FOOT_SOLE_BOX_BOUNDS_M = (
    ("heel_center", -0.070, -0.050, -0.025, 0.025),
    ("main", -0.050, 0.114, -0.045, 0.045),
)
FOOT_HEEL_CYLINDER_CENTERS_M = ((-0.050, -0.025), (-0.050, 0.025))
FOOT_HEEL_RADIUS_M = 0.020
FOOT_TOE_CENTER_X_M = 0.105
FOOT_TOE_RADIUS_M = 0.045
FOOT_ANKLE_HOUSING_RADIUS_M = 0.035
FOOT_ANKLE_HOUSING_LENGTH_M = 0.080
CAD_TO_URDF = np.asarray(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
FASTENER_OR_SPACER = re.compile(
    r"(screw|bolt|washer|\bnut\b|fastener|hshcs|circlip|retaining|spacer|shim|"
    r"너트|와셔)",
    re.IGNORECASE,
)
SMALL_BEARING = re.compile(r"(^|[_\s])bearing($|[_\s.\d])|베어링|부싱", re.IGNORECASE)
SEGMENT_CHILD_JOINT = {
    "left_hip_pitch_link": "left_hip_roll_joint",
    "left_hip_roll_link": "left_hip_yaw_joint",
    "left_hip_yaw_link": "left_knee_joint",
    "left_knee_link": "left_ankle_pitch_joint",
    "right_hip_pitch_link": "right_hip_roll_joint",
    "right_hip_roll_link": "right_hip_yaw_joint",
    "right_hip_yaw_link": "right_knee_joint",
    "right_knee_link": "right_ankle_pitch_joint",
    "left_shoulder_pitch_link": "left_shoulder_roll_joint",
    "left_shoulder_roll_link": "left_shoulder_yaw_joint",
    "left_upper_arm_link": "left_elbow_pitch_joint",
    "left_elbow_link": "left_wrist_yaw_joint",
    "left_lower_arm_link": "left_wrist_pitch_joint",
    "left_wrist_link": "left_wrist_roll_joint",
    "right_shoulder_pitch_link": "right_shoulder_roll_joint",
    "right_shoulder_roll_link": "right_shoulder_yaw_joint",
    "right_upper_arm_link": "right_elbow_pitch_joint",
    "right_elbow_link": "right_wrist_yaw_joint",
    "right_lower_arm_link": "right_wrist_pitch_joint",
    "right_wrist_link": "right_wrist_roll_joint",
}
SEGMENT_RADIUS_M = {
    "hip_pitch_link": 0.060,
    "hip_roll_link": 0.060,
    "hip_yaw_link": 0.065,
    "knee_link": 0.060,
    "shoulder_pitch_link": 0.055,
    "shoulder_roll_link": 0.055,
    "upper_arm_link": 0.050,
    "elbow_link": 0.045,
    "lower_arm_link": 0.040,
    "wrist_link": 0.040,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ownership", type=Path, required=True)
    parser.add_argument("--inertials", type=Path, required=True)
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def visual_occurrence_decision(occurrence: dict[str, Any]) -> tuple[bool, str]:
    label = f"{occurrence['source_name']} {occurrence['source_label']}"
    bounds = occurrence["bounds_mm"]
    maximum_size = max(
        maximum - minimum
        for minimum, maximum in zip(bounds["min"], bounds["max"], strict=True)
    )
    mass = float(occurrence["estimated_mass_kg"] or 0.0)
    if FASTENER_OR_SPACER.search(label):
        return False, "internal fastener, washer, or spacer"
    if SMALL_BEARING.search(label) and mass < 0.05:
        return False, "small internal bearing or bearing axis"
    if mass < 0.001 and maximum_size < 30.0:
        return False, "sub-gram CAD detail below visual threshold"
    return True, "reviewed structural exterior candidate"


def group_bounds(groups: Iterable[dict[str, Any]]) -> dict[str, list[float]]:
    values = list(groups)
    return {
        "min": [
            min(value["bounds_mm"]["min"][axis] for value in values)
            for axis in range(3)
        ],
        "max": [
            max(value["bounds_mm"]["max"][axis] for value in values)
            for axis in range(3)
        ],
    }


def vector_rpy_from_z(vector: np.ndarray) -> list[float]:
    direction = vector / np.linalg.norm(vector)
    reference = np.asarray([1.0, 0.0, 0.0])
    if abs(float(reference @ direction)) > 0.9:
        reference = np.asarray([0.0, 1.0, 0.0])
    y_axis = np.cross(direction, reference)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, direction)
    rotation = np.column_stack((x_axis, y_axis, direction))
    pitch = math.asin(max(-1.0, min(1.0, -float(rotation[2, 0]))))
    if abs(math.cos(pitch)) > 1e-9:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = 0.0
        yaw = math.atan2(-rotation[0, 1], rotation[1, 1])
    return [roll, pitch, yaw]


def primitive(
    *,
    name: str,
    kind: str,
    role: str,
    position: Iterable[float],
    rpy: Iterable[float] = (0.0, 0.0, 0.0),
    size: Iterable[float] | None = None,
    radius: float | None = None,
    length: float | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "type": kind,
        "role": role,
        "origin_xyz_m": [float(value) for value in position],
        "origin_rpy_rad": [float(value) for value in rpy],
        **({"size_m": [float(value) for value in size]} if size is not None else {}),
        **({"radius_m": float(radius)} if radius is not None else {}),
        **({"length_m": float(length)} if length is not None else {}),
    }


def local_urdf_aabb(
    bounds: dict[str, list[float]], origin_cad_mm: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    center_cad = (np.asarray(bounds["min"]) + np.asarray(bounds["max"])) * 0.5
    size_cad = np.asarray(bounds["max"]) - np.asarray(bounds["min"])
    center = CAD_TO_URDF @ ((center_cad - origin_cad_mm) * 1e-3)
    size = CAD_TO_URDF @ (size_cad * 1e-3)
    return center, size


def foot_collision_primitives(name: str) -> list[dict[str, Any]]:
    sole_center_z = (FOOT_SOLE_BOTTOM_M + FOOT_SOLE_TOP_M) * 0.5
    sole_thickness = FOOT_SOLE_TOP_M - FOOT_SOLE_BOTTOM_M
    boxes = [
        primitive(
            name=f"{name}_sole_{part}",
            kind="box",
            role="foot_contact_envelope",
            position=((minimum_x + maximum_x) * 0.5, 0.0, sole_center_z),
            size=(
                maximum_x - minimum_x,
                maximum_y - minimum_y,
                sole_thickness,
            ),
        )
        for part, minimum_x, maximum_x, minimum_y, maximum_y in (
            FOOT_SOLE_BOX_BOUNDS_M
        )
    ]
    heel_cylinders = [
        primitive(
            name=f"{name}_sole_heel_{side}",
            kind="cylinder",
            role="foot_contact_envelope",
            position=(center_x, center_y, sole_center_z),
            radius=FOOT_HEEL_RADIUS_M,
            length=sole_thickness,
        )
        for side, (center_x, center_y) in zip(
            ("negative_y", "positive_y"),
            FOOT_HEEL_CYLINDER_CENTERS_M,
            strict=True,
        )
    ]
    return [
        *boxes,
        *heel_cylinders,
        primitive(
            name=f"{name}_sole_toe",
            kind="cylinder",
            role="foot_contact_envelope",
            position=(FOOT_TOE_CENTER_X_M, 0.0, sole_center_z),
            radius=FOOT_TOE_RADIUS_M,
            length=sole_thickness,
        ),
        primitive(
            name=f"{name}_ankle_housing",
            kind="cylinder",
            role="ankle_joint_envelope",
            position=(0.0, 0.0, 0.0),
            rpy=(0.0, math.pi * 0.5, 0.0),
            radius=FOOT_ANKLE_HOUSING_RADIUS_M,
            length=FOOT_ANKLE_HOUSING_LENGTH_M,
        ),
    ]


def collision_primitives(
    ownership: dict[str, Any], link_bounds: dict[str, dict[str, list[float]]]
) -> dict[str, list[dict[str, Any]]]:
    joints = {joint["name"]: joint for joint in ownership["joints"]}
    origins = {
        "pelvis_link": np.zeros(3),
        **{
            joint["child_link"]: np.asarray(joint["origin_cad_mm"])
            for joint in ownership["joints"]
        },
    }
    result: dict[str, list[dict[str, Any]]] = {}
    for link in ownership["links"]:
        name = link["name"]
        if link["physical_geometry_policy"] == "virtual_massless_serial_coordinate":
            result[name] = []
            continue
        origin = origins[name]
        if name in SEGMENT_CHILD_JOINT:
            child_origin = np.asarray(
                joints[SEGMENT_CHILD_JOINT[name]]["origin_cad_mm"]
            )
            vector = CAD_TO_URDF @ ((child_origin - origin) * 1e-3)
            length = float(np.linalg.norm(vector))
            suffix = next(
                suffix for suffix in SEGMENT_RADIUS_M if name.endswith(suffix)
            )
            radius = SEGMENT_RADIUS_M[suffix]
            result[name] = [
                primitive(
                    name=f"{name}_segment",
                    kind="cylinder",
                    role="link_segment",
                    position=vector * 0.5,
                    rpy=vector_rpy_from_z(vector),
                    radius=radius,
                    length=length,
                ),
                primitive(
                    name=f"{name}_joint_envelope",
                    kind="sphere",
                    role="proximal_joint_envelope",
                    position=(0.0, 0.0, 0.0),
                    radius=radius,
                ),
            ]
            continue

        center, size = local_urdf_aabb(link_bounds[name], origin)
        if name in {"pelvis_link", "torso_link"}:
            scale = np.asarray([0.86, 0.90, 0.88])
            result[name] = [
                primitive(
                    name=f"{name}_box",
                    kind="box",
                    role="central_body_envelope",
                    position=center,
                    size=size * scale,
                )
            ]
        elif name == "waist_yaw_link":
            result[name] = [
                primitive(
                    name="waist_yaw_link_cylinder",
                    kind="cylinder",
                    role="waist_envelope",
                    position=center,
                    radius=0.45 * max(float(size[0]), float(size[1])),
                    length=0.88 * float(size[2]),
                )
            ]
        elif name.endswith("ankle_roll_link"):
            result[name] = foot_collision_primitives(name)
        elif name.endswith("wrist_roll_link"):
            result[name] = [
                primitive(
                    name=f"{name}_sphere",
                    kind="sphere",
                    role="terminal_wrist_envelope",
                    position=center,
                    radius=0.045,
                )
            ]
        else:
            raise ValueError(f"no collision primitive policy for {name}")
    return result


def anonymous_end_effectors(
    classification: dict[str, Any], ownership: dict[str, Any]
) -> list[dict[str, Any]]:
    wrist_joints = {
        joint["side"]: joint
        for joint in ownership["joints"]
        if joint["station"] == "wrist_roll"
    }
    result = []
    for source in classification["end_effector_policy"]["source_assemblies"]:
        bounds = source["bounds_mm"]
        center_cad = (np.asarray(bounds["min"]) + np.asarray(bounds["max"])) * 0.5
        side = "left" if center_cad[0] > 0.0 else "right"
        wrist = wrist_joints[side]
        wrist_origin = np.asarray(wrist["origin_cad_mm"])
        mount_cad = wrist_origin.copy()
        mount_cad[1] = float(bounds["max"][1])
        mount_offset = CAD_TO_URDF @ ((mount_cad - wrist_origin) * 1e-3)
        result.append(
            {
                "side": side,
                "link_name": f"{side}_end_effector_sphere_link",
                "joint_name": f"{side}_end_effector_sphere_joint",
                "parent_link": wrist["child_link"],
                "origin_xyz_parent_m": mount_offset.tolist(),
                "sphere_center_xyz_link_m": [
                    0.0,
                    0.0,
                    -ANONYMOUS_SPHERE_RADIUS_M,
                ],
                "radius_m": ANONYMOUS_SPHERE_RADIUS_M,
                "mass_kg": 0.0,
                "source_hand_id": source["id"],
                "placement_policy": (
                    "reviewed wrist centerline at excluded CAD hand proximal plane"
                ),
            }
        )
    return sorted(result, key=lambda value: value["side"])


def build_manifest(
    ownership: dict[str, Any],
    inertials: dict[str, Any],
    classification: dict[str, Any],
    *,
    ownership_sha256: str,
    inertials_sha256: str,
    classification_sha256: str,
) -> dict[str, Any]:
    if ownership.get("review_contract") != OWNERSHIP_CONTRACT:
        raise ValueError("ownership input has not passed its resolved gate")
    if inertials.get("review_contract") != INERTIAL_CONTRACT:
        raise ValueError("inertial input has not passed its build gate")
    source_hashes = {
        ownership.get("source", {}).get("sha256"),
        inertials.get("source", {}).get("sha256"),
        classification.get("source", {}).get("sha256"),
    }
    if None in source_hashes or len(source_hashes) != 1:
        raise ValueError("geometry inputs refer to different STEP sources")

    visual_instances: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fixed_visual_instances: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    excluded_structure_mass = 0.0
    groups_by_link: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in ownership["groups"]:
        groups_by_link[group["reviewed_link"]].append(group)
        if group["kind"] == "measured_assembly":
            exclusions["measured battery/electronics internals"] += group[
                "occurrence_count"
            ]
            continue
        if group["kind"] == "catalog_motor":
            instances = group.get("selection_instances")
            if not instances:
                raise ValueError(f"catalog motor {group['id']} lacks envelope geometry")
            for instance in instances:
                visual_instances[group["reviewed_link"]].append(
                    {
                        **instance,
                        "group_id": group["id"],
                        "source_kind": "catalog_motor_envelope",
                    }
                )
            continue
        source_kind = (
            "catalog_component"
            if group["kind"] == "catalog_component"
            else "structure"
        )
        for occurrence in group["occurrences"]:
            included, reason = visual_occurrence_decision(occurrence)
            if not included:
                exclusions[reason] += 1
                excluded_structure_mass += float(occurrence["estimated_mass_kg"] or 0.0)
                continue
            instance = {
                "mesh": occurrence["mesh"],
                "transform": occurrence["transform"],
                "group_id": group["id"],
                "source_kind": source_kind,
                "source_name": occurrence["source_name"],
            }
            if group.get("catalog_model") == CAMERA_CATALOG_MODEL:
                fixed_visual_instances.append(instance)
            else:
                visual_instances[group["reviewed_link"]].append(instance)

    camera_placement = find_camera_placement(ownership)
    if not any(
        instance["source_name"] == camera_placement.occurrence["source_name"]
        for instance in fixed_visual_instances
    ):
        raise ValueError("camera body was removed by the visual selection policy")
    mount_origin_urdf_m, mount_rotation = mount_pose(ownership, CAD_TO_URDF)
    fixed_geometry = [
        {
            "name": ASTRA_MOUNT_FRAME,
            "frame_origin_urdf_world_m": mount_origin_urdf_m.tolist(),
            "rotation_urdf_world_from_frame": mount_rotation.tolist(),
            "visual_instances": fixed_visual_instances,
            "collision_primitives": [camera_collision_primitive()],
        }
    ]

    physical_links = {
        link["name"]
        for link in ownership["links"]
        if link["physical_geometry_policy"] != "virtual_massless_serial_coordinate"
    }
    missing_visuals = sorted(physical_links - set(visual_instances))
    if missing_visuals:
        raise ValueError(f"physical links have no visual geometry: {missing_visuals}")
    bounds = {link: group_bounds(groups) for link, groups in groups_by_link.items()}
    collisions = collision_primitives(ownership, bounds)
    endpoints = anonymous_end_effectors(classification, ownership)
    collision_roles = Counter(
        primitive["role"]
        for link_primitives in collisions.values()
        for primitive in link_primitives
    )
    collision_roles.update(
        primitive["role"]
        for fixed in fixed_geometry
        for primitive in fixed["collision_primitives"]
    )
    collision_roles["anonymous_end_effector"] += len(endpoints)
    return {
        "schema_version": SCHEMA_VERSION,
        "review_contract": OUTPUT_CONTRACT,
        "source": ownership["source"],
        "source_manifests": {
            **ownership["source_manifests"],
            "resolved_link_ownership_sha256": ownership_sha256,
            "inertial_manifest_sha256": inertials_sha256,
            "classification_manifest_sha256": classification_sha256,
        },
        "visual_policy": {
            "mode": "minimal_exterior_candidates",
            "structure": "omit named fasteners/spacers, small bearings, and sub-gram details",
            "catalog_motor": "one decimated assembly envelope per physical actuator",
            "catalog_component": "retain the reviewed external CAD housing",
            "measured_assembly": "omit enclosed battery and electronics geometry",
            "mass_model_independent": True,
        },
        "collision_policy": {
            "mode": "primitives_only",
            "allowed_types": ["box", "cylinder", "sphere"],
            "segment_links": "one axial cylinder plus one proximal joint sphere",
            "feet": (
                "two fitted sole boxes, two vertical heel-fillet cylinders, one "
                "vertical toe cylinder, and one horizontal ankle-housing cylinder "
                "per terminal ankle-roll link"
            ),
            "foot_sole_vertical_extent_m": [FOOT_SOLE_BOTTOM_M, FOOT_SOLE_TOP_M],
            "camera": "one box fitted to the exact Compound008 CAD bounds",
        },
        "links": [
            {
                "name": link["name"],
                "visual_instances": visual_instances.get(link["name"], []),
                "collision_primitives": collisions[link["name"]],
            }
            for link in ownership["links"]
        ],
        "fixed_geometry": fixed_geometry,
        "anonymous_end_effectors": endpoints,
        "summary": {
            "visual_link_count": len(visual_instances) + len(fixed_geometry),
            "visual_instance_count": sum(map(len, visual_instances.values()))
            + len(fixed_visual_instances),
            "collision_primitive_count": sum(map(len, collisions.values()))
            + sum(
                len(fixed["collision_primitives"]) for fixed in fixed_geometry
            )
            + len(endpoints),
            "collision_primitive_role_counts": dict(sorted(collision_roles.items())),
            "excluded_visual_instance_counts": dict(sorted(exclusions.items())),
            "visually_omitted_structure_mass_kg": excluded_structure_mass,
            "anonymous_sphere_count": len(endpoints),
            "anonymous_sphere_radius_m": ANONYMOUS_SPHERE_RADIUS_M,
        },
    }


def main() -> None:
    args = parse_args()
    ownership_path = args.ownership.expanduser().resolve()
    inertials_path = args.inertials.expanduser().resolve()
    classification_path = args.classification.expanduser().resolve()
    ownership = json.loads(ownership_path.read_text(encoding="utf-8"))
    inertials = json.loads(inertials_path.read_text(encoding="utf-8"))
    classification = json.loads(classification_path.read_text(encoding="utf-8"))
    result = build_manifest(
        ownership,
        inertials,
        classification,
        ownership_sha256=file_sha256(ownership_path),
        inertials_sha256=file_sha256(inertials_path),
        classification_sha256=file_sha256(classification_path),
    )
    atomic_write_json(args.output, result)
    print(f"wrote geometry manifest: {args.output}")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
