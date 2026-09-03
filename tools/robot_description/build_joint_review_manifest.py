"""Build the source-bound Bello Mini joint and link naming review."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
REVIEW_CONTRACT = "bello_mini_joint_link_naming_v3"


@dataclass(frozen=True)
class MotorSpec:
    motor_id: str
    motor_name: str
    model: str
    side: str
    station: str
    joint_names: tuple[str, ...]
    coupling: str = "direct"


@dataclass(frozen=True)
class JointSpec:
    name: str
    parent_link: str
    child_link: str
    side: str
    station: str
    motor_ids: tuple[str, ...]
    nominal_axis_cad: tuple[float, float, float]
    coupling: str = "direct"
    joint_type: str = "revolute"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def motor_specs() -> tuple[MotorSpec, ...]:
    return (
        MotorSpec(
            "motor-12",
            "waist_yaw_motor",
            "ROBSTRIDE_03",
            "center",
            "waist_yaw",
            ("waist_yaw_joint",),
        ),
        MotorSpec(
            "motor-01",
            "waist_roll_motor",
            "ROBSTRIDE_03",
            "center",
            "waist_roll",
            ("waist_roll_joint",),
        ),
        MotorSpec(
            "motor-11",
            "left_hip_pitch_motor",
            "ROBSTRIDE_04",
            "left",
            "hip_pitch",
            ("left_hip_pitch_joint",),
        ),
        MotorSpec(
            "motor-05",
            "left_hip_roll_motor",
            "ROBSTRIDE_03",
            "left",
            "hip_roll",
            ("left_hip_roll_joint",),
        ),
        MotorSpec(
            "motor-16",
            "left_hip_yaw_motor",
            "ROBSTRIDE_03",
            "left",
            "hip_yaw",
            ("left_hip_yaw_joint",),
        ),
        MotorSpec(
            "motor-14",
            "left_knee_motor",
            "ROBSTRIDE_04",
            "left",
            "knee",
            ("left_knee_joint",),
        ),
        MotorSpec(
            "motor-15",
            "left_ankle_motor_1",
            "ROBSTRIDE_02",
            "left",
            "ankle_differential_upper",
            ("left_ankle_pitch_joint", "left_ankle_roll_joint"),
            "differential",
        ),
        MotorSpec(
            "motor-13",
            "left_ankle_motor_2",
            "ROBSTRIDE_02",
            "left",
            "ankle_differential_lower",
            ("left_ankle_pitch_joint", "left_ankle_roll_joint"),
            "differential",
        ),
        MotorSpec(
            "motor-04",
            "right_hip_pitch_motor",
            "ROBSTRIDE_04",
            "right",
            "hip_pitch",
            ("right_hip_pitch_joint",),
        ),
        MotorSpec(
            "motor-06",
            "right_hip_roll_motor",
            "ROBSTRIDE_03",
            "right",
            "hip_roll",
            ("right_hip_roll_joint",),
        ),
        MotorSpec(
            "motor-07",
            "right_hip_yaw_motor",
            "ROBSTRIDE_03",
            "right",
            "hip_yaw",
            ("right_hip_yaw_joint",),
        ),
        MotorSpec(
            "motor-09",
            "right_knee_motor",
            "ROBSTRIDE_04",
            "right",
            "knee",
            ("right_knee_joint",),
        ),
        MotorSpec(
            "motor-10",
            "right_ankle_motor_1",
            "ROBSTRIDE_02",
            "right",
            "ankle_differential_upper",
            ("right_ankle_pitch_joint", "right_ankle_roll_joint"),
            "differential",
        ),
        MotorSpec(
            "motor-08",
            "right_ankle_motor_2",
            "ROBSTRIDE_02",
            "right",
            "ankle_differential_lower",
            ("right_ankle_pitch_joint", "right_ankle_roll_joint"),
            "differential",
        ),
        MotorSpec(
            "motor-02",
            "left_shoulder_pitch_motor",
            "ROBSTRIDE_03",
            "left",
            "shoulder_pitch",
            ("left_shoulder_pitch_joint",),
        ),
        MotorSpec(
            "motor-24",
            "left_shoulder_roll_motor",
            "ROBSTRIDE_03",
            "left",
            "shoulder_roll",
            ("left_shoulder_roll_joint",),
        ),
        MotorSpec(
            "motor-26",
            "left_shoulder_yaw_motor",
            "ROBSTRIDE_06",
            "left",
            "shoulder_yaw",
            ("left_shoulder_yaw_joint",),
        ),
        MotorSpec(
            "motor-25",
            "left_elbow_pitch_motor",
            "ROBSTRIDE_06",
            "left",
            "elbow_pitch",
            ("left_elbow_pitch_joint",),
        ),
        MotorSpec(
            "motor-27",
            "left_elbow_yaw_motor",
            "ROBSTRIDE_00",
            "left",
            "elbow_yaw",
            ("left_elbow_yaw_joint",),
        ),
        MotorSpec(
            "motor-28",
            "left_wrist_pitch_motor",
            "ROBSTRIDE_00",
            "left",
            "wrist_pitch",
            ("left_wrist_pitch_joint",),
        ),
        MotorSpec(
            "motor-23",
            "left_wrist_roll_motor",
            "ROBSTRIDE_00",
            "left",
            "wrist_roll",
            ("left_wrist_roll_joint",),
        ),
        MotorSpec(
            "motor-03",
            "right_shoulder_pitch_motor",
            "ROBSTRIDE_03",
            "right",
            "shoulder_pitch",
            ("right_shoulder_pitch_joint",),
        ),
        MotorSpec(
            "motor-18",
            "right_shoulder_roll_motor",
            "ROBSTRIDE_03",
            "right",
            "shoulder_roll",
            ("right_shoulder_roll_joint",),
        ),
        MotorSpec(
            "motor-20",
            "right_shoulder_yaw_motor",
            "ROBSTRIDE_06",
            "right",
            "shoulder_yaw",
            ("right_shoulder_yaw_joint",),
        ),
        MotorSpec(
            "motor-19",
            "right_elbow_pitch_motor",
            "ROBSTRIDE_06",
            "right",
            "elbow_pitch",
            ("right_elbow_pitch_joint",),
        ),
        MotorSpec(
            "motor-21",
            "right_elbow_yaw_motor",
            "ROBSTRIDE_00",
            "right",
            "elbow_yaw",
            ("right_elbow_yaw_joint",),
        ),
        MotorSpec(
            "motor-22",
            "right_wrist_pitch_motor",
            "ROBSTRIDE_00",
            "right",
            "wrist_pitch",
            ("right_wrist_pitch_joint",),
        ),
        MotorSpec(
            "motor-17",
            "right_wrist_roll_motor",
            "ROBSTRIDE_00",
            "right",
            "wrist_roll",
            ("right_wrist_roll_joint",),
        ),
    )


def side_joint_specs(side: str) -> tuple[JointSpec, ...]:
    prefix = f"{side}_"
    return (
        JointSpec(
            f"{prefix}hip_pitch_joint",
            "pelvis_link",
            f"{prefix}hip_pitch_link",
            side,
            "hip_pitch",
            (("motor-11",) if side == "left" else ("motor-04",)),
            (1.0, 0.0, 0.0),
        ),
        JointSpec(
            f"{prefix}hip_roll_joint",
            f"{prefix}hip_pitch_link",
            f"{prefix}hip_roll_link",
            side,
            "hip_roll",
            (("motor-05",) if side == "left" else ("motor-06",)),
            (0.0, 0.0, 1.0),
        ),
        JointSpec(
            f"{prefix}hip_yaw_joint",
            f"{prefix}hip_roll_link",
            f"{prefix}hip_yaw_link",
            side,
            "hip_yaw",
            (("motor-16",) if side == "left" else ("motor-07",)),
            (0.0, 1.0, 0.0),
        ),
        JointSpec(
            f"{prefix}knee_joint",
            f"{prefix}hip_yaw_link",
            f"{prefix}knee_link",
            side,
            "knee",
            (("motor-14",) if side == "left" else ("motor-09",)),
            (1.0, 0.0, 0.0),
        ),
        JointSpec(
            f"{prefix}ankle_pitch_joint",
            f"{prefix}knee_link",
            f"{prefix}ankle_pitch_link",
            side,
            "ankle_pitch",
            (("motor-15", "motor-13") if side == "left" else ("motor-10", "motor-08")),
            (1.0, 0.0, 0.0),
            "differential",
        ),
        JointSpec(
            f"{prefix}ankle_roll_joint",
            f"{prefix}ankle_pitch_link",
            f"{prefix}ankle_roll_link",
            side,
            "ankle_roll",
            (("motor-15", "motor-13") if side == "left" else ("motor-10", "motor-08")),
            (0.0, 0.0, 1.0),
            "differential",
        ),
    )


def arm_joint_specs(side: str) -> tuple[JointSpec, ...]:
    prefix = f"{side}_"
    ids = {
        "left": (
            "motor-02",
            "motor-24",
            "motor-26",
            "motor-25",
            "motor-27",
            "motor-28",
            "motor-23",
        ),
        "right": (
            "motor-03",
            "motor-18",
            "motor-20",
            "motor-19",
            "motor-21",
            "motor-22",
            "motor-17",
        ),
    }[side]
    return (
        JointSpec(
            f"{prefix}shoulder_pitch_joint",
            "torso_link",
            f"{prefix}shoulder_pitch_link",
            side,
            "shoulder_pitch",
            (ids[0],),
            (1.0, 0.0, 0.0),
        ),
        JointSpec(
            f"{prefix}shoulder_roll_joint",
            f"{prefix}shoulder_pitch_link",
            f"{prefix}shoulder_roll_link",
            side,
            "shoulder_roll",
            (ids[1],),
            (0.0, 0.0, 1.0),
        ),
        JointSpec(
            f"{prefix}shoulder_yaw_joint",
            f"{prefix}shoulder_roll_link",
            f"{prefix}upper_arm_link",
            side,
            "shoulder_yaw",
            (ids[2],),
            (0.0, 1.0, 0.0),
        ),
        JointSpec(
            f"{prefix}elbow_pitch_joint",
            f"{prefix}upper_arm_link",
            f"{prefix}elbow_link",
            side,
            "elbow_pitch",
            (ids[3],),
            (1.0, 0.0, 0.0),
        ),
        JointSpec(
            f"{prefix}elbow_yaw_joint",
            f"{prefix}elbow_link",
            f"{prefix}lower_arm_link",
            side,
            "elbow_yaw",
            (ids[4],),
            (0.0, 1.0, 0.0),
        ),
        JointSpec(
            f"{prefix}wrist_pitch_joint",
            f"{prefix}lower_arm_link",
            f"{prefix}wrist_link",
            side,
            "wrist_pitch",
            (ids[5],),
            (1.0, 0.0, 0.0),
        ),
        JointSpec(
            f"{prefix}wrist_roll_joint",
            f"{prefix}wrist_link",
            f"{prefix}wrist_roll_link",
            side,
            "wrist_roll",
            (ids[6],),
            (0.0, 0.0, 1.0),
        ),
        JointSpec(
            f"{prefix}end_effector_sphere_joint",
            f"{prefix}wrist_roll_link",
            f"{prefix}end_effector_sphere_link",
            side,
            "end_effector_sphere",
            (),
            (0.0, 0.0, 0.0),
            "fixed",
            "fixed",
        ),
    )


def joint_specs() -> tuple[JointSpec, ...]:
    return (
        JointSpec(
            "waist_yaw_joint",
            "pelvis_link",
            "waist_yaw_link",
            "center",
            "waist_yaw",
            ("motor-12",),
            (0.0, 1.0, 0.0),
        ),
        JointSpec(
            "waist_roll_joint",
            "waist_yaw_link",
            "torso_link",
            "center",
            "waist_roll",
            ("motor-01",),
            (0.0, 0.0, 1.0),
        ),
        *side_joint_specs("left"),
        *side_joint_specs("right"),
        *arm_joint_specs("left"),
        *arm_joint_specs("right"),
    )


def unit_axis(frame_transform: list[float]) -> list[float]:
    frame = np.asarray(frame_transform, dtype=np.float64).reshape(4, 4).T
    axis = frame[:3, 2]
    norm = np.linalg.norm(axis)
    if not math.isfinite(norm) or norm < 0.999 or norm > 1.001:
        raise ValueError(f"invalid motor output axis norm: {norm}")
    return (axis / norm).tolist()


def bounds_center(bounds: dict[str, list[float]]) -> np.ndarray:
    return (
        np.asarray(bounds["min"], dtype=np.float64)
        + np.asarray(bounds["max"], dtype=np.float64)
    ) / 2.0


def ankle_pivot_origins(manifest: dict[str, Any]) -> dict[str, list[float]]:
    bearing_components = [
        component
        for component in manifest["components"]
        if component["source_label"] == "22SUS-6B2"
    ]
    if len(bearing_components) != 1:
        raise ValueError(
            "expected one 22SUS-6B2 source for the orthogonal ankle bearings"
        )
    occurrences = [
        occurrence
        for occurrence in bearing_components[0]["occurrences"]
        if occurrence["motor_assembly"] is None
        and occurrence["measured_assembly"] is None
        and occurrence["anonymous_end_effector"] is None
    ]
    by_side = {
        "left": [
            occurrence
            for occurrence in occurrences
            if bounds_center(occurrence["bounds_mm"])[0] > 0.0
        ],
        "right": [
            occurrence
            for occurrence in occurrences
            if bounds_center(occurrence["bounds_mm"])[0] < 0.0
        ],
    }
    result = {}
    for side, side_occurrences in by_side.items():
        if len(side_occurrences) != 4:
            raise ValueError(
                f"expected four orthogonal ankle bearings on {side}, "
                f"found {len(side_occurrences)}"
            )
        centers = np.asarray(
            [bounds_center(occurrence["bounds_mm"]) for occurrence in side_occurrences]
        )
        sizes = np.asarray(
            [
                np.asarray(occurrence["bounds_mm"]["max"])
                - np.asarray(occurrence["bounds_mm"]["min"])
                for occurrence in side_occurrences
            ]
        )
        shortest_axes = np.argmin(sizes, axis=1)
        if sorted(shortest_axes.tolist()) != [0, 0, 2, 2]:
            raise ValueError(
                f"{side} ankle bearings do not form two CAD-X and two CAD-Z axes"
            )
        pivot = centers.mean(axis=0)
        if np.max(np.abs(centers[:, 1] - pivot[1])) > 0.01:
            raise ValueError(f"{side} ankle bearing axes do not share one height")
        result[side] = pivot.tolist()
    if not np.allclose(
        result["left"],
        [-result["right"][0], result["right"][1], result["right"][2]],
        atol=0.01,
    ):
        raise ValueError("left/right ankle bearing pivots are not symmetric")
    return result


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    review_path = args.review.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))

    if review.get("review_contract") != manifest["review_contract"]:
        raise ValueError("classification review contract does not match manifest")
    if review.get("source", {}).get("sha256") != manifest["source"]["sha256"]:
        raise ValueError("classification review STEP checksum does not match manifest")

    by_motor_id = {motor["id"]: motor for motor in manifest["motor_assemblies"]}
    specs = motor_specs()
    expected_ids = {spec.motor_id for spec in specs}
    if expected_ids != set(by_motor_id):
        missing = sorted(expected_ids - set(by_motor_id))
        unexpected = sorted(set(by_motor_id) - expected_ids)
        raise ValueError(
            f"motor inventory mismatch: missing={missing}, unexpected={unexpected}"
        )
    if len({spec.motor_name for spec in specs}) != len(specs):
        raise ValueError("proposed motor names are not unique")

    motors = []
    for spec in specs:
        source = by_motor_id[spec.motor_id]
        selected_model = review["motor_decisions"].get(spec.motor_id)
        if selected_model != spec.model:
            raise ValueError(
                f"{spec.motor_id} expected {spec.model}, review selected {selected_model}"
            )
        axis = unit_axis(source["frame_transform"])
        motors.append(
            {
                "id": spec.motor_id,
                "motor_name": spec.motor_name,
                "model": spec.model,
                "side": spec.side,
                "station": spec.station,
                "joint_names": list(spec.joint_names),
                "coupling": spec.coupling,
                "creates_joint_at_motor_center": spec.coupling == "direct",
                "role": (
                    "remote_transmission_actuator"
                    if spec.coupling == "differential"
                    else "joint_actuator"
                ),
                "cad_object": source["cad_object"],
                "cad_label": source["cad_label"],
                "cad_path": source["path"],
                "center_mm": source["center_mm"],
                "output_axis_cad": axis,
                "output_axis_evidence": "CAD motor frame local +Z; sign pending",
                "mesh": source["mesh"],
                "shape_transform": source["shape_transform"],
                "bounds_mm": source["bounds_mm"],
            }
        )

    joints = []
    ankle_origins = ankle_pivot_origins(manifest)
    all_joint_specs = joint_specs()
    joint_names = tuple(spec.name for spec in all_joint_specs)
    child_links = tuple(spec.child_link for spec in all_joint_specs)
    if len(set(joint_names)) != len(joint_names):
        raise ValueError("proposed joint names are not unique")
    if len(set(child_links)) != len(child_links) or "pelvis_link" in child_links:
        raise ValueError("proposed links do not form a single-parent rooted tree")
    joint_spec_by_name = {spec.name: spec for spec in all_joint_specs}
    for spec in specs:
        for joint_name in spec.joint_names:
            joint_spec = joint_spec_by_name[joint_name]
            if spec.motor_id not in joint_spec.motor_ids:
                raise ValueError(
                    f"{spec.motor_name} and {joint_name} disagree on their binding"
                )
    motor_axis_by_id = {motor["id"]: motor["output_axis_cad"] for motor in motors}
    for spec in all_joint_specs:
        if spec.joint_type == "revolute" and not spec.motor_ids:
            raise ValueError(f"movable joint {spec.name} has no motors")
        if spec.coupling == "direct":
            motor_axis = np.asarray(motor_axis_by_id[spec.motor_ids[0]])
            nominal_axis = np.asarray(spec.nominal_axis_cad)
            alignment = abs(float(np.dot(motor_axis, nominal_axis)))
            if alignment < 0.999:
                raise ValueError(
                    f"{spec.name} motor axis disagrees with semantic axis: {alignment}"
                )
        else:
            alignment = None
        ankle_joint = spec.station in {"ankle_pitch", "ankle_roll"}
        joints.append(
            {
                "name": spec.name,
                "type": spec.joint_type,
                "parent_link": spec.parent_link,
                "child_link": spec.child_link,
                "side": spec.side,
                "station": spec.station,
                "motor_ids": list(spec.motor_ids),
                "coupling": spec.coupling,
                "nominal_axis_cad": list(spec.nominal_axis_cad),
                "motor_axis_alignment_abs": alignment,
                "origin_cad_mm": ankle_origins[spec.side] if ankle_joint else None,
                "origin_status": (
                    "cad_orthogonal_bearing_intersection"
                    if ankle_joint
                    else (
                        "pending_shaft_frame_audit"
                        if spec.joint_type == "revolute"
                        else "pending_wrist_mount_audit"
                    )
                ),
                "origin_evidence": (
                    "intersection of two CAD-X and two CAD-Z 22SUS-6B2 bearing occurrences"
                    if ankle_joint
                    else None
                ),
            }
        )

    link_names = {"pelvis_link"}
    for joint in joints:
        link_names.add(joint["parent_link"])
        link_names.add(joint["child_link"])
    if len(link_names) != 31:
        raise ValueError(f"expected 31 proposed links, found {len(link_names)}")

    chains = [
        {
            "id": "waist",
            "label": "Waist (2 DoF)",
            "color": "#f9c74f",
            "motor_ids": ["motor-12", "motor-01"],
        },
        {
            "id": "left_leg",
            "label": "Left leg (4 direct + 2 remote motors / 6 DoF)",
            "color": "#43aa8b",
            "motor_ids": [
                "motor-11",
                "motor-05",
                "motor-16",
                "motor-14",
                "motor-15",
                "motor-13",
            ],
        },
        {
            "id": "right_leg",
            "label": "Right leg (4 direct + 2 remote motors / 6 DoF)",
            "color": "#90be6d",
            "motor_ids": [
                "motor-04",
                "motor-06",
                "motor-07",
                "motor-09",
                "motor-10",
                "motor-08",
            ],
        },
        {
            "id": "left_arm",
            "label": "Left arm (7 DoF)",
            "color": "#577590",
            "motor_ids": [
                "motor-02",
                "motor-24",
                "motor-26",
                "motor-25",
                "motor-27",
                "motor-28",
                "motor-23",
            ],
        },
        {
            "id": "right_arm",
            "label": "Right arm (7 DoF)",
            "color": "#f3722c",
            "motor_ids": [
                "motor-03",
                "motor-18",
                "motor-20",
                "motor-19",
                "motor-21",
                "motor-22",
                "motor-17",
            ],
        },
    ]
    chained_motor_ids = [
        motor_id for chain in chains for motor_id in chain["motor_ids"]
    ]
    if (
        len(chained_motor_ids) != len(set(chained_motor_ids))
        or set(chained_motor_ids) != expected_ids
    ):
        raise ValueError("motor chains do not partition the 28 actuator candidates")

    manifest_value = {
        "schema_version": SCHEMA_VERSION,
        "review_contract": REVIEW_CONTRACT,
        "source": manifest["source"],
        "classification_review": {
            "review_contract": review["review_contract"],
            "sha256": file_sha256(review_path),
        },
        "summary": {
            "motor_count": len(motors),
            "movable_joint_count": sum(joint["type"] == "revolute" for joint in joints),
            "fixed_joint_count": sum(joint["type"] == "fixed" for joint in joints),
            "link_count": len(link_names),
            "arm_dof_per_side": 7,
            "leg_dof_per_side": 6,
            "waist_dof": 2,
            "direct_drive_joint_count": sum(
                joint["coupling"] == "direct" and joint["type"] == "revolute"
                for joint in joints
            ),
            "linkage_driven_joint_count": sum(
                joint["coupling"] == "differential" for joint in joints
            ),
            "remote_transmission_motor_count": sum(
                motor["role"] == "remote_transmission_actuator" for motor in motors
            ),
            "cad_located_joint_origin_count": sum(
                joint["origin_cad_mm"] is not None for joint in joints
            ),
        },
        "coordinate_frame": {
            "units": "mm",
            "cad_axes": {"x": "robot left", "y": "up", "z": "rear/front pending sign"},
            "origin_policy": (
                "direct-drive shaft centers are pending; ankle pitch/roll axes use "
                "the lower orthogonal-bearing intersection; motor centers are visual "
                "markers only"
            ),
        },
        "root": manifest["root"],
        "links": [
            {"name": name, "root": name == "pelvis_link"} for name in sorted(link_names)
        ],
        "joints": joints,
        "motors": motors,
        "chains": chains,
        "end_effector_policy": manifest["end_effector_policy"],
        "review_requirements": {
            "confirm_motor_names": True,
            "confirm_joint_names": True,
            "confirm_parent_child_tree": True,
            "axis_signs": "separate joint-frame gate",
            "joint_origins": (
                "ankle pivots included; direct-drive shaft centers remain a separate "
                "joint-frame gate"
            ),
            "ankle_differential_coefficients": "separate transmission gate",
            "ankle_motor_joint_policy": (
                "the four RS02 motor shafts create no URDF joints; they actuate the "
                "four lower ankle coordinates through two bilateral 2x2 linkages"
            ),
        },
    }
    atomic_write_json(args.output, manifest_value)
    print(f"wrote joint review manifest: {args.output}")
    print(json.dumps(manifest_value["summary"], indent=2))


if __name__ == "__main__":
    main()
