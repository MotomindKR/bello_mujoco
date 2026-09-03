"""Build a source-bound 3D review for Bello Mini rigid-link ownership."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
REVIEW_CONTRACT = "bello_mini_link_ownership_v1"
RESOLVED_FRAME_CONTRACT = "bello_mini_resolved_joint_frames_v1"
ARM_BRANCHES = {"LinkGroup2988": "left", "LinkGroup2926": "right"}
LOWER_BODY_BRANCH = "LinkGroup025"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--classification-review", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
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


def bounds_union(bounds: Iterable[dict[str, list[float]]]) -> dict[str, list[float]]:
    values = list(bounds)
    if not values:
        raise ValueError("cannot calculate bounds for an empty group")
    return {
        "min": [min(value["min"][axis] for value in values) for axis in range(3)],
        "max": [max(value["max"][axis] for value in values) for axis in range(3)],
    }


def bounds_center(bounds: dict[str, list[float]]) -> list[float]:
    return [
        (minimum + maximum) / 2.0
        for minimum, maximum in zip(bounds["min"], bounds["max"], strict=True)
    ]


def validate_inputs(
    manifest: dict[str, Any],
    classification: dict[str, Any],
    frames: dict[str, Any],
) -> None:
    checksums = {
        manifest.get("source", {}).get("sha256"),
        classification.get("source", {}).get("sha256"),
        frames.get("source", {}).get("sha256"),
    }
    if None in checksums or len(checksums) != 1:
        raise ValueError(
            "CAD, classification, and joint frames use different STEP files"
        )
    if classification.get("review_contract") != manifest.get("review_contract"):
        raise ValueError("classification review contract does not match CAD manifest")
    if frames.get("review_contract") != RESOLVED_FRAME_CONTRACT:
        raise ValueError("joint frames have not passed the resolved-frame gate")
    if frames.get("summary", {}).get("movable_joint_count") != len(
        frames.get("joints", [])
    ):
        raise ValueError("resolved joint-frame count is inconsistent")
    if any(
        joint.get("frame_review_decision") not in {"confirmed", "flip"}
        for joint in frames["joints"]
    ):
        raise ValueError("not every joint frame is human-approved")


def flatten_occurrences(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    flattened = []
    seen_ids: set[str] = set()
    for component in manifest["components"]:
        component_fields = {
            "source_label": component["source_label"],
            "source_name": component["source_name"],
            "volume_mm3": component["volume_mm3"],
            "mesh": component["mesh"],
        }
        for source_occurrence in component["occurrences"]:
            occurrence = {**source_occurrence, **component_fields}
            if occurrence["id"] in seen_ids:
                raise ValueError(f"duplicate physical occurrence: {occurrence['id']}")
            seen_ids.add(occurrence["id"])
            flattened.append(occurrence)
    expected = manifest["summary"]["physical_occurrence_count"]
    if len(flattened) != expected:
        raise ValueError(
            f"physical occurrence count mismatch: {len(flattened)} != {expected}"
        )
    return flattened


def motor_parent_links(frames: dict[str, Any]) -> dict[str, str]:
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for joint in frames["joints"]:
        for motor_id in joint["motor_ids"]:
            candidates[motor_id].append(joint)
    result = {}
    for motor in frames["motors"]:
        joints = candidates[motor["id"]]
        if len(joints) == 1:
            result[motor["id"]] = joints[0]["parent_link"]
            continue
        ankle_pitch = [joint for joint in joints if joint["station"] == "ankle_pitch"]
        if (
            len(ankle_pitch) != 1
            or motor.get("role") != "remote_transmission_actuator"
            or motor.get("creates_joint_at_motor_center") is not False
        ):
            raise ValueError(f"ambiguous physical housing owner for {motor['id']}")
        result[motor["id"]] = ankle_pitch[0]["parent_link"]
    return result


def occurrence_side(bounds: dict[str, list[float]]) -> str:
    minimum_x, maximum_x = bounds["min"][0], bounds["max"][0]
    center_x = (minimum_x + maximum_x) / 2.0
    if minimum_x > 1.0 or center_x > 40.0:
        return "left"
    if maximum_x < -1.0 or center_x < -40.0:
        return "right"
    return "center"


def arm_link(side: str, y_mm: float) -> str:
    stations = (
        (249.0, "torso_link"),
        (224.0, f"{side}_shoulder_pitch_link"),
        (131.22691858, f"{side}_shoulder_roll_link"),
        (26.49999934, f"{side}_upper_arm_link"),
        (-63.28812675, f"{side}_elbow_link"),
        (-156.40023324, f"{side}_lower_arm_link"),
        (-221.40023324, f"{side}_wrist_link"),
    )
    for station, parent_link in stations:
        if y_mm >= station:
            return parent_link
    return f"{side}_wrist_roll_link"


def leg_link(side: str, y_mm: float) -> str:
    stations = (
        (-180.00000114, "pelvis_link"),
        (-210.00000008, f"{side}_hip_pitch_link"),
        (-317.76182733, f"{side}_hip_roll_link"),
        (-535.12231787, f"{side}_hip_yaw_link"),
        (-827.12325379, f"{side}_knee_link"),
    )
    for station, parent_link in stations:
        if y_mm >= station:
            return parent_link
    # The serial ankle-pitch link is a massless transmission coordinate. All
    # physical foot/ankle geometry belongs to the terminal roll link.
    return f"{side}_ankle_roll_link"


def propose_structure_owner(occurrence: dict[str, Any]) -> tuple[str, str, str]:
    branch = occurrence["top_branch"]
    center_y = bounds_center(occurrence["bounds_mm"])[1]
    if branch in ARM_BRANCHES:
        side = ARM_BRANCHES[branch]
        return arm_link(side, center_y), "arm hierarchy + CAD vertical station", side
    if branch == LOWER_BODY_BRANCH:
        side = occurrence_side(occurrence["bounds_mm"])
        if side == "center":
            return "pelvis_link", "central lower-body hierarchy", side
        return leg_link(side, center_y), "leg hierarchy + CAD vertical station", side
    return "torso_link", "torso assembly hierarchy", "center"


def structural_prefix(occurrence: dict[str, Any]) -> str:
    parts = occurrence["id"].strip("/").split("/")
    branch = occurrence["top_branch"]
    try:
        branch_index = parts.index(branch)
    except ValueError:
        branch_index = 0
    prefix_length = min(len(parts), branch_index + 2)
    return "/" + "/".join(parts[:prefix_length])


def material_mass_kg(
    occurrence: dict[str, Any],
    classification: dict[str, Any],
    densities: dict[str, float],
) -> tuple[str, float]:
    try:
        decision = classification["component_decisions"][occurrence["source_name"]]
    except KeyError as error:
        raise ValueError(
            f"structural source lacks a material decision: {occurrence['source_name']}"
        ) from error
    if decision.get("classification") != "material":
        raise ValueError(
            f"structural source is not material-classified: {occurrence['source_name']}"
        )
    material = decision.get("material")
    if material not in densities:
        raise ValueError(
            f"unknown material for {occurrence['source_name']}: {material}"
        )
    mass_scale = float(decision.get("mass_scale", 1.0))
    if mass_scale <= 0.0:
        raise ValueError(f"invalid mass scale for {occurrence['source_name']}")
    mass_kg = float(occurrence["volume_mm3"]) * 1e-9 * densities[material] * mass_scale
    return material, mass_kg


def review_occurrence(
    occurrence: dict[str, Any],
    *,
    material: str | None = None,
    estimated_mass_kg: float | None = None,
) -> dict[str, Any]:
    return {
        "bounds_mm": occurrence["bounds_mm"],
        "estimated_mass_kg": estimated_mass_kg,
        "id": occurrence["id"],
        "material": material,
        "mesh": occurrence["mesh"],
        "source_label": occurrence["source_label"],
        "source_name": occurrence["source_name"],
        "transform": occurrence["transform"],
    }


def group_record(
    *,
    kind: str,
    label: str,
    occurrences: list[dict[str, Any]],
    proposed_link: str,
    reason: str,
    side: str,
    path_prefix: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    group_bounds = bounds_union(occurrence["bounds_mm"] for occurrence in occurrences)
    material_counts = Counter(
        occurrence["material"]
        for occurrence in occurrences
        if occurrence["material"] is not None
    )
    estimated_mass = sum(
        occurrence["estimated_mass_kg"] or 0.0 for occurrence in occurrences
    )
    record = {
        "bounds_mm": group_bounds,
        "center_mm": bounds_center(group_bounds),
        "estimated_material_mass_kg": estimated_mass if material_counts else None,
        "kind": kind,
        "label": label,
        "material_occurrence_counts": dict(sorted(material_counts.items())),
        "occurrence_count": len(occurrences),
        "occurrences": sorted(occurrences, key=lambda item: item["id"]),
        "path_prefix": path_prefix,
        "proposed_link": proposed_link,
        "proposal_reason": reason,
        "side": side,
    }
    if extra:
        record.update(extra)
    return record


def build_groups(
    manifest: dict[str, Any],
    classification: dict[str, Any],
    frames: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    occurrences = flatten_occurrences(manifest)
    excluded = [
        value for value in occurrences if value["anonymous_end_effector"] is not None
    ]
    included = [
        value for value in occurrences if value["anonymous_end_effector"] is None
    ]
    motor_occurrences: dict[int, list[dict[str, Any]]] = defaultdict(list)
    measured_occurrences: dict[int, list[dict[str, Any]]] = defaultdict(list)
    structural = []
    for occurrence in included:
        if occurrence["motor_assembly"] is not None:
            motor_occurrences[int(occurrence["motor_assembly"])].append(occurrence)
        elif occurrence["measured_assembly"] is not None:
            measured_occurrences[int(occurrence["measured_assembly"])].append(
                occurrence
            )
        else:
            structural.append(occurrence)

    densities = {
        material["id"]: float(material["density_kg_m3"])
        for material in manifest["materials"]
    }
    structural_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    structural_metadata: dict[tuple[str, str], tuple[str, str]] = {}
    for occurrence in structural:
        owner, reason, side = propose_structure_owner(occurrence)
        material, mass = material_mass_kg(occurrence, classification, densities)
        reviewed = review_occurrence(
            occurrence, material=material, estimated_mass_kg=mass
        )
        key = (structural_prefix(occurrence), owner)
        structural_groups[key].append(reviewed)
        structural_metadata[key] = (reason, side)

    groups = []
    for (prefix, owner), values in sorted(structural_groups.items()):
        reason, side = structural_metadata[(prefix, owner)]
        groups.append(
            group_record(
                kind="structure",
                label=prefix.rsplit("/", 1)[-1],
                occurrences=values,
                proposed_link=owner,
                reason=reason,
                side=side,
                path_prefix=prefix,
            )
        )

    motor_parents = motor_parent_links(frames)
    motors = {motor["id"]: motor for motor in frames["motors"]}
    motor_sources = manifest["motor_assemblies"]
    if set(motor_occurrences) != set(range(len(motor_sources))):
        raise ValueError(
            "motor occurrence inventory does not cover all catalog actuators"
        )
    for index, source_motor in enumerate(motor_sources):
        motor_id = source_motor["id"]
        motor = motors[motor_id]
        values = [review_occurrence(value) for value in motor_occurrences[index]]
        groups.append(
            group_record(
                kind="catalog_motor",
                label=f"{motor['motor_name']} ({motor['model']})",
                occurrences=values,
                proposed_link=motor_parents[motor_id],
                reason="catalog motor housing belongs to the physical parent side",
                side=motor["side"],
                path_prefix=source_motor["path"],
                extra={
                    "motor_id": motor_id,
                    "motor_model": motor["model"],
                    "motor_name": motor["motor_name"],
                    "coupling": motor["coupling"],
                    "selection_instances": [
                        {
                            "mesh": source_motor["mesh"],
                            "transform": source_motor["shape_transform"],
                        }
                    ],
                },
            )
        )

    measured_sources = manifest["measured_assemblies"]
    if set(measured_occurrences) != set(range(len(measured_sources))):
        raise ValueError("measured assembly occurrence inventory is incomplete")
    for index, source in enumerate(measured_sources):
        measured_id = source["id"]
        decision = classification["measured_assembly_decisions"].get(measured_id)
        if not decision or decision.get("treatment") != "measured_assembly":
            raise ValueError(f"measured assembly is not approved: {measured_id}")
        values = [review_occurrence(value) for value in measured_occurrences[index]]
        groups.append(
            group_record(
                kind="measured_assembly",
                label=source["cad_label"],
                occurrences=values,
                proposed_link="torso_link",
                reason="measured battery/electronics assembly is rigid in the torso",
                side="center",
                path_prefix=source["path"],
                extra={
                    "measured_assembly_id": measured_id,
                    "measured_mass_kg": float(decision["mass_kg"]),
                    "measured_kind": source["kind"],
                    "selection_instances": [
                        {
                            "mesh": source["mesh"],
                            "transform": source["shape_transform"],
                        }
                    ],
                },
            )
        )

    groups.sort(
        key=lambda value: (value["proposed_link"], value["kind"], value["path_prefix"])
    )
    for index, group in enumerate(groups, start=1):
        group["id"] = f"ownership-{index:03d}"

    assigned_ids = [
        occurrence["id"] for group in groups for occurrence in group["occurrences"]
    ]
    expected_ids = {occurrence["id"] for occurrence in included}
    if len(assigned_ids) != len(set(assigned_ids)):
        raise ValueError("an occurrence is assigned to more than one ownership group")
    if set(assigned_ids) != expected_ids:
        raise ValueError("ownership groups do not exactly cover non-hand geometry")
    return groups, {
        "excluded_hand_occurrence_count": len(excluded),
        "included_occurrence_count": len(included),
        "structural_occurrence_count": len(structural),
    }


def link_inventory(frames: dict[str, Any]) -> list[dict[str, Any]]:
    child_links = {joint["child_link"] for joint in frames["joints"]}
    parent_links = {joint["parent_link"] for joint in frames["joints"]}
    links = sorted(child_links | parent_links)
    return [
        {
            "name": link,
            "physical_geometry_policy": (
                "virtual_massless_serial_coordinate"
                if link.endswith("ankle_pitch_link")
                else "reviewed_cad_ownership"
            ),
        }
        for link in links
    ]


def group_region(group: dict[str, Any]) -> str:
    path = group["path_prefix"]
    if "LinkGroup2988" in path or "LinkGroup2926" in path:
        return "arm"
    if f"/{LOWER_BODY_BRANCH}/" in path:
        return "lower_body"
    return "root"


def structure_signature(group: dict[str, Any]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        sorted(
            (
                occurrence["material"],
                round(float(occurrence["estimated_mass_kg"] or 0.0), 9),
                int(occurrence["mesh"]["triangle_count"]),
            )
            for occurrence in group["occurrences"]
        )
    )


def bilateral_spatial_residual_mm(left: dict[str, Any], right: dict[str, Any]) -> float:
    region = group_region(left)
    anchor_x = 206.25 if region == "arm" else 129.5 if region == "lower_body" else 0.0
    left_center, right_center = left["center_mm"], right["center_mm"]
    left_local_x = abs(left_center[0] - anchor_x)
    right_local_x = abs(right_center[0] + anchor_x)
    x_residual = abs(left_local_x - right_local_x)
    y_residual = abs(left_center[1] - right_center[1])
    same_z = abs(left_center[2] - right_center[2])
    mirrored_z = abs((left_center[2] + 10.0) + (right_center[2] + 10.0))
    z_residual = min(same_z, mirrored_z) if region != "root" else same_z
    return math.sqrt(x_residual**2 + y_residual**2 + z_residual**2)


def bilateral_group_inventory(
    groups: list[dict[str, Any]], *, maximum_residual_mm: float = 2.0
) -> tuple[list[dict[str, Any]], list[str]]:
    left = [group for group in groups if group["center_mm"][0] > 20.0]
    right = [group for group in groups if group["center_mm"][0] < -20.0]
    candidate_edges = []
    for left_group in left:
        for right_group in right:
            if left_group["kind"] != right_group["kind"]:
                continue
            if group_region(left_group) != group_region(right_group):
                continue
            if left_group["kind"] == "catalog_motor":
                left_name = left_group["motor_name"].removeprefix("left_")
                right_name = right_group["motor_name"].removeprefix("right_")
                if left_name != right_name:
                    continue
                evidence = "matching actuator station and model"
            elif left_group["kind"] == "structure":
                if structure_signature(left_group) != structure_signature(right_group):
                    continue
                evidence = "identical material, mass, and tessellation signature"
            else:
                continue
            residual = bilateral_spatial_residual_mm(left_group, right_group)
            if residual <= maximum_residual_mm:
                candidate_edges.append((residual, left_group, right_group, evidence))

    used_left: set[str] = set()
    used_right: set[str] = set()
    pairs = []
    for residual, left_group, right_group, evidence in sorted(
        candidate_edges, key=lambda item: (item[0], item[1]["id"], item[2]["id"])
    ):
        if left_group["id"] in used_left or right_group["id"] in used_right:
            continue
        used_left.add(left_group["id"])
        used_right.add(right_group["id"])
        pairs.append(
            {
                "evidence": evidence,
                "kind": left_group["kind"],
                "left_group_id": left_group["id"],
                "right_group_id": right_group["id"],
                "spatial_residual_mm": residual,
            }
        )
    unpaired = sorted(
        group["id"]
        for group in left + right
        if group["id"] not in used_left and group["id"] not in used_right
    )
    return pairs, unpaired


def build_manifest(
    manifest: dict[str, Any],
    classification: dict[str, Any],
    frames: dict[str, Any],
    *,
    manifest_sha256: str,
    classification_sha256: str,
    frames_sha256: str,
) -> dict[str, Any]:
    validate_inputs(manifest, classification, frames)
    groups, counts = build_groups(manifest, classification, frames)
    link_counts = Counter(group["proposed_link"] for group in groups)
    material_mass = sum(group["estimated_material_mass_kg"] or 0.0 for group in groups)
    measured_mass = sum(group.get("measured_mass_kg", 0.0) for group in groups)
    material_mass_by_id: Counter[str] = Counter()
    non_motor_mass_by_link: Counter[str] = Counter()
    for group in groups:
        for occurrence in group["occurrences"]:
            if occurrence["material"] is not None:
                material_mass_by_id[occurrence["material"]] += occurrence[
                    "estimated_mass_kg"
                ]
        non_motor_mass_by_link[group["proposed_link"]] += (
            group["estimated_material_mass_kg"] or 0.0
        ) + group.get("measured_mass_kg", 0.0)
    bilateral_group_pairs, unpaired_bilateral_groups = bilateral_group_inventory(groups)
    bilateral_mass_audit = []
    for left_link in sorted(
        link for link in non_motor_mass_by_link if link.startswith("left_")
    ):
        right_link = "right_" + left_link.removeprefix("left_")
        left_mass = non_motor_mass_by_link[left_link]
        right_mass = non_motor_mass_by_link.get(right_link, 0.0)
        residual = abs(left_mass - right_mass)
        bilateral_mass_audit.append(
            {
                "left_link": left_link,
                "left_mass_kg": left_mass,
                "right_link": right_link,
                "right_mass_kg": right_mass,
                "absolute_residual_kg": residual,
                "status": "review" if residual > 0.01 else "close",
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "review_contract": REVIEW_CONTRACT,
        "review_stage": "link_ownership",
        "source": manifest["source"],
        "source_manifests": {
            "cad_manifest_sha256": manifest_sha256,
            "classification_review_sha256": classification_sha256,
            "resolved_joint_frames_sha256": frames_sha256,
        },
        "root": frames["root"],
        "coordinate_frame": frames["coordinate_frame"],
        "links": link_inventory(frames),
        "joints": frames["joints"],
        "motors": frames["motors"],
        "chains": frames["chains"],
        "ownership_policy": {
            "catalog_motor": "whole catalog motor mass on parent/stator side",
            "remote_ankle_motors": "both RS02 housings on knee link; no shaft joint",
            "ankle_pitch_link": "massless virtual serial coordinate",
            "anonymous_end_effector": "CAD hands excluded; fixed sphere added later",
            "structure": "material-derived CAD mass assigned by reviewed rigid ownership",
        },
        "groups": groups,
        "bilateral_non_motor_mass_audit": bilateral_mass_audit,
        "bilateral_group_pairs": bilateral_group_pairs,
        "unpaired_bilateral_group_ids": unpaired_bilateral_groups,
        "summary": {
            **counts,
            "ownership_group_count": len(groups),
            "structure_group_count": sum(
                group["kind"] == "structure" for group in groups
            ),
            "catalog_motor_group_count": sum(
                group["kind"] == "catalog_motor" for group in groups
            ),
            "measured_assembly_group_count": sum(
                group["kind"] == "measured_assembly" for group in groups
            ),
            "link_group_counts": dict(sorted(link_counts.items())),
            "estimated_material_mass_kg": material_mass,
            "measured_assembly_mass_kg": measured_mass,
            "material_mass_kg": dict(sorted(material_mass_by_id.items())),
            "proposed_non_motor_mass_by_link_kg": dict(
                sorted(non_motor_mass_by_link.items())
            ),
            "maximum_bilateral_non_motor_mass_residual_kg": max(
                (pair["absolute_residual_kg"] for pair in bilateral_mass_audit),
                default=0.0,
            ),
            "bilateral_link_pairs_requiring_review": sum(
                pair["status"] == "review" for pair in bilateral_mass_audit
            ),
            "bilateral_group_pair_count": len(bilateral_group_pairs),
            "unpaired_bilateral_group_count": len(unpaired_bilateral_groups),
        },
    }


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    classification_path = args.classification_review.expanduser().resolve()
    frames_path = args.frames.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    classification = json.loads(classification_path.read_text(encoding="utf-8"))
    frames = json.loads(frames_path.read_text(encoding="utf-8"))
    result = build_manifest(
        manifest,
        classification,
        frames,
        manifest_sha256=file_sha256(manifest_path),
        classification_sha256=file_sha256(classification_path),
        frames_sha256=file_sha256(frames_path),
    )
    atomic_write_json(args.output, result)
    print(f"wrote link ownership review: {args.output}")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
