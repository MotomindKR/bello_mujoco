"""Resolve reviewed Bello Mini CAD ownership and explicit symmetry completion."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = 1
INPUT_CONTRACT = "bello_mini_link_ownership_v1"
RESOLVED_CONTRACT = "bello_mini_resolved_link_ownership_v1"
MIRRORED_GROUP_ID = "ownership-010"
TRANSLATED_GROUP_IDS = (
    "ownership-021",
    "ownership-022",
    "ownership-023",
    "ownership-024",
    "ownership-028",
    "ownership-029",
)
EXPECTED_COMPLETION_GROUP_LABELS = {
    "ownership-010": "Solid038",
    "ownership-021": "Link085",
    "ownership-022": "Link086",
    "ownership-023": "Link087",
    "ownership-024": "Link088",
    "ownership-028": "LinkGroup024",
    "ownership-029": "Solid067",
}
CATALOG_COMPONENT_OVERRIDES = (
    {
        "path_prefix": "/LinkGroup2989/LinkGroup025/LinkGroup018",
        "label": "LinkGroup018",
        "reviewed_link": "pelvis_link",
        "occurrence_source_names": {"Compound007", "Compound008"},
        "catalog_model": "ORBBEC_ASTRA_STEREO_S_U3",
        "catalog_mass_kg": 0.030,
        "catalog_mass_tolerance_kg": 0.005,
        "catalog_dimensions_m": [0.0653, 0.0225, 0.0123],
        "catalog_source": {
            "url": "https://store.orbbec.com/products/astra-stereo-s-u3",
            "specification_date": "2026-09-03",
        },
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
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


def mirror_link(link: str) -> str:
    if link.startswith("left_"):
        return "right_" + link.removeprefix("left_")
    if link.startswith("right_"):
        return "left_" + link.removeprefix("right_")
    return link


def validate_review(manifest: dict[str, Any], review: dict[str, Any]) -> None:
    if manifest.get("review_contract") != INPUT_CONTRACT:
        raise ValueError("input is not a Bello Mini link-ownership manifest")
    if review.get("review_contract") != manifest["review_contract"]:
        raise ValueError("link-ownership review contract does not match manifest")
    if review.get("source", {}).get("sha256") != manifest["source"]["sha256"]:
        raise ValueError("link-ownership review STEP checksum does not match manifest")
    if review.get("source_manifests") != manifest.get("source_manifests"):
        raise ValueError("link-ownership review uses different upstream manifests")

    decisions = review.get("ownership_decisions")
    if not isinstance(decisions, dict):
        raise ValueError("ownership decisions must be a mapping")
    expected_ids = {group["id"] for group in manifest["groups"]}
    if set(decisions) != expected_ids:
        missing = sorted(expected_ids - set(decisions))
        unexpected = sorted(set(decisions) - expected_ids)
        raise ValueError(
            f"ownership decision inventory mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    unresolved = {
        group_id: decision
        for group_id, decision in decisions.items()
        if decision.get("status") != "confirmed"
    }
    if unresolved:
        raise ValueError(f"unresolved ownership decisions: {unresolved}")

    physical_links = {
        link["name"]
        for link in manifest["links"]
        if link["physical_geometry_policy"] != "virtual_massless_serial_coordinate"
    }
    invalid_links = {
        group_id: decision.get("link")
        for group_id, decision in decisions.items()
        if decision.get("link") not in physical_links
    }
    if invalid_links:
        raise ValueError(f"invalid physical owner links: {invalid_links}")

    mismatches = []
    for pair in manifest.get("bilateral_group_pairs", []):
        left_link = decisions[pair["left_group_id"]]["link"]
        right_link = decisions[pair["right_group_id"]]["link"]
        if mirror_link(left_link) != right_link:
            mismatches.append(
                {
                    "left_group_id": pair["left_group_id"],
                    "left_link": left_link,
                    "right_group_id": pair["right_group_id"],
                    "right_link": right_link,
                }
            )
    if mismatches:
        raise ValueError(f"bilateral ownership decisions do not match: {mismatches}")


def translated_transform(
    transform: list[float], translation_mm: list[float]
) -> list[float]:
    result = [float(value) for value in transform]
    for axis, offset in enumerate(translation_mm):
        result[12 + axis] += offset
    return result


def mirrored_x_transform(transform: list[float]) -> list[float]:
    result = [float(value) for value in transform]
    for index in (0, 4, 8, 12):
        result[index] *= -1.0
    return result


def translated_bounds(
    bounds: dict[str, list[float]], translation_mm: list[float]
) -> dict[str, list[float]]:
    return {
        key: [value + translation_mm[axis] for axis, value in enumerate(values)]
        for key, values in bounds.items()
    }


def mirrored_x_bounds(bounds: dict[str, list[float]]) -> dict[str, list[float]]:
    return {
        "min": [-bounds["max"][0], bounds["min"][1], bounds["min"][2]],
        "max": [-bounds["min"][0], bounds["max"][1], bounds["max"][2]],
    }


def union_bounds(values: list[dict[str, list[float]]]) -> dict[str, list[float]]:
    return {
        "min": [min(value["min"][axis] for value in values) for axis in range(3)],
        "max": [max(value["max"][axis] for value in values) for axis in range(3)],
    }


def synthesize_group(
    source: dict[str, Any],
    *,
    operation_id: str,
    destination_link: str,
    transform_operation: Callable[[list[float]], list[float]],
    bounds_operation: Callable[[dict[str, list[float]]], dict[str, list[float]]],
    operation: dict[str, Any],
) -> dict[str, Any]:
    result = deepcopy(source)
    result["id"] = f"synthetic-{operation_id}-{source['id']}"
    result["label"] = f"Synthetic right copy of {source['label']}"
    result["path_prefix"] = f"/synthetic/{operation_id}/{source['id']}"
    result["source_group_id"] = source["id"]
    result["side"] = "right"
    result["source_proposed_link"] = source.get("source_proposed_link")
    result["proposed_link"] = destination_link
    result["reviewed_link"] = destination_link
    result["ownership_review_status"] = "synthetic_user_directed"
    result["proposal_reason"] = "explicit user-directed CAD symmetry completion"
    result["synthetic_operation"] = operation
    result.pop("selection_instances", None)
    for index, occurrence in enumerate(result["occurrences"], start=1):
        occurrence["id"] = f"{result['path_prefix']}/occurrence-{index:02d}"
        occurrence["transform"] = transform_operation(occurrence["transform"])
        occurrence["bounds_mm"] = bounds_operation(occurrence["bounds_mm"])
    result["bounds_mm"] = union_bounds(
        [occurrence["bounds_mm"] for occurrence in result["occurrences"]]
    )
    result["center_mm"] = [
        (minimum + maximum) / 2.0
        for minimum, maximum in zip(
            result["bounds_mm"]["min"], result["bounds_mm"]["max"], strict=True
        )
    ]
    return result


def ankle_translation(frames: dict[str, Any]) -> list[float]:
    joints = {joint["name"]: joint for joint in frames["joints"]}
    left = joints["left_ankle_pitch_joint"]["origin_cad_mm"]
    right = joints["right_ankle_pitch_joint"]["origin_cad_mm"]
    translation = [right[index] - left[index] for index in range(3)]
    if translation[0] >= 0.0 or math.hypot(translation[1], translation[2]) > 1e-3:
        raise ValueError(f"unexpected bilateral ankle translation: {translation}")
    return translation


def validate_completion_sources(groups_by_id: dict[str, dict[str, Any]]) -> None:
    problems = []
    for group_id, expected_label in EXPECTED_COMPLETION_GROUP_LABELS.items():
        group = groups_by_id.get(group_id)
        if group is None:
            problems.append(f"{group_id}: missing")
            continue
        if group.get("label") != expected_label:
            problems.append(
                f"{group_id}: expected label {expected_label!r}, "
                f"got {group.get('label')!r}"
            )
        if group.get("kind") != "structure":
            problems.append(
                f"{group_id}: expected structure, got {group.get('kind')!r}"
            )
        if group.get("side") != "left":
            problems.append(
                f"{group_id}: expected left side, got {group.get('side')!r}"
            )
        if not group.get("reviewed_link", "").startswith("left_"):
            problems.append(
                f"{group_id}: expected a reviewed left owner link, "
                f"got {group.get('reviewed_link')!r}"
            )
    if problems:
        raise ValueError(
            "invalid CAD symmetry-completion sources: " + "; ".join(problems)
        )


def apply_catalog_component_overrides(groups: list[dict[str, Any]]) -> None:
    for override in CATALOG_COMPONENT_OVERRIDES:
        matches = [
            group
            for group in groups
            if group["path_prefix"] == override["path_prefix"]
            and group["label"] == override["label"]
        ]
        if len(matches) != 1:
            raise ValueError(
                "catalog component override must match exactly one reviewed group: "
                f"{override['catalog_model']} matched {len(matches)}"
            )
        group = matches[0]
        if group["kind"] != "structure":
            raise ValueError(
                f"catalog component {override['catalog_model']} expected structure, "
                f"got {group['kind']}"
            )
        if group["reviewed_link"] != override["reviewed_link"]:
            raise ValueError(
                f"catalog component {override['catalog_model']} expected owner "
                f"{override['reviewed_link']}, got {group['reviewed_link']}"
            )
        source_names = {
            occurrence["source_name"] for occurrence in group["occurrences"]
        }
        if source_names != override["occurrence_source_names"]:
            raise ValueError(
                f"catalog component {override['catalog_model']} CAD sources changed: "
                f"{sorted(source_names)}"
            )
        group["kind"] = "catalog_component"
        group["catalog_model"] = override["catalog_model"]
        group["catalog_mass_kg"] = override["catalog_mass_kg"]
        group["catalog_mass_tolerance_kg"] = override[
            "catalog_mass_tolerance_kg"
        ]
        group["catalog_dimensions_m"] = override["catalog_dimensions_m"]
        group["catalog_source"] = override["catalog_source"]
        group["replaced_material_mass_kg"] = group[
            "estimated_material_mass_kg"
        ]


def group_mass_kg(group: dict[str, Any]) -> float:
    if group["kind"] == "catalog_component":
        return float(group["catalog_mass_kg"])
    return float(group.get("estimated_material_mass_kg") or 0.0) + float(
        group.get("measured_mass_kg", 0.0)
    )


def mass_summary(groups: list[dict[str, Any]]) -> dict[str, Any]:
    material_mass: Counter[str] = Counter()
    link_mass: Counter[str] = Counter()
    link_group_counts: Counter[str] = Counter()
    measured_mass = 0.0
    for group in groups:
        link = group["reviewed_link"]
        link_group_counts[link] += 1
        group_mass = group_mass_kg(group)
        link_mass[link] += group_mass
        measured_mass += group.get("measured_mass_kg", 0.0)
        if group["kind"] == "catalog_component":
            continue
        for occurrence in group["occurrences"]:
            if occurrence["material"] is not None:
                material_mass[occurrence["material"]] += occurrence["estimated_mass_kg"]
    bilateral = []
    for left_link in sorted(link for link in link_mass if link.startswith("left_")):
        right_link = mirror_link(left_link)
        residual = abs(link_mass[left_link] - link_mass.get(right_link, 0.0))
        bilateral.append(
            {
                "absolute_residual_kg": residual,
                "left_link": left_link,
                "left_mass_kg": link_mass[left_link],
                "right_link": right_link,
                "right_mass_kg": link_mass.get(right_link, 0.0),
            }
        )
    synthetic_groups = [group for group in groups if group.get("source_group_id")]
    synthetic_mass = sum(group_mass_kg(group) for group in synthetic_groups)
    catalog_components = [
        group for group in groups if group["kind"] == "catalog_component"
    ]
    return {
        "ownership_group_count": len(groups),
        "structure_group_count": sum(group["kind"] == "structure" for group in groups),
        "catalog_motor_group_count": sum(
            group["kind"] == "catalog_motor" for group in groups
        ),
        "catalog_component_group_count": len(catalog_components),
        "catalog_component_mass_kg": sum(
            group_mass_kg(group) for group in catalog_components
        ),
        "catalog_component_model_counts": dict(
            sorted(Counter(group["catalog_model"] for group in catalog_components).items())
        ),
        "measured_assembly_group_count": sum(
            group["kind"] == "measured_assembly" for group in groups
        ),
        "synthetic_group_count": len(synthetic_groups),
        "synthetic_occurrence_count": sum(
            group["occurrence_count"] for group in synthetic_groups
        ),
        "synthetic_material_mass_kg": synthetic_mass,
        "estimated_material_mass_kg": sum(material_mass.values()),
        "measured_assembly_mass_kg": measured_mass,
        "resolved_non_motor_mass_kg": sum(link_mass.values()),
        "material_mass_kg": dict(sorted(material_mass.items())),
        "resolved_non_motor_mass_by_link_kg": dict(sorted(link_mass.items())),
        "link_group_counts": dict(sorted(link_group_counts.items())),
        "bilateral_non_motor_mass_audit": bilateral,
        "maximum_bilateral_non_motor_mass_residual_kg": max(
            (pair["absolute_residual_kg"] for pair in bilateral), default=0.0
        ),
    }


def resolve_ownership(
    manifest: dict[str, Any],
    review: dict[str, Any],
    *,
    manifest_sha256: str,
    review_sha256: str,
) -> dict[str, Any]:
    validate_review(manifest, review)
    result = deepcopy(manifest)
    decisions = review["ownership_decisions"]
    for group in result["groups"]:
        group["source_proposed_link"] = group["proposed_link"]
        group["reviewed_link"] = decisions[group["id"]]["link"]
        group["proposed_link"] = group["reviewed_link"]
        group["ownership_review_status"] = "reviewed_confirmed"

    apply_catalog_component_overrides(result["groups"])

    groups_by_id = {group["id"]: group for group in result["groups"]}
    validate_completion_sources(groups_by_id)
    translation = ankle_translation(result)
    operation_records = []

    mirror_source = groups_by_id[MIRRORED_GROUP_ID]
    mirror_destination = mirror_link(mirror_source["reviewed_link"])
    mirror_operation = {
        "id": "mirror-x",
        "mode": "reflect",
        "plane": "CAD x=0",
        "preserves_handedness": False,
        "source_group_id": MIRRORED_GROUP_ID,
        "destination_link": mirror_destination,
    }
    mirrored = synthesize_group(
        mirror_source,
        operation_id="mirror-x",
        destination_link=mirror_destination,
        transform_operation=mirrored_x_transform,
        bounds_operation=mirrored_x_bounds,
        operation=mirror_operation,
    )
    result["groups"].append(mirrored)
    operation_records.append({**mirror_operation, "synthetic_group_id": mirrored["id"]})

    for source_group_id in TRANSLATED_GROUP_IDS:
        source = groups_by_id[source_group_id]
        destination = mirror_link(source["reviewed_link"])
        operation_id = "translate-right-ankle-spacing"
        operation = {
            "id": operation_id,
            "mode": "translate_without_reflection",
            "translation_cad_mm": translation,
            "distance_mm": abs(translation[0]),
            "preserves_handedness": True,
            "source_group_id": source_group_id,
            "destination_link": destination,
        }
        synthetic = synthesize_group(
            source,
            operation_id=operation_id,
            destination_link=destination,
            transform_operation=lambda value, offset=translation: translated_transform(
                value, offset
            ),
            bounds_operation=lambda value, offset=translation: translated_bounds(
                value, offset
            ),
            operation=operation,
        )
        result["groups"].append(synthetic)
        operation_records.append({**operation, "synthetic_group_id": synthetic["id"]})

    result["groups"].sort(key=lambda group: group["id"])
    completion_pairs = [
        {
            "evidence": operation["mode"],
            "kind": "structure",
            "left_group_id": operation["source_group_id"],
            "right_group_id": operation["synthetic_group_id"],
            "spatial_residual_mm": 0.0,
        }
        for operation in operation_records
    ]
    result["bilateral_group_pairs"] = (
        result.get("bilateral_group_pairs", []) + completion_pairs
    )
    completed_source_ids = {
        operation["source_group_id"] for operation in operation_records
    }
    result["unpaired_bilateral_group_ids"] = [
        group_id
        for group_id in result.get("unpaired_bilateral_group_ids", [])
        if group_id not in completed_source_ids
    ]
    summary = mass_summary(result["groups"])
    result["bilateral_non_motor_mass_audit"] = summary["bilateral_non_motor_mass_audit"]
    summary.update(
        {
            "source_included_occurrence_count": manifest["summary"][
                "included_occurrence_count"
            ],
            "modeled_occurrence_count": manifest["summary"]["included_occurrence_count"]
            + summary["synthetic_occurrence_count"],
            "excluded_hand_occurrence_count": manifest["summary"][
                "excluded_hand_occurrence_count"
            ],
            "bilateral_group_pair_count": len(result["bilateral_group_pairs"]),
            "unpaired_bilateral_group_count": len(
                result["unpaired_bilateral_group_ids"]
            ),
            "bilateral_link_pairs_requiring_review": sum(
                pair["absolute_residual_kg"] > 0.01
                for pair in summary["bilateral_non_motor_mass_audit"]
            ),
        }
    )
    result["schema_version"] = SCHEMA_VERSION
    result["review_contract"] = RESOLVED_CONTRACT
    result["review_stage"] = "resolved_link_ownership"
    result["source_manifests"] = {
        **result["source_manifests"],
        "link_ownership_manifest_sha256": manifest_sha256,
        "link_ownership_review_sha256": review_sha256,
    }
    result["approved_ownership_review"] = {
        "review_contract": review["review_contract"],
        "sha256": review_sha256,
    }
    result["symmetry_completion"] = {
        "ankle_translation_cad_mm": translation,
        "operations": operation_records,
        "policy": "one reflected hip part; six ankle-linkage groups translated by bilateral ankle spacing without reflection",
    }
    result["ownership_policy"]["cad_symmetry_completion"] = result[
        "symmetry_completion"
    ]["policy"]
    result["ownership_policy"]["catalog_component"] = (
        "whole manufacturer mass lumped into the reviewed rigid owner link"
    )
    result["summary"] = summary
    return result


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    review_path = args.review.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    result = resolve_ownership(
        manifest,
        review,
        manifest_sha256=file_sha256(manifest_path),
        review_sha256=file_sha256(review_path),
    )
    atomic_write_json(args.output, result)
    print(f"wrote resolved link ownership: {args.output}")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
