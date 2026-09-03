"""Build Bello Mini link inertials from approved CAD ownership."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = 1
INPUT_CONTRACT = "bello_mini_resolved_link_ownership_v1"
OUTPUT_CONTRACT = "bello_mini_inertial_manifest_v1"
CATALOG_SOURCE = {
    "url": "https://github.com/RobStride/Product_Information",
    "revision": "6ad12f50006273b7ea4eea88980f927d97c22f0d",
    "specification_date": "2026-07-13",
}
MOTOR_MODELS = {
    "ROBSTRIDE_00": {"mass_kg": 0.310, "mass_tolerance_kg": 0.003},
    "ROBSTRIDE_02": {"mass_kg": 0.380, "mass_tolerance_kg": 0.003},
    "ROBSTRIDE_03": {"mass_kg": 0.900, "mass_tolerance_kg": 0.020},
    "ROBSTRIDE_04": {"mass_kg": 1.420, "mass_tolerance_kg": 0.020},
    "ROBSTRIDE_06": {"mass_kg": 0.621, "mass_tolerance_kg": None},
}
# CAD: +x left, +y up, +z forward. URDF: +x forward, +y left, +z up.
CAD_TO_URDF = np.asarray(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ownership", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
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


def column_major_matrix(values: list[float]) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64).reshape((4, 4), order="F")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
        raise ValueError("occurrence transform is not orthogonal")
    if not math.isclose(abs(float(np.linalg.det(rotation))), 1.0, abs_tol=1e-8):
        raise ValueError("occurrence transform is not rigid")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-10):
        raise ValueError("occurrence transform is not affine")
    return matrix


def parallel_axis(mass: float, offset: np.ndarray) -> np.ndarray:
    return mass * (float(offset @ offset) * np.eye(3) - np.outer(offset, offset))


def combine_mass_properties(
    components: Iterable[tuple[float, np.ndarray, np.ndarray]],
) -> tuple[float, np.ndarray, np.ndarray]:
    values = list(components)
    mass = sum(component_mass for component_mass, _, _ in values)
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError(f"combined mass must be positive, got {mass}")
    center = (
        sum(
            component_mass * component_center
            for component_mass, component_center, _ in values
        )
        / mass
    )
    inertia = sum(
        component_inertia + parallel_axis(component_mass, component_center - center)
        for component_mass, component_center, component_inertia in values
    )
    return mass, center, inertia


def inertia_attributes(inertia: np.ndarray) -> dict[str, float]:
    return {
        "ixx": float(inertia[0, 0]),
        "iyy": float(inertia[1, 1]),
        "izz": float(inertia[2, 2]),
        "ixy": float(inertia[0, 1]),
        "ixz": float(inertia[0, 2]),
        "iyz": float(inertia[1, 2]),
    }


def validate_inertia(inertia: np.ndarray, *, name: str) -> None:
    if not np.all(np.isfinite(inertia)):
        raise ValueError(f"{name} inertia is nonfinite")
    if not np.allclose(inertia, inertia.T, atol=1e-12):
        raise ValueError(f"{name} inertia is not symmetric")
    principal = np.linalg.eigvalsh(inertia)
    if np.any(principal <= 0.0):
        raise ValueError(f"{name} inertia is not positive definite: {principal}")
    if principal[-1] > principal[0] + principal[1] + 1e-10:
        raise ValueError(
            f"{name} principal moments violate triangle inequality: {principal}"
        )


def source_shapes(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        obj["name"]: obj["shape"]
        for obj in inventory["objects"]
        if obj.get("shape") is not None
    }


def has_solid_mass_properties(shape: dict[str, Any]) -> bool:
    return (
        int(shape["solid_count"]) > 0
        and float(shape["volume_mm3"]) > 0.0
        and shape.get("center_of_mass_mm") is not None
        and shape.get("matrix_of_inertia_mm5") is not None
    )


def occurrence_mass_properties(
    occurrence: dict[str, Any],
    shape: dict[str, Any],
    mass_kg: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    volume_mm3 = float(shape["volume_mm3"])
    center_mm = shape.get("center_of_mass_mm")
    inertia_mm5 = shape.get("matrix_of_inertia_mm5")
    if not has_solid_mass_properties(shape):
        bounds = occurrence["bounds_mm"]
        size_m = (np.asarray(bounds["max"]) - np.asarray(bounds["min"])) * 1e-3
        if np.any(size_m <= 0.0):
            raise ValueError(
                f"{occurrence['id']} has no solid properties or positive bounds"
            )
        center_world_m = (
            np.asarray(bounds["min"]) + np.asarray(bounds["max"])
        ) * 0.5e-3
        x, y, z = size_m
        inertia_world = np.diag(
            [
                mass_kg * (y * y + z * z) / 12.0,
                mass_kg * (x * x + z * z) / 12.0,
                mass_kg * (x * x + y * y) / 12.0,
            ]
        )
        return mass_kg, center_world_m, inertia_world
    transform = column_major_matrix(occurrence["transform"])
    rotation = transform[:3, :3]
    center_world_mm = rotation @ np.asarray(center_mm) + transform[:3, 3]
    # FreeCAD reports the volume integral in mm^5. Scaling it by mass/volume
    # gives kg*mm^2; 1e-6 then converts that tensor to kg*m^2.
    inertia_local = np.asarray(inertia_mm5) * (mass_kg / volume_mm3) * 1e-6
    inertia_world = rotation @ inertia_local @ rotation.T
    return mass_kg, center_world_mm * 1e-3, inertia_world


def group_mass_properties(
    group: dict[str, Any],
    shapes: dict[str, dict[str, Any]],
) -> tuple[float, np.ndarray, np.ndarray]:
    occurrences = group["occurrences"]
    volumes = [
        (
            float(shapes[value["source_name"]]["volume_mm3"])
            if has_solid_mass_properties(shapes[value["source_name"]])
            else 0.0
        )
        for value in occurrences
    ]
    if group["kind"] == "structure":
        masses = [float(value["estimated_mass_kg"]) for value in occurrences]
    else:
        if group["kind"] == "catalog_motor":
            model = group["motor_model"]
            try:
                target_mass = MOTOR_MODELS[model]["mass_kg"]
            except KeyError as error:
                raise ValueError(f"no catalog mass for motor model {model}") from error
        elif group["kind"] == "measured_assembly":
            target_mass = float(group["measured_mass_kg"])
        elif group["kind"] == "catalog_component":
            target_mass = float(group["catalog_mass_kg"])
        else:
            raise ValueError(f"unsupported ownership group kind {group['kind']}")
        total_volume = sum(volumes)
        if total_volume <= 0.0:
            raise ValueError(f"{group['id']} has no positive CAD volume")
        masses = [target_mass * volume / total_volume for volume in volumes]
    components = [
        occurrence_mass_properties(occurrence, shapes[occurrence["source_name"]], mass)
        for occurrence, mass in zip(occurrences, masses, strict=True)
        if mass > 0.0
    ]
    if not components:
        if group["kind"] != "structure" or any(mass > 0.0 for mass in masses):
            raise ValueError(f"{group['id']} has mass but no solid CAD geometry")
        return 0.0, np.zeros(3), np.zeros((3, 3))
    return combine_mass_properties(components)


def link_origins(ownership: dict[str, Any]) -> dict[str, np.ndarray]:
    result = {"pelvis_link": np.zeros(3)}
    for joint in ownership["joints"]:
        child = joint["child_link"]
        if child in result:
            raise ValueError(f"duplicate incoming joint for {child}")
        result[child] = np.asarray(joint["origin_cad_mm"], dtype=np.float64) * 1e-3
    expected = {link["name"] for link in ownership["links"]}
    if set(result) != expected:
        raise ValueError(
            f"link-frame inventory mismatch: missing={sorted(expected - set(result))}, "
            f"unexpected={sorted(set(result) - expected)}"
        )
    return result


def build_manifest(
    ownership: dict[str, Any],
    inventory: dict[str, Any],
    *,
    ownership_sha256: str,
    inventory_sha256: str,
) -> dict[str, Any]:
    if ownership.get("review_contract") != INPUT_CONTRACT:
        raise ValueError("ownership input has not passed the resolved review gate")
    if ownership.get("source", {}).get("sha256") != inventory.get("source", {}).get(
        "sha256"
    ):
        raise ValueError("ownership and inventory refer to different STEP sources")

    shapes = source_shapes(inventory)
    group_records = []
    link_components: dict[str, list[tuple[float, np.ndarray, np.ndarray]]] = (
        defaultdict(list)
    )
    kind_mass: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    model_mass: Counter[str] = Counter()
    non_solid_occurrence_count = 0
    structural_aabb_fallback_count = 0
    zero_mass_group_count = 0
    for group in ownership["groups"]:
        missing_sources = sorted(
            {
                occurrence["source_name"]
                for occurrence in group["occurrences"]
                if occurrence["source_name"] not in shapes
            }
        )
        if missing_sources:
            raise ValueError(
                f"{group['id']} references missing shapes: {missing_sources}"
            )
        mass, center, inertia = group_mass_properties(group, shapes)
        non_solid_occurrence_count += sum(
            not has_solid_mass_properties(shapes[occurrence["source_name"]])
            for occurrence in group["occurrences"]
        )
        if group["kind"] == "structure":
            structural_aabb_fallback_count += sum(
                not has_solid_mass_properties(shapes[occurrence["source_name"]])
                and float(occurrence["estimated_mass_kg"]) > 0.0
                for occurrence in group["occurrences"]
            )
        link = group["reviewed_link"]
        if mass > 0.0:
            validate_inertia(inertia, name=group["id"])
            link_components[link].append((mass, center, inertia))
        else:
            zero_mass_group_count += 1
        kind_mass[group["kind"]] += mass
        if group["kind"] == "catalog_motor":
            model_counts[group["motor_model"]] += 1
            model_mass[group["motor_model"]] += mass
        group_records.append(
            {
                "id": group["id"],
                "kind": group["kind"],
                "link": link,
                "mass_kg": mass,
                "center_cad_world_m": center.tolist(),
                "inertia_cad_world_com_kg_m2": inertia_attributes(inertia),
                **(
                    {
                        "motor_id": group["motor_id"],
                        "motor_model": group["motor_model"],
                        "motor_name": group["motor_name"],
                    }
                    if group["kind"] == "catalog_motor"
                    else {}
                ),
                **(
                    {
                        "catalog_model": group["catalog_model"],
                        "catalog_mass_tolerance_kg": group[
                            "catalog_mass_tolerance_kg"
                        ],
                        "catalog_dimensions_m": group["catalog_dimensions_m"],
                        "catalog_source": group["catalog_source"],
                    }
                    if group["kind"] == "catalog_component"
                    else {}
                ),
            }
        )

    origins = link_origins(ownership)
    link_records = []
    for link in ownership["links"]:
        name = link["name"]
        origin_cad = origins[name]
        components = link_components.get(name, [])
        if not components:
            if link["physical_geometry_policy"] != "virtual_massless_serial_coordinate":
                raise ValueError(f"physical link {name} has no mass contribution")
            link_records.append(
                {
                    "name": name,
                    "physical_geometry_policy": link["physical_geometry_policy"],
                    "frame_origin_cad_world_m": origin_cad.tolist(),
                    "frame_origin_urdf_world_m": (CAD_TO_URDF @ origin_cad).tolist(),
                    "mass_kg": 0.0,
                    "center_urdf_link_m": [0.0, 0.0, 0.0],
                    "inertia_urdf_link_com_kg_m2": None,
                }
            )
            continue
        mass, center_cad, inertia_cad = combine_mass_properties(components)
        center_link_urdf = CAD_TO_URDF @ (center_cad - origin_cad)
        inertia_urdf = CAD_TO_URDF @ inertia_cad @ CAD_TO_URDF.T
        validate_inertia(inertia_urdf, name=name)
        link_records.append(
            {
                "name": name,
                "physical_geometry_policy": link["physical_geometry_policy"],
                "frame_origin_cad_world_m": origin_cad.tolist(),
                "frame_origin_urdf_world_m": (CAD_TO_URDF @ origin_cad).tolist(),
                "mass_kg": mass,
                "center_urdf_link_m": center_link_urdf.tolist(),
                "inertia_urdf_link_com_kg_m2": inertia_attributes(inertia_urdf),
            }
        )

    modeled_mass = sum(link["mass_kg"] for link in link_records)
    expected_mass = sum(kind_mass.values())
    if not math.isclose(modeled_mass, expected_mass, abs_tol=1e-10):
        raise ValueError(f"link mass does not close: {modeled_mass} != {expected_mass}")
    non_motor_mass = (
        kind_mass["structure"]
        + kind_mass["measured_assembly"]
        + kind_mass["catalog_component"]
    )
    if not math.isclose(
        non_motor_mass,
        ownership["summary"]["resolved_non_motor_mass_kg"],
        abs_tol=1e-9,
    ):
        raise ValueError("inertial structural/measured mass does not match ownership")

    return {
        "schema_version": SCHEMA_VERSION,
        "review_contract": OUTPUT_CONTRACT,
        "source": ownership["source"],
        "source_manifests": {
            **ownership["source_manifests"],
            "resolved_link_ownership_sha256": ownership_sha256,
            "freecad_inventory_sha256": inventory_sha256,
        },
        "coordinate_frame": {
            "cad_axes": {"x": "left", "y": "up", "z": "forward"},
            "urdf_axes": {"x": "forward", "y": "left", "z": "up"},
            "cad_to_urdf_rotation": CAD_TO_URDF.tolist(),
            "link_frame_policy": "world-aligned at approved zero pose; origin at incoming joint axis",
        },
        "motor_catalog": {
            "source": CATALOG_SOURCE,
            "models": MOTOR_MODELS,
            "inertia_policy": "exact in-assembly CAD geometry scaled uniformly to official catalog mass",
        },
        "component_catalog": {
            group["catalog_model"]: {
                "mass_kg": group["catalog_mass_kg"],
                "mass_tolerance_kg": group["catalog_mass_tolerance_kg"],
                "dimensions_m": group["catalog_dimensions_m"],
                "source": group["catalog_source"],
                "inertia_policy": (
                    "exact in-assembly CAD geometry scaled uniformly to official "
                    "catalog mass"
                ),
            }
            for group in ownership["groups"]
            if group["kind"] == "catalog_component"
        },
        "groups": group_records,
        "links": link_records,
        "summary": {
            "link_count": len(link_records),
            "physical_link_count": sum(link["mass_kg"] > 0.0 for link in link_records),
            "virtual_massless_link_count": sum(
                link["mass_kg"] == 0.0 for link in link_records
            ),
            "non_solid_occurrence_count": non_solid_occurrence_count,
            "structural_aabb_fallback_count": structural_aabb_fallback_count,
            "zero_mass_structure_group_count": zero_mass_group_count,
            "structure_mass_kg": kind_mass["structure"],
            "measured_assembly_mass_kg": kind_mass["measured_assembly"],
            "catalog_component_mass_kg": kind_mass["catalog_component"],
            "catalog_motor_mass_kg": kind_mass["catalog_motor"],
            "total_mass_kg": modeled_mass,
            "motor_model_counts": dict(sorted(model_counts.items())),
            "motor_model_mass_kg": dict(sorted(model_mass.items())),
        },
    }


def main() -> None:
    args = parse_args()
    ownership_path = args.ownership.expanduser().resolve()
    inventory_path = args.inventory.expanduser().resolve()
    ownership = json.loads(ownership_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    result = build_manifest(
        ownership,
        inventory,
        ownership_sha256=file_sha256(ownership_path),
        inventory_sha256=file_sha256(inventory_path),
    )
    atomic_write_json(args.output, result)
    print(f"wrote inertial manifest: {args.output}")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
