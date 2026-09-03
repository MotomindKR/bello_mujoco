"""Build a source-bound CAD review manifest from a FreeCAD inventory."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
MATERIALS = [
    {"id": "aluminum_6061_t6", "label": "Aluminum 6061-T6", "density_kg_m3": 2700},
    {"id": "aluminum_7075_t6", "label": "Aluminum 7075-T6", "density_kg_m3": 2810},
    {"id": "steel_4140", "label": "Steel 4140", "density_kg_m3": 7850},
    {"id": "stainless_304", "label": "Stainless steel 304", "density_kg_m3": 8000},
    {"id": "abs", "label": "ABS", "density_kg_m3": 1040},
    {"id": "pa12_cf", "label": "PA12-CF", "density_kg_m3": 1050},
    {"id": "tpu", "label": "TPU (solid equivalent)", "density_kg_m3": 1200},
    {"id": "rubber", "label": "Rubber", "density_kg_m3": 1100},
    {
        "id": "carbon_fiber_laminate",
        "label": "Carbon-fiber laminate",
        "density_kg_m3": 1550,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mesh-url-prefix", default="meshes")
    return parser.parse_args()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def matrix(obj: dict[str, Any]) -> np.ndarray:
    return np.asarray(obj["local_placement"]["matrix4"], dtype=np.float64)


def flattened_column_major(value: np.ndarray) -> list[float]:
    return [float(component) for component in value.T.reshape(-1)]


def transformed_bounds(
    bounds: dict[str, list[float]], transform: np.ndarray
) -> dict[str, list[float]]:
    minimum = bounds["min"]
    maximum = bounds["max"]
    corners = np.asarray(
        [
            [x, y, z, 1.0]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ]
    )
    transformed = (transform @ corners.T).T[:, :3]
    return {
        "min": transformed.min(axis=0).tolist(),
        "max": transformed.max(axis=0).tolist(),
    }


def bounds_size(bounds: dict[str, list[float]]) -> list[float]:
    return [
        maximum - minimum
        for minimum, maximum in zip(bounds["min"], bounds["max"], strict=True)
    ]


def motor_model(label: str) -> str | None:
    normalized = label.strip().lower()
    if "60n_m0001" in normalized:
        return "ROBSTRIDE_03"
    if "120waiguan0314" in normalized:
        return "ROBSTRIDE_04"
    if re.match(r"^rs00(?:\s|$)", normalized):
        return "ROBSTRIDE_00"
    if re.match(r"^rs02(?:\s|$)", normalized):
        return "ROBSTRIDE_02"
    if re.match(r"^rs03(?:\s|$)", normalized) and "spacer" not in normalized:
        return "ROBSTRIDE_03"
    if re.match(r"^rs06(?:\s|$)", normalized) and not any(
        word in normalized for word in ("spacer", "limiter", "mount")
    ):
        return "ROBSTRIDE_06"
    return None


def measured_assembly_kind(label: str) -> str | None:
    normalized = label.lower()
    if "ec-bat" in normalized or "battery" in normalized:
        return "battery"
    if label == "Board_Assm" or "assy_bello" in normalized:
        return "electronics"
    return None


def is_anonymous_end_effector_source(label: str) -> bool:
    return "asm_human_hand" in label.lower()


def recommendation(
    label: str,
    *,
    motor_internal: bool,
    measured_internal: bool,
    end_effector_excluded: bool,
) -> dict[str, str | None]:
    normalized = label.lower()
    if end_effector_excluded:
        return {
            "classification": "exclude",
            "material": None,
            "confidence": "high",
            "reason": "inside a CAD hand replaced by an anonymous end effector",
        }
    if motor_internal:
        return {
            "classification": "catalog_actuator_internal",
            "material": None,
            "confidence": "high",
            "reason": "contained by a candidate catalog actuator assembly",
        }
    if measured_internal:
        return {
            "classification": "measured_assembly_internal",
            "material": None,
            "confidence": "high",
            "reason": "contained by a battery or electronics assembly",
        }
    if any(
        word in normalized
        for word in (
            "screw",
            "bolt",
            "washer",
            " fastener",
            "hshcs",
            "너트",
        )
    ) or re.search(r"\bm\d+(?:x|\s*x\s*)\d+", normalized):
        return {
            "classification": "material",
            "material": "steel_4140",
            "confidence": "medium",
            "reason": "fastener designation",
        }
    if any(word in normalized for word in ("bearing", "shaft", "rod", "pin")):
        return {
            "classification": "material",
            "material": "steel_4140",
            "confidence": "medium",
            "reason": "bearing, shaft, rod, or pin designation",
        }
    if "sus" in normalized or "stainless" in normalized:
        return {
            "classification": "material",
            "material": "stainless_304",
            "confidence": "medium",
            "reason": "stainless designation",
        }
    if any(word in normalized for word in ("tpu", "rubber", "bumper", "pad")):
        return {
            "classification": "material",
            "material": "tpu",
            "confidence": "low",
            "reason": "soft-part designation; verify material and infill",
        }
    if any(
        word in normalized
        for word in ("cover", "cap", "case", "housing", "shell", "罩", "壳体")
    ):
        return {
            "classification": "material",
            "material": "abs",
            "confidence": "low",
            "reason": "cover or enclosure designation; verify fabrication",
        }
    if any(
        word in normalized
        for word in ("pcb", "board", "camera", "fan", "heatsink", "heatpipe")
    ):
        return {
            "classification": "measured_component",
            "material": None,
            "confidence": "medium",
            "reason": "COTS/electronics geometry should use measured component mass",
        }
    if any(word in normalized for word in ("mount", "frame", "plate", "arm")):
        return {
            "classification": "material",
            "material": "aluminum_6061_t6",
            "confidence": "medium",
            "reason": "structural mount, frame, plate, or arm designation",
        }
    return {
        "classification": "material",
        "material": "aluminum_6061_t6",
        "confidence": "low",
        "reason": "structural default; verify drawing or fabrication process",
    }


class AssemblyWalker:
    def __init__(self, inventory: dict[str, Any]) -> None:
        self.inventory = inventory
        self.by_name = {obj["name"]: obj for obj in inventory["objects"]}
        self.occurrences: list[dict[str, Any]] = []
        self.motor_assemblies: list[dict[str, Any]] = []
        self.measured_assemblies: list[dict[str, Any]] = []
        self.excluded_end_effectors: list[dict[str, Any]] = []

    def walk(self) -> None:
        roots = self.inventory["summary"]["root_objects"]
        if len(roots) != 1:
            raise ValueError(f"expected exactly one CAD root, got {roots}")
        self._walk(
            roots[0],
            np.eye(4),
            path="",
            top_branch=None,
            active_motor=None,
            active_measured=None,
            active_end_effector=None,
            stack=(),
        )

    def _walk(
        self,
        name: str,
        parent_transform: np.ndarray,
        *,
        path: str,
        top_branch: str | None,
        active_motor: int | None,
        active_measured: int | None,
        active_end_effector: int | None,
        stack: tuple[str, ...],
    ) -> None:
        if name in stack:
            raise ValueError(f"CAD reference cycle: {'/'.join((*stack, name))}")
        obj = self.by_name[name]
        object_path = f"{path}/{name}"
        label = obj["label"] or name
        node_shape_transform = parent_transform

        if obj["type_id"] == "App::LinkGroup":
            node_shape_transform = parent_transform
        elif obj["type_id"] == "App::Link":
            node_shape_transform = parent_transform

        model = motor_model(label)
        if active_motor is None and model is not None:
            active_motor = len(self.motor_assemblies)
            motor_frame_transform = parent_transform @ matrix(obj)
            center = obj["shape"]["center_of_mass_mm"]
            center_world = None
            if center is not None:
                center_world = (node_shape_transform @ np.asarray([*center, 1.0]))[
                    :3
                ].tolist()
            self.motor_assemblies.append(
                {
                    "id": f"motor-{active_motor + 1:02d}",
                    "path": object_path,
                    "cad_object": name,
                    "cad_label": label,
                    "suggested_model": model,
                    "top_branch": top_branch,
                    "shape_transform": flattened_column_major(node_shape_transform),
                    "frame_transform": flattened_column_major(motor_frame_transform),
                    "center_mm": center_world,
                    "bounds_mm": transformed_bounds(
                        obj["shape"]["bounding_box_mm"], node_shape_transform
                    ),
                }
            )

        measured_kind = measured_assembly_kind(label)
        if active_measured is None and measured_kind is not None:
            active_measured = len(self.measured_assemblies)
            self.measured_assemblies.append(
                {
                    "id": f"measured-{active_measured + 1:02d}",
                    "path": object_path,
                    "cad_object": name,
                    "cad_label": label,
                    "kind": measured_kind,
                    "top_branch": top_branch,
                    "shape_transform": flattened_column_major(node_shape_transform),
                    "bounds_mm": transformed_bounds(
                        obj["shape"]["bounding_box_mm"], node_shape_transform
                    ),
                }
            )

        if active_end_effector is None and is_anonymous_end_effector_source(label):
            active_end_effector = len(self.excluded_end_effectors)
            self.excluded_end_effectors.append(
                {
                    "id": f"cad-hand-{active_end_effector + 1:02d}",
                    "path": object_path,
                    "cad_object": name,
                    "cad_label": label,
                    "top_branch": top_branch,
                    "shape_transform": flattened_column_major(node_shape_transform),
                    "bounds_mm": transformed_bounds(
                        obj["shape"]["bounding_box_mm"], node_shape_transform
                    ),
                }
            )

        if obj["type_id"] == "Part::Feature":
            self.occurrences.append(
                {
                    "id": object_path,
                    "source_name": name,
                    "top_branch": top_branch,
                    "transform": flattened_column_major(parent_transform),
                    "bounds_mm": transformed_bounds(
                        obj["shape"]["bounding_box_mm"], parent_transform
                    ),
                    "motor_assembly": active_motor,
                    "measured_assembly": active_measured,
                    "anonymous_end_effector": active_end_effector,
                }
            )
            return

        if obj["type_id"] == "App::LinkGroup":
            child_transform = parent_transform @ matrix(obj)
            for child in obj["children"]:
                self._walk(
                    child,
                    child_transform,
                    path=object_path,
                    top_branch=child if top_branch is None else top_branch,
                    active_motor=active_motor,
                    active_measured=active_measured,
                    active_end_effector=active_end_effector,
                    stack=(*stack, name),
                )
            return

        if obj["type_id"] == "App::Link":
            target = obj["linked_object"]
            if target is None:
                raise ValueError(f"App::Link {name} has no LinkedObject")
            target_transform = (
                parent_transform
                @ matrix(obj)
                @ np.linalg.inv(matrix(self.by_name[target]))
            )
            self._walk(
                target,
                target_transform,
                path=object_path,
                top_branch=top_branch,
                active_motor=active_motor,
                active_measured=active_measured,
                active_end_effector=active_end_effector,
                stack=(*stack, name),
            )
            return

        raise ValueError(f"unsupported FreeCAD object type {obj['type_id']}: {name}")


def main() -> None:
    args = parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    walker = AssemblyWalker(inventory)
    walker.walk()
    by_name = walker.by_name

    occurrences_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for occurrence in walker.occurrences:
        occurrences_by_source[occurrence["source_name"]].append(occurrence)

    mesh_url_prefix = args.mesh_url_prefix.rstrip("/")
    components = []
    for source_name, occurrences in sorted(occurrences_by_source.items()):
        obj = by_name[source_name]
        shape = obj["shape"]
        motor_memberships = {item["motor_assembly"] for item in occurrences}
        measured_memberships = {item["measured_assembly"] for item in occurrences}
        end_effector_memberships = {
            item["anonymous_end_effector"] for item in occurrences
        }
        is_motor_internal = None not in motor_memberships
        is_measured_internal = None not in measured_memberships
        is_end_effector_excluded = None not in end_effector_memberships
        top_counts = Counter(item["top_branch"] for item in occurrences)
        components.append(
            {
                "source_name": source_name,
                "source_label": obj["label"],
                "shape_type": shape["shape_type"],
                "solid_count": shape["solid_count"],
                "volume_mm3": shape["volume_mm3"],
                "area_mm2": shape["area_mm2"],
                "bounds_mm": shape["bounding_box_mm"],
                "size_mm": bounds_size(shape["bounding_box_mm"]),
                "occurrence_count": len(occurrences),
                "top_branches": [
                    {
                        "name": name,
                        "label": by_name[name]["label"],
                        "count": count,
                    }
                    for name, count in sorted(top_counts.items())
                ],
                "motor_internal": is_motor_internal,
                "measured_assembly_internal": is_measured_internal,
                "end_effector_excluded": is_end_effector_excluded,
                "mixed_assembly_membership": len(motor_memberships) > 1
                or len(measured_memberships) > 1
                or len(end_effector_memberships) > 1,
                "recommendation": recommendation(
                    obj["label"],
                    motor_internal=is_motor_internal,
                    measured_internal=is_measured_internal,
                    end_effector_excluded=is_end_effector_excluded,
                ),
                "mesh": {
                    "url": f"{mesh_url_prefix}/{source_name}.bello3d",
                    "triangle_count": None,
                },
                "occurrences": occurrences,
            }
        )

    root_name = inventory["summary"]["root_objects"][0]
    root = by_name[root_name]
    union_minimum = np.min(
        [occurrence["bounds_mm"]["min"] for occurrence in walker.occurrences], axis=0
    )
    union_maximum = np.max(
        [occurrence["bounds_mm"]["max"] for occurrence in walker.occurrences], axis=0
    )
    root_bounds = root["shape"]["bounding_box_mm"]
    bounds_error = max(
        np.max(np.asarray(root_bounds["min"]) - union_minimum),
        np.max(union_maximum - np.asarray(root_bounds["max"])),
    )
    if not math.isfinite(bounds_error) or bounds_error > 5.0:
        raise ValueError(
            f"expanded occurrence bounds disagree with CAD root: {bounds_error}"
        )

    mesh_targets = {component["source_name"] for component in components}
    mesh_targets.update(
        assembly["cad_object"]
        for assembly in (*walker.motor_assemblies, *walker.measured_assemblies)
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "review_contract": "bello_mini_cad_classification_v1",
        "source": inventory["source"],
        "freecad": inventory["freecad"],
        "summary": {
            "cad_object_count": inventory["summary"]["object_count"],
            "physical_occurrence_count": len(walker.occurrences),
            "unique_source_geometry_count": len(components),
            "candidate_motor_count": len(walker.motor_assemblies),
            "candidate_measured_assembly_count": len(walker.measured_assemblies),
            "excluded_cad_hand_count": len(walker.excluded_end_effectors),
            "excluded_cad_hand_occurrence_count": sum(
                occurrence["anonymous_end_effector"] is not None
                for occurrence in walker.occurrences
            ),
            "expanded_bounds_max_error_mm": float(bounds_error),
        },
        "root": {
            "name": root_name,
            "label": root["label"],
            "bounds_mm": root_bounds,
            "context_mesh": {
                "url": f"{mesh_url_prefix}/robot_context.bello3d",
                "triangle_count": None,
            },
        },
        "materials": MATERIALS,
        "motor_assemblies": walker.motor_assemblies,
        "measured_assemblies": walker.measured_assemblies,
        "end_effector_policy": {
            "mode": "anonymous_sphere",
            "cad_geometry": "excluded",
            "cad_mass": "excluded",
            "source_assemblies": walker.excluded_end_effectors,
            "sphere": {
                "radius_m": None,
                "mass_kg": None,
                "wrist_mount_frame": "pending_joint_review",
            },
        },
        "components": components,
        "mesh_targets": sorted(mesh_targets),
    }
    atomic_write_json(args.output, manifest)
    print(f"wrote review manifest: {args.output}")
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
