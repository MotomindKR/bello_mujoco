"""Extract a deterministic, reviewable inventory from a STEP assembly.

This script must run under FreeCAD's Python executable, not ordinary CPython.
It intentionally records geometric properties without inferring materials,
link ownership, joints, or physical masses.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

try:
    import FreeCAD as App
    import Import
except ImportError as error:  # pragma: no cover - only ordinary Python hits this
    raise SystemExit("run this script with FreeCAD's freecadcmd executable") from error

SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fcstd-output", type=Path)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite: {result}")
    return result


def vector(value: Any, *, name: str) -> list[float]:
    return [
        finite(value.x, name=f"{name}.x"),
        finite(value.y, name=f"{name}.y"),
        finite(value.z, name=f"{name}.z"),
    ]


def matrix3(value: Any, *, name: str) -> list[list[float]]:
    return [
        [
            finite(getattr(value, f"A{row + 1}{column + 1}"), name=name)
            for column in range(3)
        ]
        for row in range(3)
    ]


def placement(value: Any) -> dict[str, Any]:
    quaternion = value.Rotation.Q
    matrix = value.toMatrix()
    return {
        "translation_mm": vector(value.Base, name="placement translation"),
        "quaternion_xyzw": [
            finite(component, name="placement quaternion") for component in quaternion
        ],
        "matrix4": [
            [
                finite(
                    getattr(matrix, f"A{row + 1}{column + 1}"),
                    name="placement matrix",
                )
                for column in range(4)
            ]
            for row in range(4)
        ],
    }


def global_placement(obj: Any) -> Any:
    getter = getattr(obj, "getGlobalPlacement", None)
    if callable(getter):
        return getter()
    return obj.Placement


def linked_object_name(obj: Any) -> str | None:
    linked = getattr(obj, "LinkedObject", None)
    return None if linked is None else linked.Name


def simple_property_value(obj: Any, name: str) -> Any:
    try:
        value = getattr(obj, name)
    except Exception:
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, (bool, int, float, str)) for item in value
    ):
        return list(value)
    return None


def object_properties(obj: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in getattr(obj, "PropertiesList", ()):  # FreeCAD dynamic API
        if name in {
            "ExpressionEngine",
            "Label",
            "Label2",
            "Placement",
            "Shape",
            "Visibility",
        }:
            continue
        value = simple_property_value(obj, name)
        if value is not None:
            result[name] = value
    return result


def view_color(obj: Any) -> list[float] | None:
    view = getattr(obj, "ViewObject", None)
    if view is None:
        return None
    color = getattr(view, "ShapeColor", None)
    if color is None:
        return None
    return [finite(component, name="shape color") for component in color]


def geometry_sha256(shape: Any) -> str | None:
    local_shape = shape.copy()
    local_shape.Placement = App.Placement()
    exporter = getattr(local_shape, "exportBrepToString", None)
    if not callable(exporter):
        return None
    value = exporter()
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def solid_mass_properties(
    shape: Any,
) -> tuple[list[float] | None, list[list[float]] | None]:
    components: list[tuple[float, list[float], list[list[float]]]] = []
    for solid in shape.Solids:
        volume = finite(solid.Volume, name="solid volume")
        if volume <= 0.0:
            continue
        center = vector(solid.CenterOfMass, name="solid center of mass")
        inertia = matrix3(solid.MatrixOfInertia, name="solid matrix of inertia")
        components.append((volume, center, inertia))

    total_volume = sum(volume for volume, _, _ in components)
    if total_volume <= 0.0:
        return None, None

    center = [
        sum(
            volume * component_center[axis]
            for volume, component_center, _ in components
        )
        / total_volume
        for axis in range(3)
    ]
    inertia = [[0.0 for _ in range(3)] for _ in range(3)]
    for volume, component_center, component_inertia in components:
        offset = [component_center[axis] - center[axis] for axis in range(3)]
        squared_distance = sum(component * component for component in offset)
        for row in range(3):
            for column in range(3):
                parallel_axis = volume * (
                    (squared_distance if row == column else 0.0)
                    - offset[row] * offset[column]
                )
                inertia[row][column] += component_inertia[row][column] + parallel_axis

    return center, inertia


def shape_record(shape: Any) -> dict[str, Any] | None:
    if shape is None or shape.isNull():
        return None
    bounds = shape.BoundBox
    center_of_mass, matrix_of_inertia = solid_mass_properties(shape)
    record = {
        "shape_type": shape.ShapeType,
        "solid_count": len(shape.Solids),
        "shell_count": len(shape.Shells),
        "face_count": len(shape.Faces),
        "edge_count": len(shape.Edges),
        "vertex_count": len(shape.Vertexes),
        "volume_mm3": finite(shape.Volume, name="shape volume"),
        "area_mm2": finite(shape.Area, name="shape area"),
        "center_of_mass_mm": center_of_mass,
        "matrix_of_inertia_mm5": matrix_of_inertia,
        "bounding_box_mm": {
            "min": [
                finite(bounds.XMin, name="bounds xmin"),
                finite(bounds.YMin, name="bounds ymin"),
                finite(bounds.ZMin, name="bounds zmin"),
            ],
            "max": [
                finite(bounds.XMax, name="bounds xmax"),
                finite(bounds.YMax, name="bounds ymax"),
                finite(bounds.ZMax, name="bounds zmax"),
            ],
        },
        "shape_placement": placement(shape.Placement),
        "geometry_sha256": geometry_sha256(shape),
    }
    return record


def object_record(obj: Any) -> dict[str, Any]:
    shape = getattr(obj, "Shape", None)
    return {
        "name": obj.Name,
        "label": obj.Label,
        "type_id": obj.TypeId,
        "parents": sorted(parent.Name for parent in obj.InList),
        "children": sorted(child.Name for child in obj.OutList),
        "linked_object": linked_object_name(obj),
        "local_placement": placement(obj.Placement),
        "global_placement": placement(global_placement(obj)),
        "shape_color_rgb": view_color(obj),
        "properties": object_properties(obj),
        "shape": shape_record(shape),
    }


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    fcstd_output = (
        None if args.fcstd_output is None else args.fcstd_output.expanduser().resolve()
    )
    if not source.is_file():
        raise FileNotFoundError(source)

    print(f"source: {source}", flush=True)
    print(f"source bytes: {source.stat().st_size}", flush=True)
    source_digest = file_sha256(source)
    print(f"source sha256: {source_digest}", flush=True)
    print("importing STEP assembly...", flush=True)
    Import.open(str(source))
    document = App.ActiveDocument
    if document is None:
        raise RuntimeError("FreeCAD did not create an active document")
    document.recompute()
    print(f"document: {document.Name}; objects: {len(document.Objects)}", flush=True)

    records = []
    for index, obj in enumerate(document.Objects, start=1):
        records.append(object_record(obj))
        if index % 100 == 0 or index == len(document.Objects):
            print(f"inventoried {index}/{len(document.Objects)} objects", flush=True)

    shape_records = [record for record in records if record["shape"] is not None]
    geometry_hashes = [
        record["shape"]["geometry_sha256"]
        for record in shape_records
        if record["shape"]["geometry_sha256"] is not None
    ]
    inventory = {
        "schema_version": SCHEMA_VERSION,
        "robot": "bello_mini",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(source),
            "size_bytes": source.stat().st_size,
            "sha256": source_digest,
            "units": "mm",
        },
        "freecad": {
            "version": list(App.Version()),
            "document_name": document.Name,
            "document_label": document.Label,
        },
        "summary": {
            "object_count": len(records),
            "shape_object_count": len(shape_records),
            "solid_count": sum(
                record["shape"]["solid_count"] for record in shape_records
            ),
            "unique_geometry_count": len(set(geometry_hashes)),
            "root_objects": sorted(
                record["name"] for record in records if not record["parents"]
            ),
            "type_counts": dict(
                sorted(Counter(record["type_id"] for record in records).items())
            ),
        },
        "objects": records,
    }
    atomic_write_json(output, inventory)
    print(f"wrote inventory: {output}", flush=True)

    if fcstd_output is not None:
        fcstd_output.parent.mkdir(parents=True, exist_ok=True)
        document.saveAs(str(fcstd_output))
        print(f"wrote converted document: {fcstd_output}", flush=True)


if __name__ == "__main__":
    main()
