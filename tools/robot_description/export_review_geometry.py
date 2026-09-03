"""Tessellate FreeCAD objects into raw NumPy review geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import FreeCAD as App
    import numpy as np
    import Part
except ImportError as error:  # pragma: no cover - only ordinary Python hits this
    raise SystemExit(
        "run with FreeCAD's Python and its module directory on PYTHONPATH"
    ) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fcstd", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--part-deflection-mm", type=float, default=1.0)
    parser.add_argument("--context-deflection-mm", type=float, default=5.0)
    parser.add_argument("--context-only", action="store_true")
    return parser.parse_args()


def tessellate(shape: Any, deflection_mm: float) -> tuple[Any, Any]:
    vertices, triangles = shape.tessellate(deflection_mm)
    vertex_array = np.asarray(
        [[vertex.x, vertex.y, vertex.z] for vertex in vertices], dtype=np.float32
    )
    triangle_array = np.asarray(triangles, dtype=np.uint32)
    if vertex_array.ndim != 2 or vertex_array.shape[1:] != (3,):
        raise ValueError(f"invalid vertex array shape: {vertex_array.shape}")
    if triangle_array.ndim != 2 or triangle_array.shape[1:] != (3,):
        raise ValueError(f"invalid triangle array shape: {triangle_array.shape}")
    return vertex_array, triangle_array


def write_geometry(path: Path, *, shape: Any, deflection_mm: float) -> tuple[int, int]:
    vertices, triangles = tessellate(shape, deflection_mm)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, vertices=vertices, triangles=triangles)
    return len(vertices), len(triangles)


def placement_from_column_major(values: list[float]) -> Any:
    value = np.asarray(values, dtype=np.float64).reshape((4, 4), order="F")
    matrix = App.Matrix()
    for row in range(4):
        for column in range(4):
            setattr(matrix, f"A{row + 1}{column + 1}", float(value[row, column]))
    return App.Placement(matrix)


def context_shape(document: Any, manifest: dict[str, Any]) -> Any:
    if manifest.get("end_effector_policy", {}).get("cad_geometry") != "excluded":
        root_name = manifest["root"]["name"]
        root = document.getObject(root_name)
        if root is None:
            raise KeyError(f"root object missing from FCStd: {root_name}")
        return root.Shape

    shapes = []
    excluded_occurrences = 0
    for component in manifest["components"]:
        source = document.getObject(component["source_name"])
        if source is None:
            raise KeyError(
                f"source object missing from FCStd: {component['source_name']}"
            )
        for occurrence in component["occurrences"]:
            if occurrence["anonymous_end_effector"] is not None:
                excluded_occurrences += 1
                continue
            shape = source.Shape.copy()
            delta = placement_from_column_major(occurrence["transform"])
            shape.Placement = delta.multiply(shape.Placement)
            shapes.append(shape)
    if not shapes:
        raise ValueError("anonymous end-effector policy removed every CAD occurrence")
    print(
        f"context excludes {excluded_occurrences} CAD-hand occurrences; "
        f"retains {len(shapes)} occurrences",
        flush=True,
    )
    return Part.makeCompound(shapes)


def main() -> None:
    args = parse_args()
    fcstd = args.fcstd.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not fcstd.is_file():
        raise FileNotFoundError(fcstd)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    document = App.openDocument(str(fcstd))
    if document is None:
        raise RuntimeError(f"FreeCAD could not open {fcstd}")
    document.recompute()

    root_name = manifest["root"]["name"]
    vertices, triangles = write_geometry(
        output_dir / "robot_context.npz",
        shape=context_shape(document, manifest),
        deflection_mm=args.context_deflection_mm,
    )
    print(
        f"context {root_name}: {vertices} vertices, {triangles} triangles",
        flush=True,
    )
    if args.context_only:
        return

    targets = manifest["mesh_targets"]
    for index, name in enumerate(targets, start=1):
        obj = document.getObject(name)
        if obj is None:
            raise KeyError(f"mesh target missing from FCStd: {name}")
        vertices, triangles = write_geometry(
            output_dir / f"{name}.npz",
            shape=obj.Shape,
            deflection_mm=args.part_deflection_mm,
        )
        if index % 25 == 0 or index == len(targets):
            print(
                f"tessellated {index}/{len(targets)} targets; "
                f"{name}: {vertices} vertices, {triangles} triangles",
                flush=True,
            )


if __name__ == "__main__":
    main()
