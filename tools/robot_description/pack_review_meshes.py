"""Simplify raw FreeCAD tessellations and pack browser review meshes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
from typing import Any

import numpy as np
import trimesh


MAGIC = b"BELLO3D1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--part-face-limit", type=int, default=30_000)
    parser.add_argument("--context-face-limit", type=int, default=300_000)
    return parser.parse_args()


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_mesh(path: Path) -> trimesh.Trimesh:
    with np.load(path) as value:
        vertices = value["vertices"]
        triangles = value["triangles"]
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    return mesh


def simplify(mesh: trimesh.Trimesh, face_limit: int) -> trimesh.Trimesh:
    if len(mesh.faces) <= face_limit:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=face_limit, aggression=7)
    except (RuntimeError, ValueError) as error:
        print(f"warning: simplification failed ({error}); retaining original mesh")
        return mesh


def pack(path: Path, mesh: trimesh.Trimesh) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertices, dtype="<f4")
    normals = np.asarray(mesh.vertex_normals, dtype="<f4")
    indices = np.asarray(mesh.faces.reshape(-1), dtype="<u4")
    if vertices.shape != normals.shape or vertices.shape[1:] != (3,):
        raise ValueError(f"invalid mesh arrays for {path}")
    if len(vertices) > np.iinfo(np.uint32).max:
        raise ValueError(f"mesh has too many vertices for uint32 indices: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        output.write(MAGIC)
        output.write(struct.pack("<II", len(vertices), len(indices)))
        output.write(vertices.tobytes(order="C"))
        output.write(normals.tobytes(order="C"))
        output.write(indices.tobytes(order="C"))
    temporary.replace(path)
    bounds = np.asarray(mesh.bounds)
    return {
        "triangle_count": len(mesh.faces),
        "vertex_count": len(vertices),
        "bounds_mm": {"min": bounds[0].tolist(), "max": bounds[1].tolist()},
    }


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    raw_dir = args.raw_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    context = simplify(
        load_mesh(raw_dir / "robot_context.npz"), args.context_face_limit
    )
    context_record = pack(output_dir / "robot_context.bello3d", context)
    manifest["root"]["context_mesh"].update(context_record)
    print(f"packed context: {context_record['triangle_count']} triangles", flush=True)

    packed: dict[str, dict[str, Any]] = {}
    targets = manifest["mesh_targets"]
    for index, name in enumerate(targets, start=1):
        mesh = simplify(load_mesh(raw_dir / f"{name}.npz"), args.part_face_limit)
        packed[name] = pack(output_dir / f"{name}.bello3d", mesh)
        if index % 25 == 0 or index == len(targets):
            print(f"packed {index}/{len(targets)} targets: {name}", flush=True)

    for component in manifest["components"]:
        component["mesh"].update(packed[component["source_name"]])
    mesh_url_prefix = manifest["root"]["context_mesh"]["url"].rsplit("/", 1)[0]
    for collection in ("motor_assemblies", "measured_assemblies"):
        for assembly in manifest[collection]:
            name = assembly["cad_object"]
            assembly["mesh"] = {
                "url": f"{mesh_url_prefix}/{name}.bello3d",
                **packed[name],
            }

    atomic_write_json(manifest_path, manifest)
    print(f"updated review manifest: {manifest_path}")


if __name__ == "__main__":
    main()
