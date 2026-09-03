"""Export minimal per-link Bello Mini visual meshes from reviewed instances."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
from typing import Any

import numpy as np
import trimesh


MAGIC = b"BELLO3D1"
GEOMETRY_CONTRACT = "bello_mini_geometry_manifest_v1"
INERTIAL_CONTRACT = "bello_mini_inertial_manifest_v1"
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
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--inertials", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--face-limit", type=int, default=16_000)
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


def load_bello3d(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as source:
        if source.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"invalid BELLO3D header: {path}")
        vertex_count, index_count = struct.unpack("<II", source.read(8))
        if index_count % 3:
            raise ValueError(f"mesh index count is not triangular: {path}")
        vertices = np.frombuffer(
            source.read(vertex_count * 3 * 4), dtype="<f4"
        ).reshape((-1, 3))
        source.read(vertex_count * 3 * 4)  # vertex normals are regenerated
        faces = np.frombuffer(source.read(index_count * 4), dtype="<u4").reshape(
            (-1, 3)
        )
        if source.read(1):
            raise ValueError(f"unexpected trailing BELLO3D bytes: {path}")
    return vertices.astype(np.float64), faces.astype(np.int64)


def source_mesh_path(repository_root: Path, url: str) -> Path:
    if not url.startswith("/") or ".." in Path(url).parts:
        raise ValueError(f"mesh URL is not a repository-root path: {url}")
    path = repository_root / url.removeprefix("/")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def column_major_matrix(values: list[float]) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64).reshape((4, 4), order="F")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
        raise ValueError("visual transform is not orthogonal")
    return value


def transformed_instance(
    vertices: np.ndarray,
    faces: np.ndarray,
    transform_values: list[float],
    link_origin_cad_mm: np.ndarray,
    rotation_urdf_world_from_link: np.ndarray | None = None,
) -> trimesh.Trimesh:
    transform = column_major_matrix(transform_values)
    world_cad = vertices @ transform[:3, :3].T + transform[:3, 3]
    world_urdf = world_cad @ CAD_TO_URDF.T
    origin_urdf = CAD_TO_URDF @ link_origin_cad_mm
    rotation = (
        np.eye(3)
        if rotation_urdf_world_from_link is None
        else np.asarray(rotation_urdf_world_from_link, dtype=np.float64)
    )
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9):
        raise ValueError("visual link-frame rotation is not orthogonal")
    local_urdf = (world_urdf - origin_urdf) @ rotation
    transformed_faces = faces
    if np.linalg.det(transform[:3, :3]) < 0.0:
        transformed_faces = faces[:, [0, 2, 1]]
    return trimesh.Trimesh(
        vertices=local_urdf,
        faces=transformed_faces,
        process=False,
    )


def simplify(mesh: trimesh.Trimesh, face_limit: int) -> trimesh.Trimesh:
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) <= face_limit:
        return mesh
    simplified = mesh.simplify_quadric_decimation(
        face_count=face_limit,
        aggression=7,
    )
    simplified.merge_vertices()
    simplified.update_faces(simplified.nondegenerate_faces())
    simplified.remove_unreferenced_vertices()
    return simplified


def simplified_source_mesh(
    path: Path,
    face_limit: int,
    raw_cache: dict[Path, tuple[np.ndarray, np.ndarray]],
    simplified_cache: dict[tuple[Path, int], tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    key = (path, face_limit)
    cached = simplified_cache.get(key)
    if cached is not None:
        return cached
    vertices, faces = raw_cache.setdefault(path, load_bello3d(path))
    if len(faces) <= face_limit:
        result = vertices, faces
    else:
        source = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh = simplify(source, face_limit)
        if len(mesh.faces) > int(face_limit * 1.25):
            # STEP assemblies and hard-edged CAD exports can contain many
            # disconnected inner surfaces which quadric decimation cannot
            # collapse. A per-instance convex exterior envelope is preferable
            # to retaining that hidden geometry in a minimal visual asset.
            mesh = simplify(source.convex_hull, face_limit)
        result = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    simplified_cache[key] = result
    return result


def build_link_mesh(
    instances: list[dict[str, Any]],
    *,
    repository_root: Path,
    link_origin_cad_mm: np.ndarray,
    rotation_urdf_world_from_link: np.ndarray,
    raw_cache: dict[Path, tuple[np.ndarray, np.ndarray]],
    simplified_cache: dict[tuple[Path, int], tuple[np.ndarray, np.ndarray]],
    face_limit: int,
) -> tuple[trimesh.Trimesh, int]:
    meshes = []
    input_faces = 0
    instance_face_limit = max(100, face_limit // len(instances))
    for instance in instances:
        path = source_mesh_path(repository_root, instance["mesh"]["url"])
        raw_vertices, raw_faces = raw_cache.setdefault(path, load_bello3d(path))
        input_faces += len(raw_faces)
        vertices, faces = simplified_source_mesh(
            path,
            instance_face_limit,
            raw_cache,
            simplified_cache,
        )
        meshes.append(
            transformed_instance(
                vertices,
                faces,
                instance["transform"],
                link_origin_cad_mm,
                rotation_urdf_world_from_link,
            )
        )
    merged = trimesh.util.concatenate(meshes)
    return simplify(merged, face_limit), input_faces


def main() -> None:
    args = parse_args()
    geometry_path = args.geometry.expanduser().resolve()
    inertials_path = args.inertials.expanduser().resolve()
    repository_root = args.repository_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.face_limit < 100:
        raise ValueError("face limit must be at least 100")
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    inertials = json.loads(inertials_path.read_text(encoding="utf-8"))
    if geometry.get("review_contract") != GEOMETRY_CONTRACT:
        raise ValueError("geometry input has not passed its build gate")
    if inertials.get("review_contract") != INERTIAL_CONTRACT:
        raise ValueError("inertial input has not passed its build gate")
    if geometry.get("source", {}).get("sha256") != inertials.get("source", {}).get(
        "sha256"
    ):
        raise ValueError("geometry and inertials refer to different STEP sources")

    origins = {
        link["name"]: np.asarray(link["frame_origin_cad_world_m"]) * 1000.0
        for link in inertials["links"]
    }
    rotations = {link["name"]: np.eye(3) for link in inertials["links"]}
    for fixed in geometry.get("fixed_geometry", []):
        origin_urdf_world_mm = (
            np.asarray(fixed["frame_origin_urdf_world_m"], dtype=np.float64)
            * 1000.0
        )
        origins[fixed["name"]] = CAD_TO_URDF.T @ origin_urdf_world_mm
        rotations[fixed["name"]] = np.asarray(
            fixed["rotation_urdf_world_from_frame"], dtype=np.float64
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    simplified_cache: dict[tuple[Path, int], tuple[np.ndarray, np.ndarray]] = {}
    records = []
    for link in [*geometry["links"], *geometry.get("fixed_geometry", [])]:
        instances = link["visual_instances"]
        if not instances:
            continue
        name = link["name"]
        mesh, input_faces = build_link_mesh(
            instances,
            repository_root=repository_root,
            link_origin_cad_mm=origins[name],
            rotation_urdf_world_from_link=rotations[name],
            raw_cache=raw_cache,
            simplified_cache=simplified_cache,
            face_limit=args.face_limit,
        )
        path = output_dir / f"{name}.stl"
        mesh.export(path, file_type="stl")
        records.append(
            {
                "link": name,
                "path": str(path),
                "sha256": file_sha256(path),
                "instance_count": len(instances),
                "input_face_count": input_faces,
                "output_face_count": len(mesh.faces),
                "bounds_mm": {
                    "min": mesh.bounds[0].tolist(),
                    "max": mesh.bounds[1].tolist(),
                },
            }
        )
        print(
            f"{name}: {len(instances)} instances, {input_faces} -> "
            f"{len(mesh.faces)} faces",
            flush=True,
        )

    manifest = {
        "schema_version": 1,
        "review_contract": "bello_mini_visual_meshes_v1",
        "source": geometry["source"],
        "source_manifests": {
            **geometry["source_manifests"],
            "geometry_manifest_sha256": file_sha256(geometry_path),
            "inertial_manifest_sha256": file_sha256(inertials_path),
        },
        "mesh_policy": {
            "units": "mm",
            "coordinate_frame": "link-local URDF frame",
            "face_limit_per_link": args.face_limit,
        },
        "meshes": records,
        "summary": {
            "mesh_count": len(records),
            "source_mesh_count": len(raw_cache),
            "input_face_count": sum(value["input_face_count"] for value in records),
            "output_face_count": sum(value["output_face_count"] for value in records),
        },
    }
    atomic_write_json(args.manifest_output, manifest)
    print(f"wrote visual mesh manifest: {args.manifest_output}")
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
