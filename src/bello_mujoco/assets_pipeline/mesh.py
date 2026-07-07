from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import fast_simplification
import trimesh
from trimesh.exchange import stl

from bello_mujoco.assets import source_assets_root


def copy_source_inputs(source_root: Path) -> None:
    dst_root = source_assets_root()
    dst_root.mkdir(parents=True, exist_ok=True)
    for child in ("lower_body", "upper_body", "full_body"):
        src = source_root / child
        dst = dst_root / child
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(
            src,
            dst,
            ignore=shutil.ignore_patterns("usd", "*.usd", "*.usda", ".asset_hash", "__pycache__"),
        )


def rewrite_urdf_for_mujoco(source_urdf: Path, out_urdf: Path) -> dict[Path, str]:
    tree = ET.parse(source_urdf)
    root = tree.getroot()
    mesh_sources: dict[Path, str] = {}
    by_name: dict[str, Path] = {}

    for link in root.findall("link"):
        collisions = list(link.findall("collision"))
        visuals = list(link.findall("visual"))
        if collisions:
            for visual in visuals:
                link.remove(visual)
        else:
            for visual in visuals:
                visual.tag = "collision"
                visual.set("name", visual.get("name", f"{link.get('name')}_collision"))

        for mesh in link.findall(".//mesh"):
            filename = mesh.attrib["filename"]
            src = (source_urdf.parent / filename).resolve()
            dst_name = src.name
            prior = by_name.get(dst_name)
            if prior is not None and prior != src:
                raise ValueError(f"Mesh basename collision for {dst_name}: {prior} vs {src}")
            by_name[dst_name] = src
            mesh_sources[src] = dst_name
            mesh.set("filename", f"../meshes/{dst_name}")

    out_urdf.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(out_urdf, encoding="utf-8", xml_declaration=True)
    return mesh_sources


def export_binary_simplified_mesh(src: Path, dst: Path, *, face_budget: int) -> None:
    mesh = trimesh.load_mesh(src, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"{src} did not load as a Trimesh")

    while len(mesh.faces) > face_budget:
        vertices, faces = fast_simplification.simplify(
            mesh.vertices,
            mesh.faces,
            target_count=face_budget,
            agg=10.0,
        )
        simplified = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        if len(simplified.faces) >= len(mesh.faces):
            break
        mesh = simplified
    if len(mesh.faces) > 200_000:
        raise ValueError(f"{src} simplified to {len(mesh.faces)} faces, still above MuJoCo's 200000 STL cap")

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(stl.export_stl(mesh))
