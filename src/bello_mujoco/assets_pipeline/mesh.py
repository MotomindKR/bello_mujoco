from __future__ import annotations

import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import fast_simplification
import trimesh
from trimesh.exchange import stl


@dataclass(frozen=True)
class MeshSources:
    collision: dict[Path, str]
    visual: dict[Path, str]


@dataclass(frozen=True)
class JointAvailabilityConfig:
    disabled_joint_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(not name or name != name.strip() for name in self.disabled_joint_names):
            raise ValueError(
                "disabled_joint_names must contain nonempty names without "
                "surrounding whitespace"
            )
        if len(set(self.disabled_joint_names)) != len(self.disabled_joint_names):
            raise ValueError("disabled_joint_names must not contain duplicates")
        object.__setattr__(
            self,
            "disabled_joint_names",
            tuple(sorted(self.disabled_joint_names)),
        )


def rewrite_urdf_for_mujoco(
    source_urdf: Path,
    out_urdf: Path,
    *,
    joint_availability: JointAvailabilityConfig = JointAvailabilityConfig(),
) -> MeshSources:
    tree = ET.parse(source_urdf)
    root = tree.getroot()
    preserve_fixed_link_bodies(root)
    fix_disabled_joints(root, joint_availability)
    collision_sources: dict[Path, str] = {}
    visual_sources: dict[Path, str] = {}
    collision_by_name: dict[str, Path] = {}
    visual_by_name: dict[str, Path] = {}

    for link in root.findall("link"):
        collisions = list(link.findall("collision"))
        visuals = list(link.findall("visual"))
        link_name = link.get("name", "link")
        if not collisions:
            for visual in visuals:
                collision = deepcopy(visual)
                collision.tag = "collision"
                collision.set("name", f"{link_name}_collision")
                link.append(collision)
            collisions = list(link.findall("collision"))

        for visual in visuals:
            visual.set("name", visual.get("name", f"{link_name}_visual"))
            rewrite_meshes(
                visual,
                source_urdf=source_urdf,
                out_subdir="visual_meshes",
                suffix="_visual",
                extension=".stl",
                sources=visual_sources,
                by_name=visual_by_name,
            )

        for collision in collisions:
            collision.set("name", collision.get("name", f"{link_name}_collision"))
            rewrite_meshes(
                collision,
                source_urdf=source_urdf,
                out_subdir="meshes",
                suffix="",
                extension=None,
                sources=collision_sources,
                by_name=collision_by_name,
            )

    out_urdf.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(out_urdf, encoding="utf-8", xml_declaration=True)
    return MeshSources(collision=collision_sources, visual=visual_sources)


def fix_disabled_joints(root: ET.Element, config: JointAvailabilityConfig) -> None:
    joints = {
        joint.get("name"): joint
        for joint in root.findall("joint")
        if joint.get("name") is not None
    }
    for joint_name in config.disabled_joint_names:
        joint = joints.get(joint_name)
        if joint is None:
            raise ValueError(
                f"disabled joint {joint_name!r} is absent from the source URDF"
            )
        joint.set("type", "fixed")
        for tag in (
            "axis",
            "limit",
            "dynamics",
            "mimic",
            "safety_controller",
            "calibration",
        ):
            for element in joint.findall(tag):
                joint.remove(element)


def preserve_fixed_link_bodies(root: ET.Element) -> None:
    mujoco_config = root.find("mujoco")
    if mujoco_config is None:
        mujoco_config = ET.SubElement(root, "mujoco")
    compiler = mujoco_config.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(mujoco_config, "compiler")
    compiler.set("fusestatic", "false")


def rewrite_meshes(
    element: ET.Element,
    *,
    source_urdf: Path,
    out_subdir: str,
    suffix: str,
    extension: str | None,
    sources: dict[Path, str],
    by_name: dict[str, Path],
) -> None:
    for mesh in element.findall(".//mesh"):
        filename = mesh.attrib["filename"]
        src = (source_urdf.parent / filename).resolve()
        dst_name = f"{src.stem}{suffix}{extension or src.suffix.lower()}"
        prior = by_name.get(dst_name)
        if prior is not None and prior != src:
            raise ValueError(
                f"Mesh basename collision for {dst_name}: {prior} vs {src}"
            )
        by_name[dst_name] = src
        sources[src] = dst_name
        mesh.set("filename", f"../{out_subdir}/{dst_name}")


def export_binary_simplified_mesh(
    src: Path,
    dst: Path,
    *,
    face_budget: int,
    aggression: float = 10.0,
    weld_vertices: bool = False,
) -> None:
    mesh = load_trimesh(src, weld_vertices=weld_vertices)

    while len(mesh.faces) > face_budget:
        vertices, faces = fast_simplification.simplify(
            mesh.vertices,
            mesh.faces,
            target_count=face_budget,
            agg=aggression,
        )
        simplified = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        if len(simplified.faces) >= len(mesh.faces):
            break
        mesh = simplified
    if len(mesh.faces) > 200_000:
        raise ValueError(
            f"{src} simplified to {len(mesh.faces)} faces, still above MuJoCo's 200000 STL cap"
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(stl.export_stl(mesh))


def load_trimesh(src: Path, *, weld_vertices: bool = False) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(src, process=weld_vertices)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"{src} did not load as a Trimesh")
    if weld_vertices:
        mesh.merge_vertices()
        mesh.remove_unreferenced_vertices()
    return mesh
