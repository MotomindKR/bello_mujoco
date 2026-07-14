from __future__ import annotations

import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import fast_simplification
import trimesh
from trimesh.exchange import stl


@dataclass(frozen=True)
class MeshSources:
    collision: dict[Path, str]
    visual: dict[Path, str]


EndEffectorSphereMode = Literal["none", "both", "right", "left"]
END_EFFECTOR_SPHERE_RADIUS = 0.0325
END_EFFECTOR_SPHERE_OFFSET_Z = -0.02
END_EFFECTOR_SPHERE_MASS = 0.05
END_EFFECTOR_SPHERES = {
    "right": (
        "right_wrist_yaw_joint",
        "r_end_effector_sphere_link",
        "r_end_effector_sphere_fixed",
    ),
    "left": (
        "left_wrist_yaw_joint",
        "l_end_effector_sphere_link",
        "l_end_effector_sphere_fixed",
    ),
}


def rewrite_urdf_for_mujoco(
    source_urdf: Path,
    out_urdf: Path,
    *,
    end_effector_sphere_mode: EndEffectorSphereMode = "none",
) -> MeshSources:
    tree = ET.parse(source_urdf)
    root = tree.getroot()
    preserve_fixed_link_bodies(root)
    replace_end_effectors_with_spheres(root, end_effector_sphere_mode)
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


def preserve_fixed_link_bodies(root: ET.Element) -> None:
    mujoco_config = root.find("mujoco")
    if mujoco_config is None:
        mujoco_config = ET.SubElement(root, "mujoco")
    compiler = mujoco_config.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(mujoco_config, "compiler")
    compiler.set("fusestatic", "false")


def replace_end_effectors_with_spheres(
    root: ET.Element, mode: EndEffectorSphereMode
) -> None:
    if mode == "none":
        return

    sides = ("right", "left") if mode == "both" else (mode,)
    for side in sides:
        joint_name, sphere_link_name, fixed_joint_name = END_EFFECTOR_SPHERES[side]
        joint = root.find(f"./joint[@name='{joint_name}']")
        if joint is None:
            raise ValueError(f"end-effector replacement joint not found: {joint_name}")
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise ValueError(f"joint {joint_name} is missing parent or child")
        parent_link = parent.attrib["link"]
        child_link = child.attrib["link"]
        origin = joint.find("origin")
        xyz = parse_xyz(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0")
        rpy = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"

        remove_link_subtree(root, child_link)
        sphere_xyz = (xyz[0], xyz[1], xyz[2] + END_EFFECTOR_SPHERE_OFFSET_Z)
        root.append(make_sphere_link(sphere_link_name))
        fixed_joint = ET.Element("joint", {"name": fixed_joint_name, "type": "fixed"})
        ET.SubElement(
            fixed_joint, "origin", {"xyz": format_xyz(sphere_xyz), "rpy": rpy}
        )
        ET.SubElement(fixed_joint, "parent", {"link": parent_link})
        ET.SubElement(fixed_joint, "child", {"link": sphere_link_name})
        root.append(fixed_joint)


def remove_link_subtree(root: ET.Element, root_link: str) -> None:
    joints_by_parent: dict[str, list[ET.Element]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        if parent is not None:
            joints_by_parent.setdefault(parent.attrib["link"], []).append(joint)

    links_to_remove: set[str] = set()
    joints_to_remove: set[ET.Element] = set()

    def visit(link_name: str) -> None:
        links_to_remove.add(link_name)
        for joint in joints_by_parent.get(link_name, ()):
            joints_to_remove.add(joint)
            child = joint.find("child")
            if child is not None:
                visit(child.attrib["link"])

    incoming = [
        joint
        for joint in root.findall("joint")
        if joint.find("child") is not None
        and joint.find("child").attrib["link"] == root_link
    ]
    joints_to_remove.update(incoming)
    visit(root_link)

    for joint in root.findall("joint"):
        if joint in joints_to_remove:
            root.remove(joint)
    for link in root.findall("link"):
        if link.get("name") in links_to_remove:
            root.remove(link)


def make_sphere_link(name: str) -> ET.Element:
    inertia = 0.4 * END_EFFECTOR_SPHERE_MASS * END_EFFECTOR_SPHERE_RADIUS**2
    link = ET.Element("link", {"name": name})
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": f"{END_EFFECTOR_SPHERE_MASS:g}"})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": f"{inertia:g}",
            "iyy": f"{inertia:g}",
            "izz": f"{inertia:g}",
            "ixy": "0",
            "iyz": "0",
            "ixz": "0",
        },
    )
    for tag in ("visual", "collision"):
        element = ET.SubElement(link, tag, {"name": f"{name}_{tag}"})
        ET.SubElement(element, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        geometry = ET.SubElement(element, "geometry")
        ET.SubElement(geometry, "sphere", {"radius": f"{END_EFFECTOR_SPHERE_RADIUS:g}"})
    return link


def parse_xyz(value: str) -> tuple[float, float, float]:
    x, y, z = value.split()
    return float(x), float(y), float(z)


def format_xyz(value: tuple[float, float, float]) -> str:
    return " ".join(f"{item:.8g}" for item in value)


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
