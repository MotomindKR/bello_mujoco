"""Prepare Bello URDF/STL inputs for MuJoCo.

The source robot descriptions use ASCII STL meshes. MuJoCo's STL loader expects
binary STL and also rejects very large meshes, so this tool copies the source
descriptions, writes simplified binary meshes, rewrites URDF mesh references,
then compiles the URDF into MJCF, applies a training collision profile, and adds
the lower-body position actuators.
"""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mujoco
import fast_simplification
import trimesh
from trimesh.exchange import stl

from bello_mujoco.assets import generated_assets_root, generated_model_path, generated_urdf_path, source_assets_root
from bello_mujoco.contracts import (
    ACTUATOR_GAINS,
    DEFAULT_JOINT_POS,
    FIXED_JOINT_DEFAULT_POS,
    FIXED_JOINT_GAINS,
    POLICY_JOINT_ORDER,
    SIM_TIMESTEP,
    SOFT_JOINT_LIMITS,
)

DEFAULT_SOURCE_ROOT = Path(
    "/home/gmmyung/motomind_platform/systems/motomind_isaaclab/"
    "motomind_isaaclab/assets/data/bello"
)
FOOT_MESH_NAMES = frozenset({"left_ankle_roll_link", "right_ankle_roll_link"})
COLLISION_MODES = ("mesh", "foot-mesh", "box")


@dataclass(frozen=True)
class PreparedAsset:
    name: str
    urdf_path: Path
    mjcf_path: Path
    mesh_count: int


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--asset", choices=("full_body",), default="full_body")
    parser.add_argument("--mesh-face-budget", type=int, default=500)
    parser.add_argument("--collision-mode", choices=COLLISION_MODES, default="foot-mesh")
    parser.add_argument("--neighbor-exclusion-depth", type=int, default=2)
    parser.add_argument("--copy-source", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    assets = (args.asset,)
    prepared = [
        prepare_asset(
            asset=name,
            source_root=args.source_root,
            mesh_face_budget=args.mesh_face_budget,
            collision_mode=args.collision_mode,
            neighbor_exclusion_depth=args.neighbor_exclusion_depth,
            copy_source=args.copy_source,
        )
        for name in assets
    ]

    for item in prepared:
        print(f"{item.name}: {item.mesh_count} meshes -> {item.mjcf_path}")


def prepare_asset(
    *,
    asset: str,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    mesh_face_budget: int = 500,
    collision_mode: str = "foot-mesh",
    neighbor_exclusion_depth: int = 2,
    copy_source: bool = True,
) -> PreparedAsset:
    source_root = source_root.expanduser().resolve()
    source_urdf = source_root / asset / "urdf" / f"bello_{asset}.urdf"
    if not source_urdf.exists():
        raise FileNotFoundError(source_urdf)

    if copy_source:
        copy_source_inputs(source_root)

    out_urdf = generated_urdf_path(asset)
    out_mesh_dir = generated_assets_root() / asset / "meshes"
    out_mjcf = generated_model_path(asset)
    out_urdf.parent.mkdir(parents=True, exist_ok=True)
    out_mesh_dir.mkdir(parents=True, exist_ok=True)
    out_mjcf.parent.mkdir(parents=True, exist_ok=True)

    mesh_sources = rewrite_urdf_for_mujoco(source_urdf, out_urdf)
    for src, dst_name in mesh_sources.items():
        export_binary_simplified_mesh(src, out_mesh_dir / dst_name, face_budget=mesh_face_budget)

    compile_urdf_to_mjcf(
        out_urdf,
        out_mjcf,
        collision_mode=collision_mode,
        neighbor_exclusion_depth=neighbor_exclusion_depth,
    )
    return PreparedAsset(asset, out_urdf, out_mjcf, len(mesh_sources))


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


def compile_urdf_to_mjcf(
    urdf_path: Path,
    mjcf_path: Path,
    *,
    collision_mode: str = "foot-mesh",
    neighbor_exclusion_depth: int = 2,
) -> None:
    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    tmp_path = mjcf_path.with_suffix(".compiled.xml")
    mujoco.mj_saveLastXML(str(tmp_path), model)

    tree = ET.parse(tmp_path)
    root = tree.getroot()
    ensure_option(root)
    ensure_ground_plane(root)
    ensure_free_root_joint(root)
    apply_collision_mode(root, mjcf_path.parent, collision_mode)
    ensure_neighbor_collision_excludes(root, depth=neighbor_exclusion_depth)
    ensure_position_actuators(root)
    ensure_home_keyframe(root)

    ET.indent(tree, space="  ")
    tree.write(mjcf_path, encoding="utf-8", xml_declaration=True)
    tmp_path.unlink(missing_ok=True)

    # Validate the generated model immediately so failures point at the asset step.
    mujoco.MjModel.from_xml_path(str(mjcf_path))


def apply_collision_mode(root: ET.Element, mjcf_dir: Path, collision_mode: str) -> None:
    if collision_mode == "mesh":
        return
    if collision_mode not in COLLISION_MODES:
        raise ValueError(f"unknown collision mode: {collision_mode}")

    mesh_bounds = load_scaled_mesh_bounds(root, mjcf_dir)
    keep_meshes = FOOT_MESH_NAMES if collision_mode == "foot-mesh" else frozenset()
    for geom in root.findall(".//geom[@type='mesh']"):
        mesh_name = geom.get("mesh")
        if mesh_name is None or mesh_name in keep_meshes:
            continue
        if mesh_name not in mesh_bounds:
            continue
        center, half_size = mesh_bounds[mesh_name]
        quat = parse_vec(geom.get("quat", "1 0 0 0"), 4, np.asarray([1.0, 0.0, 0.0, 0.0]))
        pos = parse_vec(geom.get("pos", "0 0 0"), 3, np.zeros(3))
        adjusted_pos = pos + quat_rotate(quat, center)
        geom.attrib.pop("mesh", None)
        geom.set("type", "box")
        geom.set("pos", format_vec(adjusted_pos))
        geom.set("size", format_vec(half_size))


def load_scaled_mesh_bounds(root: ET.Element, mjcf_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for mesh in root.findall("./asset/mesh"):
        name = mesh.get("name")
        file_attr = mesh.get("file")
        if name is None or file_attr is None:
            continue
        mesh_path = (mjcf_dir / file_attr).resolve()
        scale = parse_vec(mesh.get("scale", "1 1 1"), 3, np.ones(3))
        tri_mesh = trimesh.load_mesh(mesh_path, process=False)
        mesh_bounds = np.asarray(tri_mesh.bounds, dtype=np.float64) * scale
        center = (mesh_bounds[0] + mesh_bounds[1]) * 0.5
        half_size = np.maximum((mesh_bounds[1] - mesh_bounds[0]) * 0.5, 1.0e-4)
        bounds[name] = (center, half_size)
    return bounds


def parse_vec(value: str, expected_size: int, default: np.ndarray) -> np.ndarray:
    array = np.fromstring(value, sep=" ", dtype=np.float64)
    if array.size != expected_size:
        return default.astype(np.float64)
    return array


def quat_rotate(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz
    qvec = np.asarray([x, y, z], dtype=np.float64)
    uv = np.cross(qvec, vec)
    uuv = np.cross(qvec, uv)
    return vec + 2.0 * (w * uv + uuv)


def format_vec(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.8g}" for value in values)


def ensure_neighbor_collision_excludes(root: ET.Element, *, depth: int) -> None:
    if depth <= 0:
        return
    names, edges = body_graph(root)
    graph = {name: set() for name in names}
    for parent, child in edges:
        graph[parent].add(child)
        graph[child].add(parent)

    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    existing = {
        tuple(sorted((exclude.get("body1", ""), exclude.get("body2", ""))))
        for exclude in contact.findall("exclude")
    }

    for body1, body2 in nearby_body_pairs(names, graph, depth):
        key = tuple(sorted((body1, body2)))
        if key in existing:
            continue
        ET.SubElement(contact, "exclude", {"body1": body1, "body2": body2})
        existing.add(key)


def body_graph(root: ET.Element) -> tuple[list[str], list[tuple[str, str]]]:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF has no worldbody")

    names: list[str] = []
    edges: list[tuple[str, str]] = []

    def visit(body: ET.Element, parent_name: str | None) -> None:
        name = body.get("name")
        if name is None:
            return
        names.append(name)
        if parent_name is not None:
            edges.append((parent_name, name))
        for child in body.findall("body"):
            visit(child, name)

    for body in worldbody.findall("body"):
        visit(body, None)
    return names, edges


def nearby_body_pairs(names: list[str], graph: dict[str, set[str]], depth: int) -> list[tuple[str, str]]:
    index = {name: i for i, name in enumerate(names)}
    pairs: set[tuple[str, str]] = set()
    for source in names:
        frontier = {source}
        visited = {source}
        for _ in range(depth):
            next_frontier = set()
            for body in frontier:
                next_frontier.update(graph[body] - visited)
            for target in next_frontier:
                if index[source] < index[target]:
                    pairs.add((source, target))
                elif index[target] < index[source]:
                    pairs.add((target, source))
            visited.update(next_frontier)
            frontier = next_frontier
    return sorted(pairs, key=lambda pair: (index[pair[0]], index[pair[1]]))


def ensure_option(root: ET.Element) -> None:
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(0, option)
    option.set("timestep", str(SIM_TIMESTEP))
    option.set("gravity", "0 0 -9.81")
    option.set("integrator", "implicitfast")


def ensure_ground_plane(root: ET.Element) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF has no worldbody")
    if worldbody.find("./geom[@name='floor']") is not None:
        return
    floor = ET.Element(
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "size": "5 5 0.05",
            "pos": "0 0 0",
            "rgba": "0.22 0.24 0.26 1",
            "condim": "3",
            "friction": "0.8 0.02 0.001",
        },
    )
    worldbody.insert(0, floor)


def ensure_free_root_joint(root: ET.Element) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF has no worldbody")
    if worldbody.find(".//freejoint") is not None:
        return

    robot_children = [child for child in list(worldbody) if child.get("name") != "floor"]
    if not robot_children:
        raise ValueError("MJCF worldbody has no robot geoms or bodies to wrap")
    root_body = ET.Element("body", {"name": "bello_root"})
    root_body.append(ET.Element("freejoint", {"name": "root_free"}))
    for child in robot_children:
        worldbody.remove(child)
        root_body.append(child)
    worldbody.append(root_body)


def ensure_position_actuators(root: ET.Element) -> None:
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    existing = {item.get("joint") for item in actuator.findall("position")}
    for joint_name in POLICY_JOINT_ORDER:
        if joint_name in existing:
            continue
        kp, kv = ACTUATOR_GAINS[joint_name]
        lo, hi = SOFT_JOINT_LIMITS[joint_name]
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"{joint_name}_pos",
                "joint": joint_name,
                "kp": f"{kp:g}",
                "kv": f"{kv:g}",
                "ctrlrange": f"{lo:g} {hi:g}",
                "ctrllimited": "true",
            },
        )
    for joint_name, target in FIXED_JOINT_DEFAULT_POS.items():
        if joint_name in existing:
            continue
        kp, kv = FIXED_JOINT_GAINS[joint_name]
        lo, hi = joint_range(root, joint_name)
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"{joint_name.replace(' ', '_')}_hold",
                "joint": joint_name,
                "kp": f"{kp:g}",
                "kv": f"{kv:g}",
                "ctrlrange": f"{lo:g} {hi:g}",
                "ctrllimited": "true",
                "forcelimited": "false",
                "user": f"{target:g}",
            },
        )


def ensure_home_keyframe(root: ET.Element) -> None:
    keyframe = root.find("keyframe")
    if keyframe is None:
        keyframe = ET.SubElement(root, "keyframe")
    if keyframe.find("./key[@name='home']") is not None:
        return

    joint_qpos = [joint_default(joint.get("name", "")) for joint in root.findall(".//joint")]
    qpos = " ".join(["0", "0", "0.91", "1", "0", "0", "0", *[f"{value:g}" for value in joint_qpos]])
    ctrl = " ".join(
        [
            *[f"{DEFAULT_JOINT_POS[name]:g}" for name in POLICY_JOINT_ORDER],
            *[f"{FIXED_JOINT_DEFAULT_POS[name]:g}" for name in FIXED_JOINT_DEFAULT_POS],
        ]
    )
    ET.SubElement(keyframe, "key", {"name": "home", "qpos": qpos, "ctrl": ctrl})


def joint_default(joint_name: str) -> float:
    if joint_name in DEFAULT_JOINT_POS:
        return DEFAULT_JOINT_POS[joint_name]
    return FIXED_JOINT_DEFAULT_POS.get(joint_name, 0.0)


def joint_range(root: ET.Element, joint_name: str) -> tuple[float, float]:
    eps = 1.0e-6
    joint = root.find(f".//joint[@name='{joint_name}']")
    if joint is None:
        raise ValueError(f"joint not found in MJCF: {joint_name}")
    range_attr = joint.get("range")
    if range_attr is None:
        value = FIXED_JOINT_DEFAULT_POS.get(joint_name, DEFAULT_JOINT_POS.get(joint_name, 0.0))
        return value - eps, value + eps
    lo_str, hi_str = range_attr.split()
    lo, hi = float(lo_str), float(hi_str)
    if lo == hi:
        return lo - eps, hi + eps
    return lo, hi


if __name__ == "__main__":
    main()
