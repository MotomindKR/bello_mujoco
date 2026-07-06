"""Prepare Bello URDF/STL inputs for MuJoCo.

The source robot descriptions use ASCII STL meshes. MuJoCo's STL loader expects
binary STL and also rejects very large meshes, so this tool copies the source
descriptions, writes simplified binary meshes, rewrites URDF mesh references,
then compiles the URDF into MJCF and adds the lower-body position actuators.
"""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

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
    parser.add_argument("--copy-source", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    assets = (args.asset,)
    prepared = [
        prepare_asset(
            asset=name,
            source_root=args.source_root,
            mesh_face_budget=args.mesh_face_budget,
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

    compile_urdf_to_mjcf(out_urdf, out_mjcf)
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


def compile_urdf_to_mjcf(urdf_path: Path, mjcf_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    tmp_path = mjcf_path.with_suffix(".compiled.xml")
    mujoco.mj_saveLastXML(str(tmp_path), model)

    tree = ET.parse(tmp_path)
    root = tree.getroot()
    ensure_option(root)
    ensure_ground_plane(root)
    ensure_free_root_joint(root)
    ensure_position_actuators(root)
    ensure_home_keyframe(root)

    ET.indent(tree, space="  ")
    tree.write(mjcf_path, encoding="utf-8", xml_declaration=True)
    tmp_path.unlink(missing_ok=True)

    # Validate the generated model immediately so failures point at the asset step.
    mujoco.MjModel.from_xml_path(str(mjcf_path))


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
