"""Render visual/collision confirmation images for a generated robot URDF."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw


VISUAL_RGBA = {
    "center": "0.45 0.49 0.56 1",
    "left": "0.20 0.50 0.82 1",
    "right": "0.92 0.38 0.20 1",
    "endpoint": "0.96 0.35 0.75 1",
}
COLLISION_RGBA = "0.18 0.82 1 0.28"
COLLISION_JOINT_RGBA = "0.63 0.43 1 0.32"
COLLISION_BOX_RGBA = "0.26 1 0.56 0.30"
COLLISION_ENDPOINT_RGBA = "0.96 0.35 0.75 0.42"
CAMERA_REVIEW_FRAMES = (
    "astra_mount_frame",
    "camera_link",
    "camera_depth_optical_frame",
    "camera_left_ir_optical_frame",
    "camera_right_ir_optical_frame",
    "camera_color_optical_frame",
)
CAMERA_AXIS_RGBA = (
    np.asarray([1.0, 0.25, 0.22, 1.0], dtype=np.float32),
    np.asarray([0.25, 0.90, 0.42, 1.0], dtype=np.float32),
    np.asarray([0.25, 0.52, 1.0, 1.0], dtype=np.float32),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--mjcf-output", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=1000)
    return parser.parse_args()


def link_color(name: str) -> str:
    if name.startswith("left_"):
        return VISUAL_RGBA["left"]
    if name.startswith("right_"):
        return VISUAL_RGBA["right"]
    return VISUAL_RGBA["center"]


def collision_color(geom: ET.Element) -> str:
    name = geom.get("name", "")
    if name.endswith("end_effector_sphere_link_collision"):
        return COLLISION_ENDPOINT_RGBA
    if name.endswith("_joint_envelope") or name.endswith("wrist_roll_link_sphere"):
        return COLLISION_JOINT_RGBA
    if geom.get("type") == "box":
        return COLLISION_BOX_RGBA
    return COLLISION_RGBA


def nominal_ground_translation(model: mujoco.MjModel) -> float:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bottoms = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if "_sole_" not in name:
            continue
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
            half_extent_z = float(np.abs(rotation[2]) @ model.geom_size[geom_id])
        elif model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            axis_z = float(rotation[2, 2])
            radius, half_length = model.geom_size[geom_id, :2]
            half_extent_z = abs(axis_z) * half_length + math.sqrt(
                max(0.0, 1.0 - axis_z * axis_z)
            ) * radius
        else:
            raise ValueError(f"unsupported sole geometry type for {name}")
        bottoms.append(float(data.geom_xpos[geom_id, 2]) - half_extent_z)
    if not bottoms:
        raise ValueError("generated URDF has no named sole collision geometry")
    if max(bottoms) - min(bottoms) > 1e-9:
        raise ValueError("generated URDF sole collision surfaces are not coplanar")
    return -min(bottoms)


def apply_ground_reference(root: ET.Element, translation_z: float) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("compiled review MJCF has no worldbody")
    top_level_bodies = worldbody.findall("body")
    if not top_level_bodies:
        raise ValueError("compiled review MJCF has no top-level robot body")
    for body in top_level_bodies:
        position = [float(value) for value in body.get("pos", "0 0 0").split()]
        position[2] += translation_z
        body.set("pos", " ".join(format(value, ".17g") for value in position))
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "ground_reference_z0",
            "type": "plane",
            "size": "2 2 0.02",
            "pos": "0 0 0",
            "rgba": "0.16 0.19 0.22 1",
            "contype": "0",
            "conaffinity": "0",
            "group": "2",
        },
    )


def calibrate_ground_reference(
    tree: ET.ElementTree,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=".bello-mini-ground-check-",
        suffix=".xml",
        delete=False,
    ) as handle:
        check_path = Path(handle.name)
    try:
        tree.write(check_path, encoding="utf-8", xml_declaration=True)
        correction_z = nominal_ground_translation(
            mujoco.MjModel.from_xml_path(str(check_path))
        )
        if correction_z != 0.0:
            worldbody = tree.getroot().find("worldbody")
            if worldbody is None:
                raise ValueError("compiled review MJCF has no worldbody")
            for body in worldbody.findall("body"):
                position = [
                    float(value) for value in body.get("pos", "0 0 0").split()
                ]
                position[2] += correction_z
                body.set(
                    "pos", " ".join(format(value, ".17g") for value in position)
                )
            tree.write(check_path, encoding="utf-8", xml_declaration=True)
        residual_z = nominal_ground_translation(
            mujoco.MjModel.from_xml_path(str(check_path))
        )
        if abs(residual_z) > 1e-12:
            raise ValueError(
                "review MJCF sole plane could not be calibrated to z=0: "
                f"residual {residual_z} m"
            )
    finally:
        check_path.unlink(missing_ok=True)


def rpy_to_quaternion(rpy: str) -> str:
    roll, pitch, yaw = (float(value) for value in rpy.split())
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    values = (
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        sy * cp * sr + cy * sp * cr,
        sy * cp * cr - cy * sp * sr,
    )
    return " ".join(format(value, ".10g") for value in values)


def ensure_asset(root: ET.Element) -> ET.Element:
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(0, asset)
    return asset


def add_visuals(root: ET.Element, urdf_path: Path, mjcf_path: Path) -> None:
    asset = ensure_asset(root)
    bodies = {
        body.get("name"): body
        for body in root.findall(".//body")
        if body.get("name") is not None
    }
    urdf = ET.parse(urdf_path).getroot()
    for link in urdf.findall("link"):
        link_name = link.get("name")
        body = bodies.get(link_name)
        if link_name is None or body is None:
            continue
        for visual_index, visual in enumerate(link.findall("visual")):
            name = visual.get("name", f"{link_name}_visual_{visual_index}")
            origin = visual.find("origin")
            attributes = {
                "name": name,
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
                "mass": "0",
                "rgba": (
                    VISUAL_RGBA["endpoint"]
                    if link_name.endswith("end_effector_sphere_link")
                    else link_color(link_name)
                ),
            }
            if origin is not None:
                attributes["pos"] = origin.get("xyz", "0 0 0")
                attributes["quat"] = rpy_to_quaternion(origin.get("rpy", "0 0 0"))
            mesh = visual.find("./geometry/mesh")
            sphere = visual.find("./geometry/sphere")
            if mesh is not None:
                source = (urdf_path.parent / mesh.attrib["filename"]).resolve()
                mesh_name = f"{name}_mesh"
                mesh_attributes = {
                    "name": mesh_name,
                    "file": os.path.relpath(source, mjcf_path.parent),
                }
                if "scale" in mesh.attrib:
                    mesh_attributes["scale"] = mesh.attrib["scale"]
                ET.SubElement(asset, "mesh", mesh_attributes)
                attributes.update({"type": "mesh", "mesh": mesh_name})
            elif sphere is not None:
                attributes.update({"type": "sphere", "size": sphere.attrib["radius"]})
            else:
                continue
            body.append(ET.Element("geom", attributes))


def build_review_mjcf(
    urdf_path: Path, output_path: Path, *, width: int, height: int
) -> None:
    source_model = mujoco.MjModel.from_xml_path(str(urdf_path))
    ground_translation_z = nominal_ground_translation(source_model)
    with tempfile.TemporaryDirectory(prefix="bello-mini-review-") as directory:
        compiled = Path(directory) / "compiled.xml"
        mujoco.mj_saveLastXML(str(compiled), source_model)
        tree = ET.parse(compiled)
    root = tree.getroot()
    for geom in root.findall(".//geom"):
        geom.set("rgba", collision_color(geom))
        geom.set("group", "0")
        if geom.get("name", "").endswith("end_effector_sphere_link_collision"):
            geom.set("mass", "0")
    apply_ground_reference(root, ground_translation_z)
    add_visuals(root, urdf_path, output_path)
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    headlight.set("ambient", "0.35 0.35 0.35")
    headlight.set("diffuse", "0.8 0.8 0.8")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", str(width))
    global_visual.set("offheight", str(height))
    calibrate_ground_reference(tree, output_path)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def add_arrow(
    geom: mujoco.MjvGeom,
    origin: np.ndarray,
    direction: np.ndarray,
    *,
    length: float,
    width: float,
    rgba: np.ndarray,
) -> None:
    target = origin + direction * length
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        origin,
        np.eye(3).reshape(-1),
        rgba,
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        width,
        origin,
        target,
    )


def add_joint_arrows(
    scene: mujoco.MjvScene,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> None:
    length = 0.065
    rgba = np.asarray([1.0, 0.72, 0.16, 1.0], dtype=np.float32)
    for joint_id in range(model.njnt):
        if scene.ngeom >= scene.maxgeom:
            raise ValueError("review scene ran out of geometry slots")
        axis = np.asarray(data.xaxis[joint_id])
        add_arrow(
            scene.geoms[scene.ngeom],
            np.asarray(data.xanchor[joint_id]),
            axis,
            length=length,
            width=0.006,
            rgba=rgba,
        )
        scene.ngeom += 1


def add_camera_frame_axes(
    scene: mujoco.MjvScene,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    frame_name: str,
) -> None:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, frame_name)
    if body_id < 0:
        raise ValueError(f"review model has no body {frame_name}")
    origin = np.asarray(data.xpos[body_id])
    rotation = np.asarray(data.xmat[body_id]).reshape(3, 3)
    length = 0.012
    for axis_index, rgba in enumerate(CAMERA_AXIS_RGBA):
        if scene.ngeom >= scene.maxgeom:
            raise ValueError("review scene ran out of geometry slots")
        axis = rotation[:, axis_index]
        add_arrow(
            scene.geoms[scene.ngeom],
            origin,
            axis,
            length=length,
            width=0.0012,
            rgba=rgba,
        )
        scene.ngeom += 1


def caption(
    image: np.ndarray,
    title: str,
    total_mass: float,
    *,
    camera_frame: str | None = None,
) -> Image.Image:
    value = Image.fromarray(image)
    canvas = Image.new("RGB", (value.width, value.height + 72), (12, 15, 20))
    canvas.paste(value, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, value.height + 12), title, fill=(239, 242, 246))
    optical = camera_frame is not None and camera_frame.endswith("_optical_frame")
    detail = (
        (
            "+X image-right red | +Y image-down green | +Z optical-forward blue"
            if optical
            else "+X forward red | +Y camera-left green | +Z up blue"
        )
        + " | massless frame; camera mass stays in pelvis at CAD pose"
        if camera_frame is not None
        else (
            "collision: cyan segments, green boxes, purple joint/wrist spheres, "
            f"pink endpoint spheres | soles z=0 | +axis yellow | mass {total_mass:.3f} kg"
        )
    )
    draw.text((18, value.height + 38), detail, fill=(174, 184, 198))
    return canvas


def render_images(
    mjcf_path: Path,
    image_dir: Path,
    *,
    width: int,
    height: int,
) -> None:
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    image_dir.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=height, width=width, max_geom=2000)
    camera = mujoco.MjvCamera()
    option = mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    total_mass = float(np.sum(model.body_mass))
    endpoint_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        ).endswith("end_effector_sphere_link_collision")
    ]
    visual_mask = model.geom_group == 1
    endpoint_lookat = (
        np.mean(data.geom_xpos[endpoint_geom_ids], axis=0)
        if endpoint_geom_ids
        else model.stat.center
    )
    camera_frame_lookats = {}
    for frame_name in CAMERA_REVIEW_FRAMES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, frame_name)
        if body_id < 0:
            raise ValueError(f"review model has no body {frame_name}")
        camera_frame_lookats[frame_name] = np.asarray(data.xpos[body_id])
    mount_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "astra_mount_frame"
    )
    mount_rotation = np.asarray(data.xmat[mount_body_id]).reshape(3, 3)
    camera_view_direction = mount_rotation @ np.asarray([-1.0, 0.08, -0.05])
    camera_view_direction /= np.linalg.norm(camera_view_direction)
    camera_view_azimuth = math.degrees(
        math.atan2(camera_view_direction[1], camera_view_direction[0])
    )
    camera_view_elevation = math.degrees(
        math.asin(float(camera_view_direction[2]))
    )
    full_distance = model.stat.extent * 1.30
    views = [
        ("perspective_overlay", 135.0, -8.0, True, True, model.stat.center, full_distance, None),
        ("front_overlay", 180.0, 0.0, True, True, model.stat.center, full_distance, None),
        ("side_overlay", 90.0, 0.0, True, True, model.stat.center, full_distance, None),
        ("collision_only", 135.0, -8.0, False, True, model.stat.center, full_distance, None),
        ("visual_only", 135.0, -8.0, True, False, model.stat.center, full_distance, None),
        ("eef_front_detail", 180.0, 0.0, True, True, endpoint_lookat, 0.56, None),
    ]
    views.extend(
        (
            f"camera_{frame_name}",
            camera_view_azimuth,
            camera_view_elevation,
            True,
            False,
            camera_frame_lookats[frame_name],
            0.12,
            frame_name,
        )
        for frame_name in CAMERA_REVIEW_FRAMES
    )
    for (
        name,
        azimuth,
        elevation,
        show_visual,
        show_collision,
        lookat,
        distance,
        camera_frame,
    ) in views:
        camera.azimuth = azimuth
        camera.elevation = elevation
        camera.lookat[:] = lookat
        camera.distance = distance
        collision_mask = model.geom_group == 0
        model.geom_rgba[collision_mask, 3] = 0.68 if not show_visual else 0.30
        model.geom_rgba[visual_mask, 3] = 1.0
        if camera_frame is not None:
            for geom_id in np.flatnonzero(visual_mask):
                geom_name = (
                    mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)
                    )
                    or ""
                )
                model.geom_rgba[geom_id, 3] = (
                    0.52 if geom_name == "astra_mount_frame_visual" else 0.0
                )
        option.geomgroup[:] = 1
        option.geomgroup[0] = int(show_collision)
        option.geomgroup[1] = int(show_visual)
        renderer.update_scene(data, camera=camera, scene_option=option)
        if camera_frame is not None:
            add_camera_frame_axes(renderer.scene, model, data, camera_frame)
        else:
            add_joint_arrows(renderer.scene, model, data)
        image = renderer.render()
        title = name.replace("_", " ").title()
        caption(
            image,
            title,
            total_mass,
            camera_frame=camera_frame,
        ).save(image_dir / f"{name}.png")
    renderer.close()


def main() -> None:
    args = parse_args()
    urdf_path = args.urdf.expanduser().resolve()
    mjcf_path = args.mjcf_output.expanduser().resolve()
    build_review_mjcf(
        urdf_path,
        mjcf_path,
        width=args.width,
        height=args.height,
    )
    render_images(
        mjcf_path,
        args.image_dir.expanduser().resolve(),
        width=args.width,
        height=args.height,
    )
    print(f"wrote review MJCF: {mjcf_path}")
    print(f"wrote review images: {args.image_dir}")


if __name__ == "__main__":
    main()
