from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
import trimesh

from bello_mujoco.actuators import ActuatorConfig
from bello_mujoco.contracts import SIM_TIMESTEP

CollisionMode = Literal["mesh", "box"]
TRANSPARENT_COLLISION_RGBA = "0.8 0.8 0.8 0"
ACCELERATION_STAGE_SITE = "bello_acceleration_stage_site"
ACCELERATION_STAGE_SENSOR = "bello_acceleration_stage_sensor"


@dataclass(frozen=True)
class ImuSensorConfig:
    enabled: bool = False
    body_name: str = "pelvis_link"


@dataclass(frozen=True)
class AnkleDifferentialConfig:
    motor_gain_scale: float = 0.5
    motor_effort_limit_scale: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("motor_gain_scale", self.motor_gain_scale),
            ("motor_effort_limit_scale", self.motor_effort_limit_scale),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"ankle_differential_config.{name} must be positive, got {value}"
                )


def compile_urdf_to_mjcf(
    urdf_path: Path,
    mjcf_path: Path,
    *,
    collision_mode: CollisionMode = "mesh",
    neighbor_exclusion_depth: int = 2,
    include_visuals: bool = False,
    actuator_config: ActuatorConfig = ActuatorConfig(),
    actuated_joint_order: tuple[str, ...] = (),
    enforce_policy_effort_limits: bool = True,
    imu_config: ImuSensorConfig = ImuSensorConfig(),
    ankle_differential_config: AnkleDifferentialConfig = AnkleDifferentialConfig(),
) -> None:
    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    tmp_path = mjcf_path.with_suffix(".compiled.xml")
    mujoco.mj_saveLastXML(str(tmp_path), model)

    tree = ET.parse(tmp_path)
    root = tree.getroot()
    canonicalize_compiled_mesh_metadata(root)
    ensure_option(root)
    ensure_free_root_joint(root)
    ensure_acceleration_stage_sensor(root)
    ensure_visual_geoms(root)
    apply_collision_mode(root, mjcf_path.parent, collision_mode)
    if include_visuals:
        inject_visual_geoms(root, urdf_path, mjcf_path.parent)
        ensure_visual_geoms(root)
        ensure_collision_geoms_transparent(root)
    ensure_neighbor_collision_excludes(root, depth=neighbor_exclusion_depth)
    ensure_position_actuators(
        root,
        actuator_config,
        actuated_joint_order,
        enforce_policy_effort_limits=enforce_policy_effort_limits,
    )
    ensure_ankle_differential_tendons(root)
    replace_ankle_joint_actuators(
        root,
        actuator_config,
        ankle_differential_config,
        enforce_policy_effort_limits=enforce_policy_effort_limits,
    )
    ensure_imu_sensors(root, imu_config)
    ensure_home_keyframe(root, actuator_config)

    ET.indent(tree, space="  ")
    tree.write(mjcf_path, encoding="utf-8", xml_declaration=True)
    tmp_path.unlink(missing_ok=True)

    mujoco.MjModel.from_xml_path(str(mjcf_path))


def canonicalize_compiled_mesh_metadata(root: ET.Element) -> None:
    """Remove loader-inferred metadata that MuJoCo serializes inconsistently."""

    for mesh in root.findall(".//mesh"):
        mesh.attrib.pop("content_type", None)


def ensure_ankle_differential_tendons(root: ET.Element) -> None:
    joint_names = {
        joint.get("name")
        for joint in root.findall(".//joint")
        if joint.get("name") is not None
    }
    required_joint_names = {
        f"{side}_ankle_{axis}_joint"
        for side in ("left", "right")
        for axis in ("pitch", "roll")
    }
    present_joint_names = required_joint_names & joint_names
    if present_joint_names != required_joint_names:
        missing = required_joint_names - present_joint_names
        raise ValueError(
            "ankle differential requires MuJoCo joints: " + ", ".join(sorted(missing))
        )

    tendon = root.find("tendon")
    if tendon is None:
        tendon = ET.SubElement(root, "tendon")
    existing_names = {
        item.get("name") for item in tendon if item.get("name") is not None
    }

    for side in ("left", "right"):
        pitch_joint = f"{side}_ankle_pitch_joint"
        roll_joint = f"{side}_ankle_roll_joint"
        for motor_index, pitch_coefficient in ((1, -1.0), (2, 1.0)):
            tendon_name = f"{side}_ankle_motor_{motor_index}"
            if tendon_name in existing_names:
                raise ValueError(
                    f"generated MJCF already defines tendon {tendon_name!r}"
                )
            fixed = ET.SubElement(tendon, "fixed", {"name": tendon_name})
            ET.SubElement(
                fixed,
                "joint",
                {"joint": pitch_joint, "coef": f"{pitch_coefficient:g}"},
            )
            ET.SubElement(
                fixed,
                "joint",
                {"joint": roll_joint, "coef": "1"},
            )
            existing_names.add(tendon_name)


def replace_ankle_joint_actuators(
    root: ET.Element,
    config: ActuatorConfig,
    differential_config: AnkleDifferentialConfig,
    *,
    enforce_policy_effort_limits: bool = True,
) -> None:
    actuator = root.find("actuator")
    if actuator is None:
        raise ValueError("generated MJCF has no actuator section")

    joint_configs = config.by_name
    for side in ("left", "right"):
        pitch_joint = f"{side}_ankle_pitch_joint"
        roll_joint = f"{side}_ankle_roll_joint"
        pitch_config = joint_configs[pitch_joint]
        roll_config = joint_configs[roll_joint]
        if pitch_config.mode != roll_config.mode:
            raise ValueError(
                f"{side} ankle differential joints must use the same actuator mode"
            )
        if pitch_config.gains != roll_config.gains:
            raise ValueError(
                f"{side} ankle differential joints must use identical gains"
            )
        if pitch_config.effort_limit != roll_config.effort_limit:
            raise ValueError(
                f"{side} ankle differential joints must use identical effort limits"
            )

        pitch_range = configured_joint_range(root, pitch_joint, config)
        roll_range = configured_joint_range(root, roll_joint, config)
        replacements = (
            (pitch_joint, 1, -1.0),
            (roll_joint, 2, 1.0),
        )
        for logical_joint, motor_index, pitch_coefficient in replacements:
            direct = actuator.find(f"./position[@joint='{logical_joint}']")
            if direct is None:
                raise ValueError(
                    f"missing generated position actuator for {logical_joint!r}"
                )
            index = list(actuator).index(direct)
            actuator.remove(direct)

            tendon_name = f"{side}_ankle_motor_{motor_index}"
            low, high = differential_coordinate_range(
                pitch_range, roll_range, pitch_coefficient
            )
            default = (
                pitch_coefficient * pitch_config.default_pos + roll_config.default_pos
            )
            kp, kv = (
                gain * differential_config.motor_gain_scale
                for gain in pitch_config.gains
            )
            suffix = "pos" if pitch_config.mode == "policy" else "hold"
            attributes = {
                "name": f"{tendon_name}_{suffix}",
                "tendon": tendon_name,
                "kp": f"{kp:g}",
                "kv": f"{kv:g}",
                "ctrlrange": f"{low:g} {high:g}",
                "ctrllimited": "false",
                "user": f"{default:g}",
            }
            force_range = actuator_force_range(
                root, logical_joint, pitch_config.effort_limit
            )
            if force_range is not None:
                force_low, force_high = force_range
                force_low *= differential_config.motor_effort_limit_scale
                force_high *= differential_config.motor_effort_limit_scale
                attributes["forcerange"] = f"{force_low:g} {force_high:g}"
                attributes["forcelimited"] = str(
                    pitch_config.mode != "policy" or enforce_policy_effort_limits
                ).lower()
            actuator.insert(index, ET.Element("position", attributes))


def configured_joint_range(
    root: ET.Element, joint_name: str, config: ActuatorConfig
) -> tuple[float, float]:
    joint_config = config.by_name[joint_name]
    return (
        joint_config.soft_limit
        if joint_config.soft_limit is not None
        else joint_range(root, joint_name, config)
    )


def differential_coordinate_range(
    pitch_range: tuple[float, float],
    roll_range: tuple[float, float],
    pitch_coefficient: float,
) -> tuple[float, float]:
    pitch_values = tuple(pitch_coefficient * value for value in pitch_range)
    return (
        roll_range[0] + min(pitch_values),
        roll_range[1] + max(pitch_values),
    )


def ensure_imu_sensors(root: ET.Element, config: ImuSensorConfig) -> None:
    if not config.enabled:
        return
    if not config.body_name:
        raise ValueError("IMU body name must not be empty")
    body = root.find(f".//body[@name='{config.body_name}']")
    if body is None:
        raise ValueError(f"IMU body not found in generated MJCF: {config.body_name!r}")

    site_name = "bello_imu_site"
    if root.find(f".//site[@name='{site_name}']") is not None:
        raise ValueError(f"generated MJCF already defines site {site_name!r}")
    body.append(
        ET.Element(
            "site",
            {
                "name": site_name,
                "pos": "0 0 0",
                "quat": "1 0 0 0",
                "size": "0.005",
                "rgba": "0 1 0 0",
            },
        )
    )

    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(root, "sensor")
    ET.SubElement(sensor, "gyro", {"name": "bello_imu_gyro", "site": site_name})
    ET.SubElement(
        sensor,
        "accelerometer",
        {"name": "bello_imu_accelerometer", "site": site_name},
    )
    ET.SubElement(
        sensor,
        "framequat",
        {
            "name": "bello_imu_orientation",
            "objtype": "site",
            "objname": site_name,
        },
    )


def apply_collision_mode(
    root: ET.Element, mjcf_dir: Path, collision_mode: CollisionMode
) -> None:
    if collision_mode == "mesh":
        return
    if collision_mode != "box":
        raise ValueError(
            f"collision_mode must be 'mesh' or 'box', got {collision_mode!r}"
        )

    mesh_bounds = load_scaled_mesh_bounds(root, mjcf_dir)
    for geom in root.findall(".//geom[@type='mesh']"):
        if is_visual_geom(geom):
            continue
        mesh_name = geom.get("mesh")
        if mesh_name is None:
            continue
        if mesh_name not in mesh_bounds:
            continue
        center, half_size = mesh_bounds[mesh_name]
        quat = parse_vec(
            geom.get("quat", "1 0 0 0"), 4, np.asarray([1.0, 0.0, 0.0, 0.0])
        )
        pos = parse_vec(geom.get("pos", "0 0 0"), 3, np.zeros(3))
        adjusted_pos = pos + quat_rotate(quat, center)
        geom.attrib.pop("mesh", None)
        geom.set("type", "box")
        geom.set("pos", format_vec(adjusted_pos))
        geom.set("size", format_vec(half_size))


def ensure_visual_geoms(root: ET.Element) -> None:
    for geom in root.findall(".//geom[@type='mesh']"):
        if not is_visual_geom(geom):
            continue
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
        geom.set("group", "1")


def ensure_collision_geoms_transparent(root: ET.Element) -> None:
    for geom in root.findall(".//geom"):
        if is_visual_geom(geom) or is_noncolliding_visual_geom(geom):
            continue
        geom.set("rgba", TRANSPARENT_COLLISION_RGBA)


def is_visual_geom(geom: ET.Element) -> bool:
    name = geom.get("name", "")
    mesh_name = geom.get("mesh", "")
    return name.endswith("_visual") or mesh_name.endswith("_visual")


def is_noncolliding_visual_geom(geom: ET.Element) -> bool:
    return (
        geom.get("contype") == "0"
        and geom.get("conaffinity") == "0"
        and geom.get("group") == "1"
    )


def inject_visual_geoms(root: ET.Element, urdf_path: Path, mjcf_dir: Path) -> None:
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(0, asset)

    bodies = bodies_by_name(root)
    collision_bodies = bodies_by_collision_geom(root)
    existing_meshes = {mesh.get("name") for mesh in asset.findall("mesh")}
    existing_geoms = {geom.get("name") for geom in root.findall(".//geom")}

    urdf_root = ET.parse(urdf_path).getroot()
    for link in urdf_root.findall("link"):
        link_name = link.get("name")
        if link_name is None:
            continue
        body = bodies.get(link_name)
        if body is None:
            body = collision_bodies.get(f"{link_name}_collision")
        if body is None:
            continue
        for visual in link.findall("visual"):
            mesh = visual.find("./geometry/mesh")
            if mesh is None:
                continue
            visual_name = visual.get("name", f"{link_name}_visual")
            mesh_name = visual_name
            mesh_file = mesh.get("filename")
            if mesh_file is None:
                continue
            if mesh_name not in existing_meshes:
                attrs = {
                    "name": mesh_name,
                    "file": format_mesh_file(urdf_path, mjcf_dir, mesh_file),
                }
                scale = mesh.get("scale")
                if scale is not None:
                    attrs["scale"] = scale
                ET.SubElement(asset, "mesh", attrs)
                existing_meshes.add(mesh_name)
            if visual_name in existing_geoms:
                continue
            geom_attrs = {
                "name": visual_name,
                "type": "mesh",
                "mesh": mesh_name,
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
            }
            origin = visual.find("origin")
            if origin is not None:
                xyz = origin.get("xyz")
                rpy = origin.get("rpy")
                if xyz is not None:
                    geom_attrs["pos"] = xyz
                if rpy is not None:
                    geom_attrs["quat"] = format_vec(
                        rpy_to_quat(parse_vec(rpy, 3, np.zeros(3)))
                    )
            body.append(ET.Element("geom", geom_attrs))
            existing_geoms.add(visual_name)


def bodies_by_name(root: ET.Element) -> dict[str, ET.Element]:
    return {
        body.get("name"): body
        for body in root.findall(".//body")
        if body.get("name") is not None
    }


def bodies_by_collision_geom(root: ET.Element) -> dict[str, ET.Element]:
    bodies: dict[str, ET.Element] = {}
    for body in root.findall(".//body"):
        for geom in body.findall("geom"):
            name = geom.get("name")
            if name is not None:
                bodies[name] = body
    return bodies


def format_mesh_file(urdf_path: Path, mjcf_dir: Path, mesh_file: str) -> str:
    mesh_path = (urdf_path.parent / mesh_file).resolve()
    return str(Path("..") / mesh_path.relative_to(mjcf_dir.resolve().parent))


def rpy_to_quat(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return np.asarray(
        [
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            sy * cp * sr + cy * sp * cr,
            sy * cp * cr - cy * sp * sr,
        ],
        dtype=np.float64,
    )


def load_scaled_mesh_bounds(
    root: ET.Element, mjcf_dir: Path
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
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


def nearby_body_pairs(
    names: list[str], graph: dict[str, set[str]], depth: int
) -> list[tuple[str, str]]:
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


def ensure_free_root_joint(root: ET.Element) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF has no worldbody")
    if worldbody.find(".//freejoint") is not None:
        return

    robot_children = list(worldbody)
    if not robot_children:
        raise ValueError("MJCF worldbody has no robot geoms or bodies to wrap")
    root_body = ET.Element("body", {"name": "bello_root"})
    root_body.append(ET.Element("freejoint", {"name": "root_free"}))
    for child in robot_children:
        worldbody.remove(child)
        root_body.append(child)
    worldbody.append(root_body)


def ensure_acceleration_stage_sensor(root: ET.Element) -> None:
    root_body = root.find(".//body[@name='bello_root']")
    if root_body is None:
        raise ValueError("MJCF has no generated bello_root body")
    if root.find(f".//site[@name='{ACCELERATION_STAGE_SITE}']") is None:
        ET.SubElement(
            root_body,
            "site",
            {
                "name": ACCELERATION_STAGE_SITE,
                "size": "0.001",
                "rgba": "0 0 0 0",
            },
        )

    sensors = root.find("sensor")
    if sensors is None:
        sensors = ET.SubElement(root, "sensor")
    if sensors.find(f"./accelerometer[@name='{ACCELERATION_STAGE_SENSOR}']") is None:
        ET.SubElement(
            sensors,
            "accelerometer",
            {
                "name": ACCELERATION_STAGE_SENSOR,
                "site": ACCELERATION_STAGE_SITE,
            },
        )


def ensure_position_actuators(
    root: ET.Element,
    config: ActuatorConfig,
    actuated_joint_order: tuple[str, ...],
    *,
    enforce_policy_effort_limits: bool = True,
) -> None:
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    existing = {item.get("joint") for item in actuator.findall("position")}
    joints = {joint.get("name") for joint in root.findall(".//joint")}
    joint_configs = config.by_name
    for joint_name in actuated_joint_order:
        if joint_name in existing or joint_name not in joints:
            continue
        if joint_name not in joint_configs:
            raise ValueError(f"missing actuator config for joint {joint_name!r}")
        joint_config = joint_configs[joint_name]
        kp, kv = joint_config.gains
        lo, hi = (
            joint_config.soft_limit
            if joint_config.soft_limit is not None
            else joint_range(root, joint_name, config)
        )
        suffix = "pos" if joint_config.mode == "policy" else "hold"
        attributes = {
            "name": f"{joint_name.replace(' ', '_')}_{suffix}",
            "joint": joint_name,
            "kp": f"{kp:g}",
            "kv": f"{kv:g}",
            "ctrlrange": f"{lo:g} {hi:g}",
            "ctrllimited": "false",
            "user": f"{joint_config.default_pos:g}",
        }
        force_range = actuator_force_range(root, joint_name, joint_config.effort_limit)
        if force_range is not None:
            force_low, force_high = force_range
            attributes["forcerange"] = f"{force_low:g} {force_high:g}"
            attributes["forcelimited"] = str(
                joint_config.mode != "policy" or enforce_policy_effort_limits
            ).lower()
        ET.SubElement(
            actuator,
            "position",
            attributes,
        )


def ensure_home_keyframe(root: ET.Element, config: ActuatorConfig) -> None:
    keyframe = root.find("keyframe")
    if keyframe is None:
        keyframe = ET.SubElement(root, "keyframe")
    if keyframe.find("./key[@name='home']") is not None:
        return

    joint_qpos = [
        joint_default(joint.get("name", ""), config)
        for joint in root.findall("./worldbody//joint")
    ]
    qpos = " ".join(
        ["0", "0", "0.91", "1", "0", "0", "0", *[f"{value:g}" for value in joint_qpos]]
    )
    ctrl_values = [float(item.get("user", "0")) for item in keyframe_actuators(root)]
    ctrl = " ".join(f"{value:g}" for value in ctrl_values)
    ET.SubElement(keyframe, "key", {"name": "home", "qpos": qpos, "ctrl": ctrl})


def keyframe_actuators(root: ET.Element) -> list[ET.Element]:
    actuator = root.find("actuator")
    if actuator is None:
        return []
    return list(actuator.findall("position"))


def joint_default(joint_name: str, config: ActuatorConfig) -> float:
    return config.default_joint_pos_by_name.get(joint_name, 0.0)


def joint_range(
    root: ET.Element, joint_name: str, config: ActuatorConfig
) -> tuple[float, float]:
    eps = 1.0e-6
    joint = root.find(f".//joint[@name='{joint_name}']")
    if joint is None:
        raise ValueError(f"joint not found in MJCF: {joint_name}")
    range_attr = joint.get("range")
    if range_attr is None:
        value = config.default_joint_pos_by_name.get(joint_name, 0.0)
        return value - eps, value + eps
    lo_str, hi_str = range_attr.split()
    lo, hi = float(lo_str), float(hi_str)
    if lo == hi:
        return lo - eps, hi + eps
    return lo, hi


def actuator_force_range(
    root: ET.Element, joint_name: str, effort_limit: float | None
) -> tuple[float, float] | None:
    if effort_limit is not None:
        return -effort_limit, effort_limit
    joint = root.find(f".//joint[@name='{joint_name}']")
    if joint is None:
        raise ValueError(f"joint not found in MJCF: {joint_name}")
    range_attr = joint.get("actuatorfrcrange")
    if range_attr is None:
        return None
    lo_str, hi_str = range_attr.split()
    lo, hi = float(lo_str), float(hi_str)
    if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
        return None
    return lo, hi
