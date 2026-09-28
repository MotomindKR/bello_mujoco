from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from bello_mujoco.actuators import ActuatorConfig
from bello_mujoco.contracts import SIM_TIMESTEP
from bello_mujoco.model_validation import (
    validate_home_pose,
    validate_robot_collision_geometry,
)
from bello_mujoco.robot import CoupledActuatorGroup, PayloadAttachment, RobotDefinition
from bello_mujoco.models.serialization import save_model_xml

TRANSPARENT_COLLISION_RGBA = "0.8 0.8 0.8 0"
ACCELERATION_STAGE_SITE = "bello_acceleration_stage_site"
ACCELERATION_STAGE_SENSOR = "bello_acceleration_stage_sensor"


@dataclass(frozen=True)
class ImuSensorConfig:
    enabled: bool = False
    body_name: str | None = None


def compile_urdf_to_mjcf(
    urdf_path: Path,
    mjcf_path: Path,
    viewer_mjcf_path: Path,
    *,
    neighbor_exclusion_depth: int = 2,
    actuator_config: ActuatorConfig,
    actuated_joint_order: tuple[str, ...],
    enforce_policy_effort_limits: bool = True,
    imu_config: ImuSensorConfig = ImuSensorConfig(),
    robot: RobotDefinition,
    target_total_mass_kg: float | None = None,
    end_effector_payload_mass_kg: float = 0.0,
    end_effector_payload_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    tree = save_model_xml(viewer_mjcf_path, model)
    root = tree.getroot()
    root.find("compiler").set("meshdir", str(urdf_path.parent.resolve()))
    ensure_option(root)
    ensure_free_root_joint(root, robot)
    ensure_acceleration_stage_sensor(root, robot)
    keep_fixed_link_geoms_massless(root, massless_fixed_urdf_link_names(urdf_path))
    ensure_neighbor_collision_excludes(root, depth=neighbor_exclusion_depth)
    ensure_position_actuators(
        root,
        actuator_config,
        actuated_joint_order,
        enforce_policy_effort_limits=enforce_policy_effort_limits,
    )
    ensure_coupled_actuator_tendons(root, robot.coupled_actuator_groups)
    replace_coupled_joint_actuators(
        root,
        actuator_config,
        robot.coupled_actuator_groups,
        enforce_policy_effort_limits=enforce_policy_effort_limits,
    )
    ensure_imu_sensors(root, imu_config, robot)
    ensure_home_keyframe(root, actuator_config, robot)
    if target_total_mass_kg is not None:
        scale = target_total_mass_kg / float(model.body_mass.sum())
        for inertial in root.findall(".//inertial"):
            inertial.set("mass", f"{float(inertial.attrib['mass']) * scale:.17g}")
            for key in ("diaginertia", "fullinertia"):
                if key in inertial.attrib:
                    inertial.set(
                        key,
                        format_vec(
                            np.fromstring(inertial.attrib[key], sep=" ") * scale
                        ),
                    )
    if end_effector_payload_mass_kg:
        add_end_effector_payloads(
            root,
            robot.payload_attachments,
            end_effector_payload_mass_kg,
            end_effector_payload_offset_m,
        )
    ensure_collision_geoms_transparent(root)
    ET.indent(tree, space="  ")
    tree.write(viewer_mjcf_path, encoding="utf-8", xml_declaration=True)
    # Training omits visual geoms, not physical links or collision shapes.
    for body in root.findall(".//body"):
        for geom in body.findall("geom"):
            if is_noncolliding_visual_geom(geom):
                body.remove(geom)
    used = {geom.get("mesh") for geom in root.findall(".//geom")}
    asset = root.find("asset")
    if asset is not None:
        for mesh in asset.findall("mesh"):
            if mesh.get("name") not in used:
                asset.remove(mesh)
    tree.write(mjcf_path, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    validate_robot_collision_geometry(model)
    validate_home_pose(model)
    mujoco.MjModel.from_xml_path(str(viewer_mjcf_path))


def add_end_effector_payloads(
    root: ET.Element,
    attachments: tuple[PayloadAttachment, ...],
    mass_kg: float,
    offset_m: tuple[float, float, float],
) -> None:
    if not attachments:
        raise ValueError("robot has no payload attachment frames")
    for attachment in attachments:
        body = root.find(f".//body[@name='{attachment.link_name}']")
        if body is None:
            raise ValueError(
                f"missing payload attachment frame: {attachment.link_name}"
            )
        name = f"{attachment.link_name}_payload"
        if root.find(f".//body[@name='{name}']") is not None:
            raise ValueError(f"payload body already exists: {name}")
        payload = ET.SubElement(
            body,
            "body",
            name=name,
            pos=format_vec(np.asarray(attachment.center_m) + offset_m),
        )
        inertia = 0.4 * mass_kg * attachment.radius_m**2
        ET.SubElement(
            payload,
            "inertial",
            mass=str(mass_kg),
            pos="0 0 0",
            diaginertia=f"{inertia} {inertia} {inertia}",
        )


def massless_fixed_urdf_link_names(urdf_path: Path) -> set[str]:
    """Return inertial-free fixed links and reject inertial-free dynamic links."""

    root = ET.parse(urdf_path).getroot()
    incoming_joint = {
        child.attrib["link"]: joint
        for joint in root.findall("joint")
        for child in joint.findall("child")
        if "link" in child.attrib
    }
    names: set[str] = set()
    for link in root.findall("link"):
        if link.find("inertial") is not None:
            continue
        name = link.get("name")
        if name is None:
            raise ValueError("URDF contains an unnamed link without inertial data")
        joint = incoming_joint.get(name)
        if joint is None or joint.get("type") != "fixed":
            raise ValueError(f"dynamic link {name!r} is missing explicit inertial data")
        names.add(name)
    return names


def keep_fixed_link_geoms_massless(
    root: ET.Element, massless_fixed_links: set[str]
) -> None:
    """Prevent MuJoCo from inferring mass from geoms on fixed massless frames."""

    for name in massless_fixed_links:
        body = root.find(f".//body[@name='{name}']")
        if body is None:
            raise ValueError(f"compiled MJCF is missing fixed link body {name!r}")
        for geom in body.findall("geom"):
            geom.set("mass", "0")


def ensure_coupled_actuator_tendons(
    root: ET.Element,
    groups: tuple[CoupledActuatorGroup, ...],
) -> None:
    if not groups:
        return
    joint_names = {
        joint.get("name")
        for joint in root.findall(".//joint")
        if joint.get("name") is not None
    }
    required_joint_names = {name for group in groups for name in group.joint_names}
    present_joint_names = required_joint_names & joint_names
    if present_joint_names != required_joint_names:
        missing = required_joint_names - present_joint_names
        raise ValueError(
            "coupled actuators require MuJoCo joints: " + ", ".join(sorted(missing))
        )

    tendon = root.find("tendon")
    if tendon is None:
        tendon = ET.SubElement(root, "tendon")
    existing_names = {
        item.get("name") for item in tendon if item.get("name") is not None
    }

    for group in groups:
        for coupled in group.actuators:
            tendon_name = coupled.tendon_name
            if tendon_name in existing_names:
                raise ValueError(
                    f"generated MJCF already defines tendon {tendon_name!r}"
                )
            fixed = ET.SubElement(tendon, "fixed", {"name": tendon_name})
            for joint_name, coefficient in coupled.joint_coefficients:
                ET.SubElement(
                    fixed,
                    "joint",
                    {"joint": joint_name, "coef": f"{coefficient:g}"},
                )
            existing_names.add(tendon_name)


def replace_coupled_joint_actuators(
    root: ET.Element,
    config: ActuatorConfig,
    groups: tuple[CoupledActuatorGroup, ...],
    *,
    enforce_policy_effort_limits: bool = True,
) -> None:
    if not groups:
        return
    actuator = root.find("actuator")
    if actuator is None:
        raise ValueError("generated MJCF has no actuator section")

    joint_configs = config.by_name
    for group in groups:
        group_configs = tuple(joint_configs[name] for name in group.joint_names)
        reference_config = group_configs[0]
        if any(config.mode != reference_config.mode for config in group_configs[1:]):
            raise ValueError(f"{group.name} joints must use the same actuator mode")
        if any(config.gains != reference_config.gains for config in group_configs[1:]):
            raise ValueError(f"{group.name} joints must use identical gains")
        if any(
            config.effort_limit != reference_config.effort_limit
            for config in group_configs[1:]
        ):
            raise ValueError(f"{group.name} joints must use identical effort limits")

        joint_ranges = {
            name: configured_joint_range(root, name, config)
            for name in group.joint_names
        }
        for coupled in group.actuators:
            logical_joint = coupled.logical_joint_name
            direct = actuator.find(f"./position[@joint='{logical_joint}']")
            if direct is None:
                raise ValueError(
                    f"missing generated position actuator for {logical_joint!r}"
                )
            index = list(actuator).index(direct)
            actuator.remove(direct)

            low, high = coupled_coordinate_range(
                tuple(
                    (joint_ranges[name], coefficient)
                    for name, coefficient in coupled.joint_coefficients
                )
            )
            default = sum(
                coefficient * joint_configs[name].default_pos
                for name, coefficient in coupled.joint_coefficients
            )
            kp, kv = reference_config.gains
            suffix = "pos" if reference_config.mode == "policy" else "hold"
            attributes = {
                "name": f"{coupled.tendon_name}_{suffix}",
                "tendon": coupled.tendon_name,
                "kp": f"{kp:g}",
                "kv": f"{kv:g}",
                "ctrlrange": f"{low:g} {high:g}",
                "ctrllimited": "false",
                "user": f"{default:g}",
            }
            force_range = actuator_force_range(
                root, logical_joint, reference_config.effort_limit
            )
            if force_range is not None:
                force_low, force_high = force_range
                attributes["forcerange"] = f"{force_low:g} {force_high:g}"
                attributes["forcelimited"] = str(
                    reference_config.mode != "policy" or enforce_policy_effort_limits
                ).lower()
            actuator.insert(index, ET.Element("position", attributes))


def configured_joint_range(
    root: ET.Element, joint_name: str, config: ActuatorConfig
) -> tuple[float, float]:
    joint_config = config.by_name[joint_name]
    physical = joint_range(root, joint_name, config)
    if joint_config.soft_limit is None:
        return physical
    lower, upper = joint_config.soft_limit
    if not physical[0] <= lower < upper <= physical[1]:
        raise ValueError(
            f"soft limit must stay within physical joint range: {joint_name}"
        )
    return lower, upper


def coupled_coordinate_range(
    coordinates: tuple[tuple[tuple[float, float], float], ...],
) -> tuple[float, float]:
    low = 0.0
    high = 0.0
    for (coordinate_low, coordinate_high), coefficient in coordinates:
        values = (coefficient * coordinate_low, coefficient * coordinate_high)
        low += min(values)
        high += max(values)
    return low, high


def ensure_imu_sensors(
    root: ET.Element,
    config: ImuSensorConfig,
    robot: RobotDefinition,
) -> None:
    if not config.enabled:
        return
    if config.body_name == "":
        raise ValueError("IMU body name must not be empty")
    body_name = config.body_name or robot.default_imu_body_name
    body = root.find(f".//body[@name='{body_name}']")
    if body is None:
        raise ValueError(f"IMU body not found in generated MJCF: {body_name!r}")

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


def bodies_by_name(root: ET.Element) -> dict[str, ET.Element]:
    return {
        body.get("name"): body
        for body in root.findall(".//body")
        if body.get("name") is not None
    }


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


def parse_vec(value: str, expected_size: int, default: np.ndarray) -> np.ndarray:
    array = np.fromstring(value, sep=" ", dtype=np.float64)
    if array.size != expected_size:
        return default.astype(np.float64)
    return array


def format_vec(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.17g}" for value in values)


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


def ensure_free_root_joint(root: ET.Element, robot: RobotDefinition) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF has no worldbody")
    if worldbody.find(".//freejoint") is not None:
        return

    robot_children = list(worldbody)
    if not robot_children:
        raise ValueError("MJCF worldbody has no robot geoms or bodies to wrap")
    root_body = ET.Element("body", {"name": robot.root_body_name})
    root_body.append(ET.Element("freejoint", {"name": robot.root_joint_name}))
    for child in robot_children:
        worldbody.remove(child)
        root_body.append(child)
    worldbody.append(root_body)


def ensure_acceleration_stage_sensor(root: ET.Element, robot: RobotDefinition) -> None:
    root_body = root.find(f".//body[@name='{robot.root_body_name}']")
    if root_body is None:
        raise ValueError(f"MJCF has no generated root body {robot.root_body_name!r}")
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
        lo, hi = configured_joint_range(root, joint_name, config)
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


def ensure_home_keyframe(
    root: ET.Element,
    config: ActuatorConfig,
    robot: RobotDefinition,
) -> None:
    keyframe = root.find("keyframe")
    if keyframe is None:
        keyframe = ET.SubElement(root, "keyframe")
    if keyframe.find("./key[@name='home']") is not None:
        return

    joint_qpos = [
        joint_default(joint.get("name", ""), config)
        for joint in root.findall("./worldbody//joint")
    ]
    root_qpos = (*robot.default_root_position, *robot.default_root_quaternion)
    qpos = " ".join(
        [
            *[f"{value:g}" for value in root_qpos],
            *[f"{value:g}" for value in joint_qpos],
        ]
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
