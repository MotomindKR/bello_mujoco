"""Shared rigid object physics for training and native evaluation."""

from dataclasses import dataclass
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from bello_mujoco.terrain import format_vec


GRIP_AXIS_SPAN_M = 0.10
GRIP_ANCHORS = ("grip", "grip_axis")


def grip_equality_ids(model):
    return np.array(
        [
            model.equality(f"{side}_ideal_{anchor}").id
            for side in ("left", "right")
            for anchor in GRIP_ANCHORS
        ]
    )


def grip_axis_errors(model, data):
    bar_axis = data.xmat[model.body("barbell").id].reshape(3, 3)[:, 1]
    errors = []
    for side in ("left", "right"):
        body = model.site_bodyid[model.site(f"{side}_hand_grip").id]
        hand_axis = data.xmat[body].reshape(3, 3)[:, 1]
        errors.append(
            np.arctan2(
                np.linalg.norm(np.cross(hand_axis, bar_axis)),
                np.dot(hand_axis, bar_axis),
            )
        )
    return np.asarray(errors)


def grip_wrench_columns(model, data, side):
    """Five independent hinge reactions, without redundant anchor-force columns."""
    hand, bar = (model.site(f"{side}_{kind}_grip").id for kind in ("hand", "bar"))
    hand_pos = np.zeros((3, model.nv))
    bar_pos = np.zeros_like(hand_pos)
    hand_rot = np.zeros_like(hand_pos)
    bar_rot = np.zeros_like(hand_pos)
    mujoco.mj_jacSite(model, data, hand_pos, hand_rot, hand)
    mujoco.mj_jacSite(model, data, bar_pos, bar_rot, bar)
    transverse_axes = data.site_xmat[bar].reshape(3, 3)[:, [0, 2]]
    return np.column_stack(
        ((hand_pos - bar_pos).T, (hand_rot - bar_rot).T @ transverse_axes)
    )


@dataclass(frozen=True)
class Grip:
    body: str
    side: str
    grip_local_m: tuple[float, float, float] = (0.0, 0.0, -0.09)


@dataclass(frozen=True)
class RigidBarConfig:
    mass_kg: float = 30.0
    bar_length_m: float = 1.2
    shaft_diameter_m: float = 0.03
    end_weight_radius_m: float = 0.10
    end_weight_width_m: float = 0.15
    grip_spacing_m: float = 0.40
    hands: tuple[Grip, ...] = (
        Grip("left_end_effector_sphere_link", "left"),
        Grip("right_end_effector_sphere_link", "right"),
    )
    grip_solref: tuple[float, float] = (0.02, 1.0)

    def __post_init__(self):
        values = (
            self.mass_kg,
            self.bar_length_m,
            self.shaft_diameter_m,
            self.end_weight_radius_m,
            self.end_weight_width_m,
            self.grip_spacing_m,
            *self.grip_solref,
        )
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError(
                "bar mass, dimensions and solref must be finite and positive"
            )
        if self.grip_spacing_m >= self.bar_length_m - 2 * self.end_weight_width_m:
            raise ValueError("grips must lie between the end weights")
        if len(self.hands) != 2 or {h.side for h in self.hands} != {"left", "right"}:
            raise ValueError("bar requires one left and one right grip")
        if len({h.body for h in self.hands}) != 2:
            raise ValueError("bar grip bodies must differ")
        for hand in self.hands:
            if len(hand.grip_local_m) != 3 or not np.isfinite(hand.grip_local_m).all():
                raise ValueError("grip coordinates must contain three finite values")


def two_end_weight_inertia(mass, radius, width, center_offset):
    values = np.asarray([mass, radius, width, center_offset])
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("bar dimensions and mass must be finite and positive")
    transverse = mass * ((3 * radius**2 + width**2) / 12 + center_offset**2)
    return np.asarray([transverse, mass * radius**2 / 2, transverse])


def add_barbell(root, geometry, center, mass, solref):
    """Add a free bar and shaft-axis hinges; no world/root/foot attachment."""
    world = root.find("worldbody")
    if world is None or root.find(".//body[@name='barbell']") is not None:
        raise ValueError("scene must have a worldbody and no existing barbell")
    bar = ET.SubElement(world, "body", name="barbell", pos=format_vec(center))
    ET.SubElement(bar, "freejoint", name="barbell_free")
    offset = (geometry.bar_length_m - geometry.end_weight_width_m) / 2
    inertia = two_end_weight_inertia(
        mass, geometry.end_weight_radius_m, geometry.end_weight_width_m, offset
    )
    ET.SubElement(
        bar, "inertial", pos="0 0 0", mass=str(mass), diaginertia=format_vec(inertia)
    )
    # All mass is represented exactly once in the explicit inertial above.
    ET.SubElement(
        bar,
        "geom",
        name="barbell_shaft",
        type="cylinder",
        size=format_vec((geometry.shaft_diameter_m / 2, geometry.bar_length_m / 2)),
        quat="0.7071067811865476 0.7071067811865476 0 0",
        mass="0",
        group="1",
        rgba="0.7 0.75 0.8 1",
    )
    for side, sign in (("left", 1), ("right", -1)):
        ET.SubElement(
            bar,
            "geom",
            name=f"barbell_{side}_weight",
            type="cylinder",
            size=format_vec(
                (geometry.end_weight_radius_m, geometry.end_weight_width_m / 2)
            ),
            pos=format_vec((0, sign * offset, 0)),
            quat="0.7071067811865476 0.7071067811865476 0 0",
            mass="0",
            group="1",
            rgba="0.92 0.62 0.22 1",
        )
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for hand in geometry.hands:
        body = root.find(f".//body[@name='{hand.body}']")
        if body is None:
            raise ValueError(f"missing hand attachment body: {hand.body}")
        sign = 1 if hand.side == "left" else -1
        # Two separated points define a hinge along local +Y, the shaft axis.
        # This leaves axial spin free while resisting off-axis hand rotation.
        for anchor, offset in zip(GRIP_ANCHORS, (0.0, GRIP_AXIS_SPAN_M), strict=True):
            offset *= sign
            ET.SubElement(
                body,
                "site",
                name=f"{hand.side}_hand_{anchor}",
                pos=format_vec(np.asarray(hand.grip_local_m) + [0, offset, 0]),
                size="0.005",
            )
            ET.SubElement(
                bar,
                "site",
                name=f"{hand.side}_bar_{anchor}",
                pos=format_vec((0, sign * geometry.grip_spacing_m / 2 + offset, 0)),
                size="0.005",
            )
            ET.SubElement(
                equality,
                "connect",
                name=f"{hand.side}_ideal_{anchor}",
                site1=f"{hand.side}_hand_{anchor}",
                site2=f"{hand.side}_bar_{anchor}",
                solref=format_vec(solref),
            )
        # The sphere is not a finger model; do not let it repel its gripped shaft.
        ET.SubElement(contact, "exclude", body1=hand.body, body2="barbell")


def attach_bar_to_model(path, config):
    """Extend generated MJCF and each keyframe, preserving robot coordinates."""
    root = ET.parse(path).getroot()
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    home = model.key("home").id
    mujoco.mj_resetDataKeyframe(model, data, home)
    mujoco.mj_forward(model, data)
    points = [
        data.xpos[model.body(h.body).id]
        + data.xmat[model.body(h.body).id].reshape(3, 3) @ h.grip_local_m
        for h in sorted(config.hands, key=lambda h: h.side)
    ]
    if abs(np.linalg.norm(points[0] - points[1]) - config.grip_spacing_m) > 0.002:
        raise ValueError("home pose does not preserve bar grip spacing")
    center = np.mean(points, axis=0)
    axis = (points[0] - points[1]) / np.linalg.norm(points[0] - points[1])
    for hand in config.hands:
        hand_axis = data.xmat[model.body(hand.body).id].reshape(3, 3)[:, 1]
        if np.linalg.norm(hand_axis - axis) > 1e-4:
            raise ValueError("home hand local +Y must align with the bar shaft")
    quat = np.empty(4)
    mujoco.mju_quatZ2Vec(quat, axis)
    # Z->Y local rotation makes the bar's local Y axis follow the grip line.
    rotation = np.array([2**-0.5, 2**-0.5, 0.0, 0.0])
    result = np.empty(4)
    mujoco.mju_mulQuat(result, quat, rotation)
    add_barbell(root, config, center, config.mass_kg, config.grip_solref)
    root.find(".//body[@name='barbell']").set("quat", format_vec(result))
    for key in root.findall("keyframe/key"):
        if key.get("name") != "home":
            raise ValueError("bar scene requires only the home keyframe")
        key.set("qpos", format_vec(np.r_[model.key_qpos[home], center, result]))
        if key.get("qvel") is not None:
            key.set("qvel", format_vec(np.r_[model.key_qvel[home], np.zeros(6)]))
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path)


def sample_payloads(seed, count):
    """Fixed per-world 25% unloaded, 25% 30 kg, 50% U[5,30] mixture."""
    values = []
    for world in range(count):
        rng = np.random.default_rng(np.random.SeedSequence((seed, world, 741)))
        choice = rng.random()
        values.append(
            0.0 if choice < 0.25 else 30.0 if choice < 0.5 else rng.uniform(5, 30)
        )
    return np.asarray(values, dtype=np.float32)


def free_joint_coordinates(model):
    ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    return tuple((int(model.jnt_qposadr[i]), int(model.jnt_dofadr[i])) for i in ids)
