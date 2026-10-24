"""Kinematic IK constraints used while preparing physically usable references."""

from dataclasses import dataclass
import math
import xml.etree.ElementTree as ET

import mink
import mujoco
import numpy as np

from bello_mujoco.models import robot_descriptions_root


@dataclass(frozen=True)
class ContactLimitConfig:
    floor_clearance_m: float = 0.001
    max_self_penetration_m: float = 0.01
    detection_distance_m: float = 0.03

    def __post_init__(self):
        if any(not math.isfinite(value) or value < 0 for value in vars(self).values()):
            raise ValueError("contact limit distances must be finite and nonnegative")
        if self.detection_distance_m <= self.floor_clearance_m:
            raise ValueError("contact detection distance must exceed floor clearance")


class RetargetContactLimit(mink.Limit):
    """Linearized signed-distance bounds on IK displacement, including recovery."""

    def __init__(self, model, config=ContactLimitConfig()):
        self.model = model
        self.config = config
        self.floor_geoms = frozenset(
            int(index)
            for index in np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
        )
        if not self.floor_geoms:
            raise ValueError("retarget contact limits require a ground plane")
        collidable = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        model.geom_margin[collidable] = np.maximum(
            model.geom_margin[collidable], config.detection_distance_m
        )
        self._jacobian1 = np.zeros((3, model.nv))
        self._jacobian2 = np.zeros((3, model.nv))

    def compute_qp_inequalities(self, configuration, dt):
        del dt  # Mink's QP variable is displacement, before solve_ik divides by dt.
        model, data = self.model, configuration.data
        mujoco.mj_collision(model, data)
        rows, bounds = [], []
        for contact in data.contact:
            geom1, geom2 = map(int, contact.geom)
            if geom1 < 0 or geom2 < 0:
                raise ValueError("retarget contact limits do not support flex contacts")
            normal = np.asarray(contact.frame[:3])
            mujoco.mj_jac(
                model,
                data,
                self._jacobian1,
                None,
                contact.pos,
                int(model.geom_bodyid[geom1]),
            )
            mujoco.mj_jac(
                model,
                data,
                self._jacobian2,
                None,
                contact.pos,
                int(model.geom_bodyid[geom2]),
            )
            minimum = (
                self.config.floor_clearance_m
                if geom1 in self.floor_geoms or geom2 in self.floor_geoms
                else -self.config.max_self_penetration_m
            )
            rows.append(normal @ (self._jacobian1 - self._jacobian2))
            bounds.append(float(contact.dist) - minimum)
        if not rows:
            return mink.Constraint()
        return mink.Constraint(G=np.asarray(rows), h=np.asarray(bounds))


class RetargetJointLimit(mink.Limit):
    """Canonical coupled position bounds and URDF speed bounds per source frame."""

    def __init__(self, model, timestep):
        self.qpos_indices = np.asarray(model.actuator_qpos_indices)
        self.qvel_indices = np.asarray(model.actuator_qvel_indices)
        self.matrix = model.target_limits.matrix @ model.actuator_coordinate_transform
        self.bounds = np.asarray(model.target_limits.bounds)
        root = ET.parse(
            model.robot.source_urdf_path(robot_descriptions_root())
        ).getroot()
        joints = {joint.get("name"): joint for joint in root.findall("joint")}
        self.step = np.asarray(
            [
                float(joints[name].find("limit").attrib["velocity"]) * timestep
                for name in model.actuator_joint_names
            ]
        )
        if not np.isfinite(self.step).all() or np.any(self.step <= 0):
            raise ValueError("retarget requires positive finite canonical URDF speeds")
        self.previous = None

    def compute_qp_inequalities(self, configuration, dt):
        del dt
        q = configuration.data.qpos[self.qpos_indices]
        position = self.matrix @ q
        rows = np.zeros((2 * len(position), configuration.model.nv))
        rows[: len(position), self.qvel_indices] = self.matrix
        rows[len(position) :, self.qvel_indices] = -self.matrix
        bounds = np.concatenate(
            (self.bounds[:, 1] - position, position - self.bounds[:, 0])
        )
        if self.previous is not None:
            rate_rows = np.zeros((len(q), configuration.model.nv))
            rate_rows[:, self.qvel_indices] = np.eye(len(q))
            displacement = q - self.previous
            rows = np.vstack((rows, rate_rows, -rate_rows))
            bounds = np.concatenate(
                (bounds, self.step - displacement, self.step + displacement)
            )
        return mink.Constraint(G=rows, h=bounds)
