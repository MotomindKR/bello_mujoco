"""Compile canonical motor/joint limits into connected-coordinate constraints."""

from dataclasses import dataclass
from functools import cached_property
import xml.etree.ElementTree as ET

import numpy as np
from omegaconf import OmegaConf

from bello_mujoco.models import robot_descriptions_root


@dataclass(frozen=True)
class TargetLimits:
    motor: np.ndarray
    joint: np.ndarray
    joint_from_motor: np.ndarray
    position_step: np.ndarray

    def __post_init__(self):
        n = len(self.motor)
        if (
            self.motor.shape != (n, 2)
            or self.joint.shape != (n, 2)
            or self.joint_from_motor.shape != (n, n)
            or self.position_step.shape != (n,)
            or not all(
                np.isfinite(a).all()
                for a in (
                    self.motor,
                    self.joint,
                    self.joint_from_motor,
                    self.position_step,
                )
            )
            or np.any(self.motor[:, 0] >= self.motor[:, 1])
            or np.any(self.joint[:, 0] >= self.joint[:, 1])
            or np.any(self.position_step <= 0)
        ):
            raise ValueError("invalid motor/joint target limits")

    @property
    def matrix(self):
        return np.concatenate([np.eye(len(self.motor)), self.joint_from_motor])

    @property
    def bounds(self):
        return np.concatenate([self.motor, self.joint])

    @cached_property
    def groups(self):
        # Coordinates sharing a logical joint must move by the same fraction.
        connected = np.eye(len(self.motor), dtype=bool)
        for row in self.joint_from_motor:
            indices = np.flatnonzero(row)
            if not 1 <= len(indices) <= 2:
                raise ValueError(
                    "target limiting supports direct and differential joints"
                )
            connected[np.ix_(indices, indices)] = True
        for k in range(len(connected)):
            connected |= connected[:, k, None] & connected[None, k, :]
        return connected

    @property
    def membership(self):
        return self.groups.astype(int) @ (self.matrix != 0).T.astype(int) > 0

    def validate_reference(self, reference):
        values = self.matrix @ reference
        if (
            not np.all(np.isfinite(values))
            or np.any(values < self.bounds[:, 0])
            or np.any(values > self.bounds[:, 1])
        ):
            raise ValueError("nominal motor target violates motor/joint limits")


def model_target_limits(model):
    root = robot_descriptions_root()
    source = ET.parse(model.robot.source_urdf_path(root)).getroot()
    joints = {j.attrib["name"]: j.find("limit") for j in source.findall("joint")}
    joint = np.asarray(
        [
            [float(joints[name].attrib[key]) for key in ("lower", "upper")]
            for name in model.actuator_joint_names
        ]
    )
    # Task limits also constrain logical joints in differential transmissions.
    joint[:, 0] = np.maximum(joint[:, 0], model.joint_target_low)
    joint[:, 1] = np.minimum(joint[:, 1], model.joint_target_high)
    document = OmegaConf.to_container(
        OmegaConf.load(model.robot.hardware_description_path(root)), resolve=True
    )
    motors = {m["coordinate"]: m for m in document["motors"]}
    steps = np.asarray(
        [
            motors[name]["command_policy"]["maximum_position_step"]
            for name in model.actuator_coordinate_names
        ]
    )
    limits = TargetLimits(
        np.asarray(model.mj_model.actuator_ctrlrange).copy(),
        joint,
        np.linalg.inv(model.actuator_coordinate_transform),
        steps,
    )
    limits.validate_reference(model.default_ctrl)
    return limits
