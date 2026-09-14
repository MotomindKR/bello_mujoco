"""Rated output torques and strict physical-motor binding for reward shaping."""

from dataclasses import dataclass
from types import MappingProxyType

import mujoco
import numpy as np
from omegaconf import OmegaConf

from bello_mujoco.assets_pipeline import robot_descriptions_root
from bello_mujoco.robot import RobotDefinition
from bello_mujoco.robstride.can import load_robstride_description


# Manufacturer rated output torque (Nm), not protocol/peak torque. Sources and
# rating conditions: docs/mini_settling.md#all-motor-standing-torque-penalty.
# The pinned protocol catalog has no rated-torque field; do not infer from peak.
RATED_OUTPUT_TORQUE_NM = MappingProxyType(
    {
        "ROBSTRIDE_00": 5.0,
        "ROBSTRIDE_02": 6.0,
        "ROBSTRIDE_03": 20.0,
        "ROBSTRIDE_04": 40.0,
        "ROBSTRIDE_06": 11.0,
    }
)


@dataclass(frozen=True)
class MotorTorqueRatings:
    names: tuple[str, ...]
    models: tuple[str, ...]
    continuous_torque_nm: tuple[float, ...]


def motor_torque_ratings(
    model: mujoco.MjModel, robot: RobotDefinition
) -> MotorTorqueRatings:
    """Bind every simulated actuator, including holds, to a physical motor."""
    root = robot_descriptions_root()
    hardware_path = robot.hardware_description_path(root)
    document = OmegaConf.to_container(OmegaConf.load(hardware_path), resolve=True)
    buses = {entry["name"]: entry["name"] for entry in document["logical_buses"]}
    description = load_robstride_description(
        hardware_path, robot.motor_model_catalog_path(root), buses
    )
    by_target = {
        (motor.target.type, motor.target.name): motor for motor in description.motors
    }
    if len(by_target) != len(description.motors):
        raise ValueError("standing motor torque reward requires unique motor targets")
    names, models, ratings = [], [], []
    for index in range(model.nu):
        transmission = model.actuator_trntype[index]
        if transmission == mujoco.mjtTrn.mjTRN_JOINT:
            kind, object_type = "joint", mujoco.mjtObj.mjOBJ_JOINT
        elif transmission == mujoco.mjtTrn.mjTRN_TENDON:
            kind, object_type = "tendon", mujoco.mjtObj.mjOBJ_TENDON
        else:
            raise ValueError(
                "standing motor torque reward requires joint/tendon actuators"
            )
        target = mujoco.mj_id2name(model, object_type, model.actuator_trnid[index, 0])
        motor = by_target.get((kind, target))
        if motor is None:
            raise ValueError(f"no physical motor rating for {kind} {target!r}")
        if not np.array_equal(model.actuator_gear[index], [1, 0, 0, 0, 0, 0]):
            raise ValueError("standing motor torque reward requires unit actuator gear")
        if motor.model.name not in RATED_OUTPUT_TORQUE_NM:
            raise ValueError(
                f"no verified continuous torque rating for {motor.model.name}"
            )
        names.append(motor.name)
        models.append(motor.model.name)
        ratings.append(RATED_OUTPUT_TORQUE_NM[motor.model.name])
    if len(set(names)) != len(names) or set(names) != {
        m.name for m in description.motors
    }:
        raise ValueError(
            "standing motor torque reward requires exactly one actuator per physical motor"
        )
    return MotorTorqueRatings(tuple(names), tuple(models), tuple(ratings))
