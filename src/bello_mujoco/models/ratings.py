"""Bind manufacturer-rated torque to canonical physical motor coordinates."""

from importlib.resources import files
from typing import TYPE_CHECKING

import numpy as np
from omegaconf import OmegaConf

from bello_mujoco.models import robot_descriptions_root

if TYPE_CHECKING:
    from bello_mujoco.sim import BelloModel


def motor_continuous_torque(model: "BelloModel") -> np.ndarray:
    """Return output-shaft ratings in actuator order, including held motors."""
    path = model.robot.hardware_description_path(robot_descriptions_root())
    motors = OmegaConf.to_container(OmegaConf.load(path), resolve=True)["motors"]
    by_coordinate = {motor["coordinate"]: motor for motor in motors}
    coordinates = model.actuator_coordinate_names
    if (
        len(by_coordinate) != len(motors)
        or len({motor["name"] for motor in motors}) != len(motors)
        or len(set(coordinates)) != len(coordinates)
        or set(coordinates) != set(by_coordinate)
    ):
        raise ValueError("torque ratings require one actuator per physical motor")
    if not np.all(model.mj_model.actuator_gear == [1, 0, 0, 0, 0, 0]):
        raise ValueError("motor torque ratings require unit actuator gear")
    resource = files("bello_mujoco").joinpath("robots", "motor_torque_ratings.yaml")
    with resource.open("r", encoding="utf-8") as stream:
        ratings = OmegaConf.to_container(OmegaConf.load(stream), resolve=True)
    values = []
    for coordinate in coordinates:
        model_name = by_coordinate[coordinate]["model"]
        if model_name not in ratings:
            raise ValueError(f"no verified continuous torque rating for {model_name}")
        values.append(ratings[model_name])
    result = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(result) & (result > 0)):
        raise ValueError("continuous torque ratings must be positive and finite")
    return result
