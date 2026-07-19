"""MuJoCo IMU sensor binding for the EBIMU serial emulator."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class ImuSample:
    simulation_time: float
    orientation_xyzw: tuple[float, float, float, float]
    angular_velocity_xyz: tuple[float, float, float]
    linear_acceleration_xyz: tuple[float, float, float]
    local_velocity_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    world_position_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class MujocoImu:
    orientation_adr: int
    gyro_adr: int
    accelerometer_adr: int
    site_id: int

    @classmethod
    def bind(cls, model: mujoco.MjModel) -> MujocoImu:
        orientation_adr = _sensor_address(model, "bello_imu_orientation", 4)
        gyro_adr = _sensor_address(model, "bello_imu_gyro", 3)
        accelerometer_adr = _sensor_address(model, "bello_imu_accelerometer", 3)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "bello_imu_site")
        if site_id < 0:
            raise ValueError("MuJoCo IMU site is missing: 'bello_imu_site'")
        return cls(orientation_adr, gyro_adr, accelerometer_adr, site_id)

    def sample(self, data: mujoco.MjData) -> ImuSample:
        orientation_wxyz = data.sensordata[
            self.orientation_adr : self.orientation_adr + 4
        ]
        gyro = data.sensordata[self.gyro_adr : self.gyro_adr + 3]
        accelerometer = data.sensordata[
            self.accelerometer_adr : self.accelerometer_adr + 3
        ]
        velocity = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            data.model,
            data,
            mujoco.mjtObj.mjOBJ_SITE,
            self.site_id,
            velocity,
            1,
        )
        return ImuSample(
            simulation_time=float(data.time),
            orientation_xyzw=(
                float(orientation_wxyz[1]),
                float(orientation_wxyz[2]),
                float(orientation_wxyz[3]),
                float(orientation_wxyz[0]),
            ),
            angular_velocity_xyz=tuple(float(value) for value in gyro),
            linear_acceleration_xyz=tuple(float(value) for value in accelerometer),
            local_velocity_xyz=tuple(float(value) for value in velocity[3:]),
            world_position_xyz=tuple(
                float(value) for value in data.site_xpos[self.site_id]
            ),
        )


def _sensor_address(model: mujoco.MjModel, name: str, expected_dim: int) -> int:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
        raise ValueError(f"MuJoCo IMU sensor is missing: {name!r}")
    dimension = int(model.sensor_dim[sensor_id])
    if dimension != expected_dim:
        raise ValueError(
            f"MuJoCo IMU sensor {name!r} has dimension {dimension}, "
            f"expected {expected_dim}"
        )
    return int(model.sensor_adr[sensor_id])
