"""MuJoCo IMU sensor binding for the EBIMU serial emulator."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

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
class ImuNoiseConfig:
    """Independent Gaussian noise applied to an emulated IMU sample."""

    enabled: bool = False
    orientation_std_radians: float = 0.0
    linear_acceleration_std_mps2: float = 0.0
    seed: int = 0


class ImuNoiseInjector:
    """Add reproducible sensor noise without changing simulated dynamics."""

    def __init__(self, config: ImuNoiseConfig) -> None:
        validate_imu_noise_config(config)
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    def apply(self, sample: ImuSample) -> ImuSample:
        if not self.config.enabled:
            return sample

        orientation = np.asarray(sample.orientation_xyzw, dtype=np.float64)
        orientation_norm = float(np.linalg.norm(orientation))
        if (
            orientation.shape != (4,)
            or not np.all(np.isfinite(orientation))
            or orientation_norm <= 0.0
        ):
            raise ValueError("IMU orientation must be a finite nonzero quaternion")
        orientation /= orientation_norm

        rotation_vector = self._rng.normal(
            0.0, self.config.orientation_std_radians, size=3
        )
        angle = float(np.linalg.norm(rotation_vector))
        if angle > 0.0:
            delta_xyz = rotation_vector * (math.sin(0.5 * angle) / angle)
            delta_xyzw = np.asarray(
                (*delta_xyz, math.cos(0.5 * angle)), dtype=np.float64
            )
            orientation = _multiply_quaternions_xyzw(orientation, delta_xyzw)
            orientation /= np.linalg.norm(orientation)

        acceleration = np.asarray(sample.linear_acceleration_xyz, dtype=np.float64)
        if acceleration.shape != (3,) or not np.all(np.isfinite(acceleration)):
            raise ValueError("IMU linear acceleration must contain three finite values")
        acceleration += self._rng.normal(
            0.0, self.config.linear_acceleration_std_mps2, size=3
        )
        return replace(
            sample,
            orientation_xyzw=tuple(float(value) for value in orientation),
            linear_acceleration_xyz=tuple(float(value) for value in acceleration),
        )


def validate_imu_noise_config(config: ImuNoiseConfig) -> None:
    for name, value in (
        ("orientation_std_radians", config.orientation_std_radians),
        ("linear_acceleration_std_mps2", config.linear_acceleration_std_mps2),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"imu_noise.{name} must be finite and nonnegative")
    if config.seed < 0:
        raise ValueError("imu_noise.seed must be nonnegative")
    if (
        config.enabled
        and config.orientation_std_radians == 0.0
        and config.linear_acceleration_std_mps2 == 0.0
    ):
        raise ValueError("enabled imu_noise must configure at least one positive std")


def _multiply_quaternions_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.asarray(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dtype=np.float64,
    )


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
