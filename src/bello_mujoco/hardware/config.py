"""Shared immutable configuration for physical policy control."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareControlConfig:
    """Transport, timing, and fail-closed limits shared by hardware tasks."""

    endpoint: str = "http://127.0.0.1:50051"
    groups: tuple[str, ...] = ("lower_body", "upper_body")
    imu_name: str = "imu"
    client_name: str = "bello-policy"
    arm: bool = False
    allow_git_commit_mismatch: bool = False
    connection_timeout_seconds: float = 3.0
    telemetry_timeout_seconds: float = 2.0
    command_validity_seconds: float = 0.08
    acknowledgement_timeout_seconds: float = 0.08
    rpc_timeout_seconds: float = 2.0
    ramp_seconds: float = 10.0
    max_duration_seconds: float | None = None
    max_motor_age_ms: int = 100
    max_imu_age_ms: int = 100
    max_tilt_radians: float = 0.70
    max_angular_velocity: float = 20.0
    max_linear_acceleration: float = 100.0
    max_loop_lag_seconds: float = 0.10
    status_interval_seconds: float | None = None
