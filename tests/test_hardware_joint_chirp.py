from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bello_mujoco.hardware.joint_chirp import (
    JointChirpConfig,
    actuator_layout,
    build_hardware_adapter,
    chirp_point,
    parse_args,
    validate_experiment,
)


def test_joint_chirp_layout_round_trips_logical_ankle_coordinates() -> None:
    physical_names = (
        "left_hip_pitch_joint",
        "left_ankle_motor_1",
        "left_ankle_motor_2",
    )
    logical_names, transform = actuator_layout(physical_names)
    logical = np.asarray((0.1, 0.3, -0.05))

    physical = transform @ logical

    assert logical_names == (
        "left_hip_pitch_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    )
    assert physical == pytest.approx((0.1, -0.35, 0.25))
    assert np.linalg.solve(transform, physical) == pytest.approx(logical)


def test_joint_chirp_adapter_uses_generated_differential_motor_gains() -> None:
    adapter = build_hardware_adapter(("left_ankle_motor_1", "left_ankle_motor_2"), 50.0)

    assert adapter.kp == pytest.approx((11.0, 11.0))
    assert adapter.kd == pytest.approx((0.5, 0.5))
    assert adapter.control_timestep == pytest.approx(0.02)


def test_joint_chirp_profile_has_hold_segments_and_smooth_envelope() -> None:
    config = JointChirpConfig(joint="left_hip_roll_joint")

    assert chirp_point(0.0, config).phase == "settle_before"
    start = chirp_point(config.settle_before_seconds, config)
    assert start.phase == "chirp"
    assert start.offset_radians == pytest.approx(0.0)
    assert start.instantaneous_frequency_hz == pytest.approx(config.start_frequency_hz)
    end = chirp_point(
        config.settle_before_seconds + config.duration_seconds,
        config,
    )
    assert end.phase == "settle_after"
    assert end.offset_radians == pytest.approx(0.0)
    samples = [
        chirp_point(config.settle_before_seconds + time, config).offset_radians
        for time in np.linspace(0.0, config.duration_seconds, 1000, endpoint=False)
    ]
    assert max(abs(value) for value in samples) <= config.amplitude_radians


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"amplitude_radians": 0.051}, "must not exceed"),
        ({"end_frequency_hz": 20.0}, "must remain below"),
        ({"envelope_seconds": 16.0}, "half the chirp duration"),
    ],
)
def test_joint_chirp_rejects_unsafe_profiles(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        JointChirpConfig(joint="left_hip_roll_joint", **overrides)


def test_joint_chirp_validates_logical_soft_limit_after_differential_mapping() -> None:
    adapter = build_hardware_adapter(("left_ankle_motor_1", "left_ankle_motor_2"), 50.0)
    limits = SimpleNamespace(
        position_min=-2.0,
        position_max=2.0,
        velocity_min=-20.0,
        velocity_max=20.0,
        kp_min=0.0,
        kp_max=300.0,
        kd_min=0.0,
        kd_max=30.0,
    )
    robot = SimpleNamespace(
        motors=(
            SimpleNamespace(coordinate="left_ankle_motor_1", limits=limits),
            SimpleNamespace(coordinate="left_ankle_motor_2", limits=limits),
        )
    )
    initial_logical = np.asarray((0.3, 0.07))
    initial_physical = adapter.transform @ initial_logical

    with pytest.raises(ValueError, match="soft limit"):
        validate_experiment(
            robot,
            adapter,
            initial_physical,
            JointChirpConfig(joint="left_ankle_roll_joint", amplitude_radians=0.02),
            1.0,
        )


def test_joint_chirp_cli_defaults_to_read_only() -> None:
    args = parse_args(("--joint", "left_hip_roll_joint"))

    assert args.arm is False
    assert args.groups == ["lower_body", "upper_body"]
    assert args.rate == pytest.approx(50.0)
