from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from bello_mujoco.hardware.joint_chirp import (
    JointChirpConfig,
    _config_record,
    actuator_layout,
    build_hardware_adapter,
    chirp_point,
    parse_args,
    run,
    validate_experiment,
)
from bello_mujoco.hardware.config import HardwareControlConfig
from bello_mujoco.robot import (
    CoupledActuator,
    CoupledActuatorGroup,
    robot_definition,
)


BELLO = robot_definition("bello")


def test_joint_chirp_layout_round_trips_logical_ankle_coordinates() -> None:
    physical_names = (
        "left_hip_pitch_joint",
        "left_ankle_motor_1",
        "left_ankle_motor_2",
    )
    logical_names, transform = actuator_layout(physical_names, BELLO)
    logical = np.asarray((0.1, 0.3, -0.05))

    physical = transform @ logical

    assert logical_names == (
        "left_hip_pitch_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    )
    assert physical == pytest.approx((0.1, -0.35, 0.25))
    assert np.linalg.solve(transform, physical) == pytest.approx(logical)


def test_joint_chirp_layout_uses_selected_robot_coupling_coefficients() -> None:
    custom_group = CoupledActuatorGroup(
        name="custom_left_ankle",
        actuators=(
            CoupledActuator(
                tendon_name="custom_motor_1",
                logical_joint_name="left_ankle_pitch_joint",
                joint_coefficients=(
                    ("left_ankle_pitch_joint", 2.0),
                    ("left_ankle_roll_joint", -3.0),
                ),
            ),
            CoupledActuator(
                tendon_name="custom_motor_2",
                logical_joint_name="left_ankle_roll_joint",
                joint_coefficients=(
                    ("left_ankle_pitch_joint", 4.0),
                    ("left_ankle_roll_joint", 5.0),
                ),
            ),
        ),
    )
    logical_names, transform = actuator_layout(
        ("custom_motor_1", "custom_motor_2"),
        replace(BELLO, coupled_actuator_groups=(custom_group,)),
    )

    assert logical_names == (
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    )
    assert transform == pytest.approx(np.asarray(((2.0, -3.0), (4.0, 5.0))))


@pytest.mark.parametrize(
    "physical_names",
    [
        ("left_ankle_motor_1",),
        ("left_ankle_motor_1", "left_ankle_roll_joint"),
        ("left_ankle_pitch_joint", "left_ankle_roll_joint"),
    ],
)
def test_joint_chirp_rejects_incomplete_or_logical_coupled_inventory(
    physical_names,
) -> None:
    with pytest.raises(ValueError, match="coupled actuator group"):
        actuator_layout(physical_names, BELLO)


def test_joint_chirp_adapter_uses_generated_differential_motor_gains() -> None:
    adapter = build_hardware_adapter(
        ("left_ankle_motor_1", "left_ankle_motor_2"),
        50.0,
        BELLO,
    )

    assert adapter.kp == pytest.approx((11.0, 11.0))
    assert adapter.kd == pytest.approx((0.5, 0.5))
    assert adapter.control_timestep == pytest.approx(0.02)


def test_joint_chirp_configuration_records_selected_asset() -> None:
    adapter = build_hardware_adapter(
        ("left_ankle_motor_1", "left_ankle_motor_2"),
        50.0,
        BELLO,
    )
    record = _config_record(
        HardwareControlConfig(),
        JointChirpConfig(joint="left_ankle_pitch_joint"),
        adapter,
        np.zeros(2),
        np.zeros(2),
    )

    assert record["asset"] == "bello"


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
    adapter = build_hardware_adapter(
        ("left_ankle_motor_1", "left_ankle_motor_2"),
        50.0,
        BELLO,
    )
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
    assert args.asset == "bello"
    assert args.groups == ["lower_body", "upper_body"]
    assert args.rate == pytest.approx(50.0)


def test_joint_chirp_cli_accepts_known_robot_asset() -> None:
    args = parse_args(("--joint", "left_hip_roll_joint", "--asset", "bello_mini"))

    assert args.asset == "bello_mini"


def test_joint_chirp_accepts_hardware_ready_mini(
    monkeypatch,
    tmp_path,
) -> None:
    def expected_import():
        raise RuntimeError("hardware binding reached")

    monkeypatch.setattr(
        "bello_mujoco.hardware.joint_chirp.import_hardware_binding",
        expected_import,
    )

    with pytest.raises(RuntimeError, match="hardware binding reached"):
        run(
            HardwareControlConfig(),
            JointChirpConfig(
                joint="left_hip_roll_joint",
                asset="bello_mini",
            ),
            tmp_path / "chirp.jsonl",
        )
