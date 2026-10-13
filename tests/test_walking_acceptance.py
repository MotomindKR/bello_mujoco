from dataclasses import replace
import math

import numpy as np
import pytest

from bello_mujoco.evaluation.walking import (
    WalkingAcceptanceConfig,
    WalkingDiagnostics,
    _lateral_tilt,
)
from bello_mujoco.training.locomotion import TrainingVideoConfig, validate_video_config


def score(
    *, slip=0.01, error=0.0, alternate=True, terminal_at=None, samples=500, tilt=None
):
    diagnostics = WalkingDiagnostics(0.02, WalkingAcceptanceConfig(enabled=True))
    for step in range(samples):
        left = (step // 20) % 2 == 0 if alternate else True
        diagnostics.record(
            dict(
                left_contact=float(left),
                right_contact=float(not left),
                stance_slip_speed=slip,
                velocity_error_sq=error,
                yaw_rate_error_sq=0.0,
                orientation_lateral_tilt=tilt(step) if tilt else 0.0,
                terminal=float(step == terminal_at),
            )
        )
    return diagnostics.summary()


def test_alternating_low_slip_walking_passes():
    assert score()["walking/accepted"] == 1


@pytest.mark.parametrize(
    "kwargs,failed",
    [
        ({"slip": 0.3}, "stance_slip"),
        ({"error": 0.04}, "velocity"),
        ({"alternate": False}, "alternation"),
        ({"terminal_at": 10}, "survival"),
        ({"samples": 150}, "duration"),
        ({"slip": float("nan")}, "survival"),
        ({"tilt": lambda step: math.radians(16)}, "lateral_tilt_rms"),
        (
            {"tilt": lambda step: math.radians(25) if step == 250 else 0.0},
            "lateral_tilt_peak",
        ),
        ({"tilt": lambda step: float("nan")}, "survival"),
    ],
)
def test_gate_rejects_degenerate_or_incomplete_rollouts(kwargs, failed):
    result = score(**kwargs)
    assert result["walking/accepted"] == 0
    assert result[f"walking/pass_{failed}"] == 0


def test_short_contact_chatter_does_not_count_as_alternating_support():
    diagnostics = WalkingDiagnostics(0.02, WalkingAcceptanceConfig(enabled=True))
    for step in range(500):
        diagnostics.record(
            dict(
                left_contact=step % 2,
                right_contact=1 - step % 2,
                stance_slip_speed=0.0,
                velocity_error_sq=0.0,
                yaw_rate_error_sq=0.0,
                orientation_lateral_tilt=0.0,
                terminal=0.0,
            )
        )
    assert diagnostics.summary()["walking/accepted"] == 0
    assert diagnostics.summary()["walking/support_alternations"] == 0


def test_acceptance_cannot_score_standing_or_resetting_video():
    config = TrainingVideoConfig(
        enabled=True, walking_acceptance=WalkingAcceptanceConfig(enabled=True)
    )
    with pytest.raises(ValueError, match="translation"):
        validate_video_config(config)
    with pytest.raises(ValueError, match="stop_on_done"):
        validate_video_config(replace(config, command=(0.5, 0, 0), stop_on_done=False))


def test_tilt_gate_keeps_normal_balancing_and_discards_warmup():
    result = score(
        tilt=lambda step: math.radians(30 if step < 99 else 8 * (-1) ** step)
    )
    assert result["walking/accepted"] == 1
    assert result["walking/lateral_tilt_rms"] == pytest.approx(math.radians(8))
    assert result["walking/lateral_tilt_peak"] == pytest.approx(math.radians(8))
    assert result["walking/lateral_tilt_peak_to_peak"] == pytest.approx(
        math.radians(16)
    )


@pytest.mark.parametrize("axis", [(1, 0, 0), (0, 1, 0)])
@pytest.mark.parametrize("yaw", [0.0, 0.8, -2.0])
def test_lateral_tilt_uses_robot_forward_axis_and_heading(axis, yaw):
    heading = yaw + math.atan2(axis[1], axis[0])
    forward = np.array([math.cos(heading), math.sin(heading), 0])
    lateral = np.array([-math.sin(heading), math.cos(heading), 0])
    vertical = np.array([0, 0, 1])
    for angle in (-0.3, 0.0, 0.3):
        up = math.sin(angle) * lateral + math.cos(angle) * vertical
        assert _lateral_tilt(up, yaw, axis) == pytest.approx(angle)
        pitched_up = math.sin(angle) * forward + math.cos(angle) * vertical
        assert _lateral_tilt(pitched_up, yaw, axis) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("field", ["lateral_tilt_rms_max", "lateral_tilt_peak_max"])
@pytest.mark.parametrize("value", [0.0, -0.1, float("nan"), float("inf")])
def test_invalid_tilt_threshold_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        WalkingAcceptanceConfig(**{field: value})
