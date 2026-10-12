from dataclasses import replace

import pytest

from bello_mujoco.evaluation.walking import WalkingAcceptanceConfig, WalkingDiagnostics
from bello_mujoco.training.locomotion import TrainingVideoConfig, validate_video_config


def score(*, slip=0.01, error=0.0, alternate=True, terminal_at=None, samples=500):
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
