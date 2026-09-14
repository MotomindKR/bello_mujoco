import math

import pytest

from bello_mujoco.evaluation.settling import SettlingDiagnostics


def sample(**changes):
    values = dict.fromkeys(
        (
            "speed",
            "yaw_rate",
            "tilt",
            "foreaft_error",
            "width_error",
            "heading_error",
            "torque_saturation",
            "self_collision_contacts",
            "x",
            "y",
            "yaw",
            "terminal",
        ),
        0.0,
    )
    values.update(left_contact=1.0, right_contact=1.0)
    values.update(changes)
    return values


def test_survival_is_not_settling_and_push_requires_settled_precondition():
    diagnostics = SettlingDiagnostics(0.02, push_at_step=100)
    for _ in range(100):
        diagnostics.record(sample(right_contact=0, foreaft_error=0.18))
    for _ in range(100):
        diagnostics.record(sample())
    summary = diagnostics.summary()
    assert summary["pre_push/settled_at_end"] == 0
    assert summary["post_push/settled_at_end"] == 1
    assert summary["push_recovery_settled"] == 0


def test_recovery_clock_restarts_at_push_and_reports_drift():
    diagnostics = SettlingDiagnostics(0.02, push_at_step=100)
    for _ in range(100):
        diagnostics.record(sample())
    for _ in range(25):
        diagnostics.record(sample(speed=0.2))
    for i in range(100):
        diagnostics.record(sample(x=0.01 * i))
    summary = diagnostics.summary()
    assert summary["push_precondition_settled"] == 1
    assert summary["post_push/time_to_settle_seconds"] == pytest.approx(1.5)
    assert summary["push_recovery_settled"] == 1
    assert summary["post_push/max_position_drift_m"] == pytest.approx(0.99)
    # A later loss of stance invalidates settled-at-end, even after an earlier success.
    diagnostics.record(sample(terminal=1))
    assert diagnostics.summary()["push_recovery_settled"] == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"speed": 0.06},
        {"yaw_rate": 0.2},
        {"tilt": math.radians(18)},
        {"foreaft_error": 0.18},
        {"width_error": 0.12},
        {"heading_error": math.radians(20)},
        {"speed": float("nan")},
    ],
)
def test_each_physical_settling_requirement_is_enforced(changes):
    diagnostics = SettlingDiagnostics(0.02)
    for _ in range(100):
        diagnostics.record(sample(**changes))
    assert diagnostics.summary()["standing/settled_at_end"] == 0
