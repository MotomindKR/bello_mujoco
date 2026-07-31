import numpy as np
import pytest

from bello_mujoco.hardware.pose_metrics import (
    PoseObservation,
    calculate_pose_error_metrics,
)


def observation(
    source_time: float,
    position: tuple[float, float, float],
) -> PoseObservation:
    return PoseObservation(
        source_time=source_time,
        position=np.asarray(position, dtype=np.float64),
        orientation_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64),
    )


def test_pose_metrics_remove_the_initial_translation_offset() -> None:
    truth = [
        observation(0.0, (1.0, 2.0, 0.8)),
        observation(1.0, (1.1, 2.0, 0.8)),
        observation(2.0, (1.2, 2.0, 0.8)),
    ]
    estimate = [
        observation(0.001, (0.0, 0.0, 0.8)),
        observation(1.001, (0.1, 0.0, 0.8)),
        observation(2.001, (0.2, 0.0, 0.8)),
    ]

    metrics = calculate_pose_error_metrics(
        estimate,
        truth,
        maximum_pair_time_error=0.01,
    )

    assert metrics.sample_count == 3
    assert metrics.position_rmse_m == pytest.approx(0.0, abs=1.0e-12)
    assert metrics.absolute_z_rmse_m == pytest.approx(0.0)
    assert metrics.orientation_rmse_radians == pytest.approx(0.0)


def test_pose_metrics_report_incremental_vertical_error() -> None:
    truth = [
        observation(0.0, (0.0, 0.0, 0.8)),
        observation(1.0, (0.0, 0.0, 0.8)),
        observation(2.0, (0.0, 0.0, 0.8)),
    ]
    estimate = [
        observation(0.0, (0.0, 0.0, 0.8)),
        observation(1.0, (0.0, 0.0, 0.79)),
        observation(2.0, (0.0, 0.0, 0.78)),
    ]

    metrics = calculate_pose_error_metrics(
        estimate,
        truth,
        maximum_pair_time_error=0.01,
    )

    assert metrics.position_rmse_xyz_m[2] == pytest.approx(
        np.sqrt((0.0**2 + 0.01**2 + 0.02**2) / 3.0)
    )
    assert metrics.final_position_error_xyz_m[2] == pytest.approx(-0.02)
    assert metrics.maximum_position_error_m == pytest.approx(0.02)
