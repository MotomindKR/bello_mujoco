from dataclasses import replace
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from bello_mujoco.evaluation.world_model_audit import (
    MotionWindow,
    WorldModelAuditConfig,
    contact_groups,
    prediction_errors,
    rollout_predictions,
    summarize_prediction_errors,
)
from bello_mujoco.supermimic.state import PhysicalState


class DriftModel(eqx.Module):
    timestep: float = eqx.field(static=True, default=0.02)

    def step(self, state, action, context, input_norm, output_norm):
        return replace(
            state, position=state.position + self.timestep * state.linear_velocity
        )


def test_open_loop_prediction_and_errors_have_correct_horizon_and_units():
    state = PhysicalState(
        jnp.zeros((2, 3)),
        jnp.tile(jnp.array([1.0, 0, 0, 0]), (2, 1)),
        jnp.array([[1.0, 0, 0], [2.0, 0, 0]]),
        jnp.zeros((2, 3)),
        jnp.zeros((2, 1)),
        jnp.zeros((2, 1)),
    )
    controls = jnp.zeros((2, 8, 1))
    predicted = rollout_predictions(DriftModel(), state, controls, controls, None, None)
    np.testing.assert_allclose(predicted.position[:, 0], [0.16, 0.32], atol=1e-6)
    errors = {k: np.asarray(v) for k, v in prediction_errors(predicted, state).items()}
    np.testing.assert_allclose(errors["root_position_m"], [0.16, 0.32], atol=1e-6)
    np.testing.assert_array_equal(errors["joint_position_rms_rad"], 0)
    summary = summarize_prediction_errors(errors, np.ones(2, dtype=bool))
    assert summary["errors"]["root_position_m"]["mean"] == pytest.approx(0.24)


def test_contact_strata_detect_changes_anywhere_in_a_prediction_window():
    signatures = [
        frozenset(),
        frozenset(),
        frozenset({1}),
        frozenset({1}),
        frozenset({2}),
    ]
    groups = contact_groups(signatures, np.array([0, 1, 2]), 2)
    np.testing.assert_array_equal(groups["ground_contact_change"], [True, True, True])
    groups = contact_groups(signatures, np.array([0, 2]), 1)
    np.testing.assert_array_equal(groups["airborne"], [True, False])
    np.testing.assert_array_equal(groups["ground_contact_stable"], [True, True])


def test_missing_and_nonfinite_prediction_coverage_are_explicit():
    errors = {"distance_m": np.array([1.0, np.nan])}
    summary = summarize_prediction_errors(errors, np.array([True, True]))
    assert summary["windows"] == 2
    assert summary["finite_windows"] == summary["nonfinite_windows"] == 1
    assert summary["errors"]["distance_m"]["mean"] == 1
    empty = summarize_prediction_errors(errors, np.zeros(2, dtype=bool))
    assert empty["windows"] == 0
    assert empty["errors"]["distance_m"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizons": (0,)},
        {"anchor_stride": True},
        {"max_windows_per_horizon": 0},
        {"windows": ()},
    ],
)
def test_model_audit_config_rejects_invalid_requests(kwargs):
    options = {"windows": (MotionWindow("clip"),), **kwargs}
    with pytest.raises(ValueError):
        WorldModelAuditConfig(
            Path("checkpoint"), Path("corpus"), Path("report.json"), **options
        )
