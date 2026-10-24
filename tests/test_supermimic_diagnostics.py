import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bello_mujoco.supermimic.control import MotorCommand, motor_limit_flags
from bello_mujoco.training.supermimic import (
    require_valid_metrics,
    training_metric_snapshot,
)


def test_command_and_force_limits_are_independent_under_jit_and_batching():
    command = MotorCommand(
        jnp.zeros((2, 3)),
        jnp.array([[2.0, 0.0, 2.0], [0.0, 0.0, 0.0]]),
        jnp.array([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
        jnp.array([[1.0, 0.1, 0.5], [0.0, 0.0, 0.0]]),
    )
    flags = jax.jit(motor_limit_flags)(
        command, jnp.array([[0.0, -10.0, 0.0], [0.0, 0.0, 10.0]]), jnp.full(3, 10.0)
    )
    np.testing.assert_array_equal(
        flags["position_projected"], [[True, False, True], [False, False, False]]
    )
    np.testing.assert_array_equal(
        flags["rate_limited"], [[False, True, True], [False, False, False]]
    )
    np.testing.assert_array_equal(
        flags["torque_saturated"], [[False, True, False], [False, False, True]]
    )


def test_compact_motor_metrics_expand_only_when_requested():
    metrics = {
        "collection/reward": jnp.asarray(0.1),
        "motor/rate_limited_fraction": jnp.array([0.2, 0.3]),
    }
    require_valid_metrics(metrics)
    assert training_metric_snapshot(metrics, ("a", "b"), include_per_motor=False) == {
        "collection/reward": pytest.approx(0.1)
    }
    expanded = training_metric_snapshot(metrics, ("a", "b"), include_per_motor=True)
    assert expanded["motor/a/rate_limited_fraction"] == pytest.approx(0.2)
    assert expanded["motor/b/rate_limited_fraction"] == pytest.approx(0.3)
    with pytest.raises(ValueError, match="one value per physical motor"):
        training_metric_snapshot(metrics, ("a",), include_per_motor=True)
    with pytest.raises(RuntimeError, match="motor/rate_limited_fraction"):
        require_valid_metrics(
            {"motor/rate_limited_fraction": jnp.array([0.0, jnp.nan])}
        )
