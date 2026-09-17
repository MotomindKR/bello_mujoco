"""Synthetic motor/joint envelope checks, independent of robot configuration."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bello_mujoco.envs.actions import limit_motor_step, project_motor_target
from bello_mujoco.models.target_limits import TargetLimits


def differential_limits():
    return TargetLimits(
        np.array([[-2, 2], [-2, 2], [-3, 3.0]]),
        np.array([[-0.4, 0.6], [-0.3, 0.5], [-1, 2.0]]),
        np.array([[0.5, 0.5, 0], [-0.5, 0.5, 0], [0, 0, 1.0]]),
        np.array([0.1, 0.2, 0.3]),
    )


def test_projection_and_step_remain_in_all_envelopes_under_jit_and_vmap():
    limits = differential_limits()
    reference = jnp.array([0.1, -0.1, 0.2])
    limits.validate_reference(reference)
    rng = np.random.default_rng(41)
    requests = jnp.asarray(rng.uniform(-5, 5, (1000, 3)), dtype=jnp.float32)
    project = jax.jit(
        jax.vmap(partial(project_motor_target, reference=reference, limits=limits))
    )
    feasible = project(requests)
    previous = project(requests[::-1])
    result = jax.jit(jax.vmap(partial(limit_motor_step, limits=limits)))(
        feasible, previous
    )
    for values in (feasible, result):
        mapped = np.asarray(values) @ limits.matrix.T
        assert np.all(mapped >= limits.bounds[:, 0] - 1e-7)
        assert np.all(mapped <= limits.bounds[:, 1] + 1e-7)
    assert np.all(np.abs(np.asarray(result - previous)) <= limits.position_step + 1e-7)
    np.testing.assert_array_equal(project(feasible), feasible)


def test_coupled_pair_scales_together_without_moving_unrelated_motor():
    limits = differential_limits()
    requested = jnp.array([1.5, 1.5, 0.2])
    projected = project_motor_target(requested, jnp.zeros(3), limits)
    np.testing.assert_allclose(projected, [0.6 * (1 - 1e-6)] * 2 + [0.2], atol=1e-7)
    limited = limit_motor_step(projected, jnp.zeros(3), limits)
    np.testing.assert_allclose(limited, [0.1 * (1 - 1e-6)] * 2 + [0.2], atol=1e-7)


def test_invalid_reference_is_not_silently_repaired():
    limits = differential_limits()
    for reference in ([2, 2, 0], [float("nan"), 0, 0]):
        with pytest.raises(ValueError, match="violates"):
            limits.validate_reference(reference)


def test_runtime_intersects_task_joint_limits_with_physical_limits(tmp_path):
    from types import SimpleNamespace
    from bello_mujoco.models.target_limits import model_target_limits

    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        '<robot><joint name="hinge"><limit lower="-2" upper="2"/></joint></robot>'
    )
    hardware = tmp_path / "hardware.yaml"
    hardware.write_text(
        "motors: [{coordinate: motor, command_policy: {maximum_position_step: 0.1}}]"
    )
    model = SimpleNamespace(
        robot=SimpleNamespace(
            source_urdf_path=lambda _: urdf,
            hardware_description_path=lambda _: hardware,
        ),
        actuator_joint_names=("hinge",),
        actuator_coordinate_names=("motor",),
        mj_model=SimpleNamespace(actuator_ctrlrange=np.array([[-3.0, 3.0]])),
        actuator_coordinate_transform=np.eye(1),
        default_ctrl=np.zeros(1),
        joint_target_low=np.array([-0.4]),
        joint_target_high=np.array([0.8]),
    )
    np.testing.assert_array_equal(model_target_limits(model).joint, [[-0.4, 0.8]])
    model.joint_target_low[:], model.joint_target_high[:] = -4, 4
    np.testing.assert_array_equal(model_target_limits(model).joint, [[-2, 2]])
