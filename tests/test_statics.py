import mujoco
import numpy as np
import pytest

from bello_mujoco.evaluation.statics import static_equilibrium, two_end_weight_inertia


def test_fixed_base_lever_gravity_matches_analytic_torque():
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><worldbody><body><joint axis="0 1 0"/>
        <inertial pos="0.4 0 0" mass="3" diaginertia="0.1 0.1 0.1"/>
      </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    jac = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jac, None, np.array([0.4, 0, 0]), 1)
    required = data.qfrc_bias - jac.T @ np.array([0, 0, -2 * 9.81])
    result = static_equilibrium(
        np.eye(1), np.empty((1, 0)), required, np.eye(1), np.array([[-10.0, 10.0]])
    )
    np.testing.assert_allclose(result.motor_torque, [-5 * 9.81 * 0.4])
    assert result.maximum_limit_fraction == pytest.approx(5 * 9.81 * 0.4 / 10)


def test_free_root_equilibrium_rejects_tipping_and_negative_ground_force():
    # Generalized coordinates: root Z, root pitch, one actuated joint.
    motors = np.array([[0], [0], [1.0]])
    supports = np.array([[1, 1], [-0.5, 0.5], [0, 0]])
    result = static_equilibrium(
        motors, supports, np.array([100, 20, 12]), np.eye(1), np.array([[-6, 6]])
    )
    np.testing.assert_allclose(result.normal_force, [30, 70])
    np.testing.assert_allclose(result.motor_torque, [12])
    assert result.maximum_limit_fraction == pytest.approx(2)
    assert result.residual < 1e-8
    assert (
        static_equilibrium(
            motors, supports, np.array([100, 60, 12]), np.eye(1), np.array([[-6, 6]])
        )
        is None
    )
    assert (
        static_equilibrium(
            motors, supports, np.array([-1, 0, 0]), np.eye(1), np.array([[-6, 6]])
        )
        is None
    )


def test_differential_virtual_work_uses_transpose_not_forward_mapping():
    coordinate_transform = np.array([[2.0, 1.0], [-2.0, 1.0]])
    result = static_equilibrium(
        coordinate_transform.T,
        np.empty((2, 0)),
        np.array([20, 6]),
        np.eye(2),
        np.array([[-10, 10], [-10, 10]]),
    )
    np.testing.assert_allclose(result.motor_torque, [8, -2])
    velocity = np.array([0.3, -0.4])
    assert result.motor_torque @ (coordinate_transform @ velocity) == pytest.approx(
        np.array([20, 6]) @ velocity
    )


def test_two_end_weights_include_parallel_axis_and_full_mass():
    inertia = two_end_weight_inertia(10, 0.2, 0.1, 0.5)
    np.testing.assert_allclose(
        inertia, [10 * (0.13 / 12 + 0.25), 0.2, 10 * (0.13 / 12 + 0.25)]
    )
    np.testing.assert_allclose(two_end_weight_inertia(20, 0.2, 0.1, 0.5), 2 * inertia)


@pytest.mark.parametrize("bad_rating", [0, -1, float("nan")])
def test_invalid_rating_fails(bad_rating):
    with pytest.raises(ValueError, match="invalid static"):
        static_equilibrium(
            np.eye(1),
            np.empty((1, 0)),
            np.ones(1),
            np.eye(1),
            np.array([[-bad_rating, bad_rating]]),
        )


def test_combined_joint_limit_can_bind_before_either_motor_limit():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body>
      <joint name="joint" actuatorfrclimited="true" actuatorfrcrange="-17 17"/>
      <geom size=".1"/></body></worldbody><actuator>
      <motor joint="joint" forcelimited="true" forcerange="-10 10"/>
      <motor joint="joint" forcelimited="true" forcerange="-10 10"/>
      </actuator></mujoco>""")
    data = mujoco.MjData(model)
    data.ctrl[:] = 9
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.actuator_force, [9, 9])
    np.testing.assert_allclose(data.qfrc_actuator, [17])
    result = static_equilibrium(
        np.array([[1.0, 1.0]]),
        np.empty((1, 0)),
        np.array([18.0]),
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        np.array([[-10.0, 10.0], [-10.0, 10.0], [-17.0, 17.0]]),
    )
    assert result.maximum_limit_fraction == pytest.approx(18 / 17)
