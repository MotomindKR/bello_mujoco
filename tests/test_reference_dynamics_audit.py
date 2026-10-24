from pathlib import Path

import mujoco
import numpy as np
import pytest

from bello_mujoco.evaluation.reference_audit import (
    ReferenceAuditConfig,
    foot_support_samples,
    ground_contact_bodies,
    inverse_requirements,
    support_margin,
)


def test_inverse_residual_distinguishes_unsupported_hold_from_ballistic_fall():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body pos="0 0 1"><freejoint/>'
        '<geom type="sphere" size=".1" mass="2"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    _, held = inverse_requirements(
        model,
        data,
        model.qpos0,
        np.zeros(6),
        np.zeros(6),
        np.array([], dtype=int),
        np.zeros((0, 0)),
    )
    np.testing.assert_allclose(held[:3], [0, 0, 19.62], atol=1e-8)
    _, falling = inverse_requirements(
        model,
        data,
        model.qpos0,
        np.zeros(6),
        np.array([0, 0, -9.81, 0, 0, 0]),
        np.array([], dtype=int),
        np.zeros((0, 0)),
    )
    np.testing.assert_allclose(falling, 0, atol=1e-8)


def test_inverse_motor_coordinates_preserve_differential_torque_budget():
    transform = np.array([[-1.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(np.linalg.solve(transform.T, [34, 0]), [-17, 17])
    np.testing.assert_allclose(np.linalg.solve(transform.T, [20, 20]), [0, 20])


def test_near_foot_material_point_slip_and_degenerate_support():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom name="ground" type="plane" size="1 1 .1"/>'
        '<body name="foot" pos="0 0 .099"><freejoint/>'
        '<geom name="sole" type="sphere" size=".1" mass="1"/>'
        "</body></worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    data.qvel[0] = 0.7
    mujoco.mj_forward(model, data)
    floor, sole = model.geom("ground").id, model.geom("sole").id
    samples = foot_support_samples(model, data, [sole], [floor], 0.03)
    assert len(samples) == 1
    assert samples[0][1] == pytest.approx(0.7)
    assert ground_contact_bodies(model, data, [floor]) == {model.body("foot").id}
    assert support_margin([point for point, _ in samples], [0, 0]) is None
    points = [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert support_margin(points, [0.5, 0.5]) == pytest.approx(0.5)
    assert support_margin(points, [1.2, 0.5]) == pytest.approx(-0.2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frame_stride": 0},
        {"frame_stride": True},
        {"slip_speed_m_s": float("nan")},
        {"clips": ("a", "a")},
    ],
)
def test_reference_audit_rejects_invalid_sampling(kwargs):
    with pytest.raises(ValueError):
        ReferenceAuditConfig(Path("corpus"), Path("report.json"), **kwargs)
