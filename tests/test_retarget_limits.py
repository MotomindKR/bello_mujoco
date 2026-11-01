import mink
import mujoco
import numpy as np
import pytest
from types import SimpleNamespace

from bello_mujoco.motion.retarget_limits import (
    RetargetContactLimit,
    RetargetRootTranslationLimit,
    project_contacts,
)


@pytest.mark.parametrize("timestep", [0.005, 0.02])
def test_retarget_contact_limit_recovers_existing_floor_penetration(timestep):
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="plane" size="1 1 .01"/>'
        '<body pos="0 0 .05"><freejoint/><geom type="sphere" size=".1"/>'
        "</body></worldbody></mujoco>"
    )
    configuration = mink.Configuration(model)
    limit = RetargetContactLimit(model)
    constraint = limit.compute_qp_inequalities(configuration, timestep)
    assert constraint.G.shape == (1, model.nv)
    np.testing.assert_allclose(constraint.G[0, :3], [0, 0, -1])
    assert constraint.h[0] == pytest.approx(-0.051)
    displacement = np.zeros(model.nv)
    displacement[2] = 0.051
    assert (constraint.G @ displacement <= constraint.h + 1e-12).all()
    configuration.integrate_inplace(displacement / timestep, timestep)
    limit.compute_qp_inequalities(configuration, timestep)
    assert configuration.data.contact[0].dist == pytest.approx(0.001)


def test_nonlinear_projection_repairs_floor_without_chasing_pose_targets():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="plane" size="1 1 .01"/>'
        '<body pos="0 0 .04"><freejoint/><geom type="box" size=".2 .1 .1"/>'
        "</body></worldbody></mujoco>"
    )
    configuration = mink.Configuration(model)
    displacement = np.array([0, 0, 0, 0.4, 0.2, 0])
    configuration.integrate_inplace(displacement, 1)
    limit = RetargetContactLimit(model)
    backend = SimpleNamespace(
        model=model,
        configuration=configuration,
        solver="daqp",
        ik_limits=[limit],
    )
    pose = project_contacts(backend, limit, iterations=20, tolerance=1e-6)
    assert np.isfinite(pose).all()
    constraint = limit.compute_qp_inequalities(configuration, model.opt.timestep)
    assert constraint.inactive or np.min(constraint.h) >= -1e-6


def test_root_correction_bound_preserves_fast_source_motion():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1"/>'
        "</body></worldbody></mujoco>"
    )
    configuration = mink.Configuration(model)
    limit = RetargetRootTranslationLimit(model, 0.02)
    assert limit.compute_qp_inequalities(configuration, 0.02).inactive
    limit.previous = configuration.q[:3].copy()
    limit.source_displacement = np.array([0, 0, 0.2])
    constraint = limit.compute_qp_inequalities(configuration, 0.02)
    source_step = np.zeros(model.nv)
    source_step[2] = 0.2
    assert (constraint.G @ source_step <= constraint.h).all()
    source_step[2] = 0.23
    assert not (constraint.G @ source_step <= constraint.h).all()
