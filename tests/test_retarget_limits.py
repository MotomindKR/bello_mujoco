import mink
import mujoco
import numpy as np
import pytest

from bello_mujoco.motion.retarget_limits import RetargetContactLimit


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
