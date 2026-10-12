from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from bello_mujoco.models.barbell import Grip, grip_wrench_columns

from bello_mujoco.evaluation.barbell_hold import (
    add_barbell,
    add_feedforward,
    grip_measurements,
    unexpected_contacts,
)


def geometry():
    return SimpleNamespace(
        bar_length_m=1.0,
        end_weight_width_m=0.1,
        end_weight_radius_m=0.1,
        shaft_diameter_m=0.02,
        grip_spacing_m=0.4,
        hands=tuple(
            Grip(body=side, side=side, grip_local_m=(0, 0, 0))
            for side in ("left", "right")
        ),
    )


def test_free_bar_mass_inertia_and_grip_sites():
    root = ET.fromstring("""<mujoco><worldbody>
      <body name="left" pos="0 .2 1"><geom size=".02"/></body>
      <body name="right" pos="0 -.2 1"><geom size=".02"/></body>
    </worldbody></mujoco>""")
    add_barbell(root, geometry(), (0, 0, 1), 8.0, (0.02, 1.0))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert model.body("barbell").mass == pytest.approx([8])
    np.testing.assert_allclose(
        model.body("barbell").inertia,
        [8 * (0.04 / 12 + 0.45**2), 0.04, 8 * (0.04 / 12 + 0.45**2)],
    )
    assert model.joint("barbell_free").type == mujoco.mjtJoint.mjJNT_FREE
    assert model.neq == 4
    gaps, forces = grip_measurements(model, data, geometry())
    np.testing.assert_allclose(gaps, 0, atol=1e-12)
    assert np.isfinite(forces).all()
    # Soft constraints support the free bar without any root/world weld.
    for _ in range(500):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    _, forces = grip_measurements(model, data, geometry())
    np.testing.assert_allclose(forces, [8 * 9.81 / 2] * 2, rtol=0.01)
    assert data.xpos[model.body("barbell").id, 2] > 0.99


def test_feedforward_keeps_native_pd_and_torque_clamp():
    root = ET.fromstring("""<mujoco><worldbody><body>
      <joint name="hinge"/><geom size=".1"/></body></worldbody>
      <actuator><position name="servo" joint="hinge" kp="20" kv="2"
        ctrllimited="false" ctrlrange="-1 1" forcelimited="true" forcerange="-5 5"/>
      </actuator></mujoco>""")
    add_feedforward(root, np.array([3.0]))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.actuator_force, [3])
    data.qpos[0] = 0.1
    data.qvel[0] = 0.2
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.actuator_force, [0.6])
    data.ctrl[0] = 1
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.actuator_force, [5])
    np.testing.assert_allclose(model.actuator_ctrlrange, [[-1, 1]])


def test_hinge_grips_leave_bar_axial_spin_free():
    root = ET.fromstring("""<mujoco><option gravity="0 0 0"/><worldbody>
      <body name="left" pos="0 .2 1"><geom size=".02"/></body>
      <body name="right" pos="0 -.2 1"><geom size=".02"/></body>
    </worldbody></mujoco>""")
    add_barbell(root, geometry(), (0, 0, 1), 8.0, (0.02, 1.0))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    data.qvel[4] = 1
    for _ in range(100):
        mujoco.mj_step(model, data)
    np.testing.assert_allclose(data.qvel[3:], [0, 1, 0], atol=1e-8)


def test_missing_grip_body_fails():
    with pytest.raises(ValueError, match="missing hand"):
        add_barbell(
            ET.fromstring("<mujoco><worldbody/></mujoco>"),
            geometry(),
            (0, 0, 1),
            8,
            (0.02, 1),
        )


def test_contact_force_is_reported_on_second_geom_in_world_frame():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom name="ground" type="plane" size="1 1 .1"/>
      <body pos="0 0 .1"><freejoint/><geom name="ball" size=".1" mass="2"/>
      </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    for _ in range(1000):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    contacts = unexpected_contacts(model, data, set())
    assert len(contacts) == 1
    contact = contacts[0]
    assert contact["geom2"] == "ball"
    np.testing.assert_allclose(
        contact["force_on_geom2_world_n"], [0, 0, 2 * 9.81], atol=1e-4
    )


@pytest.mark.parametrize(
    "axis,allowed", [([0, 1, 0], True), ([1, 0, 0], False), ([0, 0, 1], False)]
)
def test_each_hand_can_only_rotate_about_the_shaft(axis, allowed):
    root = ET.fromstring("""<mujoco><option gravity="0 0 0"/><worldbody>
      <body name="left" pos="0 .2 1"><freejoint/><geom size=".01"/></body>
      <body name="right" pos="0 -.2 1"><geom size=".01"/></body>
    </worldbody></mujoco>""")
    add_barbell(root, geometry(), (0, 0, 1), 8.0, (0.02, 1.0))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mju_axisAngle2Quat(data.qpos[3:7], np.array(axis, dtype=float), 0.5)
    mujoco.mj_forward(model, data)
    # Main grip position remains coincident for all three rotations.
    np.testing.assert_allclose(
        data.site_xpos[model.site("left_hand_grip").id],
        data.site_xpos[model.site("left_bar_grip").id],
        atol=1e-12,
    )
    residual = data.efc_pos[data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY]
    if allowed:
        np.testing.assert_allclose(residual, 0, atol=1e-12)
    else:
        assert np.linalg.norm(residual) > 0.04


def test_hinge_constraints_have_five_independent_relative_constraints():
    root = ET.fromstring("""<mujoco><worldbody>
      <body name="left" pos="0 .2 1"><freejoint/><geom size=".01"/></body>
      <body name="right" pos="0 -.2 1"><geom size=".01"/></body>
    </worldbody></mujoco>""")
    add_barbell(root, geometry(), (0, 0, 1), 8.0, (0.02, 1.0))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ids = [model.equality(f"left_ideal_{a}").id for a in ("grip", "grip_axis")]
    rows = (data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & np.isin(
        data.efc_id, ids
    )
    assert np.linalg.matrix_rank(data.efc_J.reshape(data.nefc, model.nv)[rows, :6]) == 5


def test_hinge_reactions_cannot_resist_axial_spin():
    root = ET.fromstring("""<mujoco><worldbody>
      <body name="left" pos="0 .2 1"><freejoint/><geom size=".01"/></body>
      <body name="right" pos="0 -.2 1"><geom size=".01"/></body>
    </worldbody></mujoco>""")
    add_barbell(root, geometry(), (0, 0, 1), 8.0, (0.02, 1.0))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    reactions = grip_wrench_columns(model, data, "left")
    assert np.linalg.matrix_rank(reactions[:6]) == 5
    np.testing.assert_allclose(reactions[4], 0, atol=1e-12)
    assert np.linalg.norm(reactions[3]) > 0.9
    assert np.linalg.norm(reactions[5]) > 0.9
