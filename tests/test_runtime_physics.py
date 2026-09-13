"""Topology-preserving runtime rebuilds must retain the loaded physical model."""

import mujoco
import numpy as np
import pytest

from bello_mujoco.sim import preserve_runtime_physics


def test_rebuild_preserves_inertias_and_keeps_added_support_separate():
    spec = mujoco.MjSpec.from_string("""<mujoco><worldbody><body name="base">
      <freejoint/><inertial mass="1.2345678912345" pos=".001234567891 0 .009876543219"
        quat=".998750260395 0 .049979169271 0" diaginertia=".000123456789123 .000223456789123 .000234567891234"/>
      <geom size=".1"/></body></worldbody></mujoco>""")
    original = spec.compile()
    rebuilt_spec = mujoco.MjSpec.from_string(spec.to_xml())
    support = rebuilt_spec.worldbody.add_body(name="support")
    support.explicitinertial = True
    support.mass = 0.1
    support.inertia = [1e-4] * 3
    support.add_joint(type=mujoco.mjtJoint.mjJNT_SLIDE)
    rebuilt = rebuilt_spec.compile()
    assert not np.array_equal(original.body_mass, rebuilt.body_mass[: original.nbody])
    preserve_runtime_physics(original, rebuilt)
    for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat"):
        np.testing.assert_array_equal(
            getattr(rebuilt, field)[: original.nbody], getattr(original, field)
        )
    assert rebuilt.body("support").mass[0] == pytest.approx(0.1)
    assert rebuilt.body_subtreemass[0] == pytest.approx(
        original.body_subtreemass[0] + 0.1
    )


def test_rebuild_rejects_replaced_body_topology():
    source = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="a"><geom size=".1"/></body></worldbody></mujoco>'
    )
    target = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="b"><geom size=".1"/></body></worldbody></mujoco>'
    )
    with pytest.raises(ValueError, match="body topology"):
        preserve_runtime_physics(source, target)
