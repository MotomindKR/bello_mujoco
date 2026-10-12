"""Compare contact decoding with MuJoCo's force API, including pyramid edges."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from bello_mujoco.envs.contacts import contact_normal_forces


@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
@pytest.mark.parametrize("dim", [1, 3, 4, 6])
def test_contact_normal_force_matches_mujoco(cone, dim):
    model = mujoco.MjModel.from_xml_string(f"""
    <mujoco><option cone="{cone}"/><worldbody>
      <geom type="plane" size="1 1 .1" condim="{dim}"/>
      <body pos="0 0 .099"><freejoint/>
        <geom type="sphere" size=".1" mass="1" condim="{dim}"/>
      </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon == 1
    # A zero first pyramid edge must not hide a loaded contact.
    data.efc_force[:] = np.arange(data.nefc) + 1.0
    if cone == "pyramidal" and dim > 1:
        data.efc_force[0] = 0.0
    expected = np.zeros(6)
    mujoco.mj_contactForce(model, data, 0, expected)

    def decode(force):
        return contact_normal_forces(
            model,
            SimpleNamespace(
                contact__efc_address=jnp.array([[0], [-1]]),
                contact__dim=jnp.array([dim, dim]),
                efc__force=force,
            ),
        )

    actual = jax.jit(jax.vmap(decode))(
        jnp.asarray(np.stack([data.efc_force, 2 * data.efc_force]))
    )
    np.testing.assert_allclose(
        actual, [[expected[0], 0], [2 * expected[0], 0]], atol=1e-6
    )


def test_normal_force_supports_weight_in_static_contact():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><option cone="pyramidal"/><worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body pos="0 0 .1"><freejoint/><geom type="sphere" size=".1" mass="1"/>
      </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    for _ in range(500):
        mujoco.mj_step(model, data)
    force = contact_normal_forces(
        model,
        SimpleNamespace(
            contact__efc_address=jnp.asarray(data.contact.efc_address[:, None]),
            contact__dim=jnp.asarray(data.contact.dim),
            efc__force=jnp.asarray(data.efc_force),
        ),
    )
    np.testing.assert_allclose(force.sum(), 9.81, atol=0.01)
    assert data.efc_force[0] < 0.3 * float(force.sum())


def test_empty_constraint_buffer_and_inactive_slots():
    model = SimpleNamespace(opt=SimpleNamespace(cone=0))
    impl = SimpleNamespace(
        contact__efc_address=jnp.array([[-1]]),
        contact__dim=jnp.array([3]),
        efc__force=jnp.zeros(0),
    )
    np.testing.assert_array_equal(contact_normal_forces(model, impl), [0])


@pytest.mark.integration
@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
def test_warp_contact_load_matches_native_static_support(cone):
    from mujoco import mjx

    if jax.default_backend() != "gpu":
        pytest.skip("MJX-Warp contact parity requires a GPU")
    model = mujoco.MjModel.from_xml_string(f"""
    <mujoco><option cone="{cone}"/><worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body pos="0 0 .099"><freejoint/><geom type="sphere" size=".1" mass="1"/>
      </body></worldbody></mujoco>""")
    native = mujoco.MjData(model)
    mujoco.mj_forward(model, native)
    warp_model = mjx.put_model(model, impl="warp")
    data = mjx.put_data(model, native, impl="warp", naconmax=32, njmax=128)
    step = jax.jit(lambda state: mjx.step(warp_model, state))
    for _ in range(100):
        data = step(data)
    native = mjx.get_data(model, data)
    # get_data does not copy the solver force buffers; solve at the same state.
    mujoco.mj_forward(model, native)
    expected = np.zeros(6)
    mujoco.mj_contactForce(model, native, 0, expected)
    actual = jax.jit(lambda state: contact_normal_forces(model, state._impl))(data)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(actual[0], 9.81, atol=0.01)
