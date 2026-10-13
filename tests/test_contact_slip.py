"""Ground contact velocity must distinguish sliding from rolling in native and JAX."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from bello_mujoco.envs.contacts import foot_contact_slip


def fixture(cone="pyramidal", height=0.099, shape="sphere", tilt=0):
    size = ".1" if shape == "sphere" else ".1 .05 .1"
    model = mujoco.MjModel.from_xml_string(f"""
    <mujoco><compiler angle="radian"/><option cone="{cone}"/><worldbody>
      <geom type="plane" size="1 1 .1" euler="0 {tilt} 0"/>
      <body pos="0 0 {height}"><freejoint/>
        <geom type="{shape}" size="{size}" mass="1"/>
      </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    wrapper = SimpleNamespace(
        mj_model=model,
        geom_body_ids=model.geom_bodyid,
        foot_body_ids=np.array([1]),
        foot_geom_ids=np.array([1]),
        terrain_geom_ids=np.array([0]),
    )
    return model, data, wrapper


def view(data):
    return SimpleNamespace(
        cvel=jnp.asarray(data.cvel),
        subtree_com=jnp.asarray(data.subtree_com),
        _impl=SimpleNamespace(
            contact__geom=jnp.asarray(data.contact.geom),
            contact__pos=jnp.asarray(data.contact.pos),
            contact__frame=jnp.asarray(data.contact.frame.reshape(-1, 3, 3)),
            contact__worldid=jnp.zeros(data.ncon, dtype=jnp.int32),
            contact__efc_address=jnp.asarray(data.contact.efc_address[:, None]),
            contact__dim=jnp.asarray(data.contact.dim),
            nacon=jnp.array([data.ncon]),
            efc__force=jnp.asarray(data.efc_force),
        ),
    )


@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
@pytest.mark.parametrize(
    "velocity", [(0.3, 0.0, 0.0), (0.2, 1.0, 0.0), (0.0, 0.0, 0.4)]
)
def test_point_slip_matches_native_contact_velocity(cone, velocity):
    model, data, wrapper = fixture(cone)
    data.qvel[[0, 4, 2]] = velocity
    mujoco.mj_forward(model, data)
    assert data.ncon == 1
    twist = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, 1, twist, 0)
    contact = data.contact[0]
    point = twist[3:] + np.cross(twist[:3], contact.pos - data.xpos[1])
    normal = contact.frame[:3]
    expected = np.linalg.norm(point - np.dot(point, normal) * normal)
    state = view(data)
    # Fix the contact load so upward velocity does not unload this kinematic test.
    state._impl.efc__force = jnp.ones_like(state._impl.efc__force) * 10
    cfg = SimpleNamespace(foot_contact_force_threshold=1.0)
    actual = jax.jit(
        lambda v: foot_contact_slip(
            wrapper,
            cfg,
            SimpleNamespace(cvel=v, subtree_com=state.subtree_com, _impl=state._impl),
        )
    )(state.cvel)
    np.testing.assert_allclose(actual, [expected], atol=1e-6)


def test_rolling_contact_is_stationary_despite_moving_body_origin():
    model, data, wrapper = fixture()
    mujoco.mj_forward(model, data)
    radius = data.xpos[1, 2] - data.contact[0].pos[2]
    data.qvel[0] = radius
    data.qvel[4] = 1.0
    mujoco.mj_forward(model, data)
    state = view(data)
    cfg = SimpleNamespace(foot_contact_force_threshold=1.0)
    assert data.cvel[1, 3] > 0.09
    np.testing.assert_allclose(foot_contact_slip(wrapper, cfg, state), [0], atol=1e-6)
    # Sliding added to the same rolling motion remains observable.
    shifts = jnp.array([0.0, 0.1, 0.4])

    def compute(shift):
        v = state.cvel.at[1, 3].add(shift)
        return foot_contact_slip(
            wrapper,
            cfg,
            SimpleNamespace(cvel=v, subtree_com=state.subtree_com, _impl=state._impl),
        )

    np.testing.assert_allclose(
        jax.jit(jax.vmap(compute))(shifts)[:, 0], shifts, atol=1e-6
    )
    np.testing.assert_array_equal(
        foot_contact_slip(wrapper, cfg, state, world_id=1), [0]
    )
    state._impl.nacon = jnp.array([0])
    np.testing.assert_array_equal(foot_contact_slip(wrapper, cfg, state), [0])


def test_empty_contact_buffer_has_no_slip():
    model, data, wrapper = fixture(height=1)
    mujoco.mj_forward(model, data)
    assert data.ncon == 0
    np.testing.assert_array_equal(
        foot_contact_slip(
            wrapper, SimpleNamespace(foot_contact_force_threshold=1.0), view(data)
        ),
        [0],
    )


def test_slip_uses_terrain_tangent_instead_of_horizontal_speed():
    model, data, wrapper = fixture(tilt=0.4)
    mujoco.mj_forward(model, data)
    normal = data.contact[0].frame[:3].copy()
    data.qvel[:3] = normal * 0.3
    mujoco.mj_forward(model, data)
    state = view(data)
    state._impl.efc__force = jnp.ones_like(state._impl.efc__force) * 10
    assert abs(data.qvel[0]) > 0.1
    np.testing.assert_allclose(
        foot_contact_slip(
            wrapper, SimpleNamespace(foot_contact_force_threshold=1.0), state
        ),
        [0],
        atol=1e-6,
    )


def test_multiple_contact_speeds_are_weighted_by_normal_load():
    model, data, wrapper = fixture(shape="box")
    data.qvel[0] = 0.2
    data.qvel[5] = 2.0
    mujoco.mj_forward(model, data)
    assert data.ncon == 4
    speeds, loads = [], []
    twist = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, 1, twist, 0)
    for index, contact in enumerate(data.contact):
        address = contact.efc_address
        count = 2 * (contact.dim - 1)
        data.efc_force[address : address + count] = 10 ** (index + 1) / count
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force)
        velocity = twist[3:] + np.cross(twist[:3], contact.pos - data.xpos[1])
        normal = contact.frame[:3]
        speeds.append(np.linalg.norm(velocity - np.dot(velocity, normal) * normal))
        loads.append(force[0])
    expected = np.average(speeds, weights=loads)
    assert abs(expected - np.mean(speeds)) > 0.01
    np.testing.assert_allclose(
        foot_contact_slip(
            wrapper, SimpleNamespace(foot_contact_force_threshold=1.0), view(data)
        ),
        [expected],
        atol=1e-6,
    )


@pytest.mark.integration
def test_warp_contact_point_slip_matches_native():
    from mujoco import mjx

    if jax.default_backend() != "gpu":
        pytest.skip("MJX-Warp contact slip parity requires a GPU")
    model, data, wrapper = fixture()
    data.qvel[0] = 0.3
    data.qvel[4] = 1.0
    mujoco.mj_forward(model, data)
    warp_model = mjx.put_model(model, impl="warp")
    state = mjx.put_data(model, data, impl="warp", naconmax=32, njmax=128)
    state = jax.jit(lambda x: mjx.forward(warp_model, x))(state)
    config = SimpleNamespace(foot_contact_force_threshold=1.0)
    actual = jax.jit(lambda x: foot_contact_slip(wrapper, config, x))(state)
    expected = foot_contact_slip(wrapper, config, view(data))
    assert float(expected[0]) > 0.1
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
