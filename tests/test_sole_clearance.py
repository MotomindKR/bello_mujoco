import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from bello_mujoco.envs.foot_geometry import sole_lateral_clearance


def clearance(
    position, rotation, *, shape="BOX", size=(0.12, 0.04, 0.015), axis=(0, 1, 0), xp=np
):
    return sole_lateral_clearance(
        xp.asarray(position),
        xp.asarray(rotation),
        xp.tile(xp.asarray(size), (2, 1)),
        xp.full(2, int(getattr(mujoco.mjtGeom, "mjGEOM_" + shape))),
        xp.asarray([10, 20]),
        xp.asarray([10, 20]),
        xp.asarray(axis),
        array_module=xp,
    )


def yaw(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def test_toe_in_detects_overlap_despite_unchanged_centers():
    position = np.array([[0, 0.07, 0], [0, -0.07, 0]], dtype=np.float32)
    assert clearance(position, [np.eye(3)] * 2) == pytest.approx(0.06)
    rotation = np.array([yaw(np.pi / 6), yaw(-np.pi / 6)])
    assert clearance(position, rotation) < 0
    result = jax.jit(jax.vmap(lambda p: clearance(p, rotation, xp=jnp)))(
        jnp.array([position, position[::-1]])
    )
    np.testing.assert_allclose(result[0], clearance(position, rotation), atol=1e-6)
    assert result[1] < result[0]


@pytest.mark.parametrize(
    "shape,extent",
    [
        ("BOX", 0.04),
        ("SPHERE", 0.12),
        ("CYLINDER", 0.12),
        ("CAPSULE", 0.12),
        ("ELLIPSOID", 0.04),
    ],
)
def test_analytic_primitive_projection(shape, extent):
    position = [[0, 0.5, 0], [0, -0.5, 0]]
    assert clearance(position, [np.eye(3)] * 2, shape=shape) == pytest.approx(
        1 - 2 * extent
    )


def test_roll_accounts_for_sole_thickness_and_is_heading_invariant():
    angle = np.pi / 3
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[[1, 0, 0], [0, c, -s], [0, s, c]]] * 2)
    position = np.array([[0, 0.1, 0], [0, -0.1, 0]])
    gap = clearance(position, rotation)
    assert gap == pytest.approx(0.2 - 2 * (0.04 * c + 0.015 * s))
    heading = yaw(1.1)
    assert clearance(
        position @ heading.T, heading @ rotation, axis=heading[:, 1]
    ) == pytest.approx(gap)


def test_crossing_cannot_escape_by_foreaft_or_vertical_separation():
    assert clearance([[1, -0.1, 1], [-1, 0.1, 0]], [np.eye(3)] * 2) < 0


def test_union_uses_innermost_primitive_and_foot_ownership():
    result = sole_lateral_clearance(
        np.array([[0, 0.2, 0], [0, 0.08, 0], [0, -0.2, 0]]),
        np.tile(np.eye(3), (3, 1, 1)),
        np.full((3, 3), 0.05),
        np.full(3, int(mujoco.mjtGeom.mjGEOM_SPHERE)),
        np.array([10, 10, 20]),
        np.array([10, 20]),
        np.array([0, 1, 0]),
        array_module=np,
    )
    assert result == pytest.approx(0.18)
