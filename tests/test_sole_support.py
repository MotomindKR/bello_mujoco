import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from bello_mujoco.envs.foot_geometry import sole_support_positions


def supports(position, slope, xp):
    normal = xp.asarray([-slope, 0.0, 1.0], dtype=xp.float32)
    normal = normal / xp.linalg.norm(normal)
    kinds = [
        mujoco.mjtGeom.mjGEOM_SPHERE,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        mujoco.mjtGeom.mjGEOM_BOX,
    ]
    return sole_support_positions(
        geom_position=position,
        geom_rotation=xp.tile(xp.eye(3, dtype=xp.float32), (5, 1, 1)),
        geom_size=xp.tile(xp.asarray([0.2, 0.4, 0.6], dtype=xp.float32), (5, 1)),
        geom_type=xp.asarray([int(kind) for kind in kinds]),
        geom_body_ids=xp.arange(5),
        foot_body_ids=xp.arange(5),
        terrain_up=xp.broadcast_to(normal, (5, 3)),
        terrain_height=lambda xy: slope * xy[..., 0],
        array_module=xp,
    )


def test_flat_support_matches_analytic_primitive_extents():
    position = np.tile(np.asarray([0, 0, 1], dtype=np.float32), (5, 1))
    result = supports(position, 0.0, np)
    np.testing.assert_allclose(result[:, 2], [0.8, 0.4, 0.4, 0.6, 0.4], atol=1e-6)


@pytest.mark.parametrize("slope", [-0.4, 0.0, 0.3])
def test_compiled_jax_support_matches_native_numpy(slope):
    position = np.random.default_rng(3).normal(size=(5, 3)).astype(np.float32)
    actual = jax.jit(lambda point: supports(point, slope, jnp))(jnp.asarray(position))
    np.testing.assert_allclose(actual, supports(position, slope, np), atol=1e-6)


def test_support_selects_lowest_primitive_belonging_to_each_foot():
    result = sole_support_positions(
        geom_position=np.asarray([[0, 0, 3], [1, 0, 1], [2, 0, 2]], dtype=np.float32),
        geom_rotation=np.tile(np.eye(3), (3, 1, 1)),
        geom_size=np.full((3, 3), 0.1),
        geom_type=np.full(3, int(mujoco.mjtGeom.mjGEOM_SPHERE)),
        geom_body_ids=np.asarray([10, 10, 20]),
        foot_body_ids=np.asarray([20, 10]),
        terrain_up=np.tile([0, 0, 1], (3, 1)),
        terrain_height=lambda xy: np.zeros(xy.shape[:-1]),
        array_module=np,
    )
    np.testing.assert_allclose(result, [[2, 0, 1.9], [1, 0, 0.9]], atol=1e-6)
