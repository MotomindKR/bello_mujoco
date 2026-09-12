"""JAX-compatible rigid-body and contact array operations."""

import jax
import jax.numpy as jnp


def quat_rotate_inverse(quat_wxyz: jax.Array, vec: jax.Array) -> jax.Array:
    w, x, y, z = quat_wxyz
    qvec = jnp.asarray([x, y, z], dtype=jnp.float32)
    uv = jnp.cross(qvec, vec)
    uuv = jnp.cross(qvec, uv)
    return vec + 2.0 * (-w * uv + uuv)


def quat_rotate(quat_wxyz: jax.Array, vec: jax.Array) -> jax.Array:
    w, x, y, z = quat_wxyz
    qvec = jnp.asarray([x, y, z], dtype=jnp.float32)
    uv = jnp.cross(qvec, vec)
    uuv = jnp.cross(qvec, uv)
    return vec + 2.0 * (w * uv + uuv)


def yaw_quat(quat_wxyz: jax.Array) -> jax.Array:
    w, x, y, z = quat_wxyz
    yaw = jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half_yaw = 0.5 * yaw
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.stack([jnp.cos(half_yaw), zero, zero, jnp.sin(half_yaw)])


def yaw_quat_from_angle(yaw: jax.Array) -> jax.Array:
    half_yaw = 0.5 * yaw
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.stack([jnp.cos(half_yaw), zero, zero, jnp.sin(half_yaw)]).astype(
        jnp.float32
    )


def projected_gravity_tilt_angle(projected_gravity: jax.Array) -> jax.Array:
    return jnp.arccos(jnp.clip(-projected_gravity[2], -1.0, 1.0))


def tree_where(mask: jax.Array, true_tree, false_tree):
    def select(true_value, false_value):
        if true_value.ndim < mask.ndim or true_value.shape[: mask.ndim] != mask.shape:
            return false_value
        select_mask = mask.reshape(mask.shape + (1,) * (true_value.ndim - mask.ndim))
        return jnp.where(select_mask, true_value, false_value)

    return jax.tree_util.tree_map(select, true_tree, false_tree)


def pair_contains(pair: jax.Array, candidates) -> jax.Array:
    candidates = jnp.asarray(candidates, dtype=pair.dtype)
    if candidates.size == 0:
        return jnp.zeros(pair.shape[0], dtype=bool)
    return jnp.any(pair[:, :, None] == candidates[None, None, :], axis=(1, 2))


def pair_contains_each(pair: jax.Array, candidates) -> jax.Array:
    candidates = jnp.asarray(candidates, dtype=pair.dtype)
    if candidates.size == 0:
        return jnp.zeros((pair.shape[0], 0), dtype=bool)
    return jnp.any(pair[:, :, None] == candidates[None, None, :], axis=1)


def foot_terrain_contact_membership(
    contact_geom: jax.Array,
    geom_body_ids,
    foot_body_ids,
    foot_geom_ids,
    terrain_geom_ids,
) -> jax.Array:
    """Map sole-terrain contact pairs to their configured left/right foot."""

    contact_geom = jnp.asarray(contact_geom, dtype=jnp.int32)
    contact_body = jnp.asarray(geom_body_ids, dtype=jnp.int32)[contact_geom]
    sole_terrain = pair_contains(contact_geom, foot_geom_ids) & pair_contains(
        contact_geom, terrain_geom_ids
    )
    return sole_terrain[:, None] & pair_contains_each(contact_body, foot_body_ids)


def pair_both_contained(pair: jax.Array, candidates) -> jax.Array:
    candidates = jnp.asarray(candidates, dtype=pair.dtype)
    if candidates.size == 0:
        return jnp.zeros(pair.shape[0], dtype=bool)
    contained = jnp.any(pair[:, :, None] == candidates[None, None, :], axis=2)
    return jnp.all(contained, axis=1)
