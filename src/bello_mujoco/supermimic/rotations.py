"""Differentiable WXYZ rotations, including finite derivatives at zero angle."""

import jax.numpy as jnp


def normalize(quaternion):
    return quaternion / jnp.sqrt(
        jnp.maximum(jnp.sum(quaternion**2, axis=-1, keepdims=True), 1e-12)
    )


def conjugate(quaternion):
    return quaternion * jnp.asarray([1.0, -1.0, -1.0, -1.0])


def multiply(left, right):
    lw, lv = left[..., :1], left[..., 1:]
    rw, rv = right[..., :1], right[..., 1:]
    return jnp.concatenate(
        (
            lw * rw - jnp.sum(lv * rv, axis=-1, keepdims=True),
            lw * rv + rw * lv + jnp.cross(lv, rv),
        ),
        axis=-1,
    )


def rotate(quaternion, vector):
    q = normalize(quaternion)
    cross = 2.0 * jnp.cross(q[..., 1:], vector)
    return vector + q[..., :1] * cross + jnp.cross(q[..., 1:], cross)


def exp(vector):
    squared = jnp.sum(vector**2, axis=-1, keepdims=True)
    angle = jnp.sqrt(jnp.maximum(squared, 1e-12))
    scale = jnp.where(
        squared < 1e-6,
        0.5 - squared / 48.0 + squared**2 / 3840.0,
        jnp.sin(0.5 * angle) / angle,
    )
    scalar = jnp.where(
        squared < 1e-6,
        1.0 - squared / 8.0 + squared**2 / 384.0,
        jnp.cos(0.5 * angle),
    )
    return normalize(jnp.concatenate((scalar, scale * vector), axis=-1))


def log(quaternion):
    q = normalize(quaternion)
    q = jnp.where(q[..., :1] < 0.0, -q, q)
    squared = jnp.sum(q[..., 1:] ** 2, axis=-1, keepdims=True)
    length = jnp.sqrt(jnp.maximum(squared, 1e-12))
    scale = jnp.where(
        squared < 1e-6,
        2.0 + squared / 3.0 + 3.0 * squared**2 / 20.0,
        2.0 * jnp.arctan2(length, q[..., :1]) / length,
    )
    return scale * q[..., 1:]


def heading(quaternion):
    q = normalize(quaternion)
    w, x, y, z = (q[..., i] for i in range(4))
    sine, cosine = 2 * (w * z + x * y), 1 - 2 * (y * y + z * z)
    # Heading is defined as zero when the forward axis is vertical.
    defined = sine**2 + cosine**2 >= 1e-12
    yaw = jnp.arctan2(jnp.where(defined, sine, 0.0), jnp.where(defined, cosine, 1.0))
    zero = jnp.zeros_like(yaw)
    return jnp.stack((jnp.cos(yaw / 2), zero, zero, jnp.sin(yaw / 2)), axis=-1)


def rotation_6d(quaternion):
    columns = jnp.stack(
        (
            rotate(quaternion, jnp.asarray([1.0, 0.0, 0.0])),
            rotate(quaternion, jnp.asarray([0.0, 1.0, 0.0])),
        ),
        axis=-1,
    )
    return columns.reshape(columns.shape[:-2] + (6,))
