"""Pure locomotion features in world and command-heading coordinates."""

import jax
import jax.numpy as jnp


def centroidal_angular_momentum(cinert: jax.Array, cvel: jax.Array) -> jax.Array:
    """Sum MuJoCo spatial momenta about the single robot's subtree COM."""
    indices = jnp.asarray([[0, 3, 4], [3, 1, 5], [4, 5, 2]])
    inertia = cinert[:, indices]
    spin = jnp.einsum("bij,bj->bi", inertia, cvel[:, :3])
    orbital = jnp.cross(cinert[:, 6:9], cvel[:, 3:])
    return jnp.sum(spin + orbital, axis=0)


def swing_cross_track_velocity(
    command: jax.Array, foot_position: jax.Array, foot_velocity: jax.Array
) -> jax.Array:
    """Transverse world foot speed, with every input expressed in heading axes.

    The desired tangent includes yaw about the pelvis. Near zero local command
    speed the penalty fades smoothly, leaving recovery foot placement free.
    """
    tangent = command[:2] + command[2] * jnp.stack(
        (-foot_position[:, 1], foot_position[:, 0]), axis=-1
    )
    speed = jnp.linalg.norm(tangent, axis=-1)
    cross = tangent[:, 0] * foot_velocity[:, 1] - tangent[:, 1] * foot_velocity[:, 0]
    return cross / jnp.maximum(speed, 0.1)
