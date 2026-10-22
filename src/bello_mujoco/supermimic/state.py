"""Learning state uses world-frame root twists and logical joint coordinates."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PhysicalState:
    position: jax.Array
    quaternion: jax.Array
    linear_velocity: jax.Array
    angular_velocity: jax.Array
    joint_position: jax.Array
    joint_velocity: jax.Array

    def flatten(self):
        return jnp.concatenate(jax.tree.leaves(self), axis=-1)

    @classmethod
    def unflatten(cls, value, joint_count):
        if value.shape[-1] != 13 + 2 * joint_count:
            raise ValueError("physical state dimension does not match joint layout")
        return cls(*jnp.split(value, (3, 7, 10, 13, 13 + joint_count), axis=-1))


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BodyState:
    position: jax.Array
    quaternion: jax.Array
    linear_velocity: jax.Array
    angular_velocity: jax.Array
