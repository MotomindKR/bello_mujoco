"""JAX-compatible native-MuJoCo data views for motion policy inference."""

from __future__ import annotations

from dataclasses import dataclass

import jax


@dataclass(frozen=True)
class NativeContactData:
    contact__geom: jax.Array
    nacon: jax.Array
    contact__worldid: jax.Array
    contact__efc_address: jax.Array
    efc__force: jax.Array


@dataclass(frozen=True)
class NativeMotionData:
    qpos: jax.Array
    qvel: jax.Array
    ctrl: jax.Array
    xpos: jax.Array
    xquat: jax.Array
    cvel: jax.Array
    subtree_com: jax.Array
    actuator_force: jax.Array
    _impl: NativeContactData
