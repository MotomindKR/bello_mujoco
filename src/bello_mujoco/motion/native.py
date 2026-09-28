"""JAX-compatible native-MuJoCo data views for motion policy inference."""

from __future__ import annotations

from dataclasses import dataclass

import jax


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NativeContactData:
    contact__geom: jax.Array
    nacon: jax.Array
    contact__worldid: jax.Array
    contact__efc_address: jax.Array
    efc__force: jax.Array
    nefc: jax.Array | None = None
    efc__id: jax.Array | None = None
    efc__type: jax.Array | None = None


@jax.tree_util.register_dataclass
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
