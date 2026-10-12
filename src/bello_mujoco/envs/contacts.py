"""Contact-force classification and debounce math for JAX/MJX rollouts."""

import jax
import jax.numpy as jnp
import mujoco

from bello_mujoco.envs.geometry import (
    pair_contains,
    pair_contains_each,
    pair_both_contained,
    foot_terrain_contact_membership,
)


def contact_normal_forces(mj_model, impl):
    """Decode normal loads from MJX-Warp constraint rows, as mj_contactForce does."""
    base = jnp.asarray(impl.contact__efc_address, dtype=jnp.int32)[:, 0]
    force = jnp.asarray(impl.efc__force, dtype=jnp.float32)
    if force.shape[0] == 0:
        return jnp.zeros(base.shape, dtype=jnp.float32)
    if mj_model.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL:
        dim = jnp.asarray(impl.contact__dim, dtype=jnp.int32)
        count = jnp.where(dim == 1, 1, 2 * (dim - 1))
        # Six-dimensional contacts have ten pyramid edges. The first edge alone
        # can be zero even when the contact carries the robot's weight.
        offsets = jnp.arange(10)
        addresses = base[:, None] + offsets
        valid = (base[:, None] >= 0) & (offsets < count[:, None])
        valid &= addresses < force.shape[0]
        values = force[jnp.clip(addresses, 0, force.shape[0] - 1)]
        return jnp.abs(jnp.sum(jnp.where(valid, values, 0.0), axis=1))
    return jnp.where(
        (base >= 0) & (base < force.shape[0]),
        jnp.abs(force[jnp.clip(base, 0, force.shape[0] - 1)]),
        0.0,
    )


def contact_forces(model, config, data, world_id: jax.Array | int = 0):
    foot_body_ids = jnp.asarray(model.foot_body_ids)
    foot_count = foot_body_ids.shape[0]
    if model.terrain_geom_ids.size == 0 and model.robot_geom_ids.size == 0:
        return (
            jnp.zeros(foot_count, dtype=jnp.float32),
            jnp.zeros(foot_count, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
        )

    impl = data._impl
    contact_geom = jnp.asarray(impl.contact__geom, dtype=jnp.int32)
    contact_geom = jnp.clip(contact_geom, 0, model.mj_model.ngeom - 1)
    nacon = jnp.ravel(jnp.asarray(impl.nacon, dtype=jnp.int32))[0]
    contact_worldid = jnp.asarray(impl.contact__worldid, dtype=jnp.int32)
    active = (jnp.arange(contact_geom.shape[0], dtype=jnp.int32) < nacon) & (
        contact_worldid == jnp.asarray(world_id, dtype=jnp.int32)
    )
    normal_force = contact_normal_forces(model.mj_model, impl)
    foot_forceful_contact = normal_force > jnp.asarray(
        config.foot_contact_force_threshold, dtype=jnp.float32
    )
    undesired_forceful_contact = normal_force > jnp.asarray(
        config.undesired_contact_force_threshold, dtype=jnp.float32
    )
    self_collision_forceful_contact = normal_force > jnp.asarray(
        config.self_collision_force_threshold, dtype=jnp.float32
    )

    geom_body_ids = jnp.asarray(model.geom_body_ids)
    contact_body = geom_body_ids[contact_geom]
    terrain_contact = pair_contains(contact_geom, model.terrain_geom_ids)
    self_collision_slot = (
        active
        & self_collision_forceful_contact
        & pair_both_contained(contact_geom, model.robot_geom_ids)
    )
    foot_membership = foot_terrain_contact_membership(
        contact_geom,
        model.geom_body_ids,
        model.foot_body_ids,
        model.foot_geom_ids,
        model.terrain_geom_ids,
    )
    foot_terrain_contact_slot = active & jnp.any(foot_membership, axis=1)
    foot_contact_slot = foot_terrain_contact_slot & foot_forceful_contact
    foot_contact = jnp.any(
        foot_contact_slot[:, None] & foot_membership,
        axis=0,
    ).astype(jnp.float32)
    foot_normal_force = jnp.sum(
        jnp.where(
            foot_terrain_contact_slot[:, None] & foot_membership,
            normal_force[:, None],
            0.0,
        ),
        axis=0,
    )
    undesired_contact_slot = (
        active
        & terrain_contact
        & undesired_forceful_contact
        & pair_contains(contact_geom, model.nonfoot_robot_geom_ids)
    )
    undesired_contact = jnp.sum(
        jnp.any(
            undesired_contact_slot[:, None]
            & pair_contains_each(contact_body, model.nonfoot_robot_body_ids),
            axis=0,
        ).astype(jnp.float32)
    )
    self_collision = jnp.sum(self_collision_slot.astype(jnp.float32))
    return (
        foot_contact,
        foot_normal_force,
        undesired_contact,
        self_collision,
    )


def debounce_contact(
    raw_foot_contact, previous_contact, debounce_time, done, config, timestep
):
    raw_contact = raw_foot_contact.astype(bool)
    previous_contact = previous_contact.astype(bool)
    changed = raw_contact != previous_contact
    on_threshold = jnp.asarray(config.foot_contact_on_seconds, dtype=jnp.float32)
    off_threshold = jnp.asarray(config.foot_contact_off_seconds, dtype=jnp.float32)
    threshold = jnp.where(raw_contact, on_threshold, off_threshold)
    next_debounce_time = jnp.where(
        changed,
        debounce_time + timestep,
        jnp.zeros_like(debounce_time),
    )
    accepted = changed & (next_debounce_time + 1e-6 >= threshold)
    filtered_contact = jnp.where(accepted, raw_contact, previous_contact)
    next_debounce_time = jnp.where(
        accepted | ~changed,
        jnp.zeros_like(next_debounce_time),
        next_debounce_time,
    )
    return (
        filtered_contact.astype(jnp.float32),
        jnp.where(done, jnp.zeros_like(next_debounce_time), next_debounce_time),
    )
