"""Grip activation, arm reference sampling and contact checks for rigid loads."""

from pathlib import Path
import hashlib
import jax
import jax.numpy as jnp
import numpy as np

from bello_mujoco.envs.geometry import quat_rotate, quat_multiply, quat_conjugate
from bello_mujoco.models.barbell import GRIP_ANCHORS, GRIP_AXIS_SPAN_M


def load_arm_bank(path: Path, model, sha256=None, control_timestep=None):
    if sha256 is not None and hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise ValueError("arm reference checksum differs from checkpoint")
    with np.load(path, allow_pickle=False) as source:
        if not bool(source["passed"]):
            raise ValueError(
                "arm reference failed collision/torque feasibility screening"
            )
        targets = np.array(source["targets"], dtype=np.float32)
        loaded = np.array(source["loaded"], dtype=bool)
        fps = float(source["fps"])
        names = tuple(source["joint_names"].tolist())
    if names != model.actuator_joint_names:
        raise ValueError("arm bank joint mapping differs from model")
    if (
        targets.ndim != 3
        or targets.shape[2] != model.mj_model.nu
        or targets.shape[1] < 2
    ):
        raise ValueError("arm bank must have shape [poses, frames, motors]")
    if loaded.shape != (len(targets),) or not loaded.any() or loaded.all():
        raise ValueError("arm bank requires grip and unloaded rest trajectories")
    if not np.isfinite(targets).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError("invalid arm reference values")
    for target in targets.reshape(-1, targets.shape[-1]):
        model.target_limits.validate_reference(target)
    if not np.allclose(targets[:, 0], model.default_ctrl, atol=1e-5) or not np.allclose(
        targets[:, -1], model.default_ctrl, atol=1e-5
    ):
        raise ValueError("arm trajectories must start and end at the common home pose")
    policy = model.policy_ctrl_indices
    if not np.allclose(targets[:, :, policy], model.default_ctrl[policy], atol=1e-5):
        raise ValueError("arm trajectories must leave policy motor targets unchanged")
    if control_timestep is not None:
        step = (
            np.max(np.abs(np.diff(targets, axis=1)), axis=(0, 1))
            * fps
            * control_timestep
        )
        if np.any(step > model.target_limits.position_step + 1e-6):
            raise ValueError("arm reference exceeds the motor target step limits")
    return targets, loaded, fps


def arm_target(bank, step, timestep, world_id, mass, seed):
    targets, loaded, fps = bank
    duration = (targets.shape[1] - 1) / fps
    cycle = jnp.floor(step * timestep / duration).astype(jnp.int32)
    key = jax.random.fold_in(jax.random.PRNGKey(seed), world_id)
    key = jax.random.fold_in(key, cycle)
    eligible = jnp.asarray(loaded) | (mass == 0)
    index = jax.random.categorical(key, jnp.where(eligible, 0.0, -jnp.inf))
    frame = jnp.mod(step * timestep, duration) * fps
    lower = jnp.floor(frame).astype(jnp.int32)
    upper = jnp.minimum(lower + 1, targets.shape[1] - 1)
    table = jnp.asarray(targets)
    return table[index, lower] + (frame - lower) * (
        table[index, upper] - table[index, lower]
    )


def coherent_bar_reset(model, data, world_id):
    """Move bar with robot's rigid reset; unloaded worlds disconnect and park."""
    root = model.root_qpos_adr
    bar = model.bar_qpos_adr
    nominal = jnp.asarray(model.default_qpos)
    rotation = quat_multiply(
        data.qpos[root + 3 : root + 7], quat_conjugate(nominal[root + 3 : root + 7])
    )
    position = data.qpos[root : root + 3] + quat_rotate(
        rotation, nominal[bar : bar + 3] - nominal[root : root + 3]
    )
    quaternion = quat_multiply(rotation, nominal[bar + 3 : bar + 7])
    loaded = jnp.asarray(model.payload_mass)[world_id] > 0
    position = jnp.where(loaded, position, jnp.array([10.0, 0.0, 0.15]))
    qpos = (
        data.qpos.at[bar : bar + 3].set(position).at[bar + 3 : bar + 7].set(quaternion)
    )
    active = data.eq_active.at[jnp.asarray(model.grip_equality_ids)].set(loaded)
    return data.replace(qpos=qpos, eq_active=active)


def bar_contact_force(model, data, world_id):
    impl = data._impl
    geoms = jnp.clip(impl.contact__geom, 0, model.mj_model.ngeom - 1)
    bodies = jnp.asarray(model.geom_body_ids)[geoms]
    active = (jnp.arange(geoms.shape[0]) < jnp.ravel(impl.nacon)[0]) & (
        impl.contact__worldid == world_id
    )
    address = impl.contact__efc_address[:, 0]
    force = jnp.abs(impl.efc__force[jnp.clip(address, 0, impl.efc__force.shape[0] - 1)])
    # Gripped hand pairs are excluded in the shared scene; any remaining bar
    # contact (including the ground) must not support the loaded task.
    return jnp.sum(
        jnp.where(
            active & jnp.any(bodies == model.bar_body_id, axis=1) & (address >= 0),
            force,
            0.0,
        )
    )


def bar_metrics(model, data, world_id=0):
    errors, forces, angles, moments = [], [], [], []
    impl = data._impl
    for side in ("left", "right"):
        positions = []
        for kind in ("hand", "bar"):
            site = model.mj_model.site(f"{side}_{kind}_grip").id
            body = int(model.mj_model.site_bodyid[site])
            positions.append(
                data.xpos[body]
                + quat_rotate(
                    data.xquat[body], jnp.asarray(model.mj_model.site_pos[site])
                )
            )
        errors.append(jnp.linalg.norm(positions[0] - positions[1]))
        hand_body = int(
            model.mj_model.site_bodyid[model.mj_model.site(f"{side}_hand_grip").id]
        )
        grip = model.mj_model.site(f"{side}_hand_grip").id
        axis = model.mj_model.site(f"{side}_hand_grip_axis").id
        local_axis = model.mj_model.site_pos[axis] - model.mj_model.site_pos[grip]
        local_axis = (
            local_axis / np.linalg.norm(local_axis) * (1 if side == "left" else -1)
        )
        hand_axis = quat_rotate(data.xquat[hand_body], jnp.asarray(local_axis))
        bar_axis = quat_rotate(
            data.xquat[model.bar_body_id], jnp.array([0.0, 1.0, 0.0])
        )
        angles.append(
            jnp.arctan2(
                jnp.linalg.norm(jnp.cross(hand_axis, bar_axis)),
                jnp.dot(hand_axis, bar_axis),
            )
        )
        anchor_forces = []
        for anchor in GRIP_ANCHORS:
            eq = model.mj_model.equality(f"{side}_ideal_{anchor}").id
            mask = (
                (jnp.arange(impl.efc__force.shape[0]) < jnp.ravel(impl.nefc)[0])
                & (impl.efc__type == 0)
                & (impl.efc__id == eq)
            )
            indices = jnp.nonzero(mask, size=3, fill_value=0)[0]
            anchor_forces.append(
                jnp.where(jnp.any(mask), impl.efc__force[indices], 0.0)
            )
        forces.append(jnp.linalg.norm(anchor_forces[0] + anchor_forces[1]))
        moments.append(
            GRIP_AXIS_SPAN_M * jnp.linalg.norm(jnp.cross(hand_axis, anchor_forces[1]))
        )
    loaded = jnp.asarray(model.payload_mass)[world_id] > 0
    return {
        "barbell/grip_error_m": jnp.where(loaded, jnp.max(jnp.stack(errors)), 0.0),
        "barbell/grip_axis_error_rad": jnp.where(
            loaded, jnp.max(jnp.stack(angles)), 0.0
        ),
        "barbell/grip_transverse_torque_nm": jnp.where(
            loaded, jnp.max(jnp.stack(moments)), 0.0
        ),
        "barbell/grip_force_n": jnp.where(loaded, jnp.max(jnp.stack(forces)), 0.0),
        "barbell/unintended_contact_force_n": jnp.where(
            loaded, bar_contact_force(model, data, world_id), 0.0
        ),
    }
