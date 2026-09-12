"""Reset and disturbance sampling, kept outside environment stepping."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from bello_mujoco.envs.geometry import yaw_quat_from_angle


def randomize_spawn(model, config, terrain_height, data, rng, world_id=0):
    if rng is None:
        return data
    xy_key, yaw_key, qpos_key, scale_key, qvel_key, root_vel_key = jax.random.split(
        rng, 6
    )
    root_qpos = int(model.root_qpos_adr)
    qpos = data.qpos
    root_xy = jax.random.uniform(
        xy_key,
        (2,),
        minval=config.root_xy_range[0],
        maxval=config.root_xy_range[1],
        dtype=jnp.float32,
    )
    default_root_xy = qpos[root_qpos : root_qpos + 2]
    randomized_root_xy = default_root_xy + root_xy
    terrain_height_delta = terrain_height(
        randomized_root_xy, world_id
    ) - terrain_height(default_root_xy, world_id)
    yaw = jax.random.uniform(
        yaw_key,
        (),
        minval=config.root_yaw_range[0],
        maxval=config.root_yaw_range[1],
        dtype=jnp.float32,
    )
    qpos = (
        qpos.at[root_qpos : root_qpos + 2]
        .add(root_xy)
        .at[root_qpos + 2]
        .add(terrain_height_delta)
        .at[root_qpos + 3 : root_qpos + 7]
        .set(yaw_quat_from_angle(yaw))
    )
    qpos_idx = jnp.asarray(model.actuator_qpos_indices)
    joint_noise = jax.random.uniform(
        scale_key,
        qpos_idx.shape,
        minval=config.joint_pos_noise[0],
        maxval=config.joint_pos_noise[1],
        dtype=jnp.float32,
    )
    joint_ids = jnp.asarray(model.actuator_joint_ids)
    joint_range = jnp.asarray(model.mj_model.jnt_range)[joint_ids]
    joint_scale = jax.random.uniform(
        qpos_key,
        qpos_idx.shape,
        minval=config.joint_pos_scale_range[0],
        maxval=config.joint_pos_scale_range[1],
        dtype=jnp.float32,
    )
    default_joint_pos = jnp.asarray(model.default_qpos)[qpos_idx]
    joint_pos = jnp.clip(
        default_joint_pos * joint_scale + joint_noise,
        joint_range[:, 0],
        joint_range[:, 1],
    )
    qpos = qpos.at[qpos_idx].set(joint_pos)
    qvel_idx = jnp.asarray(model.actuator_qvel_indices)
    joint_vel = jax.random.uniform(
        qvel_key,
        qvel_idx.shape,
        minval=config.joint_vel_range[0],
        maxval=config.joint_vel_range[1],
        dtype=jnp.float32,
    )
    qvel = data.qvel.at[qvel_idx].set(joint_vel)
    root_qvel = int(model.root_qvel_adr)
    root_velocity = jax.random.uniform(
        root_vel_key,
        (6,),
        minval=config.root_vel_range[0],
        maxval=config.root_vel_range[1],
        dtype=jnp.float32,
    )
    qvel = qvel.at[root_qvel : root_qvel + 6].set(root_velocity)
    return data.replace(qpos=qpos, qvel=qvel)


def apply_push_batch(model, config, control_timestep, data, step_count, rng):
    batch_size = int(step_count.shape[0])
    push_applied = jnp.zeros((batch_size,), dtype=bool)
    push_velocity = jnp.zeros((batch_size, 2), dtype=jnp.float32)
    if rng is None or not config.enabled:
        return data, push_applied, push_velocity

    interval_steps = max(
        1,
        int(round(config.interval_seconds / control_timestep)),
    )
    push_applied = (step_count > 0) & (step_count % interval_steps == 0)
    x_key, y_key = jax.random.split(rng)
    push_velocity = jnp.stack(
        [
            jax.random.uniform(
                x_key,
                (batch_size,),
                minval=config.lin_vel_x[0],
                maxval=config.lin_vel_x[1],
                dtype=jnp.float32,
            ),
            jax.random.uniform(
                y_key,
                (batch_size,),
                minval=config.lin_vel_y[0],
                maxval=config.lin_vel_y[1],
                dtype=jnp.float32,
            ),
        ],
        axis=1,
    )
    root_qvel = int(model.root_qvel_adr)
    current_qvel_xy = data.qvel[:, root_qvel : root_qvel + 2]
    qvel_xy = jnp.where(
        push_applied[:, None],
        current_qvel_xy + push_velocity,
        current_qvel_xy,
    )
    data = data.replace(qvel=data.qvel.at[:, root_qvel : root_qvel + 2].set(qvel_xy))
    push_velocity = jnp.where(
        push_applied[:, None], push_velocity, jnp.zeros_like(push_velocity)
    )
    return data, push_applied, push_velocity
