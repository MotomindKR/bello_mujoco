"""Locomotion rollout diagnostics, separate from stepping and rewards."""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp
from bello_mujoco.envs.gait import (
    gait_is_active,
    gait_frequency,
    desired_gait_foot_contact,
)
from bello_mujoco.envs.locomotion_state import BelloState

if TYPE_CHECKING:
    from bello_mujoco.envs.locomotion import BelloLocomotionEnv


def gait_diagnostics(
    env: BelloLocomotionEnv,
    state: BelloState,
    next_state: BelloState,
    foot_contact: jax.Array,
    foot_touchdown: jax.Array,
    foot_normal_impulse: jax.Array,
    foot_peak_normal_force: jax.Array,
) -> dict[str, jax.Array]:
    contact = foot_contact.astype(bool)
    previous = state.last_foot_contact.astype(bool)
    initialized = state.step_count > 0
    touchdown = foot_touchdown.astype(bool)
    liftoff = jnp.where(initialized, ~contact & previous, False)
    moving = gait_is_active(
        next_state.command,
        env.command_config,
        next_state.standing_transition_pending,
        env.config.gait,
    )
    desired_contact = desired_gait_foot_contact(
        next_state.gait_phase, env.config.gait
    ).astype(bool)
    contact_count = jnp.sum(contact.astype(jnp.int32))
    normalized_joint_force = env._leg_joint_force(next_state) / jnp.maximum(
        env._robot_weight(), 1.0e-6
    )
    normalized_contact_force = foot_peak_normal_force / jnp.maximum(
        env._robot_weight(), 1.0e-6
    )
    normalized_contact_impulse = foot_normal_impulse / jnp.maximum(
        env._robot_weight() * env.control_timestep, 1.0e-6
    )
    joint_force_excess = jnp.maximum(
        normalized_joint_force
        - env.reward_config.gait.joint_force_soft_limit_body_weights,
        0.0,
    )
    foot_heading_error_deg = (
        jnp.abs(env._foot_heading_error(next_state)) * 180.0 / jnp.pi
    )
    placement = env._foot_placement_inputs(next_state)
    placement_metrics = {}
    if "foot_separation" in placement:
        placement_metrics = {
            "gait/sole_signed_separation": placement["foot_separation"],
            "gait/sole_foreaft_error": jnp.abs(placement["foot_foreaft"]),
            "gait/sole_width_error": jnp.abs(
                placement["foot_separation"] - placement["nominal_stance_width"]
            ),
            "gait/feet_crossed": (placement["foot_separation"] < 0).astype(jnp.float32),
            "gait/sole_clearance_violation": (
                placement["foot_separation"] < placement["minimum_foot_separation"]
            ).astype(jnp.float32),
        }
    return {
        **placement_metrics,
        "gait/stop_steps_remaining": next_state.standing_transition_pending.astype(
            jnp.float32
        ),
        "gait/standing_reward_blend": jnp.asarray(next_state.standing_reward_blend),
        "gait/step_frequency_hz": jnp.where(
            moving, 2 * gait_frequency(next_state.command, env.config.gait), 0.0
        ),
        "gait/stance_width": env._stance_width(
            next_state, env._reward_frame_link("track_lin_vel_xy")
        ),
        "gait/foot_heading_error_mean_deg": jnp.mean(foot_heading_error_deg),
        "gait/foot_heading_error_max_deg": jnp.max(foot_heading_error_deg),
        "gait/active": moving.astype(jnp.float32),
        "gait/phase": jnp.where(moving, next_state.gait_phase, 0.0),
        "gait/desired_left_contact": jnp.where(
            moving, desired_contact[0], False
        ).astype(jnp.float32),
        "gait/desired_right_contact": jnp.where(
            moving, desired_contact[1], False
        ).astype(jnp.float32),
        "gait/contact_match": jnp.where(
            moving, jnp.mean((contact == desired_contact).astype(jnp.float32)), 0.0
        ),
        "gait/touchdown": jnp.sum(touchdown.astype(jnp.float32)),
        "gait/liftoff": jnp.sum(liftoff.astype(jnp.float32)),
        "gait/single_support": (contact_count == 1).astype(jnp.float32),
        "gait/double_support": (contact_count == 2).astype(jnp.float32),
        "gait/flight": (contact_count == 0).astype(jnp.float32),
        "gait/joint_force_mean_body_weight": jnp.mean(normalized_joint_force),
        "gait/joint_force_max_body_weight": jnp.max(normalized_joint_force),
        "gait/joint_force_peak_excess": jnp.mean(
            jnp.sort(joint_force_excess)[-env.reward_config.gait.joint_force_top_k :]
            ** 2
        ),
        "gait/contact_force_mean_body_weight": jnp.mean(normalized_contact_force),
        "gait/contact_force_max_body_weight": jnp.max(normalized_contact_force),
        "gait/contact_impulse_mean_body_weight": jnp.mean(normalized_contact_impulse),
        "gait/contact_impulse_max_body_weight": jnp.max(normalized_contact_impulse),
    }


def command_diagnostics(
    env: BelloLocomotionEnv, command: jax.Array, pending=0
) -> dict[str, jax.Array]:
    standing = ~gait_is_active(command, env.command_config, pending, env.config.gait)
    return {
        "command/standing": standing.astype(jnp.float32),
        "command/moving": (~standing).astype(jnp.float32),
    }
