"""Immutable locomotion rollout state."""

from __future__ import annotations
import chex
import jax
from mujoco import mjx


@chex.dataclass(frozen=True)
class BelloState:
    data: mjx.Data
    last_action: jax.Array
    last_action_delta: jax.Array
    tracking_ang_vel_z: jax.Array
    actor_feedback_history: jax.Array
    actor_feedback_delay_steps: jax.Array
    foot_contact_debounce_time: jax.Array
    last_foot_contact: jax.Array
    command: jax.Array
    command_target: jax.Array
    gait_phase: jax.Array
    standing_transition_pending: (
        jax.Array
    )  # Boundaries left; -1 waits for zero command.
    command_resample_step_count: jax.Array
    step_count: jax.Array
    push_steps_remaining: jax.Array
    episode_return: jax.Array
    episode_track_lin_vel_xy: jax.Array
    episode_length: jax.Array
    last_episode_return: jax.Array
    last_episode_track_lin_vel_xy: jax.Array
    last_episode_length: jax.Array
    next_start_left: jax.Array
    standing_reward_blend: jax.Array
