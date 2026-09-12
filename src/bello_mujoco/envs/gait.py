"""Pure gait and command transitions shared across simulation and deployment."""

from __future__ import annotations
from dataclasses import dataclass
import math
import jax
import jax.numpy as jnp
from bello_mujoco.envs.rewards import (
    CommandConfig,
    command_is_moving,
    command_is_standing,
)


@dataclass(frozen=True)
class GaitConfig:
    """Alternating gait clock shared by training, evaluation, and hardware."""

    frequency_hz: float = 0.9
    stance_fraction: float = 0.60
    speed_dependent: bool = False
    step_frequency_base: float = 1.6
    step_frequency_speed_gain: float = 0.35
    stop_extra_steps: int = 0
    command_slew_enabled: bool = False
    command_linear_slew_rate: float = 1.0
    command_yaw_slew_rate: float = 1.0
    directional_start: bool = False
    lateral_start_deadband: float = 0.1
    standing_reward_ramp_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.frequency_hz) or self.frequency_hz <= 0.0:
            raise ValueError("gait.frequency_hz must be positive and finite")
        if not math.isfinite(self.stance_fraction) or not (
            0.5 <= self.stance_fraction < 1.0
        ):
            raise ValueError("gait.stance_fraction must be in [0.5, 1.0)")
        if type(self.stop_extra_steps) is not int or self.stop_extra_steps < 0:
            raise ValueError("gait.stop_extra_steps must be a nonnegative integer")
        for name in (
            "step_frequency_base",
            "step_frequency_speed_gain",
            "lateral_start_deadband",
            "standing_reward_ramp_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"gait.{name} must be finite and nonnegative")
        if self.step_frequency_base <= 0:
            raise ValueError("gait.step_frequency_base must be positive")
        for value in (self.command_linear_slew_rate, self.command_yaw_slew_rate):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("gait command slew rates must be positive and finite")


def gait_frequency(command: jax.Array, config: GaitConfig) -> jax.Array:
    if config.speed_dependent:
        return 0.5 * (
            config.step_frequency_base
            + config.step_frequency_speed_gain
            * jnp.linalg.norm(command[..., :2], axis=-1)
        )
    return jnp.full(command.shape[:-1], config.frequency_hz, dtype=jnp.float32)


def gait_is_active(
    command: jax.Array,
    command_config: CommandConfig,
    pending: jax.Array | int = 0,
    gait_config: GaitConfig = GaitConfig(),
) -> jax.Array:
    moving = (
        jnp.any(command != 0, axis=-1)
        if gait_config.command_slew_enabled
        else command_is_moving(command, command_config)
    )
    return moving | (jnp.asarray(pending) != 0)


def next_start_preference(
    command: jax.Array,
    next_command: jax.Array,
    pending: jax.Array,
    preference: jax.Array,
    command_config: CommandConfig,
    gait_config: GaitConfig = GaitConfig(),
) -> jax.Array:
    starting = ~gait_is_active(
        command, command_config, pending, gait_config
    ) & gait_is_active(next_command, command_config, 0, gait_config)
    return jnp.where(starting, ~jnp.asarray(preference), preference)


def advance_standing_blend(
    blend: jax.Array,
    command: jax.Array,
    pending: jax.Array,
    command_config: CommandConfig,
    gait_config: GaitConfig,
    dt: float,
) -> jax.Array:
    increment = dt / max(gait_config.standing_reward_ramp_seconds, dt)
    return jnp.where(
        gait_is_active(command, command_config, pending, gait_config),
        0.0,
        jnp.minimum(jnp.asarray(blend) + increment, 1.0),
    )


def advance_gait_phase(
    phase: jax.Array,
    command: jax.Array,
    command_config: CommandConfig,
    gait_config: GaitConfig,
    control_timestep: float,
    standing_transition_pending: jax.Array = 0,
) -> jax.Array:
    """Advance the normalized gait clock, resetting it for standing commands."""

    next_phase = jnp.mod(
        phase + gait_frequency(command, gait_config) * control_timestep,
        1.0,
    )
    next_phase = jnp.where(
        jnp.asarray(standing_transition_pending) < 0, phase, next_phase
    )
    return jnp.where(
        gait_is_active(
            command, command_config, standing_transition_pending, gait_config
        ),
        next_phase,
        0.0,
    )


def gait_phase_features(
    phase: jax.Array,
    command: jax.Array,
    command_config: CommandConfig,
    standing_transition_pending: jax.Array = 0,
    gait_config: GaitConfig = GaitConfig(),
) -> jax.Array:
    """Return sin/cos phase plus an explicit active bit; standing is all zero."""

    active = gait_is_active(
        command, command_config, standing_transition_pending, gait_config
    )
    angle = 2.0 * jnp.pi * phase
    features = jnp.stack(
        (jnp.sin(angle), jnp.cos(angle), active.astype(jnp.float32)), axis=-1
    )
    return jnp.where(active[..., None], features, jnp.zeros_like(features))


def desired_gait_foot_contact(
    phase: jax.Array,
    gait_config: GaitConfig,
) -> jax.Array:
    """Return desired left/right stance indicators at half-cycle phase offset."""

    offsets = jnp.asarray((0.0, 0.5), dtype=jnp.float32)
    foot_phase = jnp.mod(phase[..., None] + offsets, 1.0)
    return (foot_phase < gait_config.stance_fraction).astype(jnp.float32)


def resolve_gait_command_transition(
    command: jax.Array,
    requested_command: jax.Array,
    previous_phase: jax.Array,
    next_phase: jax.Array,
    standing_transition_pending: jax.Array,
    command_config: CommandConfig,
    gait_config: GaitConfig = GaitConfig(),
    next_start_left: jax.Array = True,
    left_axis: jax.Array = (0.0, 1.0),
    elapsed_seconds: float | jax.Array = 0.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Resolve a command request without interrupting an active gait step.

    Positive pending counts remaining half-cycle boundaries; -1 waits at double
    support for the command slew to finish. Zero means no stop is pending.
    """

    command = jnp.asarray(command, dtype=jnp.float32)
    requested_command = jnp.asarray(requested_command, dtype=jnp.float32)
    previous_phase = jnp.asarray(previous_phase, dtype=jnp.float32)
    next_phase = jnp.asarray(next_phase, dtype=jnp.float32)
    pending = jnp.asarray(standing_transition_pending, dtype=jnp.int32)
    requested_standing = command_is_standing(requested_command, command_config)
    command_standing = ~gait_is_active(command, command_config, pending, gait_config)
    lateral = jnp.sum(requested_command[..., :2] * jnp.asarray(left_axis), axis=-1)
    start_left = jnp.where(
        jnp.abs(lateral) > gait_config.lateral_start_deadband,
        lateral > 0,
        next_start_left,
    )
    start_phase = (
        jnp.where(start_left, 0.5, 0.0) if gait_config.directional_start else 0.0
    )
    if gait_config.command_slew_enabled:
        target = jnp.where(requested_standing[..., None], 0.0, requested_command)
        effective = slew_command(command, target, gait_config, elapsed_seconds)
        crossed = ((previous_phase < 0.5) & (next_phase >= 0.5)) | (
            next_phase < previous_phase
        )
        remaining = jnp.where(pending != 0, pending, 1 + gait_config.stop_extra_steps)
        remaining -= ((pending > 0) & crossed).astype(jnp.int32)
        remaining = jnp.where(remaining == 0, -1, remaining)
        stopping = requested_standing & ~command_standing
        if not command_config.standing_transitions_enabled:
            remaining = jnp.full_like(remaining, -1)
        finished = (remaining < 0) & jnp.all(effective == 0, axis=-1)
        pending_next = jnp.where(stopping & ~finished, remaining, 0)
        phase = jnp.where(
            command_standing & ~requested_standing, start_phase, next_phase
        )
        # Buffer complete: hold double support until the velocity slew reaches zero.
        phase = jnp.where(pending_next < 0, jnp.floor(phase * 2) * 0.5, phase)
        phase = jnp.where(
            requested_standing & (pending_next == 0) & jnp.all(effective == 0, axis=-1),
            0.0,
            phase,
        )
        return effective, phase, pending_next

    if not command_config.standing_transitions_enabled:
        reset_phase = requested_standing | command_standing
        return (
            jnp.where(
                requested_standing[..., None],
                jnp.zeros_like(requested_command),
                requested_command,
            ),
            jnp.where(
                reset_phase, jnp.where(requested_standing, 0.0, start_phase), next_phase
            ),
            jnp.zeros_like(pending),
        )

    crossed_half_cycle = (previous_phase < 0.5) & (next_phase >= 0.5)
    crossed_cycle = next_phase < previous_phase
    remaining = jnp.where(pending > 0, pending, 1 + gait_config.stop_extra_steps)
    reached_next_pattern = (pending > 0) & (crossed_half_cycle | crossed_cycle)
    remaining = remaining - reached_next_pattern.astype(jnp.int32)
    finish_standing_transition = (
        requested_standing & ~command_standing & (remaining == 0)
    )
    keep_moving_until_boundary = (
        requested_standing & ~command_standing & ~finish_standing_transition
    )
    effective_command = jnp.where(
        keep_moving_until_boundary[..., None], command, requested_command
    )
    effective_command = jnp.where(
        requested_standing[..., None] & ~keep_moving_until_boundary[..., None],
        jnp.zeros_like(effective_command),
        effective_command,
    )
    reset_phase = finish_standing_transition | command_standing
    effective_phase = jnp.where(
        reset_phase, jnp.where(requested_standing, 0.0, start_phase), next_phase
    )
    return (
        effective_command,
        effective_phase,
        jnp.where(keep_moving_until_boundary, remaining, 0),
    )


def slew_command(
    command: jax.Array,
    target: jax.Array,
    config: GaitConfig,
    elapsed_seconds: float | jax.Array,
) -> jax.Array:
    delta = target - command
    planar_norm = jnp.linalg.norm(delta[..., :2], axis=-1, keepdims=True)
    planar_scale = jnp.minimum(
        1.0,
        config.command_linear_slew_rate
        * elapsed_seconds
        / jnp.maximum(planar_norm, 1e-8),
    )
    planar = jnp.where(
        planar_scale >= 1.0,
        target[..., :2],
        command[..., :2] + delta[..., :2] * planar_scale,
    )
    yaw_change = config.command_yaw_slew_rate * elapsed_seconds
    yaw = jnp.where(
        jnp.abs(delta[..., 2]) <= yaw_change,
        target[..., 2],
        command[..., 2] + jnp.clip(delta[..., 2], -yaw_change, yaw_change),
    )
    return jnp.concatenate((planar, yaw[..., None]), axis=-1)
