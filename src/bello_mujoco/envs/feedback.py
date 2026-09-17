"""Shared feedback-filter configuration and control transport timing."""

from dataclasses import dataclass
import math

import jax.numpy as jnp


@dataclass(frozen=True)
class ObservationLowPassFilterConfig:
    enabled: bool = False
    cutoff_hz: float = 10.0
    noise_before_filter: bool = False
    residual_noise_scale: float = 0.0


def validate_feedback_filter(config: ObservationLowPassFilterConfig) -> None:
    if not math.isfinite(config.cutoff_hz) or config.cutoff_hz <= 0.0:
        raise ValueError(
            "observation_low_pass_filter.cutoff_hz must be positive and finite"
        )
    if (
        not math.isfinite(config.residual_noise_scale)
        or not 0 <= config.residual_noise_scale <= 1
    ):
        raise ValueError(
            "observation_low_pass_filter.residual_noise_scale must be finite and in [0, 1]"
        )
    if config.noise_before_filter and not config.enabled:
        raise ValueError(
            "observation_low_pass_filter.noise_before_filter requires enabled=true"
        )


def feedback_filter_alpha(timestep, cutoff_hz):
    return 1.0 - jnp.exp(-2.0 * jnp.pi * cutoff_hz * timestep)


def validate_command_delay(delay: float, physics_dt: float, decimation: int) -> None:
    if not math.isfinite(delay) or not 0 <= delay < decimation * physics_dt:
        raise ValueError(
            "command_delay_seconds must be finite and in [0, control_timestep)"
        )
    if not math.isclose(delay / physics_dt, round(delay / physics_dt), abs_tol=1e-6):
        raise ValueError(
            "command_delay_seconds must be a whole number of physics substeps"
        )


def delayed_ctrl(previous, requested, substep, *, delay_seconds, physics_dt):
    return jnp.where(
        jnp.asarray(substep) < round(delay_seconds / physics_dt), previous, requested
    )
