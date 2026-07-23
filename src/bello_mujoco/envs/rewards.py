"""Command sampling and pure reward functions for Bello locomotion."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math

import chex
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class CommandConfig:
    """Uniform ``[forward, lateral, yaw]`` velocity command sampler."""

    enabled: bool = True
    lin_vel_x: tuple[float, float] = (-0.1, 0.1)
    lin_vel_y: tuple[float, float] = (-0.1, 0.1)
    ang_vel_yaw: tuple[float, float] = (-0.1, 0.1)
    standing_probability: float = 0.10
    resample_seconds: float = 10.0

    def __post_init__(self) -> None:
        validate_range("lin_vel_x", self.lin_vel_x)
        validate_range("lin_vel_y", self.lin_vel_y)
        validate_range("ang_vel_yaw", self.ang_vel_yaw)
        if not 0.0 <= self.standing_probability <= 1.0:
            raise ValueError(
                "standing_probability must be in [0, 1], got "
                f"{self.standing_probability}"
            )
        if not math.isfinite(self.resample_seconds) or self.resample_seconds <= 0:
            raise ValueError(
                f"resample_seconds must be positive, got {self.resample_seconds}"
            )

    def range_array(self) -> jax.Array:
        return jnp.asarray(
            (self.lin_vel_x, self.lin_vel_y, self.ang_vel_yaw), dtype=jnp.float32
        )

    def sample(
        self,
        rng: jax.Array,
        shape: tuple[int, ...] = (),
    ) -> jax.Array:
        if not self.enabled:
            return jnp.zeros(shape + (3,), dtype=jnp.float32)
        bounds = self.range_array()
        command_key, stand_key = jax.random.split(rng)
        command = jax.random.uniform(
            command_key,
            shape + (3,),
            minval=bounds[:, 0].astype(jnp.float32),
            maxval=bounds[:, 1].astype(jnp.float32),
        )
        standing = jax.random.bernoulli(stand_key, self.standing_probability, shape)
        return jnp.where(standing[..., None], jnp.zeros_like(command), command)


@dataclass(frozen=True)
class WeightedRewardConfig:
    """A reward is disabled exactly when its weight is zero."""

    weight: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.weight != 0.0


@dataclass(frozen=True)
class LinkRewardConfig(WeightedRewardConfig):
    link: str | None = None


@dataclass(frozen=True)
class TrackLinVelXYRewardConfig(LinkRewardConfig):
    moving_std: float = 0.75
    still_std: float = 0.25
    command_deadband: float = 0.10
    frame_link: str | None = None


@dataclass(frozen=True)
class TrackAngVelZRewardConfig(LinkRewardConfig):
    moving_std: float = 0.50
    still_std: float = 0.20
    command_deadband: float = 0.10


@dataclass(frozen=True)
class LinVelZRewardConfig(LinkRewardConfig):
    pass


@dataclass(frozen=True)
class AngVelXYRewardConfig(LinkRewardConfig):
    pass


@dataclass(frozen=True)
class FlatOrientationRewardConfig(LinkRewardConfig):
    pass


@dataclass(frozen=True)
class BaseHeightRewardConfig(LinkRewardConfig):
    target_height: float = 0.86


@dataclass(frozen=True)
class ActionRateRewardConfig(WeightedRewardConfig):
    pass


@dataclass(frozen=True)
class ActionLimitRewardConfig(WeightedRewardConfig):
    soft_limit_ratio: float = 0.8


@dataclass(frozen=True)
class TorqueL1RewardConfig(WeightedRewardConfig):
    pass


@dataclass(frozen=True)
class JointPoseRewardConfig(WeightedRewardConfig):
    pass


@dataclass(frozen=True)
class StandDualContactRewardConfig(WeightedRewardConfig):
    command_speed_threshold: float = 0.10
    linear_velocity_sigma: float = 0.04
    yaw_velocity_sigma: float = 0.04


@dataclass(frozen=True)
class HipDeviationRewardConfig(WeightedRewardConfig):
    lateral_command_index: int = 1
    lateral_command_threshold: float = 0.1


@dataclass(frozen=True)
class KneeDeviationRewardConfig(WeightedRewardConfig):
    pass


@dataclass(frozen=True)
class JointLimitRewardConfig(WeightedRewardConfig):
    soft_limit_factor: float = 0.9


@dataclass(frozen=True)
class InvalidActionRewardConfig(WeightedRewardConfig):
    pass


@dataclass(frozen=True)
class FootSlipRewardConfig(WeightedRewardConfig):
    angular_speed_scale: float = 0.0


@dataclass(frozen=True)
class FlightRewardConfig(WeightedRewardConfig):
    command_speed_threshold: float = 0.10
    body_speed_threshold: float = 0.10
    yaw_speed_threshold: float = 0.10


@dataclass(frozen=True)
class ContactTimingRewardConfig(WeightedRewardConfig):
    min_air_time: float = 0.20
    min_contact_time: float = 0.30
    command_speed_threshold: float = 0.10
    body_speed_threshold: float = 0.10
    yaw_speed_threshold: float = 0.10


@dataclass(frozen=True)
class TouchdownAirTimeRewardConfig(WeightedRewardConfig):
    min_air_time: float = 0.20
    max_air_time: float = 0.50
    command_speed_threshold: float = 0.10
    body_speed_threshold: float = 0.10
    yaw_speed_threshold: float = 0.10


@dataclass(frozen=True)
class UndesiredContactRewardConfig(WeightedRewardConfig):
    pass


@dataclass(frozen=True)
class SelfCollisionRewardConfig(WeightedRewardConfig):
    pass


@dataclass(frozen=True)
class TerminationRewardConfig(WeightedRewardConfig):
    pass


@dataclass(frozen=True)
class RewardConfig:
    """Canonical Bello locomotion reward configuration."""

    track_lin_vel_xy: TrackLinVelXYRewardConfig = TrackLinVelXYRewardConfig(weight=1.0)
    track_ang_vel_z: TrackAngVelZRewardConfig = TrackAngVelZRewardConfig(weight=1.0)
    lin_vel_z: LinVelZRewardConfig = LinVelZRewardConfig(weight=-2.0)
    ang_vel_xy: AngVelXYRewardConfig = AngVelXYRewardConfig(weight=-0.15)
    flat_orientation: FlatOrientationRewardConfig = FlatOrientationRewardConfig(
        weight=-2.0
    )
    base_height: BaseHeightRewardConfig = BaseHeightRewardConfig(weight=-10.0)
    action_rate: ActionRateRewardConfig = ActionRateRewardConfig(weight=-0.01)
    action_limit: ActionLimitRewardConfig = ActionLimitRewardConfig(weight=-1.0)
    torque_l1: TorqueL1RewardConfig = TorqueL1RewardConfig(weight=-2.5e-5)
    joint_pose: JointPoseRewardConfig = JointPoseRewardConfig(weight=-0.1)
    stand_dual_contact: StandDualContactRewardConfig = StandDualContactRewardConfig(
        weight=2.0
    )
    hip_deviation: HipDeviationRewardConfig = HipDeviationRewardConfig(weight=-0.25)
    knee_deviation: KneeDeviationRewardConfig = KneeDeviationRewardConfig(weight=-0.1)
    joint_limit: JointLimitRewardConfig = JointLimitRewardConfig(weight=-1.0)
    invalid_action: InvalidActionRewardConfig = InvalidActionRewardConfig()
    foot_slip: FootSlipRewardConfig = FootSlipRewardConfig(weight=-0.25)
    flight: FlightRewardConfig = FlightRewardConfig(weight=-2.0)
    contact_timing: ContactTimingRewardConfig = ContactTimingRewardConfig(weight=-5.0)
    touchdown_air_time: TouchdownAirTimeRewardConfig = TouchdownAirTimeRewardConfig(
        weight=0.5
    )
    undesired_contact: UndesiredContactRewardConfig = UndesiredContactRewardConfig()
    self_collision: SelfCollisionRewardConfig = SelfCollisionRewardConfig()
    termination: TerminationRewardConfig = TerminationRewardConfig(weight=-100.0)

    def __post_init__(self) -> None:
        validate_reward_config(self)

    @property
    def uses_foot_contact_metadata(self) -> bool:
        return any(
            term.enabled
            for term in (
                self.foot_slip,
                self.stand_dual_contact,
                self.flight,
                self.contact_timing,
                self.touchdown_air_time,
            )
        )

    @property
    def uses_foot_metadata(self) -> bool:
        return self.uses_foot_contact_metadata

    @property
    def uses_terrain_contact_metadata(self) -> bool:
        return self.uses_foot_contact_metadata or self.undesired_contact.enabled

    @property
    def enabled_link_names(self) -> tuple[str, ...]:
        terms = (
            self.track_lin_vel_xy,
            self.track_ang_vel_z,
            self.lin_vel_z,
            self.ang_vel_xy,
            self.flat_orientation,
            self.base_height,
        )
        links = {term.link for term in terms if term.enabled and term.link}
        if self.track_lin_vel_xy.enabled and self.track_lin_vel_xy.frame_link:
            links.add(self.track_lin_vel_xy.frame_link)
        return tuple(sorted(links))


@chex.dataclass(frozen=True)
class RewardInputs:
    """Complete, backend-independent feature packet consumed by rewards."""

    command: jax.Array
    lin_vel_xy: jax.Array
    yaw_vel_z: jax.Array
    lin_vel_z: jax.Array
    ang_vel_xy: jax.Array
    projected_gravity: jax.Array
    base_height: jax.Array
    joint_pos: jax.Array
    joint_pos_rel: jax.Array
    action: jax.Array
    requested_action: jax.Array
    last_action: jax.Array
    actuator_force: jax.Array
    joint_limit_low: jax.Array
    joint_limit_high: jax.Array
    policy_joint_indices: jax.Array
    action_target_default: jax.Array
    action_target_low: jax.Array
    action_target_high: jax.Array
    foot_contact: jax.Array
    foot_lateral_speed_sq: jax.Array
    foot_angular_speed: jax.Array
    foot_air_time: jax.Array
    foot_contact_time: jax.Array
    last_foot_contact: jax.Array
    joint_pose_weights: jax.Array
    hip_joint_indices: jax.Array
    knee_joint_indices: jax.Array
    undesired_contact: jax.Array
    self_collision: jax.Array
    done: jax.Array
    control_timestep: jax.Array
    action_clip_limit: jax.Array


def compute_locomotion_reward(
    config: RewardConfig, inputs: RewardInputs
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Return the canonical weighted reward and its weighted components."""

    linear_still = (
        jnp.abs(inputs.command[:2]) < config.track_lin_vel_xy.command_deadband
    )
    linear_target = jnp.where(linear_still, 0.0, inputs.command[:2])
    linear_std = jnp.where(
        linear_still,
        config.track_lin_vel_xy.still_std,
        config.track_lin_vel_xy.moving_std,
    )
    lin_vel_error = jnp.sum(((inputs.lin_vel_xy - linear_target) / linear_std) ** 2)
    yaw_still = jnp.abs(inputs.command[2]) < config.track_ang_vel_z.command_deadband
    yaw_target = jnp.where(yaw_still, 0.0, inputs.command[2])
    yaw_std = jnp.where(
        yaw_still,
        config.track_ang_vel_z.still_std,
        config.track_ang_vel_z.moving_std,
    )
    yaw_vel_error = ((inputs.yaw_vel_z - yaw_target) / yaw_std) ** 2
    track_lin_vel_xy = jnp.exp(-lin_vel_error)
    track_ang_vel_z = jnp.exp(-yaw_vel_error)
    lin_vel_z = inputs.lin_vel_z**2
    ang_vel_xy = jnp.sum(inputs.ang_vel_xy**2)
    flat_orientation = jnp.sum(inputs.projected_gravity[:2] ** 2)
    base_height = (inputs.base_height - config.base_height.target_height) ** 2
    action_rate = jnp.sum((inputs.action - inputs.last_action) ** 2)

    action_soft_limit = config.action_limit.soft_limit_ratio * inputs.action_clip_limit
    action_limit_excess = jnp.maximum(jnp.abs(inputs.action) - action_soft_limit, 0.0)
    action_limit = jnp.mean(
        (
            action_limit_excess
            / jnp.maximum(inputs.action_clip_limit - action_soft_limit, 1.0e-6)
        )
        ** 2
    )
    torque_l1 = jnp.sum(jnp.abs(inputs.actuator_force[inputs.policy_joint_indices]))
    joint_pose = jnp.sum(inputs.joint_pos_rel**2 * inputs.joint_pose_weights)
    command_speed = jnp.linalg.norm(inputs.command)
    body_speed = jnp.linalg.norm(inputs.lin_vel_xy)
    standing_command = (
        command_speed <= config.stand_dual_contact.command_speed_threshold
    )
    both_feet_contact = jnp.all(inputs.foot_contact.astype(jnp.bool_))
    stand_stationarity = jnp.exp(
        -(body_speed**2) / config.stand_dual_contact.linear_velocity_sigma
        - inputs.yaw_vel_z**2 / config.stand_dual_contact.yaw_velocity_sigma
    )
    stand_dual_contact = jnp.where(
        standing_command,
        both_feet_contact.astype(jnp.float32) * stand_stationarity,
        0.0,
    )
    hip_deviation = jnp.sum(jnp.abs(inputs.joint_pos_rel[inputs.hip_joint_indices]))
    lateral_command = inputs.command[config.hip_deviation.lateral_command_index]
    hip_deviation *= (
        jnp.abs(lateral_command) > config.hip_deviation.lateral_command_threshold
    )
    knee_deviation = jnp.sum(jnp.abs(inputs.joint_pos_rel[inputs.knee_joint_indices]))

    joint_limit_center = 0.5 * (inputs.joint_limit_low + inputs.joint_limit_high)
    joint_limit_radius = (
        0.5
        * (inputs.joint_limit_high - inputs.joint_limit_low)
        * config.joint_limit.soft_limit_factor
    )
    joint_limit_low = joint_limit_center - joint_limit_radius
    joint_limit_high = joint_limit_center + joint_limit_radius
    joint_limit = jnp.sum(
        jnp.maximum(joint_limit_low - inputs.joint_pos, 0.0)
        + jnp.maximum(inputs.joint_pos - joint_limit_high, 0.0)
    )

    normalized_requested_action = inputs.requested_action / inputs.action_clip_limit
    requested_target_span = jnp.where(
        normalized_requested_action >= 0.0,
        inputs.action_target_high - inputs.action_target_default,
        inputs.action_target_default - inputs.action_target_low,
    )
    requested_target = (
        inputs.action_target_default
        + requested_target_span * normalized_requested_action
    )
    target_violation = jnp.maximum(
        inputs.action_target_low - requested_target, 0.0
    ) + jnp.maximum(requested_target - inputs.action_target_high, 0.0)
    unit_violation = target_violation / jnp.maximum(
        jnp.abs(requested_target_span), 1.0e-6
    )
    clip_violation = jnp.maximum(
        jnp.abs(inputs.requested_action) - inputs.action_clip_limit, 0.0
    )
    invalid_action = jnp.sum(unit_violation**2 + clip_violation**2)

    foot_speed = jnp.sqrt(jnp.maximum(inputs.foot_lateral_speed_sq, 0.0))
    foot_slip = jnp.sum(
        inputs.foot_contact
        * (
            foot_speed
            + config.foot_slip.angular_speed_scale * inputs.foot_angular_speed
        )
    )
    contact_motion_active = motion_active(
        command_speed,
        body_speed,
        jnp.abs(inputs.yaw_vel_z),
        command_speed_threshold=config.contact_timing.command_speed_threshold,
        body_speed_threshold=config.contact_timing.body_speed_threshold,
        yaw_speed_threshold=config.contact_timing.yaw_speed_threshold,
    )
    contact = inputs.foot_contact.astype(bool)
    previous_contact = inputs.last_foot_contact.astype(bool)
    filtered_contact = contact | previous_contact
    flight_motion_active = motion_active(
        command_speed,
        body_speed,
        jnp.abs(inputs.yaw_vel_z),
        command_speed_threshold=config.flight.command_speed_threshold,
        body_speed_threshold=config.flight.body_speed_threshold,
        yaw_speed_threshold=config.flight.yaw_speed_threshold,
    )
    flight = jnp.where(
        flight_motion_active,
        ~jnp.any(filtered_contact),
        False,
    ).astype(jnp.float32)
    touchdown = contact & ~previous_contact
    liftoff = ~contact & previous_contact
    short_air = (
        jnp.maximum(config.contact_timing.min_air_time - inputs.foot_air_time, 0.0)
        / config.contact_timing.min_air_time
    )
    short_contact = (
        jnp.maximum(
            config.contact_timing.min_contact_time - inputs.foot_contact_time, 0.0
        )
        / config.contact_timing.min_contact_time
    )
    contact_timing = jnp.where(
        contact_motion_active,
        jnp.sum(
            touchdown.astype(jnp.float32) * short_air**2
            + liftoff.astype(jnp.float32) * short_contact**2
        ),
        0.0,
    )

    first_contact = (inputs.foot_air_time > 0.0) & filtered_contact
    touchdown_air_time = jnp.sum(
        jnp.minimum(
            (inputs.foot_air_time - config.touchdown_air_time.min_air_time)
            * first_contact.astype(jnp.float32),
            config.touchdown_air_time.max_air_time
            - config.touchdown_air_time.min_air_time,
        )
    )
    touchdown_motion_active = motion_active(
        command_speed,
        body_speed,
        jnp.abs(inputs.yaw_vel_z),
        command_speed_threshold=config.touchdown_air_time.command_speed_threshold,
        body_speed_threshold=config.touchdown_air_time.body_speed_threshold,
        yaw_speed_threshold=config.touchdown_air_time.yaw_speed_threshold,
    )
    touchdown_air_time = jnp.where(
        touchdown_motion_active,
        touchdown_air_time,
        0.0,
    )

    values = {
        "track_lin_vel_xy": track_lin_vel_xy,
        "track_ang_vel_z": track_ang_vel_z,
        "lin_vel_z": lin_vel_z,
        "ang_vel_xy": ang_vel_xy,
        "flat_orientation": flat_orientation,
        "base_height": base_height,
        "action_rate": action_rate,
        "action_limit": action_limit,
        "torque_l1": torque_l1,
        "joint_pose": joint_pose,
        "stand_dual_contact": stand_dual_contact,
        "hip_deviation": hip_deviation,
        "knee_deviation": knee_deviation,
        "joint_limit": joint_limit,
        "invalid_action": invalid_action,
        "foot_slip": foot_slip,
        "flight": flight,
        "contact_timing": contact_timing,
        "touchdown_air_time": touchdown_air_time,
        "undesired_contact": inputs.undesired_contact,
        "self_collision": inputs.self_collision,
        "termination": inputs.done.astype(jnp.float32),
    }
    timestep = jnp.asarray(inputs.control_timestep, dtype=jnp.float32)
    terms = {
        f"reward/{name}": value * reward_terms(config)[name].weight * timestep
        for name, value in values.items()
    }
    reward = sum(terms.values(), jnp.asarray(0.0, dtype=jnp.float32))
    return reward, terms


def reward_terms(config: RewardConfig) -> dict[str, WeightedRewardConfig]:
    return {field.name: getattr(config, field.name) for field in fields(config)}


def motion_active(
    command_speed: jax.Array,
    body_speed: jax.Array,
    yaw_speed: jax.Array,
    *,
    command_speed_threshold: float,
    body_speed_threshold: float,
    yaw_speed_threshold: float,
) -> jax.Array:
    return (
        (command_speed > command_speed_threshold)
        | (body_speed > body_speed_threshold)
        | (yaw_speed > yaw_speed_threshold)
    )


def validate_reward_config(config: RewardConfig) -> None:
    for name, term in reward_terms(config).items():
        if not math.isfinite(term.weight):
            raise ValueError(f"{name}.weight must be finite, got {term.weight}")
    for name, sigma in (
        ("track_lin_vel_xy.moving_std", config.track_lin_vel_xy.moving_std),
        ("track_lin_vel_xy.still_std", config.track_lin_vel_xy.still_std),
        ("track_ang_vel_z.moving_std", config.track_ang_vel_z.moving_std),
        ("track_ang_vel_z.still_std", config.track_ang_vel_z.still_std),
        (
            "stand_dual_contact.linear_velocity_sigma",
            config.stand_dual_contact.linear_velocity_sigma,
        ),
        (
            "stand_dual_contact.yaw_velocity_sigma",
            config.stand_dual_contact.yaw_velocity_sigma,
        ),
    ):
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"{name} must be positive and finite, got {sigma}")
    if not 0.0 <= config.action_limit.soft_limit_ratio < 1.0:
        raise ValueError("action_limit.soft_limit_ratio must be in [0, 1)")
    if not 0.0 < config.joint_limit.soft_limit_factor <= 1.0:
        raise ValueError("joint_limit.soft_limit_factor must be in (0, 1]")
    if config.hip_deviation.lateral_command_index not in {0, 1}:
        raise ValueError("hip_deviation.lateral_command_index must be 0 or 1")
    nonnegative = {
        "foot_slip.angular_speed_scale": config.foot_slip.angular_speed_scale,
        "hip_deviation.lateral_command_threshold": (
            config.hip_deviation.lateral_command_threshold
        ),
        "track_lin_vel_xy.command_deadband": (config.track_lin_vel_xy.command_deadband),
        "track_ang_vel_z.command_deadband": config.track_ang_vel_z.command_deadband,
        "contact_timing.command_speed_threshold": (
            config.contact_timing.command_speed_threshold
        ),
        "flight.command_speed_threshold": config.flight.command_speed_threshold,
        "touchdown_air_time.command_speed_threshold": (
            config.touchdown_air_time.command_speed_threshold
        ),
        "contact_timing.body_speed_threshold": (
            config.contact_timing.body_speed_threshold
        ),
        "flight.body_speed_threshold": config.flight.body_speed_threshold,
        "contact_timing.yaw_speed_threshold": (
            config.contact_timing.yaw_speed_threshold
        ),
        "flight.yaw_speed_threshold": config.flight.yaw_speed_threshold,
        "touchdown_air_time.body_speed_threshold": (
            config.touchdown_air_time.body_speed_threshold
        ),
        "touchdown_air_time.yaw_speed_threshold": (
            config.touchdown_air_time.yaw_speed_threshold
        ),
        "stand_dual_contact.command_speed_threshold": (
            config.stand_dual_contact.command_speed_threshold
        ),
    }
    for name, value in nonnegative.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be nonnegative and finite, got {value}")
    if (
        not math.isfinite(config.base_height.target_height)
        or config.base_height.target_height <= 0
    ):
        raise ValueError("base_height.target_height must be positive and finite")
    if (
        not math.isfinite(config.contact_timing.min_air_time)
        or config.contact_timing.min_air_time <= 0
    ):
        raise ValueError("contact_timing.min_air_time must be positive and finite")
    if (
        not math.isfinite(config.contact_timing.min_contact_time)
        or config.contact_timing.min_contact_time <= 0
    ):
        raise ValueError("contact_timing.min_contact_time must be positive and finite")
    if (
        not math.isfinite(config.touchdown_air_time.min_air_time)
        or not math.isfinite(config.touchdown_air_time.max_air_time)
        or config.touchdown_air_time.min_air_time < 0
        or config.touchdown_air_time.max_air_time
        <= config.touchdown_air_time.min_air_time
    ):
        raise ValueError(
            "touchdown_air_time bounds must be finite and require "
            "0 <= min_air_time < max_air_time"
        )


def validate_range(name: str, bounds: tuple[float, float]) -> None:
    if len(bounds) != 2:
        raise ValueError(f"{name} must have exactly two bounds, got {bounds}")
    low, high = bounds
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError(f"{name} bounds must be finite, got {bounds}")
    if low > high:
        raise ValueError(f"{name} lower bound must be <= upper bound, got {bounds}")
