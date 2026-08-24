"""Command sampling and pure reward functions for Bello locomotion."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math

import chex
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class CommandConfig:
    """Explicit standing plus conditioned moving velocity command sampler."""

    enabled: bool = True
    lin_vel_x: tuple[float, float] = (-0.5, 0.5)
    lin_vel_y: tuple[float, float] = (-0.5, 0.5)
    ang_vel_yaw: tuple[float, float] = (-0.5, 0.5)
    standing_probability: float = 0.10
    linear_deadband: float = 0.10
    yaw_deadband: float = 0.10
    minimum_moving_linear_speed: float = 0.20
    minimum_moving_yaw_speed: float = 0.20
    resample_seconds: float = 10.0
    standing_transitions_enabled: bool = False

    def __post_init__(self) -> None:
        validate_range("lin_vel_x", self.lin_vel_x)
        validate_range("lin_vel_y", self.lin_vel_y)
        validate_range("ang_vel_yaw", self.ang_vel_yaw)
        if not 0.0 <= self.standing_probability <= 1.0:
            raise ValueError(
                "standing_probability must be in [0, 1], got "
                f"{self.standing_probability}"
            )
        for name, value in (
            ("linear_deadband", self.linear_deadband),
            ("yaw_deadband", self.yaw_deadband),
            ("minimum_moving_linear_speed", self.minimum_moving_linear_speed),
            ("minimum_moving_yaw_speed", self.minimum_moving_yaw_speed),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and finite, got {value}")
        if self.minimum_moving_linear_speed <= self.linear_deadband:
            raise ValueError(
                "minimum_moving_linear_speed must exceed linear_deadband, got "
                f"{self.minimum_moving_linear_speed} <= {self.linear_deadband}"
            )
        if self.minimum_moving_yaw_speed <= self.yaw_deadband:
            raise ValueError(
                "minimum_moving_yaw_speed must exceed yaw_deadband, got "
                f"{self.minimum_moving_yaw_speed} <= {self.yaw_deadband}"
            )
        if self.enabled and self.standing_probability < 1.0:
            moving_axes = self._moving_axis_mask()
            if not any(moving_axes):
                raise ValueError(
                    "at least one command range must reach its minimum moving "
                    "speed when standing_probability is below 1"
                )
        if not math.isfinite(self.resample_seconds) or self.resample_seconds <= 0:
            raise ValueError(
                f"resample_seconds must be positive, got {self.resample_seconds}"
            )

    def range_array(self) -> jax.Array:
        return jnp.asarray(
            (self.lin_vel_x, self.lin_vel_y, self.ang_vel_yaw), dtype=jnp.float32
        )

    def _minimum_moving_speeds(self) -> tuple[float, float, float]:
        return (
            self.minimum_moving_linear_speed,
            self.minimum_moving_linear_speed,
            self.minimum_moving_yaw_speed,
        )

    def _moving_axis_mask(self) -> tuple[bool, bool, bool]:
        return tuple(
            low <= -minimum or high >= minimum
            for (low, high), minimum in zip(
                (self.lin_vel_x, self.lin_vel_y, self.ang_vel_yaw),
                self._minimum_moving_speeds(),
                strict=True,
            )
        )

    def sample(
        self,
        rng: jax.Array,
        shape: tuple[int, ...] = (),
    ) -> jax.Array:
        if not self.enabled:
            return jnp.zeros(shape + (3,), dtype=jnp.float32)
        if self.standing_probability == 1.0:
            return jnp.zeros(shape + (3,), dtype=jnp.float32)

        moving_key, stand_key = jax.random.split(rng)
        moving_command = self.sample_moving(moving_key, shape)
        standing = jax.random.bernoulli(stand_key, self.standing_probability, shape)
        return jnp.where(
            standing[..., None], jnp.zeros_like(moving_command), moving_command
        )

    def sample_moving(
        self,
        rng: jax.Array,
        shape: tuple[int, ...] = (),
    ) -> jax.Array:
        """Sample commands that are guaranteed to be outside the deadbands."""

        if not self.enabled or self.standing_probability == 1.0:
            return jnp.zeros(shape + (3,), dtype=jnp.float32)

        bounds = self.range_array()
        minimums = jnp.asarray(self._minimum_moving_speeds(), dtype=jnp.float32)
        negative_available = bounds[:, 0] <= -minimums
        positive_available = bounds[:, 1] >= minimums
        axis_available = negative_available | positive_available
        axis_probability = axis_available.astype(jnp.float32)
        axis_probability /= jnp.sum(axis_probability)

        command_key, axis_key, sign_key, magnitude_key = jax.random.split(rng, 4)
        command_uniform = jax.random.uniform(
            command_key, shape + (3,), dtype=jnp.float32
        )
        command = bounds[:, 0] + command_uniform * (bounds[:, 1] - bounds[:, 0])
        active_axis = jax.random.choice(
            axis_key,
            3,
            shape=shape,
            p=axis_probability,
        )
        selected_negative_available = negative_available[active_axis]
        selected_positive_available = positive_available[active_axis]
        positive = jnp.where(
            selected_negative_available & selected_positive_available,
            jax.random.bernoulli(sign_key, shape=shape),
            selected_positive_available,
        )
        active_low = jnp.where(
            positive,
            jnp.maximum(minimums[active_axis], bounds[active_axis, 0]),
            bounds[active_axis, 0],
        )
        active_high = jnp.where(
            positive,
            bounds[active_axis, 1],
            jnp.minimum(-minimums[active_axis], bounds[active_axis, 1]),
        )
        magnitude_uniform = jax.random.uniform(
            magnitude_key, shape=shape, dtype=jnp.float32
        )
        active_value = active_low + magnitude_uniform * (active_high - active_low)
        active_mask = jax.nn.one_hot(active_axis, 3, dtype=jnp.float32)
        return command * (1.0 - active_mask) + active_value[..., None] * active_mask

    def sample_stratified_batch(self, rng: jax.Array, batch_size: int) -> jax.Array:
        """Sample a batch with a fixed, randomly assigned standing cohort."""

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if not self.enabled or self.standing_probability == 1.0:
            return jnp.zeros((batch_size, 3), dtype=jnp.float32)

        moving_key, cohort_key = jax.random.split(rng)
        command = self.sample_moving(moving_key, (batch_size,))
        standing_count = int(math.floor(self.standing_probability * batch_size + 0.5))
        if standing_count == 0:
            return command
        cohort = jax.random.permutation(cohort_key, batch_size)[:standing_count]
        standing = jnp.zeros(batch_size, dtype=bool).at[cohort].set(True)
        return jnp.where(standing[:, None], jnp.zeros_like(command), command)


def command_is_standing(command: jax.Array, config: CommandConfig) -> jax.Array:
    """Return whether every command axis is inside its standing deadband."""

    linear_still = jnp.all(jnp.abs(command[..., :2]) < config.linear_deadband, axis=-1)
    yaw_still = jnp.abs(command[..., 2]) < config.yaw_deadband
    return linear_still & yaw_still


def command_is_moving(command: jax.Array, config: CommandConfig) -> jax.Array:
    """Return whether any command axis requests motion."""

    return ~command_is_standing(command, config)


@dataclass(frozen=True)
class RewardWeights:
    """Scalar weights for the stable locomotion reward term names."""

    track_lin_vel_xy: float = 2.0
    track_ang_vel_z: float = 2.0
    healthy_survival: float = 1.5
    lin_vel_z: float = -2.0
    ang_vel_xy: float = -0.15
    flat_orientation: float = -2.0
    base_height: float = -10.0
    action_rate: float = -0.01
    action_limit: float = -1.0
    torque_l1: float = -1.2
    torque_peak: float = -12.0
    joint_force_peak: float = -4.0
    touchdown_impact: float = -4.0
    joint_pose: float = -0.1
    stand_still: float = 1.0
    stand_foot_motion: float = -1.0
    stand_foot_heading: float = -0.5
    stand_contact: float = -0.5
    hip_deviation: float = -0.25
    knee_deviation: float = -0.1
    joint_limit: float = -1.0
    invalid_action: float = 0.0
    gait_contact: float = -1.0
    gait_stance_slip: float = -0.25
    gait_swing_clearance: float = -0.5
    undesired_contact: float = 0.0
    self_collision: float = 0.0
    termination: float = -100.0


@dataclass(frozen=True)
class RewardLinks:
    """Optional link overrides for link-derived reward features."""

    track_lin_vel_xy: str | None = None
    track_lin_vel_xy_frame: str | None = None
    track_ang_vel_z: str | None = None
    lin_vel_z: str | None = None
    ang_vel_xy: str | None = None
    flat_orientation: str | None = None
    base_height: str | None = None


@dataclass(frozen=True)
class TrackingRewardConfig:
    lin_vel_xy_moving_std: float = 0.30
    lin_vel_xy_still_std: float = 0.25
    ang_vel_z_moving_std: float = 0.25
    ang_vel_z_still_std: float = 0.20
    ang_vel_z_lpf_cutoff_hz: float = 5.0


@dataclass(frozen=True)
class StandingRewardConfig:
    foot_linear_speed_scale: float = 0.15
    foot_angular_speed_scale: float = 1.0
    foot_angular_weight: float = 0.25
    foot_heading_deadband: float = math.radians(5.0)


@dataclass(frozen=True)
class GaitRewardConfig:
    stance_slip_angular_speed_scale: float = 0.0
    swing_clearance_target: float = 0.08
    joint_force_soft_limit_body_weights: float = 0.8
    joint_force_top_k: int = 2
    joint_force_apply_while_standing: bool = False
    touchdown_force_soft_limit_body_weights: float = 1.0
    touchdown_vertical_speed_scale: float = 0.3


@dataclass(frozen=True)
class LimitsPoseRewardConfig:
    base_height_target: float = 0.86
    action_soft_limit_ratio: float = 0.8
    joint_soft_limit_factor: float = 0.9
    hip_lateral_command_index: int = 1
    hip_lateral_command_threshold: float = 0.1


@dataclass(frozen=True)
class RewardConfig:
    """Canonical Bello locomotion reward configuration."""

    weights: RewardWeights = RewardWeights()
    links: RewardLinks = RewardLinks()
    tracking: TrackingRewardConfig = TrackingRewardConfig()
    standing: StandingRewardConfig = StandingRewardConfig()
    gait: GaitRewardConfig = GaitRewardConfig()
    limits_pose: LimitsPoseRewardConfig = LimitsPoseRewardConfig()

    def __post_init__(self) -> None:
        validate_reward_config(self)

    def enabled(self, term_name: str) -> bool:
        return getattr(self.weights, term_name) != 0.0

    @property
    def uses_foot_contact_metadata(self) -> bool:
        return any(
            self.enabled(name)
            for name in (
                "gait_contact",
                "gait_stance_slip",
                "touchdown_impact",
                "stand_contact",
                "stand_still",
                "stand_foot_heading",
            )
        )

    @property
    def uses_foot_metadata(self) -> bool:
        return (
            self.uses_foot_contact_metadata
            or self.enabled("gait_swing_clearance")
            or self.enabled("stand_foot_motion")
            or self.enabled("stand_foot_heading")
        )

    @property
    def enabled_link_names(self) -> tuple[str, ...]:
        term_names = (
            "track_lin_vel_xy",
            "track_ang_vel_z",
            "lin_vel_z",
            "ang_vel_xy",
            "flat_orientation",
            "base_height",
        )
        links = {
            link
            for name in term_names
            if self.enabled(name) and (link := getattr(self.links, name))
        }
        if self.enabled("track_lin_vel_xy") and self.links.track_lin_vel_xy_frame:
            links.add(self.links.track_lin_vel_xy_frame)
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
    actuator_peak_force: jax.Array
    joint_force: jax.Array
    robot_weight: jax.Array
    joint_limit_low: jax.Array
    joint_limit_high: jax.Array
    policy_joint_indices: jax.Array
    action_target_default: jax.Array
    action_target_low: jax.Array
    action_target_high: jax.Array
    foot_contact: jax.Array
    foot_touchdown: jax.Array
    foot_normal_force: jax.Array
    previous_foot_linear_velocity: jax.Array
    foot_linear_velocity: jax.Array
    foot_angular_speed: jax.Array
    foot_heading_error: jax.Array
    foot_height: jax.Array
    gait_active: jax.Array
    desired_foot_contact: jax.Array
    joint_pose_weights: jax.Array
    hip_joint_indices: jax.Array
    knee_joint_indices: jax.Array
    undesired_contact: jax.Array
    self_collision: jax.Array
    done: jax.Array
    control_timestep: jax.Array
    action_clip_limit: jax.Array


def compute_locomotion_reward(
    config: RewardConfig,
    inputs: RewardInputs,
    command_config: CommandConfig | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Return the canonical weighted reward and its weighted components."""

    if command_config is None:
        command_config = CommandConfig()

    standing_command = command_is_standing(inputs.command, command_config)
    standing_double_support = standing_command & jnp.all(inputs.foot_contact)
    linear_still = jnp.abs(inputs.command[:2]) < command_config.linear_deadband
    linear_target = jnp.where(linear_still, 0.0, inputs.command[:2])
    linear_std = jnp.where(
        linear_still,
        config.tracking.lin_vel_xy_still_std,
        config.tracking.lin_vel_xy_moving_std,
    )
    lin_vel_error = jnp.sum(((inputs.lin_vel_xy - linear_target) / linear_std) ** 2)
    yaw_still = jnp.abs(inputs.command[2]) < command_config.yaw_deadband
    yaw_target = jnp.where(yaw_still, 0.0, inputs.command[2])
    yaw_std = jnp.where(
        yaw_still,
        config.tracking.ang_vel_z_still_std,
        config.tracking.ang_vel_z_moving_std,
    )
    yaw_vel_error = ((inputs.yaw_vel_z - yaw_target) / yaw_std) ** 2
    track_lin_vel_xy = jnp.exp(-lin_vel_error)
    track_ang_vel_z = jnp.exp(-yaw_vel_error)
    tracking_supported = ~standing_command | standing_double_support
    track_lin_vel_xy = jnp.where(tracking_supported, track_lin_vel_xy, 0.0)
    track_ang_vel_z = jnp.where(tracking_supported, track_ang_vel_z, 0.0)
    lin_vel_z = inputs.lin_vel_z**2
    ang_vel_xy = jnp.sum(inputs.ang_vel_xy**2)
    flat_orientation = jnp.sum(inputs.projected_gravity[:2] ** 2)
    base_height = (inputs.base_height - config.limits_pose.base_height_target) ** 2
    action_rate = jnp.sum((inputs.action - inputs.last_action) ** 2)

    action_soft_limit = (
        config.limits_pose.action_soft_limit_ratio * inputs.action_clip_limit
    )
    action_limit_excess = jnp.maximum(jnp.abs(inputs.action) - action_soft_limit, 0.0)
    action_limit = jnp.mean(
        (
            action_limit_excess
            / jnp.maximum(inputs.action_clip_limit - action_soft_limit, 1.0e-6)
        )
        ** 2
    )
    policy_force = jnp.abs(inputs.actuator_force[inputs.policy_joint_indices])
    policy_peak_force = inputs.actuator_peak_force[inputs.policy_joint_indices]
    normalized_torque = policy_force / policy_peak_force
    torque_l1 = jnp.mean(normalized_torque)
    torque_peak = jnp.mean(jnp.maximum(normalized_torque - 1.0, 0.0) ** 2)
    normalized_joint_force = inputs.joint_force / jnp.maximum(
        inputs.robot_weight, 1.0e-6
    )
    gait_active = inputs.gait_active.astype(jnp.float32)
    joint_force_excess = jnp.maximum(
        normalized_joint_force - config.gait.joint_force_soft_limit_body_weights,
        0.0,
    )
    joint_force_peak = jnp.mean(
        jnp.sort(joint_force_excess)[-config.gait.joint_force_top_k :] ** 2
    )
    joint_force_peak = jnp.where(
        inputs.gait_active | config.gait.joint_force_apply_while_standing,
        joint_force_peak,
        0.0,
    )
    foot_normal_impulse = inputs.foot_normal_force * inputs.control_timestep
    normalized_foot_impulse = foot_normal_impulse / jnp.maximum(
        inputs.robot_weight * inputs.control_timestep, 1.0e-6
    )
    touchdown_force_excess = jnp.maximum(
        normalized_foot_impulse - config.gait.touchdown_force_soft_limit_body_weights,
        0.0,
    )
    touchdown_speed_ratio = (
        jnp.maximum(-inputs.previous_foot_linear_velocity[:, 2], 0.0)
        / config.gait.touchdown_vertical_speed_scale
    )
    touchdown_impact = jnp.mean(
        inputs.foot_touchdown.astype(jnp.float32)
        * (jnp.tanh(touchdown_force_excess**2) + jnp.tanh(touchdown_speed_ratio**2))
    )
    joint_pose = jnp.sum(inputs.joint_pos_rel**2 * inputs.joint_pose_weights)
    foot_linear_speed_sq = jnp.sum(inputs.foot_linear_velocity**2, axis=1)
    stand_foot_linear_motion = jnp.tanh(
        foot_linear_speed_sq / config.standing.foot_linear_speed_scale**2
    )
    stand_foot_angular_motion = jnp.tanh(
        (inputs.foot_angular_speed / config.standing.foot_angular_speed_scale) ** 2
    )
    foot_motion = jnp.mean(
        stand_foot_linear_motion
        + config.standing.foot_angular_weight * stand_foot_angular_motion
    )
    stand_foot_motion = jnp.where(standing_command, foot_motion, 0.0)
    foot_heading_excess = jnp.maximum(
        jnp.abs(inputs.foot_heading_error) - config.standing.foot_heading_deadband,
        0.0,
    )
    stand_foot_heading = jnp.where(
        standing_double_support & ~inputs.done,
        jnp.mean(foot_heading_excess**2),
        0.0,
    )
    actual_contact = inputs.foot_contact.astype(jnp.float32)
    support_fraction = jnp.mean(actual_contact)
    stand_contact = jnp.where(
        standing_command,
        1.0 - support_fraction,
        0.0,
    )
    standing_stillness = jnp.clip(
        1.0 - foot_motion / (1.0 + config.standing.foot_angular_weight),
        0.0,
        1.0,
    )
    stand_still = jnp.where(
        standing_double_support & ~inputs.done,
        standing_stillness,
        0.0,
    )
    hip_deviation = jnp.sum(jnp.abs(inputs.joint_pos_rel[inputs.hip_joint_indices]))
    lateral_command = inputs.command[config.limits_pose.hip_lateral_command_index]
    hip_deviation *= (
        jnp.abs(lateral_command) > config.limits_pose.hip_lateral_command_threshold
    )
    knee_deviation = jnp.sum(jnp.abs(inputs.joint_pos_rel[inputs.knee_joint_indices]))

    joint_limit_center = 0.5 * (inputs.joint_limit_low + inputs.joint_limit_high)
    joint_limit_radius = (
        0.5
        * (inputs.joint_limit_high - inputs.joint_limit_low)
        * config.limits_pose.joint_soft_limit_factor
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

    desired_contact = inputs.desired_foot_contact.astype(jnp.float32)
    gait_contact = gait_active * jnp.mean(jnp.abs(actual_contact - desired_contact))
    foot_speed = jnp.linalg.norm(inputs.foot_linear_velocity[:, :2], axis=1)
    gait_stance_slip = gait_active * jnp.sum(
        desired_contact
        * (
            foot_speed
            + config.gait.stance_slip_angular_speed_scale * inputs.foot_angular_speed
        )
    )
    desired_swing = 1.0 - desired_contact
    normalized_clearance_shortfall = (
        jnp.maximum(
            config.gait.swing_clearance_target - inputs.foot_height,
            0.0,
        )
        / config.gait.swing_clearance_target
    )
    gait_swing_clearance = gait_active * jnp.mean(
        desired_swing * normalized_clearance_shortfall**2
    )

    values = {
        "track_lin_vel_xy": track_lin_vel_xy,
        "track_ang_vel_z": track_ang_vel_z,
        "healthy_survival": (~inputs.done).astype(jnp.float32),
        "lin_vel_z": lin_vel_z,
        "ang_vel_xy": ang_vel_xy,
        "flat_orientation": flat_orientation,
        "base_height": base_height,
        "action_rate": action_rate,
        "action_limit": action_limit,
        "torque_l1": torque_l1,
        "torque_peak": torque_peak,
        "joint_force_peak": joint_force_peak,
        "touchdown_impact": touchdown_impact,
        "joint_pose": joint_pose,
        "stand_still": stand_still,
        "stand_foot_motion": stand_foot_motion,
        "stand_foot_heading": stand_foot_heading,
        "stand_contact": stand_contact,
        "hip_deviation": hip_deviation,
        "knee_deviation": knee_deviation,
        "joint_limit": joint_limit,
        "invalid_action": invalid_action,
        "gait_contact": gait_contact,
        "gait_stance_slip": gait_stance_slip,
        "gait_swing_clearance": gait_swing_clearance,
        "undesired_contact": inputs.undesired_contact,
        "self_collision": inputs.self_collision,
        "termination": inputs.done.astype(jnp.float32),
    }
    timestep = jnp.asarray(inputs.control_timestep, dtype=jnp.float32)
    weights = reward_terms(config)
    terms = {
        f"reward/{name}": value * weights[name] * timestep
        for name, value in values.items()
    }
    reward = sum(terms.values(), jnp.asarray(0.0, dtype=jnp.float32))
    return reward, terms


def reward_terms(config: RewardConfig) -> dict[str, float]:
    return {
        field.name: getattr(config.weights, field.name)
        for field in fields(config.weights)
    }


def validate_reward_config(config: RewardConfig) -> None:
    for name, weight in reward_terms(config).items():
        if not math.isfinite(weight):
            raise ValueError(f"weights.{name} must be finite, got {weight}")
    for name, sigma in (
        (
            "tracking.lin_vel_xy_moving_std",
            config.tracking.lin_vel_xy_moving_std,
        ),
        (
            "tracking.lin_vel_xy_still_std",
            config.tracking.lin_vel_xy_still_std,
        ),
        (
            "tracking.ang_vel_z_moving_std",
            config.tracking.ang_vel_z_moving_std,
        ),
        (
            "tracking.ang_vel_z_still_std",
            config.tracking.ang_vel_z_still_std,
        ),
        (
            "tracking.ang_vel_z_lpf_cutoff_hz",
            config.tracking.ang_vel_z_lpf_cutoff_hz,
        ),
        (
            "standing.foot_linear_speed_scale",
            config.standing.foot_linear_speed_scale,
        ),
        (
            "standing.foot_angular_speed_scale",
            config.standing.foot_angular_speed_scale,
        ),
        (
            "gait.swing_clearance_target",
            config.gait.swing_clearance_target,
        ),
        (
            "gait.joint_force_soft_limit_body_weights",
            config.gait.joint_force_soft_limit_body_weights,
        ),
        (
            "gait.touchdown_force_soft_limit_body_weights",
            config.gait.touchdown_force_soft_limit_body_weights,
        ),
        (
            "gait.touchdown_vertical_speed_scale",
            config.gait.touchdown_vertical_speed_scale,
        ),
    ):
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"{name} must be positive and finite, got {sigma}")
    if not 0.0 <= config.limits_pose.action_soft_limit_ratio < 1.0:
        raise ValueError("limits_pose.action_soft_limit_ratio must be in [0, 1)")
    if not 0.0 < config.limits_pose.joint_soft_limit_factor <= 1.0:
        raise ValueError("limits_pose.joint_soft_limit_factor must be in (0, 1]")
    if config.limits_pose.hip_lateral_command_index not in {0, 1}:
        raise ValueError("limits_pose.hip_lateral_command_index must be 0 or 1")
    nonnegative = {
        "gait.stance_slip_angular_speed_scale": (
            config.gait.stance_slip_angular_speed_scale
        ),
        "standing.foot_angular_weight": config.standing.foot_angular_weight,
        "standing.foot_heading_deadband": config.standing.foot_heading_deadband,
        "limits_pose.hip_lateral_command_threshold": (
            config.limits_pose.hip_lateral_command_threshold
        ),
    }
    for name, value in nonnegative.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be nonnegative and finite, got {value}")
    if (
        type(config.gait.joint_force_top_k) is not int
        or config.gait.joint_force_top_k <= 0
    ):
        raise ValueError("gait.joint_force_top_k must be a positive integer")
    if type(config.gait.joint_force_apply_while_standing) is not bool:
        raise TypeError("gait.joint_force_apply_while_standing must be a boolean")
    if (
        not math.isfinite(config.limits_pose.base_height_target)
        or config.limits_pose.base_height_target <= 0
    ):
        raise ValueError("limits_pose.base_height_target must be positive and finite")


def validate_range(name: str, bounds: tuple[float, float]) -> None:
    if len(bounds) != 2:
        raise ValueError(f"{name} must have exactly two bounds, got {bounds}")
    low, high = bounds
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError(f"{name} bounds must be finite, got {bounds}")
    if low > high:
        raise ValueError(f"{name} lower bound must be <= upper bound, got {bounds}")
