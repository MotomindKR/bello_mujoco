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
    moving_lin_vel_xy_error: float = 0.0
    track_ang_vel_z: float = 2.0
    moving_ang_vel_z_error: float = 0.0
    healthy_survival: float = 1.5
    lin_vel_z: float = -2.0
    ang_vel_xy: float = -0.15
    flat_orientation: float = -2.0
    base_height: float = -10.0
    action_rate: float = -0.01
    action_acceleration: float = -0.005
    action_limit: float = -1.0
    torque_l1: float = -1.2
    stand_motor_continuous_torque_excess: float = 0.0
    torque_peak: float = -12.0
    joint_force_peak: float = -4.0
    touchdown_impact: float = -4.0
    joint_pose: float = -0.1
    stand_still: float = 1.0
    stand_foot_motion: float = -1.0
    stand_foot_heading: float = -0.5
    stand_contact: float = -0.5
    stand_upright: float = -8.0
    stand_foot_foreaft: float = 0.0
    stand_foot_width: float = 0.0
    foot_heading: float = 0.0
    foot_slip: float = 0.0
    foot_crossing: float = 0.0
    foot_stumble: float = 0.0
    hip_deviation: float = -0.25
    knee_deviation: float = -0.1
    joint_limit: float = -1.0
    gait_contact: float = -1.0
    gait_stance_slip: float = -0.25
    gait_swing_clearance: float = -0.5
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
    lin_vel_xy_error_delta: float = 0.25
    ang_vel_z_moving_std: float = 0.25
    ang_vel_z_still_std: float = 0.20
    ang_vel_z_error_delta: float = 0.25
    ang_vel_z_lpf_cutoff_hz: float = 5.0


@dataclass(frozen=True)
class StandingRewardConfig:
    foot_linear_speed_scale: float = 0.15
    foot_angular_speed_scale: float = 1.0
    foot_angular_weight: float = 0.25
    foot_heading_deadband: float = math.radians(5.0)
    foot_heading_scale: float = math.radians(15.0)
    position_deadband: float = 0.02
    position_scale: float = 0.03
    contact_independent_alignment: bool = False


@dataclass(frozen=True)
class GaitRewardConfig:
    stance_slip_angular_speed_scale: float = 0.0
    swing_clearance_target: float = 0.08
    joint_force_soft_limit_body_weights: float = 0.8
    joint_force_top_k: int = 2
    joint_force_apply_while_standing: bool = False
    touchdown_force_soft_limit_body_weights: float = 0.6
    touchdown_vertical_speed_scale: float = 0.3
    phase_shaped_clearance: bool = False
    crossing_scale: float = 0.02
    sole_clearance_margin: float = 0.01
    stumble_grace_fraction: float = 0.15
    stumble_load_body_weights: float = 0.05
    stumble_speed_scale: float = 0.15


@dataclass(frozen=True)
class SmoothnessRewardConfig:
    ankle_action_weight: float = 2.0
    waist_action_weight: float = 2.0


@dataclass(frozen=True)
class LimitsPoseRewardConfig:
    base_height_target: float = 0.86
    action_soft_limit_ratio: float = 0.8
    joint_soft_limit_factor: float = 0.9
    hip_deviation_command_indices: tuple[int, ...] = (0,)
    hip_deviation_command_threshold: float = 0.1
    knee_deviation_apply_while_moving: bool = False
    joint_pose_weights: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class RewardConfig:
    """Canonical Bello locomotion reward configuration."""

    weights: RewardWeights = RewardWeights()
    links: RewardLinks = RewardLinks()
    tracking: TrackingRewardConfig = TrackingRewardConfig()
    standing: StandingRewardConfig = StandingRewardConfig()
    gait: GaitRewardConfig = GaitRewardConfig()
    smoothness: SmoothnessRewardConfig = SmoothnessRewardConfig()
    limits_pose: LimitsPoseRewardConfig = LimitsPoseRewardConfig()

    def __post_init__(self) -> None:
        validate_reward_config(self)

    def enabled(self, term_name: str) -> bool:
        return getattr(self.weights, term_name) != 0.0

    @property
    def uses_planar_velocity(self) -> bool:
        return self.enabled("track_lin_vel_xy") or self.enabled(
            "moving_lin_vel_xy_error"
        )

    @property
    def uses_yaw_velocity(self) -> bool:
        return self.enabled("track_ang_vel_z") or self.enabled("moving_ang_vel_z_error")

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
                "stand_foot_foreaft",
                "stand_foot_width",
                "foot_stumble",
                "foot_slip",
            )
        )

    @property
    def uses_foot_metadata(self) -> bool:
        return (
            self.uses_foot_contact_metadata
            or self.enabled("gait_swing_clearance")
            or self.enabled("stand_foot_motion")
            or self.enabled("stand_foot_heading")
            or self.enabled("foot_crossing")
            or self.enabled("foot_heading")
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
        links = set()
        for name in term_names:
            if name == "track_lin_vel_xy":
                enabled = self.uses_planar_velocity
            elif name == "track_ang_vel_z":
                enabled = self.uses_yaw_velocity
            else:
                enabled = self.enabled(name)
            if enabled and (link := getattr(self.links, name)):
                links.add(link)
        if self.enabled("stand_upright") and self.links.flat_orientation:
            links.add(self.links.flat_orientation)
        if self.uses_planar_velocity and self.links.track_lin_vel_xy_frame:
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
    last_action: jax.Array
    last_action_delta: jax.Array
    action_smoothness_weights: jax.Array
    actuator_force: jax.Array
    requested_actuator_force: jax.Array
    actuator_peak_force: jax.Array
    joint_force: jax.Array
    robot_weight: jax.Array
    joint_limit_low: jax.Array
    joint_limit_high: jax.Array
    policy_joint_indices: jax.Array
    foot_contact: jax.Array
    foot_touchdown: jax.Array
    foot_normal_impulse: jax.Array
    foot_peak_normal_force: jax.Array
    touchdown_foot_linear_velocity: jax.Array
    foot_linear_velocity: jax.Array
    foot_angular_speed: jax.Array
    foot_heading_error: jax.Array
    foot_height: jax.Array
    gait_active: jax.Array
    desired_foot_contact: jax.Array
    joint_pose_weights: jax.Array
    hip_joint_indices: jax.Array
    knee_joint_indices: jax.Array
    done: jax.Array
    control_timestep: jax.Array
    action_clip_limit: jax.Array
    standing_reward_blend: jax.Array = 1.0
    foot_foreaft: jax.Array = 0.0
    foot_separation: jax.Array = 0.0
    nominal_stance_width: jax.Array = 0.0
    minimum_foot_separation: jax.Array = 0.0
    foot_clearance_samples: jax.Array | None = None
    swing_progress: jax.Array | None = None
    motor_continuous_torque: jax.Array | None = None


def positive_huber(value: jax.Array) -> jax.Array:
    value = jnp.maximum(value, 0.0)
    return jnp.where(value <= 1.0, 0.5 * value**2, value - 0.5)


def deadband_huber(error: jax.Array, deadband: float, scale: float) -> jax.Array:
    return positive_huber((jnp.abs(error) - deadband) / scale)


def compute_locomotion_reward(
    config: RewardConfig,
    inputs: RewardInputs,
    command_config: CommandConfig | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Return the canonical weighted reward and its weighted components."""

    if command_config is None:
        command_config = CommandConfig()

    standing_command = (
        command_is_standing(inputs.command, command_config) & ~inputs.gait_active
    )
    standing_double_support = standing_command & jnp.all(inputs.foot_contact)
    linear_still = jnp.abs(inputs.command[:2]) < command_config.linear_deadband
    linear_target = jnp.where(linear_still, 0.0, inputs.command[:2])
    linear_std = jnp.where(
        linear_still,
        config.tracking.lin_vel_xy_still_std,
        config.tracking.lin_vel_xy_moving_std,
    )
    lin_vel_error = jnp.sum(((inputs.lin_vel_xy - linear_target) / linear_std) ** 2)
    linear_error_abs = jnp.abs(inputs.lin_vel_xy - linear_target)
    linear_error_delta = config.tracking.lin_vel_xy_error_delta
    moving_lin_vel_xy_error = jnp.sum(
        jnp.where(
            linear_error_abs <= linear_error_delta,
            0.5 * linear_error_abs**2 / linear_error_delta,
            linear_error_abs - 0.5 * linear_error_delta,
        )
    )
    moving_lin_vel_xy_error = jnp.where(standing_command, 0.0, moving_lin_vel_xy_error)
    yaw_still = jnp.abs(inputs.command[2]) < command_config.yaw_deadband
    yaw_target = jnp.where(yaw_still, 0.0, inputs.command[2])
    yaw_std = jnp.where(
        yaw_still,
        config.tracking.ang_vel_z_still_std,
        config.tracking.ang_vel_z_moving_std,
    )
    yaw_vel_error = ((inputs.yaw_vel_z - yaw_target) / yaw_std) ** 2
    angular_error_abs = jnp.abs(inputs.yaw_vel_z - yaw_target)
    angular_error_delta = config.tracking.ang_vel_z_error_delta
    moving_ang_vel_z_error = jnp.where(
        angular_error_abs <= angular_error_delta,
        0.5 * angular_error_abs**2 / angular_error_delta,
        angular_error_abs - 0.5 * angular_error_delta,
    )
    moving_ang_vel_z_error = jnp.where(standing_command, 0.0, moving_ang_vel_z_error)
    track_lin_vel_xy = jnp.exp(-lin_vel_error)
    track_ang_vel_z = jnp.exp(-yaw_vel_error)
    # Recovery steps must retain the incentive to arrest velocity. Unsupported
    # flight still earns no standing tracking reward.
    tracking_supported = ~standing_command | jnp.any(inputs.foot_contact)
    track_lin_vel_xy = jnp.where(tracking_supported, track_lin_vel_xy, 0.0)
    track_ang_vel_z = jnp.where(tracking_supported, track_ang_vel_z, 0.0)
    lin_vel_z = inputs.lin_vel_z**2
    ang_vel_xy = jnp.sum(inputs.ang_vel_xy**2)
    flat_orientation = jnp.sum(inputs.projected_gravity[:2] ** 2)
    stand_upright = jnp.where(standing_command, flat_orientation, 0.0)
    base_height = (inputs.base_height - config.limits_pose.base_height_target) ** 2
    action_delta = inputs.action - inputs.last_action
    action_rate = jnp.sum(inputs.action_smoothness_weights * action_delta**2)
    action_acceleration = jnp.sum(
        inputs.action_smoothness_weights
        * (action_delta - inputs.last_action_delta) ** 2
    )

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
    stand_motor_continuous_torque_excess = jnp.asarray(0.0)
    if config.enabled("stand_motor_continuous_torque_excess"):
        if inputs.motor_continuous_torque is None:
            raise ValueError("standing motor torque reward requires continuous ratings")
        if inputs.motor_continuous_torque.shape != inputs.actuator_force.shape:
            raise ValueError("continuous torque ratings must cover all actuator forces")
        excess = jnp.maximum(
            jnp.abs(inputs.actuator_force) / (0.5 * inputs.motor_continuous_torque)
            - 1.0,
            0.0,
        )
        stand_motor_continuous_torque_excess = jnp.where(
            standing_command, jnp.mean(excess), 0.0
        )
    requested_policy_force = jnp.abs(
        inputs.requested_actuator_force[inputs.policy_joint_indices]
    )
    normalized_requested_torque = requested_policy_force / policy_peak_force
    torque_peak = jnp.mean(jnp.maximum(normalized_requested_torque - 1.0, 0.0) ** 2)
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
    normalized_foot_impulse = inputs.foot_normal_impulse / jnp.maximum(
        inputs.robot_weight * inputs.control_timestep, 1.0e-6
    )
    normalized_foot_peak_force = inputs.foot_peak_normal_force / jnp.maximum(
        inputs.robot_weight, 1.0e-6
    )
    touchdown_load = jnp.maximum(normalized_foot_impulse, normalized_foot_peak_force)
    touchdown_force_excess = jnp.maximum(
        touchdown_load - config.gait.touchdown_force_soft_limit_body_weights,
        0.0,
    )
    touchdown_speed_ratio = (
        jnp.maximum(-inputs.touchdown_foot_linear_velocity[:, 2], 0.0)
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
    bounded_foot_heading_error = foot_heading_excess / (
        foot_heading_excess + config.standing.foot_heading_scale
    )
    stand_foot_heading = jnp.where(
        standing_double_support & ~inputs.done,
        jnp.mean(bounded_foot_heading_error),
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
    hip_deviation_commands = inputs.command[
        jnp.asarray(config.limits_pose.hip_deviation_command_indices, dtype=jnp.int32)
    ]
    hip_deviation *= standing_command | jnp.any(
        jnp.abs(hip_deviation_commands)
        > config.limits_pose.hip_deviation_command_threshold
    )
    knee_deviation = jnp.where(
        standing_command | config.limits_pose.knee_deviation_apply_while_moving,
        jnp.sum(jnp.abs(inputs.joint_pos_rel[inputs.knee_joint_indices])),
        0.0,
    )
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

    desired_contact = inputs.desired_foot_contact.astype(jnp.float32)
    gait_contact = gait_active * jnp.mean(jnp.abs(actual_contact - desired_contact))
    foot_speed = jnp.linalg.norm(inputs.foot_linear_velocity[:, :2], axis=1)
    foot_slip = jnp.sum(actual_contact * foot_speed**2)
    foot_heading = jnp.mean(inputs.foot_heading_error**2)
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
    if config.gait.phase_shaped_clearance:
        if inputs.swing_progress is None or inputs.foot_clearance_samples is None:
            raise ValueError(
                "phase-shaped clearance requires sole samples and swing progress"
            )
        target = (
            config.gait.swing_clearance_target
            * jnp.sin(jnp.pi * inputs.swing_progress) ** 2
        )
        shortfall = jnp.maximum(target[:, None] - inputs.foot_clearance_samples, 0.0)
        gait_swing_clearance = gait_active * jnp.mean(
            desired_swing[:, None]
            * (shortfall / config.gait.swing_clearance_target) ** 2
        )

    alignment_active = (
        standing_command
        if config.standing.contact_independent_alignment
        else standing_double_support
    )
    alignment_gate = alignment_active & ~inputs.done
    stand_foot_foreaft = deadband_huber(
        inputs.foot_foreaft,
        config.standing.position_deadband,
        config.standing.position_scale,
    )
    stand_foot_width = deadband_huber(
        inputs.foot_separation - inputs.nominal_stance_width,
        config.standing.position_deadband,
        config.standing.position_scale,
    )
    stand_foot_foreaft *= alignment_gate
    stand_foot_width *= alignment_gate
    foot_crossing = positive_huber(
        (inputs.minimum_foot_separation - inputs.foot_separation)
        / config.gait.crossing_scale
    )
    foot_stumble = jnp.asarray(0.0)
    if config.enabled("foot_stumble"):
        if inputs.swing_progress is None:
            raise ValueError("foot_stumble requires swing progress")
        grace = config.gait.stumble_grace_fraction
        mid_swing = (inputs.swing_progress > grace) & (
            inputs.swing_progress < 1.0 - grace
        )
        load = jnp.maximum(touchdown_load - config.gait.stumble_load_body_weights, 0.0)
        severity = jnp.tanh(load / config.gait.stumble_load_body_weights)
        motion = jnp.tanh((foot_speed / config.gait.stumble_speed_scale) ** 2)
        # Standing recovery has no scheduled swing; detect loaded sliding instead.
        foot_stumble = jnp.mean(
            severity
            * jnp.where(
                inputs.gait_active,
                desired_swing * mid_swing * (0.5 + 0.5 * motion),
                motion,
            )
        )

    values = {
        "track_lin_vel_xy": track_lin_vel_xy,
        "moving_lin_vel_xy_error": moving_lin_vel_xy_error,
        "track_ang_vel_z": track_ang_vel_z,
        "moving_ang_vel_z_error": moving_ang_vel_z_error,
        "healthy_survival": (~inputs.done).astype(jnp.float32),
        "lin_vel_z": lin_vel_z,
        "ang_vel_xy": ang_vel_xy,
        "flat_orientation": flat_orientation,
        "base_height": base_height,
        "action_rate": action_rate,
        "action_acceleration": action_acceleration,
        "action_limit": action_limit,
        "torque_l1": torque_l1,
        "stand_motor_continuous_torque_excess": stand_motor_continuous_torque_excess,
        "torque_peak": torque_peak,
        "joint_force_peak": joint_force_peak,
        "touchdown_impact": touchdown_impact,
        "joint_pose": joint_pose,
        "stand_still": stand_still,
        "stand_foot_motion": stand_foot_motion,
        "stand_foot_heading": stand_foot_heading,
        "stand_contact": stand_contact,
        "stand_upright": stand_upright,
        "stand_foot_foreaft": stand_foot_foreaft,
        "stand_foot_width": stand_foot_width,
        "foot_crossing": foot_crossing,
        "foot_heading": foot_heading,
        "foot_slip": foot_slip,
        "foot_stumble": foot_stumble,
        "hip_deviation": hip_deviation,
        "knee_deviation": knee_deviation,
        "joint_limit": joint_limit,
        "gait_contact": gait_contact,
        "gait_stance_slip": gait_stance_slip,
        "gait_swing_clearance": gait_swing_clearance,
        "termination": inputs.done.astype(jnp.float32),
    }
    timestep = jnp.asarray(inputs.control_timestep, dtype=jnp.float32)
    for name in values:
        if name.startswith("stand_"):
            values[name] *= inputs.standing_reward_blend
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
    if type(config.standing.contact_independent_alignment) is not bool:
        raise TypeError("standing.contact_independent_alignment must be a boolean")
    pose_weights = config.limits_pose.joint_pose_weights
    if type(pose_weights) is not tuple:
        raise TypeError("limits_pose.joint_pose_weights must be a tuple")
    names = set()
    for entry in pose_weights:
        if type(entry) is not tuple or len(entry) != 2:
            raise ValueError("joint_pose_weights entries must be (joint_name, weight)")
        name, weight = entry
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("joint_pose_weights requires unique nonempty joint names")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise TypeError("joint_pose_weights values must be numbers")
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("joint_pose_weights values must be finite and nonnegative")
        names.add(name)
    for name, value in {
        "standing.position_scale": config.standing.position_scale,
        "gait.crossing_scale": config.gait.crossing_scale,
        "gait.stumble_load_body_weights": config.gait.stumble_load_body_weights,
        "gait.stumble_speed_scale": config.gait.stumble_speed_scale,
    }.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    for value in (config.standing.position_deadband, config.gait.sole_clearance_margin):
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                "sole margin and position deadband must be finite and nonnegative"
            )
    if not 0 <= config.gait.stumble_grace_fraction < 0.5:
        raise ValueError("gait.stumble_grace_fraction must be in [0, 0.5)")
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
            "tracking.lin_vel_xy_error_delta",
            config.tracking.lin_vel_xy_error_delta,
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
            "tracking.ang_vel_z_error_delta",
            config.tracking.ang_vel_z_error_delta,
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
            "standing.foot_heading_scale",
            config.standing.foot_heading_scale,
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
    hip_command_indices = config.limits_pose.hip_deviation_command_indices
    if type(hip_command_indices) is not tuple:
        raise TypeError("limits_pose.hip_deviation_command_indices must be a tuple")
    if not hip_command_indices:
        raise ValueError("limits_pose.hip_deviation_command_indices must not be empty")
    if any(
        type(index) is not int or index not in {0, 1, 2}
        for index in hip_command_indices
    ):
        raise ValueError(
            "limits_pose.hip_deviation_command_indices must contain command "
            "indices 0, 1, or 2"
        )
    if len(set(hip_command_indices)) != len(hip_command_indices):
        raise ValueError(
            "limits_pose.hip_deviation_command_indices must not contain duplicates"
        )
    if type(config.limits_pose.knee_deviation_apply_while_moving) is not bool:
        raise TypeError(
            "limits_pose.knee_deviation_apply_while_moving must be a boolean"
        )
    nonnegative = {
        "gait.stance_slip_angular_speed_scale": (
            config.gait.stance_slip_angular_speed_scale
        ),
        "standing.foot_angular_weight": config.standing.foot_angular_weight,
        "standing.foot_heading_deadband": config.standing.foot_heading_deadband,
        "smoothness.ankle_action_weight": config.smoothness.ankle_action_weight,
        "smoothness.waist_action_weight": config.smoothness.waist_action_weight,
        "limits_pose.hip_deviation_command_threshold": (
            config.limits_pose.hip_deviation_command_threshold
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
