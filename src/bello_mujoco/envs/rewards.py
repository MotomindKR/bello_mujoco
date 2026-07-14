"""Pure-JAX reward terms for Bello velocity-tracking locomotion."""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class CommandCurriculumConfig:
    """Unitree-style command range expansion for velocity commands."""

    enabled: bool = False
    limit_lin_vel_x: tuple[float, float] = (-0.5, 1.0)
    limit_lin_vel_y: tuple[float, float] = (-0.3, 0.3)
    limit_ang_vel_yaw: tuple[float, float] = (-0.2, 0.2)
    expand_ang_vel_yaw: bool = False
    increment: float = 0.1
    success_threshold: float = 0.8


@dataclass(frozen=True)
class CommandConfig:
    """Uniform velocity command sampler.

    Commands are `[lin_vel_x, lin_vel_y, yaw_vel]`.
    """

    enabled: bool = True
    lin_vel_x: tuple[float, float] = (-0.1, 0.1)
    lin_vel_y: tuple[float, float] = (-0.1, 0.1)
    ang_vel_yaw: tuple[float, float] = (-0.1, 0.1)
    standing_probability: float = 0.10
    resample_seconds: float = 10.0
    curriculum: CommandCurriculumConfig = CommandCurriculumConfig()

    def __post_init__(self) -> None:
        validate_range("lin_vel_x", self.lin_vel_x)
        validate_range("lin_vel_y", self.lin_vel_y)
        validate_range("ang_vel_yaw", self.ang_vel_yaw)
        validate_range("curriculum.limit_lin_vel_x", self.curriculum.limit_lin_vel_x)
        validate_range("curriculum.limit_lin_vel_y", self.curriculum.limit_lin_vel_y)
        validate_range(
            "curriculum.limit_ang_vel_yaw", self.curriculum.limit_ang_vel_yaw
        )
        if (
            not math.isfinite(self.standing_probability)
            or not 0.0 <= self.standing_probability <= 1.0
        ):
            raise ValueError(
                "standing_probability must be in [0, 1], got "
                f"{self.standing_probability}"
            )
        if not math.isfinite(self.resample_seconds) or self.resample_seconds <= 0:
            raise ValueError(
                f"resample_seconds must be positive, got {self.resample_seconds}"
            )
        if (
            not math.isfinite(self.curriculum.increment)
            or self.curriculum.increment <= 0
        ):
            raise ValueError(
                "curriculum.increment must be positive and finite, got "
                f"{self.curriculum.increment}"
            )
        if (
            not math.isfinite(self.curriculum.success_threshold)
            or self.curriculum.success_threshold <= 0
        ):
            raise ValueError(
                "curriculum.success_threshold must be positive and finite, got "
                f"{self.curriculum.success_threshold}"
            )
        if self.curriculum.enabled:
            validate_subrange(
                "lin_vel_x", self.lin_vel_x, self.curriculum.limit_lin_vel_x
            )
            validate_subrange(
                "lin_vel_y", self.lin_vel_y, self.curriculum.limit_lin_vel_y
            )
            validate_subrange(
                "ang_vel_yaw", self.ang_vel_yaw, self.curriculum.limit_ang_vel_yaw
            )

    def range_array(self) -> jax.Array:
        return jnp.asarray(
            [self.lin_vel_x, self.lin_vel_y, self.ang_vel_yaw],
            dtype=jnp.float32,
        )

    def sample(
        self,
        rng: jax.Array,
        shape: tuple[int, ...] = (),
        ranges: jax.Array | None = None,
    ) -> jax.Array:
        if not self.enabled:
            return jnp.zeros(shape + (3,), dtype=jnp.float32)
        ranges = self.range_array() if ranges is None else jnp.asarray(ranges)
        low = ranges[:, 0].astype(jnp.float32)
        high = ranges[:, 1].astype(jnp.float32)
        command_key, stand_key = jax.random.split(rng)
        command = jax.random.uniform(command_key, shape + (3,), minval=low, maxval=high)
        standing = jax.random.bernoulli(stand_key, self.standing_probability, shape)
        return jnp.where(standing[..., None], jnp.zeros_like(command), command)


@dataclass(frozen=True)
class WeightedRewardConfig:
    enabled: bool = True
    weight: float = 1.0


@dataclass(frozen=True)
class LinkRewardConfig(WeightedRewardConfig):
    link: str | None = None


@dataclass(frozen=True)
class TrackLinVelXYRewardConfig(LinkRewardConfig):
    sigma: float = 0.25
    frame_link: str | None = None


@dataclass(frozen=True)
class TrackAngVelZRewardConfig(LinkRewardConfig):
    sigma: float = 0.25


@dataclass(frozen=True)
class AliveRewardConfig(WeightedRewardConfig):
    pass


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
    target: float = 0.86


@dataclass(frozen=True)
class JointVelRewardConfig(WeightedRewardConfig):
    pass


@dataclass(frozen=True)
class JointAccRewardConfig(WeightedRewardConfig):
    pass


@dataclass(frozen=True)
class ActionRateRewardConfig(WeightedRewardConfig):
    pass


@dataclass(frozen=True)
class EnergyRewardConfig(WeightedRewardConfig):
    pass


@dataclass(frozen=True)
class TorqueRewardConfig(WeightedRewardConfig):
    pass


@dataclass(frozen=True)
class TorqueL1RewardConfig(WeightedRewardConfig):
    pass


@dataclass(frozen=True)
class JointPoseRewardConfig(WeightedRewardConfig):
    pass


@dataclass(frozen=True)
class HipDeviationRewardConfig(WeightedRewardConfig):
    lateral_command_index: int = 0
    lateral_command_threshold: float = 0.1


@dataclass(frozen=True)
class KneeDeviationRewardConfig(WeightedRewardConfig):
    pass


@dataclass(frozen=True)
class JointLimitRewardConfig(WeightedRewardConfig):
    pass


@dataclass(frozen=True)
class InvalidActionRewardConfig(WeightedRewardConfig):
    pass


@dataclass(frozen=True)
class HoldJointDeviationRewardConfig(WeightedRewardConfig):
    pass


@dataclass(frozen=True)
class PolicyJointDeviationRewardConfig(WeightedRewardConfig):
    pass


@dataclass(frozen=True)
class FootContactRewardConfig(WeightedRewardConfig):
    pass


@dataclass(frozen=True)
class FootSlipRewardConfig(WeightedRewardConfig):
    angular_speed_scale: float = 0.05


@dataclass(frozen=True)
class FootClearanceRewardConfig(WeightedRewardConfig):
    target: float = 0.10
    std: float = 0.05
    tanh_mult: float = 2.0
    command_speed_threshold: float = 0.05


@dataclass(frozen=True)
class SwingFootVelocityRewardConfig(LinkRewardConfig):
    std: float = 0.25
    max_target_speed: float = 2.0
    command_speed_threshold: float = 0.10


@dataclass(frozen=True)
class FootDistanceRewardConfig(WeightedRewardConfig):
    pass


@dataclass(frozen=True)
class StanceWidthRewardConfig(LinkRewardConfig):
    min_width: float = 0.24
    max_width: float = 0.34


@dataclass(frozen=True)
class GaitContactRewardConfig(WeightedRewardConfig):
    command_speed_threshold: float = 0.10


@dataclass(frozen=True)
class ContactTimingRewardConfig(WeightedRewardConfig):
    min_air_time: float = 0.20
    min_contact_time: float = 0.30
    command_speed_threshold: float = 0.10


@dataclass(frozen=True)
class FeetAirTimeRewardConfig(WeightedRewardConfig):
    max_duration: float = 0.40
    command_speed_threshold: float = 0.10


@dataclass(frozen=True)
class TouchdownAirTimeRewardConfig(WeightedRewardConfig):
    min_air_time: float = 0.20
    max_air_time: float = 0.50
    command_speed_threshold: float = 0.10


@dataclass(frozen=True)
class FootPhaseRewardConfig(WeightedRewardConfig):
    sigma: float = 0.01


@dataclass(frozen=True)
class StandingFootVelRewardConfig(WeightedRewardConfig):
    command_speed_threshold: float = 0.05


@dataclass(frozen=True)
class StandingFootAirTimeRewardConfig(WeightedRewardConfig):
    command_speed_threshold: float = 0.05


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
    """Per-term reward config for the Bello velocity-tracking baseline."""

    track_lin_vel_xy: TrackLinVelXYRewardConfig = TrackLinVelXYRewardConfig(weight=1.0)
    track_ang_vel_z: TrackAngVelZRewardConfig = TrackAngVelZRewardConfig(weight=0.5)
    alive: AliveRewardConfig = AliveRewardConfig(weight=0.15)
    lin_vel_z: LinVelZRewardConfig = LinVelZRewardConfig(weight=-2.0)
    ang_vel_xy: AngVelXYRewardConfig = AngVelXYRewardConfig(weight=-0.05)
    flat_orientation: FlatOrientationRewardConfig = FlatOrientationRewardConfig(
        weight=-5.0
    )
    base_height: BaseHeightRewardConfig = BaseHeightRewardConfig(weight=-10.0)
    joint_vel: JointVelRewardConfig = JointVelRewardConfig(weight=-0.001)
    joint_acc: JointAccRewardConfig = JointAccRewardConfig(weight=-2.5e-7)
    action_rate: ActionRateRewardConfig = ActionRateRewardConfig(weight=-0.05)
    energy: EnergyRewardConfig = EnergyRewardConfig(weight=-2.0e-5)
    torque: TorqueRewardConfig = TorqueRewardConfig(enabled=False, weight=-1.0e-5)
    torque_l1: TorqueL1RewardConfig = TorqueL1RewardConfig(
        enabled=False, weight=-2.5e-5
    )
    joint_pose: JointPoseRewardConfig = JointPoseRewardConfig(
        enabled=False, weight=-1.0
    )
    hip_deviation: HipDeviationRewardConfig = HipDeviationRewardConfig(
        enabled=False, weight=-0.25
    )
    knee_deviation: KneeDeviationRewardConfig = KneeDeviationRewardConfig(
        enabled=False, weight=-0.1
    )
    joint_limit: JointLimitRewardConfig = JointLimitRewardConfig(weight=-5.0)
    invalid_action: InvalidActionRewardConfig = InvalidActionRewardConfig(
        enabled=False, weight=-1.0
    )
    hold_joint_deviation: HoldJointDeviationRewardConfig = (
        HoldJointDeviationRewardConfig(weight=-0.1)
    )
    policy_joint_deviation: PolicyJointDeviationRewardConfig = (
        PolicyJointDeviationRewardConfig(enabled=False, weight=-0.1)
    )
    foot_contact: FootContactRewardConfig = FootContactRewardConfig(weight=0.05)
    foot_slip: FootSlipRewardConfig = FootSlipRewardConfig(weight=-0.2)
    foot_clearance: FootClearanceRewardConfig = FootClearanceRewardConfig(weight=1.0)
    swing_foot_velocity: SwingFootVelocityRewardConfig = SwingFootVelocityRewardConfig(
        enabled=False, weight=0.5
    )
    foot_distance: FootDistanceRewardConfig = FootDistanceRewardConfig(
        enabled=False, weight=1.0
    )
    stance_width: StanceWidthRewardConfig = StanceWidthRewardConfig(
        enabled=False, weight=-2.0
    )
    gait_contact: GaitContactRewardConfig = GaitContactRewardConfig(weight=0.5)
    contact_timing: ContactTimingRewardConfig = ContactTimingRewardConfig(
        enabled=False, weight=-0.10
    )
    feet_air_time: FeetAirTimeRewardConfig = FeetAirTimeRewardConfig(
        enabled=False, weight=0.25
    )
    touchdown_air_time: TouchdownAirTimeRewardConfig = TouchdownAirTimeRewardConfig(
        enabled=False, weight=2.0
    )
    foot_phase: FootPhaseRewardConfig = FootPhaseRewardConfig(enabled=False, weight=1.0)
    standing_foot_vel: StandingFootVelRewardConfig = StandingFootVelRewardConfig(
        enabled=False, weight=-1.0
    )
    standing_foot_air_time: StandingFootAirTimeRewardConfig = (
        StandingFootAirTimeRewardConfig(enabled=False, weight=-1.0)
    )
    undesired_contact: UndesiredContactRewardConfig = UndesiredContactRewardConfig(
        weight=-1.0
    )
    self_collision: SelfCollisionRewardConfig = SelfCollisionRewardConfig(weight=-1.0)
    termination: TerminationRewardConfig = TerminationRewardConfig(weight=-1.0)
    clip_negative_total: bool = False
    scale_termination_by_timestep: bool = False

    def __post_init__(self) -> None:
        validate_reward_config(self)

    @property
    def uses_foot_contact_metadata(self) -> bool:
        return (
            self.foot_contact.enabled
            or self.foot_slip.enabled
            or self.foot_clearance.enabled
            or self.gait_contact.enabled
            or self.contact_timing.enabled
            or self.feet_air_time.enabled
            or self.touchdown_air_time.enabled
            or self.standing_foot_vel.enabled
            or self.standing_foot_air_time.enabled
        )

    @property
    def uses_foot_metadata(self) -> bool:
        return (
            self.uses_foot_contact_metadata
            or self.swing_foot_velocity.enabled
            or self.foot_distance.enabled
            or self.stance_width.enabled
            or self.foot_phase.enabled
        )

    @property
    def uses_terrain_contact_metadata(self) -> bool:
        return self.uses_foot_contact_metadata or self.undesired_contact.enabled

    @property
    def enabled_link_names(self) -> tuple[str, ...]:
        terms = [
            self.track_lin_vel_xy,
            self.track_ang_vel_z,
            self.lin_vel_z,
            self.ang_vel_xy,
            self.flat_orientation,
            self.base_height,
            self.swing_foot_velocity,
            self.stance_width,
        ]
        links = {term.link for term in terms if term.enabled and term.link}
        if self.track_lin_vel_xy.enabled and self.track_lin_vel_xy.frame_link:
            links.add(self.track_lin_vel_xy.frame_link)
        return tuple(sorted(links))


def compute_velocity_reward(
    *,
    config: RewardConfig,
    command: jax.Array,
    lin_vel_xy: jax.Array,
    yaw_vel_z: jax.Array,
    lin_vel_z: jax.Array,
    ang_vel_xy: jax.Array,
    projected_gravity: jax.Array,
    base_height: jax.Array,
    joint_pos: jax.Array,
    joint_pos_rel: jax.Array,
    joint_vel: jax.Array,
    last_joint_vel: jax.Array,
    action: jax.Array,
    last_action: jax.Array,
    actuator_force: jax.Array,
    joint_limit_low: jax.Array,
    joint_limit_high: jax.Array,
    hold_joint_indices: jax.Array,
    policy_joint_indices: jax.Array,
    foot_contact: jax.Array,
    foot_height: jax.Array,
    foot_lateral_speed_sq: jax.Array,
    foot_distance: jax.Array,
    gait_contact_mask: jax.Array,
    foot_air_time: jax.Array,
    last_foot_contact: jax.Array,
    undesired_contact: jax.Array,
    control_timestep: float,
    done: jax.Array,
    foot_angular_speed: jax.Array | None = None,
    stance_width: jax.Array | None = None,
    foot_contact_time: jax.Array | None = None,
    requested_action: jax.Array | None = None,
    action_clip_limit: float | None = None,
    self_collision: jax.Array | None = None,
    action_target_default: jax.Array | None = None,
    action_target_scale: jax.Array | None = None,
    action_target_low: jax.Array | None = None,
    action_target_high: jax.Array | None = None,
    foot_relative_lin_vel_xy: jax.Array | None = None,
    swing_velocity_target_scale: float = 1.0,
    foot_phase_height_target: jax.Array | None = None,
    joint_pose_weights: jax.Array | None = None,
    hip_joint_indices: jax.Array | None = None,
    knee_joint_indices: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Return weighted scalar reward and weighted component terms."""
    lin_vel_error = jnp.sum((lin_vel_xy - command[:2]) ** 2)
    yaw_vel_error = (yaw_vel_z - command[2]) ** 2

    track_lin_vel_xy = jnp.exp(-lin_vel_error / config.track_lin_vel_xy.sigma)
    track_ang_vel_z = jnp.exp(-yaw_vel_error / config.track_ang_vel_z.sigma)
    alive = 1.0 - done.astype(jnp.float32)

    lin_vel_z = lin_vel_z**2
    ang_vel_xy = jnp.sum(ang_vel_xy**2)
    flat_orientation = jnp.sum(projected_gravity[:2] ** 2)
    base_height_error = (base_height - config.base_height.target) ** 2
    joint_vel_l2 = jnp.sum(joint_vel**2)
    joint_acc_l2 = jnp.sum(((joint_vel - last_joint_vel) / control_timestep) ** 2)
    action_rate_l2 = jnp.sum((action - last_action) ** 2)
    energy = jnp.sum(jnp.abs(actuator_force * joint_vel))
    torque_l2 = jnp.sum(actuator_force**2)
    torque_l1 = jnp.sum(jnp.abs(actuator_force[policy_joint_indices]))
    pose_weights = (
        jnp.ones_like(joint_pos_rel)
        if joint_pose_weights is None
        else joint_pose_weights
    )
    joint_pose = jnp.sum(joint_pos_rel**2 * pose_weights)
    hip_joint_indices = (
        jnp.asarray([], dtype=jnp.int32)
        if hip_joint_indices is None
        else hip_joint_indices
    )
    hip_deviation = jnp.sum(jnp.abs(joint_pos_rel[hip_joint_indices]))
    lateral_command = command[config.hip_deviation.lateral_command_index]
    hip_deviation *= (
        jnp.abs(lateral_command) > config.hip_deviation.lateral_command_threshold
    )
    knee_joint_indices = (
        jnp.asarray([], dtype=jnp.int32)
        if knee_joint_indices is None
        else knee_joint_indices
    )
    knee_deviation = jnp.sum(jnp.abs(joint_pos_rel[knee_joint_indices]))

    lower_violation = jnp.maximum(joint_limit_low - joint_pos, 0.0)
    upper_violation = jnp.maximum(joint_pos - joint_limit_high, 0.0)
    joint_limit = jnp.sum(lower_violation + upper_violation)
    if config.invalid_action.enabled:
        if (
            action_target_default is None
            or action_target_scale is None
            or action_target_low is None
            or action_target_high is None
        ):
            raise ValueError(
                "invalid_action reward requires action target default, scale, "
                "low, and high arrays"
            )
        requested_action = action if requested_action is None else requested_action
        action_target = action_target_default + action_target_scale * requested_action
        action_low_violation = jnp.maximum(action_target_low - action_target, 0.0)
        action_high_violation = jnp.maximum(action_target - action_target_high, 0.0)
        action_target_violation = action_low_violation + action_high_violation
        action_unit_violation = action_target_violation / jnp.maximum(
            jnp.abs(action_target_scale), 1.0e-6
        )
        action_clip_violation = (
            jnp.zeros_like(requested_action)
            if action_clip_limit is None
            else jnp.maximum(jnp.abs(requested_action) - action_clip_limit, 0.0)
        )
        invalid_action = jnp.sum(action_unit_violation**2 + action_clip_violation**2)
    else:
        invalid_action = jnp.asarray(0.0, dtype=jnp.float32)
    hold_joint_deviation = (
        jnp.sum(jnp.abs(joint_pos_rel[hold_joint_indices]))
        if hold_joint_indices.size
        else jnp.asarray(0.0, dtype=jnp.float32)
    )
    policy_joint_deviation = (
        jnp.sum(jnp.abs(joint_pos_rel[policy_joint_indices]))
        if policy_joint_indices.size
        else jnp.asarray(0.0, dtype=jnp.float32)
    )
    foot_contact_rate = safe_mean(foot_contact)
    foot_lateral_speed = jnp.sqrt(jnp.maximum(foot_lateral_speed_sq, 0.0))
    if foot_angular_speed is None:
        foot_angular_speed = jnp.zeros_like(foot_lateral_speed)
    foot_slip_speed = (
        foot_lateral_speed + config.foot_slip.angular_speed_scale * foot_angular_speed
    )
    foot_slip = jnp.sum(foot_contact * foot_slip_speed)
    foot_clearance_error = (foot_height - config.foot_clearance.target) ** 2
    moving_foot_gate = jnp.tanh(config.foot_clearance.tanh_mult * foot_lateral_speed)
    swing_gate = (1.0 - foot_contact) * moving_foot_gate
    clearance_score = jnp.exp(-foot_clearance_error / config.foot_clearance.std)
    active_swing = jnp.sum(swing_gate)
    foot_clearance = jnp.where(
        active_swing > 0.0,
        jnp.sum(swing_gate * clearance_score) / jnp.maximum(active_swing, 1.0e-6),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    if config.swing_foot_velocity.enabled:
        if foot_relative_lin_vel_xy is None:
            raise ValueError(
                "swing_foot_velocity reward requires relative foot XY velocities"
            )
        target_velocity = command[:2] * swing_velocity_target_scale
        target_speed = jnp.linalg.norm(target_velocity)
        target_velocity = target_velocity * jnp.minimum(
            1.0,
            config.swing_foot_velocity.max_target_speed
            / jnp.maximum(target_speed, 1.0e-6),
        )
        swing_velocity_error = jnp.sum(
            (foot_relative_lin_vel_xy - target_velocity) ** 2, axis=1
        )
        swing_velocity_score = jnp.exp(
            -swing_velocity_error / config.swing_foot_velocity.std
        )
        scheduled_swing = (~gait_contact_mask.astype(bool)).astype(jnp.float32)
        swing_foot_velocity = jnp.sum(
            scheduled_swing * swing_velocity_score
        ) / jnp.maximum(jnp.sum(scheduled_swing), 1.0)
        swing_foot_velocity = jnp.where(
            jnp.linalg.norm(command[:2])
            > config.swing_foot_velocity.command_speed_threshold,
            swing_foot_velocity,
            jnp.asarray(0.0, dtype=jnp.float32),
        )
    else:
        swing_foot_velocity = jnp.asarray(0.0, dtype=jnp.float32)
    foot_distance = jnp.asarray(foot_distance, dtype=jnp.float32)
    stance_width = (
        jnp.asarray(0.0, dtype=jnp.float32)
        if stance_width is None
        else jnp.asarray(stance_width, dtype=jnp.float32)
    )
    stance_width_error = (
        jnp.maximum(config.stance_width.min_width - stance_width, 0.0) ** 2
        + jnp.maximum(stance_width - config.stance_width.max_width, 0.0) ** 2
    )
    command_speed = jnp.linalg.norm(command)
    foot_clearance = jnp.where(
        command_speed > config.foot_clearance.command_speed_threshold,
        foot_clearance,
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    gait_match = foot_contact.astype(bool) == gait_contact_mask.astype(bool)
    gait_contact = jnp.sum(gait_match.astype(jnp.float32))
    gait_contact = jnp.where(
        command_speed > config.gait_contact.command_speed_threshold,
        gait_contact,
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    if config.contact_timing.enabled:
        if foot_contact_time is None:
            raise ValueError("contact_timing reward requires foot contact durations")
        contact = foot_contact.astype(bool)
        previous_contact = last_foot_contact.astype(bool)
        touchdown = contact & ~previous_contact
        liftoff = ~contact & previous_contact
        short_air = (
            jnp.maximum(config.contact_timing.min_air_time - foot_air_time, 0.0)
            / config.contact_timing.min_air_time
        )
        short_contact = (
            jnp.maximum(config.contact_timing.min_contact_time - foot_contact_time, 0.0)
            / config.contact_timing.min_contact_time
        )
        contact_timing = jnp.sum(
            touchdown.astype(jnp.float32) * short_air**2
            + liftoff.astype(jnp.float32) * short_contact**2
        )
        contact_timing = jnp.where(
            jnp.linalg.norm(command[:2])
            > config.contact_timing.command_speed_threshold,
            contact_timing,
            jnp.asarray(0.0, dtype=jnp.float32),
        )
    else:
        contact_timing = jnp.asarray(0.0, dtype=jnp.float32)
    if config.feet_air_time.enabled:
        if foot_contact_time is None:
            raise ValueError("feet_air_time reward requires foot contact durations")
        contact = foot_contact.astype(bool)
        current_air_time = jnp.where(
            contact,
            jnp.zeros_like(foot_air_time),
            foot_air_time + control_timestep,
        )
        current_contact_time = jnp.where(
            contact,
            foot_contact_time + control_timestep,
            jnp.zeros_like(foot_contact_time),
        )
        in_mode_time = jnp.where(contact, current_contact_time, current_air_time)
        single_stance = jnp.sum(contact.astype(jnp.int32)) == 1
        feet_air_time = jnp.min(
            jnp.where(single_stance, in_mode_time, jnp.zeros_like(in_mode_time))
        )
        feet_air_time = jnp.minimum(feet_air_time, config.feet_air_time.max_duration)
        feet_air_time = jnp.where(
            jnp.linalg.norm(command[:2]) > config.feet_air_time.command_speed_threshold,
            feet_air_time,
            jnp.asarray(0.0, dtype=jnp.float32),
        )
    else:
        feet_air_time = jnp.asarray(0.0, dtype=jnp.float32)
    if config.touchdown_air_time.enabled:
        contact = foot_contact.astype(bool)
        contact_filtered = contact | last_foot_contact.astype(bool)
        first_contact = (foot_air_time > 0.0) & contact_filtered
        touchdown_air_time = (
            foot_air_time - config.touchdown_air_time.min_air_time
        ) * first_contact.astype(jnp.float32)
        touchdown_air_time = jnp.sum(
            jnp.minimum(
                touchdown_air_time,
                config.touchdown_air_time.max_air_time
                - config.touchdown_air_time.min_air_time,
            )
        )
        touchdown_air_time = jnp.where(
            jnp.linalg.norm(command[:2])
            > config.touchdown_air_time.command_speed_threshold,
            touchdown_air_time,
            jnp.asarray(0.0, dtype=jnp.float32),
        )
    else:
        touchdown_air_time = jnp.asarray(0.0, dtype=jnp.float32)
    if config.foot_phase.enabled:
        if foot_phase_height_target is None:
            raise ValueError("foot_phase reward requires foot height targets")
        foot_phase = jnp.exp(
            -jnp.sum((foot_height - foot_phase_height_target) ** 2)
            / config.foot_phase.sigma
        )
    else:
        foot_phase = jnp.asarray(0.0, dtype=jnp.float32)
    standing_foot_vel = safe_mean(foot_lateral_speed_sq)
    standing_foot_vel = jnp.where(
        command_speed <= config.standing_foot_vel.command_speed_threshold,
        standing_foot_vel,
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    standing_foot_air_time = safe_mean(foot_air_time)
    standing_foot_air_time = jnp.where(
        command_speed <= config.standing_foot_air_time.command_speed_threshold,
        standing_foot_air_time,
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    termination = done.astype(jnp.float32)
    self_collision = (
        jnp.asarray(0.0, dtype=jnp.float32)
        if self_collision is None
        else jnp.asarray(self_collision, dtype=jnp.float32)
    )

    raw_terms = {
        "reward/track_lin_vel_xy": weighted_term(
            config.track_lin_vel_xy.enabled,
            config.track_lin_vel_xy.weight,
            track_lin_vel_xy,
        ),
        "reward/track_ang_vel_z": weighted_term(
            config.track_ang_vel_z.enabled,
            config.track_ang_vel_z.weight,
            track_ang_vel_z,
        ),
        "reward/alive": weighted_term(config.alive.enabled, config.alive.weight, alive),
        "reward/lin_vel_z": weighted_term(
            config.lin_vel_z.enabled, config.lin_vel_z.weight, lin_vel_z
        ),
        "reward/ang_vel_xy": weighted_term(
            config.ang_vel_xy.enabled, config.ang_vel_xy.weight, ang_vel_xy
        ),
        "reward/flat_orientation": weighted_term(
            config.flat_orientation.enabled,
            config.flat_orientation.weight,
            flat_orientation,
        ),
        "reward/base_height": weighted_term(
            config.base_height.enabled,
            config.base_height.weight,
            base_height_error,
        ),
        "reward/joint_vel": weighted_term(
            config.joint_vel.enabled, config.joint_vel.weight, joint_vel_l2
        ),
        "reward/joint_acc": weighted_term(
            config.joint_acc.enabled, config.joint_acc.weight, joint_acc_l2
        ),
        "reward/action_rate": weighted_term(
            config.action_rate.enabled, config.action_rate.weight, action_rate_l2
        ),
        "reward/energy": weighted_term(
            config.energy.enabled, config.energy.weight, energy
        ),
        "reward/torque": weighted_term(
            config.torque.enabled, config.torque.weight, torque_l2
        ),
        "reward/torque_l1": weighted_term(
            config.torque_l1.enabled, config.torque_l1.weight, torque_l1
        ),
        "reward/joint_pose": weighted_term(
            config.joint_pose.enabled, config.joint_pose.weight, joint_pose
        ),
        "reward/hip_deviation": weighted_term(
            config.hip_deviation.enabled,
            config.hip_deviation.weight,
            hip_deviation,
        ),
        "reward/knee_deviation": weighted_term(
            config.knee_deviation.enabled,
            config.knee_deviation.weight,
            knee_deviation,
        ),
        "reward/joint_limit": weighted_term(
            config.joint_limit.enabled, config.joint_limit.weight, joint_limit
        ),
        "reward/invalid_action": weighted_term(
            config.invalid_action.enabled,
            config.invalid_action.weight,
            invalid_action,
        ),
        "reward/hold_joint_deviation": (
            weighted_term(
                config.hold_joint_deviation.enabled,
                config.hold_joint_deviation.weight,
                hold_joint_deviation,
            )
        ),
        "reward/policy_joint_deviation": (
            weighted_term(
                config.policy_joint_deviation.enabled,
                config.policy_joint_deviation.weight,
                policy_joint_deviation,
            )
        ),
        "reward/foot_contact": weighted_term(
            config.foot_contact.enabled,
            config.foot_contact.weight,
            foot_contact_rate,
        ),
        "reward/foot_slip": weighted_term(
            config.foot_slip.enabled, config.foot_slip.weight, foot_slip
        ),
        "reward/foot_clearance": weighted_term(
            config.foot_clearance.enabled,
            config.foot_clearance.weight,
            foot_clearance,
        ),
        "reward/swing_foot_velocity": weighted_term(
            config.swing_foot_velocity.enabled,
            config.swing_foot_velocity.weight,
            swing_foot_velocity,
        ),
        "reward/foot_distance": weighted_term(
            config.foot_distance.enabled,
            config.foot_distance.weight,
            foot_distance,
        ),
        "reward/stance_width": weighted_term(
            config.stance_width.enabled,
            config.stance_width.weight,
            stance_width_error,
        ),
        "reward/gait_contact": weighted_term(
            config.gait_contact.enabled,
            config.gait_contact.weight,
            gait_contact,
        ),
        "reward/contact_timing": weighted_term(
            config.contact_timing.enabled,
            config.contact_timing.weight,
            contact_timing,
        ),
        "reward/feet_air_time": weighted_term(
            config.feet_air_time.enabled,
            config.feet_air_time.weight,
            feet_air_time,
        ),
        "reward/touchdown_air_time": weighted_term(
            config.touchdown_air_time.enabled,
            config.touchdown_air_time.weight,
            touchdown_air_time,
        ),
        "reward/foot_phase": weighted_term(
            config.foot_phase.enabled, config.foot_phase.weight, foot_phase
        ),
        "reward/standing_foot_vel": weighted_term(
            config.standing_foot_vel.enabled,
            config.standing_foot_vel.weight,
            standing_foot_vel,
        ),
        "reward/standing_foot_air_time": weighted_term(
            config.standing_foot_air_time.enabled,
            config.standing_foot_air_time.weight,
            standing_foot_air_time,
        ),
        "reward/undesired_contact": (
            weighted_term(
                config.undesired_contact.enabled,
                config.undesired_contact.weight,
                undesired_contact,
            )
        ),
        "reward/self_collision": weighted_term(
            config.self_collision.enabled,
            config.self_collision.weight,
            self_collision,
        ),
        "reward/termination": weighted_term(
            config.termination.enabled,
            config.termination.weight,
            termination,
        ),
    }
    timestep = jnp.asarray(control_timestep, dtype=jnp.float32)
    event_terms = {"reward/contact_timing"}
    if not config.scale_termination_by_timestep:
        event_terms.add("reward/termination")
    terms = {
        name: value if name in event_terms else value * timestep
        for name, value in raw_terms.items()
    }
    reward = sum(terms.values(), jnp.asarray(0.0, dtype=jnp.float32))
    if config.clip_negative_total:
        reward = jnp.maximum(reward, 0.0)
    return reward, terms


def safe_mean(values: jax.Array) -> jax.Array:
    if values.size == 0:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.mean(values)


def weighted_term(enabled: bool, weight: float, value: jax.Array) -> jax.Array:
    if not enabled:
        return jnp.zeros_like(value, dtype=jnp.float32)
    return weight * value


def validate_reward_config(config: RewardConfig) -> None:
    if config.track_lin_vel_xy.enabled and (
        not math.isfinite(config.track_lin_vel_xy.sigma)
        or config.track_lin_vel_xy.sigma <= 0
    ):
        raise ValueError(
            "track_lin_vel_xy.sigma must be positive when enabled, got "
            f"{config.track_lin_vel_xy.sigma}"
        )
    if config.track_ang_vel_z.enabled and (
        not math.isfinite(config.track_ang_vel_z.sigma)
        or config.track_ang_vel_z.sigma <= 0
    ):
        raise ValueError(
            "track_ang_vel_z.sigma must be positive when enabled, got "
            f"{config.track_ang_vel_z.sigma}"
        )
    for name, term in reward_terms(config).items():
        if not math.isfinite(term.weight):
            raise ValueError(f"{name}.weight must be finite, got {term.weight}")
    if config.foot_slip.enabled and (
        not math.isfinite(config.foot_slip.angular_speed_scale)
        or config.foot_slip.angular_speed_scale < 0
    ):
        raise ValueError(
            "foot_slip.angular_speed_scale must be nonnegative and finite, got "
            f"{config.foot_slip.angular_speed_scale}"
        )
    if config.foot_clearance.enabled:
        if (
            not math.isfinite(config.foot_clearance.target)
            or config.foot_clearance.target < 0
        ):
            raise ValueError(
                "foot_clearance.target must be nonnegative and finite, got "
                f"{config.foot_clearance.target}"
            )
        if (
            not math.isfinite(config.foot_clearance.std)
            or config.foot_clearance.std <= 0
        ):
            raise ValueError(
                "foot_clearance.std must be positive when enabled, got "
                f"{config.foot_clearance.std}"
            )
        if (
            not math.isfinite(config.foot_clearance.tanh_mult)
            or config.foot_clearance.tanh_mult < 0
        ):
            raise ValueError(
                "foot_clearance.tanh_mult must be nonnegative and finite, got "
                f"{config.foot_clearance.tanh_mult}"
            )
        if (
            not math.isfinite(config.foot_clearance.command_speed_threshold)
            or config.foot_clearance.command_speed_threshold < 0
        ):
            raise ValueError(
                "foot_clearance.command_speed_threshold must be nonnegative and "
                f"finite, got {config.foot_clearance.command_speed_threshold}"
            )
    if config.swing_foot_velocity.enabled:
        if (
            not math.isfinite(config.swing_foot_velocity.std)
            or config.swing_foot_velocity.std <= 0
        ):
            raise ValueError(
                "swing_foot_velocity.std must be positive when enabled, got "
                f"{config.swing_foot_velocity.std}"
            )
        if (
            not math.isfinite(config.swing_foot_velocity.max_target_speed)
            or config.swing_foot_velocity.max_target_speed <= 0
        ):
            raise ValueError(
                "swing_foot_velocity.max_target_speed must be positive and finite, "
                f"got {config.swing_foot_velocity.max_target_speed}"
            )
        if (
            not math.isfinite(config.swing_foot_velocity.command_speed_threshold)
            or config.swing_foot_velocity.command_speed_threshold < 0
        ):
            raise ValueError(
                "swing_foot_velocity.command_speed_threshold must be nonnegative "
                "and finite, got "
                f"{config.swing_foot_velocity.command_speed_threshold}"
            )
    if config.gait_contact.enabled and (
        not math.isfinite(config.gait_contact.command_speed_threshold)
        or config.gait_contact.command_speed_threshold < 0
    ):
        raise ValueError(
            "gait_contact.command_speed_threshold must be nonnegative and "
            f"finite, got {config.gait_contact.command_speed_threshold}"
        )
    if config.contact_timing.enabled:
        if (
            not math.isfinite(config.contact_timing.min_air_time)
            or config.contact_timing.min_air_time <= 0
        ):
            raise ValueError(
                "contact_timing.min_air_time must be positive and finite, got "
                f"{config.contact_timing.min_air_time}"
            )
        if (
            not math.isfinite(config.contact_timing.min_contact_time)
            or config.contact_timing.min_contact_time <= 0
        ):
            raise ValueError(
                "contact_timing.min_contact_time must be positive and finite, got "
                f"{config.contact_timing.min_contact_time}"
            )
        if (
            not math.isfinite(config.contact_timing.command_speed_threshold)
            or config.contact_timing.command_speed_threshold < 0
        ):
            raise ValueError(
                "contact_timing.command_speed_threshold must be nonnegative and "
                f"finite, got {config.contact_timing.command_speed_threshold}"
            )
    if config.feet_air_time.enabled:
        if (
            not math.isfinite(config.feet_air_time.max_duration)
            or config.feet_air_time.max_duration <= 0
        ):
            raise ValueError(
                "feet_air_time.max_duration must be positive and finite, got "
                f"{config.feet_air_time.max_duration}"
            )
        if (
            not math.isfinite(config.feet_air_time.command_speed_threshold)
            or config.feet_air_time.command_speed_threshold < 0
        ):
            raise ValueError(
                "feet_air_time.command_speed_threshold must be nonnegative and "
                f"finite, got {config.feet_air_time.command_speed_threshold}"
            )
    if config.touchdown_air_time.enabled:
        if (
            not math.isfinite(config.touchdown_air_time.min_air_time)
            or config.touchdown_air_time.min_air_time < 0
            or not math.isfinite(config.touchdown_air_time.max_air_time)
            or config.touchdown_air_time.max_air_time
            <= config.touchdown_air_time.min_air_time
        ):
            raise ValueError(
                "touchdown_air_time bounds must be finite with max greater than "
                "min, got "
                f"{config.touchdown_air_time.min_air_time}, "
                f"{config.touchdown_air_time.max_air_time}"
            )
    if config.foot_phase.enabled and (
        not math.isfinite(config.foot_phase.sigma) or config.foot_phase.sigma <= 0
    ):
        raise ValueError(
            "foot_phase.sigma must be positive and finite, got "
            f"{config.foot_phase.sigma}"
        )
    if config.stance_width.enabled:
        if (
            not math.isfinite(config.stance_width.min_width)
            or config.stance_width.min_width < 0
        ):
            raise ValueError(
                "stance_width.min_width must be nonnegative and finite, got "
                f"{config.stance_width.min_width}"
            )
        if (
            not math.isfinite(config.stance_width.max_width)
            or config.stance_width.max_width <= config.stance_width.min_width
        ):
            raise ValueError(
                "stance_width.max_width must be finite and greater than min_width, "
                f"got {config.stance_width.max_width} <= "
                f"{config.stance_width.min_width}"
            )
    if config.standing_foot_vel.enabled and (
        not math.isfinite(config.standing_foot_vel.command_speed_threshold)
        or config.standing_foot_vel.command_speed_threshold < 0
    ):
        raise ValueError(
            "standing_foot_vel.command_speed_threshold must be nonnegative and "
            f"finite, got {config.standing_foot_vel.command_speed_threshold}"
        )
    if config.standing_foot_air_time.enabled and (
        not math.isfinite(config.standing_foot_air_time.command_speed_threshold)
        or config.standing_foot_air_time.command_speed_threshold < 0
    ):
        raise ValueError(
            "standing_foot_air_time.command_speed_threshold must be nonnegative and "
            f"finite, got {config.standing_foot_air_time.command_speed_threshold}"
        )


def reward_terms(config: RewardConfig):
    return {
        "track_lin_vel_xy": config.track_lin_vel_xy,
        "track_ang_vel_z": config.track_ang_vel_z,
        "alive": config.alive,
        "lin_vel_z": config.lin_vel_z,
        "ang_vel_xy": config.ang_vel_xy,
        "flat_orientation": config.flat_orientation,
        "base_height": config.base_height,
        "joint_vel": config.joint_vel,
        "joint_acc": config.joint_acc,
        "action_rate": config.action_rate,
        "energy": config.energy,
        "torque": config.torque,
        "torque_l1": config.torque_l1,
        "joint_pose": config.joint_pose,
        "hip_deviation": config.hip_deviation,
        "knee_deviation": config.knee_deviation,
        "joint_limit": config.joint_limit,
        "invalid_action": config.invalid_action,
        "hold_joint_deviation": config.hold_joint_deviation,
        "policy_joint_deviation": config.policy_joint_deviation,
        "foot_contact": config.foot_contact,
        "foot_slip": config.foot_slip,
        "foot_clearance": config.foot_clearance,
        "swing_foot_velocity": config.swing_foot_velocity,
        "foot_distance": config.foot_distance,
        "stance_width": config.stance_width,
        "gait_contact": config.gait_contact,
        "contact_timing": config.contact_timing,
        "feet_air_time": config.feet_air_time,
        "touchdown_air_time": config.touchdown_air_time,
        "foot_phase": config.foot_phase,
        "standing_foot_vel": config.standing_foot_vel,
        "standing_foot_air_time": config.standing_foot_air_time,
        "undesired_contact": config.undesired_contact,
        "self_collision": config.self_collision,
        "termination": config.termination,
    }


def validate_range(name: str, bounds: tuple[float, float]) -> None:
    if len(bounds) != 2:
        raise ValueError(f"{name} must have exactly two bounds, got {bounds}")
    low, high = bounds
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError(f"{name} bounds must be finite, got {bounds}")
    if low > high:
        raise ValueError(f"{name} lower bound must be <= upper bound, got {bounds}")


def validate_subrange(
    name: str,
    bounds: tuple[float, float],
    limits: tuple[float, float],
) -> None:
    low, high = bounds
    limit_low, limit_high = limits
    if low < limit_low or high > limit_high:
        raise ValueError(
            f"{name} bounds {bounds} must be inside curriculum limits {limits}"
        )
