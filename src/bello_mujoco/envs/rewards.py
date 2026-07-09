"""Pure-JAX reward terms for Bello velocity-tracking locomotion."""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class CommandConfig:
    """Uniform velocity command sampler.

    Commands are `[lin_vel_x, lin_vel_y, yaw_vel]`.
    """

    enabled: bool = True
    lin_vel_x: tuple[float, float] = (-0.1, 0.1)
    lin_vel_y: tuple[float, float] = (-0.1, 0.1)
    ang_vel_yaw: tuple[float, float] = (-0.1, 0.1)
    standing_probability: float = 0.02
    resample_seconds: float = 10.0

    def __post_init__(self) -> None:
        validate_range("lin_vel_x", self.lin_vel_x)
        validate_range("lin_vel_y", self.lin_vel_y)
        validate_range("ang_vel_yaw", self.ang_vel_yaw)
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

    def sample(self, rng: jax.Array, shape: tuple[int, ...] = ()) -> jax.Array:
        if not self.enabled:
            return jnp.zeros(shape + (3,), dtype=jnp.float32)
        low = jnp.asarray(
            [self.lin_vel_x[0], self.lin_vel_y[0], self.ang_vel_yaw[0]],
            dtype=jnp.float32,
        )
        high = jnp.asarray(
            [self.lin_vel_x[1], self.lin_vel_y[1], self.ang_vel_yaw[1]],
            dtype=jnp.float32,
        )
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
    link: str = "bello_root"


@dataclass(frozen=True)
class TrackLinVelXYRewardConfig(LinkRewardConfig):
    sigma: float = 0.25


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
class ActionRateRewardConfig(WeightedRewardConfig):
    pass


@dataclass(frozen=True)
class EnergyRewardConfig(WeightedRewardConfig):
    pass


@dataclass(frozen=True)
class TorqueRewardConfig(WeightedRewardConfig):
    pass


@dataclass(frozen=True)
class JointLimitRewardConfig(WeightedRewardConfig):
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
    pass


@dataclass(frozen=True)
class FootClearanceRewardConfig(WeightedRewardConfig):
    target: float = 0.08


@dataclass(frozen=True)
class FootDistanceRewardConfig(WeightedRewardConfig):
    pass


@dataclass(frozen=True)
class GaitContactRewardConfig(WeightedRewardConfig):
    pass


@dataclass(frozen=True)
class FeetAirTimeRewardConfig(WeightedRewardConfig):
    min_air_time: float = 0.20
    max_air_time: float = 0.60
    command_speed_threshold: float = 0.05


@dataclass(frozen=True)
class UndesiredContactRewardConfig(WeightedRewardConfig):
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
    action_rate: ActionRateRewardConfig = ActionRateRewardConfig(weight=-0.05)
    energy: EnergyRewardConfig = EnergyRewardConfig(weight=-2.0e-5)
    torque: TorqueRewardConfig = TorqueRewardConfig(enabled=False, weight=-1.0e-5)
    joint_limit: JointLimitRewardConfig = JointLimitRewardConfig(weight=-5.0)
    hold_joint_deviation: HoldJointDeviationRewardConfig = (
        HoldJointDeviationRewardConfig(weight=-0.1)
    )
    policy_joint_deviation: PolicyJointDeviationRewardConfig = (
        PolicyJointDeviationRewardConfig(enabled=False, weight=-0.1)
    )
    foot_contact: FootContactRewardConfig = FootContactRewardConfig(weight=0.05)
    foot_slip: FootSlipRewardConfig = FootSlipRewardConfig(weight=-0.2)
    foot_clearance: FootClearanceRewardConfig = FootClearanceRewardConfig(weight=-2.0)
    foot_distance: FootDistanceRewardConfig = FootDistanceRewardConfig(
        enabled=False, weight=1.0
    )
    gait_contact: GaitContactRewardConfig = GaitContactRewardConfig(weight=-0.1)
    feet_air_time: FeetAirTimeRewardConfig = FeetAirTimeRewardConfig(weight=0.2)
    undesired_contact: UndesiredContactRewardConfig = UndesiredContactRewardConfig(
        weight=-1.0
    )
    termination: TerminationRewardConfig = TerminationRewardConfig(weight=-1.0)

    def __post_init__(self) -> None:
        validate_reward_config(self)

    @property
    def uses_foot_contact_metadata(self) -> bool:
        return (
            self.foot_contact.enabled
            or self.foot_slip.enabled
            or self.foot_clearance.enabled
            or self.gait_contact.enabled
            or self.feet_air_time.enabled
        )

    @property
    def uses_foot_metadata(self) -> bool:
        return self.uses_foot_contact_metadata or self.foot_distance.enabled

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
        ]
        return tuple(
            sorted({term.link for term in terms if term.enabled and term.link})
        )


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
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Return weighted scalar reward and weighted component terms."""
    lin_vel_error = jnp.sum((lin_vel_xy - command[:2]) ** 2)
    yaw_vel_error = (yaw_vel_z - command[2]) ** 2

    track_lin_vel_xy = jnp.exp(-lin_vel_error / config.track_lin_vel_xy.sigma)
    track_ang_vel_z = jnp.exp(-yaw_vel_error / config.track_ang_vel_z.sigma)
    alive = jnp.asarray(1.0, dtype=jnp.float32)

    lin_vel_z = lin_vel_z**2
    ang_vel_xy = jnp.sum(ang_vel_xy**2)
    flat_orientation = jnp.sum(projected_gravity[:2] ** 2)
    base_height_error = (base_height - config.base_height.target) ** 2
    joint_vel_l2 = jnp.mean(joint_vel**2)
    action_rate_l2 = jnp.mean((action - last_action) ** 2)
    energy = jnp.mean(jnp.abs(actuator_force * joint_vel))
    torque_l2 = jnp.mean(actuator_force**2)

    lower_violation = jnp.maximum(joint_limit_low - joint_pos, 0.0)
    upper_violation = jnp.maximum(joint_pos - joint_limit_high, 0.0)
    joint_limit = jnp.mean(lower_violation**2 + upper_violation**2)
    hold_joint_deviation = (
        jnp.mean(jnp.abs(joint_pos_rel[hold_joint_indices]))
        if hold_joint_indices.size
        else jnp.asarray(0.0, dtype=jnp.float32)
    )
    policy_joint_deviation = (
        jnp.mean(jnp.abs(joint_pos_rel[policy_joint_indices]))
        if policy_joint_indices.size
        else jnp.asarray(0.0, dtype=jnp.float32)
    )
    foot_contact_rate = safe_mean(foot_contact)
    foot_slip = safe_mean(foot_contact * foot_lateral_speed_sq)
    swing = 1.0 - foot_contact
    foot_clearance = safe_mean(
        swing * jnp.maximum(config.foot_clearance.target - foot_height, 0.0) ** 2
    )
    foot_distance = jnp.asarray(foot_distance, dtype=jnp.float32)
    gait_contact = safe_mean(
        (foot_contact - gait_contact_mask.astype(jnp.float32)) ** 2
    )
    first_contact = foot_contact.astype(bool) & ~last_foot_contact.astype(bool)
    completed_air_time = foot_air_time + control_timestep
    max_bonus = config.feet_air_time.max_air_time - config.feet_air_time.min_air_time
    air_time_bonus = jnp.clip(
        completed_air_time - config.feet_air_time.min_air_time,
        0.0,
        max_bonus,
    )
    command_speed = jnp.linalg.norm(command)
    feet_air_time = safe_mean(first_contact.astype(jnp.float32) * air_time_bonus)
    feet_air_time = jnp.where(
        command_speed > config.feet_air_time.command_speed_threshold,
        feet_air_time,
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    termination = done.astype(jnp.float32)

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
        "reward/action_rate": weighted_term(
            config.action_rate.enabled, config.action_rate.weight, action_rate_l2
        ),
        "reward/energy": weighted_term(
            config.energy.enabled, config.energy.weight, energy
        ),
        "reward/torque": weighted_term(
            config.torque.enabled, config.torque.weight, torque_l2
        ),
        "reward/joint_limit": weighted_term(
            config.joint_limit.enabled, config.joint_limit.weight, joint_limit
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
        "reward/foot_distance": weighted_term(
            config.foot_distance.enabled,
            config.foot_distance.weight,
            foot_distance,
        ),
        "reward/gait_contact": weighted_term(
            config.gait_contact.enabled,
            config.gait_contact.weight,
            gait_contact,
        ),
        "reward/feet_air_time": weighted_term(
            config.feet_air_time.enabled,
            config.feet_air_time.weight,
            feet_air_time,
        ),
        "reward/undesired_contact": (
            weighted_term(
                config.undesired_contact.enabled,
                config.undesired_contact.weight,
                undesired_contact,
            )
        ),
        "reward/termination": weighted_term(
            config.termination.enabled,
            config.termination.weight,
            termination,
        ),
    }
    timestep = jnp.asarray(control_timestep, dtype=jnp.float32)
    event_terms = frozenset(("reward/feet_air_time", "reward/termination"))
    terms = {
        name: value if name in event_terms else value * timestep
        for name, value in raw_terms.items()
    }
    reward = sum(terms.values(), jnp.asarray(0.0, dtype=jnp.float32))
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
    if config.feet_air_time.enabled:
        if (
            not math.isfinite(config.feet_air_time.min_air_time)
            or config.feet_air_time.min_air_time < 0
        ):
            raise ValueError(
                "feet_air_time.min_air_time must be nonnegative and finite, got "
                f"{config.feet_air_time.min_air_time}"
            )
        if (
            not math.isfinite(config.feet_air_time.max_air_time)
            or config.feet_air_time.max_air_time <= config.feet_air_time.min_air_time
        ):
            raise ValueError(
                "feet_air_time.max_air_time must be finite and greater than "
                "min_air_time, got "
                f"{config.feet_air_time.max_air_time}"
            )
        if (
            not math.isfinite(config.feet_air_time.command_speed_threshold)
            or config.feet_air_time.command_speed_threshold < 0
        ):
            raise ValueError(
                "feet_air_time.command_speed_threshold must be nonnegative and "
                f"finite, got {config.feet_air_time.command_speed_threshold}"
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
        "action_rate": config.action_rate,
        "energy": config.energy,
        "torque": config.torque,
        "joint_limit": config.joint_limit,
        "hold_joint_deviation": config.hold_joint_deviation,
        "policy_joint_deviation": config.policy_joint_deviation,
        "foot_contact": config.foot_contact,
        "foot_slip": config.foot_slip,
        "foot_clearance": config.foot_clearance,
        "foot_distance": config.foot_distance,
        "gait_contact": config.gait_contact,
        "feet_air_time": config.feet_air_time,
        "undesired_contact": config.undesired_contact,
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
