"""Pure-JAX reward terms for Bello velocity-tracking locomotion."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class CommandConfig:
    """Uniform velocity command sampler.

    Commands are `[lin_vel_x, lin_vel_y, yaw_vel]`. The limit ranges are kept for
    later curriculum code; the v1 environment samples from the initial ranges.
    """

    enabled: bool = True
    lin_vel_x: tuple[float, float] = (-0.1, 0.1)
    lin_vel_y: tuple[float, float] = (-0.1, 0.1)
    ang_vel_yaw: tuple[float, float] = (-0.1, 0.1)
    limit_lin_vel_x: tuple[float, float] = (-0.5, 1.0)
    limit_lin_vel_y: tuple[float, float] = (-0.3, 0.3)
    limit_ang_vel_yaw: tuple[float, float] = (-0.2, 0.2)
    standing_probability: float = 0.02
    resample_seconds: float = 10.0

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
    target: float = 0.91


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
class JointLimitRewardConfig(WeightedRewardConfig):
    pass


@dataclass(frozen=True)
class HoldJointDeviationRewardConfig(WeightedRewardConfig):
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
    joint_limit: JointLimitRewardConfig = JointLimitRewardConfig(weight=-5.0)
    hold_joint_deviation: HoldJointDeviationRewardConfig = (
        HoldJointDeviationRewardConfig(weight=-0.1)
    )
    foot_contact: FootContactRewardConfig = FootContactRewardConfig(weight=0.05)
    foot_slip: FootSlipRewardConfig = FootSlipRewardConfig(weight=-0.2)
    foot_clearance: FootClearanceRewardConfig = FootClearanceRewardConfig(weight=-2.0)
    undesired_contact: UndesiredContactRewardConfig = UndesiredContactRewardConfig(
        weight=-1.0
    )
    termination: TerminationRewardConfig = TerminationRewardConfig(weight=-1.0)

    @property
    def uses_foot_metadata(self) -> bool:
        return (
            self.foot_contact.enabled
            or self.foot_slip.enabled
            or self.foot_clearance.enabled
        )

    @property
    def uses_terrain_contact_metadata(self) -> bool:
        return self.uses_foot_metadata or self.undesired_contact.enabled

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
    foot_contact: jax.Array,
    foot_height: jax.Array,
    foot_lateral_speed_sq: jax.Array,
    undesired_contact: jax.Array,
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

    lower_violation = jnp.maximum(joint_limit_low - joint_pos, 0.0)
    upper_violation = jnp.maximum(joint_pos - joint_limit_high, 0.0)
    joint_limit = jnp.mean(lower_violation**2 + upper_violation**2)
    hold_joint_deviation = (
        jnp.mean(jnp.abs(joint_pos_rel[hold_joint_indices]))
        if hold_joint_indices.size
        else jnp.asarray(0.0, dtype=jnp.float32)
    )
    foot_contact_rate = safe_mean(foot_contact)
    foot_slip = safe_mean(foot_contact * foot_lateral_speed_sq)
    swing = 1.0 - foot_contact
    foot_clearance = safe_mean(
        swing * jnp.maximum(config.foot_clearance.target - foot_height, 0.0) ** 2
    )
    termination = done.astype(jnp.float32)

    terms = {
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
