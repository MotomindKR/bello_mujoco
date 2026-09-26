"""Typed locomotion configuration and initialization constraints."""

from __future__ import annotations
from dataclasses import dataclass, fields
import math
from bello_mujoco.envs.rewards import (
    CommandConfig,
    RewardConfig,
    validate_range,
    validate_reward_config,
)
from bello_mujoco.contracts import (
    BELLO_ACTION_CLIP,
    CONTROL_DECIMATION,
    MAX_EPISODE_STEPS,
    DEFAULT_SPAWN_CLEARANCE,
)
from bello_mujoco.envs.gait import GaitConfig
from bello_mujoco.envs.feedback import (
    ObservationLowPassFilterConfig,
    validate_feedback_filter,
    validate_command_delay,
)
from bello_mujoco.sim import ModelRandomizationConfig, SimulationPhysicsConfig
from bello_mujoco.terrain import FLAT_TERRAIN, TerrainConfig


@dataclass(frozen=True)
class BaseDoneConfig:
    enabled: bool = True


@dataclass(frozen=True)
class BaseHeightDoneConfig(BaseDoneConfig):
    min_height: float = 0.75
    link: str | None = None


@dataclass(frozen=True)
class BadOrientationDoneConfig(BaseDoneConfig):
    limit_angle: float = 1.0
    link: str | None = None


@dataclass(frozen=True)
class DoneConfig:
    """Termination thresholds for Bello episodes."""

    base_height: BaseHeightDoneConfig = BaseHeightDoneConfig()
    bad_orientation: BadOrientationDoneConfig = BadOrientationDoneConfig()
    timeout: BaseDoneConfig = BaseDoneConfig()
    nonfinite: BaseDoneConfig = BaseDoneConfig()

    @property
    def enabled_link_names(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    term.link
                    for term in (self.base_height, self.bad_orientation)
                    if term.enabled and term.link
                }
            )
        )


@dataclass(frozen=True)
class ObservationScaleConfig:
    base_lin_vel: float = 1.0
    base_lin_acc: float = 1.0
    base_ang_vel: float = 1.0
    projected_gravity: float = 1.0
    base_height: float = 1.0
    command: float = 1.0
    gait_phase: float = 1.0
    joint_pos: float = 1.0
    joint_vel: float = 1.0
    actuator_force: float = 1.0
    last_action: float = 1.0


@dataclass(frozen=True)
class ObservationNoiseConfig:
    enabled: bool = False
    base_lin_vel: tuple[float, float] = (0.0, 0.0)
    base_lin_acc: tuple[float, float] = (0.0, 0.0)
    base_ang_vel: tuple[float, float] = (-0.2, 0.2)
    projected_gravity: tuple[float, float] = (-0.05, 0.05)
    base_height: tuple[float, float] = (0.0, 0.0)
    command: tuple[float, float] = (0.0, 0.0)
    gait_phase: tuple[float, float] = (0.0, 0.0)
    joint_pos: tuple[float, float] = (-0.01, 0.01)
    joint_vel: tuple[float, float] = (-1.5, 1.5)
    actuator_force: tuple[float, float] = (0.0, 0.0)
    last_action: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class ObservationLatencyConfig:
    enabled: bool = False
    delay_steps: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class TorsoComCurriculumConfig:
    """Ramp fixed per-world torso COM offsets over policy steps."""

    enabled: bool = False
    ramp_steps: int = 1


@dataclass(frozen=True)
class ObservationConfig:
    """Fixed deployable observation, optionally extended for the critic."""

    privileged: bool = False
    scale: ObservationScaleConfig = ObservationScaleConfig()
    noise: ObservationNoiseConfig = ObservationNoiseConfig()


@dataclass(frozen=True)
class SpawnConfig:
    clearance: float = DEFAULT_SPAWN_CLEARANCE
    root_xy_range: tuple[float, float] = (0.0, 0.0)
    root_yaw_range: tuple[float, float] = (0.0, 0.0)
    joint_pos_noise: tuple[float, float] = (0.0, 0.0)
    joint_pos_scale_range: tuple[float, float] = (1.0, 1.0)
    joint_vel_range: tuple[float, float] = (0.0, 0.0)
    root_vel_range: tuple[float, float] = (0.0, 0.0)
    stagger_initial_step_count: bool = False


@dataclass(frozen=True)
class ContactConfig:
    foot_contact_on_seconds: float = 0.0
    foot_contact_off_seconds: float = 0.0
    foot_contact_force_threshold: float = 1.0
    undesired_contact_force_threshold: float = 1.0
    self_collision_force_threshold: float = 1.0


@dataclass(frozen=True)
class RandomPushConfig:
    enabled: bool = False
    interval_seconds: tuple[float, float] = (3.0, 7.0)
    lin_vel_x: tuple[float, float] = (-0.5, 0.5)
    lin_vel_y: tuple[float, float] = (-0.5, 0.5)
    curriculum_steps: int = 0
    initial_scale: float = 1.0


@dataclass(frozen=True)
class NumericalSafetyConfig:
    """Broad physical bounds used to reject finite solver explosions."""

    max_abs_position: float = 100.0
    max_abs_velocity: float = 1_000.0
    max_abs_acceleration: float = 1_000_000.0
    max_abs_actuator_force: float = 100_000.0
    max_abs_constraint_force: float = 1_000_000.0
    max_abs_reward_term: float = 1_000.0


@dataclass(frozen=True)
class BelloEnvConfig:
    """Runtime environment settings that affect stepping and model loading."""

    decimation: int = CONTROL_DECIMATION
    max_episode_steps: int = MAX_EPISODE_STEPS
    action_clip: float = BELLO_ACTION_CLIP
    root_link: str | None = None
    terrain: TerrainConfig = FLAT_TERRAIN
    simulation_physics: SimulationPhysicsConfig = SimulationPhysicsConfig()
    model_randomization: ModelRandomizationConfig = ModelRandomizationConfig()
    spawn: SpawnConfig = SpawnConfig()
    contact: ContactConfig = ContactConfig()
    random_push: RandomPushConfig = RandomPushConfig()
    numerical_safety: NumericalSafetyConfig = NumericalSafetyConfig()
    observation: ObservationConfig = ObservationConfig()
    observation_low_pass_filter: ObservationLowPassFilterConfig = (
        ObservationLowPassFilterConfig()
    )
    torso_com_curriculum: TorsoComCurriculumConfig = TorsoComCurriculumConfig()
    observation_latency: ObservationLatencyConfig = ObservationLatencyConfig()
    command_delay_seconds: float = 0.0
    gait: GaitConfig = GaitConfig()
    critic_observation: ObservationConfig = ObservationConfig(privileged=True)


def validate_locomotion_config(
    config: BelloEnvConfig,
    reward: RewardConfig,
    command: CommandConfig,
    done: DoneConfig,
    *,
    physics_dt: float,
) -> None:
    control_timestep = config.decimation * physics_dt
    if config.decimation <= 0:
        raise ValueError(f"decimation must be positive, got {config.decimation}")
    validate_command_delay(config.command_delay_seconds, physics_dt, config.decimation)
    maximum_speed = math.hypot(
        max(map(abs, command.lin_vel_x)),
        max(map(abs, command.lin_vel_y)),
    )
    maximum_frequency = (
        0.5
        * (
            config.gait.step_frequency_base
            + config.gait.step_frequency_speed_gain * maximum_speed
        )
        if config.gait.speed_dependent
        else config.gait.frequency_hz
    )
    gait_phase_advance = maximum_frequency * control_timestep
    if gait_phase_advance >= 0.5:
        raise ValueError(
            "gait.frequency_hz must advance the gait by less than half a cycle "
            f"per control step, got {gait_phase_advance} cycles"
        )
    if config.max_episode_steps <= 0:
        raise ValueError(
            f"max_episode_steps must be positive, got {config.max_episode_steps}"
        )
    if not math.isfinite(config.action_clip) or config.action_clip <= 0:
        raise ValueError(
            f"action_clip must be a positive finite value, got {config.action_clip}"
        )
    if config.root_link is not None and (
        not config.root_link or config.root_link != config.root_link.strip()
    ):
        raise ValueError(
            "root_link must be null or a non-empty body name without "
            "surrounding whitespace"
        )
    if not math.isfinite(config.spawn.clearance) or config.spawn.clearance < 0.0:
        raise ValueError(
            "spawn.clearance must be a finite nonnegative value, got "
            f"{config.spawn.clearance}"
        )
    for name in (
        "root_xy_range",
        "root_yaw_range",
        "joint_pos_noise",
        "joint_pos_scale_range",
        "joint_vel_range",
        "root_vel_range",
    ):
        validate_range(f"spawn.{name}", getattr(config.spawn, name))
    if config.observation.privileged:
        raise ValueError(
            "actor observation must remain deployable; privileged=true is only "
            "valid for critic_observation"
        )
    _validate_observation("observation", config.observation)
    validate_feedback_filter(config.observation_low_pass_filter)
    _validate_torso_curriculum(
        config.torso_com_curriculum, config.model_randomization.enabled
    )
    _validate_observation_latency(config.observation_latency)
    _validate_observation("critic_observation", config.critic_observation)
    if config.critic_observation.noise.enabled:
        raise ValueError(
            "critic_observation.noise.enabled must be false; the privileged "
            "critic observation is intentionally clean"
        )
    for field in fields(config.contact):
        value = getattr(config.contact, field.name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"contact.{field.name} must be finite and nonnegative, got {value}"
            )
    validate_range("random_push.interval_seconds", config.random_push.interval_seconds)
    low, high = config.random_push.interval_seconds
    if low < control_timestep or math.ceil(low / control_timestep) > math.floor(
        high / control_timestep
    ):
        raise ValueError(
            "random_push.interval_seconds must contain at least one control step "
            "and start at or above the control timestep"
        )
    validate_range("random_push.lin_vel_x", config.random_push.lin_vel_x)
    validate_range("random_push.lin_vel_y", config.random_push.lin_vel_y)
    push = config.random_push
    if type(push.curriculum_steps) is not int or push.curriculum_steps < 0:
        raise ValueError("random_push.curriculum_steps must be a nonnegative integer")
    if not math.isfinite(push.initial_scale) or not 0 < push.initial_scale <= 1:
        raise ValueError("random_push.initial_scale must be in (0, 1]")
    if push.curriculum_steps and not push.enabled:
        raise ValueError("random_push curriculum requires enabled pushes")
    if not push.curriculum_steps and push.initial_scale != 1.0:
        raise ValueError("random_push.initial_scale requires a curriculum")
    for name, value in config.numerical_safety.__dict__.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"numerical_safety.{name} must be positive and finite, got {value}"
            )
    validate_reward_config(reward)
    if command.torso.variable:
        if not command.enabled or reward.weights.track_torso_orientation <= 0:
            raise ValueError(
                "variable torso commands require commands and positive torso tracking"
            )
        if reward.enabled("flat_orientation") or reward.enabled("stand_upright"):
            raise ValueError(
                "torso commands conflict with fixed upright rewards; disable those terms"
            )
    if done.base_height.enabled and not math.isfinite(done.base_height.min_height):
        raise ValueError(
            "done_config.base_height.min_height must be finite, got "
            f"{done.base_height.min_height}"
        )
    if done.bad_orientation.enabled and (
        not math.isfinite(done.bad_orientation.limit_angle)
        or done.bad_orientation.limit_angle < 0
    ):
        raise ValueError(
            "done_config.bad_orientation.limit_angle must be finite and "
            "nonnegative, got "
            f"{done.bad_orientation.limit_angle}"
        )


def _validate_observation(config_name: str, observation: ObservationConfig) -> None:
    for name, value in observation.scale.__dict__.items():
        if not math.isfinite(value):
            raise ValueError(f"{config_name}.scale.{name} must be finite, got {value}")
    for name, bounds in observation.noise.__dict__.items():
        if name == "enabled":
            continue
        validate_range(f"{config_name}.noise.{name}", bounds)


def _validate_observation_latency(config: ObservationLatencyConfig) -> None:
    delay_steps = config.delay_steps
    if len(delay_steps) != 2 or any(
        type(value) is not int or value < 0 for value in delay_steps
    ):
        raise ValueError(
            "observation_latency.delay_steps must contain two nonnegative integers"
        )
    if delay_steps[0] > delay_steps[1]:
        raise ValueError(
            "observation_latency.delay_steps lower bound exceeds upper bound"
        )


def _validate_torso_curriculum(
    config: TorsoComCurriculumConfig, randomization_enabled: bool
) -> None:
    if type(config.ramp_steps) is not int or config.ramp_steps <= 0:
        raise ValueError(
            "torso_com_curriculum.ramp_steps must be a positive integer, got "
            f"{config.ramp_steps}"
        )
    if config.enabled and not randomization_enabled:
        raise ValueError(
            "torso_com_curriculum.enabled requires model_randomization.enabled"
        )
