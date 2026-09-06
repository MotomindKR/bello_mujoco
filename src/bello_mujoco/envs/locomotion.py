"""Bello MuJoCo/MJX locomotion environment scaffold."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
import math
from typing import Literal

import chex
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from bello_mujoco.actuators import ActuatorConfig
from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig, PreparedAsset
from bello_mujoco.contracts import (
    ActionLayout,
    BELLO_ACTION_CLIP,
    CONTROL_DECIMATION,
    MAX_EPISODE_STEPS,
    DEFAULT_SPAWN_CLEARANCE,
    ObservationLayout,
)
from bello_mujoco.envs.rewards import (
    CommandConfig,
    RewardConfig,
    RewardInputs,
    command_is_moving,
    command_is_standing,
    compute_locomotion_reward,
    validate_range,
    validate_reward_config,
)
from bello_mujoco.sim import (
    BelloModel,
    ModelRandomizationConfig,
    SimulationPhysicsConfig,
    WarpConfig,
    load_bello_model,
    load_prepared_bello_model,
)
from bello_mujoco.terrain import (
    FLAT_TERRAIN,
    TerrainConfig,
    terrain_surface_height_jax,
)


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
    gait_phase: jax.Array
    standing_transition_pending: jax.Array
    command_resample_step_count: jax.Array
    step_count: jax.Array
    episode_return: jax.Array
    episode_track_lin_vel_xy: jax.Array
    episode_length: jax.Array
    last_episode_return: jax.Array
    last_episode_track_lin_vel_xy: jax.Array
    last_episode_length: jax.Array


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
class TimeoutDoneConfig(BaseDoneConfig):
    pass


@dataclass(frozen=True)
class NonfiniteDoneConfig(BaseDoneConfig):
    pass


@dataclass(frozen=True)
class DoneConfig:
    """Termination thresholds for Bello episodes."""

    base_height: BaseHeightDoneConfig = BaseHeightDoneConfig()
    bad_orientation: BadOrientationDoneConfig = BadOrientationDoneConfig()
    timeout: TimeoutDoneConfig = TimeoutDoneConfig()
    nonfinite: NonfiniteDoneConfig = NonfiniteDoneConfig()

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
class ObservationLowPassFilterConfig:
    """First-order filtering for hardware-derived actor feedback."""

    enabled: bool = False
    cutoff_hz: float = 10.0
    noise_before_filter: bool = False
    residual_noise_scale: float = 0.0


@dataclass(frozen=True)
class TorsoComCurriculumConfig:
    """Ramp fixed per-world torso COM offsets over policy steps."""

    enabled: bool = False
    ramp_steps: int = 1


@dataclass(frozen=True)
class GaitConfig:
    """Fixed alternating gait clock shared by training, evaluation, and hardware."""

    frequency_hz: float = 0.9
    stance_fraction: float = 0.60

    def __post_init__(self) -> None:
        if not math.isfinite(self.frequency_hz) or self.frequency_hz <= 0.0:
            raise ValueError("gait.frequency_hz must be positive and finite")
        if not math.isfinite(self.stance_fraction) or not (
            0.5 <= self.stance_fraction < 1.0
        ):
            raise ValueError("gait.stance_fraction must be in [0.5, 1.0)")


def advance_gait_phase(
    phase: jax.Array,
    command: jax.Array,
    command_config: CommandConfig,
    gait_config: GaitConfig,
    control_timestep: float,
) -> jax.Array:
    """Advance the normalized gait clock, resetting it for standing commands."""

    next_phase = jnp.mod(
        phase
        + jnp.asarray(gait_config.frequency_hz * control_timestep, dtype=jnp.float32),
        1.0,
    )
    return jnp.where(command_is_moving(command, command_config), next_phase, 0.0)


def gait_phase_features(
    phase: jax.Array,
    command: jax.Array,
    command_config: CommandConfig,
) -> jax.Array:
    """Return sin/cos phase plus an explicit active bit; standing is all zero."""

    active = command_is_moving(command, command_config)
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
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Resolve a command request without interrupting an active gait step.

    Standing requests remain pending until the next half-cycle phase boundary.
    Moving requests replace the effective command immediately.
    """

    command = jnp.asarray(command, dtype=jnp.float32)
    requested_command = jnp.asarray(requested_command, dtype=jnp.float32)
    previous_phase = jnp.asarray(previous_phase, dtype=jnp.float32)
    next_phase = jnp.asarray(next_phase, dtype=jnp.float32)
    pending = jnp.asarray(standing_transition_pending, dtype=bool)
    requested_standing = command_is_standing(requested_command, command_config)
    command_standing = command_is_standing(command, command_config)

    if not command_config.standing_transitions_enabled:
        reset_phase = requested_standing | command_standing
        return (
            jnp.where(
                requested_standing[..., None],
                jnp.zeros_like(requested_command),
                requested_command,
            ),
            jnp.where(reset_phase, 0.0, next_phase),
            jnp.zeros_like(pending),
        )

    crossed_half_cycle = (previous_phase < 0.5) & (next_phase >= 0.5)
    crossed_cycle = next_phase < previous_phase
    reached_next_pattern = pending & (crossed_half_cycle | crossed_cycle)
    finish_standing_transition = (
        requested_standing & ~command_standing & reached_next_pattern
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
    effective_phase = jnp.where(reset_phase, 0.0, next_phase)
    return effective_command, effective_phase, keep_moving_until_boundary


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
    interval_seconds: float = 5.0
    lin_vel_x: tuple[float, float] = (-0.5, 0.5)
    lin_vel_y: tuple[float, float] = (-0.5, 0.5)


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
    gait: GaitConfig = GaitConfig()
    critic_observation: ObservationConfig = ObservationConfig(privileged=True)


@dataclass(frozen=True)
class BelloLocomotionEnv:
    """Bello sim scaffold with actuator-order action and observation.

    Actions select policy-controlled logical joint targets within their soft
    limits. Held actuators stay at their configured default targets.
    """

    model: BelloModel
    config: BelloEnvConfig = BelloEnvConfig()
    reward_config: RewardConfig = RewardConfig()
    command_config: CommandConfig = CommandConfig()
    done_config: DoneConfig = DoneConfig()

    def __post_init__(self) -> None:
        self._validate_config()
        self._validate_model_metadata()

    @property
    def task(self) -> Literal["locomotion"]:
        return "locomotion"

    @property
    def root_link_name(self) -> str:
        return self.config.root_link or self.model.robot.root_body_name

    def _validate_config(self) -> None:
        if self.decimation <= 0:
            raise ValueError(f"decimation must be positive, got {self.decimation}")
        gait_phase_advance = self.config.gait.frequency_hz * self.control_timestep
        if gait_phase_advance >= 0.5:
            raise ValueError(
                "gait.frequency_hz must advance the gait by less than half a cycle "
                f"per control step, got {gait_phase_advance} cycles"
            )
        if self.max_episode_steps <= 0:
            raise ValueError(
                f"max_episode_steps must be positive, got {self.max_episode_steps}"
            )
        if not math.isfinite(self.config.action_clip) or self.config.action_clip <= 0:
            raise ValueError(
                "action_clip must be a positive finite value, got "
                f"{self.config.action_clip}"
            )
        if self.config.root_link is not None and (
            not self.config.root_link
            or self.config.root_link != self.config.root_link.strip()
        ):
            raise ValueError(
                "root_link must be null or a non-empty body name without "
                "surrounding whitespace"
            )
        if (
            not math.isfinite(self.config.spawn.clearance)
            or self.config.spawn.clearance < 0.0
        ):
            raise ValueError(
                "spawn.clearance must be a finite nonnegative value, got "
                f"{self.config.spawn.clearance}"
            )
        validate_range("spawn.root_xy_range", self.config.spawn.root_xy_range)
        validate_range("spawn.root_yaw_range", self.config.spawn.root_yaw_range)
        validate_range("spawn.joint_pos_noise", self.config.spawn.joint_pos_noise)
        validate_range(
            "spawn.joint_pos_scale_range", self.config.spawn.joint_pos_scale_range
        )
        validate_range("spawn.joint_vel_range", self.config.spawn.joint_vel_range)
        validate_range("spawn.root_vel_range", self.config.spawn.root_vel_range)
        if self.config.observation.privileged:
            raise ValueError(
                "actor observation must remain deployable; privileged=true is only "
                "valid for critic_observation"
            )
        self._validate_observation_config("observation", self.config.observation)
        self._validate_observation_low_pass_filter()
        self._validate_torso_com_curriculum()
        self._validate_observation_latency()
        self._validate_observation_config(
            "critic_observation", self.config.critic_observation
        )
        if self.config.critic_observation.noise.enabled:
            raise ValueError(
                "critic_observation.noise.enabled must be false; the privileged "
                "critic observation is intentionally clean"
            )
        if (
            not math.isfinite(self.config.contact.foot_contact_on_seconds)
            or self.config.contact.foot_contact_on_seconds < 0.0
        ):
            raise ValueError(
                "contact.foot_contact_on_seconds must be finite and nonnegative, got "
                f"{self.config.contact.foot_contact_on_seconds}"
            )
        if (
            not math.isfinite(self.config.contact.foot_contact_off_seconds)
            or self.config.contact.foot_contact_off_seconds < 0.0
        ):
            raise ValueError(
                "contact.foot_contact_off_seconds must be finite and nonnegative, got "
                f"{self.config.contact.foot_contact_off_seconds}"
            )
        if (
            not math.isfinite(self.config.contact.foot_contact_force_threshold)
            or self.config.contact.foot_contact_force_threshold < 0.0
        ):
            raise ValueError(
                "contact.foot_contact_force_threshold must be finite and "
                "nonnegative, got "
                f"{self.config.contact.foot_contact_force_threshold}"
            )
        if (
            not math.isfinite(self.config.contact.undesired_contact_force_threshold)
            or self.config.contact.undesired_contact_force_threshold < 0.0
        ):
            raise ValueError(
                "contact.undesired_contact_force_threshold must be finite and "
                "nonnegative, got "
                f"{self.config.contact.undesired_contact_force_threshold}"
            )
        if (
            not math.isfinite(self.config.contact.self_collision_force_threshold)
            or self.config.contact.self_collision_force_threshold < 0.0
        ):
            raise ValueError(
                "contact.self_collision_force_threshold must be finite and "
                "nonnegative, got "
                f"{self.config.contact.self_collision_force_threshold}"
            )
        if (
            not math.isfinite(self.config.random_push.interval_seconds)
            or self.config.random_push.interval_seconds <= 0.0
        ):
            raise ValueError(
                "random_push.interval_seconds must be positive and finite, got "
                f"{self.config.random_push.interval_seconds}"
            )
        validate_range("random_push.lin_vel_x", self.config.random_push.lin_vel_x)
        validate_range("random_push.lin_vel_y", self.config.random_push.lin_vel_y)
        for name, value in self.config.numerical_safety.__dict__.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"numerical_safety.{name} must be positive and finite, got {value}"
                )
        validate_reward_config(self.reward_config)
        if self.done_config.base_height.enabled and not math.isfinite(
            self.done_config.base_height.min_height
        ):
            raise ValueError(
                "done_config.base_height.min_height must be finite, got "
                f"{self.done_config.base_height.min_height}"
            )
        if self.done_config.bad_orientation.enabled and (
            not math.isfinite(self.done_config.bad_orientation.limit_angle)
            or self.done_config.bad_orientation.limit_angle < 0
        ):
            raise ValueError(
                "done_config.bad_orientation.limit_angle must be finite and "
                "nonnegative, got "
                f"{self.done_config.bad_orientation.limit_angle}"
            )

    def _validate_observation_config(
        self, config_name: str, observation: ObservationConfig
    ) -> None:
        for name, value in observation.scale.__dict__.items():
            if not math.isfinite(value):
                raise ValueError(
                    f"{config_name}.scale.{name} must be finite, got {value}"
                )
        for name, bounds in observation.noise.__dict__.items():
            if name == "enabled":
                continue
            validate_range(f"{config_name}.noise.{name}", bounds)

    def _validate_observation_latency(self) -> None:
        delay_steps = self.config.observation_latency.delay_steps
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

    def _validate_observation_low_pass_filter(self) -> None:
        config = self.config.observation_low_pass_filter
        cutoff_hz = config.cutoff_hz
        if not math.isfinite(cutoff_hz) or cutoff_hz <= 0.0:
            raise ValueError(
                "observation_low_pass_filter.cutoff_hz must be positive and "
                f"finite, got {cutoff_hz}"
            )
        if (
            not math.isfinite(config.residual_noise_scale)
            or config.residual_noise_scale < 0.0
            or config.residual_noise_scale > 1.0
        ):
            raise ValueError(
                "observation_low_pass_filter.residual_noise_scale must be finite "
                f"and in [0, 1], got {config.residual_noise_scale}"
            )
        if config.noise_before_filter and not config.enabled:
            raise ValueError(
                "observation_low_pass_filter.noise_before_filter requires enabled=true"
            )

    def _validate_torso_com_curriculum(self) -> None:
        config = self.config.torso_com_curriculum
        if type(config.ramp_steps) is not int or config.ramp_steps <= 0:
            raise ValueError(
                "torso_com_curriculum.ramp_steps must be a positive integer, got "
                f"{config.ramp_steps}"
            )
        if config.enabled and not self.config.model_randomization.enabled:
            raise ValueError(
                "torso_com_curriculum.enabled requires model_randomization.enabled"
            )

    @classmethod
    def load(
        cls,
        config: AssetPrepConfig = AssetPrepConfig(),
        *,
        warp: WarpConfig = WarpConfig(),
        batch_size: int = 1,
        env_config: BelloEnvConfig = BelloEnvConfig(),
        reward_config: RewardConfig = RewardConfig(),
        command_config: CommandConfig = CommandConfig(),
        done_config: DoneConfig = DoneConfig(),
        use_viewer_model: bool = False,
        load_mjx: bool = True,
    ) -> "BelloLocomotionEnv":
        return cls(
            load_bello_model(
                config,
                warp=warp,
                batch_size=batch_size,
                terrain=env_config.terrain,
                simulation_physics=env_config.simulation_physics,
                model_randomization=env_config.model_randomization,
                spawn_clearance=env_config.spawn.clearance,
                use_viewer_model=use_viewer_model,
                load_mjx=load_mjx,
            ),
            config=env_config,
            reward_config=reward_config,
            command_config=command_config,
            done_config=done_config,
        )

    @classmethod
    def load_prepared(
        cls,
        prepared: PreparedAsset,
        *,
        warp: WarpConfig = WarpConfig(),
        batch_size: int = 1,
        env_config: BelloEnvConfig = BelloEnvConfig(),
        actuator_config: ActuatorConfig | None = None,
        imu_body_name: str | None = None,
        reward_config: RewardConfig = RewardConfig(),
        command_config: CommandConfig = CommandConfig(),
        done_config: DoneConfig = DoneConfig(),
        use_viewer_model: bool = False,
        load_mjx: bool = True,
    ) -> "BelloLocomotionEnv":
        return cls(
            load_prepared_bello_model(
                prepared,
                warp=warp,
                batch_size=batch_size,
                terrain=env_config.terrain,
                simulation_physics=env_config.simulation_physics,
                model_randomization=env_config.model_randomization,
                actuator_config=actuator_config,
                imu_body_name=imu_body_name,
                spawn_clearance=env_config.spawn.clearance,
                use_viewer_model=use_viewer_model,
                load_mjx=load_mjx,
            ),
            config=env_config,
            reward_config=reward_config,
            command_config=command_config,
            done_config=done_config,
        )

    @property
    def decimation(self) -> int:
        return self.config.decimation

    @property
    def max_episode_steps(self) -> int:
        return self.config.max_episode_steps

    @property
    def terrain_config(self) -> TerrainConfig:
        return self.config.terrain

    @property
    def action_clip(self) -> float:
        return self.config.action_clip

    @property
    def ppo_stagger_initial_step_count(self) -> bool:
        return self.config.spawn.stagger_initial_step_count

    @property
    def action_size(self) -> int:
        return int(self.model.policy_ctrl_indices.size)

    @property
    def action_layout(self) -> ActionLayout:
        return ActionLayout(
            tuple(
                self.model.actuator_joint_names[index]
                for index in self.model.policy_ctrl_indices
            )
        )

    @property
    def observation_layout(self) -> ObservationLayout:
        return self._observation_layout(self.config.observation)

    @property
    def critic_observation_layout(self) -> ObservationLayout:
        return self._observation_layout(self.critic_observation_config)

    @property
    def observation_size(self) -> int:
        return self.observation_layout.size

    @property
    def critic_observation_config(self) -> ObservationConfig:
        return self.config.critic_observation

    @property
    def critic_observation_size(self) -> int:
        return self.critic_observation_layout.size

    @property
    def maximum_observation_delay_steps(self) -> int:
        if not self.config.observation_latency.enabled:
            return 0
        return self.config.observation_latency.delay_steps[1]

    @property
    def actor_feedback_size(self) -> int:
        return 9 + 2 * self.action_size

    def initial_task_state(self) -> jax.Array:
        if not self.config.torso_com_curriculum.enabled:
            return jnp.zeros((0,), dtype=jnp.float32)
        return jnp.zeros((1,), dtype=jnp.float32)

    def with_task_state(self, task_state: jax.Array) -> "BelloLocomotionEnv":
        if not self.config.torso_com_curriculum.enabled:
            return self
        if task_state.shape != (1,):
            raise ValueError(
                "torso COM curriculum task state must have shape (1,), got "
                f"{task_state.shape}"
            )
        if self.model.mjx_model is None:
            raise RuntimeError("torso COM curriculum requires an MJX model")
        progress = jnp.clip(
            task_state[0] / self.config.torso_com_curriculum.ramp_steps, 0.0, 1.0
        )
        torso_body_id = mujoco.mj_name2id(
            self.model.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
        )
        offsets = jnp.asarray(self.model.torso_com_offset)
        body_ipos = self.model.mjx_model.body_ipos.at[:, torso_body_id].add(
            (progress - 1.0) * offsets
        )
        model = replace(
            self.model,
            mjx_model=self.model.mjx_model.replace(body_ipos=body_ipos),
        )
        return replace(self, model=model)

    def update_task_state(
        self, task_state: jax.Array, info: dict[str, jax.Array]
    ) -> jax.Array:
        del info
        if not self.config.torso_com_curriculum.enabled:
            return task_state
        return jnp.minimum(
            task_state + 1.0,
            jnp.asarray(
                [self.config.torso_com_curriculum.ramp_steps], dtype=jnp.float32
            ),
        )

    def task_state_metrics(self, task_state: jax.Array) -> dict[str, jax.Array]:
        if not self.config.torso_com_curriculum.enabled:
            return {}
        return {
            "curriculum/torso_com_scale": jnp.clip(
                task_state[0] / self.config.torso_com_curriculum.ramp_steps,
                0.0,
                1.0,
            )
        }

    def _observation_layout(self, observation: ObservationConfig) -> ObservationLayout:
        components: list[tuple[str, int]] = [
            ("imu_linear_acceleration", 3),
            ("imu_angular_velocity", 3),
            ("projected_gravity", 3),
        ]
        if self.command_config.enabled:
            components.append(("command", 3))
            components.append(("gait_phase", 3))
        components.extend(
            (
                ("joint_position", self.action_size),
                ("joint_velocity", self.action_size),
                ("last_action", self.action_size),
            )
        )
        if observation.privileged:
            components.extend(
                (
                    ("base_linear_velocity", 3),
                    ("base_height", 1),
                    ("actuator_force", self.action_size),
                )
            )
        return ObservationLayout(tuple(components))

    def _validate_model_metadata(self) -> None:
        model = self.model
        mj_model = model.mj_model
        require_index_range(
            "policy_ctrl_indices", model.policy_ctrl_indices, mj_model.nu
        )
        require_index_range("hold_ctrl_indices", model.hold_ctrl_indices, mj_model.nu)
        require_disjoint(
            "policy_ctrl_indices",
            model.policy_ctrl_indices,
            "hold_ctrl_indices",
            model.hold_ctrl_indices,
        )
        if model.policy_ctrl_indices.size + model.hold_ctrl_indices.size != mj_model.nu:
            raise ValueError(
                "policy_ctrl_indices and hold_ctrl_indices must cover every actuator "
                f"exactly once, got {model.policy_ctrl_indices.size} policy and "
                f"{model.hold_ctrl_indices.size} hold controls for nu={mj_model.nu}"
            )
        mirror_joint_names = {
            mirror.joint_name for mirror in model.robot.locomotion_action_mirror
        }
        action_joint_names = set(self.action_layout.names)
        if mirror_joint_names != action_joint_names:
            missing = sorted(action_joint_names - mirror_joint_names)
            unknown = sorted(mirror_joint_names - action_joint_names)
            details = []
            if missing:
                details.append(f"missing={missing}")
            if unknown:
                details.append(f"unknown={unknown}")
            raise ValueError(
                "locomotion action mirror must match the environment action layout: "
                + ", ".join(details)
            )
        policy_peak_force = model.actuator_peak_force[model.policy_ctrl_indices]
        if not np.all(np.isfinite(policy_peak_force)) or np.any(
            policy_peak_force <= 0.0
        ):
            raise ValueError(
                "policy actuators require positive finite peak-force metadata"
            )
        if self.reward_config.enabled("torque_peak") and (
            np.any(mj_model.actuator_gaintype != mujoco.mjtGain.mjGAIN_FIXED)
            or np.any(mj_model.actuator_biastype != mujoco.mjtBias.mjBIAS_AFFINE)
            or np.any(mj_model.actuator_dyntype != mujoco.mjtDyn.mjDYN_NONE)
        ):
            raise ValueError(
                "torque_peak demand reconstruction requires stateless position "
                "actuators with fixed gain and affine bias"
            )
        require_index_range(
            "actuator_qpos_indices", model.actuator_qpos_indices, mj_model.nq
        )
        require_index_range(
            "actuator_qvel_indices", model.actuator_qvel_indices, mj_model.nv
        )
        require_index_range(
            "actuator_joint_ids", model.actuator_joint_ids, mj_model.njnt
        )
        require_index_range(
            "policy_qpos_indices", model.policy_qpos_indices, mj_model.nq
        )
        require_index_range(
            "policy_qvel_indices", model.policy_qvel_indices, mj_model.nv
        )
        require_index_range("foot_body_ids", model.foot_body_ids, mj_model.nbody)
        if self.command_config.enabled and model.foot_body_ids.size != 2:
            raise ValueError(
                "gait-conditioned locomotion requires exactly two foot bodies"
            )
        require_index_range("robot_geom_ids", model.robot_geom_ids, mj_model.ngeom)
        require_index_range("foot_geom_ids", model.foot_geom_ids, mj_model.ngeom)
        require_index_range("terrain_geom_ids", model.terrain_geom_ids, mj_model.ngeom)
        require_index_range(
            "nonfoot_robot_geom_ids", model.nonfoot_robot_geom_ids, mj_model.ngeom
        )
        require_index_range(
            "nonfoot_robot_body_ids", model.nonfoot_robot_body_ids, mj_model.nbody
        )
        if isinstance(mj_model, mujoco.MjModel):
            required_link_names = {
                self.root_link_name,
                model.imu_body_name,
                model.robot.locomotion_command_frame_body_name,
                *self.reward_config.enabled_link_names,
                *self.done_config.enabled_link_names,
            }
            if self.reward_config.enabled("stand_foot_heading"):
                required_link_names.update(
                    name for name, _ in self.model.robot.foot_forward_axes
                )
            for link_name in required_link_names:
                if self._link_body_id(link_name) < 0:
                    raise ValueError(f"configured link/body not found: {link_name!r}")
        if model.actuator_coordinate_transform.shape != (mj_model.nu, mj_model.nu):
            raise ValueError(
                "actuator_coordinate_transform must have shape "
                f"({mj_model.nu}, {mj_model.nu}), got "
                f"{model.actuator_coordinate_transform.shape}"
            )
        if (
            model.joint_angle_offset.ndim != 2
            or model.joint_angle_offset.shape[1] != mj_model.nu
            or model.joint_angle_offset.shape[0] < 1
        ):
            raise ValueError(
                "joint_angle_offset must have shape (worlds, nu), got "
                f"{model.joint_angle_offset.shape} for nu={mj_model.nu}"
            )
        for name in ("default_joint_target", "joint_target_low", "joint_target_high"):
            value = getattr(model, name)
            if value.shape != (mj_model.nu,):
                raise ValueError(
                    f"{name} must have shape ({mj_model.nu},), got {value.shape}"
                )
        if np.any(model.joint_target_low >= model.joint_target_high):
            raise ValueError("joint target lower limits must be below upper limits")
        if model.actuator_qpos_indices.shape != (mj_model.nu,):
            raise ValueError(
                "actuator_qpos_indices must have one entry per actuator, got "
                f"{model.actuator_qpos_indices.shape} for nu={mj_model.nu}"
            )
        if model.actuator_qvel_indices.shape != (mj_model.nu,):
            raise ValueError(
                "actuator_qvel_indices must have one entry per actuator, got "
                f"{model.actuator_qvel_indices.shape} for nu={mj_model.nu}"
            )
        if self.reward_config.uses_foot_metadata and model.foot_body_ids.shape != (2,):
            raise ValueError(
                "enabled foot reward terms require exactly two foot bodies, got "
                f"{model.foot_body_ids.shape}. Disable foot-dependent rewards for "
                "assets without feet."
            )
        if (
            self.reward_config.uses_foot_contact_metadata
            and model.foot_geom_ids.size == 0
        ):
            raise ValueError(
                "enabled foot reward terms require foot collision geoms. Disable "
                "contact-dependent foot rewards for assets without feet."
            )
        if self.terrain_config.mode != "none" and model.terrain_geom_ids.size == 0:
            raise ValueError(
                f"terrain mode {self.terrain_config.mode!r} requires a terrain geom "
                f"named {self.terrain_config.name!r}"
            )
        if model.default_qpos.shape != (mj_model.nq,):
            raise ValueError(
                f"default_qpos shape {model.default_qpos.shape} does not match nq={mj_model.nq}"
            )
        if model.default_qvel.shape != (mj_model.nv,):
            raise ValueError(
                f"default_qvel shape {model.default_qvel.shape} does not match nv={mj_model.nv}"
            )
        if model.default_ctrl.shape != (mj_model.nu,):
            raise ValueError(
                f"default_ctrl shape {model.default_ctrl.shape} does not match nu={mj_model.nu}"
            )

    def reset(
        self,
        command: jax.Array | None = None,
        rng: jax.Array | None = None,
    ) -> tuple[BelloState, jax.Array]:
        return self._reset(command, rng, world_id=0)

    def _reset(
        self,
        command: jax.Array | None,
        rng: jax.Array | None,
        world_id: jax.Array | int,
        *,
        forward: bool = True,
        align_clearance: bool = True,
    ) -> tuple[BelloState, jax.Array]:
        command_rng = reset_rng = feedback_rng = obs_rng = latency_rng = gait_rng = rng
        if rng is not None:
            (
                command_rng,
                reset_rng,
                obs_rng,
                latency_rng,
                gait_rng,
            ) = jax.random.split(rng, 5)
            feedback_rng = jax.random.fold_in(obs_rng, 1)
            obs_rng = jax.random.fold_in(obs_rng, 2)
        command = self._resolve_command(command, command_rng)
        gait_phase = jnp.asarray(0.0, dtype=jnp.float32)
        if gait_rng is not None:
            sampled_phase = jax.random.uniform(gait_rng, (), dtype=jnp.float32)
            gait_phase = jnp.where(
                command_is_moving(command, self.command_config), sampled_phase, 0.0
            )
        data = self.model.make_data()
        data = data.replace(
            qpos=jnp.asarray(self.model.default_qpos),
            qvel=jnp.asarray(self.model.default_qvel),
            ctrl=self.action_to_ctrl(jnp.zeros(self.action_size, dtype=jnp.float32)),
        )
        data = self._apply_reset_randomization(data, reset_rng, world_id)
        data = mjx.kinematics(self.model.mjx_model, data)
        state = BelloState(
            data=data,
            last_action=jnp.zeros(self.action_size, dtype=jnp.float32),
            last_action_delta=jnp.zeros(self.action_size, dtype=jnp.float32),
            tracking_ang_vel_z=jnp.asarray(0.0, dtype=jnp.float32),
            actor_feedback_history=jnp.zeros(
                (
                    self.maximum_observation_delay_steps + 1,
                    self.actor_feedback_size,
                ),
                dtype=jnp.float32,
            ),
            actor_feedback_delay_steps=self._sample_observation_delay(latency_rng),
            foot_contact_debounce_time=jnp.zeros(
                self.model.foot_body_ids.size, dtype=jnp.float32
            ),
            last_foot_contact=jnp.zeros(self.model.foot_body_ids.size, dtype=bool),
            command=command,
            gait_phase=gait_phase,
            standing_transition_pending=jnp.asarray(False),
            command_resample_step_count=jnp.asarray(0, dtype=jnp.int32),
            step_count=jnp.asarray(0, dtype=jnp.int32),
            episode_return=jnp.asarray(0.0, dtype=jnp.float32),
            episode_track_lin_vel_xy=jnp.asarray(0.0, dtype=jnp.float32),
            episode_length=jnp.asarray(0, dtype=jnp.int32),
            last_episode_return=jnp.asarray(0.0, dtype=jnp.float32),
            last_episode_track_lin_vel_xy=jnp.asarray(0.0, dtype=jnp.float32),
            last_episode_length=jnp.asarray(0, dtype=jnp.int32),
        )
        if align_clearance:
            state = self._align_reset_foot_clearance(state, world_id)
            data = (
                mjx.forward(self.model.mjx_model, state.data)
                if forward
                else mjx.kinematics(self.model.mjx_model, state.data)
            )
            state = state.replace(data=data)
        actor_feedback = self._actor_feedback(state)
        if self.config.observation_low_pass_filter.noise_before_filter:
            actor_feedback = self._noisy_actor_feedback(actor_feedback, feedback_rng)
        state = state.replace(
            tracking_ang_vel_z=self._raw_tracking_ang_vel_z(state),
            actor_feedback_history=jnp.repeat(
                actor_feedback[None, :],
                self.maximum_observation_delay_steps + 1,
                axis=0,
            ),
        )
        return state, self.observation(state, obs_rng)

    def reset_batch(
        self,
        batch_size: int,
        command: jax.Array | None = None,
        rng: jax.Array | None = None,
    ) -> tuple[BelloState, jax.Array]:
        command_rng = reset_rng = feedback_rng = obs_rng = rng
        if rng is not None:
            command_rng, reset_rng, obs_rng = jax.random.split(rng, 3)
            feedback_rng = jax.random.fold_in(obs_rng, 1)
            obs_rng = jax.random.fold_in(obs_rng, 2)
        command = self._resolve_batch_command(batch_size, command, command_rng)
        world_ids = jnp.arange(batch_size, dtype=jnp.int32)
        if reset_rng is None:
            state, _ = jax.vmap(
                lambda item, world_id: self._reset(
                    item,
                    None,
                    world_id,
                    forward=False,
                    align_clearance=False,
                )
            )(command, world_ids)
        else:
            reset_keys = jax.random.split(reset_rng, batch_size)
            state, _ = jax.vmap(
                lambda item, key, world_id: self._reset(
                    item,
                    key,
                    world_id,
                    forward=False,
                    align_clearance=False,
                )
            )(command, reset_keys, world_ids)
        data = state.data.replace(_impl=state.data._impl.replace(nworld=batch_size))
        data = data.replace(
            ctrl=jax.vmap(self.action_to_ctrl)(
                jnp.zeros((batch_size, self.action_size), dtype=jnp.float32),
                self._joint_angle_offset_batch(batch_size),
            )
        )
        data = mjx.kinematics(self.model.mjx_model, data)
        state = state.replace(data=data)
        clearance_adjustment = jax.vmap(self._reset_foot_clearance_adjustment)(
            state, world_ids
        )
        root_qpos = int(self.model.root_qpos_adr)
        data = data.replace(
            qpos=data.qpos.at[:, root_qpos + 2].add(clearance_adjustment)
        )
        data = mjx.forward(self.model.mjx_model, data)
        state = state.replace(data=data)
        state = state.replace(
            tracking_ang_vel_z=jax.vmap(self._raw_tracking_ang_vel_z)(state)
        )
        actor_feedback = jax.vmap(self._actor_feedback)(state)
        if (
            feedback_rng is not None
            and self.config.observation_low_pass_filter.noise_before_filter
        ):
            feedback_keys = jax.random.split(feedback_rng, batch_size)
            actor_feedback = jax.vmap(self._noisy_actor_feedback)(
                actor_feedback, feedback_keys
            )
        state = state.replace(
            actor_feedback_history=jnp.repeat(
                actor_feedback[:, None, :],
                self.maximum_observation_delay_steps + 1,
                axis=1,
            )
        )
        return state, self.observation_batch(state, obs_rng)

    def _reset_foot_clearance_adjustment(
        self, state: BelloState, world_id: jax.Array | int
    ) -> jax.Array:
        minimum_clearance = jnp.min(self._foot_height(state, world_id))
        return (
            jnp.asarray(self.config.spawn.clearance, dtype=jnp.float32)
            - minimum_clearance
        )

    def _align_reset_foot_clearance(
        self, state: BelloState, world_id: jax.Array | int
    ) -> BelloState:
        root_qpos = int(self.model.root_qpos_adr)
        adjustment = self._reset_foot_clearance_adjustment(state, world_id)
        return state.replace(
            data=state.data.replace(
                qpos=state.data.qpos.at[root_qpos + 2].add(adjustment)
            )
        )

    def initial_step_counts(
        self, batch_size: int, rng: jax.Array | None = None
    ) -> jax.Array:
        if rng is None or not self.config.spawn.stagger_initial_step_count:
            return jnp.zeros((batch_size,), dtype=jnp.int32)
        return jax.random.randint(
            rng,
            (batch_size,),
            minval=0,
            maxval=self.max_episode_steps,
            dtype=jnp.int32,
        )

    def _apply_reset_randomization(
        self,
        data: mjx.Data,
        rng: jax.Array | None,
        world_id: jax.Array | int = 0,
    ) -> mjx.Data:
        if rng is None:
            return data
        xy_key, yaw_key, qpos_key, scale_key, qvel_key, root_vel_key = jax.random.split(
            rng, 6
        )
        root_qpos = int(self.model.root_qpos_adr)
        qpos = data.qpos
        root_xy = jax.random.uniform(
            xy_key,
            (2,),
            minval=self.config.spawn.root_xy_range[0],
            maxval=self.config.spawn.root_xy_range[1],
            dtype=jnp.float32,
        )
        default_root_xy = qpos[root_qpos : root_qpos + 2]
        randomized_root_xy = default_root_xy + root_xy
        terrain_height_delta = self._terrain_height(
            randomized_root_xy, world_id
        ) - self._terrain_height(default_root_xy, world_id)
        yaw = jax.random.uniform(
            yaw_key,
            (),
            minval=self.config.spawn.root_yaw_range[0],
            maxval=self.config.spawn.root_yaw_range[1],
            dtype=jnp.float32,
        )
        qpos = (
            qpos.at[root_qpos : root_qpos + 2]
            .add(root_xy)
            .at[root_qpos + 2]
            .add(terrain_height_delta)
            .at[root_qpos + 3 : root_qpos + 7]
            .set(yaw_quat_from_angle(yaw))
        )
        qpos_idx = jnp.asarray(self.model.actuator_qpos_indices)
        joint_noise = jax.random.uniform(
            scale_key,
            qpos_idx.shape,
            minval=self.config.spawn.joint_pos_noise[0],
            maxval=self.config.spawn.joint_pos_noise[1],
            dtype=jnp.float32,
        )
        joint_ids = jnp.asarray(self.model.actuator_joint_ids)
        joint_range = jnp.asarray(self.model.mj_model.jnt_range)[joint_ids]
        joint_scale = jax.random.uniform(
            qpos_key,
            qpos_idx.shape,
            minval=self.config.spawn.joint_pos_scale_range[0],
            maxval=self.config.spawn.joint_pos_scale_range[1],
            dtype=jnp.float32,
        )
        default_joint_pos = jnp.asarray(self.model.default_qpos)[qpos_idx]
        joint_pos = jnp.clip(
            default_joint_pos * joint_scale + joint_noise,
            joint_range[:, 0],
            joint_range[:, 1],
        )
        qpos = qpos.at[qpos_idx].set(joint_pos)
        qvel_idx = jnp.asarray(self.model.actuator_qvel_indices)
        joint_vel = jax.random.uniform(
            qvel_key,
            qvel_idx.shape,
            minval=self.config.spawn.joint_vel_range[0],
            maxval=self.config.spawn.joint_vel_range[1],
            dtype=jnp.float32,
        )
        qvel = data.qvel.at[qvel_idx].set(joint_vel)
        root_qvel = int(self.model.root_qvel_adr)
        root_velocity = jax.random.uniform(
            root_vel_key,
            (6,),
            minval=self.config.spawn.root_vel_range[0],
            maxval=self.config.spawn.root_vel_range[1],
            dtype=jnp.float32,
        )
        qvel = qvel.at[root_qvel : root_qvel + 6].set(root_velocity)
        return data.replace(qpos=qpos, qvel=qvel)

    def _step_physics_with_impacts(
        self, state: BelloState, data: mjx.Data
    ) -> tuple[mjx.Data, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        foot_count = int(self.model.foot_body_ids.size)
        touchdown_eligible = (
            (state.step_count > 0)
            & ~state.last_foot_contact.astype(bool)
            & (state.foot_contact_debounce_time <= 0.0)
        )
        zeros = jnp.zeros(foot_count, dtype=jnp.float32)
        initial = (
            data,
            jnp.zeros(foot_count, dtype=bool),
            zeros,
            zeros,
            jnp.zeros((foot_count, 3), dtype=jnp.float32),
            jnp.zeros(self.model.mj_model.nu, dtype=jnp.float32),
        )
        physics_timestep = jnp.asarray(
            self.model.mj_model.opt.timestep, dtype=jnp.float32
        )

        def substep(_, carry):
            (
                current_data,
                touchdown,
                normal_impulse,
                peak_normal_force,
                touchdown_velocity,
                requested_force_peak,
            ) = carry
            current_state = state.replace(data=current_data)
            preimpact_velocity = self._foot_world_linear_velocity(current_state)
            next_data = mjx.step(self.model.mjx_model, current_data)
            next_state = state.replace(data=next_data)
            raw_contact, normal_force, *_ = self._contact_terms(next_state, 0)
            new_touchdown = touchdown_eligible & ~touchdown & raw_contact.astype(bool)
            touchdown_velocity = jnp.where(
                new_touchdown[:, None], preimpact_velocity, touchdown_velocity
            )
            return (
                next_data,
                touchdown | new_touchdown,
                normal_impulse + normal_force * physics_timestep,
                jnp.maximum(peak_normal_force, normal_force),
                touchdown_velocity,
                jnp.maximum(
                    requested_force_peak,
                    jnp.abs(self.requested_actuator_force(next_data)),
                ),
            )

        return jax.lax.fori_loop(0, self.decimation, substep, initial)

    def _step_physics_with_impacts_batch(
        self, state: BelloState, data: mjx.Data, batch_size: int
    ) -> tuple[mjx.Data, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        foot_count = int(self.model.foot_body_ids.size)
        world_ids = jnp.arange(batch_size, dtype=jnp.int32)
        touchdown_eligible = (
            (state.step_count > 0)[:, None]
            & ~state.last_foot_contact.astype(bool)
            & (state.foot_contact_debounce_time <= 0.0)
        )
        zeros = jnp.zeros((batch_size, foot_count), dtype=jnp.float32)
        initial = (
            data,
            jnp.zeros((batch_size, foot_count), dtype=bool),
            zeros,
            zeros,
            jnp.zeros((batch_size, foot_count, 3), dtype=jnp.float32),
            jnp.zeros((batch_size, self.model.mj_model.nu), dtype=jnp.float32),
        )
        physics_timestep = jnp.asarray(
            self.model.mj_model.opt.timestep, dtype=jnp.float32
        )

        def substep(_, carry):
            (
                current_data,
                touchdown,
                normal_impulse,
                peak_normal_force,
                touchdown_velocity,
                requested_force_peak,
            ) = carry
            current_state = state.replace(data=current_data)
            preimpact_velocity = jax.vmap(self._foot_world_linear_velocity)(
                current_state
            )
            next_data = mjx.step(self.model.mjx_model, current_data)
            next_state = state.replace(data=next_data)
            raw_contact, normal_force, *_ = jax.vmap(self._contact_terms)(
                next_state, world_ids
            )
            new_touchdown = touchdown_eligible & ~touchdown & raw_contact.astype(bool)
            touchdown_velocity = jnp.where(
                new_touchdown[:, :, None], preimpact_velocity, touchdown_velocity
            )
            return (
                next_data,
                touchdown | new_touchdown,
                normal_impulse + normal_force * physics_timestep,
                jnp.maximum(peak_normal_force, normal_force),
                touchdown_velocity,
                jnp.maximum(
                    requested_force_peak,
                    jnp.abs(self.requested_actuator_force(next_data)),
                ),
            )

        return jax.lax.fori_loop(0, self.decimation, substep, initial)

    def step(
        self, state: BelloState, action: jax.Array
    ) -> tuple[BelloState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        requested_action = jnp.asarray(action, dtype=jnp.float32)
        action = jnp.clip(
            requested_action,
            -self.action_clip,
            self.action_clip,
        )
        action_delta = action - state.last_action
        ctrl, joint_saturation, motor_saturation = self._action_target_details(action)
        data = state.data.replace(ctrl=ctrl)
        (
            data,
            foot_touchdown,
            foot_normal_impulse,
            foot_peak_normal_force,
            touchdown_foot_linear_velocity,
            requested_actuator_force,
        ) = self._step_physics_with_impacts(state, data)
        next_data_state = state.replace(data=data)
        next_gait_phase = advance_gait_phase(
            state.gait_phase,
            state.command,
            self.command_config,
            self.config.gait,
            self.control_timestep,
        )
        reward_state = BelloState(
            data=data,
            last_action=action,
            last_action_delta=action_delta,
            tracking_ang_vel_z=self.filter_tracking_ang_vel_z(
                state.tracking_ang_vel_z,
                self._raw_tracking_ang_vel_z(next_data_state),
            ),
            actor_feedback_history=self._advance_actor_feedback_history(
                state.actor_feedback_history, next_data_state
            ),
            actor_feedback_delay_steps=state.actor_feedback_delay_steps,
            foot_contact_debounce_time=state.foot_contact_debounce_time,
            last_foot_contact=state.last_foot_contact,
            command=state.command,
            gait_phase=next_gait_phase,
            standing_transition_pending=state.standing_transition_pending,
            command_resample_step_count=state.command_resample_step_count + 1,
            step_count=state.step_count + 1,
            episode_return=state.episode_return,
            episode_track_lin_vel_xy=state.episode_track_lin_vel_xy,
            episode_length=state.episode_length,
            last_episode_return=state.last_episode_return,
            last_episode_track_lin_vel_xy=state.last_episode_track_lin_vel_xy,
            last_episode_length=state.last_episode_length,
        )
        done_flags = self.done_flags(reward_state)
        done = done_flags["done/any"]
        terminal = done_flags["done/terminal"]
        (
            raw_foot_contact,
            foot_normal_force,
            foot_linear_velocity,
            foot_angular_speed,
            undesired_contact,
            self_collision,
        ) = self._contact_terms(reward_state, 0)
        foot_contact, foot_contact_debounce_time = self.debounce_foot_contact(
            reward_state, raw_foot_contact, done
        )
        reward, reward_info = self.reward(
            state,
            reward_state,
            action,
            requested_action,
            terminal,
            foot_contact,
            foot_touchdown,
            foot_normal_impulse,
            foot_peak_normal_force,
            touchdown_foot_linear_velocity,
            requested_actuator_force,
            foot_linear_velocity,
            foot_angular_speed,
            undesired_contact,
            self_collision,
        )
        reward, reward_info = self.sanitize_nonfinite_reward(
            reward, reward_info, done_flags
        )
        episode_return = state.episode_return + reward
        episode_track_lin_vel_xy = (
            state.episode_track_lin_vel_xy + reward_info["reward/track_lin_vel_xy"]
        )
        episode_length = state.episode_length + 1
        pending_command = jnp.where(
            reward_state.standing_transition_pending[..., None],
            jnp.zeros_like(reward_state.command),
            reward_state.command,
        )
        next_command, next_gait_phase, standing_transition_pending = (
            resolve_gait_command_transition(
                reward_state.command,
                pending_command,
                state.gait_phase,
                reward_state.gait_phase,
                reward_state.standing_transition_pending,
                self.command_config,
            )
        )
        next_state = BelloState(
            data=reward_state.data,
            last_action=reward_state.last_action,
            last_action_delta=reward_state.last_action_delta,
            tracking_ang_vel_z=reward_state.tracking_ang_vel_z,
            actor_feedback_history=reward_state.actor_feedback_history,
            actor_feedback_delay_steps=reward_state.actor_feedback_delay_steps,
            foot_contact_debounce_time=foot_contact_debounce_time,
            last_foot_contact=foot_contact.astype(bool),
            command=next_command,
            gait_phase=next_gait_phase,
            standing_transition_pending=standing_transition_pending,
            command_resample_step_count=reward_state.command_resample_step_count,
            step_count=reward_state.step_count,
            episode_return=episode_return,
            episode_track_lin_vel_xy=episode_track_lin_vel_xy,
            episode_length=episode_length,
            last_episode_return=jnp.where(
                done, episode_return, state.last_episode_return
            ),
            last_episode_track_lin_vel_xy=jnp.where(
                done, episode_track_lin_vel_xy, state.last_episode_track_lin_vel_xy
            ),
            last_episode_length=jnp.where(
                done, episode_length, state.last_episode_length
            ),
        )
        info = {
            "base_height": self._base_height(next_state),
            "bootstrap_observation": self.observation(next_state),
            "bootstrap_critic_observation": self.critic_observation(next_state),
            "episode_return": jnp.where(done, episode_return, 0.0),
            "episode_reward/track_lin_vel_xy": jnp.where(
                done, episode_track_lin_vel_xy, 0.0
            ),
            "episode_length": jnp.where(done, episode_length, 0),
            "foot_contact/raw": raw_foot_contact,
            "foot_contact/filtered": foot_contact,
            "foot_contact/debounce_time": foot_contact_debounce_time,
            "target/joint_saturation_fraction": joint_saturation,
            "target/motor_saturation_fraction": motor_saturation,
            "latency/observation_steps": (
                next_state.actor_feedback_delay_steps.astype(jnp.float32)
            ),
            **self._command_diagnostics(reward_state.command),
            **self._gait_diagnostics(
                state,
                reward_state,
                foot_contact,
                foot_touchdown,
                foot_normal_impulse,
                foot_peak_normal_force,
            ),
            "returned_episode": done,
            **done_flags,
            **reward_info,
        }
        return next_state, self.observation(next_state), reward, done, info

    def step_batch(
        self,
        state: BelloState,
        action: jax.Array,
        rng: jax.Array | None = None,
    ) -> tuple[BelloState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        batch_size = int(action.shape[0])
        resample_rng = reset_obs_rng = None
        push_rng = feedback_rng = obs_rng = None
        if rng is not None:
            (
                resample_rng,
                _,
                reset_obs_rng,
                push_rng,
                obs_rng,
            ) = jax.random.split(rng, 5)
            feedback_rng = jax.random.fold_in(obs_rng, 1)
            obs_rng = jax.random.fold_in(obs_rng, 2)
        requested_action = jnp.asarray(action, dtype=jnp.float32)
        action = jnp.clip(
            requested_action,
            -self.action_clip,
            self.action_clip,
        )
        action_delta = action - state.last_action
        joint_angle_offset = self._joint_angle_offset_batch(batch_size)
        ctrl, joint_saturation, motor_saturation = jax.vmap(
            self._action_target_details
        )(action, joint_angle_offset)
        data = state.data.replace(
            ctrl=ctrl,
            _impl=state.data._impl.replace(nworld=batch_size),
        )
        data, push_applied, push_velocity = self._maybe_apply_random_push_batch(
            data, state.step_count, push_rng
        )
        (
            data,
            foot_touchdown,
            foot_normal_impulse,
            foot_peak_normal_force,
            touchdown_foot_linear_velocity,
            requested_actuator_force,
        ) = self._step_physics_with_impacts_batch(state, data, batch_size)
        next_data_state = state.replace(data=data)
        if feedback_rng is None:
            actor_feedback_history = jax.vmap(self._advance_actor_feedback_history)(
                state.actor_feedback_history, next_data_state
            )
        else:
            feedback_keys = jax.random.split(feedback_rng, batch_size)
            actor_feedback_history = jax.vmap(self._advance_actor_feedback_history)(
                state.actor_feedback_history, next_data_state, feedback_keys
            )
        next_gait_phase = advance_gait_phase(
            state.gait_phase,
            state.command,
            self.command_config,
            self.config.gait,
            self.control_timestep,
        )
        stepped_state = BelloState(
            data=data,
            last_action=action,
            last_action_delta=action_delta,
            tracking_ang_vel_z=self.filter_tracking_ang_vel_z(
                state.tracking_ang_vel_z,
                jax.vmap(self._raw_tracking_ang_vel_z)(next_data_state),
            ),
            actor_feedback_history=actor_feedback_history,
            actor_feedback_delay_steps=state.actor_feedback_delay_steps,
            foot_contact_debounce_time=state.foot_contact_debounce_time,
            last_foot_contact=state.last_foot_contact,
            command=state.command,
            gait_phase=next_gait_phase,
            standing_transition_pending=state.standing_transition_pending,
            command_resample_step_count=state.command_resample_step_count + 1,
            step_count=state.step_count + 1,
            episode_return=state.episode_return,
            episode_track_lin_vel_xy=state.episode_track_lin_vel_xy,
            episode_length=state.episode_length,
            last_episode_return=state.last_episode_return,
            last_episode_track_lin_vel_xy=state.last_episode_track_lin_vel_xy,
            last_episode_length=state.last_episode_length,
        )
        world_ids = jnp.arange(batch_size, dtype=jnp.int32)
        done_flags = jax.vmap(self.done_flags)(stepped_state, world_ids)
        done = done_flags["done/any"]
        terminal = done_flags["done/terminal"]
        (
            raw_foot_contact,
            foot_normal_force,
            foot_linear_velocity,
            foot_angular_speed,
            undesired_contact,
            self_collision,
        ) = jax.vmap(self._contact_terms)(stepped_state, world_ids)
        foot_contact, foot_contact_debounce_time = jax.vmap(self.debounce_foot_contact)(
            stepped_state, raw_foot_contact, done
        )
        reward, reward_info = jax.vmap(self.reward)(
            state,
            stepped_state,
            action,
            requested_action,
            terminal,
            foot_contact,
            foot_touchdown,
            foot_normal_impulse,
            foot_peak_normal_force,
            touchdown_foot_linear_velocity,
            requested_actuator_force,
            foot_linear_velocity,
            foot_angular_speed,
            undesired_contact,
            self_collision,
            joint_angle_offset,
            world_ids,
        )
        reward, reward_info = self.sanitize_nonfinite_reward(
            reward, reward_info, done_flags
        )
        episode_return = state.episode_return + reward
        episode_track_lin_vel_xy = (
            state.episode_track_lin_vel_xy + reward_info["reward/track_lin_vel_xy"]
        )
        episode_length = state.episode_length + 1
        reward_state = stepped_state
        pending_command = jnp.where(
            reward_state.standing_transition_pending[:, None],
            jnp.zeros_like(reward_state.command),
            reward_state.command,
        )
        (
            transitioned_command,
            transitioned_gait_phase,
            transitioned_standing_pending,
        ) = resolve_gait_command_transition(
            reward_state.command,
            pending_command,
            state.gait_phase,
            reward_state.gait_phase,
            reward_state.standing_transition_pending,
            self.command_config,
        )
        transitioned_state = reward_state.replace(
            command=transitioned_command,
            gait_phase=transitioned_gait_phase,
            standing_transition_pending=transitioned_standing_pending,
        )
        (
            next_command,
            next_gait_phase,
            standing_transition_pending,
            command_resample_step_count,
        ) = self._maybe_resample_batch_command_and_phase(
            transitioned_state, resample_rng
        )
        stepped_state = BelloState(
            data=reward_state.data,
            last_action=reward_state.last_action,
            last_action_delta=reward_state.last_action_delta,
            tracking_ang_vel_z=reward_state.tracking_ang_vel_z,
            actor_feedback_history=reward_state.actor_feedback_history,
            actor_feedback_delay_steps=reward_state.actor_feedback_delay_steps,
            foot_contact_debounce_time=foot_contact_debounce_time,
            last_foot_contact=foot_contact.astype(bool),
            command=next_command,
            gait_phase=next_gait_phase,
            standing_transition_pending=standing_transition_pending,
            command_resample_step_count=command_resample_step_count,
            step_count=reward_state.step_count,
            episode_return=episode_return,
            episode_track_lin_vel_xy=episode_track_lin_vel_xy,
            episode_length=episode_length,
            last_episode_return=jnp.where(
                done, episode_return, state.last_episode_return
            ),
            last_episode_track_lin_vel_xy=jnp.where(
                done, episode_track_lin_vel_xy, state.last_episode_track_lin_vel_xy
            ),
            last_episode_length=jnp.where(
                done, episode_length, state.last_episode_length
            ),
        )
        stepped_obs = self.observation_batch(stepped_state, obs_rng)
        stepped_critic_obs = self.critic_observation_batch(stepped_state)
        info = {
            "base_height": jax.vmap(self._base_height)(stepped_state, world_ids),
            "bootstrap_observation": stepped_obs,
            "bootstrap_critic_observation": stepped_critic_obs,
            "episode_return": jnp.where(done, episode_return, 0.0),
            "episode_reward/track_lin_vel_xy": jnp.where(
                done, episode_track_lin_vel_xy, 0.0
            ),
            "episode_length": jnp.where(done, episode_length, 0),
            "foot_contact/raw": raw_foot_contact,
            "foot_contact/filtered": foot_contact,
            "foot_contact/debounce_time": foot_contact_debounce_time,
            "target/joint_saturation_fraction": joint_saturation,
            "target/motor_saturation_fraction": motor_saturation,
            "latency/observation_steps": (
                stepped_state.actor_feedback_delay_steps.astype(jnp.float32)
            ),
            **jax.vmap(self._gait_diagnostics)(
                state,
                reward_state,
                foot_contact,
                foot_touchdown,
                foot_normal_impulse,
                foot_peak_normal_force,
            ),
            "random_push/applied": push_applied,
            "random_push/lin_vel_x": push_velocity[:, 0],
            "random_push/lin_vel_y": push_velocity[:, 1],
            **self._command_diagnostics(reward_state.command),
            "returned_episode": done,
            **done_flags,
            **reward_info,
        }

        def reset_terminated_worlds(_):
            reset_source_command = jnp.where(
                stepped_state.standing_transition_pending[:, None],
                jnp.zeros_like(stepped_state.command),
                stepped_state.command,
            )
            reset_state, reset_obs = self.reset_batch(
                batch_size, command=reset_source_command, rng=reset_obs_rng
            )
            reset_state = reset_state.replace(
                command_resample_step_count=(stepped_state.command_resample_step_count),
                last_episode_return=stepped_state.last_episode_return,
                last_episode_track_lin_vel_xy=(
                    stepped_state.last_episode_track_lin_vel_xy
                ),
                last_episode_length=stepped_state.last_episode_length,
            )
            return (
                tree_where(done, reset_state, stepped_state),
                jnp.where(done[:, None], reset_obs, stepped_obs),
            )

        next_state, next_obs = jax.lax.cond(
            jnp.any(done),
            reset_terminated_worlds,
            lambda _: (stepped_state, stepped_obs),
            operand=None,
        )
        return next_state, next_obs, reward, done, info

    def observation_batch(
        self, state: BelloState, rng: jax.Array | None = None
    ) -> jax.Array:
        world_ids = jnp.arange(int(state.command.shape[0]), dtype=jnp.int32)
        if rng is None:
            return jax.vmap(
                lambda item, world_id: self.observation(item, None, world_id)
            )(state, world_ids)
        keys = jax.random.split(rng, int(state.command.shape[0]))
        return jax.vmap(self.observation)(state, keys, world_ids)

    def critic_observation_batch(
        self, state: BelloState, rng: jax.Array | None = None
    ) -> jax.Array:
        world_ids = jnp.arange(int(state.command.shape[0]), dtype=jnp.int32)
        if rng is None:
            return jax.vmap(
                lambda item, world_id: self.critic_observation(item, None, world_id)
            )(state, world_ids)
        keys = jax.random.split(rng, int(state.command.shape[0]))
        return jax.vmap(self.critic_observation)(state, keys, world_ids)

    def observation(
        self,
        state: BelloState,
        rng: jax.Array | None = None,
        world_id: jax.Array | int = 0,
    ) -> jax.Array:
        return self._observation(
            state,
            self.config.observation,
            rng,
            world_id,
            _feedback=self._delayed_actor_feedback(state),
        )

    def critic_observation(
        self,
        state: BelloState,
        rng: jax.Array | None = None,
        world_id: jax.Array | int = 0,
    ) -> jax.Array:
        return self._observation(state, self.critic_observation_config, rng, world_id)

    def _observation(
        self,
        state: BelloState,
        observation: ObservationConfig,
        rng: jax.Array | None = None,
        world_id: jax.Array | int = 0,
        *,
        _feedback: jax.Array | None = None,
    ) -> jax.Array:
        (
            base_lin_vel,
            base_lin_acc,
            base_ang_vel,
            projected_gravity,
            base_height,
            command,
            joint_pos,
            joint_vel,
            actuator_force,
            last_action,
        ) = self._frame_terms(state, world_id)
        if _feedback is not None:
            (
                base_lin_acc,
                base_ang_vel,
                projected_gravity,
                joint_pos,
                joint_vel,
            ) = self._unpack_actor_feedback(_feedback)
        scale = observation.scale
        noise = observation.noise
        feedback_noise_scale = 1.0
        joint_position_noise_scale = 1.0
        if (
            _feedback is not None
            and self.config.observation_low_pass_filter.noise_before_filter
        ):
            feedback_noise_scale = (
                self.config.observation_low_pass_filter.residual_noise_scale
            )
            joint_position_noise_scale = 0.0
        keys = (
            jax.random.split(rng, 11)
            if rng is not None and noise.enabled
            else (None,) * 11
        )
        parts = [
            self._observation_component(
                base_lin_acc,
                noise.base_lin_acc,
                scale.base_lin_acc,
                keys[0],
                noise_scale=feedback_noise_scale,
            ),
            self._observation_component(
                base_ang_vel,
                noise.base_ang_vel,
                scale.base_ang_vel,
                keys[1],
                noise_scale=feedback_noise_scale,
            ),
            self._observation_component(
                projected_gravity,
                noise.projected_gravity,
                scale.projected_gravity,
                keys[2],
                noise_scale=feedback_noise_scale,
            ),
        ]
        if self.command_config.enabled:
            parts.append(
                self._observation_component(
                    command, noise.command, scale.command, keys[3]
                )
            )
            parts.append(
                self._observation_component(
                    gait_phase_features(state.gait_phase, command, self.command_config),
                    noise.gait_phase,
                    scale.gait_phase,
                    keys[4],
                )
            )
        parts.extend(
            (
                self._observation_component(
                    joint_pos,
                    noise.joint_pos,
                    scale.joint_pos,
                    keys[5],
                    noise_scale=joint_position_noise_scale,
                ),
                self._observation_component(
                    joint_vel,
                    noise.joint_vel,
                    scale.joint_vel,
                    keys[6],
                    noise_scale=feedback_noise_scale,
                ),
                self._observation_component(
                    last_action, noise.last_action, scale.last_action, keys[7]
                ),
            )
        )
        if observation.privileged:
            parts.extend(
                (
                    self._observation_component(
                        base_lin_vel,
                        noise.base_lin_vel,
                        scale.base_lin_vel,
                        keys[8],
                    ),
                    self._observation_component(
                        base_height, noise.base_height, scale.base_height, keys[9]
                    ),
                    self._observation_component(
                        actuator_force,
                        noise.actuator_force,
                        scale.actuator_force,
                        keys[10],
                    ),
                )
            )
        return jnp.concatenate(parts)

    def _actor_feedback(self, state: BelloState) -> jax.Array:
        policy_indices = jnp.asarray(self.model.policy_ctrl_indices)
        qpos_indices = jnp.asarray(self.model.actuator_qpos_indices)[policy_indices]
        qvel_indices = jnp.asarray(self.model.actuator_qvel_indices)[policy_indices]
        joint_position = (
            state.data.qpos[qpos_indices]
            - jnp.asarray(self.model.default_joint_target)[policy_indices]
        )
        return jnp.concatenate(
            (
                self._imu_linear_acceleration(state),
                self._imu_angular_velocity(state),
                self._link_projected_gravity(state, self.model.imu_body_name),
                joint_position,
                state.data.qvel[qvel_indices],
            )
        )

    def _unpack_actor_feedback(
        self, feedback: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        joint_start = 9
        joint_end = joint_start + self.action_size
        return (
            feedback[0:3],
            feedback[3:6],
            feedback[6:9],
            feedback[joint_start:joint_end],
            feedback[joint_end : joint_end + self.action_size],
        )

    def _delayed_actor_feedback(self, state: BelloState) -> jax.Array:
        if (
            not self.config.observation_low_pass_filter.enabled
            and not self.config.observation_latency.enabled
        ):
            return self._actor_feedback(state)
        return state.actor_feedback_history[state.actor_feedback_delay_steps]

    def _advance_actor_feedback_history(
        self,
        history: jax.Array,
        state: BelloState,
        rng: jax.Array | None = None,
    ) -> jax.Array:
        measured = self._actor_feedback(state)
        if self.config.observation_low_pass_filter.noise_before_filter:
            measured = self._noisy_actor_feedback(measured, rng)
        feedback = self._low_pass_actor_feedback(
            history[0], measured, self.control_timestep
        )
        return jnp.concatenate((feedback[None, :], history[:-1]))

    def _noisy_actor_feedback(
        self, measured: jax.Array, rng: jax.Array | None
    ) -> jax.Array:
        noise = self.config.observation.noise
        if rng is None or not noise.enabled:
            return measured
        keys = jax.random.split(rng, 5)
        parts = self._unpack_actor_feedback(measured)
        return jnp.concatenate(
            (
                self._add_uniform_noise(parts[0], noise.base_lin_acc, keys[0]),
                self._add_uniform_noise(parts[1], noise.base_ang_vel, keys[1]),
                self._add_uniform_noise(parts[2], noise.projected_gravity, keys[2]),
                self._add_uniform_noise(parts[3], noise.joint_pos, keys[3]),
                self._add_uniform_noise(parts[4], noise.joint_vel, keys[4]),
            )
        )

    def _low_pass_actor_feedback(
        self, previous: jax.Array, measured: jax.Array, timestep: float | jax.Array
    ) -> jax.Array:
        config = self.config.observation_low_pass_filter
        if not config.enabled:
            return measured
        alpha = 1.0 - jnp.exp(
            -2.0
            * jnp.pi
            * jnp.asarray(config.cutoff_hz, dtype=jnp.float32)
            * jnp.asarray(timestep, dtype=jnp.float32)
        )
        previous_parts = self._unpack_actor_feedback(previous)
        measured_parts = self._unpack_actor_feedback(measured)
        linear_acceleration = previous_parts[0] + alpha * (
            measured_parts[0] - previous_parts[0]
        )
        angular_velocity = previous_parts[1] + alpha * (
            measured_parts[1] - previous_parts[1]
        )
        projected_gravity = previous_parts[2] + alpha * (
            measured_parts[2] - previous_parts[2]
        )
        projected_gravity_norm = jnp.linalg.norm(projected_gravity)
        projected_gravity = jnp.where(
            projected_gravity_norm > jnp.finfo(projected_gravity.dtype).eps,
            projected_gravity / projected_gravity_norm,
            measured_parts[2],
        )
        joint_velocity = previous_parts[4] + alpha * (
            measured_parts[4] - previous_parts[4]
        )
        return jnp.concatenate(
            (
                linear_acceleration,
                angular_velocity,
                projected_gravity,
                measured_parts[3],
                joint_velocity,
            )
        )

    def _sample_observation_delay(self, rng: jax.Array | None) -> jax.Array:
        config = self.config.observation_latency
        if not config.enabled or rng is None:
            return jnp.asarray(0, dtype=jnp.int32)
        return jax.random.randint(
            rng,
            (),
            minval=config.delay_steps[0],
            maxval=config.delay_steps[1] + 1,
            dtype=jnp.int32,
        )

    def _observation_component(
        self,
        value: jax.Array,
        noise_range: tuple[float, float],
        scale: float,
        rng: jax.Array | None,
        *,
        noise_scale: float = 1.0,
    ) -> jax.Array:
        if rng is not None and noise_scale > 0.0:
            scaled_range = (
                noise_scale * noise_range[0],
                noise_scale * noise_range[1],
            )
            value = self._add_uniform_noise(value, scaled_range, rng)
        return value * jnp.asarray(scale, dtype=value.dtype)

    @staticmethod
    def _add_uniform_noise(
        value: jax.Array,
        noise_range: tuple[float, float],
        rng: jax.Array,
    ) -> jax.Array:
        return value + jax.random.uniform(
            rng,
            value.shape,
            minval=noise_range[0],
            maxval=noise_range[1],
            dtype=value.dtype,
        )

    def action_to_ctrl(
        self,
        action: jax.Array,
        joint_angle_offset: jax.Array | None = None,
    ) -> jax.Array:
        ctrl, _, _ = self._action_target_details(action, joint_angle_offset)
        return ctrl

    def _locomotion_right_axis(self) -> jax.Array:
        forward_x, forward_y, _ = self.model.robot.locomotion_forward_axis
        return jnp.asarray((forward_y, -forward_x, 0.0), dtype=jnp.float32)

    def _sagittal_reflection(self) -> jax.Array:
        forward = jnp.asarray(
            self.model.robot.locomotion_forward_axis, dtype=jnp.float32
        )
        right = self._locomotion_right_axis()
        up = jnp.asarray((0.0, 0.0, 1.0), dtype=jnp.float32)
        return jnp.outer(forward, forward) - jnp.outer(right, right) + jnp.outer(up, up)

    @cached_property
    def _imu_sagittal_reflection(self) -> np.ndarray:
        command_frame_name = self.model.robot.locomotion_command_frame_body_name
        if self.model.imu_body_name == command_frame_name:
            forward = np.asarray(
                self.model.robot.locomotion_forward_axis, dtype=np.float64
            )
            right = np.asarray((forward[1], -forward[0], 0.0), dtype=np.float64)
            up = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
            return (
                np.outer(forward, forward) - np.outer(right, right) + np.outer(up, up)
            ).astype(np.float32)

        data = mujoco.MjData(self.model.mj_model)
        data.qpos[:] = self.model.default_qpos
        data.qvel[:] = self.model.default_qvel
        mujoco.mj_forward(self.model.mj_model, data)
        command_frame_id = self._link_body_id(command_frame_name)
        imu_frame_id = self._link_body_id(self.model.imu_body_name)
        command_rotation = data.xmat[command_frame_id].reshape(3, 3)
        imu_rotation = data.xmat[imu_frame_id].reshape(3, 3)
        command_forward = np.asarray(
            self.model.robot.locomotion_forward_axis, dtype=np.float64
        )
        command_right = np.asarray(
            (command_forward[1], -command_forward[0], 0.0), dtype=np.float64
        )
        forward = imu_rotation.T @ command_rotation @ command_forward
        right = imu_rotation.T @ command_rotation @ command_right
        up = np.cross(right, forward)
        reflection = (
            np.outer(forward, forward) - np.outer(right, right) + np.outer(up, up)
        )
        reflection[np.abs(reflection) < 1.0e-7] = 0.0
        return reflection.astype(np.float32)

    def mirror_action(self, action: jax.Array) -> jax.Array:
        """Reflect logical policy actions across the robot's sagittal plane."""

        index_by_name = {
            name: index for index, name in enumerate(self.action_layout.names)
        }
        mirror_by_name = {
            mirror.joint_name: mirror
            for mirror in self.model.robot.locomotion_action_mirror
        }
        source_indices = [
            index_by_name[mirror_by_name[name].counterpart_joint_name]
            for name in self.action_layout.names
        ]
        signs = [mirror_by_name[name].sign for name in self.action_layout.names]
        return action[..., jnp.asarray(source_indices, dtype=jnp.int32)] * jnp.asarray(
            signs, dtype=action.dtype
        )

    def mirror_actor_observation(self, observation: jax.Array) -> jax.Array:
        """Reflect one or more stacked deployable actor observations."""

        frame_size = self.observation_size
        if observation.shape[-1] % frame_size != 0:
            raise ValueError(
                "stacked actor observation size must be divisible by one frame, got "
                f"{observation.shape[-1]} and {frame_size}"
            )
        frames = observation.reshape(
            observation.shape[:-1] + (observation.shape[-1] // frame_size, frame_size)
        )
        mirrored = frames
        offset = 0
        for name, size in self.observation_layout.components:
            value = frames[..., offset : offset + size]
            if name in {"imu_linear_acceleration", "projected_gravity"}:
                value = jnp.matmul(
                    value,
                    jnp.asarray(self._imu_sagittal_reflection, dtype=value.dtype).T,
                )
            elif name == "imu_angular_velocity":
                value = -jnp.matmul(
                    value,
                    jnp.asarray(self._imu_sagittal_reflection, dtype=value.dtype).T,
                )
            elif name == "command":
                planar = jnp.matmul(
                    value[..., :2], self._sagittal_reflection()[:2, :2].T
                )
                value = jnp.concatenate((planar, -value[..., 2:3]), axis=-1)
            elif name == "gait_phase":
                value = value * jnp.asarray((-1.0, -1.0, 1.0), dtype=value.dtype)
            elif name in {"joint_position", "joint_velocity", "last_action"}:
                value = self.mirror_action(value)
            else:
                raise ValueError(
                    f"unsupported actor observation symmetry component: {name!r}"
                )
            mirrored = mirrored.at[..., offset : offset + size].set(value)
            offset += size
        return mirrored.reshape(observation.shape)

    def ppo_action_diagnostics(self, env_action: jax.Array) -> dict[str, jax.Array]:
        flat_action = env_action.reshape((-1, self.action_size))
        num_steps, num_envs = env_action.shape[:2]
        joint_angle_offset = self._joint_angle_offset_batch(num_envs)
        flat_joint_angle_offset = jnp.broadcast_to(
            joint_angle_offset[None, ...],
            (num_steps, *joint_angle_offset.shape),
        ).reshape((-1, joint_angle_offset.shape[-1]))
        ctrl = jax.vmap(self.action_to_ctrl)(flat_action, flat_joint_angle_offset)
        policy_ctrl_indices = jnp.asarray(self.model.policy_ctrl_indices)
        policy_ctrl = ctrl[:, policy_ctrl_indices]
        ctrlrange = jnp.asarray(self.model.mj_model.actuator_ctrlrange)[
            policy_ctrl_indices
        ]
        ctrl_low = ctrlrange[:, 0]
        ctrl_high = ctrlrange[:, 1]
        ctrl_margin = jnp.minimum(policy_ctrl - ctrl_low, ctrl_high - policy_ctrl)
        saturated = (policy_ctrl <= ctrl_low + 1.0e-6) | (
            policy_ctrl >= ctrl_high - 1.0e-6
        )
        return {
            "ctrl_saturation_fraction": saturated.astype(jnp.float32).mean(),
            "ctrl_margin_min": jnp.min(ctrl_margin),
        }

    def action_to_joint_target(
        self,
        action: jax.Array,
        joint_angle_offset: jax.Array | None = None,
    ) -> jax.Array:
        """Map bounded policy actions to logical joint targets."""
        if joint_angle_offset is None:
            joint_angle_offset = jnp.asarray(self.model.joint_angle_offset[0])
        action = jnp.clip(
            jnp.asarray(action, dtype=jnp.float32),
            -self.action_clip,
            self.action_clip,
        )
        policy_indices = jnp.asarray(self.model.policy_ctrl_indices)
        all_target_low = jnp.asarray(self.model.joint_target_low)
        all_target_high = jnp.asarray(self.model.joint_target_high)
        default_target = jnp.clip(
            jnp.asarray(self.model.default_joint_target) + joint_angle_offset,
            all_target_low,
            all_target_high,
        )
        target_low = all_target_low[policy_indices]
        target_high = all_target_high[policy_indices]
        policy_default = default_target[policy_indices]
        normalized_action = action / self.action_clip
        target_span = jnp.where(
            normalized_action >= 0.0,
            target_high - policy_default,
            policy_default - target_low,
        )
        return default_target.at[policy_indices].set(
            policy_default + target_span * normalized_action
        )

    def _action_target_details(
        self,
        action: jax.Array,
        joint_angle_offset: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        requested_joint_target = self.action_to_joint_target(action, joint_angle_offset)
        policy_indices = jnp.asarray(self.model.policy_ctrl_indices)
        joint_target = jnp.clip(
            requested_joint_target,
            jnp.asarray(self.model.joint_target_low),
            jnp.asarray(self.model.joint_target_high),
        )
        requested_ctrl = jnp.matmul(
            jnp.asarray(self.model.actuator_coordinate_transform),
            joint_target,
            precision=jax.lax.Precision.HIGHEST,
        )
        ctrlrange = jnp.asarray(self.model.mj_model.actuator_ctrlrange)
        ctrl = jnp.clip(requested_ctrl, ctrlrange[:, 0], ctrlrange[:, 1])
        joint_saturation = jnp.mean(
            (
                jnp.abs(
                    joint_target[policy_indices]
                    - requested_joint_target[policy_indices]
                )
                > 1.0e-6
            ).astype(jnp.float32)
        )
        motor_saturation = jnp.mean(
            (
                jnp.abs(ctrl[policy_indices] - requested_ctrl[policy_indices]) > 1.0e-6
            ).astype(jnp.float32)
        )
        return ctrl, joint_saturation, motor_saturation

    def _joint_angle_offset_batch(self, batch_size: int) -> jax.Array:
        offset = jnp.asarray(self.model.joint_angle_offset, dtype=jnp.float32)
        if offset.shape[0] == 1:
            return jnp.broadcast_to(offset, (batch_size, offset.shape[1]))
        if offset.shape[0] != batch_size:
            raise ValueError(
                "model randomization batch does not match environment batch: "
                f"{offset.shape[0]} != {batch_size}"
            )
        return offset

    def _resolve_command(
        self,
        command: jax.Array | None,
        rng: jax.Array | None,
    ) -> jax.Array:
        if not self.command_config.enabled:
            return jnp.zeros(3, dtype=jnp.float32)
        if command is not None:
            command = jnp.asarray(command, dtype=jnp.float32)
            if command.shape != (3,):
                raise ValueError(f"command must have shape (3,), got {command.shape}")
            return command
        if rng is not None:
            return self.command_config.sample(rng)
        return jnp.zeros(3, dtype=jnp.float32)

    def _resolve_batch_command(
        self,
        batch_size: int,
        command: jax.Array | None,
        rng: jax.Array | None,
    ) -> jax.Array:
        if not self.command_config.enabled:
            return jnp.zeros((batch_size, 3), dtype=jnp.float32)
        if command is None:
            if rng is not None:
                return self.command_config.sample_stratified_batch(rng, batch_size)
            return jnp.zeros((batch_size, 3), dtype=jnp.float32)
        command = jnp.asarray(command, dtype=jnp.float32)
        if command.shape == (3,):
            return jnp.repeat(command[None, :], batch_size, axis=0)
        if command.shape != (batch_size, 3):
            raise ValueError(
                "batch command must have shape (3,) or "
                f"({batch_size}, 3), got {command.shape}"
            )
        return command

    def _maybe_resample_batch_command_and_phase(
        self,
        state: BelloState,
        rng: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        resample_steps = max(
            1, int(round(self.command_config.resample_seconds / self.control_timestep))
        )
        if rng is None or not self.command_config.enabled:
            return (
                state.command,
                state.gait_phase,
                state.standing_transition_pending,
                jnp.minimum(state.command_resample_step_count, resample_steps),
            )
        resample = state.command_resample_step_count >= resample_steps
        sampled = self._resample_batch_command(
            state.command, int(state.command.shape[0]), rng
        )
        pending_command = jnp.where(
            state.standing_transition_pending[:, None],
            jnp.zeros_like(state.command),
            state.command,
        )
        requested_command = jnp.where(resample[:, None], sampled, pending_command)
        command, gait_phase, standing_transition_pending = (
            resolve_gait_command_transition(
                state.command,
                requested_command,
                state.gait_phase,
                state.gait_phase,
                state.standing_transition_pending,
                self.command_config,
            )
        )
        return (
            command,
            gait_phase,
            standing_transition_pending,
            jnp.where(resample, 0, state.command_resample_step_count),
        )

    def _resample_batch_command(
        self,
        command: jax.Array,
        batch_size: int,
        rng: jax.Array | None,
    ) -> jax.Array:
        if rng is None:
            return command
        if not self.command_config.enabled:
            return jnp.zeros_like(command)
        if self.command_config.standing_transitions_enabled:
            return self.command_config.sample_stratified_batch(rng, batch_size)
        return self._reset_batch_command(command, batch_size, rng)

    def _reset_batch_command(
        self,
        command: jax.Array,
        batch_size: int,
        rng: jax.Array | None,
    ) -> jax.Array:
        if rng is None:
            return command
        if not self.command_config.enabled:
            return jnp.zeros_like(command)
        standing = command_is_standing(command, self.command_config)
        moving = self.command_config.sample_moving(rng, (batch_size,))
        return jnp.where(standing[:, None], jnp.zeros_like(moving), moving)

    def _maybe_apply_random_push_batch(
        self,
        data: mjx.Data,
        step_count: jax.Array,
        rng: jax.Array | None,
    ) -> tuple[mjx.Data, jax.Array, jax.Array]:
        batch_size = int(step_count.shape[0])
        push_applied = jnp.zeros((batch_size,), dtype=bool)
        push_velocity = jnp.zeros((batch_size, 2), dtype=jnp.float32)
        if rng is None or not self.config.random_push.enabled:
            return data, push_applied, push_velocity

        interval_steps = max(
            1,
            int(
                round(self.config.random_push.interval_seconds / self.control_timestep)
            ),
        )
        push_applied = (step_count > 0) & (step_count % interval_steps == 0)
        x_key, y_key = jax.random.split(rng)
        push_velocity = jnp.stack(
            [
                jax.random.uniform(
                    x_key,
                    (batch_size,),
                    minval=self.config.random_push.lin_vel_x[0],
                    maxval=self.config.random_push.lin_vel_x[1],
                    dtype=jnp.float32,
                ),
                jax.random.uniform(
                    y_key,
                    (batch_size,),
                    minval=self.config.random_push.lin_vel_y[0],
                    maxval=self.config.random_push.lin_vel_y[1],
                    dtype=jnp.float32,
                ),
            ],
            axis=1,
        )
        root_qvel = int(self.model.root_qvel_adr)
        qvel_xy = jnp.where(
            push_applied[:, None],
            push_velocity,
            data.qvel[:, root_qvel : root_qvel + 2],
        )
        data = data.replace(
            qvel=data.qvel.at[:, root_qvel : root_qvel + 2].set(qvel_xy)
        )
        push_velocity = jnp.where(
            push_applied[:, None], push_velocity, jnp.zeros_like(push_velocity)
        )
        return data, push_applied, push_velocity

    @property
    def control_timestep(self) -> float:
        return self.decimation * float(self.model.mj_model.opt.timestep)

    def _frame_terms(
        self, state: BelloState, world_id: jax.Array | int = 0
    ) -> tuple[jax.Array, ...]:
        data = state.data
        policy_idx = jnp.asarray(self.model.policy_ctrl_indices)
        qpos_idx = jnp.asarray(self.model.actuator_qpos_indices)[policy_idx]
        qvel_idx = jnp.asarray(self.model.actuator_qvel_indices)[policy_idx]
        joint_pos = (
            data.qpos[qpos_idx]
            - jnp.asarray(self.model.default_joint_target)[policy_idx]
        )
        joint_vel = data.qvel[qvel_idx]
        actuator_force = data.actuator_force[policy_idx]
        base_lin_vel = self._link_lin_vel(state, self.root_link_name)
        imu_lin_acc = self._imu_linear_acceleration(state)
        imu_ang_vel = self._imu_angular_velocity(state)
        base_height = jnp.asarray(
            [self._link_clearance(state, self.root_link_name, world_id)],
            dtype=jnp.float32,
        )
        projected_gravity = self._link_projected_gravity(
            state, self.model.imu_body_name
        )
        return (
            base_lin_vel,
            imu_lin_acc,
            imu_ang_vel,
            projected_gravity,
            base_height,
            state.command,
            joint_pos,
            joint_vel,
            actuator_force,
            state.last_action,
        )

    def _actuator_indices_named(self, *suffixes: str) -> np.ndarray:
        indices = [
            index
            for index, name in enumerate(self.model.actuator_joint_names)
            if name.endswith(suffixes)
        ]
        return np.asarray(indices, dtype=np.int32)

    def _leg_joint_body_ids(self) -> np.ndarray:
        actuator_indices = self._actuator_indices_named(
            "hip_pitch_joint",
            "hip_roll_joint",
            "hip_yaw_joint",
            "knee_joint",
            "ankle_pitch_joint",
            "ankle_roll_joint",
        )
        policy_indices = np.asarray(self.model.policy_ctrl_indices, dtype=np.int32)
        actuator_indices = actuator_indices[
            np.isin(actuator_indices, policy_indices, assume_unique=True)
        ]
        joint_ids = np.asarray(self.model.actuator_joint_ids)[actuator_indices]
        body_ids = np.asarray(self.model.mj_model.jnt_bodyid)[joint_ids]
        if body_ids.size == 0 or np.unique(body_ids).size != body_ids.size:
            raise ValueError(
                "locomotion joint-force reward requires unique policy-controlled "
                "hip, knee, and ankle child bodies"
            )
        return body_ids.astype(np.int32)

    def _leg_joint_force(self, state: BelloState) -> jax.Array:
        body_ids = jnp.asarray(self._leg_joint_body_ids())
        linear_force = state.data._impl.cfrc_int[body_ids, 3:]
        return jnp.linalg.norm(linear_force, axis=1)

    def _robot_weight(self) -> jax.Array:
        total_mass = float(np.sum(self.model.mj_model.body_mass))
        gravity = float(np.linalg.norm(self.model.mj_model.opt.gravity))
        return jnp.asarray(total_mass * gravity, dtype=jnp.float32)

    def _joint_pose_weights(self) -> jax.Array:
        weights = np.zeros(self.model.mj_model.nu, dtype=np.float32)
        weights[np.asarray(self.model.policy_ctrl_indices)] = 1.0
        low_weight_indices = self._actuator_indices_named(
            "hip_pitch_joint", "knee_joint"
        )
        weights[low_weight_indices] = 0.01
        return jnp.asarray(weights)

    def _action_smoothness_weights(self) -> jax.Array:
        names = self.action_layout.names
        return jnp.asarray(
            [
                (
                    self.reward_config.smoothness.ankle_action_weight
                    if "_ankle_" in name
                    else self.reward_config.smoothness.waist_action_weight
                    if name in {"waist_yaw_joint", "waist_roll_joint"}
                    else 1.0
                )
                for name in names
            ],
            dtype=jnp.float32,
        )

    def requested_actuator_force(self, data: mjx.Data) -> jax.Array:
        """Return the unclipped force requested by the position servos."""

        model = self.model.mj_model
        gain = jnp.asarray(model.actuator_gainprm[:, 0], dtype=jnp.float32)
        bias_position = jnp.asarray(model.actuator_biasprm[:, 1], dtype=jnp.float32)
        bias_velocity = jnp.asarray(model.actuator_biasprm[:, 2], dtype=jnp.float32)
        return (
            gain * data.ctrl
            + bias_position * data.actuator_length
            + bias_velocity * data._impl.actuator_velocity
        )

    def _all_actuator_joint_terms(
        self, state: BelloState
    ) -> tuple[jax.Array, jax.Array]:
        qpos_idx = jnp.asarray(self.model.actuator_qpos_indices)
        qvel_idx = jnp.asarray(self.model.actuator_qvel_indices)
        joint_pos = (
            state.data.qpos[qpos_idx] - jnp.asarray(self.model.default_qpos)[qpos_idx]
        )
        joint_vel = state.data.qvel[qvel_idx]
        return joint_pos, joint_vel

    def _link_body_id(self, link_name: str) -> int:
        return int(
            mujoco.mj_name2id(self.model.mj_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
        )

    def _link_quat(self, state: BelloState, link_name: str) -> jax.Array:
        if link_name == self.model.robot.root_body_name:
            root_qpos = int(self.model.root_qpos_adr)
            return state.data.qpos[root_qpos + 3 : root_qpos + 7]
        return state.data.xquat[self._link_body_id(link_name)]

    def _link_position(self, state: BelloState, link_name: str) -> jax.Array:
        if link_name == self.model.robot.root_body_name:
            root_qpos = int(self.model.root_qpos_adr)
            return state.data.qpos[root_qpos : root_qpos + 3]
        return state.data.xpos[self._link_body_id(link_name)]

    def _link_clearance(
        self,
        state: BelloState,
        link_name: str,
        world_id: jax.Array | int = 0,
    ) -> jax.Array:
        position = self._link_position(state, link_name)
        return position[2] - self._terrain_height(position[:2], world_id)

    def _link_lin_vel(self, state: BelloState, link_name: str) -> jax.Array:
        return self._link_body_lin_vel(state, link_name)

    def _link_body_lin_vel(self, state: BelloState, link_name: str) -> jax.Array:
        if link_name == self.model.robot.root_body_name:
            root_qpos = int(self.model.root_qpos_adr)
            root_qvel = int(self.model.root_qvel_adr)
            return quat_rotate_inverse(
                state.data.qpos[root_qpos + 3 : root_qpos + 7],
                state.data.qvel[root_qvel : root_qvel + 3],
            )
        return quat_rotate_inverse(
            self._link_quat(state, link_name),
            self._link_world_lin_vel(state, link_name),
        )

    def _link_body_lin_acc(self, state: BelloState, link_name: str) -> jax.Array:
        body_id = self._link_body_id(link_name)
        root_body_id = int(self.model.mj_model.body_rootid[body_id])
        offset = state.data.xpos[body_id] - state.data.subtree_com[root_body_id]
        cacc = state.data._impl.cacc[body_id]
        cvel = state.data.cvel[body_id]
        world_lin_acc = cacc[3:6] - jnp.cross(offset, cacc[:3])
        world_lin_vel = cvel[3:6] - jnp.cross(offset, cvel[:3])
        world_lin_acc = world_lin_acc + jnp.cross(cvel[:3], world_lin_vel)
        return quat_rotate_inverse(self._link_quat(state, link_name), world_lin_acc)

    def _imu_linear_acceleration(self, state: BelloState) -> jax.Array:
        """MuJoCo accelerometer-equivalent output at the configured body origin."""

        return self._link_body_lin_acc(state, self.model.imu_body_name)

    def _imu_angular_velocity(self, state: BelloState) -> jax.Array:
        """MuJoCo gyro-equivalent output in the configured IMU body frame."""

        return self._link_body_ang_vel(state, self.model.imu_body_name)

    def _link_world_lin_vel(self, state: BelloState, link_name: str) -> jax.Array:
        if link_name == self.model.robot.root_body_name:
            root_qvel = int(self.model.root_qvel_adr)
            return state.data.qvel[root_qvel : root_qvel + 3]
        return self._body_world_lin_vel(
            state, np.asarray([self._link_body_id(link_name)], dtype=np.int32)
        )[0]

    def _body_world_lin_vel(self, state: BelloState, body_ids: np.ndarray) -> jax.Array:
        body_ids = np.asarray(body_ids, dtype=np.int32)
        root_body_ids = np.asarray(
            self.model.mj_model.body_rootid[body_ids], dtype=np.int32
        )
        body_ids_jax = jnp.asarray(body_ids)
        cvel = state.data.cvel[body_ids_jax]
        offset = (
            state.data.xpos[body_ids_jax]
            - state.data.subtree_com[jnp.asarray(root_body_ids)]
        )
        return cvel[:, 3:6] - jnp.cross(offset, cvel[:, :3])

    def _link_yaw_frame_lin_vel(
        self, state: BelloState, link_name: str, frame_link_name: str | None = None
    ) -> jax.Array:
        frame_link_name = link_name if frame_link_name is None else frame_link_name
        return quat_rotate_inverse(
            yaw_quat(self._link_quat(state, frame_link_name)),
            self._link_world_lin_vel(state, link_name),
        )

    def _link_body_ang_vel(self, state: BelloState, link_name: str) -> jax.Array:
        if link_name == self.model.robot.root_body_name:
            root_qvel = int(self.model.root_qvel_adr)
            return state.data.qvel[root_qvel + 3 : root_qvel + 6]
        body_id = self._link_body_id(link_name)
        return quat_rotate_inverse(
            self._link_quat(state, link_name), state.data.cvel[body_id, :3]
        )

    def _link_projected_gravity(self, state: BelloState, link_name: str) -> jax.Array:
        return quat_rotate_inverse(
            self._link_quat(state, link_name),
            jnp.asarray([0.0, 0.0, -1.0], dtype=jnp.float32),
        )

    def done_flags(
        self, state: BelloState, world_id: jax.Array | int = 0
    ) -> dict[str, jax.Array]:
        config = self.done_config
        bad_height = (
            self._link_clearance(state, self._done_link(config.base_height), world_id)
            < config.base_height.min_height
            if config.base_height.enabled
            else jnp.asarray(False)
        )
        if config.bad_orientation.enabled:
            projected_gravity = self._link_projected_gravity(
                state, self._done_link(config.bad_orientation)
            )
            bad_orientation = (
                projected_gravity_tilt_angle(projected_gravity)
                >= config.bad_orientation.limit_angle
            )
        else:
            bad_orientation = jnp.asarray(False)
        bad_pose = bad_height | bad_orientation
        timeout = (
            state.step_count >= self.max_episode_steps
            if config.timeout.enabled
            else jnp.asarray(False)
        )
        position_fields = (state.data.qpos, state.data.xpos, state.data.xquat)
        velocity_fields = (state.data.qvel, state.data.cvel)
        acceleration_fields = (state.data.qacc, state.data._impl.cacc)
        actuator_force_fields = (state.data.actuator_force,)
        constraint_force_fields = (
            state.data.qfrc_constraint,
            state.data._impl.efc__force,
        )
        all_fields = (
            *position_fields,
            *velocity_fields,
            *acceleration_fields,
            *actuator_force_fields,
            *constraint_force_fields,
        )
        finite = jnp.all(
            jnp.stack([jnp.all(jnp.isfinite(value)) for value in all_fields])
        )
        nonfinite = ~finite if config.nonfinite.enabled else jnp.asarray(False)
        safety = self.config.numerical_safety
        unstable = finite & (
            jnp.any(
                jnp.stack(
                    [
                        jnp.any(jnp.abs(value) > safety.max_abs_position)
                        for value in position_fields
                    ]
                )
            )
            | jnp.any(
                jnp.stack(
                    [
                        jnp.any(jnp.abs(value) > safety.max_abs_velocity)
                        for value in velocity_fields
                    ]
                )
            )
            | jnp.any(
                jnp.stack(
                    [
                        jnp.any(jnp.abs(value) > safety.max_abs_acceleration)
                        for value in acceleration_fields
                    ]
                )
            )
            | jnp.any(
                jnp.stack(
                    [
                        jnp.any(jnp.abs(value) > safety.max_abs_actuator_force)
                        for value in actuator_force_fields
                    ]
                )
            )
            | jnp.any(
                jnp.stack(
                    [
                        jnp.any(jnp.abs(value) > safety.max_abs_constraint_force)
                        for value in constraint_force_fields
                    ]
                )
            )
        )
        terminal = bad_pose | nonfinite | unstable
        done = terminal | timeout
        return {
            "done/any": done,
            "done/terminal": terminal,
            "done/base_height": bad_height,
            "done/bad_orientation": bad_orientation,
            "done/timeout": timeout,
            "done/nonfinite": nonfinite,
            "done/unstable": unstable,
        }

    def _base_height(
        self, state: BelloState, world_id: jax.Array | int = 0
    ) -> jax.Array:
        return self._link_clearance(state, self.root_link_name, world_id)

    def reward(
        self,
        state: BelloState,
        next_state: BelloState,
        action: jax.Array,
        requested_action: jax.Array,
        done: jax.Array,
        foot_contact: jax.Array,
        foot_touchdown: jax.Array,
        foot_normal_impulse: jax.Array,
        foot_peak_normal_force: jax.Array,
        touchdown_foot_linear_velocity: jax.Array,
        requested_actuator_force: jax.Array,
        foot_linear_velocity: jax.Array,
        foot_angular_speed: jax.Array,
        undesired_contact: jax.Array,
        self_collision: jax.Array,
        joint_angle_offset: jax.Array | None = None,
        world_id: jax.Array | int = 0,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        command = next_state.command
        qpos_idx = jnp.asarray(self.model.actuator_qpos_indices)
        joint_pos_rel, _ = self._all_actuator_joint_terms(next_state)
        lin_vel_xy = self._link_yaw_frame_lin_vel(
            next_state,
            self._reward_link("track_lin_vel_xy"),
            self._reward_frame_link("track_lin_vel_xy"),
        )[:2]
        yaw_vel_z = next_state.tracking_ang_vel_z
        lin_vel_z = self._link_body_lin_vel(next_state, self._reward_link("lin_vel_z"))[
            2
        ]
        ang_vel_xy = self._link_body_ang_vel(
            next_state, self._reward_link("ang_vel_xy")
        )[:2]
        projected_gravity = self._link_projected_gravity(
            next_state, self._reward_link("flat_orientation")
        )
        base_height = self._link_clearance(
            next_state,
            self._reward_link("base_height"),
            world_id,
        )
        return compute_locomotion_reward(
            self.reward_config,
            RewardInputs(
                command=command,
                lin_vel_xy=lin_vel_xy,
                yaw_vel_z=yaw_vel_z,
                lin_vel_z=lin_vel_z,
                ang_vel_xy=ang_vel_xy,
                projected_gravity=projected_gravity,
                base_height=base_height,
                joint_pos=next_state.data.qpos[qpos_idx],
                joint_pos_rel=joint_pos_rel,
                action=action,
                requested_action=requested_action,
                last_action=state.last_action,
                last_action_delta=state.last_action_delta,
                action_smoothness_weights=self._action_smoothness_weights(),
                actuator_force=next_state.data.actuator_force,
                requested_actuator_force=requested_actuator_force,
                actuator_peak_force=jnp.asarray(self.model.actuator_peak_force),
                joint_force=self._leg_joint_force(next_state),
                robot_weight=self._robot_weight(),
                joint_limit_low=self._actuator_joint_limit_low(),
                joint_limit_high=self._actuator_joint_limit_high(),
                policy_joint_indices=jnp.asarray(self.model.policy_ctrl_indices),
                action_target_default=self._policy_action_target_default(
                    joint_angle_offset
                ),
                action_target_low=self._policy_action_target_low(),
                action_target_high=self._policy_action_target_high(),
                foot_contact=foot_contact,
                foot_touchdown=foot_touchdown,
                foot_normal_impulse=foot_normal_impulse,
                foot_peak_normal_force=foot_peak_normal_force,
                touchdown_foot_linear_velocity=touchdown_foot_linear_velocity,
                foot_linear_velocity=foot_linear_velocity,
                foot_angular_speed=foot_angular_speed,
                foot_heading_error=self._foot_heading_error(next_state),
                foot_height=self._foot_height(next_state, world_id),
                gait_active=command_is_moving(command, self.command_config),
                desired_foot_contact=desired_gait_foot_contact(
                    next_state.gait_phase, self.config.gait
                ),
                joint_pose_weights=self._joint_pose_weights(),
                hip_joint_indices=self._actuator_indices_named(
                    "hip_roll_joint", "hip_yaw_joint"
                ),
                knee_joint_indices=self._actuator_indices_named("knee_joint"),
                undesired_contact=undesired_contact,
                self_collision=self_collision,
                done=done,
                control_timestep=jnp.asarray(self.control_timestep, dtype=jnp.float32),
                action_clip_limit=jnp.asarray(self.action_clip, dtype=jnp.float32),
            ),
            self.command_config,
        )

    def _raw_tracking_ang_vel_z(self, state: BelloState) -> jax.Array:
        return self._link_body_ang_vel(state, self._reward_link("track_ang_vel_z"))[2]

    def filter_tracking_ang_vel_z(
        self, previous: jax.Array, measured: jax.Array
    ) -> jax.Array:
        """Low-pass the measured yaw rate used by the tracking reward."""

        alpha = 1.0 - jnp.exp(
            -2.0
            * jnp.pi
            * jnp.asarray(
                self.reward_config.tracking.ang_vel_z_lpf_cutoff_hz,
                dtype=jnp.float32,
            )
            * jnp.asarray(self.control_timestep, dtype=jnp.float32)
        )
        return previous + alpha * (measured - previous)

    def sanitize_nonfinite_reward(
        self,
        reward: jax.Array,
        reward_info: dict[str, jax.Array],
        done_flags: dict[str, jax.Array],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        invalid_physics = done_flags["done/nonfinite"] | done_flags["done/unstable"]
        limit = jnp.asarray(
            self.config.numerical_safety.max_abs_reward_term, dtype=jnp.float32
        )
        sanitized = {}
        for name, value in reward_info.items():
            value = jnp.nan_to_num(value, nan=0.0, posinf=limit, neginf=-limit)
            value = jnp.clip(value, -limit, limit)
            if name == "reward/termination":
                sanitized[name] = value
            else:
                sanitized[name] = jnp.where(
                    invalid_physics, jnp.zeros_like(value), value
                )
        reward = sum(sanitized.values(), jnp.zeros_like(reward))
        return reward, sanitized

    def debounce_foot_contact(
        self, state: BelloState, raw_foot_contact: jax.Array, done: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        raw_contact = raw_foot_contact.astype(bool)
        previous_contact = state.last_foot_contact.astype(bool)
        changed = raw_contact != previous_contact
        on_threshold = jnp.asarray(
            self.config.contact.foot_contact_on_seconds, dtype=jnp.float32
        )
        off_threshold = jnp.asarray(
            self.config.contact.foot_contact_off_seconds, dtype=jnp.float32
        )
        threshold = jnp.where(raw_contact, on_threshold, off_threshold)
        next_debounce_time = jnp.where(
            changed,
            state.foot_contact_debounce_time + self.control_timestep,
            jnp.zeros_like(state.foot_contact_debounce_time),
        )
        accepted = changed & (next_debounce_time + 1e-6 >= threshold)
        filtered_contact = jnp.where(accepted, raw_contact, previous_contact)
        next_debounce_time = jnp.where(
            accepted | ~changed,
            jnp.zeros_like(next_debounce_time),
            next_debounce_time,
        )
        return (
            filtered_contact.astype(jnp.float32),
            jnp.where(done, jnp.zeros_like(next_debounce_time), next_debounce_time),
        )

    def _reward_link(self, term_name: str) -> str:
        link = getattr(self.reward_config.links, term_name)
        if self.reward_config.enabled(term_name) and link is not None:
            return link
        return self.root_link_name

    def _reward_frame_link(self, term_name: str) -> str:
        velocity_link = self._reward_link(term_name)
        frame_link = (
            self.reward_config.links.track_lin_vel_xy_frame
            if term_name == "track_lin_vel_xy"
            else None
        )
        if self.reward_config.enabled(term_name) and frame_link is not None:
            return frame_link
        if term_name == "track_lin_vel_xy":
            return self.model.robot.locomotion_command_frame_body_name
        return velocity_link

    def _done_link(self, term) -> str:
        if term.enabled and term.link is not None:
            return term.link
        return self.root_link_name

    def _contact_terms(
        self, state: BelloState, world_id: jax.Array
    ) -> tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
    ]:
        foot_body_ids = jnp.asarray(self.model.foot_body_ids)
        foot_count = foot_body_ids.shape[0]
        if (
            self.model.terrain_geom_ids.size == 0
            and self.model.robot_geom_ids.size == 0
        ):
            return (
                jnp.zeros(foot_count, dtype=jnp.float32),
                jnp.zeros(foot_count, dtype=jnp.float32),
                self._foot_world_linear_velocity(state),
                self._foot_angular_speed(state),
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(0.0, dtype=jnp.float32),
            )

        impl = state.data._impl
        contact_geom = jnp.asarray(impl.contact__geom, dtype=jnp.int32)
        contact_geom = jnp.clip(contact_geom, 0, self.model.mj_model.ngeom - 1)
        nacon = jnp.ravel(jnp.asarray(impl.nacon, dtype=jnp.int32))[0]
        contact_worldid = jnp.asarray(impl.contact__worldid, dtype=jnp.int32)
        active = (jnp.arange(contact_geom.shape[0], dtype=jnp.int32) < nacon) & (
            contact_worldid == jnp.asarray(world_id, dtype=jnp.int32)
        )
        contact_efc_address = jnp.asarray(impl.contact__efc_address, dtype=jnp.int32)
        normal_efc_address = contact_efc_address[:, 0]
        efc_force = jnp.asarray(impl.efc__force, dtype=jnp.float32)
        clipped_address = jnp.clip(normal_efc_address, 0, efc_force.shape[0] - 1)
        normal_force = jnp.where(
            normal_efc_address >= 0,
            jnp.abs(efc_force[clipped_address]),
            jnp.zeros_like(normal_efc_address, dtype=jnp.float32),
        )
        foot_forceful_contact = normal_force > jnp.asarray(
            self.config.contact.foot_contact_force_threshold, dtype=jnp.float32
        )
        undesired_forceful_contact = normal_force > jnp.asarray(
            self.config.contact.undesired_contact_force_threshold, dtype=jnp.float32
        )
        self_collision_forceful_contact = normal_force > jnp.asarray(
            self.config.contact.self_collision_force_threshold, dtype=jnp.float32
        )

        geom_body_ids = jnp.asarray(self.model.geom_body_ids)
        contact_body = geom_body_ids[contact_geom]
        terrain_contact = pair_contains(contact_geom, self.model.terrain_geom_ids)
        self_collision_slot = (
            active
            & self_collision_forceful_contact
            & pair_both_contained(contact_geom, self.model.robot_geom_ids)
        )
        foot_membership = foot_terrain_contact_membership(
            contact_geom,
            self.model.geom_body_ids,
            self.model.foot_body_ids,
            self.model.foot_geom_ids,
            self.model.terrain_geom_ids,
        )
        foot_terrain_contact_slot = active & jnp.any(foot_membership, axis=1)
        foot_contact_slot = foot_terrain_contact_slot & foot_forceful_contact
        foot_contact = jnp.any(
            foot_contact_slot[:, None] & foot_membership,
            axis=0,
        ).astype(jnp.float32)
        foot_normal_force = jnp.sum(
            jnp.where(
                foot_terrain_contact_slot[:, None] & foot_membership,
                normal_force[:, None],
                0.0,
            ),
            axis=0,
        )
        undesired_contact_slot = (
            active
            & terrain_contact
            & undesired_forceful_contact
            & pair_contains(contact_geom, self.model.nonfoot_robot_geom_ids)
        )
        undesired_contact = jnp.sum(
            jnp.any(
                undesired_contact_slot[:, None]
                & pair_contains_each(contact_body, self.model.nonfoot_robot_body_ids),
                axis=0,
            ).astype(jnp.float32)
        )
        self_collision = jnp.sum(self_collision_slot.astype(jnp.float32))
        return (
            foot_contact,
            foot_normal_force,
            self._foot_world_linear_velocity(state),
            self._foot_angular_speed(state),
            undesired_contact,
            self_collision,
        )

    def _foot_touchdown(
        self, state: BelloState, raw_foot_contact: jax.Array
    ) -> jax.Array:
        first_changed_sample = state.foot_contact_debounce_time <= 0.0
        return (
            (state.step_count > 0)
            & raw_foot_contact.astype(bool)
            & ~state.last_foot_contact.astype(bool)
            & first_changed_sample
        )

    def _foot_height(
        self, state: BelloState, world_id: jax.Array | int = 0
    ) -> jax.Array:
        sole_position = self._foot_sole_positions(state, world_id)
        return sole_position[:, 2] - self._terrain_height(
            sole_position[:, :2], world_id
        )

    def _foot_sole_positions(
        self, state: BelloState, world_id: jax.Array | int = 0
    ) -> jax.Array:
        """Collision support point with minimum local terrain clearance per foot."""

        geom_ids = jnp.asarray(self.model.foot_geom_ids, dtype=jnp.int32)
        geom_rotation = state.data.geom_xmat[geom_ids]
        geom_position = state.data.geom_xpos[geom_ids]
        terrain_up = self._terrain_surface_normal(geom_position[:, :2], world_id)
        local_down = jnp.einsum("gji,gj->gi", geom_rotation, -terrain_up)
        geom_size = jnp.asarray(self.model.mj_model.geom_size)[geom_ids]
        geom_type = jnp.asarray(self.model.mj_model.geom_type)[geom_ids]

        radial_direction = local_down[:, :2]
        radial_norm = jnp.linalg.norm(radial_direction, axis=1, keepdims=True)
        unit_radial = jnp.where(
            radial_norm > 1.0e-8,
            radial_direction / jnp.maximum(radial_norm, 1.0e-8),
            jnp.zeros_like(radial_direction),
        )
        sphere_support = geom_size[:, :1] * local_down
        capsule_support = sphere_support.at[:, 2].add(
            jnp.sign(local_down[:, 2]) * geom_size[:, 1]
        )
        ellipsoid_denominator = jnp.sqrt(
            jnp.sum((geom_size * local_down) ** 2, axis=1, keepdims=True)
        )
        ellipsoid_support = (
            geom_size**2 * local_down / jnp.maximum(ellipsoid_denominator, 1.0e-8)
        )
        cylinder_support = jnp.concatenate(
            [
                geom_size[:, :1] * unit_radial,
                (jnp.sign(local_down[:, 2]) * geom_size[:, 1])[:, None],
            ],
            axis=1,
        )
        support = sphere_support
        for shape, candidate in (
            (mujoco.mjtGeom.mjGEOM_CAPSULE, capsule_support),
            (mujoco.mjtGeom.mjGEOM_ELLIPSOID, ellipsoid_support),
            (mujoco.mjtGeom.mjGEOM_CYLINDER, cylinder_support),
        ):
            support = jnp.where((geom_type == int(shape))[:, None], candidate, support)
        support_world = geom_position + jnp.einsum("gij,gj->gi", geom_rotation, support)

        box_corner_signs = jnp.asarray(
            [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
            dtype=jnp.float32,
        )
        box_corners = box_corner_signs[None, :, :] * geom_size[:, None, :]
        box_corners_world = geom_position[:, None, :] + jnp.einsum(
            "gij,gkj->gki", geom_rotation, box_corners
        )
        box_corner_clearance = box_corners_world[:, :, 2] - self._terrain_height(
            box_corners_world[:, :, :2], world_id
        )
        lowest_box_corner = jnp.argmin(box_corner_clearance, axis=1)
        box_support_world = box_corners_world[
            jnp.arange(box_corners_world.shape[0]), lowest_box_corner
        ]
        support_world = jnp.where(
            (geom_type == int(mujoco.mjtGeom.mjGEOM_BOX))[:, None],
            box_support_world,
            support_world,
        )
        support_clearance = support_world[:, 2] - self._terrain_height(
            support_world[:, :2], world_id
        )

        geom_body_ids = jnp.asarray(self.model.geom_body_ids)[geom_ids]
        foot_body_ids = jnp.asarray(self.model.foot_body_ids)
        belongs_to_foot = foot_body_ids[:, None] == geom_body_ids[None, :]
        clearance_by_foot = jnp.where(
            belongs_to_foot,
            support_clearance[None, :],
            jnp.asarray(jnp.inf, dtype=jnp.float32),
        )
        lowest_geom = jnp.argmin(clearance_by_foot, axis=1)
        return support_world[lowest_geom]

    def _terrain_height(
        self, world_xy: jax.Array, world_id: jax.Array | int = 0
    ) -> jax.Array:
        if self.terrain_config.mode == "none":
            return jnp.full(
                jnp.asarray(world_xy).shape[:-1],
                self.terrain_config.pos[2],
                dtype=jnp.float32,
            )
        return terrain_surface_height_jax(
            self.model.mj_model,
            int(self.model.terrain_geom_ids[0]),
            world_xy,
            geom_quat=jnp.asarray(self.model.terrain_geom_quat)[
                jnp.minimum(
                    jnp.asarray(world_id, dtype=jnp.int32),
                    self.model.terrain_geom_quat.shape[0] - 1,
                )
            ],
        )

    def _terrain_surface_normal(
        self, world_xy: jax.Array, world_id: jax.Array | int = 0
    ) -> jax.Array:
        world_xy = jnp.asarray(world_xy, dtype=jnp.float32)
        output_shape = world_xy.shape[:-1] + (3,)
        if self.terrain_config.mode == "none":
            return jnp.broadcast_to(
                jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float32), output_shape
            )
        if self.terrain_config.mode == "plane":
            terrain_quat = jnp.asarray(self.model.terrain_geom_quat)[
                jnp.minimum(
                    jnp.asarray(world_id, dtype=jnp.int32),
                    self.model.terrain_geom_quat.shape[0] - 1,
                )
            ]
            terrain_up = quat_rotate(
                terrain_quat,
                jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float32),
            )
            return jnp.broadcast_to(terrain_up, output_shape)

        epsilon = jnp.asarray(1.0e-3, dtype=jnp.float32)
        x_offset = jnp.asarray([1.0e-3, 0.0], dtype=jnp.float32)
        y_offset = jnp.asarray([0.0, 1.0e-3], dtype=jnp.float32)
        gradient_x = (
            self._terrain_height(world_xy + x_offset, world_id)
            - self._terrain_height(world_xy - x_offset, world_id)
        ) / (2.0 * epsilon)
        gradient_y = (
            self._terrain_height(world_xy + y_offset, world_id)
            - self._terrain_height(world_xy - y_offset, world_id)
        ) / (2.0 * epsilon)
        normal = jnp.stack(
            [-gradient_x, -gradient_y, jnp.ones_like(gradient_x)], axis=-1
        )
        return normal / jnp.maximum(
            jnp.linalg.norm(normal, axis=-1, keepdims=True), 1.0e-8
        )

    def _foot_world_linear_velocity(self, state: BelloState) -> jax.Array:
        return self._body_world_lin_vel(state, self.model.foot_body_ids)

    def _foot_angular_speed(self, state: BelloState) -> jax.Array:
        foot_body_ids = jnp.asarray(self.model.foot_body_ids)
        foot_ang_vel = state.data.cvel[foot_body_ids, :3]
        return jnp.linalg.norm(foot_ang_vel, axis=1)

    def _foot_heading_error(self, state: BelloState) -> jax.Array:
        command_forward = quat_rotate(
            self._link_quat(state, self._reward_frame_link("track_lin_vel_xy")),
            jnp.asarray(self.model.robot.locomotion_forward_axis, dtype=jnp.float32),
        )[:2]
        command_forward /= jnp.maximum(jnp.linalg.norm(command_forward), 1.0e-6)
        errors = []
        for link_name, local_forward in self.model.robot.foot_forward_axes:
            foot_forward = quat_rotate(
                self._link_quat(state, link_name),
                jnp.asarray(local_forward, dtype=jnp.float32),
            )[:2]
            foot_forward /= jnp.maximum(jnp.linalg.norm(foot_forward), 1.0e-6)
            cross_z = (
                command_forward[0] * foot_forward[1]
                - command_forward[1] * foot_forward[0]
            )
            dot = jnp.dot(command_forward, foot_forward)
            errors.append(jnp.arctan2(cross_z, dot))
        return jnp.stack(errors)

    def _stance_width(self, state: BelloState, frame_link_name: str) -> jax.Array:
        if self.model.foot_body_ids.size != 2:
            return jnp.asarray(0.0, dtype=jnp.float32)
        foot_position = state.data.xpos[jnp.asarray(self.model.foot_body_ids)]
        foot_delta = foot_position[0] - foot_position[1]
        frame_yaw = yaw_quat(self._link_quat(state, frame_link_name))
        frame_delta = quat_rotate_inverse(frame_yaw, foot_delta)
        return jnp.abs(jnp.dot(frame_delta, self._locomotion_right_axis()))

    def _gait_diagnostics(
        self,
        state: BelloState,
        next_state: BelloState,
        foot_contact: jax.Array,
        foot_touchdown: jax.Array,
        foot_normal_impulse: jax.Array,
        foot_peak_normal_force: jax.Array,
    ) -> dict[str, jax.Array]:
        contact = foot_contact.astype(bool)
        previous = state.last_foot_contact.astype(bool)
        initialized = state.step_count > 0
        touchdown = foot_touchdown.astype(bool)
        liftoff = jnp.where(initialized, ~contact & previous, False)
        moving = command_is_moving(next_state.command, self.command_config)
        desired_contact = desired_gait_foot_contact(
            next_state.gait_phase, self.config.gait
        ).astype(bool)
        contact_count = jnp.sum(contact.astype(jnp.int32))
        normalized_joint_force = self._leg_joint_force(next_state) / jnp.maximum(
            self._robot_weight(), 1.0e-6
        )
        normalized_contact_force = foot_peak_normal_force / jnp.maximum(
            self._robot_weight(), 1.0e-6
        )
        normalized_contact_impulse = foot_normal_impulse / jnp.maximum(
            self._robot_weight() * self.control_timestep, 1.0e-6
        )
        joint_force_excess = jnp.maximum(
            normalized_joint_force
            - self.reward_config.gait.joint_force_soft_limit_body_weights,
            0.0,
        )
        foot_heading_error_deg = (
            jnp.abs(self._foot_heading_error(next_state)) * 180.0 / jnp.pi
        )
        return {
            "gait/stance_width": self._stance_width(
                next_state, self._reward_frame_link("track_lin_vel_xy")
            ),
            "gait/foot_heading_error_mean_deg": jnp.mean(foot_heading_error_deg),
            "gait/foot_heading_error_max_deg": jnp.max(foot_heading_error_deg),
            "gait/active": moving.astype(jnp.float32),
            "gait/phase": jnp.where(moving, next_state.gait_phase, 0.0),
            "gait/desired_left_contact": jnp.where(
                moving, desired_contact[0], False
            ).astype(jnp.float32),
            "gait/desired_right_contact": jnp.where(
                moving, desired_contact[1], False
            ).astype(jnp.float32),
            "gait/contact_match": jnp.where(
                moving, jnp.mean((contact == desired_contact).astype(jnp.float32)), 0.0
            ),
            "gait/touchdown": jnp.sum(touchdown.astype(jnp.float32)),
            "gait/liftoff": jnp.sum(liftoff.astype(jnp.float32)),
            "gait/single_support": (contact_count == 1).astype(jnp.float32),
            "gait/double_support": (contact_count == 2).astype(jnp.float32),
            "gait/flight": (contact_count == 0).astype(jnp.float32),
            "gait/joint_force_mean_body_weight": jnp.mean(normalized_joint_force),
            "gait/joint_force_max_body_weight": jnp.max(normalized_joint_force),
            "gait/joint_force_peak_excess": jnp.mean(
                jnp.sort(joint_force_excess)[
                    -self.reward_config.gait.joint_force_top_k :
                ]
                ** 2
            ),
            "gait/contact_force_mean_body_weight": jnp.mean(normalized_contact_force),
            "gait/contact_force_max_body_weight": jnp.max(normalized_contact_force),
            "gait/contact_impulse_mean_body_weight": jnp.mean(
                normalized_contact_impulse
            ),
            "gait/contact_impulse_max_body_weight": jnp.max(normalized_contact_impulse),
        }

    def _command_diagnostics(self, command: jax.Array) -> dict[str, jax.Array]:
        standing = command_is_standing(command, self.command_config)
        return {
            "command/standing": standing.astype(jnp.float32),
            "command/moving": (~standing).astype(jnp.float32),
        }

    def _actuator_joint_limit_low(self) -> jax.Array:
        joint_ids = jnp.asarray(self.model.actuator_joint_ids)
        return jnp.asarray(self.model.mj_model.jnt_range)[joint_ids, 0]

    def _actuator_joint_limit_high(self) -> jax.Array:
        joint_ids = jnp.asarray(self.model.actuator_joint_ids)
        return jnp.asarray(self.model.mj_model.jnt_range)[joint_ids, 1]

    def _policy_action_target_default(
        self, joint_angle_offset: jax.Array | None = None
    ) -> jax.Array:
        if joint_angle_offset is None:
            joint_angle_offset = jnp.asarray(self.model.joint_angle_offset[0])
        policy_indices = jnp.asarray(self.model.policy_ctrl_indices)
        target = (
            jnp.asarray(self.model.default_joint_target)[policy_indices]
            + joint_angle_offset[policy_indices]
        )
        return jnp.clip(
            target,
            self._policy_action_target_low(),
            self._policy_action_target_high(),
        )

    def _policy_action_target_low(self) -> jax.Array:
        policy_indices = jnp.asarray(self.model.policy_ctrl_indices)
        return jnp.asarray(self.model.joint_target_low)[policy_indices]

    def _policy_action_target_high(self) -> jax.Array:
        policy_indices = jnp.asarray(self.model.policy_ctrl_indices)
        return jnp.asarray(self.model.joint_target_high)[policy_indices]


def quat_rotate_inverse(quat_wxyz: jax.Array, vec: jax.Array) -> jax.Array:
    w, x, y, z = quat_wxyz
    qvec = jnp.asarray([x, y, z], dtype=jnp.float32)
    uv = jnp.cross(qvec, vec)
    uuv = jnp.cross(qvec, uv)
    return vec + 2.0 * (-w * uv + uuv)


def quat_rotate(quat_wxyz: jax.Array, vec: jax.Array) -> jax.Array:
    w, x, y, z = quat_wxyz
    qvec = jnp.asarray([x, y, z], dtype=jnp.float32)
    uv = jnp.cross(qvec, vec)
    uuv = jnp.cross(qvec, uv)
    return vec + 2.0 * (w * uv + uuv)


def yaw_quat(quat_wxyz: jax.Array) -> jax.Array:
    w, x, y, z = quat_wxyz
    yaw = jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half_yaw = 0.5 * yaw
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.stack([jnp.cos(half_yaw), zero, zero, jnp.sin(half_yaw)])


def yaw_quat_from_angle(yaw: jax.Array) -> jax.Array:
    half_yaw = 0.5 * yaw
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.stack([jnp.cos(half_yaw), zero, zero, jnp.sin(half_yaw)]).astype(
        jnp.float32
    )


def projected_gravity_tilt_angle(projected_gravity: jax.Array) -> jax.Array:
    return jnp.arccos(jnp.clip(-projected_gravity[2], -1.0, 1.0))


def tree_where(mask: jax.Array, true_tree, false_tree):
    def select(true_value, false_value):
        if true_value.ndim < mask.ndim or true_value.shape[: mask.ndim] != mask.shape:
            return false_value
        select_mask = mask.reshape(mask.shape + (1,) * (true_value.ndim - mask.ndim))
        return jnp.where(select_mask, true_value, false_value)

    return jax.tree_util.tree_map(select, true_tree, false_tree)


def pair_contains(pair: jax.Array, candidates) -> jax.Array:
    candidates = jnp.asarray(candidates, dtype=pair.dtype)
    if candidates.size == 0:
        return jnp.zeros(pair.shape[0], dtype=bool)
    return jnp.any(pair[:, :, None] == candidates[None, None, :], axis=(1, 2))


def pair_contains_each(pair: jax.Array, candidates) -> jax.Array:
    candidates = jnp.asarray(candidates, dtype=pair.dtype)
    if candidates.size == 0:
        return jnp.zeros((pair.shape[0], 0), dtype=bool)
    return jnp.any(pair[:, :, None] == candidates[None, None, :], axis=1)


def foot_terrain_contact_membership(
    contact_geom: jax.Array,
    geom_body_ids,
    foot_body_ids,
    foot_geom_ids,
    terrain_geom_ids,
) -> jax.Array:
    """Map sole-terrain contact pairs to their configured left/right foot."""

    contact_geom = jnp.asarray(contact_geom, dtype=jnp.int32)
    contact_body = jnp.asarray(geom_body_ids, dtype=jnp.int32)[contact_geom]
    sole_terrain = pair_contains(contact_geom, foot_geom_ids) & pair_contains(
        contact_geom, terrain_geom_ids
    )
    return sole_terrain[:, None] & pair_contains_each(contact_body, foot_body_ids)


def pair_both_contained(pair: jax.Array, candidates) -> jax.Array:
    candidates = jnp.asarray(candidates, dtype=pair.dtype)
    if candidates.size == 0:
        return jnp.zeros(pair.shape[0], dtype=bool)
    contained = jnp.any(pair[:, :, None] == candidates[None, None, :], axis=2)
    return jnp.all(contained, axis=1)


def require_index_range(name: str, indices: np.ndarray, upper_bound: int) -> None:
    indices = np.asarray(indices)
    if indices.size == 0:
        return
    if np.any(indices < 0) or np.any(indices >= upper_bound):
        raise ValueError(
            f"{name} contains out-of-range references for upper bound {upper_bound}: "
            f"{indices.tolist()}"
        )


def require_disjoint(
    left_name: str,
    left: np.ndarray,
    right_name: str,
    right: np.ndarray,
) -> None:
    overlap = sorted(set(np.asarray(left).tolist()) & set(np.asarray(right).tolist()))
    if overlap:
        raise ValueError(
            f"{left_name} must not reference held actuator controls in {right_name}: "
            f"{overlap}"
        )
