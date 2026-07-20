"""Bello MuJoCo/MJX locomotion environment scaffold."""

from __future__ import annotations

from dataclasses import dataclass
import math

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
    foot_air_time: jax.Array
    foot_contact_time: jax.Array
    foot_contact_debounce_time: jax.Array
    last_foot_contact: jax.Array
    command: jax.Array
    gait_frequency: jax.Array
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
    foot_phase: float = 1.0
    command: float = 1.0
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
    foot_phase: tuple[float, float] = (0.0, 0.0)
    command: tuple[float, float] = (0.0, 0.0)
    joint_pos: tuple[float, float] = (-0.01, 0.01)
    joint_vel: tuple[float, float] = (-1.5, 1.5)
    actuator_force: tuple[float, float] = (0.0, 0.0)
    last_action: tuple[float, float] = (0.0, 0.0)


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
class GaitConfig:
    cycle_time: float = 0.80
    command_speed_frequency_gain: float = 0.55
    min_frequency: float = 1.20
    max_frequency: float = 2.20
    duty_factor: float = 0.55
    phase_offsets: tuple[float, float] = (0.0, 0.5)
    standing_command_threshold: float = 0.05
    randomize_frequency: bool = False
    frequency_range: tuple[float, float] = (1.25, 1.50)
    foot_height: float = 0.10


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
    root_link: str = "bello_root"
    terrain: TerrainConfig = FLAT_TERRAIN
    simulation_physics: SimulationPhysicsConfig = SimulationPhysicsConfig()
    model_randomization: ModelRandomizationConfig = ModelRandomizationConfig()
    spawn: SpawnConfig = SpawnConfig()
    gait: GaitConfig = GaitConfig()
    contact: ContactConfig = ContactConfig()
    random_push: RandomPushConfig = RandomPushConfig()
    numerical_safety: NumericalSafetyConfig = NumericalSafetyConfig()
    observation: ObservationConfig = ObservationConfig()
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

    def _validate_config(self) -> None:
        if self.decimation <= 0:
            raise ValueError(f"decimation must be positive, got {self.decimation}")
        if self.max_episode_steps <= 0:
            raise ValueError(
                f"max_episode_steps must be positive, got {self.max_episode_steps}"
            )
        if not math.isfinite(self.config.action_clip) or self.config.action_clip <= 0:
            raise ValueError(
                "action_clip must be a positive finite value, got "
                f"{self.config.action_clip}"
            )
        if not self.config.root_link:
            raise ValueError("root_link must be a non-empty body name")
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
        self._validate_observation_config(
            "critic_observation", self.config.critic_observation
        )
        if self.config.critic_observation.noise.enabled:
            raise ValueError(
                "critic_observation.noise.enabled must be false; the privileged "
                "critic observation is intentionally clean"
            )
        if (
            not math.isfinite(self.config.gait.cycle_time)
            or self.config.gait.cycle_time <= 0
        ):
            raise ValueError(
                f"gait.cycle_time must be positive, got {self.config.gait.cycle_time}"
            )
        if (
            not math.isfinite(self.config.gait.command_speed_frequency_gain)
            or self.config.gait.command_speed_frequency_gain < 0
        ):
            raise ValueError(
                "gait.command_speed_frequency_gain must be finite and nonnegative, "
                f"got {self.config.gait.command_speed_frequency_gain}"
            )
        if (
            not math.isfinite(self.config.gait.min_frequency)
            or self.config.gait.min_frequency <= 0
        ):
            raise ValueError(
                "gait.min_frequency must be finite and positive, got "
                f"{self.config.gait.min_frequency}"
            )
        if (
            not math.isfinite(self.config.gait.max_frequency)
            or self.config.gait.max_frequency < self.config.gait.min_frequency
        ):
            raise ValueError(
                "gait.max_frequency must be finite and at least min_frequency, got "
                f"{self.config.gait.max_frequency}"
            )
        if not 0.0 <= self.config.gait.duty_factor <= 1.0:
            raise ValueError(
                "gait.duty_factor must be in [0, 1], got "
                f"{self.config.gait.duty_factor}"
            )
        if len(self.config.gait.phase_offsets) != 2:
            raise ValueError(
                "gait.phase_offsets must have two values, got "
                f"{self.config.gait.phase_offsets}"
            )
        if not all(math.isfinite(offset) for offset in self.config.gait.phase_offsets):
            raise ValueError(
                "gait.phase_offsets must be finite, got "
                f"{self.config.gait.phase_offsets}"
            )
        if (
            not math.isfinite(self.config.gait.standing_command_threshold)
            or self.config.gait.standing_command_threshold < 0
        ):
            raise ValueError(
                "gait.standing_command_threshold must be nonnegative, got "
                f"{self.config.gait.standing_command_threshold}"
            )
        validate_range("gait.frequency_range", self.config.gait.frequency_range)
        if self.config.gait.frequency_range[0] <= 0:
            raise ValueError("gait.frequency_range must contain positive frequencies")
        if (
            not math.isfinite(self.config.gait.foot_height)
            or self.config.gait.foot_height < 0
        ):
            raise ValueError("gait.foot_height must be finite and nonnegative")
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
        actuator_config: ActuatorConfig = ActuatorConfig(),
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

    def _observation_layout(self, observation: ObservationConfig) -> ObservationLayout:
        components: list[tuple[str, int]] = [
            ("imu_linear_acceleration", 3),
            ("imu_angular_velocity", 3),
            ("projected_gravity", 3),
            ("foot_phase", 4),
        ]
        if self.command_config.enabled:
            components.append(("command", 3))
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
            for link_name in {
                self.config.root_link,
                model.imu_body_name,
                *self.reward_config.enabled_link_names,
                *self.done_config.enabled_link_names,
            }:
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
        if self.reward_config.foot_phase.enabled:
            supported_foot_geom_types = {
                int(mujoco.mjtGeom.mjGEOM_SPHERE),
                int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
                int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                int(mujoco.mjtGeom.mjGEOM_BOX),
            }
            foot_geom_types = np.asarray(mj_model.geom_type)[model.foot_geom_ids]
            unsupported = sorted(
                {int(value) for value in foot_geom_types} - supported_foot_geom_types
            )
            if unsupported:
                raise ValueError(
                    "foot_phase requires primitive foot collision geoms so physical "
                    f"sole clearance can be measured; unsupported geom types: {unsupported}"
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
        command_rng = reset_rng = gait_rng = obs_rng = rng
        if rng is not None:
            command_rng, reset_rng, gait_rng, obs_rng = jax.random.split(rng, 4)
        command = self._resolve_command(command, command_rng)
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
            foot_air_time=jnp.zeros(self.model.foot_body_ids.size, dtype=jnp.float32),
            foot_contact_time=jnp.zeros(
                self.model.foot_body_ids.size, dtype=jnp.float32
            ),
            foot_contact_debounce_time=jnp.zeros(
                self.model.foot_body_ids.size, dtype=jnp.float32
            ),
            last_foot_contact=jnp.zeros(self.model.foot_body_ids.size, dtype=bool),
            command=command,
            gait_frequency=self.sample_gait_frequency(gait_rng),
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
        return state, self.observation(state, obs_rng)

    def reset_batch(
        self,
        batch_size: int,
        command: jax.Array | None = None,
        rng: jax.Array | None = None,
    ) -> tuple[BelloState, jax.Array]:
        command_rng = reset_rng = obs_rng = rng
        if rng is not None:
            command_rng, reset_rng, obs_rng = jax.random.split(rng, 3)
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

    def step(
        self, state: BelloState, action: jax.Array
    ) -> tuple[BelloState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        requested_action = jnp.asarray(action, dtype=jnp.float32)
        action = jnp.clip(
            requested_action,
            -self.action_clip,
            self.action_clip,
        )
        ctrl, joint_saturation, motor_saturation = self._action_target_details(action)
        data = state.data.replace(ctrl=ctrl)
        data = jax.lax.fori_loop(
            0, self.decimation, lambda _, d: mjx.step(self.model.mjx_model, d), data
        )
        next_state = BelloState(
            data=data,
            last_action=action,
            foot_air_time=state.foot_air_time,
            foot_contact_time=state.foot_contact_time,
            foot_contact_debounce_time=state.foot_contact_debounce_time,
            last_foot_contact=state.last_foot_contact,
            command=state.command,
            gait_frequency=state.gait_frequency,
            step_count=state.step_count + 1,
            episode_return=state.episode_return,
            episode_track_lin_vel_xy=state.episode_track_lin_vel_xy,
            episode_length=state.episode_length,
            last_episode_return=state.last_episode_return,
            last_episode_track_lin_vel_xy=state.last_episode_track_lin_vel_xy,
            last_episode_length=state.last_episode_length,
        )
        done_flags = self.done_flags(next_state)
        done = done_flags["done/any"]
        terminal = done_flags["done/terminal"]
        (
            raw_foot_contact,
            foot_height,
            foot_lateral_speed_sq,
            foot_angular_speed,
            undesired_contact,
            self_collision,
        ) = self._contact_terms(next_state, 0)
        foot_contact, foot_contact_debounce_time = self.debounce_foot_contact(
            next_state, raw_foot_contact, done
        )
        reward, reward_info = self.reward(
            state,
            next_state,
            action,
            requested_action,
            terminal,
            foot_contact,
            foot_height,
            foot_lateral_speed_sq,
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
        foot_air_time, foot_contact_time, last_foot_contact = self.next_foot_memory(
            state, foot_contact, done
        )
        next_state = BelloState(
            data=next_state.data,
            last_action=next_state.last_action,
            foot_air_time=foot_air_time,
            foot_contact_time=foot_contact_time,
            foot_contact_debounce_time=foot_contact_debounce_time,
            last_foot_contact=last_foot_contact,
            command=next_state.command,
            gait_frequency=next_state.gait_frequency,
            step_count=next_state.step_count,
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
            **self._gait_diagnostics(state, next_state, foot_contact),
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
        resample_rng = reset_command_rng = reset_obs_rng = push_rng = obs_rng = None
        if rng is not None:
            resample_rng, reset_command_rng, reset_obs_rng, push_rng, obs_rng = (
                jax.random.split(rng, 5)
            )
        requested_action = jnp.asarray(action, dtype=jnp.float32)
        action = jnp.clip(
            requested_action,
            -self.action_clip,
            self.action_clip,
        )
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
        data = jax.lax.fori_loop(
            0, self.decimation, lambda _, d: mjx.step(self.model.mjx_model, d), data
        )
        stepped_state = BelloState(
            data=data,
            last_action=action,
            foot_air_time=state.foot_air_time,
            foot_contact_time=state.foot_contact_time,
            foot_contact_debounce_time=state.foot_contact_debounce_time,
            last_foot_contact=state.last_foot_contact,
            command=state.command,
            gait_frequency=state.gait_frequency,
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
            foot_height,
            foot_lateral_speed_sq,
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
            foot_height,
            foot_lateral_speed_sq,
            foot_angular_speed,
            undesired_contact,
            self_collision,
            joint_angle_offset,
        )
        reward, reward_info = self.sanitize_nonfinite_reward(
            reward, reward_info, done_flags
        )
        episode_return = state.episode_return + reward
        episode_track_lin_vel_xy = (
            state.episode_track_lin_vel_xy + reward_info["reward/track_lin_vel_xy"]
        )
        episode_length = state.episode_length + 1
        foot_air_time, foot_contact_time, last_foot_contact = jax.vmap(
            self.next_foot_memory
        )(state, foot_contact, done)
        stepped_state = BelloState(
            data=stepped_state.data,
            last_action=stepped_state.last_action,
            foot_air_time=foot_air_time,
            foot_contact_time=foot_contact_time,
            foot_contact_debounce_time=foot_contact_debounce_time,
            last_foot_contact=last_foot_contact,
            command=self._maybe_resample_batch_command(
                stepped_state, done, resample_rng
            ),
            gait_frequency=stepped_state.gait_frequency,
            step_count=stepped_state.step_count,
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
            **jax.vmap(self._gait_diagnostics)(state, stepped_state, foot_contact),
            "random_push/applied": push_applied,
            "random_push/lin_vel_x": push_velocity[:, 0],
            "random_push/lin_vel_y": push_velocity[:, 1],
            "returned_episode": done,
            **done_flags,
            **reward_info,
        }

        def reset_terminated_worlds(_):
            reset_command = self._reset_batch_command(
                stepped_state.command, batch_size, reset_command_rng
            )
            reset_state, reset_obs = self.reset_batch(
                batch_size, command=reset_command, rng=reset_obs_rng
            )
            reset_state = reset_state.replace(
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
        return self._observation(state, self.config.observation, rng, world_id)

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
    ) -> jax.Array:
        (
            base_lin_vel,
            base_lin_acc,
            base_ang_vel,
            projected_gravity,
            base_height,
            foot_phase,
            command,
            joint_pos,
            joint_vel,
            actuator_force,
            last_action,
        ) = self._frame_terms(state, world_id)
        scale = observation.scale
        noise = observation.noise
        keys = (
            jax.random.split(rng, 11)
            if rng is not None and noise.enabled
            else (None,) * 11
        )
        parts = [
            self._observation_component(
                base_lin_acc, noise.base_lin_acc, scale.base_lin_acc, keys[0]
            ),
            self._observation_component(
                base_ang_vel, noise.base_ang_vel, scale.base_ang_vel, keys[1]
            ),
            self._observation_component(
                projected_gravity,
                noise.projected_gravity,
                scale.projected_gravity,
                keys[2],
            ),
            self._observation_component(
                foot_phase, noise.foot_phase, scale.foot_phase, keys[3]
            ),
        ]
        if self.command_config.enabled:
            parts.append(
                self._observation_component(
                    command, noise.command, scale.command, keys[4]
                )
            )
        parts.extend(
            (
                self._observation_component(
                    joint_pos, noise.joint_pos, scale.joint_pos, keys[5]
                ),
                self._observation_component(
                    joint_vel, noise.joint_vel, scale.joint_vel, keys[6]
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

    def _observation_component(
        self,
        value: jax.Array,
        noise_range: tuple[float, float],
        scale: float,
        rng: jax.Array | None,
    ) -> jax.Array:
        if rng is not None:
            value = value + jax.random.uniform(
                rng,
                value.shape,
                minval=noise_range[0],
                maxval=noise_range[1],
                dtype=value.dtype,
            )
        return value * jnp.asarray(scale, dtype=value.dtype)

    def action_to_ctrl(
        self,
        action: jax.Array,
        joint_angle_offset: jax.Array | None = None,
    ) -> jax.Array:
        ctrl, _, _ = self._action_target_details(action, joint_angle_offset)
        return ctrl

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
                return self.command_config.sample(rng, (batch_size,))
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

    def _maybe_resample_batch_command(
        self,
        state: BelloState,
        done: jax.Array,
        rng: jax.Array | None,
    ) -> jax.Array:
        if rng is None or not self.command_config.enabled:
            return state.command
        resample_steps = max(
            1, int(round(self.command_config.resample_seconds / self.control_timestep))
        )
        resample = (state.step_count % resample_steps == 0) & ~done
        sampled = self.command_config.sample(rng, (int(state.command.shape[0]),))
        return jnp.where(resample[:, None], sampled, state.command)

    def _reset_batch_command(
        self,
        command: jax.Array,
        batch_size: int,
        rng: jax.Array | None,
    ) -> jax.Array:
        if rng is None:
            return command
        return self.command_config.sample(rng, (batch_size,))

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
        base_lin_vel = self._link_lin_vel(state, self.config.root_link)
        imu_lin_acc = self._imu_linear_acceleration(state)
        imu_ang_vel = self._imu_angular_velocity(state)
        base_height = jnp.asarray(
            [self._link_clearance(state, self.config.root_link, world_id)],
            dtype=jnp.float32,
        )
        projected_gravity = self._link_projected_gravity(
            state, self.model.imu_body_name
        )
        foot_phase = self._foot_phase_sin_cos(state)
        return (
            base_lin_vel,
            imu_lin_acc,
            imu_ang_vel,
            projected_gravity,
            base_height,
            foot_phase,
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

    def _joint_pose_weights(self) -> jax.Array:
        weights = np.zeros(self.model.mj_model.nu, dtype=np.float32)
        weights[np.asarray(self.model.policy_ctrl_indices)] = 1.0
        low_weight_indices = self._actuator_indices_named(
            "hip_pitch_joint", "knee_joint"
        )
        weights[low_weight_indices] = 0.01
        return jnp.asarray(weights)

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
        if link_name == "bello_root":
            root_qpos = int(self.model.root_qpos_adr)
            return state.data.qpos[root_qpos + 3 : root_qpos + 7]
        return state.data.xquat[self._link_body_id(link_name)]

    def _link_height(self, state: BelloState, link_name: str) -> jax.Array:
        return self._link_position(state, link_name)[2]

    def _link_position(self, state: BelloState, link_name: str) -> jax.Array:
        if link_name == "bello_root":
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
        if link_name == "bello_root":
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
        if link_name == "bello_root":
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

    def _link_ang_vel(self, state: BelloState, link_name: str) -> jax.Array:
        return self._link_body_ang_vel(state, link_name)

    def _link_body_ang_vel(self, state: BelloState, link_name: str) -> jax.Array:
        if link_name == "bello_root":
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
        return self._link_clearance(state, self.config.root_link, world_id)

    def reward(
        self,
        state: BelloState,
        next_state: BelloState,
        action: jax.Array,
        requested_action: jax.Array,
        done: jax.Array,
        foot_contact: jax.Array,
        foot_height: jax.Array,
        foot_lateral_speed_sq: jax.Array,
        foot_angular_speed: jax.Array,
        undesired_contact: jax.Array,
        self_collision: jax.Array,
        joint_angle_offset: jax.Array | None = None,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        command = next_state.command
        qpos_idx = jnp.asarray(self.model.actuator_qpos_indices)
        joint_pos_rel, _ = self._all_actuator_joint_terms(next_state)
        lin_vel_xy = self._link_yaw_frame_lin_vel(
            next_state,
            self._reward_link(self.reward_config.track_lin_vel_xy),
            self._reward_frame_link(self.reward_config.track_lin_vel_xy),
        )[:2]
        yaw_vel_z = self._link_body_ang_vel(
            next_state, self._reward_link(self.reward_config.track_ang_vel_z)
        )[2]
        lin_vel_z = self._link_body_lin_vel(
            next_state, self._reward_link(self.reward_config.lin_vel_z)
        )[2]
        ang_vel_xy = self._link_body_ang_vel(
            next_state, self._reward_link(self.reward_config.ang_vel_xy)
        )[:2]
        projected_gravity = self._link_projected_gravity(
            next_state, self._reward_link(self.reward_config.flat_orientation)
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
                joint_pos=next_state.data.qpos[qpos_idx],
                joint_pos_rel=joint_pos_rel,
                action=action,
                requested_action=requested_action,
                last_action=state.last_action,
                actuator_force=next_state.data.actuator_force,
                joint_limit_low=self._actuator_joint_limit_low(),
                joint_limit_high=self._actuator_joint_limit_high(),
                policy_joint_indices=jnp.asarray(self.model.policy_ctrl_indices),
                action_target_default=self._policy_action_target_default(
                    joint_angle_offset
                ),
                action_target_low=self._policy_action_target_low(),
                action_target_high=self._policy_action_target_high(),
                foot_contact=foot_contact,
                foot_height=foot_height,
                foot_lateral_speed_sq=foot_lateral_speed_sq,
                foot_angular_speed=foot_angular_speed,
                foot_air_time=state.foot_air_time,
                foot_contact_time=state.foot_contact_time,
                last_foot_contact=state.last_foot_contact,
                foot_phase_height_target=self._foot_phase_height_target(next_state),
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
        )

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

    def _gait_frequency(self, command: jax.Array) -> jax.Array:
        base_frequency = 1.0 / jnp.asarray(
            self.config.gait.cycle_time, dtype=jnp.float32
        )
        command_speed = jnp.linalg.norm(command[:2])
        frequency = (
            base_frequency
            + self.config.gait.command_speed_frequency_gain * command_speed
        )
        return jnp.clip(
            frequency,
            self.config.gait.min_frequency,
            self.config.gait.max_frequency,
        )

    def sample_gait_frequency(self, rng: jax.Array | None) -> jax.Array:
        if not self.config.gait.randomize_frequency:
            return jnp.asarray(0.0, dtype=jnp.float32)
        low, high = self.config.gait.frequency_range
        if rng is None:
            return jnp.asarray((low + high) / 2.0, dtype=jnp.float32)
        return jax.random.uniform(rng, (), minval=low, maxval=high, dtype=jnp.float32)

    def _gait_phase_from_step_count(
        self, step_count: jax.Array, command: jax.Array
    ) -> jax.Array:
        phase = (
            step_count.astype(jnp.float32)
            * self.control_timestep
            * self._gait_frequency(command)
        )
        return jnp.where(
            self._is_standing_command(command),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.mod(phase, 1.0),
        )

    def _is_standing_command(self, command: jax.Array) -> jax.Array:
        return jnp.linalg.norm(command) <= self.config.gait.standing_command_threshold

    def _gait_phase(self, state: BelloState) -> jax.Array:
        if self.config.gait.randomize_frequency:
            phase = (
                state.step_count.astype(jnp.float32)
                * self.control_timestep
                * state.gait_frequency
            )
            return jnp.where(
                self._is_standing_command(state.command),
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.mod(phase, 1.0),
            )
        return self._gait_phase_from_step_count(state.step_count, state.command)

    def _gait_phase_sin_cos(self, state: BelloState) -> jax.Array:
        angle = 2.0 * jnp.pi * self._gait_phase(state)
        return jnp.stack([jnp.sin(angle), jnp.cos(angle)]).astype(jnp.float32)

    def _foot_phase_angles(self, state: BelloState) -> jax.Array:
        phase_offsets = jnp.asarray(self.config.gait.phase_offsets, dtype=jnp.float32)
        return 2.0 * jnp.pi * jnp.mod(self._gait_phase(state) + phase_offsets, 1.0)

    def _foot_phase_sin_cos(self, state: BelloState) -> jax.Array:
        angles = self._foot_phase_angles(state)
        phase = jnp.concatenate([jnp.cos(angles), jnp.sin(angles)]).astype(jnp.float32)
        return jnp.where(
            self._is_standing_command(state.command),
            jnp.zeros_like(phase),
            phase,
        )

    def _foot_phase_height_target(self, state: BelloState) -> jax.Array:
        phase = jnp.mod(
            self._gait_phase(state)
            + jnp.asarray(self.config.gait.phase_offsets, dtype=jnp.float32),
            1.0,
        )
        duty_factor = jnp.asarray(self.config.gait.duty_factor, dtype=jnp.float32)
        swing_duration = jnp.maximum(1.0 - duty_factor, 1.0e-6)
        swing_phase = jnp.clip((phase - duty_factor) / swing_duration, 0.0, 1.0)
        swing_height = self.config.gait.foot_height * jnp.sin(jnp.pi * swing_phase) ** 2
        target = jnp.where(phase < duty_factor, 0.0, swing_height)
        return jnp.where(
            self._is_standing_command(state.command),
            jnp.zeros_like(target),
            target,
        )

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

    def next_foot_memory(
        self, state: BelloState, foot_contact: jax.Array, done: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        contact = foot_contact.astype(bool)
        next_air_time = jnp.where(
            contact,
            jnp.zeros_like(state.foot_air_time),
            state.foot_air_time + self.control_timestep,
        )
        next_contact_time = jnp.where(
            contact,
            state.foot_contact_time + self.control_timestep,
            jnp.zeros_like(state.foot_contact_time),
        )
        reset_air_time = jnp.zeros_like(next_air_time)
        reset_contact_time = jnp.zeros_like(next_contact_time)
        return (
            jnp.where(done, reset_air_time, next_air_time),
            jnp.where(done, reset_contact_time, next_contact_time),
            contact,
        )

    def _reward_link(self, term) -> str:
        if term.enabled and term.link is not None:
            return term.link
        return self.config.root_link

    def _reward_frame_link(self, term) -> str:
        velocity_link = self._reward_link(term)
        frame_link = getattr(term, "frame_link", None)
        if term.enabled and frame_link is not None:
            return frame_link
        return velocity_link

    def _done_link(self, term) -> str:
        if term.enabled and term.link is not None:
            return term.link
        return self.config.root_link

    def _contact_terms(
        self, state: BelloState, world_id: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        foot_body_ids = jnp.asarray(self.model.foot_body_ids)
        foot_count = foot_body_ids.shape[0]
        if (
            self.model.terrain_geom_ids.size == 0
            and self.model.robot_geom_ids.size == 0
        ):
            return (
                jnp.zeros(foot_count, dtype=jnp.float32),
                self._foot_height(state, world_id),
                jnp.zeros(foot_count, dtype=jnp.float32),
                jnp.zeros(foot_count, dtype=jnp.float32),
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
        foot_contact_slot = (
            active
            & terrain_contact
            & foot_forceful_contact
            & pair_contains(contact_body, self.model.foot_body_ids)
        )
        foot_contact = jnp.any(
            foot_contact_slot[:, None]
            & pair_contains_each(contact_body, self.model.foot_body_ids),
            axis=0,
        ).astype(jnp.float32)
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
            self._foot_height(state, world_id),
            self._foot_lateral_speed_sq(state),
            self._foot_angular_speed(state),
            undesired_contact,
            self_collision,
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

    def _foot_lateral_speed_sq(self, state: BelloState) -> jax.Array:
        foot_world_vel = self._body_world_lin_vel(state, self.model.foot_body_ids)
        return jnp.sum(foot_world_vel[:, :2] ** 2, axis=1)

    def _foot_angular_speed(self, state: BelloState) -> jax.Array:
        foot_body_ids = jnp.asarray(self.model.foot_body_ids)
        foot_ang_vel = state.data.cvel[foot_body_ids, :3]
        return jnp.linalg.norm(foot_ang_vel, axis=1)

    def _stance_width(self, state: BelloState, frame_link_name: str) -> jax.Array:
        if self.model.foot_body_ids.size != 2:
            return jnp.asarray(0.0, dtype=jnp.float32)
        foot_position = state.data.xpos[jnp.asarray(self.model.foot_body_ids)]
        foot_delta = foot_position[0] - foot_position[1]
        frame_yaw = yaw_quat(self._link_quat(state, frame_link_name))
        frame_delta = quat_rotate_inverse(frame_yaw, foot_delta)
        return jnp.abs(frame_delta[1])

    def _gait_diagnostics(
        self,
        state: BelloState,
        next_state: BelloState,
        foot_contact: jax.Array,
    ) -> dict[str, jax.Array]:
        contact = foot_contact.astype(bool)
        previous = state.last_foot_contact.astype(bool)
        initialized = state.step_count > 0
        touchdown = jnp.where(initialized, contact & ~previous, False)
        liftoff = jnp.where(initialized, ~contact & previous, False)
        moving = (
            jnp.linalg.norm(next_state.command)
            > self.reward_config.contact_timing.command_speed_threshold
        )
        premature_touchdown = (
            touchdown
            & (state.foot_air_time < self.reward_config.contact_timing.min_air_time)
            & moving
        )
        premature_liftoff = (
            liftoff
            & (
                state.foot_contact_time
                < self.reward_config.contact_timing.min_contact_time
            )
            & moving
        )
        contact_count = jnp.sum(contact.astype(jnp.int32))
        return {
            "gait/stance_width": self._stance_width(next_state, self.config.root_link),
            "gait/touchdown": jnp.sum(touchdown.astype(jnp.float32)),
            "gait/liftoff": jnp.sum(liftoff.astype(jnp.float32)),
            "gait/premature_touchdown": jnp.sum(
                premature_touchdown.astype(jnp.float32)
            ),
            "gait/premature_liftoff": jnp.sum(premature_liftoff.astype(jnp.float32)),
            "gait/single_support": (contact_count == 1).astype(jnp.float32),
            "gait/double_support": (contact_count == 2).astype(jnp.float32),
            "gait/flight": (contact_count == 0).astype(jnp.float32),
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
