"""Bello MuJoCo/MJX locomotion environment scaffold."""

from __future__ import annotations

from dataclasses import dataclass
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
    ACTION_CLIP,
    CONTROL_DECIMATION,
    MAX_EPISODE_STEPS,
    SIM_TIMESTEP,
    DEFAULT_SPAWN_CLEARANCE,
)
from bello_mujoco.envs.rewards import (
    CommandConfig,
    RewardConfig,
    compute_velocity_reward,
    validate_range,
    validate_reward_config,
)
from bello_mujoco.sim import (
    BelloModel,
    WarpConfig,
    load_bello_model,
    load_prepared_bello_model,
)
from bello_mujoco.terrain import FLAT_TERRAIN, TerrainConfig


@chex.dataclass(frozen=True)
class BelloState:
    data: mjx.Data
    last_action: jax.Array
    foot_air_time: jax.Array
    foot_contact_debounce_time: jax.Array
    last_foot_contact: jax.Array
    command: jax.Array
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
class RollPitchDoneConfig(BaseDoneConfig):
    max_abs_roll: float = 1.0
    max_abs_pitch: float = 1.0
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
    roll_pitch: RollPitchDoneConfig = RollPitchDoneConfig()
    timeout: TimeoutDoneConfig = TimeoutDoneConfig()
    nonfinite: NonfiniteDoneConfig = NonfiniteDoneConfig()

    @property
    def enabled_link_names(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    term.link
                    for term in (self.base_height, self.roll_pitch)
                    if term.enabled and term.link
                }
            )
        )


@dataclass(frozen=True)
class ObservationScaleConfig:
    base_lin_vel: float = 1.0
    base_ang_vel: float = 1.0
    projected_gravity: float = 1.0
    base_height: float = 1.0
    gait_phase: float = 1.0
    command: float = 1.0
    joint_pos: float = 1.0
    joint_vel: float = 1.0
    actuator_force: float = 1.0
    last_action: float = 1.0


@dataclass(frozen=True)
class ObservationNoiseConfig:
    enabled: bool = False
    base_lin_vel: tuple[float, float] = (0.0, 0.0)
    base_ang_vel: tuple[float, float] = (-0.2, 0.2)
    projected_gravity: tuple[float, float] = (-0.05, 0.05)
    base_height: tuple[float, float] = (0.0, 0.0)
    gait_phase: tuple[float, float] = (0.0, 0.0)
    command: tuple[float, float] = (0.0, 0.0)
    joint_pos: tuple[float, float] = (-0.01, 0.01)
    joint_vel: tuple[float, float] = (-1.5, 1.5)
    actuator_force: tuple[float, float] = (0.0, 0.0)
    last_action: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class ObservationConfig:
    joint_mode: Literal["all_actuated", "policy"] = "all_actuated"
    base_lin_vel: bool = False
    base_ang_vel: bool = True
    projected_gravity: bool = True
    base_height: bool = False
    gait_phase: bool = False
    command: bool = True
    joint_pos: bool = True
    joint_vel: bool = True
    actuator_force: bool = True
    last_action: bool = True
    scale: ObservationScaleConfig = ObservationScaleConfig()
    noise: ObservationNoiseConfig = ObservationNoiseConfig()


@dataclass(frozen=True)
class SpawnConfig:
    clearance: float = DEFAULT_SPAWN_CLEARANCE


@dataclass(frozen=True)
class GaitConfig:
    cycle_time: float = 0.80
    duty_factor: float = 0.55
    phase_offsets: tuple[float, float] = (0.0, 0.5)
    standing_command_threshold: float = 0.05


@dataclass(frozen=True)
class ContactConfig:
    foot_contact_on_seconds: float = 0.0
    foot_contact_off_seconds: float = 0.0


@dataclass(frozen=True)
class RandomPushConfig:
    enabled: bool = False
    interval_seconds: float = 5.0
    lin_vel_x: tuple[float, float] = (-0.5, 0.5)
    lin_vel_y: tuple[float, float] = (-0.5, 0.5)


@dataclass(frozen=True)
class BelloEnvConfig:
    """Runtime environment settings that affect stepping and model loading."""

    decimation: int = CONTROL_DECIMATION
    max_episode_steps: int = MAX_EPISODE_STEPS
    root_link: str = "bello_root"
    terrain: TerrainConfig = FLAT_TERRAIN
    spawn: SpawnConfig = SpawnConfig()
    gait: GaitConfig = GaitConfig()
    contact: ContactConfig = ContactConfig()
    random_push: RandomPushConfig = RandomPushConfig()
    observation: ObservationConfig = ObservationConfig()


@dataclass(frozen=True)
class BelloLocomotionEnv:
    """Bello sim scaffold with actuator-order action and observation.

    Action contains policy-controlled actuator residuals. Held actuators stay at
    their configured default targets.
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
        if self.config.observation.joint_mode not in {"all_actuated", "policy"}:
            raise ValueError(
                "observation.joint_mode must be 'all_actuated' or 'policy', got "
                f"{self.config.observation.joint_mode!r}"
            )
        self._validate_observation_config()
        if (
            not math.isfinite(self.config.gait.cycle_time)
            or self.config.gait.cycle_time <= 0
        ):
            raise ValueError(
                f"gait.cycle_time must be positive, got {self.config.gait.cycle_time}"
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
            not math.isfinite(self.config.random_push.interval_seconds)
            or self.config.random_push.interval_seconds <= 0.0
        ):
            raise ValueError(
                "random_push.interval_seconds must be positive and finite, got "
                f"{self.config.random_push.interval_seconds}"
            )
        validate_range("random_push.lin_vel_x", self.config.random_push.lin_vel_x)
        validate_range("random_push.lin_vel_y", self.config.random_push.lin_vel_y)
        validate_reward_config(self.reward_config)
        if self.done_config.base_height.enabled and not math.isfinite(
            self.done_config.base_height.min_height
        ):
            raise ValueError(
                "done_config.base_height.min_height must be finite, got "
                f"{self.done_config.base_height.min_height}"
            )
        if self.done_config.roll_pitch.enabled:
            if self.done_config.roll_pitch.max_abs_roll < 0:
                raise ValueError(
                    "done_config.roll_pitch.max_abs_roll must be nonnegative, got "
                    f"{self.done_config.roll_pitch.max_abs_roll}"
                )
            if self.done_config.roll_pitch.max_abs_pitch < 0:
                raise ValueError(
                    "done_config.roll_pitch.max_abs_pitch must be nonnegative, got "
                    f"{self.done_config.roll_pitch.max_abs_pitch}"
                )

    def _validate_observation_config(self) -> None:
        for name, value in self.config.observation.scale.__dict__.items():
            if not math.isfinite(value):
                raise ValueError(
                    f"observation.scale.{name} must be finite, got {value}"
                )
        for name, bounds in self.config.observation.noise.__dict__.items():
            if name == "enabled":
                continue
            validate_range(f"observation.noise.{name}", bounds)

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
    ) -> "BelloLocomotionEnv":
        return cls(
            load_bello_model(
                config,
                warp=warp,
                batch_size=batch_size,
                terrain=env_config.terrain,
                spawn_clearance=env_config.spawn.clearance,
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
    ) -> "BelloLocomotionEnv":
        return cls(
            load_prepared_bello_model(
                prepared,
                warp=warp,
                batch_size=batch_size,
                terrain=env_config.terrain,
                actuator_config=actuator_config,
                spawn_clearance=env_config.spawn.clearance,
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
        return ACTION_CLIP

    @property
    def action_size(self) -> int:
        return int(self.model.policy_ctrl_indices.size)

    @property
    def observation_size(self) -> int:
        observation = self.config.observation
        return (
            (3 if observation.base_lin_vel else 0)
            + (3 if observation.base_ang_vel else 0)
            + (3 if observation.projected_gravity else 0)
            + (1 if observation.base_height else 0)
            + (2 if observation.gait_phase else 0)
            + self.command_observation_size
            + (self.observed_joint_count if observation.joint_pos else 0)
            + (self.observed_joint_count if observation.joint_vel else 0)
            + (self.observed_actuator_count if observation.actuator_force else 0)
            + (self.action_size if observation.last_action else 0)
        )

    @property
    def command_observation_size(self) -> int:
        return (
            3 if self.config.observation.command and self.command_config.enabled else 0
        )

    @property
    def observed_joint_count(self) -> int:
        if self.config.observation.joint_mode == "policy":
            return int(self.model.policy_ctrl_indices.size)
        return int(self.model.mj_model.nu)

    @property
    def observed_actuator_count(self) -> int:
        if self.config.observation.joint_mode == "policy":
            return int(self.model.policy_ctrl_indices.size)
        return int(self.model.mj_model.nu)

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
            "policy_qpos_indices", model.policy_qpos_indices, mj_model.nq
        )
        require_index_range(
            "policy_qvel_indices", model.policy_qvel_indices, mj_model.nv
        )
        require_index_range("foot_body_ids", model.foot_body_ids, mj_model.nbody)
        require_index_range("foot_geom_ids", model.foot_geom_ids, mj_model.ngeom)
        require_index_range("terrain_geom_ids", model.terrain_geom_ids, mj_model.ngeom)
        require_index_range(
            "nonfoot_robot_geom_ids", model.nonfoot_robot_geom_ids, mj_model.ngeom
        )
        if isinstance(mj_model, mujoco.MjModel):
            for link_name in {
                self.config.root_link,
                *self.reward_config.enabled_link_names,
                *self.done_config.enabled_link_names,
            }:
                if self._link_body_id(link_name) < 0:
                    raise ValueError(f"configured link/body not found: {link_name!r}")

        if model.policy_action_scale.shape != model.policy_ctrl_indices.shape:
            raise ValueError(
                "policy_action_scale must have one value per policy action, got "
                f"{model.policy_action_scale.shape} for "
                f"{model.policy_ctrl_indices.shape} policy controls"
            )
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
        if (
            self.terrain_config.mode != "none"
            and self.reward_config.uses_terrain_contact_metadata
            and model.terrain_geom_ids.size == 0
        ):
            raise ValueError(
                f"terrain mode {self.terrain_config.mode!r} requires a terrain geom "
                f"named {self.terrain_config.name!r} for enabled contact rewards"
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
        command_ranges: jax.Array | None = None,
    ) -> tuple[BelloState, jax.Array]:
        command_rng = obs_rng = rng
        if rng is not None:
            command_rng, obs_rng = jax.random.split(rng)
        command = self._resolve_command(command, command_rng, command_ranges)
        data = self.model.make_data()
        data = data.replace(
            qpos=jnp.asarray(self.model.default_qpos),
            qvel=jnp.asarray(self.model.default_qvel),
            ctrl=jnp.asarray(self.model.default_ctrl),
        )
        state = BelloState(
            data=data,
            last_action=jnp.zeros(self.action_size, dtype=jnp.float32),
            foot_air_time=jnp.zeros(self.model.foot_body_ids.size, dtype=jnp.float32),
            foot_contact_debounce_time=jnp.zeros(
                self.model.foot_body_ids.size, dtype=jnp.float32
            ),
            last_foot_contact=jnp.ones(self.model.foot_body_ids.size, dtype=bool),
            command=command,
            step_count=jnp.asarray(0, dtype=jnp.int32),
            episode_return=jnp.asarray(0.0, dtype=jnp.float32),
            episode_track_lin_vel_xy=jnp.asarray(0.0, dtype=jnp.float32),
            episode_length=jnp.asarray(0, dtype=jnp.int32),
            last_episode_return=jnp.asarray(0.0, dtype=jnp.float32),
            last_episode_track_lin_vel_xy=jnp.asarray(0.0, dtype=jnp.float32),
            last_episode_length=jnp.asarray(0, dtype=jnp.int32),
        )
        return state, self.observation(state, obs_rng)

    def reset_batch(
        self,
        batch_size: int,
        command: jax.Array | None = None,
        rng: jax.Array | None = None,
        command_ranges: jax.Array | None = None,
    ) -> tuple[BelloState, jax.Array]:
        command_rng = obs_rng = rng
        if rng is not None:
            command_rng, obs_rng = jax.random.split(rng)
        command = self._resolve_batch_command(
            batch_size, command, command_rng, command_ranges
        )
        state, obs = jax.vmap(self.reset)(command)
        data = state.data.replace(_impl=state.data._impl.replace(nworld=batch_size))
        state = state.replace(data=data)
        return state, self.observation_batch(state, obs_rng)

    def step(
        self, state: BelloState, action: jax.Array
    ) -> tuple[BelloState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        action = jnp.clip(
            jnp.asarray(action, dtype=jnp.float32), -ACTION_CLIP, ACTION_CLIP
        )
        data = state.data.replace(ctrl=self._action_to_ctrl(action))
        data = jax.lax.fori_loop(
            0, self.decimation, lambda _, d: mjx.step(self.model.mjx_model, d), data
        )
        next_state = BelloState(
            data=data,
            last_action=action,
            foot_air_time=state.foot_air_time,
            foot_contact_debounce_time=state.foot_contact_debounce_time,
            last_foot_contact=state.last_foot_contact,
            command=state.command,
            step_count=state.step_count + 1,
            episode_return=state.episode_return,
            episode_track_lin_vel_xy=state.episode_track_lin_vel_xy,
            episode_length=state.episode_length,
            last_episode_return=state.last_episode_return,
            last_episode_track_lin_vel_xy=state.last_episode_track_lin_vel_xy,
            last_episode_length=state.last_episode_length,
        )
        done_flags = self._done_flags(next_state)
        done = done_flags["done/any"]
        terminal = done_flags["done/terminal"]
        raw_foot_contact, foot_height, foot_lateral_speed_sq, undesired_contact = (
            self._contact_terms(next_state, 0)
        )
        foot_contact, foot_contact_debounce_time = self._debounce_foot_contact(
            next_state, raw_foot_contact, done
        )
        reward, reward_info = self._reward(
            state,
            next_state,
            action,
            terminal,
            foot_contact,
            foot_height,
            foot_lateral_speed_sq,
            undesired_contact,
        )
        episode_return = state.episode_return + reward
        episode_track_lin_vel_xy = (
            state.episode_track_lin_vel_xy + reward_info["reward/track_lin_vel_xy"]
        )
        episode_length = state.episode_length + 1
        foot_air_time, last_foot_contact = self._next_foot_memory(
            state, foot_contact, done
        )
        next_state = BelloState(
            data=next_state.data,
            last_action=next_state.last_action,
            foot_air_time=foot_air_time,
            foot_contact_debounce_time=foot_contact_debounce_time,
            last_foot_contact=last_foot_contact,
            command=next_state.command,
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
            "episode_return": jnp.where(done, episode_return, 0.0),
            "episode_reward/track_lin_vel_xy": jnp.where(
                done, episode_track_lin_vel_xy, 0.0
            ),
            "episode_length": jnp.where(done, episode_length, 0),
            "foot_contact/raw": raw_foot_contact,
            "foot_contact/filtered": foot_contact,
            "foot_contact/debounce_time": foot_contact_debounce_time,
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
        command_ranges: jax.Array | None = None,
    ) -> tuple[BelloState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        batch_size = int(action.shape[0])
        resample_rng = reset_command_rng = reset_obs_rng = push_rng = obs_rng = None
        if rng is not None:
            resample_rng, reset_command_rng, reset_obs_rng, push_rng, obs_rng = (
                jax.random.split(rng, 5)
            )
        action = jnp.clip(
            jnp.asarray(action, dtype=jnp.float32), -ACTION_CLIP, ACTION_CLIP
        )
        ctrl = jax.vmap(self._action_to_ctrl)(action)
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
            foot_contact_debounce_time=state.foot_contact_debounce_time,
            last_foot_contact=state.last_foot_contact,
            command=state.command,
            step_count=state.step_count + 1,
            episode_return=state.episode_return,
            episode_track_lin_vel_xy=state.episode_track_lin_vel_xy,
            episode_length=state.episode_length,
            last_episode_return=state.last_episode_return,
            last_episode_track_lin_vel_xy=state.last_episode_track_lin_vel_xy,
            last_episode_length=state.last_episode_length,
        )
        done_flags = jax.vmap(self._done_flags)(stepped_state)
        done = done_flags["done/any"]
        terminal = done_flags["done/terminal"]
        world_ids = jnp.arange(batch_size, dtype=jnp.int32)
        raw_foot_contact, foot_height, foot_lateral_speed_sq, undesired_contact = (
            jax.vmap(self._contact_terms)(stepped_state, world_ids)
        )
        foot_contact, foot_contact_debounce_time = jax.vmap(
            self._debounce_foot_contact
        )(stepped_state, raw_foot_contact, done)
        reward, reward_info = jax.vmap(self._reward)(
            state,
            stepped_state,
            action,
            terminal,
            foot_contact,
            foot_height,
            foot_lateral_speed_sq,
            undesired_contact,
        )
        episode_return = state.episode_return + reward
        episode_track_lin_vel_xy = (
            state.episode_track_lin_vel_xy + reward_info["reward/track_lin_vel_xy"]
        )
        episode_length = state.episode_length + 1
        foot_air_time, last_foot_contact = jax.vmap(self._next_foot_memory)(
            state, foot_contact, done
        )
        stepped_state = BelloState(
            data=stepped_state.data,
            last_action=stepped_state.last_action,
            foot_air_time=foot_air_time,
            foot_contact_debounce_time=foot_contact_debounce_time,
            last_foot_contact=last_foot_contact,
            command=self._maybe_resample_batch_command(
                stepped_state, done, resample_rng, command_ranges
            ),
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
        info = {
            "base_height": jax.vmap(self._base_height)(stepped_state),
            "bootstrap_observation": stepped_obs,
            "episode_return": jnp.where(done, episode_return, 0.0),
            "episode_reward/track_lin_vel_xy": jnp.where(
                done, episode_track_lin_vel_xy, 0.0
            ),
            "episode_length": jnp.where(done, episode_length, 0),
            "foot_contact/raw": raw_foot_contact,
            "foot_contact/filtered": foot_contact,
            "foot_contact/debounce_time": foot_contact_debounce_time,
            "random_push/applied": push_applied,
            "random_push/lin_vel_x": push_velocity[:, 0],
            "random_push/lin_vel_y": push_velocity[:, 1],
            "returned_episode": done,
            **done_flags,
            **reward_info,
        }
        reset_command = self._reset_batch_command(
            stepped_state.command, batch_size, reset_command_rng, command_ranges
        )
        reset_state, reset_obs = self.reset_batch(
            batch_size, command=reset_command, rng=reset_obs_rng
        )
        reset_state = reset_state.replace(
            last_episode_return=stepped_state.last_episode_return,
            last_episode_track_lin_vel_xy=stepped_state.last_episode_track_lin_vel_xy,
            last_episode_length=stepped_state.last_episode_length,
        )
        next_state = tree_where(done, reset_state, stepped_state)
        next_obs = jnp.where(done[:, None], reset_obs, stepped_obs)
        return next_state, next_obs, reward, done, info

    def observation_batch(
        self, state: BelloState, rng: jax.Array | None = None
    ) -> jax.Array:
        if rng is None:
            return jax.vmap(self.observation)(state)
        keys = jax.random.split(rng, int(state.command.shape[0]))
        return jax.vmap(self.observation)(state, keys)

    def observation(self, state: BelloState, rng: jax.Array | None = None) -> jax.Array:
        (
            base_lin_vel,
            base_ang_vel,
            projected_gravity,
            base_height,
            gait_phase,
            command,
            joint_pos,
            joint_vel,
            actuator_force,
            last_action,
        ) = self._frame_terms(state)
        observation = self.config.observation
        scale = observation.scale
        noise = observation.noise
        keys = (
            jax.random.split(rng, 10)
            if rng is not None and noise.enabled
            else (None,) * 10
        )
        parts = []
        if observation.base_lin_vel:
            parts.append(
                self._observation_component(
                    base_lin_vel, noise.base_lin_vel, scale.base_lin_vel, keys[0]
                )
            )
        if observation.base_ang_vel:
            parts.append(
                self._observation_component(
                    base_ang_vel, noise.base_ang_vel, scale.base_ang_vel, keys[1]
                )
            )
        if observation.projected_gravity:
            parts.append(
                self._observation_component(
                    projected_gravity,
                    noise.projected_gravity,
                    scale.projected_gravity,
                    keys[2],
                )
            )
        if observation.base_height:
            parts.append(
                self._observation_component(
                    base_height, noise.base_height, scale.base_height, keys[3]
                )
            )
        if observation.gait_phase:
            parts.append(
                self._observation_component(
                    gait_phase, noise.gait_phase, scale.gait_phase, keys[4]
                )
            )
        if observation.command and self.command_config.enabled:
            parts.append(
                self._observation_component(
                    command, noise.command, scale.command, keys[5]
                )
            )
        if observation.joint_pos:
            parts.append(
                self._observation_component(
                    joint_pos, noise.joint_pos, scale.joint_pos, keys[6]
                )
            )
        if observation.joint_vel:
            parts.append(
                self._observation_component(
                    joint_vel, noise.joint_vel, scale.joint_vel, keys[7]
                )
            )
        if observation.actuator_force:
            parts.append(
                self._observation_component(
                    actuator_force,
                    noise.actuator_force,
                    scale.actuator_force,
                    keys[8],
                )
            )
        if observation.last_action:
            parts.append(
                self._observation_component(
                    last_action, noise.last_action, scale.last_action, keys[9]
                )
            )
        if not parts:
            return jnp.zeros((0,), dtype=jnp.float32)
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

    def _action_to_ctrl(self, action: jax.Array) -> jax.Array:
        default = jnp.asarray(self.model.default_ctrl)
        scale = self._action_scale()
        policy_indices = jnp.asarray(self.model.policy_ctrl_indices)
        ctrl = default.at[policy_indices].set(default[policy_indices] + scale * action)
        ctrlrange = jnp.asarray(self.model.mj_model.actuator_ctrlrange)
        return jnp.clip(ctrl, ctrlrange[:, 0], ctrlrange[:, 1])

    def _action_scale(self) -> jax.Array:
        return jnp.asarray(self.model.policy_action_scale, dtype=jnp.float32)

    def _resolve_command(
        self,
        command: jax.Array | None,
        rng: jax.Array | None,
        command_ranges: jax.Array | None = None,
    ) -> jax.Array:
        if not self.command_config.enabled:
            return jnp.zeros(3, dtype=jnp.float32)
        if command is not None:
            command = jnp.asarray(command, dtype=jnp.float32)
            if command.shape != (3,):
                raise ValueError(f"command must have shape (3,), got {command.shape}")
            return command
        if rng is not None:
            return self.command_config.sample(rng, ranges=command_ranges)
        return jnp.zeros(3, dtype=jnp.float32)

    def _resolve_batch_command(
        self,
        batch_size: int,
        command: jax.Array | None,
        rng: jax.Array | None,
        command_ranges: jax.Array | None = None,
    ) -> jax.Array:
        if not self.command_config.enabled:
            return jnp.zeros((batch_size, 3), dtype=jnp.float32)
        if command is None:
            if rng is not None:
                return self.command_config.sample(
                    rng, (batch_size,), ranges=command_ranges
                )
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
        command_ranges: jax.Array | None = None,
    ) -> jax.Array:
        if rng is None or not self.command_config.enabled:
            return state.command
        resample_steps = max(
            1, int(round(self.command_config.resample_seconds / self.control_timestep))
        )
        resample = (state.step_count % resample_steps == 0) & ~done
        sampled = self.command_config.sample(
            rng, (int(state.command.shape[0]),), ranges=command_ranges
        )
        return jnp.where(resample[:, None], sampled, state.command)

    def _reset_batch_command(
        self,
        command: jax.Array,
        batch_size: int,
        rng: jax.Array | None,
        command_ranges: jax.Array | None = None,
    ) -> jax.Array:
        if rng is None:
            return command
        return self.command_config.sample(rng, (batch_size,), ranges=command_ranges)

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
        return self.decimation * SIM_TIMESTEP

    def _frame_terms(self, state: BelloState) -> tuple[jax.Array, ...]:
        data = state.data
        qpos_idx = jnp.asarray(self._observed_qpos_indices())
        qvel_idx = jnp.asarray(self._observed_qvel_indices())
        actuator_idx = jnp.asarray(self._observed_actuator_indices())
        joint_pos = data.qpos[qpos_idx] - jnp.asarray(self.model.default_qpos)[qpos_idx]
        joint_vel = data.qvel[qvel_idx]
        actuator_force = data.actuator_force[actuator_idx]
        base_lin_vel = self._link_lin_vel(state, self.config.root_link)
        base_ang_vel = self._link_ang_vel(state, self.config.root_link)
        base_height = jnp.asarray(
            [self._link_height(state, self.config.root_link)], dtype=jnp.float32
        )
        projected_gravity = self._link_projected_gravity(state, self.config.root_link)
        gait_phase = self._gait_phase_sin_cos(state)
        return (
            base_lin_vel,
            base_ang_vel,
            projected_gravity,
            base_height,
            gait_phase,
            state.command,
            joint_pos,
            joint_vel,
            actuator_force,
            state.last_action,
        )

    def _observed_qpos_indices(self):
        if self.config.observation.joint_mode == "policy":
            return self.model.policy_qpos_indices
        return self.model.actuator_qpos_indices

    def _observed_qvel_indices(self):
        if self.config.observation.joint_mode == "policy":
            return self.model.policy_qvel_indices
        return self.model.actuator_qvel_indices

    def _observed_actuator_indices(self):
        if self.config.observation.joint_mode == "policy":
            return self.model.policy_ctrl_indices
        return np.arange(self.model.mj_model.nu, dtype=np.int32)

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
        if link_name == "bello_root":
            return state.data.qpos[int(self.model.root_qpos_adr) + 2]
        return state.data.xpos[self._link_body_id(link_name), 2]

    def _link_lin_vel(self, state: BelloState, link_name: str) -> jax.Array:
        return self._link_body_lin_vel(state, link_name)

    def _link_body_lin_vel(self, state: BelloState, link_name: str) -> jax.Array:
        if link_name == "bello_root":
            root_qpos = int(self.model.root_qpos_adr)
            root_qvel = int(self.model.root_qvel_adr)
            quat = state.data.qpos[root_qpos + 3 : root_qpos + 7]
            return quat_rotate_inverse(quat, state.data.qvel[root_qvel : root_qvel + 3])
        return state.data.cvel[self._link_body_id(link_name), 3:6]

    def _link_world_lin_vel(self, state: BelloState, link_name: str) -> jax.Array:
        if link_name == "bello_root":
            root_qvel = int(self.model.root_qvel_adr)
            return state.data.qvel[root_qvel : root_qvel + 3]
        return quat_rotate(
            self._link_quat(state, link_name), self._link_body_lin_vel(state, link_name)
        )

    def _link_yaw_frame_lin_vel(self, state: BelloState, link_name: str) -> jax.Array:
        return quat_rotate_inverse(
            yaw_quat(self._link_quat(state, link_name)),
            self._link_world_lin_vel(state, link_name),
        )

    def _link_ang_vel(self, state: BelloState, link_name: str) -> jax.Array:
        return self._link_body_ang_vel(state, link_name)

    def _link_body_ang_vel(self, state: BelloState, link_name: str) -> jax.Array:
        if link_name == "bello_root":
            root_qpos = int(self.model.root_qpos_adr)
            root_qvel = int(self.model.root_qvel_adr)
            quat = state.data.qpos[root_qpos + 3 : root_qpos + 7]
            return quat_rotate_inverse(
                quat, state.data.qvel[root_qvel + 3 : root_qvel + 6]
            )
        return state.data.cvel[self._link_body_id(link_name), :3]

    def _link_projected_gravity(self, state: BelloState, link_name: str) -> jax.Array:
        return quat_rotate_inverse(
            self._link_quat(state, link_name),
            jnp.asarray([0.0, 0.0, -1.0], dtype=jnp.float32),
        )

    def _done(self, state: BelloState) -> jax.Array:
        return self._done_flags(state)["done/any"]

    def _done_flags(self, state: BelloState) -> dict[str, jax.Array]:
        config = self.done_config
        quat = self._link_quat(state, self._done_link(config.roll_pitch))
        roll, pitch = roll_pitch_from_quat(quat)
        bad_height = (
            self._link_height(state, self._done_link(config.base_height))
            < config.base_height.min_height
            if config.base_height.enabled
            else jnp.asarray(False)
        )
        bad_orientation = (
            (jnp.abs(roll) > config.roll_pitch.max_abs_roll)
            | (jnp.abs(pitch) > config.roll_pitch.max_abs_pitch)
            if config.roll_pitch.enabled
            else jnp.asarray(False)
        )
        bad_pose = bad_height | bad_orientation
        timeout = (
            state.step_count >= self.max_episode_steps
            if config.timeout.enabled
            else jnp.asarray(False)
        )
        finite = jnp.all(jnp.isfinite(state.data.qpos)) & jnp.all(
            jnp.isfinite(state.data.qvel)
        )
        nonfinite = ~finite if config.nonfinite.enabled else jnp.asarray(False)
        terminal = bad_pose | nonfinite
        done = terminal | timeout
        return {
            "done/any": done,
            "done/terminal": terminal,
            "done/base_height": bad_height,
            "done/roll_pitch": bad_orientation,
            "done/timeout": timeout,
            "done/nonfinite": nonfinite,
        }

    def _base_height(self, state: BelloState) -> jax.Array:
        return self._link_height(
            state, self._reward_link(self.reward_config.base_height)
        )

    def _reward(
        self,
        state: BelloState,
        next_state: BelloState,
        action: jax.Array,
        done: jax.Array,
        foot_contact: jax.Array,
        foot_height: jax.Array,
        foot_lateral_speed_sq: jax.Array,
        undesired_contact: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        command = next_state.command
        qpos_idx = jnp.asarray(self.model.actuator_qpos_indices)
        joint_pos_rel, joint_vel = self._all_actuator_joint_terms(next_state)
        _, last_joint_vel = self._all_actuator_joint_terms(state)
        lin_vel_xy = self._link_yaw_frame_lin_vel(
            next_state, self._reward_link(self.reward_config.track_lin_vel_xy)
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
        base_height = self._link_height(
            next_state, self._reward_link(self.reward_config.base_height)
        )
        return compute_velocity_reward(
            config=self.reward_config,
            command=command,
            lin_vel_xy=lin_vel_xy,
            yaw_vel_z=yaw_vel_z,
            lin_vel_z=lin_vel_z,
            ang_vel_xy=ang_vel_xy,
            projected_gravity=projected_gravity,
            base_height=base_height,
            joint_pos=next_state.data.qpos[qpos_idx],
            joint_pos_rel=joint_pos_rel,
            joint_vel=joint_vel,
            last_joint_vel=last_joint_vel,
            action=action,
            last_action=state.last_action,
            actuator_force=next_state.data.actuator_force,
            joint_limit_low=self._actuator_joint_limit_low(),
            joint_limit_high=self._actuator_joint_limit_high(),
            hold_joint_indices=jnp.asarray(self.model.hold_ctrl_indices),
            policy_joint_indices=jnp.asarray(self.model.policy_ctrl_indices),
            foot_contact=foot_contact,
            foot_height=foot_height,
            foot_lateral_speed_sq=foot_lateral_speed_sq,
            foot_distance=self._foot_distance(next_state),
            gait_contact_mask=self._gait_contact_mask(next_state),
            foot_air_time=state.foot_air_time,
            last_foot_contact=state.last_foot_contact,
            undesired_contact=undesired_contact,
            control_timestep=self.control_timestep,
            done=done,
        )

    def _gait_phase(self, state: BelloState) -> jax.Array:
        phase = (
            state.step_count.astype(jnp.float32)
            * self.control_timestep
            / self.config.gait.cycle_time
        )
        return jnp.mod(phase, 1.0)

    def _gait_phase_sin_cos(self, state: BelloState) -> jax.Array:
        angle = 2.0 * jnp.pi * self._gait_phase(state)
        return jnp.stack([jnp.sin(angle), jnp.cos(angle)]).astype(jnp.float32)

    def _gait_contact_mask(self, state: BelloState) -> jax.Array:
        foot_count = self.model.foot_body_ids.size
        phase_offsets = jnp.asarray(self.config.gait.phase_offsets, dtype=jnp.float32)[
            :foot_count
        ]
        phase = jnp.mod(self._gait_phase(state) + phase_offsets, 1.0)
        scheduled = phase < self.config.gait.duty_factor
        standing = (
            jnp.linalg.norm(state.command)
            <= self.config.gait.standing_command_threshold
        )
        return jnp.where(standing, jnp.ones_like(scheduled), scheduled)

    def _debounce_foot_contact(
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

    def _next_foot_memory(
        self, state: BelloState, foot_contact: jax.Array, done: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        contact = foot_contact.astype(bool)
        next_air_time = jnp.where(
            contact,
            jnp.zeros_like(state.foot_air_time),
            state.foot_air_time + self.control_timestep,
        )
        reset_air_time = jnp.zeros_like(next_air_time)
        reset_contact = jnp.ones_like(contact)
        return (
            jnp.where(done, reset_air_time, next_air_time),
            jnp.where(done, reset_contact, contact),
        )

    def _reward_link(self, term) -> str:
        if term.enabled and term.link is not None:
            return term.link
        return self.config.root_link

    def _done_link(self, term) -> str:
        if term.enabled and term.link is not None:
            return term.link
        return self.config.root_link

    def _contact_terms(
        self, state: BelloState, world_id: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        foot_body_ids = jnp.asarray(self.model.foot_body_ids)
        foot_count = foot_body_ids.shape[0]
        if self.model.terrain_geom_ids.size == 0:
            return (
                jnp.zeros(foot_count, dtype=jnp.float32),
                self._foot_height(state),
                jnp.zeros(foot_count, dtype=jnp.float32),
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

        geom_body_ids = jnp.asarray(self.model.geom_body_ids)
        contact_body = geom_body_ids[contact_geom]
        terrain_contact = pair_contains(contact_geom, self.model.terrain_geom_ids)
        foot_contact_slot = (
            active
            & terrain_contact
            & pair_contains(contact_body, self.model.foot_body_ids)
        )
        foot_contact = jnp.any(
            foot_contact_slot[:, None]
            & pair_contains_each(contact_body, self.model.foot_body_ids),
            axis=0,
        ).astype(jnp.float32)
        undesired_contact = jnp.sum(
            (
                active
                & terrain_contact
                & pair_contains(contact_geom, self.model.nonfoot_robot_geom_ids)
            ).astype(jnp.float32)
        )
        return (
            foot_contact,
            self._foot_height(state),
            self._foot_lateral_speed_sq(state),
            undesired_contact,
        )

    def _foot_height(self, state: BelloState) -> jax.Array:
        terrain_z = jnp.asarray(self.terrain_config.pos[2], dtype=jnp.float32)
        return state.data.xpos[jnp.asarray(self.model.foot_body_ids), 2] - terrain_z

    def _foot_lateral_speed_sq(self, state: BelloState) -> jax.Array:
        foot_body_ids = jnp.asarray(self.model.foot_body_ids)
        foot_cvel = state.data.cvel[foot_body_ids, 3:6]
        foot_quat = state.data.xquat[foot_body_ids]
        foot_world_vel = jax.vmap(quat_rotate)(foot_quat, foot_cvel)
        return jnp.sum(foot_world_vel[:, :2] ** 2, axis=1)

    def _foot_distance(self, state: BelloState) -> jax.Array:
        if self.model.foot_body_ids.size != 2:
            return jnp.asarray(0.0, dtype=jnp.float32)
        foot_xy = state.data.xpos[jnp.asarray(self.model.foot_body_ids), :2]
        return jnp.linalg.norm(foot_xy[0] - foot_xy[1])

    def _actuator_joint_limit_low(self) -> jax.Array:
        joint_ids = self.model.mj_model.actuator_trnid[:, 0]
        return jnp.asarray(self.model.mj_model.jnt_range[joint_ids, 0])

    def _actuator_joint_limit_high(self) -> jax.Array:
        joint_ids = self.model.mj_model.actuator_trnid[:, 0]
        return jnp.asarray(self.model.mj_model.jnt_range[joint_ids, 1])


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


def roll_pitch_from_quat(quat_wxyz: jax.Array) -> tuple[jax.Array, jax.Array]:
    w, x, y, z = quat_wxyz
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = jnp.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = jnp.arcsin(jnp.clip(sinp, -1.0, 1.0))
    return roll, pitch


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
