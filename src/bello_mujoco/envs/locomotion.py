"""Minimal full-body Bello MuJoCo/MJX environment scaffold."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
)
from bello_mujoco.envs.rewards import (
    CommandConfig,
    RewardConfig,
    compute_velocity_reward,
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
    command: jax.Array
    step_count: jax.Array
    episode_return: jax.Array
    episode_length: jax.Array
    last_episode_return: jax.Array
    last_episode_length: jax.Array


@dataclass(frozen=True)
class BaseDoneConfig:
    enabled: bool = True


@dataclass(frozen=True)
class BaseHeightDoneConfig(BaseDoneConfig):
    min_height: float = 0.80
    link: str = "bello_root"


@dataclass(frozen=True)
class RollPitchDoneConfig(BaseDoneConfig):
    max_abs_roll: float = 1.0
    max_abs_pitch: float = 1.0
    link: str = "bello_root"


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
                    if term.enabled
                }
            )
        )


@dataclass(frozen=True)
class ObservationConfig:
    joint_mode: Literal["all_actuated", "policy"] = "all_actuated"
    base_ang_vel: bool = True
    projected_gravity: bool = True
    command: bool = True
    joint_pos: bool = True
    joint_vel: bool = True
    actuator_force: bool = True
    last_action: bool = True


@dataclass(frozen=True)
class BelloEnvConfig:
    """Runtime environment settings that affect stepping and model loading."""

    decimation: int = CONTROL_DECIMATION
    max_episode_steps: int = MAX_EPISODE_STEPS
    terrain: TerrainConfig = FLAT_TERRAIN
    observation: ObservationConfig = ObservationConfig()


@dataclass(frozen=True)
class BelloLocomotionEnv:
    """Full-body sim scaffold with actuator-order action and observation.

    Action is full-body actuator residuals in MuJoCo actuator order. Zero action
    keeps every actuator at its default target.
    """

    model: BelloModel
    config: BelloEnvConfig = BelloEnvConfig()
    reward_config: RewardConfig = RewardConfig()
    command_config: CommandConfig = CommandConfig()
    done_config: DoneConfig = DoneConfig()

    def __post_init__(self) -> None:
        self._validate_model_metadata()

    @classmethod
    def load(
        cls,
        config: AssetPrepConfig = AssetPrepConfig(),
        *,
        warp: WarpConfig = WarpConfig(),
        batch_size: int = 1,
        env_config: BelloEnvConfig = BelloEnvConfig(),
        terrain: TerrainConfig | None = None,
        reward_config: RewardConfig = RewardConfig(),
        command_config: CommandConfig = CommandConfig(),
        done_config: DoneConfig = DoneConfig(),
    ) -> "BelloLocomotionEnv":
        env_config = (
            env_config if terrain is None else replace(env_config, terrain=terrain)
        )
        return cls(
            load_bello_model(
                config,
                warp=warp,
                batch_size=batch_size,
                terrain=env_config.terrain,
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
        terrain: TerrainConfig | None = None,
        actuator_config: ActuatorConfig = ActuatorConfig(),
        reward_config: RewardConfig = RewardConfig(),
        command_config: CommandConfig = CommandConfig(),
        done_config: DoneConfig = DoneConfig(),
    ) -> "BelloLocomotionEnv":
        env_config = (
            env_config if terrain is None else replace(env_config, terrain=terrain)
        )
        return cls(
            load_prepared_bello_model(
                prepared,
                warp=warp,
                batch_size=batch_size,
                terrain=env_config.terrain,
                actuator_config=actuator_config,
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
    def action_size(self) -> int:
        return int(self.model.policy_ctrl_indices.size)

    @property
    def observation_size(self) -> int:
        observation = self.config.observation
        return (
            (3 if observation.base_ang_vel else 0)
            + (3 if observation.projected_gravity else 0)
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
        for link_name in (
            *self.reward_config.enabled_link_names,
            *self.done_config.enabled_link_names,
        ):
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
                f"{model.foot_body_ids.shape}. Disable foot_contact, foot_slip, and "
                "foot_clearance rewards for assets without feet."
            )
        if self.reward_config.uses_foot_metadata and model.foot_geom_ids.size == 0:
            raise ValueError(
                "enabled foot reward terms require foot collision geoms. Disable "
                "foot_contact, foot_slip, and foot_clearance rewards for assets "
                "without feet."
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
        self, command: jax.Array | None = None, rng: jax.Array | None = None
    ) -> tuple[BelloState, jax.Array]:
        command = self._resolve_command(command, rng)
        data = self.model.make_data()
        data = data.replace(
            qpos=jnp.asarray(self.model.default_qpos),
            qvel=jnp.asarray(self.model.default_qvel),
            ctrl=jnp.asarray(self.model.default_ctrl),
        )
        state = BelloState(
            data=data,
            last_action=jnp.zeros(self.action_size, dtype=jnp.float32),
            command=command,
            step_count=jnp.asarray(0, dtype=jnp.int32),
            episode_return=jnp.asarray(0.0, dtype=jnp.float32),
            episode_length=jnp.asarray(0, dtype=jnp.int32),
            last_episode_return=jnp.asarray(0.0, dtype=jnp.float32),
            last_episode_length=jnp.asarray(0, dtype=jnp.int32),
        )
        return state, self.observation(state)

    def reset_batch(
        self,
        batch_size: int,
        command: jax.Array | None = None,
        rng: jax.Array | None = None,
    ) -> tuple[BelloState, jax.Array]:
        command = self._resolve_batch_command(batch_size, command, rng)
        return jax.vmap(self.reset)(command)

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
            command=state.command,
            step_count=state.step_count + 1,
            episode_return=state.episode_return,
            episode_length=state.episode_length,
            last_episode_return=state.last_episode_return,
            last_episode_length=state.last_episode_length,
        )
        done = self._done(next_state)
        reward, reward_info = self._reward(state, next_state, action, done)
        episode_return = state.episode_return + reward
        episode_length = state.episode_length + 1
        next_state = BelloState(
            data=next_state.data,
            last_action=next_state.last_action,
            command=next_state.command,
            step_count=next_state.step_count,
            episode_return=episode_return,
            episode_length=episode_length,
            last_episode_return=jnp.where(
                done, episode_return, state.last_episode_return
            ),
            last_episode_length=jnp.where(
                done, episode_length, state.last_episode_length
            ),
        )
        info = {
            "base_height": self._base_height(next_state),
            "episode_return": jnp.where(done, episode_return, 0.0),
            "episode_length": jnp.where(done, episode_length, 0),
            "returned_episode": done,
            **reward_info,
        }
        return next_state, self.observation(next_state), reward, done, info

    def step_batch(
        self,
        state: BelloState,
        action: jax.Array,
        rng: jax.Array | None = None,
    ) -> tuple[BelloState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        stepped_state, stepped_obs, reward, done, info = jax.vmap(self.step)(
            state, action
        )
        batch_size = int(action.shape[0])
        reset_command = (
            stepped_state.command
            if rng is None
            else self.command_config.sample(rng, (batch_size,))
        )
        reset_state, reset_obs = self.reset_batch(batch_size, command=reset_command)
        next_state = tree_where(done, reset_state, stepped_state)
        next_obs = jnp.where(done[:, None], reset_obs, stepped_obs)
        return next_state, next_obs, reward, done, info

    def observation(self, state: BelloState) -> jax.Array:
        (
            base_ang_vel,
            projected_gravity,
            command,
            joint_pos,
            joint_vel,
            actuator_force,
            last_action,
        ) = self._frame_terms(state)
        observation = self.config.observation
        parts = []
        if observation.base_ang_vel:
            parts.append(base_ang_vel)
        if observation.projected_gravity:
            parts.append(projected_gravity)
        if observation.command and self.command_config.enabled:
            parts.append(command)
        if observation.joint_pos:
            parts.append(joint_pos)
        if observation.joint_vel:
            parts.append(joint_vel)
        if observation.actuator_force:
            parts.append(actuator_force)
        if observation.last_action:
            parts.append(last_action)
        if not parts:
            return jnp.zeros((0,), dtype=jnp.float32)
        return jnp.concatenate(parts)

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
        self, command: jax.Array | None, rng: jax.Array | None
    ) -> jax.Array:
        if not self.command_config.enabled:
            return jnp.zeros(3, dtype=jnp.float32)
        if command is not None:
            return jnp.asarray(command, dtype=jnp.float32)
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
        return command

    def _frame_terms(self, state: BelloState) -> tuple[jax.Array, ...]:
        data = state.data
        qpos_idx = jnp.asarray(self._observed_qpos_indices())
        qvel_idx = jnp.asarray(self._observed_qvel_indices())
        actuator_idx = jnp.asarray(self._observed_actuator_indices())
        joint_pos = data.qpos[qpos_idx] - jnp.asarray(self.model.default_qpos)[qpos_idx]
        joint_vel = data.qvel[qvel_idx]
        actuator_force = data.actuator_force[actuator_idx]
        root_qpos = int(self.model.root_qpos_adr)
        root_qvel = int(self.model.root_qvel_adr)
        quat = data.qpos[root_qpos + 3 : root_qpos + 7]
        base_ang_vel = data.qvel[root_qvel + 3 : root_qvel + 6]
        projected_gravity = quat_rotate_inverse(
            quat, jnp.asarray([0.0, 0.0, -1.0], dtype=jnp.float32)
        )
        return (
            base_ang_vel,
            projected_gravity,
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
        if link_name == "bello_root":
            root_qpos = int(self.model.root_qpos_adr)
            root_qvel = int(self.model.root_qvel_adr)
            quat = state.data.qpos[root_qpos + 3 : root_qpos + 7]
            return quat_rotate_inverse(quat, state.data.qvel[root_qvel : root_qvel + 3])
        return state.data.cvel[self._link_body_id(link_name), 3:6]

    def _link_ang_vel(self, state: BelloState, link_name: str) -> jax.Array:
        if link_name == "bello_root":
            root_qvel = int(self.model.root_qvel_adr)
            return state.data.qvel[root_qvel + 3 : root_qvel + 6]
        return state.data.cvel[self._link_body_id(link_name), :3]

    def _link_projected_gravity(self, state: BelloState, link_name: str) -> jax.Array:
        return quat_rotate_inverse(
            self._link_quat(state, link_name),
            jnp.asarray([0.0, 0.0, -1.0], dtype=jnp.float32),
        )

    def _done(self, state: BelloState) -> jax.Array:
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
        return bad_pose | timeout | nonfinite

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
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        _, _, command, _, _, _, _ = self._frame_terms(next_state)
        qpos_idx = jnp.asarray(self.model.actuator_qpos_indices)
        joint_pos_rel, joint_vel = self._all_actuator_joint_terms(next_state)
        foot_contact, foot_height, foot_lateral_speed_sq, undesired_contact = (
            self._contact_terms(next_state)
        )
        lin_vel_xy = self._link_lin_vel(
            next_state, self._reward_link(self.reward_config.track_lin_vel_xy)
        )[:2]
        yaw_vel_z = self._link_ang_vel(
            next_state, self._reward_link(self.reward_config.track_ang_vel_z)
        )[2]
        lin_vel_z = self._link_lin_vel(
            next_state, self._reward_link(self.reward_config.lin_vel_z)
        )[2]
        ang_vel_xy = self._link_ang_vel(
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
            action=action,
            last_action=state.last_action,
            actuator_force=next_state.data.actuator_force,
            joint_limit_low=self._actuator_joint_limit_low(),
            joint_limit_high=self._actuator_joint_limit_high(),
            hold_joint_indices=jnp.asarray(self.model.hold_ctrl_indices),
            foot_contact=foot_contact,
            foot_height=foot_height,
            foot_lateral_speed_sq=foot_lateral_speed_sq,
            undesired_contact=undesired_contact,
            done=done,
        )

    def _reward_link(self, term) -> str:
        return term.link if term.enabled else "bello_root"

    def _done_link(self, term) -> str:
        return term.link if term.enabled else "bello_root"

    def _contact_terms(
        self, state: BelloState
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
        active = jnp.arange(contact_geom.shape[0], dtype=jnp.int32) < nacon

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
        foot_cvel = state.data.cvel[jnp.asarray(self.model.foot_body_ids)]
        return jnp.sum(foot_cvel[:, 3:5] ** 2, axis=1)

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
