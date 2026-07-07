"""Minimal full-body Bello MuJoCo/MJX environment scaffold."""

from __future__ import annotations

from dataclasses import dataclass
import chex
import jax
import jax.numpy as jnp
from mujoco import mjx

from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig, PreparedAsset
from bello_mujoco.contracts import (
    ACTION_CLIP,
    ACTION_SCALE,
    CONTROL_DECIMATION,
    MAX_EPISODE_STEPS,
    POLICY_JOINT_ORDER,
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
    require_full_body_config,
)


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
class DoneConfig:
    """Termination thresholds for Bello episodes."""

    min_base_height: float = 0.80
    max_abs_roll: float = 1.0
    max_abs_pitch: float = 1.0
    terminate_on_timeout: bool = True
    terminate_on_nonfinite: bool = True


@dataclass(frozen=True)
class BelloLocomotionEnv:
    """Full-body sim scaffold with actuator-order action and observation.

    Action is full-body actuator residuals in MuJoCo actuator order. Zero action
    keeps every actuator at its default target.
    """

    model: BelloModel
    decimation: int = CONTROL_DECIMATION
    max_episode_steps: int = MAX_EPISODE_STEPS
    reward_config: RewardConfig = RewardConfig()
    command_config: CommandConfig = CommandConfig()
    done_config: DoneConfig = DoneConfig()

    @classmethod
    def load(
        cls,
        config: AssetPrepConfig = AssetPrepConfig(),
        *,
        warp: WarpConfig = WarpConfig(),
        batch_size: int = 1,
    ) -> "BelloLocomotionEnv":
        require_full_body_config(config)
        return cls(load_bello_model(config, warp=warp, batch_size=batch_size))

    @classmethod
    def load_prepared(
        cls,
        prepared: PreparedAsset,
        *,
        warp: WarpConfig = WarpConfig(),
        batch_size: int = 1,
    ) -> "BelloLocomotionEnv":
        if prepared.name != "full_body":
            raise ValueError("BelloLocomotionEnv currently requires a full_body asset")
        return cls(
            load_prepared_bello_model(prepared, warp=warp, batch_size=batch_size)
        )

    @property
    def action_size(self) -> int:
        return int(self.model.mj_model.nu)

    @property
    def observation_size(self) -> int:
        return 9 + 3 * self.action_size

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
        base_ang_vel, projected_gravity, command, joint_pos, joint_vel, last_action = (
            self._frame_terms(state)
        )
        return jnp.concatenate(
            [
                base_ang_vel,
                projected_gravity,
                command,
                joint_pos,
                joint_vel,
                last_action,
            ]
        )

    def _action_to_ctrl(self, action: jax.Array) -> jax.Array:
        default = jnp.asarray(self.model.default_ctrl)
        scale = self._action_scale()
        ctrl = default + scale * action
        ctrlrange = jnp.asarray(self.model.mj_model.actuator_ctrlrange)
        return jnp.clip(ctrl, ctrlrange[:, 0], ctrlrange[:, 1])

    def _action_scale(self) -> jax.Array:
        ctrlrange = jnp.asarray(self.model.mj_model.actuator_ctrlrange)
        scale = 0.25 * (ctrlrange[:, 1] - ctrlrange[:, 0])
        lower_scale = jnp.asarray(
            [ACTION_SCALE[name] for name in POLICY_JOINT_ORDER], dtype=jnp.float32
        )
        return scale.at[jnp.asarray(self.model.policy_ctrl_indices)].set(lower_scale)

    def _resolve_command(
        self, command: jax.Array | None, rng: jax.Array | None
    ) -> jax.Array:
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
        qpos_idx = jnp.asarray(self.model.actuator_qpos_indices)
        qvel_idx = jnp.asarray(self.model.actuator_qvel_indices)
        joint_pos = data.qpos[qpos_idx] - jnp.asarray(self.model.default_qpos)[qpos_idx]
        joint_vel = data.qvel[qvel_idx]
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
            state.last_action,
        )

    def _done(self, state: BelloState) -> jax.Array:
        config = self.done_config
        root_qpos = int(self.model.root_qpos_adr)
        quat = state.data.qpos[root_qpos + 3 : root_qpos + 7]
        roll, pitch = roll_pitch_from_quat(quat)
        bad_pose = (
            (self._base_height(state) < config.min_base_height)
            | (jnp.abs(roll) > config.max_abs_roll)
            | (jnp.abs(pitch) > config.max_abs_pitch)
        )
        timeout = (
            state.step_count >= self.max_episode_steps
            if config.terminate_on_timeout
            else jnp.asarray(False)
        )
        finite = jnp.all(jnp.isfinite(state.data.qpos)) & jnp.all(
            jnp.isfinite(state.data.qvel)
        )
        nonfinite = ~finite if config.terminate_on_nonfinite else jnp.asarray(False)
        return bad_pose | timeout | nonfinite

    def _base_height(self, state: BelloState) -> jax.Array:
        return state.data.qpos[int(self.model.root_qpos_adr) + 2]

    def _reward(
        self,
        state: BelloState,
        next_state: BelloState,
        action: jax.Array,
        done: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        (
            base_ang_vel,
            projected_gravity,
            command,
            joint_pos,
            joint_vel,
            _,
        ) = self._frame_terms(next_state)
        root_qpos = int(self.model.root_qpos_adr)
        root_qvel = int(self.model.root_qvel_adr)
        quat = next_state.data.qpos[root_qpos + 3 : root_qpos + 7]
        qpos_idx = jnp.asarray(self.model.actuator_qpos_indices)
        base_lin_vel = quat_rotate_inverse(
            quat, next_state.data.qvel[root_qvel : root_qvel + 3]
        )
        return compute_velocity_reward(
            config=self.reward_config,
            command=command,
            base_lin_vel=base_lin_vel,
            base_ang_vel=base_ang_vel,
            projected_gravity=projected_gravity,
            base_height=self._base_height(next_state),
            joint_pos=next_state.data.qpos[qpos_idx],
            joint_pos_rel=joint_pos,
            joint_vel=joint_vel,
            action=action,
            last_action=state.last_action,
            actuator_force=next_state.data.actuator_force,
            joint_limit_low=self._actuator_joint_limit_low(),
            joint_limit_high=self._actuator_joint_limit_high(),
            fixed_joint_indices=jnp.asarray(self.model.fixed_ctrl_indices),
            done=done,
        )

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
