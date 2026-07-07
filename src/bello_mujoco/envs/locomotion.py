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


@dataclass(frozen=True)
class BelloLocomotionEnv:
    """Full-body sim scaffold with actuator-order action and observation.

    Action is full-body actuator residuals in MuJoCo actuator order. Zero action
    keeps every actuator at its default target.
    """

    model: BelloModel
    decimation: int = CONTROL_DECIMATION
    max_episode_steps: int = MAX_EPISODE_STEPS

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

    def reset(self, command: jax.Array | None = None) -> tuple[BelloState, jax.Array]:
        command = (
            jnp.zeros(3, dtype=jnp.float32)
            if command is None
            else jnp.asarray(command, dtype=jnp.float32)
        )
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
        )
        return state, self.observation(state)

    def reset_batch(
        self, batch_size: int, command: jax.Array | None = None
    ) -> tuple[BelloState, jax.Array]:
        if command is None:
            command = jnp.zeros((batch_size, 3), dtype=jnp.float32)
        else:
            command = jnp.asarray(command, dtype=jnp.float32)
            if command.shape == (3,):
                command = jnp.repeat(command[None, :], batch_size, axis=0)
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
        )
        reward = jnp.asarray(0.0, dtype=jnp.float32)
        done = self._done(next_state)
        info = {"base_height": self._base_height(next_state)}
        return next_state, self.observation(next_state), reward, done, info

    def step_batch(
        self, state: BelloState, action: jax.Array
    ) -> tuple[BelloState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        return jax.vmap(self.step)(state, action)

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
        root_qpos = int(self.model.root_qpos_adr)
        quat = state.data.qpos[root_qpos + 3 : root_qpos + 7]
        roll, pitch = roll_pitch_from_quat(quat)
        bad_pose = (
            (self._base_height(state) < 0.80)
            | (jnp.abs(roll) > 1.0)
            | (jnp.abs(pitch) > 1.0)
        )
        timeout = state.step_count >= self.max_episode_steps
        finite = jnp.all(jnp.isfinite(state.data.qpos)) & jnp.all(
            jnp.isfinite(state.data.qvel)
        )
        return bad_pose | timeout | (~finite)

    def _base_height(self, state: BelloState) -> jax.Array:
        return state.data.qpos[int(self.model.root_qpos_adr) + 2]


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
