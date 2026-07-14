"""Whole-body reference-motion tracking for fixed-waist Bello."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from bello_mujoco.actuators import ActuatorConfig
from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig, PreparedAsset
from bello_mujoco.contracts import CONTROL_DECIMATION, SIM_TIMESTEP
from bello_mujoco.envs.locomotion import (
    quat_rotate,
    quat_rotate_inverse,
    tree_where,
    yaw_quat,
    yaw_quat_from_angle,
)
from bello_mujoco.motion import (
    MotionClip,
    MotionSample,
    load_motion_clip,
    sample_motion,
)
from bello_mujoco.sim import (
    BelloModel,
    WarpConfig,
    load_bello_model,
    load_prepared_bello_model,
)
from bello_mujoco.terrain import FLAT_TERRAIN, TerrainConfig


@chex.dataclass(frozen=True)
class MotionTrackingState:
    data: mjx.Data
    last_action: jax.Array
    reference_time: jax.Array
    alignment_xy: jax.Array
    alignment_yaw: jax.Array
    step_count: jax.Array
    episode_return: jax.Array
    episode_length: jax.Array
    last_episode_return: jax.Array
    last_episode_length: jax.Array


@dataclass(frozen=True)
class MotionResetConfig:
    random_start: bool = True
    initial_xy_range: tuple[float, float] = (-1.0, 1.0)
    initial_yaw_range: tuple[float, float] = (-math.pi, math.pi)
    joint_pos_noise: tuple[float, float] = (-0.02, 0.02)
    joint_vel_noise: tuple[float, float] = (-0.1, 0.1)


@dataclass(frozen=True)
class MotionRewardConfig:
    root_position_weight: float = 0.15
    root_orientation_weight: float = 0.15
    joint_position_weight: float = 0.20
    joint_velocity_weight: float = 0.05
    body_position_weight: float = 0.20
    body_orientation_weight: float = 0.10
    body_velocity_weight: float = 0.05
    action_rate_weight: float = -0.01
    joint_limit_weight: float = -0.02
    root_position_sigma: float = 0.20
    root_orientation_sigma: float = 0.35
    joint_position_sigma: float = 0.35
    joint_velocity_sigma: float = 3.0
    body_position_sigma: float = 0.15
    body_orientation_sigma: float = 0.45
    body_velocity_sigma: float = 2.0


@dataclass(frozen=True)
class MotionDoneConfig:
    root_height_error: float = 0.35
    root_orientation_error: float = 1.2
    end_effector_position_error: float = 0.65


@dataclass(frozen=True)
class MotionTrackingEnvConfig:
    motion_path: Path = Path("motions/local/walk1_subject1_fixed_waist.npz")
    decimation: int = CONTROL_DECIMATION
    action_clip: float = 1.0
    terrain: TerrainConfig = FLAT_TERRAIN
    reset: MotionResetConfig = MotionResetConfig()
    reward: MotionRewardConfig = MotionRewardConfig()
    done: MotionDoneConfig = MotionDoneConfig()


@dataclass(frozen=True)
class BelloMotionTrackingEnv:
    model: BelloModel
    motion: MotionClip
    config: MotionTrackingEnvConfig = MotionTrackingEnvConfig()

    def __post_init__(self) -> None:
        if self.model.mjx_model is None:
            raise ValueError("motion tracking requires an MJX-loaded model")
        if self.model.actuator_joint_names != self.motion.joint_names:
            raise ValueError("model and motion actuator orders do not match")
        if self.model.mj_model.nq != 34 or self.model.mj_model.nv != 33:
            raise ValueError(
                "fixed-waist Bello must have nq=34 and nv=33, got "
                f"nq={self.model.mj_model.nq}, nv={self.model.mj_model.nv}"
            )
        if self.decimation <= 0:
            raise ValueError("decimation must be positive")
        if not math.isfinite(self.config.action_clip) or self.config.action_clip <= 0:
            raise ValueError("action_clip must be positive and finite")
        self._validate_range("initial_xy_range", self.config.reset.initial_xy_range)
        self._validate_range("initial_yaw_range", self.config.reset.initial_yaw_range)
        self._validate_range("joint_pos_noise", self.config.reset.joint_pos_noise)
        self._validate_range("joint_vel_noise", self.config.reset.joint_vel_noise)
        if self.motion.body_names[0] != "pelvis_link":
            raise ValueError("the first tracked body must be pelvis_link")
        for name in self.motion.body_names:
            if self._body_id(name) < 0:
                raise ValueError(f"tracked body not found in model: {name!r}")

    @classmethod
    def load(
        cls,
        asset_config: AssetPrepConfig,
        env_config: MotionTrackingEnvConfig,
        *,
        warp: WarpConfig = WarpConfig(),
        batch_size: int = 1,
        use_viewer_model: bool = False,
    ) -> "BelloMotionTrackingEnv":
        model = load_bello_model(
            asset_config,
            warp=warp,
            batch_size=batch_size,
            terrain=env_config.terrain,
            use_viewer_model=use_viewer_model,
        )
        return cls(model, load_motion_clip(env_config.motion_path, model), env_config)

    @classmethod
    def load_prepared(
        cls,
        prepared: PreparedAsset,
        env_config: MotionTrackingEnvConfig,
        *,
        actuator_config: ActuatorConfig = ActuatorConfig(),
        warp: WarpConfig = WarpConfig(),
        batch_size: int = 1,
        use_viewer_model: bool = False,
    ) -> "BelloMotionTrackingEnv":
        model = load_prepared_bello_model(
            prepared,
            warp=warp,
            batch_size=batch_size,
            terrain=env_config.terrain,
            actuator_config=actuator_config,
            use_viewer_model=use_viewer_model,
        )
        return cls(model, load_motion_clip(env_config.motion_path, model), env_config)

    @property
    def decimation(self) -> int:
        return self.config.decimation

    @property
    def control_timestep(self) -> float:
        return self.decimation * SIM_TIMESTEP

    @property
    def max_episode_steps(self) -> int:
        return max(1, math.ceil(self.motion.duration / self.control_timestep))

    @property
    def action_clip(self) -> float:
        return self.config.action_clip

    @property
    def action_size(self) -> int:
        return int(self.model.policy_ctrl_indices.size)

    @property
    def observation_size(self) -> int:
        return 5 * self.action_size + 14

    @property
    def critic_observation_size(self) -> int:
        return self.observation_size + 6 * len(self.motion.body_names)

    @property
    def tracked_body_ids(self) -> np.ndarray:
        return np.asarray([self._body_id(name) for name in self.motion.body_names])

    @property
    def end_effector_indices(self) -> np.ndarray:
        names = self.motion.body_names
        return np.asarray([names.index("l_wrist_link"), names.index("r_wrist_link")])

    def reset(
        self,
        rng: jax.Array | None = None,
        reference_time: jax.Array | None = None,
    ) -> tuple[MotionTrackingState, jax.Array]:
        time_key = pose_key = noise_key = None
        if rng is not None:
            time_key, pose_key, noise_key = jax.random.split(rng, 3)
        if reference_time is None:
            if self.config.reset.random_start and time_key is not None:
                max_time = max(self.motion.duration - self.control_timestep, 0.0)
                reference_time = jax.random.uniform(time_key, (), maxval=max_time)
            else:
                reference_time = jnp.asarray(0.0, dtype=jnp.float32)
        reference_time = jnp.asarray(reference_time, dtype=jnp.float32)
        source = sample_motion(self.motion, reference_time)
        desired_xy = jnp.zeros(2, dtype=jnp.float32)
        alignment_yaw = jnp.asarray(0.0, dtype=jnp.float32)
        if pose_key is not None:
            xy_key, yaw_key = jax.random.split(pose_key)
            desired_xy = jax.random.uniform(
                xy_key,
                (2,),
                minval=self.config.reset.initial_xy_range[0],
                maxval=self.config.reset.initial_xy_range[1],
            )
            alignment_yaw = jax.random.uniform(
                yaw_key,
                (),
                minval=self.config.reset.initial_yaw_range[0],
                maxval=self.config.reset.initial_yaw_range[1],
            )
        rotated_source_xy = quat_rotate(
            yaw_quat_from_angle(alignment_yaw),
            jnp.concatenate([source.qpos[:2], jnp.zeros(1)]),
        )[:2]
        alignment_xy = desired_xy - rotated_source_xy
        reference = self._aligned_reference(reference_time, alignment_xy, alignment_yaw)
        qpos, qvel = reference.qpos, reference.qvel
        if noise_key is not None:
            qpos_key, qvel_key = jax.random.split(noise_key)
            qpos_indices = jnp.asarray(self.model.actuator_qpos_indices)
            qvel_indices = jnp.asarray(self.model.actuator_qvel_indices)
            qpos = qpos.at[qpos_indices].add(
                jax.random.uniform(
                    qpos_key,
                    qpos_indices.shape,
                    minval=self.config.reset.joint_pos_noise[0],
                    maxval=self.config.reset.joint_pos_noise[1],
                )
            )
            qvel = qvel.at[qvel_indices].add(
                jax.random.uniform(
                    qvel_key,
                    qvel_indices.shape,
                    minval=self.config.reset.joint_vel_noise[0],
                    maxval=self.config.reset.joint_vel_noise[1],
                )
            )
        data = self.model.make_data().replace(
            qpos=qpos,
            qvel=qvel,
            ctrl=self._reference_ctrl(reference),
        )
        data = mjx.forward(self.model.mjx_model, data)
        state = MotionTrackingState(
            data=data,
            last_action=jnp.zeros(self.action_size, dtype=jnp.float32),
            reference_time=reference_time,
            alignment_xy=alignment_xy,
            alignment_yaw=alignment_yaw,
            step_count=jnp.asarray(0, dtype=jnp.int32),
            episode_return=jnp.asarray(0.0, dtype=jnp.float32),
            episode_length=jnp.asarray(0, dtype=jnp.int32),
            last_episode_return=jnp.asarray(0.0, dtype=jnp.float32),
            last_episode_length=jnp.asarray(0, dtype=jnp.int32),
        )
        return state, self.observation(state)

    def reset_batch(
        self, batch_size: int, rng: jax.Array | None = None
    ) -> tuple[MotionTrackingState, jax.Array]:
        if rng is None:
            state, _ = jax.vmap(lambda _: self.reset())(jnp.arange(batch_size))
        else:
            keys = jax.random.split(rng, batch_size)
            state, _ = jax.vmap(self.reset)(keys)
        data = state.data.replace(_impl=state.data._impl.replace(nworld=batch_size))
        state = state.replace(data=data)
        return state, self.observation_batch(state)

    def step(
        self, state: MotionTrackingState, action: jax.Array
    ) -> tuple[
        MotionTrackingState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]
    ]:
        action = jnp.clip(jnp.asarray(action), -self.action_clip, self.action_clip)
        data = state.data.replace(ctrl=self._action_to_ctrl(action))
        data = jax.lax.fori_loop(
            0,
            self.decimation,
            lambda _, value: mjx.step(self.model.mjx_model, value),
            data,
        )
        stepped = state.replace(
            data=data,
            last_action=action,
            reference_time=state.reference_time + self.control_timestep,
            step_count=state.step_count + 1,
            episode_length=state.episode_length + 1,
        )
        reward, reward_info = self._reward(state, stepped)
        done_flags = self._done_flags(stepped)
        reward, reward_info = self._sanitize_nonfinite_reward(
            reward, reward_info, done_flags
        )
        done = done_flags["done/any"]
        episode_return = state.episode_return + reward
        stepped = stepped.replace(
            episode_return=episode_return,
            last_episode_return=jnp.where(
                done, episode_return, state.last_episode_return
            ),
            last_episode_length=jnp.where(
                done, stepped.episode_length, state.last_episode_length
            ),
        )
        observation = self.observation(stepped)
        info = self._info(stepped, observation, reward_info, done_flags)
        return stepped, observation, reward, done, info

    def step_batch(
        self,
        state: MotionTrackingState,
        action: jax.Array,
        rng: jax.Array | None = None,
    ) -> tuple[
        MotionTrackingState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]
    ]:
        batch_size = int(action.shape[0])
        action = jnp.clip(jnp.asarray(action), -self.action_clip, self.action_clip)
        ctrl = jax.vmap(self._action_to_ctrl)(action)
        data = state.data.replace(
            ctrl=ctrl, _impl=state.data._impl.replace(nworld=batch_size)
        )
        data = jax.lax.fori_loop(
            0,
            self.decimation,
            lambda _, value: mjx.step(self.model.mjx_model, value),
            data,
        )
        stepped = state.replace(
            data=data,
            last_action=action,
            reference_time=state.reference_time + self.control_timestep,
            step_count=state.step_count + 1,
            episode_length=state.episode_length + 1,
        )
        reward, reward_info = jax.vmap(self._reward)(state, stepped)
        done_flags = jax.vmap(self._done_flags)(stepped)
        reward, reward_info = self._sanitize_nonfinite_reward(
            reward, reward_info, done_flags
        )
        done = done_flags["done/any"]
        episode_return = state.episode_return + reward
        stepped = stepped.replace(
            episode_return=episode_return,
            last_episode_return=jnp.where(
                done, episode_return, state.last_episode_return
            ),
            last_episode_length=jnp.where(
                done, stepped.episode_length, state.last_episode_length
            ),
        )
        stepped_obs = self.observation_batch(stepped)
        info = self._info(stepped, stepped_obs, reward_info, done_flags)
        reset_state, reset_obs = self.reset_batch(batch_size, rng)
        reset_state = reset_state.replace(
            last_episode_return=stepped.last_episode_return,
            last_episode_length=stepped.last_episode_length,
        )
        return (
            tree_where(done, reset_state, stepped),
            jnp.where(done[:, None], reset_obs, stepped_obs),
            reward,
            done,
            info,
        )

    def observation(self, state: MotionTrackingState) -> jax.Array:
        reference = self._aligned_reference(
            jnp.minimum(
                state.reference_time + self.control_timestep, self.motion.duration
            ),
            state.alignment_xy,
            state.alignment_yaw,
        )
        qpos_indices = jnp.asarray(self.model.policy_qpos_indices)
        qvel_indices = jnp.asarray(self.model.policy_qvel_indices)
        heading = yaw_quat(state.data.qpos[3:7])
        root_position_error = quat_rotate_inverse(
            heading, reference.qpos[:3] - state.data.qpos[:3]
        )
        root_orientation_error = quat_to_rotvec(
            quat_multiply(reference.qpos[3:7], quat_conjugate(state.data.qpos[3:7]))
        )
        root_velocity_error = quat_rotate_inverse(
            heading, reference.qvel[:3] - state.data.qvel[:3]
        )
        root_angular_velocity_error = quat_rotate_inverse(
            heading, reference.qvel[3:6] - state.data.qvel[3:6]
        )
        joint_default = jnp.asarray(self.model.default_qpos)[qpos_indices]
        phase = 2.0 * jnp.pi * state.reference_time / self.motion.duration
        return jnp.concatenate(
            [
                reference.qpos[qpos_indices] - joint_default,
                reference.qvel[qvel_indices],
                state.data.qpos[qpos_indices] - joint_default,
                state.data.qvel[qvel_indices],
                state.last_action,
                root_position_error,
                root_orientation_error,
                root_velocity_error,
                root_angular_velocity_error,
                jnp.asarray([jnp.sin(phase), jnp.cos(phase)]),
            ]
        )

    def observation_batch(self, state: MotionTrackingState) -> jax.Array:
        return jax.vmap(self.observation)(state)

    def critic_observation(self, state: MotionTrackingState) -> jax.Array:
        reference = self._aligned_reference(
            jnp.minimum(
                state.reference_time + self.control_timestep, self.motion.duration
            ),
            state.alignment_xy,
            state.alignment_yaw,
        )
        ids = jnp.asarray(self.tracked_body_ids)
        current_pos = state.data.xpos[ids]
        relative_position_error = (reference.body_pos - reference.body_pos[0]) - (
            current_pos - current_pos[0]
        )
        orientation_error = jax.vmap(
            lambda ref, current: quat_to_rotvec(
                quat_multiply(ref, quat_conjugate(current))
            )
        )(reference.body_quat, state.data.xquat[ids])
        return jnp.concatenate(
            [
                self.observation(state),
                relative_position_error.reshape(-1),
                orientation_error.reshape(-1),
            ]
        )

    def critic_observation_batch(self, state: MotionTrackingState) -> jax.Array:
        return jax.vmap(self.critic_observation)(state)

    def initial_step_counts(
        self, batch_size: int, rng: jax.Array | None = None
    ) -> jax.Array:
        del rng
        return jnp.zeros((batch_size,), dtype=jnp.int32)

    def _reward(
        self, previous: MotionTrackingState, state: MotionTrackingState
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        reference = self._aligned_reference(
            jnp.minimum(state.reference_time, self.motion.duration),
            state.alignment_xy,
            state.alignment_yaw,
        )
        qpos_indices = jnp.asarray(self.model.policy_qpos_indices)
        qvel_indices = jnp.asarray(self.model.policy_qvel_indices)
        ids = jnp.asarray(self.tracked_body_ids)
        current_pos = state.data.xpos[ids]
        body_position_error = (current_pos - current_pos[0]) - (
            reference.body_pos - reference.body_pos[0]
        )
        body_orientation_error = jax.vmap(quat_distance)(
            state.data.xquat[ids], reference.body_quat
        )
        body_velocity_error = self._body_world_velocity(state, ids) - jnp.concatenate(
            [reference.body_ang_vel, reference.body_lin_vel], axis=-1
        )
        cfg = self.config.reward
        components = {
            "reward/root_position": cfg.root_position_weight
            * gaussian_reward(
                state.data.qpos[:3] - reference.qpos[:3], cfg.root_position_sigma
            ),
            "reward/root_orientation": cfg.root_orientation_weight
            * gaussian_reward(
                quat_distance(state.data.qpos[3:7], reference.qpos[3:7]),
                cfg.root_orientation_sigma,
            ),
            "reward/joint_position": cfg.joint_position_weight
            * gaussian_reward(
                state.data.qpos[qpos_indices] - reference.qpos[qpos_indices],
                cfg.joint_position_sigma,
            ),
            "reward/joint_velocity": cfg.joint_velocity_weight
            * gaussian_reward(
                state.data.qvel[qvel_indices] - reference.qvel[qvel_indices],
                cfg.joint_velocity_sigma,
            ),
            "reward/body_position": cfg.body_position_weight
            * gaussian_reward(body_position_error, cfg.body_position_sigma),
            "reward/body_orientation": cfg.body_orientation_weight
            * gaussian_reward(body_orientation_error, cfg.body_orientation_sigma),
            "reward/body_velocity": cfg.body_velocity_weight
            * gaussian_reward(body_velocity_error, cfg.body_velocity_sigma),
            "reward/action_rate": cfg.action_rate_weight
            * jnp.mean((state.last_action - previous.last_action) ** 2),
            "reward/joint_limit": cfg.joint_limit_weight
            * self._joint_limit_cost(state),
        }
        return jnp.sum(jnp.stack(tuple(components.values()))), components

    def _done_flags(self, state: MotionTrackingState) -> dict[str, jax.Array]:
        reference = self._aligned_reference(
            jnp.minimum(state.reference_time, self.motion.duration),
            state.alignment_xy,
            state.alignment_yaw,
        )
        ids = jnp.asarray(self.tracked_body_ids)
        height = (
            jnp.abs(state.data.qpos[2] - reference.qpos[2])
            > self.config.done.root_height_error
        )
        orientation = (
            quat_distance(state.data.qpos[3:7], reference.qpos[3:7])
            > self.config.done.root_orientation_error
        )
        ee_error = jnp.linalg.norm(
            state.data.xpos[ids[jnp.asarray(self.end_effector_indices)]]
            - reference.body_pos[jnp.asarray(self.end_effector_indices)],
            axis=-1,
        )
        end_effector = jnp.any(ee_error > self.config.done.end_effector_position_error)
        nonfinite = ~(
            jnp.all(jnp.isfinite(state.data.qpos))
            & jnp.all(jnp.isfinite(state.data.qvel))
            & jnp.all(jnp.isfinite(state.data.xpos))
            & jnp.all(jnp.isfinite(state.data.xquat))
            & jnp.all(jnp.isfinite(state.data.cvel))
            & jnp.all(jnp.isfinite(state.data.actuator_force))
            & jnp.all(jnp.isfinite(state.data.ctrl))
        )
        timeout = state.reference_time >= self.motion.duration
        terminal = height | orientation | end_effector | nonfinite
        return {
            "done/height": height,
            "done/orientation": orientation,
            "done/end_effector": end_effector,
            "done/nonfinite": nonfinite,
            "done/timeout": timeout,
            "done/terminal": terminal,
            "done/any": terminal | timeout,
        }

    def _info(self, state, observation, reward_info, done_flags):
        done = done_flags["done/any"]
        return {
            "bootstrap_observation": observation,
            "bootstrap_critic_observation": self.critic_observation_batch(state)
            if observation.ndim == 2
            else self.critic_observation(state),
            "episode_return": jnp.where(done, state.episode_return, 0.0),
            "episode_length": jnp.where(done, state.episode_length, 0),
            "returned_episode": done,
            "reference/time": state.reference_time,
            **done_flags,
            **reward_info,
        }

    def _sanitize_nonfinite_reward(
        self,
        reward: jax.Array,
        reward_info: dict[str, jax.Array],
        done_flags: dict[str, jax.Array],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        nonfinite = done_flags["done/nonfinite"] | ~jnp.isfinite(reward)
        sanitized = {
            name: jnp.where(
                nonfinite,
                jnp.zeros_like(value),
                jnp.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0),
            )
            for name, value in reward_info.items()
        }
        return sum(sanitized.values(), jnp.zeros_like(reward)), sanitized

    def _aligned_reference(
        self, time: jax.Array, alignment_xy: jax.Array, alignment_yaw: jax.Array
    ) -> MotionSample:
        source = sample_motion(self.motion, time)
        rotation = yaw_quat_from_angle(alignment_yaw)
        qpos = source.qpos
        root_pos = quat_rotate(rotation, qpos[:3]).at[:2].add(alignment_xy)
        qpos = qpos.at[:3].set(root_pos).at[3:7].set(quat_multiply(rotation, qpos[3:7]))
        qvel = source.qvel.at[:3].set(quat_rotate(rotation, source.qvel[:3]))
        qvel = qvel.at[3:6].set(quat_rotate(rotation, source.qvel[3:6]))
        body_pos = (
            jax.vmap(lambda value: quat_rotate(rotation, value))(source.body_pos)
            .at[:, :2]
            .add(alignment_xy)
        )
        return MotionSample(
            qpos=qpos,
            qvel=qvel,
            body_pos=body_pos,
            body_quat=jax.vmap(lambda quat: quat_multiply(rotation, quat))(
                source.body_quat
            ),
            body_lin_vel=jax.vmap(lambda value: quat_rotate(rotation, value))(
                source.body_lin_vel
            ),
            body_ang_vel=jax.vmap(lambda value: quat_rotate(rotation, value))(
                source.body_ang_vel
            ),
            foot_contacts=source.foot_contacts,
        )

    def _reference_ctrl(self, reference: MotionSample) -> jax.Array:
        ctrl = jnp.asarray(self.model.default_ctrl)
        return ctrl.at[jnp.asarray(self.model.policy_ctrl_indices)].set(
            reference.qpos[jnp.asarray(self.model.policy_qpos_indices)]
        )

    def _action_to_ctrl(self, action: jax.Array) -> jax.Array:
        default = jnp.asarray(self.model.default_ctrl)
        policy = jnp.asarray(self.model.policy_ctrl_indices)
        target = default[policy] + jnp.asarray(self.model.policy_action_scale) * action
        limits = jnp.asarray(self.model.mj_model.actuator_ctrlrange)[policy]
        return default.at[policy].set(jnp.clip(target, limits[:, 0], limits[:, 1]))

    def _joint_limit_cost(self, state: MotionTrackingState) -> jax.Array:
        positions = state.data.qpos[jnp.asarray(self.model.policy_qpos_indices)]
        limits = jnp.asarray(self.model.mj_model.actuator_ctrlrange)[
            jnp.asarray(self.model.policy_ctrl_indices)
        ]
        span = jnp.maximum(limits[:, 1] - limits[:, 0], 1.0e-6)
        margin = 0.02 * span
        cost = jnp.mean(
            jax.nn.relu(limits[:, 0] + margin - positions) ** 2
            + jax.nn.relu(positions - limits[:, 1] + margin) ** 2
        )
        return jnp.clip(cost, 0.0, 1.0)

    def _body_world_velocity(
        self, state: MotionTrackingState, body_ids: jax.Array
    ) -> jax.Array:
        cvel = state.data.cvel[body_ids]
        root_body_ids = jnp.asarray(self.model.mj_model.body_rootid)[body_ids]
        offset = state.data.xpos[body_ids] - state.data.subtree_com[root_body_ids]
        linear = cvel[:, 3:6] - jnp.cross(offset, cvel[:, :3])
        return jnp.concatenate([cvel[:, :3], linear], axis=-1)

    def _body_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)

    @staticmethod
    def _validate_range(name: str, value: tuple[float, float]) -> None:
        if len(value) != 2 or not all(math.isfinite(item) for item in value):
            raise ValueError(f"{name} must contain two finite values")
        if value[0] > value[1]:
            raise ValueError(f"{name} lower bound exceeds upper bound")


def gaussian_reward(error: jax.Array, sigma: float) -> jax.Array:
    return jnp.exp(-jnp.mean(jnp.square(error)) / (sigma * sigma))


def quat_conjugate(quat: jax.Array) -> jax.Array:
    return quat * jnp.asarray([1.0, -1.0, -1.0, -1.0])


def quat_multiply(left: jax.Array, right: jax.Array) -> jax.Array:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return jnp.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def quat_distance(left: jax.Array, right: jax.Array) -> jax.Array:
    dot = jnp.clip(jnp.abs(jnp.dot(left, right)), 0.0, 1.0)
    return 2.0 * jnp.arccos(dot)


def quat_to_rotvec(quat: jax.Array) -> jax.Array:
    quat = jnp.where(quat[0] < 0.0, -quat, quat)
    vector_norm = jnp.linalg.norm(quat[1:])
    angle = 2.0 * jnp.arctan2(vector_norm, jnp.maximum(quat[0], 1.0e-8))
    return quat[1:] * angle / jnp.maximum(vector_norm, 1.0e-8)
