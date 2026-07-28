"""Whole-body reference-motion tracking for fixed-waist, fixed-head Bello."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
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
    CONTROL_DECIMATION,
    SIM_TIMESTEP,
    ActionLayout,
    ObservationLayout,
)
from bello_mujoco.envs.locomotion import (
    pair_contains_each,
    quat_rotate,
    quat_rotate_inverse,
    tree_where,
    yaw_quat,
    yaw_quat_from_angle,
)
from bello_mujoco.motion.clip import (
    MotionClip,
    MotionSample,
    load_motion_clip,
    motion_tracking_asset_config,
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
    joint_default_offset: jax.Array
    next_push_step: jax.Array
    sampling_failure_ema: jax.Array
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
    root_position_noise: tuple[float, float, float] = (0.0, 0.0, 0.0)
    root_rotation_noise: tuple[float, float, float] = (0.0, 0.0, 0.0)
    root_linear_velocity_noise: tuple[float, float, float] = (0.0, 0.0, 0.0)
    root_angular_velocity_noise: tuple[float, float, float] = (0.0, 0.0, 0.0)
    joint_default_offset_noise: float = 0.0


@dataclass(frozen=True)
class MotionObservationNoiseConfig:
    enabled: bool = False
    anchor_position: float = 0.0
    anchor_orientation: float = 0.0
    base_linear_velocity: float = 0.0
    base_angular_velocity: float = 0.0
    joint_position: float = 0.0
    joint_velocity: float = 0.0


@dataclass(frozen=True)
class MotionAdaptiveSamplingConfig:
    enabled: bool = False
    bin_seconds: float = 1.0
    ema_alpha: float = 0.001
    kernel_size: int = 1
    kernel_decay: float = 0.8
    uniform_ratio: float = 0.1


@dataclass(frozen=True)
class MotionPushConfig:
    enabled: bool = False
    interval_seconds: tuple[float, float] = (1.0, 3.0)
    linear_velocity: tuple[float, float, float] = (0.5, 0.5, 0.2)
    angular_velocity: tuple[float, float, float] = (0.52, 0.52, 0.78)


@dataclass(frozen=True)
class MotionRewardConfig:
    root_position_weight: float = 0.15
    root_orientation_weight: float = 0.15
    joint_position_weight: float = 0.20
    joint_velocity_weight: float = 0.05
    body_position_weight: float = 0.20
    body_orientation_weight: float = 0.10
    body_velocity_weight: float = 0.05
    body_linear_velocity_weight: float = 0.0
    body_angular_velocity_weight: float = 0.0
    action_rate_weight: float = -0.01
    joint_limit_weight: float = -0.02
    undesired_contact_weight: float = 0.0
    root_position_sigma: float = 0.20
    root_orientation_sigma: float = 0.35
    joint_position_sigma: float = 0.35
    joint_velocity_sigma: float = 3.0
    body_position_sigma: float = 0.15
    body_orientation_sigma: float = 0.45
    body_velocity_sigma: float = 2.0
    body_linear_velocity_sigma: float = 1.0
    body_angular_velocity_sigma: float = math.pi
    undesired_contact_force_threshold: float = 1.0


@dataclass(frozen=True)
class MotionDoneConfig:
    root_height_error: float = 0.35
    root_orientation_error: float = 1.2
    end_effector_position_error: float = 0.65


@dataclass(frozen=True)
class MotionTrackingEnvConfig:
    motion_path: Path = Path(
        "motions/asap_fixed_head/video_single_foot_balance_level1_filter_amass.npz"
    )
    decimation: int = CONTROL_DECIMATION
    action_clip: float = 1.0
    formulation: Literal[
        "reference_residual",
        "beyondmimic",
        "beyondmimic_joint_limits",
    ] = "reference_residual"
    anchor_body_name: str = "pelvis_link"
    terrain: TerrainConfig = FLAT_TERRAIN
    reset: MotionResetConfig = MotionResetConfig()
    observation_noise: MotionObservationNoiseConfig = MotionObservationNoiseConfig()
    adaptive_sampling: MotionAdaptiveSamplingConfig = MotionAdaptiveSamplingConfig()
    push: MotionPushConfig = MotionPushConfig()
    reward: MotionRewardConfig = MotionRewardConfig()
    done: MotionDoneConfig = MotionDoneConfig()


@dataclass(frozen=True)
class BelloMotionTrackingEnv:
    model: BelloModel
    motion: MotionClip
    config: MotionTrackingEnvConfig = MotionTrackingEnvConfig()
    actuator_config: ActuatorConfig = motion_tracking_asset_config().actuator_config

    def __post_init__(self) -> None:
        if self.model.actuator_joint_names != self.motion.joint_names:
            raise ValueError("model and motion actuator orders do not match")
        actuator_count = self.model.mj_model.nu
        expected_nq = 7 + actuator_count
        expected_nv = 6 + actuator_count
        if (
            self.model.mj_model.nq != expected_nq
            or self.model.mj_model.nv != expected_nv
        ):
            raise ValueError(
                "motion-tracking Bello must contain only its floating base and "
                "actuated one-DoF joints; expected "
                f"nq={expected_nq}, nv={expected_nv}, got "
                f"nq={self.model.mj_model.nq}, nv={self.model.mj_model.nv}"
            )
        expected_policy_indices = np.arange(actuator_count, dtype=np.int32)
        if not np.array_equal(self.model.policy_ctrl_indices, expected_policy_indices):
            raise ValueError(
                "motion tracking requires every logical actuator to be policy "
                f"controlled; got {self.action_size} policy joints for "
                f"nu={actuator_count}"
            )
        configured_actuators = self.actuator_config.by_name
        for name in self.model.actuator_joint_names:
            configured = configured_actuators.get(name)
            if configured is None or configured.mode != "policy":
                raise ValueError(
                    f"motion tracking requires a policy actuator config for {name!r}"
                )
        if self.decimation <= 0:
            raise ValueError("decimation must be positive")
        if not math.isfinite(self.config.action_clip) or self.config.action_clip <= 0:
            raise ValueError("action_clip must be positive and finite")
        if self.config.formulation not in {
            "reference_residual",
            "beyondmimic",
            "beyondmimic_joint_limits",
        }:
            raise ValueError(
                "formulation must be 'reference_residual', 'beyondmimic', or "
                "'beyondmimic_joint_limits'"
            )
        self._validate_range("initial_xy_range", self.config.reset.initial_xy_range)
        self._validate_range("initial_yaw_range", self.config.reset.initial_yaw_range)
        self._validate_range("joint_pos_noise", self.config.reset.joint_pos_noise)
        self._validate_range("joint_vel_noise", self.config.reset.joint_vel_noise)
        for name in (
            "root_position_noise",
            "root_rotation_noise",
            "root_linear_velocity_noise",
            "root_angular_velocity_noise",
        ):
            self._validate_nonnegative_vector(name, getattr(self.config.reset, name))
        if self.config.reset.joint_default_offset_noise < 0.0:
            raise ValueError("joint_default_offset_noise must be nonnegative")
        self._validate_observation_noise()
        self._validate_adaptive_sampling()
        self._validate_push()
        if self.motion.body_names[0] != "pelvis_link":
            raise ValueError("the first tracked body must be pelvis_link")
        for name in self.motion.body_names:
            if self._body_id(name) < 0:
                raise ValueError(f"tracked body not found in model: {name!r}")
        if self.config.anchor_body_name not in self.motion.body_names:
            raise ValueError(
                f"anchor body {self.config.anchor_body_name!r} is not tracked"
            )
        for name in self.end_effector_names:
            if name not in self.motion.body_names:
                raise ValueError(f"end effector {name!r} is not tracked")

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
        return cls(
            model,
            load_motion_clip(env_config.motion_path, model),
            env_config,
            asset_config.actuator_config,
        )

    @classmethod
    def load_native(
        cls,
        asset_config: AssetPrepConfig,
        env_config: MotionTrackingEnvConfig,
        *,
        use_viewer_model: bool = False,
    ) -> "BelloMotionTrackingEnv":
        model = load_bello_model(
            asset_config,
            terrain=env_config.terrain,
            use_viewer_model=use_viewer_model,
            load_mjx=False,
        )
        return cls(
            model,
            load_motion_clip(env_config.motion_path, model),
            env_config,
            asset_config.actuator_config,
        )

    @classmethod
    def load_prepared(
        cls,
        prepared: PreparedAsset,
        env_config: MotionTrackingEnvConfig,
        *,
        actuator_config: ActuatorConfig = motion_tracking_asset_config().actuator_config,
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
        return cls(
            model,
            load_motion_clip(env_config.motion_path, model),
            env_config,
            actuator_config,
        )

    @classmethod
    def load_prepared_native(
        cls,
        prepared: PreparedAsset,
        env_config: MotionTrackingEnvConfig,
        *,
        actuator_config: ActuatorConfig = motion_tracking_asset_config().actuator_config,
        use_viewer_model: bool = False,
    ) -> "BelloMotionTrackingEnv":
        model = load_prepared_bello_model(
            prepared,
            terrain=env_config.terrain,
            actuator_config=actuator_config,
            use_viewer_model=use_viewer_model,
            load_mjx=False,
        )
        return cls(
            model,
            load_motion_clip(env_config.motion_path, model),
            env_config,
            actuator_config,
        )

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
    def uses_beyondmimic_formulation(self) -> bool:
        return self.config.formulation in {
            "beyondmimic",
            "beyondmimic_joint_limits",
        }

    @property
    def task(self) -> Literal["motion_tracking"]:
        return "motion_tracking"

    @property
    def ppo_stagger_initial_step_count(self) -> bool:
        return False

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
        action_size = self.action_size
        if self.uses_beyondmimic_formulation:
            return ObservationLayout(
                (
                    ("reference_joint_position", action_size),
                    ("reference_joint_velocity", action_size),
                    ("anchor_position_error", 3),
                    ("anchor_orientation_error_6d", 6),
                    ("base_linear_velocity", 3),
                    ("base_angular_velocity", 3),
                    ("joint_position", action_size),
                    ("joint_velocity", action_size),
                    ("last_action", action_size),
                )
            )
        return ObservationLayout(
            (
                ("reference_joint_position", action_size),
                ("reference_joint_velocity", action_size),
                ("joint_position", action_size),
                ("joint_velocity", action_size),
                ("last_action", action_size),
                ("root_position_error", 3),
                ("root_orientation_error", 3),
                ("root_linear_velocity_error", 3),
                ("root_angular_velocity_error", 3),
                ("reference_phase", 2),
            )
        )

    @property
    def critic_observation_layout(self) -> ObservationLayout:
        body_count = len(self.motion.body_names)
        if self.uses_beyondmimic_formulation:
            extra = (
                ("tracked_body_position_relative_to_anchor", 3 * body_count),
                ("tracked_body_orientation_relative_to_anchor_6d", 6 * body_count),
            )
        else:
            extra = (
                ("tracked_body_position_error", 3 * body_count),
                ("tracked_body_orientation_error", 3 * body_count),
            )
        return ObservationLayout(self.observation_layout.components + extra)

    @property
    def observation_size(self) -> int:
        return self.observation_layout.size

    @property
    def critic_observation_size(self) -> int:
        return self.critic_observation_layout.size

    @property
    def tracked_body_ids(self) -> np.ndarray:
        return np.asarray([self._body_id(name) for name in self.motion.body_names])

    @property
    def end_effector_indices(self) -> np.ndarray:
        names = self.motion.body_names
        return np.asarray([names.index(name) for name in self.end_effector_names])

    @property
    def end_effector_names(self) -> tuple[str, ...]:
        if self.uses_beyondmimic_formulation:
            return (
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "l_wrist_link",
                "r_wrist_link",
            )
        return ("l_wrist_link", "r_wrist_link")

    @property
    def anchor_body_index(self) -> int:
        return self.motion.body_names.index(self.config.anchor_body_name)

    @property
    def anchor_body_id(self) -> int:
        return int(self.tracked_body_ids[self.anchor_body_index])

    @property
    def sampling_bin_count(self) -> int:
        return max(
            1,
            math.ceil(
                max(self.motion.duration, self.control_timestep)
                / self.config.adaptive_sampling.bin_seconds
            ),
        )

    def reset(
        self,
        rng: jax.Array | None = None,
        reference_time: jax.Array | None = None,
        sampling_failure_ema: jax.Array | None = None,
    ) -> tuple[MotionTrackingState, jax.Array]:
        self._require_mjx()
        time_key = pose_key = joint_key = root_key = offset_key = push_key = None
        if rng is not None:
            time_key, pose_key, joint_key, root_key, offset_key, push_key = (
                jax.random.split(rng, 6)
            )
        if sampling_failure_ema is None:
            sampling_failure_ema = jnp.zeros(self.sampling_bin_count, dtype=jnp.float32)
        if reference_time is None:
            if self.config.reset.random_start and time_key is not None:
                reference_time = self._sample_reference_time(
                    time_key, sampling_failure_ema
                )
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
        reference = self.aligned_reference(reference_time, alignment_xy, alignment_yaw)
        qpos, qvel = reference.qpos, reference.qvel
        if joint_key is not None:
            qpos_key, qvel_key = jax.random.split(joint_key)
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
        if root_key is not None:
            qpos, qvel = self._perturb_root(qpos, qvel, root_key)
        joint_default_offset = jnp.zeros(self.action_size, dtype=jnp.float32)
        if offset_key is not None:
            magnitude = self.config.reset.joint_default_offset_noise
            joint_default_offset = jax.random.uniform(
                offset_key,
                (self.action_size,),
                minval=-magnitude,
                maxval=magnitude,
                dtype=jnp.float32,
            )
        next_push_step = self._sample_push_interval(push_key)
        data = self.model.make_data().replace(
            qpos=qpos,
            qvel=qvel,
            ctrl=self._initial_ctrl(reference, joint_default_offset),
        )
        data = mjx.forward(self.model.mjx_model, data)
        state = MotionTrackingState(
            data=data,
            last_action=jnp.zeros(self.action_size, dtype=jnp.float32),
            reference_time=reference_time,
            alignment_xy=alignment_xy,
            alignment_yaw=alignment_yaw,
            joint_default_offset=joint_default_offset,
            next_push_step=next_push_step,
            sampling_failure_ema=sampling_failure_ema,
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
        rng: jax.Array | None = None,
        sampling_failure_ema: jax.Array | None = None,
    ) -> tuple[MotionTrackingState, jax.Array]:
        reset_rng = obs_rng = rng
        if rng is not None:
            reset_rng, obs_rng = jax.random.split(rng)
        if sampling_failure_ema is None:
            sampling_failure_ema = jnp.zeros(self.sampling_bin_count, dtype=jnp.float32)
        if rng is None:
            state, _ = jax.vmap(
                lambda _: self.reset(sampling_failure_ema=sampling_failure_ema)
            )(jnp.arange(batch_size))
        else:
            keys = jax.random.split(reset_rng, batch_size)
            state, _ = jax.vmap(
                lambda key: self.reset(key, sampling_failure_ema=sampling_failure_ema)
            )(keys)
        data = state.data.replace(_impl=state.data._impl.replace(nworld=batch_size))
        state = state.replace(data=data)
        return state, self.observation_batch(state, obs_rng)

    def step(
        self, state: MotionTrackingState, action: jax.Array
    ) -> tuple[
        MotionTrackingState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]
    ]:
        self._require_mjx()
        action = jnp.clip(jnp.asarray(action), -self.action_clip, self.action_clip)
        data = state.data.replace(ctrl=self.action_target(state, action))
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
        reward, reward_info = self._reward(
            state, stepped, jnp.asarray(0, dtype=jnp.int32)
        )
        done_flags = self.termination_flags(stepped)
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
        self._require_mjx()
        batch_size = int(action.shape[0])
        reset_rng = push_rng = obs_rng = rng
        if rng is not None:
            reset_rng, push_rng, obs_rng = jax.random.split(rng, 3)
        action = jnp.clip(jnp.asarray(action), -self.action_clip, self.action_clip)
        ctrl = jax.vmap(self.action_target)(state, action)
        data = state.data.replace(
            ctrl=ctrl, _impl=state.data._impl.replace(nworld=batch_size)
        )
        data, next_push_step, push_applied = self._maybe_apply_push_batch(
            data, state, push_rng
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
            next_push_step=next_push_step,
        )
        world_ids = jnp.arange(batch_size, dtype=jnp.int32)
        reward, reward_info = jax.vmap(self._reward)(state, stepped, world_ids)
        done_flags = jax.vmap(self.termination_flags)(stepped)
        reward, reward_info = self._sanitize_nonfinite_reward(
            reward, reward_info, done_flags
        )
        done = done_flags["done/any"]
        sampling_failure_ema = self._update_sampling_failure_ema(
            stepped, done_flags["done/terminal"]
        )
        repeated_sampling_ema = jnp.repeat(
            sampling_failure_ema[None, :], batch_size, axis=0
        )
        episode_return = state.episode_return + reward
        stepped = stepped.replace(
            episode_return=episode_return,
            sampling_failure_ema=repeated_sampling_ema,
            last_episode_return=jnp.where(
                done, episode_return, state.last_episode_return
            ),
            last_episode_length=jnp.where(
                done, stepped.episode_length, state.last_episode_length
            ),
        )
        stepped_obs = self.observation_batch(stepped, obs_rng)
        info = self._info(stepped, stepped_obs, reward_info, done_flags)
        info["push/applied"] = push_applied.astype(jnp.float32)
        reset_state, reset_obs = self.reset_batch(
            batch_size, reset_rng, sampling_failure_ema=sampling_failure_ema
        )
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

    def observation(
        self, state: MotionTrackingState, rng: jax.Array | None = None
    ) -> jax.Array:
        reference = self.aligned_reference(
            jnp.minimum(
                state.reference_time + self.control_timestep, self.motion.duration
            ),
            state.alignment_xy,
            state.alignment_yaw,
        )
        qpos_indices = jnp.asarray(self.model.policy_qpos_indices)
        qvel_indices = jnp.asarray(self.model.policy_qvel_indices)
        policy_indices = jnp.asarray(self.model.policy_ctrl_indices)
        joint_default = jnp.asarray(self.model.default_joint_target)[policy_indices]
        if self.uses_beyondmimic_formulation:
            anchor_id = self.anchor_body_id
            anchor_index = self.anchor_body_index
            anchor_quat = state.data.xquat[anchor_id]
            anchor_position_error = quat_rotate_inverse(
                anchor_quat,
                reference.body_pos[anchor_index] - state.data.xpos[anchor_id],
            )
            anchor_orientation_error = rotation_6d(
                quat_multiply(
                    quat_conjugate(anchor_quat),
                    reference.body_quat[anchor_index],
                )
            )
            root_quat = state.data.qpos[3:7]
            base_linear_velocity = quat_rotate_inverse(root_quat, state.data.qvel[:3])
            base_angular_velocity = state.data.qvel[3:6]
            current_joint_position = (
                state.data.qpos[qpos_indices]
                - joint_default
                - state.joint_default_offset
            )
            terms = (
                reference.qpos[qpos_indices] - joint_default,
                reference.qvel[qvel_indices],
                anchor_position_error,
                anchor_orientation_error,
                base_linear_velocity,
                base_angular_velocity,
                current_joint_position,
                state.data.qvel[qvel_indices],
                state.last_action,
            )
            return self._corrupt_observation_terms(terms, rng)

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
        reference_root_angular_velocity_world = quat_rotate(
            reference.qpos[3:7], reference.qvel[3:6]
        )
        root_angular_velocity_world = quat_rotate(
            state.data.qpos[3:7], state.data.qvel[3:6]
        )
        root_angular_velocity_error = quat_rotate_inverse(
            heading,
            reference_root_angular_velocity_world - root_angular_velocity_world,
        )
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

    def observation_batch(
        self, state: MotionTrackingState, rng: jax.Array | None = None
    ) -> jax.Array:
        if rng is None:
            return jax.vmap(self.observation)(state)
        keys = jax.random.split(rng, int(state.last_action.shape[0]))
        return jax.vmap(self.observation)(state, keys)

    def critic_observation(self, state: MotionTrackingState) -> jax.Array:
        if self.uses_beyondmimic_formulation:
            ids = jnp.asarray(self.tracked_body_ids)
            anchor_id = self.anchor_body_id
            anchor_position = state.data.xpos[anchor_id]
            anchor_quat = state.data.xquat[anchor_id]
            body_position = jax.vmap(
                lambda value: quat_rotate_inverse(anchor_quat, value - anchor_position)
            )(state.data.xpos[ids])
            body_orientation = jax.vmap(
                lambda value: rotation_6d(
                    quat_multiply(quat_conjugate(anchor_quat), value)
                )
            )(state.data.xquat[ids])
            return jnp.concatenate(
                [
                    self.observation(state),
                    body_position.reshape(-1),
                    body_orientation.reshape(-1),
                ]
            )

        reference = self.aligned_reference(
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

    def critic_observation_batch(
        self, state: MotionTrackingState, rng: jax.Array | None = None
    ) -> jax.Array:
        del rng
        return jax.vmap(self.critic_observation)(state)

    def initial_step_counts(
        self, batch_size: int, rng: jax.Array | None = None
    ) -> jax.Array:
        del rng
        return jnp.zeros((batch_size,), dtype=jnp.int32)

    def ppo_action_diagnostics(self, env_action: jax.Array) -> dict[str, jax.Array]:
        del env_action
        return {}

    def _reward(
        self,
        previous: MotionTrackingState,
        state: MotionTrackingState,
        world_id: jax.Array | int = 0,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        reference = self.aligned_reference(
            jnp.minimum(state.reference_time, self.motion.duration),
            state.alignment_xy,
            state.alignment_yaw,
        )
        if self.uses_beyondmimic_formulation:
            return self._beyondmimic_reward(previous, state, reference, world_id)
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

    def _beyondmimic_reward(
        self,
        previous: MotionTrackingState,
        state: MotionTrackingState,
        reference: MotionSample,
        world_id: jax.Array | int,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        ids = jnp.asarray(self.tracked_body_ids)
        anchor_id = self.anchor_body_id
        anchor_index = self.anchor_body_index
        body_target_pos, body_target_quat = self._relative_body_targets(
            state, reference
        )
        body_velocity = self._body_world_velocity(state, ids)
        position_error = state.data.xpos[ids] - body_target_pos
        orientation_error = jax.vmap(quat_distance)(
            state.data.xquat[ids], body_target_quat
        )
        linear_velocity_error = body_velocity[:, 3:6] - reference.body_lin_vel
        angular_velocity_error = body_velocity[:, :3] - reference.body_ang_vel
        cfg = self.config.reward
        components = {
            "reward/global_anchor_position": cfg.root_position_weight
            * squared_norm_gaussian_reward(
                state.data.xpos[anchor_id] - reference.body_pos[anchor_index],
                cfg.root_position_sigma,
            ),
            "reward/global_anchor_orientation": cfg.root_orientation_weight
            * scalar_gaussian_reward(
                quat_distance(
                    state.data.xquat[anchor_id], reference.body_quat[anchor_index]
                ),
                cfg.root_orientation_sigma,
            ),
            "reward/relative_body_position": cfg.body_position_weight
            * squared_norm_gaussian_reward(position_error, cfg.body_position_sigma),
            "reward/relative_body_orientation": cfg.body_orientation_weight
            * scalar_gaussian_reward(orientation_error, cfg.body_orientation_sigma),
            "reward/body_linear_velocity": cfg.body_linear_velocity_weight
            * squared_norm_gaussian_reward(
                linear_velocity_error, cfg.body_linear_velocity_sigma
            ),
            "reward/body_angular_velocity": cfg.body_angular_velocity_weight
            * squared_norm_gaussian_reward(
                angular_velocity_error, cfg.body_angular_velocity_sigma
            ),
            "reward/action_rate": cfg.action_rate_weight
            * jnp.sum((state.last_action - previous.last_action) ** 2),
            "reward/joint_limit": cfg.joint_limit_weight
            * self._joint_limit_cost(state),
            "reward/undesired_contact": cfg.undesired_contact_weight
            * self._undesired_contact_count(state, world_id),
        }
        return jnp.sum(jnp.stack(tuple(components.values()))), components

    def termination_flags(self, state: MotionTrackingState) -> dict[str, jax.Array]:
        """Return named termination conditions for one policy frame."""

        reference = self.aligned_reference(
            jnp.minimum(state.reference_time, self.motion.duration),
            state.alignment_xy,
            state.alignment_yaw,
        )
        ids = jnp.asarray(self.tracked_body_ids)
        if self.uses_beyondmimic_formulation:
            anchor_id = self.anchor_body_id
            anchor_index = self.anchor_body_index
            target_pos, _ = self._relative_body_targets(state, reference)
            height = (
                jnp.abs(
                    state.data.xpos[anchor_id, 2] - reference.body_pos[anchor_index, 2]
                )
                > self.config.done.root_height_error
            )
            reference_gravity = quat_rotate_inverse(
                reference.body_quat[anchor_index],
                jnp.asarray([0.0, 0.0, -1.0], dtype=jnp.float32),
            )
            robot_gravity = quat_rotate_inverse(
                state.data.xquat[anchor_id],
                jnp.asarray([0.0, 0.0, -1.0], dtype=jnp.float32),
            )
            orientation = (
                jnp.abs(reference_gravity[2] - robot_gravity[2])
                > self.config.done.root_orientation_error
            )
            ee_indices = jnp.asarray(self.end_effector_indices)
            ee_error = jnp.abs(
                state.data.xpos[ids[ee_indices], 2] - target_pos[ee_indices, 2]
            )
            end_effector = jnp.any(
                ee_error > self.config.done.end_effector_position_error
            )
            return self._finish_termination_flags(
                state, height, orientation, end_effector
            )
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
        return self._finish_termination_flags(state, height, orientation, end_effector)

    def _finish_termination_flags(
        self,
        state: MotionTrackingState,
        height: jax.Array,
        orientation: jax.Array,
        end_effector: jax.Array,
    ) -> dict[str, jax.Array]:
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
        failure_ema = (
            state.sampling_failure_ema[0]
            if observation.ndim == 2
            else state.sampling_failure_ema
        )
        sampling_probabilities = self._sampling_probabilities(failure_ema)
        sampling_entropy = -jnp.sum(
            sampling_probabilities * jnp.log(sampling_probabilities + 1.0e-12)
        ) / jnp.maximum(jnp.log(float(self.sampling_bin_count)), 1.0)
        sampling_top_probability = jnp.max(sampling_probabilities)
        sampling_top_bin = (
            jnp.argmax(sampling_probabilities).astype(jnp.float32)
            / self.sampling_bin_count
        )
        sampling_failure_mean = jnp.mean(failure_ema)
        sampling_failure_max = jnp.max(failure_ema)
        if observation.ndim == 2:
            sampling_entropy = jnp.full_like(done, sampling_entropy, dtype=jnp.float32)
            sampling_top_probability = jnp.full_like(
                done, sampling_top_probability, dtype=jnp.float32
            )
            sampling_top_bin = jnp.full_like(done, sampling_top_bin, dtype=jnp.float32)
            sampling_failure_mean = jnp.full_like(
                done, sampling_failure_mean, dtype=jnp.float32
            )
            sampling_failure_max = jnp.full_like(
                done, sampling_failure_max, dtype=jnp.float32
            )
        return {
            "bootstrap_observation": observation,
            "bootstrap_critic_observation": self.critic_observation_batch(state)
            if observation.ndim == 2
            else self.critic_observation(state),
            "episode_return": jnp.where(done, state.episode_return, 0.0),
            "episode_length": jnp.where(done, state.episode_length, 0),
            "returned_episode": done,
            "reference/time": state.reference_time,
            "sampling/entropy": sampling_entropy,
            "sampling/top_probability": sampling_top_probability,
            "sampling/top_bin": sampling_top_bin,
            "sampling/failure_ema_mean": sampling_failure_mean,
            "sampling/failure_ema_max": sampling_failure_max,
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

    def aligned_reference(
        self, time: jax.Array, alignment_xy: jax.Array, alignment_yaw: jax.Array
    ) -> MotionSample:
        """Sample the motion after applying the episode's planar alignment."""

        source = sample_motion(self.motion, time)
        rotation = yaw_quat_from_angle(alignment_yaw)
        qpos = source.qpos
        root_pos = quat_rotate(rotation, qpos[:3]).at[:2].add(alignment_xy)
        qpos = qpos.at[:3].set(root_pos).at[3:7].set(quat_multiply(rotation, qpos[3:7]))
        qvel = source.qvel.at[:3].set(quat_rotate(rotation, source.qvel[:3]))
        # Free-joint angular qvel is root-local, so world-yaw alignment leaves it.
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
        logical_target = reference.qpos[jnp.asarray(self.model.actuator_qpos_indices)]
        return self._logical_target_to_ctrl(logical_target)

    def _logical_target_to_ctrl(self, logical_target: jax.Array) -> jax.Array:
        requested_ctrl = jnp.matmul(
            jnp.asarray(self.model.actuator_coordinate_transform),
            jnp.asarray(logical_target),
            precision=jax.lax.Precision.HIGHEST,
        )
        limits = jnp.asarray(self.model.mj_model.actuator_ctrlrange)
        return jnp.clip(requested_ctrl, limits[:, 0], limits[:, 1])

    @property
    def reference_residual_action_scale(self) -> jax.Array:
        policy = jnp.asarray(self.model.policy_ctrl_indices)
        logical_span = jnp.asarray(self.model.joint_target_high) - jnp.asarray(
            self.model.joint_target_low
        )
        return 0.25 * logical_span[policy]

    @property
    def beyondmimic_action_scale(self) -> jax.Array:
        configured = self.actuator_config.by_name
        scales = []
        for index in self.model.policy_ctrl_indices:
            joint = configured[self.model.actuator_joint_names[index]]
            if joint.effort_limit is None:
                raise ValueError(
                    "BeyondMimic action scaling requires an effort limit for "
                    f"{self.model.actuator_joint_names[index]!r}"
                )
            stiffness = joint.gains[0]
            if not math.isfinite(stiffness) or stiffness <= 0.0:
                raise ValueError(
                    "BeyondMimic action scaling requires positive stiffness for "
                    f"{self.model.actuator_joint_names[index]!r}"
                )
            scales.append(0.25 * joint.effort_limit / stiffness)
        return jnp.asarray(scales, dtype=jnp.float32)

    def beyondmimic_action_to_joint_target(
        self,
        action: jax.Array,
        joint_default_offset: jax.Array | None = None,
    ) -> jax.Array:
        """Map a BeyondMimic action to logical joint targets."""
        policy = jnp.asarray(self.model.policy_ctrl_indices)
        logical_target = jnp.asarray(self.model.default_joint_target)
        if joint_default_offset is None:
            joint_default_offset = jnp.zeros(self.action_size, dtype=jnp.float32)
        policy_default = logical_target[policy] + joint_default_offset
        if not self.uses_beyondmimic_formulation:
            raise ValueError("BeyondMimic actions require a BeyondMimic formulation")
        if self.config.formulation == "beyondmimic":
            policy_target = policy_default + self.beyondmimic_action_scale * action
            return logical_target.at[policy].set(policy_target)

        target_low = jnp.asarray(self.model.joint_target_low)[policy]
        target_high = jnp.asarray(self.model.joint_target_high)[policy]
        policy_default = jnp.clip(policy_default, target_low, target_high)
        normalized_action = jnp.clip(
            jnp.asarray(action, dtype=jnp.float32) / self.action_clip,
            -1.0,
            1.0,
        )
        target_span = jnp.where(
            normalized_action >= 0.0,
            target_high - policy_default,
            policy_default - target_low,
        )
        policy_target = policy_default + target_span * normalized_action
        return logical_target.at[policy].set(policy_target)

    def _initial_ctrl(
        self, reference: MotionSample, joint_default_offset: jax.Array
    ) -> jax.Array:
        if self.uses_beyondmimic_formulation:
            logical_target = jnp.asarray(self.model.default_joint_target)
            policy = jnp.asarray(self.model.policy_ctrl_indices)
            logical_target = logical_target.at[policy].add(joint_default_offset)
            return self._logical_target_to_ctrl(logical_target)
        return self._reference_ctrl(reference)

    def action_target(self, state: MotionTrackingState, action: jax.Array) -> jax.Array:
        """Convert one normalized policy action into native actuator targets."""

        policy = jnp.asarray(self.model.policy_ctrl_indices)
        if self.uses_beyondmimic_formulation:
            return self._logical_target_to_ctrl(
                self.beyondmimic_action_to_joint_target(
                    action, state.joint_default_offset
                )
            )
        reference = sample_motion(
            self.motion,
            jnp.minimum(
                state.reference_time + self.control_timestep, self.motion.duration
            ),
        )
        logical_target = reference.qpos[jnp.asarray(self.model.actuator_qpos_indices)]
        policy_target = (
            logical_target[policy] + self.reference_residual_action_scale * action
        )
        logical_target = logical_target.at[policy].set(policy_target)
        logical_target = jnp.clip(
            logical_target,
            jnp.asarray(self.model.joint_target_low),
            jnp.asarray(self.model.joint_target_high),
        )
        return self._logical_target_to_ctrl(logical_target)

    def _relative_body_targets(
        self, state: MotionTrackingState, reference: MotionSample
    ) -> tuple[jax.Array, jax.Array]:
        anchor_index = self.anchor_body_index
        anchor_id = self.anchor_body_id
        reference_anchor_pos = reference.body_pos[anchor_index]
        reference_anchor_quat = reference.body_quat[anchor_index]
        robot_anchor_pos = state.data.xpos[anchor_id]
        robot_anchor_quat = state.data.xquat[anchor_id]
        target_origin = robot_anchor_pos.at[2].set(reference_anchor_pos[2])
        yaw_alignment = yaw_quat(
            quat_multiply(robot_anchor_quat, quat_conjugate(reference_anchor_quat))
        )
        target_position = target_origin + jax.vmap(
            lambda value: quat_rotate(yaw_alignment, value - reference_anchor_pos)
        )(reference.body_pos)
        target_orientation = jax.vmap(
            lambda value: quat_multiply(yaw_alignment, value)
        )(reference.body_quat)
        return target_position, target_orientation

    def _corrupt_observation_terms(
        self, terms: tuple[jax.Array, ...], rng: jax.Array | None
    ) -> jax.Array:
        config = self.config.observation_noise
        if rng is None or not config.enabled:
            return jnp.concatenate(terms)
        keys = jax.random.split(rng, 6)
        mutable = list(terms)
        for index, magnitude, key in zip(
            (2, 3, 4, 5, 6, 7),
            (
                config.anchor_position,
                config.anchor_orientation,
                config.base_linear_velocity,
                config.base_angular_velocity,
                config.joint_position,
                config.joint_velocity,
            ),
            keys,
            strict=True,
        ):
            mutable[index] = mutable[index] + jax.random.uniform(
                key,
                mutable[index].shape,
                minval=-magnitude,
                maxval=magnitude,
                dtype=jnp.float32,
            )
        return jnp.concatenate(tuple(mutable))

    def _sample_reference_time(
        self, rng: jax.Array, sampling_failure_ema: jax.Array
    ) -> jax.Array:
        max_time = max(self.motion.duration - self.control_timestep, 0.0)
        if not self.config.adaptive_sampling.enabled:
            return jax.random.uniform(rng, (), maxval=max_time)
        bin_key, offset_key = jax.random.split(rng)
        probabilities = self._sampling_probabilities(sampling_failure_ema)
        sampled_bin = jax.random.choice(
            bin_key, self.sampling_bin_count, (), p=probabilities
        )
        offset = jax.random.uniform(offset_key, ())
        phase = (sampled_bin.astype(jnp.float32) + offset) / self.sampling_bin_count
        return phase * jnp.asarray(max_time, dtype=jnp.float32)

    def _sampling_probabilities(self, failure_ema: jax.Array) -> jax.Array:
        config = self.config.adaptive_sampling
        uniform = jnp.full(
            (self.sampling_bin_count,),
            1.0 / self.sampling_bin_count,
            dtype=jnp.float32,
        )
        scores = failure_ema + config.uniform_ratio / self.sampling_bin_count
        padded = jnp.pad(
            scores,
            (0, config.kernel_size - 1),
            mode="edge",
        )
        kernel = jnp.asarray(
            [config.kernel_decay**index for index in range(config.kernel_size)],
            dtype=jnp.float32,
        )
        kernel = kernel / jnp.sum(kernel)
        convolved = sum(
            padded[offset : offset + self.sampling_bin_count] * kernel[offset]
            for offset in range(config.kernel_size)
        )
        total = jnp.sum(convolved)
        return jnp.where(
            total > 0.0,
            convolved / jnp.maximum(total, 1.0e-8),
            uniform,
        )

    def _reference_time_to_sampling_bin(self, reference_time: jax.Array) -> jax.Array:
        max_time = max(self.motion.duration - self.control_timestep, 0.0)
        phase = jnp.where(
            max_time > 0.0,
            reference_time / max_time,
            jnp.zeros_like(reference_time),
        )
        return jnp.clip(
            jnp.floor(phase * self.sampling_bin_count).astype(jnp.int32),
            0,
            self.sampling_bin_count - 1,
        )

    def _update_sampling_failure_ema(
        self, state: MotionTrackingState, terminal: jax.Array
    ) -> jax.Array:
        current = state.sampling_failure_ema[0]
        if not self.config.adaptive_sampling.enabled:
            return current
        bins = self._reference_time_to_sampling_bin(state.reference_time)
        failures = jnp.zeros_like(current).at[bins].add(terminal.astype(jnp.float32))
        alpha = self.config.adaptive_sampling.ema_alpha
        return alpha * failures + (1.0 - alpha) * current

    def _perturb_root(
        self, qpos: jax.Array, qvel: jax.Array, rng: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        position_key, rotation_key, linear_key, angular_key = jax.random.split(rng, 4)
        position_noise = symmetric_vector_noise(
            position_key, self.config.reset.root_position_noise
        )
        rotation_noise = symmetric_vector_noise(
            rotation_key, self.config.reset.root_rotation_noise
        )
        linear_noise = symmetric_vector_noise(
            linear_key, self.config.reset.root_linear_velocity_noise
        )
        angular_noise = symmetric_vector_noise(
            angular_key, self.config.reset.root_angular_velocity_noise
        )
        rotation_delta = quat_from_euler_xyz(rotation_noise)
        qpos = (
            qpos.at[:3]
            .add(position_noise)
            .at[3:7]
            .set(quat_multiply(rotation_delta, qpos[3:7]))
        )
        qvel = qvel.at[:3].add(linear_noise).at[3:6].add(angular_noise)
        return qpos, qvel

    def _sample_push_interval(self, rng: jax.Array | None) -> jax.Array:
        if rng is None or not self.config.push.enabled:
            return jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32)
        seconds = jax.random.uniform(
            rng,
            (),
            minval=self.config.push.interval_seconds[0],
            maxval=self.config.push.interval_seconds[1],
        )
        return jnp.maximum(
            1, jnp.rint(seconds / self.control_timestep).astype(jnp.int32)
        )

    def _maybe_apply_push_batch(
        self,
        data: mjx.Data,
        state: MotionTrackingState,
        rng: jax.Array | None,
    ) -> tuple[mjx.Data, jax.Array, jax.Array]:
        batch_size = int(state.step_count.shape[0])
        applied = jnp.zeros((batch_size,), dtype=bool)
        if rng is None or not self.config.push.enabled:
            return data, state.next_push_step, applied
        velocity_key, interval_key = jax.random.split(rng)
        applied = state.step_count >= state.next_push_step
        velocity_keys = jax.random.split(velocity_key, batch_size)
        maxima = self.config.push.linear_velocity + self.config.push.angular_velocity
        velocity = jax.vmap(lambda key: symmetric_vector_noise(key, maxima))(
            velocity_keys
        )
        root_qvel = int(self.model.root_qvel_adr)
        root_velocity = jnp.where(
            applied[:, None],
            velocity,
            data.qvel[:, root_qvel : root_qvel + 6],
        )
        data = data.replace(
            qvel=data.qvel.at[:, root_qvel : root_qvel + 6].set(root_velocity)
        )
        interval_keys = jax.random.split(interval_key, batch_size)
        intervals = jax.vmap(self._sample_push_interval)(interval_keys)
        next_push_step = jnp.where(
            applied, state.step_count + intervals, state.next_push_step
        )
        return data, next_push_step, applied

    def _undesired_contact_count(
        self, state: MotionTrackingState, world_id: jax.Array | int
    ) -> jax.Array:
        if self.model.robot_geom_ids.size == 0:
            return jnp.asarray(0.0, dtype=jnp.float32)
        impl = state.data._impl
        contact_geom = jnp.asarray(impl.contact__geom, dtype=jnp.int32)
        contact_geom = jnp.clip(contact_geom, 0, self.model.mj_model.ngeom - 1)
        nacon = jnp.ravel(jnp.asarray(impl.nacon, dtype=jnp.int32))[0]
        active = (jnp.arange(contact_geom.shape[0], dtype=jnp.int32) < nacon) & (
            jnp.asarray(impl.contact__worldid, dtype=jnp.int32)
            == jnp.asarray(world_id, dtype=jnp.int32)
        )
        addresses = jnp.asarray(impl.contact__efc_address, dtype=jnp.int32)[:, 0]
        forces = jnp.asarray(impl.efc__force, dtype=jnp.float32)
        clipped = jnp.clip(addresses, 0, forces.shape[0] - 1)
        normal_force = jnp.where(
            addresses >= 0, jnp.abs(forces[clipped]), jnp.zeros_like(addresses)
        )
        forceful = normal_force > self.config.reward.undesired_contact_force_threshold
        contact_body = jnp.asarray(self.model.geom_body_ids)[contact_geom]
        robot_body_ids = np.unique(self.model.geom_body_ids[self.model.robot_geom_ids])
        allowed = {self._body_id(name) for name in self.end_effector_names}
        undesired_body_ids = jnp.asarray(
            [body_id for body_id in robot_body_ids if body_id not in allowed],
            dtype=jnp.int32,
        )
        body_contact = (
            active[:, None]
            & forceful[:, None]
            & pair_contains_each(contact_body, undesired_body_ids)
        )
        return jnp.sum(jnp.any(body_contact, axis=0).astype(jnp.float32))

    def _joint_limit_cost(self, state: MotionTrackingState) -> jax.Array:
        positions = state.data.qpos[jnp.asarray(self.model.policy_qpos_indices)]
        policy = jnp.asarray(self.model.policy_ctrl_indices)
        lower = jnp.asarray(self.model.joint_target_low)[policy]
        upper = jnp.asarray(self.model.joint_target_high)[policy]
        span = jnp.maximum(upper - lower, 1.0e-6)
        margin = 0.02 * span
        cost = jnp.mean(
            jax.nn.relu(lower + margin - positions) ** 2
            + jax.nn.relu(positions - upper + margin) ** 2
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

    def _require_mjx(self) -> None:
        if self.model.mjx_model is None:
            raise RuntimeError(
                "this motion environment was loaded for native MuJoCo only"
            )

    @staticmethod
    def _validate_range(name: str, value: tuple[float, float]) -> None:
        if len(value) != 2 or not all(math.isfinite(item) for item in value):
            raise ValueError(f"{name} must contain two finite values")
        if value[0] > value[1]:
            raise ValueError(f"{name} lower bound exceeds upper bound")

    @staticmethod
    def _validate_nonnegative_vector(name: str, value: tuple[float, ...]) -> None:
        if not value or not all(math.isfinite(item) and item >= 0.0 for item in value):
            raise ValueError(f"{name} must contain finite nonnegative values")

    def _validate_observation_noise(self) -> None:
        config = self.config.observation_noise
        for name in (
            "anchor_position",
            "anchor_orientation",
            "base_linear_velocity",
            "base_angular_velocity",
            "joint_position",
            "joint_velocity",
        ):
            value = getattr(config, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"observation_noise.{name} must be nonnegative")

    def _validate_adaptive_sampling(self) -> None:
        config = self.config.adaptive_sampling
        if not math.isfinite(config.bin_seconds) or config.bin_seconds <= 0.0:
            raise ValueError("adaptive_sampling.bin_seconds must be positive")
        if not 0.0 < config.ema_alpha <= 1.0:
            raise ValueError("adaptive_sampling.ema_alpha must be in (0, 1]")
        if config.kernel_size <= 0:
            raise ValueError("adaptive_sampling.kernel_size must be positive")
        if not 0.0 < config.kernel_decay <= 1.0:
            raise ValueError("adaptive_sampling.kernel_decay must be in (0, 1]")
        if not 0.0 <= config.uniform_ratio <= 1.0:
            raise ValueError("adaptive_sampling.uniform_ratio must be in [0, 1]")

    def _validate_push(self) -> None:
        config = self.config.push
        self._validate_range("push.interval_seconds", config.interval_seconds)
        if config.interval_seconds[0] <= 0.0:
            raise ValueError("push.interval_seconds must be positive")
        self._validate_nonnegative_vector(
            "push.linear_velocity", config.linear_velocity
        )
        self._validate_nonnegative_vector(
            "push.angular_velocity", config.angular_velocity
        )


def gaussian_reward(error: jax.Array, sigma: float) -> jax.Array:
    return jnp.exp(-jnp.mean(jnp.square(error)) / (sigma * sigma))


def squared_norm_gaussian_reward(error: jax.Array, sigma: float) -> jax.Array:
    squared_norm = jnp.sum(jnp.square(error), axis=-1)
    return jnp.exp(-jnp.mean(squared_norm) / (sigma * sigma))


def scalar_gaussian_reward(error: jax.Array, sigma: float) -> jax.Array:
    return jnp.exp(-jnp.mean(jnp.square(error)) / (sigma * sigma))


def symmetric_vector_noise(rng: jax.Array, maximum: tuple[float, ...]) -> jax.Array:
    maximum_array = jnp.asarray(maximum, dtype=jnp.float32)
    return jax.random.uniform(
        rng,
        maximum_array.shape,
        minval=-maximum_array,
        maxval=maximum_array,
        dtype=jnp.float32,
    )


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


def rotation_6d(quat: jax.Array) -> jax.Array:
    w, x, y, z = quat / jnp.maximum(jnp.linalg.norm(quat), 1.0e-8)
    matrix = jnp.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w)],
        ]
    )
    return matrix.reshape(-1)


def quat_from_euler_xyz(angles: jax.Array) -> jax.Array:
    half = 0.5 * angles
    cr, cp, cy = jnp.cos(half)
    sr, sp, sy = jnp.sin(half)
    return jnp.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )
