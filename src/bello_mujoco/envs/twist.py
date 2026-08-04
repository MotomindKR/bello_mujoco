"""Stage-isolated TWIST observations on the Bello MJX motion environment."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Literal

import chex
import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.contracts import ActionLayout, ObservationLayout
from bello_mujoco.envs.locomotion import (
    pair_contains_each,
    quat_rotate,
    quat_rotate_inverse,
    tree_where,
    yaw_quat_from_angle,
)
from bello_mujoco.envs.motion_tracking import (
    BelloMotionTrackingEnv,
    MotionTrackingEnvConfig,
    quat_multiply,
    rotation_6d,
)
from bello_mujoco.motion.clip import MotionSample, motion_tracking_asset_config
from bello_mujoco.motion.twist_dataset import (
    TWIST_FUTURE_OFFSETS,
    PackedReferenceBank,
)
from bello_mujoco.sim import WarpConfig

TwistStage = Literal["teacher", "student"]
TWIST_SIMULATION_HZ = 500
TWIST_POLICY_HZ = 50
TWIST_DECIMATION = TWIST_SIMULATION_HZ // TWIST_POLICY_HZ
TWIST_EPISODE_STEPS = 10 * TWIST_POLICY_HZ
TWIST_HISTORY_FRAMES = 11

DEFAULT_KEY_BODY_NAMES = (
    "pelvis_link",
    "torso_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "l_wrist_link",
    "r_wrist_link",
    "left_knee_link",
    "right_knee_link",
    "l_elbow_link",
)


@dataclass(frozen=True)
class TwistObservationContract:
    joint_names: tuple[str, ...]
    key_body_names: tuple[str, ...] = DEFAULT_KEY_BODY_NAMES
    future_offsets: tuple[int, ...] = TWIST_FUTURE_OFFSETS
    history_frames: int = TWIST_HISTORY_FRAMES
    randomization_size: int = 27

    def __post_init__(self) -> None:
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("TWIST joint_names must be nonempty and unique")
        if len(self.key_body_names) != 9 or len(set(self.key_body_names)) != 9:
            raise ValueError("TWIST requires exactly nine unique key bodies")
        if self.future_offsets != TWIST_FUTURE_OFFSETS:
            raise ValueError(
                "TWIST future offsets must match the released 20-offset contract"
            )
        if self.history_frames != TWIST_HISTORY_FRAMES:
            raise ValueError("TWIST student history must contain exactly 11 frames")
        if self.randomization_size <= 0:
            raise ValueError("randomization_size must be positive")

    @property
    def action_size(self) -> int:
        return len(self.joint_names)

    @property
    def reference_frame_size(self) -> int:
        # height + orientation 6D + local linear velocity + yaw rate +
        # joint position + nine local key-body positions.
        return 1 + 6 + 3 + 1 + self.action_size + 9 * 3

    @property
    def proprioception_frame_size(self) -> int:
        # base angular velocity + roll/pitch + q + qdot + previous action.
        return 3 + 2 + 3 * self.action_size

    @property
    def teacher_observation_size(self) -> int:
        return (
            len(self.future_offsets) * self.reference_frame_size
            + self.proprioception_frame_size
        )

    @property
    def student_observation_size(self) -> int:
        return (
            self.reference_frame_size
            + self.history_frames * self.proprioception_frame_size
        )

    @property
    def critic_observation_size(self) -> int:
        return (
            self.teacher_observation_size
            + 3  # base linear velocity
            + 1  # root height
            + 9 * 3  # current key-body positions
            + 2  # physical foot contacts
            + self.randomization_size
        )

    def actor_size(self, stage: TwistStage) -> int:
        return (
            self.teacher_observation_size
            if stage == "teacher"
            else self.student_observation_size
        )

    def actor_layout(self, stage: TwistStage) -> ObservationLayout:
        if stage == "teacher":
            return ObservationLayout(
                (
                    (
                        "future_reference",
                        len(self.future_offsets) * self.reference_frame_size,
                    ),
                    ("current_proprioception", self.proprioception_frame_size),
                )
            )
        return ObservationLayout(
            (
                ("current_reference", self.reference_frame_size),
                (
                    "proprioception_history",
                    self.history_frames * self.proprioception_frame_size,
                ),
            )
        )

    @property
    def critic_layout(self) -> ObservationLayout:
        return ObservationLayout(
            (
                ("teacher_actor_observation", self.teacher_observation_size),
                ("base_linear_velocity", 3),
                ("root_height", 1),
                ("current_key_body_positions", 27),
                ("foot_contacts", 2),
                ("domain_randomization", self.randomization_size),
            )
        )


@dataclass(frozen=True)
class TwistObservationNoiseConfig:
    enabled: bool = True
    base_angular_velocity: float = 0.05
    roll_pitch: float = 0.02
    joint_position: float = 0.01
    joint_velocity: float = 0.1


@dataclass(frozen=True)
class TwistEnvConfig:
    max_episode_steps: int = TWIST_EPISODE_STEPS
    action_delay_steps: tuple[int, int] = (0, 2)
    observation_noise: TwistObservationNoiseConfig = TwistObservationNoiseConfig()
    key_body_names: tuple[str, ...] = DEFAULT_KEY_BODY_NAMES

    def __post_init__(self) -> None:
        if self.max_episode_steps != TWIST_EPISODE_STEPS:
            raise ValueError("TWIST episodes must be exactly 10 seconds at 50 Hz")
        low, high = self.action_delay_steps
        if low < 0 or high < low:
            raise ValueError(
                "action_delay_steps must be an increasing nonnegative range"
            )
        if high > 8:
            raise ValueError("action_delay_steps above 8 are not supported")
        if len(self.key_body_names) != 9:
            raise ValueError("TWIST requires exactly nine key bodies")


@chex.dataclass(frozen=True)
class TwistState:
    motion_state: object
    proprioception_history: jax.Array
    action_history: jax.Array
    action_delay_steps: jax.Array
    clip_index: jax.Array


@dataclass(frozen=True)
class BelloTwistEnv:
    """A TWIST stage view over the existing JAX/MJX whole-body tracker."""

    base: BelloMotionTrackingEnv
    stage: TwistStage
    config: TwistEnvConfig = TwistEnvConfig()
    reference_bank: PackedReferenceBank | None = None

    def __post_init__(self) -> None:
        if self.stage not in {"teacher", "student"}:
            raise ValueError(f"invalid TWIST stage: {self.stage!r}")
        if self.base.decimation != TWIST_DECIMATION:
            raise ValueError(
                f"TWIST requires decimation={TWIST_DECIMATION} for 500/50 Hz"
            )
        if not math.isclose(
            self.base.control_timestep,
            1.0 / TWIST_POLICY_HZ,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("TWIST policy timestep must be 0.02 seconds")
        if self.base.motion.num_frames <= max(TWIST_FUTURE_OFFSETS):
            raise ValueError("TWIST motion is too short for its future horizon")
        missing = set(self.config.key_body_names) - set(self.base.motion.body_names)
        if missing:
            raise ValueError(f"TWIST key bodies missing from motion: {sorted(missing)}")
        if self.contract.randomization_size != self.action_size + 2:
            raise ValueError("TWIST randomization contract has drifted")
        if self.reference_bank is not None:
            clip_count = len(self.reference_bank.sampling_probabilities)
            if self.reference_bank.lengths.shape != (clip_count,) or (
                self.reference_bank.offsets.shape != (clip_count,)
            ):
                raise ValueError("TWIST reference bank clip metadata has drifted")
            total_frames = int(np.sum(self.reference_bank.lengths, dtype=np.int64))
            if any(
                value.shape[0] != total_frames
                for value in (
                    self.reference_bank.qpos,
                    self.reference_bank.qvel,
                    self.reference_bank.body_pos,
                    self.reference_bank.body_quat_wxyz,
                    self.reference_bank.body_lin_vel,
                    self.reference_bank.body_ang_vel,
                    self.reference_bank.foot_contacts,
                )
            ):
                raise ValueError("TWIST reference bank frame arrays have drifted")
            if tuple(self.reference_bank.qpos.shape[1:]) != (
                self.base.model.mj_model.nq,
            ):
                raise ValueError("TWIST reference bank qpos shape differs from model")

    @classmethod
    def load(
        cls,
        stage: TwistStage,
        asset_config: AssetPrepConfig,
        motion_config: MotionTrackingEnvConfig,
        *,
        config: TwistEnvConfig = TwistEnvConfig(),
        warp: WarpConfig = WarpConfig(),
        batch_size: int = 1,
        reference_bank: PackedReferenceBank | None = None,
    ) -> "BelloTwistEnv":
        motion_config = replace(motion_config, decimation=TWIST_DECIMATION)
        base = BelloMotionTrackingEnv.load(
            asset_config,
            motion_config,
            warp=warp,
            batch_size=batch_size,
            simulation_timestep=1.0 / TWIST_SIMULATION_HZ,
        )
        return cls(base, stage, config, reference_bank)

    @property
    def contract(self) -> TwistObservationContract:
        return TwistObservationContract(
            joint_names=self.base.model.actuator_joint_names,
            key_body_names=self.config.key_body_names,
            randomization_size=self.action_size + 2,
        )

    @property
    def task(self) -> Literal["twist_teacher", "twist_student"]:
        return f"twist_{self.stage}"  # type: ignore[return-value]

    @property
    def observation_layout(self) -> ObservationLayout:
        return self.contract.actor_layout(self.stage)

    @property
    def critic_observation_layout(self) -> ObservationLayout:
        return self.contract.critic_layout

    @property
    def observation_size(self) -> int:
        return self.observation_layout.size

    @property
    def critic_observation_size(self) -> int:
        return self.critic_observation_layout.size

    @property
    def action_size(self) -> int:
        return self.base.action_size

    @property
    def action_layout(self) -> ActionLayout:
        return self.base.action_layout

    @property
    def action_clip(self) -> float:
        return self.base.action_clip

    @property
    def control_timestep(self) -> float:
        return self.base.control_timestep

    @property
    def ppo_stagger_initial_step_count(self) -> bool:
        return False

    @property
    def key_body_ids(self) -> np.ndarray:
        return np.asarray(
            [
                self.base.model.mj_model.body(name).id
                for name in self.config.key_body_names
            ]
        )

    def reset(self, rng: jax.Array | None = None) -> tuple[TwistState, jax.Array]:
        motion_key = clip_key = delay_key = obs_key = rng
        if rng is not None:
            motion_key, clip_key, delay_key, obs_key = jax.random.split(rng, 4)
        motion_state, _ = self.base.reset(motion_key)
        clip_index, reference_time = self._sample_clip_start(clip_key)
        if self.reference_bank is not None:
            motion_state = motion_state.replace(reference_time=reference_time)
            reference = self._aligned_reference(motion_state, clip_index, 0)
            data = motion_state.data.replace(
                qpos=reference.qpos,
                qvel=reference.qvel,
                ctrl=self.base._initial_ctrl(
                    reference, motion_state.joint_default_offset
                ),
            )
            data = mjx.forward(self.base.model.mjx_model, data)
            motion_state = motion_state.replace(data=data)
        proprioception = self._proprioception(motion_state, obs_key)
        history = jnp.repeat(proprioception[None, :], TWIST_HISTORY_FRAMES, axis=0)
        delay = self._sample_action_delay(delay_key)
        action_history = jnp.zeros(
            (self.config.action_delay_steps[1] + 1, self.action_size),
            dtype=jnp.float32,
        )
        state = TwistState(
            motion_state=motion_state,
            proprioception_history=history,
            action_history=action_history,
            action_delay_steps=delay,
            clip_index=clip_index,
        )
        return state, self.observation(state)

    def reset_batch(
        self, batch_size: int, rng: jax.Array | None = None
    ) -> tuple[TwistState, jax.Array]:
        if rng is None:
            state, observation = jax.vmap(lambda _: self.reset())(
                jnp.arange(batch_size)
            )
        else:
            state, observation = jax.vmap(self.reset)(jax.random.split(rng, batch_size))
        data = state.motion_state.data.replace(
            _impl=state.motion_state.data._impl.replace(nworld=batch_size)
        )
        return (
            state.replace(motion_state=state.motion_state.replace(data=data)),
            observation,
        )

    def initial_step_counts(
        self, batch_size: int, rng: jax.Array | None = None
    ) -> jax.Array:
        del rng
        return jnp.zeros((batch_size,), dtype=jnp.int32)

    def with_initial_step_counts(
        self, state: TwistState, step_count: jax.Array
    ) -> TwistState:
        return state.replace(
            motion_state=state.motion_state.replace(
                step_count=step_count, episode_length=step_count
            )
        )

    def step(
        self, state: TwistState, action: jax.Array, rng: jax.Array | None = None
    ) -> tuple[TwistState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        if self.reference_bank is not None:
            return self._step_reference_bank(state, action, rng)
        action_history = jnp.concatenate(
            (state.action_history[1:], action[None, :]), axis=0
        )
        applied = jax.lax.dynamic_index_in_dim(
            action_history,
            action_history.shape[0] - 1 - state.action_delay_steps,
            keepdims=False,
        )
        motion_state, _, reward, done, info = self.base.step(
            state.motion_state, applied
        )
        timeout = motion_state.step_count >= self.config.max_episode_steps
        done = done | timeout
        proprioception = self._proprioception(motion_state, rng)
        history = jnp.concatenate(
            (state.proprioception_history[1:], proprioception[None, :]), axis=0
        )
        next_state = TwistState(
            motion_state=motion_state,
            proprioception_history=history,
            action_history=action_history,
            action_delay_steps=state.action_delay_steps,
            clip_index=state.clip_index,
        )
        info = {
            **info,
            "done/twist_timeout": timeout.astype(jnp.float32),
            "delay/steps": state.action_delay_steps.astype(jnp.float32),
        }
        return next_state, self.observation(next_state), reward, done, info

    def step_batch(
        self,
        state: TwistState,
        action: jax.Array,
        rng: jax.Array | None = None,
    ) -> tuple[TwistState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        if self.reference_bank is not None:
            return self._step_reference_bank_batch(state, action, rng)
        batch_size = int(action.shape[0])
        step_key = reset_key = obs_key = rng
        if rng is not None:
            step_key, reset_key, obs_key = jax.random.split(rng, 3)
        action_history = jnp.concatenate(
            (state.action_history[:, 1:], action[:, None, :]), axis=1
        )
        indices = action_history.shape[1] - 1 - state.action_delay_steps
        applied = jax.vmap(
            lambda values, index: jax.lax.dynamic_index_in_dim(
                values, index, keepdims=False
            )
        )(action_history, indices)
        motion_state, _, reward, base_done, info = self.base.step_batch(
            state.motion_state, applied, step_key
        )
        timeout = state.motion_state.step_count + 1 >= self.config.max_episode_steps
        done = base_done | timeout
        if obs_key is None:
            proprioception = jax.vmap(self._proprioception)(motion_state)
        else:
            keys = jax.random.split(obs_key, batch_size)
            proprioception = jax.vmap(self._proprioception)(motion_state, keys)
        history = jnp.concatenate(
            (state.proprioception_history[:, 1:], proprioception[:, None, :]), axis=1
        )
        stepped = TwistState(
            motion_state=motion_state,
            proprioception_history=history,
            action_history=action_history,
            action_delay_steps=state.action_delay_steps,
            clip_index=state.clip_index,
        )
        stepped_observation = self.observation_batch(stepped)
        stepped_critic_observation = self.critic_observation_batch(stepped)
        reset_state, _ = self.reset_batch(batch_size, reset_key)
        stepped = tree_where(done, reset_state, stepped)
        info = {
            **info,
            "bootstrap_observation": stepped_observation,
            "bootstrap_critic_observation": stepped_critic_observation,
            "done/twist_timeout": timeout.astype(jnp.float32),
            "delay/steps": state.action_delay_steps.astype(jnp.float32),
        }
        return stepped, self.observation_batch(stepped), reward, done, info

    def observation(self, state: TwistState, rng: jax.Array | None = None) -> jax.Array:
        del rng
        if self.stage == "teacher":
            return self._teacher_observation(state)
        current = self._reference_frame(state, 1)
        return jnp.concatenate((current, state.proprioception_history.reshape(-1)))

    def observation_batch(
        self, state: TwistState, rng: jax.Array | None = None
    ) -> jax.Array:
        del rng
        return jax.vmap(self.observation)(state)

    def critic_observation(self, state: TwistState) -> jax.Array:
        motion_state = state.motion_state
        root_quat = motion_state.data.qpos[3:7]
        base_linear_velocity = quat_rotate_inverse(
            root_quat, motion_state.data.qvel[:3]
        )
        key_positions = self._current_key_body_positions(motion_state)
        contacts = self._foot_contacts(motion_state, 0)
        randomization = jnp.concatenate(
            (
                motion_state.joint_default_offset,
                jnp.asarray(
                    [
                        state.action_delay_steps,
                        motion_state.next_push_step
                        / max(self.config.max_episode_steps, 1),
                    ],
                    dtype=jnp.float32,
                ),
            )
        )
        return jnp.concatenate(
            (
                self._teacher_observation(state),
                base_linear_velocity,
                motion_state.data.qpos[2:3],
                key_positions,
                contacts,
                randomization,
            )
        )

    def critic_observation_batch(
        self, state: TwistState, rng: jax.Array | None = None
    ) -> jax.Array:
        del rng
        world_ids = jnp.arange(state.action_delay_steps.shape[0], dtype=jnp.int32)
        return jax.vmap(self.critic_observation_with_world)(state, world_ids)

    def critic_observation_with_world(
        self, state: TwistState, world_id: jax.Array
    ) -> jax.Array:
        motion_state = state.motion_state
        root_quat = motion_state.data.qpos[3:7]
        base_linear_velocity = quat_rotate_inverse(
            root_quat, motion_state.data.qvel[:3]
        )
        randomization = jnp.concatenate(
            (
                motion_state.joint_default_offset,
                jnp.asarray(
                    [
                        state.action_delay_steps,
                        motion_state.next_push_step
                        / max(self.config.max_episode_steps, 1),
                    ],
                    dtype=jnp.float32,
                ),
            )
        )
        return jnp.concatenate(
            (
                self._teacher_observation(state),
                base_linear_velocity,
                motion_state.data.qpos[2:3],
                self._current_key_body_positions(motion_state),
                self._foot_contacts(motion_state, world_id),
                randomization,
            )
        )

    def ppo_action_diagnostics(self, env_action: jax.Array) -> dict[str, jax.Array]:
        return self.base.ppo_action_diagnostics(env_action)

    def _teacher_observation(self, state: TwistState) -> jax.Array:
        future = jnp.concatenate(
            tuple(
                self._reference_frame(state, offset) for offset in TWIST_FUTURE_OFFSETS
            )
        )
        return jnp.concatenate((future, state.proprioception_history[-1]))

    def _reference_frame(self, state: TwistState, offset: int) -> jax.Array:
        motion_state = state.motion_state
        reference = self._aligned_reference(motion_state, state.clip_index, offset)
        root_quat = reference.qpos[3:7]
        local_velocity = quat_rotate_inverse(root_quat, reference.qvel[:3])
        key_indices = jnp.asarray(
            [
                self.base.motion.body_names.index(name)
                for name in self.config.key_body_names
            ]
        )
        key_positions = reference.body_pos[key_indices]
        root_position = reference.qpos[:3]
        local_positions = jax.vmap(
            lambda value: quat_rotate_inverse(root_quat, value - root_position)
        )(key_positions)
        return jnp.concatenate(
            (
                reference.qpos[2:3],
                rotation_6d(root_quat),
                local_velocity,
                reference.qvel[5:6],
                reference.qpos[jnp.asarray(self.base.model.policy_qpos_indices)],
                local_positions.reshape(-1),
            )
        )

    def _step_reference_bank(
        self, state: TwistState, action: jax.Array, rng: jax.Array | None
    ):
        action = jnp.clip(jnp.asarray(action), -self.action_clip, self.action_clip)
        action_history = jnp.concatenate(
            (state.action_history[1:], action[None, :]), axis=0
        )
        applied = jax.lax.dynamic_index_in_dim(
            action_history,
            action_history.shape[0] - 1 - state.action_delay_steps,
            keepdims=False,
        )
        data = state.motion_state.data.replace(
            ctrl=self.base.action_target(state.motion_state, applied)
        )
        data = jax.lax.fori_loop(
            0,
            self.base.decimation,
            lambda _, value: mjx.step(self.base.model.mjx_model, value),
            data,
        )
        motion_state = state.motion_state.replace(
            data=data,
            last_action=applied,
            reference_time=(state.motion_state.reference_time + self.control_timestep),
            step_count=state.motion_state.step_count + 1,
            episode_length=state.motion_state.episode_length + 1,
        )
        reference = self._aligned_reference(motion_state, state.clip_index, 0)
        reward, reward_info = self.base._beyondmimic_reward(
            state.motion_state, motion_state, reference, 0
        )
        done_flags = self._bank_termination_flags(
            motion_state, state.clip_index, reference
        )
        reward, reward_info = self.base._sanitize_nonfinite_reward(
            reward, reward_info, done_flags
        )
        episode_return = state.motion_state.episode_return + reward
        motion_state = motion_state.replace(
            episode_return=episode_return,
            last_episode_return=jnp.where(
                done_flags["done/any"],
                episode_return,
                state.motion_state.last_episode_return,
            ),
            last_episode_length=jnp.where(
                done_flags["done/any"],
                motion_state.episode_length,
                state.motion_state.last_episode_length,
            ),
        )
        proprioception = self._proprioception(motion_state, rng)
        history = jnp.concatenate(
            (state.proprioception_history[1:], proprioception[None, :]), axis=0
        )
        stepped = TwistState(
            motion_state=motion_state,
            proprioception_history=history,
            action_history=action_history,
            action_delay_steps=state.action_delay_steps,
            clip_index=state.clip_index,
        )
        observation = self.observation(stepped)
        info = self._bank_info(stepped, observation, reward_info, done_flags)
        return (
            stepped,
            observation,
            reward,
            done_flags["done/any"],
            info,
        )

    def _step_reference_bank_batch(
        self, state: TwistState, action: jax.Array, rng: jax.Array | None
    ):
        batch_size = int(action.shape[0])
        action = jnp.clip(jnp.asarray(action), -self.action_clip, self.action_clip)
        push_key = reset_key = obs_key = rng
        if rng is not None:
            push_key, reset_key, obs_key = jax.random.split(rng, 3)
        action_history = jnp.concatenate(
            (state.action_history[:, 1:], action[:, None, :]), axis=1
        )
        indices = action_history.shape[1] - 1 - state.action_delay_steps
        applied = jax.vmap(
            lambda values, index: jax.lax.dynamic_index_in_dim(
                values, index, keepdims=False
            )
        )(action_history, indices)
        ctrl = jax.vmap(self.base.action_target)(state.motion_state, applied)
        data = state.motion_state.data.replace(
            ctrl=ctrl,
            _impl=state.motion_state.data._impl.replace(nworld=batch_size),
        )
        data, next_push_step, push_applied = self.base._maybe_apply_push_batch(
            data, state.motion_state, push_key
        )
        data = jax.lax.fori_loop(
            0,
            self.base.decimation,
            lambda _, value: mjx.step(self.base.model.mjx_model, value),
            data,
        )
        motion_state = state.motion_state.replace(
            data=data,
            last_action=applied,
            reference_time=(state.motion_state.reference_time + self.control_timestep),
            step_count=state.motion_state.step_count + 1,
            episode_length=state.motion_state.episode_length + 1,
            next_push_step=next_push_step,
        )
        world_ids = jnp.arange(batch_size, dtype=jnp.int32)

        def reward_one(previous, current, clip_index, world_id):
            reference = self._aligned_reference(current, clip_index, 0)
            return self.base._beyondmimic_reward(previous, current, reference, world_id)

        reward, reward_info = jax.vmap(reward_one)(
            state.motion_state, motion_state, state.clip_index, world_ids
        )
        references = jax.vmap(
            lambda current, clip: self._aligned_reference(current, clip, 0)
        )(motion_state, state.clip_index)
        done_flags = jax.vmap(self._bank_termination_flags)(
            motion_state, state.clip_index, references
        )
        reward, reward_info = self.base._sanitize_nonfinite_reward(
            reward, reward_info, done_flags
        )
        done = done_flags["done/any"]
        episode_return = state.motion_state.episode_return + reward
        motion_state = motion_state.replace(
            episode_return=episode_return,
            last_episode_return=jnp.where(
                done, episode_return, state.motion_state.last_episode_return
            ),
            last_episode_length=jnp.where(
                done,
                motion_state.episode_length,
                state.motion_state.last_episode_length,
            ),
        )
        if obs_key is None:
            proprioception = jax.vmap(self._proprioception)(motion_state)
        else:
            keys = jax.random.split(obs_key, batch_size)
            proprioception = jax.vmap(self._proprioception)(motion_state, keys)
        history = jnp.concatenate(
            (state.proprioception_history[:, 1:], proprioception[:, None, :]),
            axis=1,
        )
        stepped = TwistState(
            motion_state=motion_state,
            proprioception_history=history,
            action_history=action_history,
            action_delay_steps=state.action_delay_steps,
            clip_index=state.clip_index,
        )
        stepped_observation = self.observation_batch(stepped)
        info = self._bank_info(stepped, stepped_observation, reward_info, done_flags)
        info["push/applied"] = push_applied.astype(jnp.float32)
        reset_state, reset_observation = self.reset_batch(batch_size, reset_key)
        reset_state = reset_state.replace(
            motion_state=reset_state.motion_state.replace(
                last_episode_return=motion_state.last_episode_return,
                last_episode_length=motion_state.last_episode_length,
            )
        )
        next_state = tree_where(done, reset_state, stepped)
        next_observation = jnp.where(
            done[:, None], reset_observation, stepped_observation
        )
        return next_state, next_observation, reward, done, info

    def _bank_info(
        self,
        state: TwistState,
        observation: jax.Array,
        reward_info: dict[str, jax.Array],
        done_flags: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        done = done_flags["done/any"]
        return {
            "bootstrap_observation": observation,
            "bootstrap_critic_observation": (
                self.critic_observation_batch(state)
                if observation.ndim == 2
                else self.critic_observation(state)
            ),
            "episode_return": jnp.where(done, state.motion_state.episode_return, 0.0),
            "episode_length": jnp.where(done, state.motion_state.episode_length, 0),
            "returned_episode": done,
            "reference/time": state.motion_state.reference_time,
            "reference/clip_index": state.clip_index.astype(jnp.float32),
            "delay/steps": state.action_delay_steps.astype(jnp.float32),
            **done_flags,
            **reward_info,
        }

    def _bank_termination_flags(
        self,
        state,
        clip_index: jax.Array,
        reference: MotionSample,
    ) -> dict[str, jax.Array]:
        ids = jnp.asarray(self.base.tracked_body_ids)
        anchor_id = self.base.anchor_body_id
        anchor_index = self.base.anchor_body_index
        target_position, _ = self.base._relative_body_targets(state, reference)
        height = (
            jnp.abs(state.data.xpos[anchor_id, 2] - reference.body_pos[anchor_index, 2])
            > self.base.config.done.root_height_error
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
            > self.base.config.done.root_orientation_error
        )
        end_effector_indices = jnp.asarray(self.base.end_effector_indices)
        end_effector_error = jnp.abs(
            state.data.xpos[ids[end_effector_indices], 2]
            - target_position[end_effector_indices, 2]
        )
        end_effector = jnp.any(
            end_effector_error > self.base.config.done.end_effector_position_error
        )
        nonfinite = ~(
            jnp.all(jnp.isfinite(state.data.qpos))
            & jnp.all(jnp.isfinite(state.data.qvel))
            & jnp.all(jnp.isfinite(state.data.xpos))
            & jnp.all(jnp.isfinite(state.data.xquat))
            & jnp.all(jnp.isfinite(state.data.cvel))
            & jnp.all(jnp.isfinite(state.data.actuator_force))
            & jnp.all(jnp.isfinite(state.data.ctrl))
        )
        frame = jnp.rint(state.reference_time * TWIST_POLICY_HZ).astype(jnp.int32)
        assert self.reference_bank is not None
        last_safe_frame = (
            jnp.asarray(self.reference_bank.lengths)[clip_index]
            - max(TWIST_FUTURE_OFFSETS)
            - 1
        )
        timeout = (state.step_count >= self.config.max_episode_steps) | (
            frame >= last_safe_frame
        )
        terminal = height | orientation | end_effector | nonfinite
        return {
            "done/height": height,
            "done/orientation": orientation,
            "done/end_effector": end_effector,
            "done/nonfinite": nonfinite,
            "done/timeout": timeout,
            "done/twist_timeout": (state.step_count >= self.config.max_episode_steps),
            "done/terminal": terminal,
            "done/any": terminal | timeout,
        }

    def _sample_clip_start(self, rng: jax.Array | None) -> tuple[jax.Array, jax.Array]:
        if self.reference_bank is None:
            return (
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0.0, dtype=jnp.float32),
            )
        clip_key = frame_key = rng
        if rng is not None:
            clip_key, frame_key = jax.random.split(rng)
        probabilities = jnp.asarray(self.reference_bank.sampling_probabilities)
        if clip_key is None:
            clip_index = jnp.asarray(0, dtype=jnp.int32)
        else:
            clip_index = jax.random.categorical(
                clip_key, jnp.log(probabilities)
            ).astype(jnp.int32)
        length = jnp.asarray(self.reference_bank.lengths)[clip_index]
        maximum_start = jnp.maximum(length - max(TWIST_FUTURE_OFFSETS) - 1, 0)
        if frame_key is None:
            start_frame = jnp.asarray(0, dtype=jnp.int32)
        else:
            start_frame = jnp.floor(
                jax.random.uniform(frame_key) * (maximum_start + 1)
            ).astype(jnp.int32)
        return clip_index, start_frame.astype(jnp.float32) / TWIST_POLICY_HZ

    def _aligned_reference(
        self, state, clip_index: jax.Array, offset: int
    ) -> MotionSample:
        if self.reference_bank is None:
            time = jnp.minimum(
                state.reference_time + offset * self.control_timestep,
                self.base.motion.duration,
            )
            return self.base.aligned_reference(
                time, state.alignment_xy, state.alignment_yaw
            )
        frame = (
            jnp.rint(state.reference_time * TWIST_POLICY_HZ).astype(jnp.int32) + offset
        )
        length = jnp.asarray(self.reference_bank.lengths)[clip_index]
        frame = jnp.clip(frame, 0, length - 1)
        absolute_frame = jnp.asarray(self.reference_bank.offsets)[clip_index] + frame

        def value(name: str) -> jax.Array:
            return jnp.asarray(getattr(self.reference_bank, name))[absolute_frame]

        qpos = value("qpos")
        qvel = value("qvel")
        body_pos = value("body_pos")
        body_quat = value("body_quat_wxyz")
        body_lin_vel = value("body_lin_vel")
        body_ang_vel = value("body_ang_vel")
        rotation = yaw_quat_from_angle(state.alignment_yaw)
        root_position = quat_rotate(rotation, qpos[:3]).at[:2].add(state.alignment_xy)
        return MotionSample(
            qpos=qpos.at[:3]
            .set(root_position)
            .at[3:7]
            .set(quat_multiply(rotation, qpos[3:7])),
            qvel=qvel.at[:3].set(quat_rotate(rotation, qvel[:3])),
            body_pos=jax.vmap(lambda item: quat_rotate(rotation, item))(body_pos)
            .at[:, :2]
            .add(state.alignment_xy),
            body_quat=jax.vmap(lambda item: quat_multiply(rotation, item))(body_quat),
            body_lin_vel=jax.vmap(lambda item: quat_rotate(rotation, item))(
                body_lin_vel
            ),
            body_ang_vel=jax.vmap(lambda item: quat_rotate(rotation, item))(
                body_ang_vel
            ),
            foot_contacts=value("foot_contacts"),
        )

    def _proprioception(self, state, rng: jax.Array | None = None) -> jax.Array:
        root_quat = state.data.qpos[3:7]
        base_angular_velocity = state.data.qvel[3:6]
        roll_pitch = _roll_pitch(root_quat)
        qpos = state.data.qpos[jnp.asarray(self.base.model.policy_qpos_indices)]
        qvel = state.data.qvel[jnp.asarray(self.base.model.policy_qvel_indices)]
        if rng is not None and self.config.observation_noise.enabled:
            keys = jax.random.split(rng, 4)
            cfg = self.config.observation_noise
            base_angular_velocity = _uniform_noise(
                base_angular_velocity, cfg.base_angular_velocity, keys[0]
            )
            roll_pitch = _uniform_noise(roll_pitch, cfg.roll_pitch, keys[1])
            qpos = _uniform_noise(qpos, cfg.joint_position, keys[2])
            qvel = _uniform_noise(qvel, cfg.joint_velocity, keys[3])
        return jnp.concatenate(
            (base_angular_velocity, roll_pitch, qpos, qvel, state.last_action)
        )

    def _current_key_body_positions(self, state) -> jax.Array:
        root_quat = state.data.qpos[3:7]
        root_position = state.data.qpos[:3]
        positions = state.data.xpos[jnp.asarray(self.key_body_ids)]
        return jax.vmap(
            lambda value: quat_rotate_inverse(root_quat, value - root_position)
        )(positions).reshape(-1)

    def _sample_action_delay(self, rng: jax.Array | None) -> jax.Array:
        low, high = self.config.action_delay_steps
        if rng is None or low == high:
            return jnp.asarray(low, dtype=jnp.int32)
        return jax.random.randint(rng, (), low, high + 1, dtype=jnp.int32)

    def _foot_contacts(self, state, world_id: jax.Array | int) -> jax.Array:
        impl = state.data._impl
        contact_geom = jnp.asarray(impl.contact__geom, dtype=jnp.int32)
        contact_geom = jnp.clip(contact_geom, 0, self.base.model.mj_model.ngeom - 1)
        nacon = jnp.ravel(jnp.asarray(impl.nacon, dtype=jnp.int32))[0]
        active = (jnp.arange(contact_geom.shape[0]) < nacon) & (
            jnp.asarray(impl.contact__worldid) == jnp.asarray(world_id)
        )
        geom_body = jnp.asarray(self.base.model.geom_body_ids)[contact_geom]
        return jnp.any(
            active[:, None]
            & pair_contains_each(geom_body, self.base.model.foot_body_ids),
            axis=0,
        ).astype(jnp.float32)


def teacher_observation_from_student_critic(
    critic_observation: jax.Array, contract: TwistObservationContract
) -> jax.Array:
    """Return the exact teacher actor input stored in a student critic input."""

    return critic_observation[..., : contract.teacher_observation_size]


def _roll_pitch(quaternion: jax.Array) -> jax.Array:
    w, x, y, z = quaternion
    roll = jnp.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = jnp.arcsin(jnp.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return jnp.asarray((roll, pitch))


def _uniform_noise(value: jax.Array, magnitude: float, rng: jax.Array) -> jax.Array:
    if not math.isfinite(magnitude) or magnitude < 0.0:
        raise ValueError("TWIST observation noise magnitudes must be nonnegative")
    return value + jax.random.uniform(
        rng, value.shape, minval=-magnitude, maxval=magnitude
    )


def default_twist_asset_config() -> AssetPrepConfig:
    return motion_tracking_asset_config()
