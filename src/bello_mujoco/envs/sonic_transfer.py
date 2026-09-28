"""Multi-clip SONIC transfer environment for Bello Mini."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import jax
import jax.numpy as jnp

from bello_mujoco.contracts import ActionLayout, ObservationLayout
from bello_mujoco.envs.geometry import quat_rotate_inverse
from bello_mujoco.envs.motion_tracking import (
    BelloMotionTrackingEnv,
    MotionTrackingEnvConfig,
    MotionTrackingState,
    quat_conjugate,
    quat_distance,
    quat_multiply,
    rotation_6d,
)
from bello_mujoco.models.compiler import ModelConfig
from bello_mujoco.motion.clip import MotionSample
from bello_mujoco.robot_tasks import motion_task_definition
from bello_mujoco.sim import WarpConfig, load_model
from bello_mujoco.sonic.control import sonic_action_target
from bello_mujoco.sonic.embodiment import MiniEmbodimentAdapter
from bello_mujoco.sonic.observations import SonicObservationAssembler
from bello_mujoco.sonic.reference import ReferenceBank, load_reference_bank


@dataclass(frozen=True)
class BelloSonicTransferEnv(BelloMotionTrackingEnv):
    reference_bank: ReferenceBank | None = None
    embodiment: MiniEmbodimentAdapter | None = None
    observation_assembler: SonicObservationAssembler | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.model.robot.asset != "bello_mini":
            raise ValueError("sonic_transfer currently supports only bello_mini")
        if self.reference_bank is None or self.embodiment is None:
            raise ValueError("sonic_transfer requires a reference bank and embodiment")
        if self.observation_assembler is None:
            raise ValueError("sonic_transfer requires an observation assembler")
        if self.reference_bank.joint_names != self.motion.joint_names:
            raise ValueError("SONIC reference bank and model joint orders differ")
        expected_bodies = motion_task_definition(
            self.model.robot.asset
        ).beyondmimic_tracked_body_names
        if self.reference_bank.body_names != expected_bodies:
            raise ValueError(
                "SONIC release critic requires the canonical 14-body tracking "
                f"order; got {self.reference_bank.body_names}"
            )
        if self.reference_bank.split_counts[0] == 0:
            raise ValueError("SONIC reference bank requires at least one training clip")
        step_frames = self.reference_bank.fps * self.control_timestep
        if not math.log2(step_frames).is_integer():
            raise ValueError(
                "SONIC clips must be resampled to a power-of-two multiple of the control rate"
            )
        clock_end = self.reference_bank.clock_end
        if clock_end >= 2**24 * min(1.0, step_frames):
            raise ValueError(
                "SONIC reference bank exceeds exact float32 frame-clock capacity"
            )

    @classmethod
    def load_corpus(
        cls,
        asset_config: ModelConfig,
        env_config: MotionTrackingEnvConfig,
        motion_paths: tuple[Path, ...],
        *,
        evaluation_names: tuple[str, ...] = (),
        warp: WarpConfig = WarpConfig(),
        batch_size: int = 1,
        use_viewer_model: bool = False,
        load_mjx: bool = True,
    ) -> BelloSonicTransferEnv:
        if not motion_paths:
            raise ValueError("sonic_transfer requires at least one motion clip")
        model = load_model(
            asset_config,
            warp=warp,
            batch_size=batch_size,
            terrain=env_config.terrain,
            simulation_physics=env_config.simulation_physics,
            model_randomization=env_config.model_randomization,
            use_viewer_model=use_viewer_model,
            load_mjx=load_mjx,
        )
        bank = load_reference_bank(
            motion_paths, model, evaluation_names=evaluation_names
        )
        embodiment = MiniEmbodimentAdapter.from_model(model)
        assembler = SonicObservationAssembler(embodiment)
        first = bank.sample_frames(jnp.asarray(0), jnp.arange(int(bank.lengths[0])))
        from bello_mujoco.motion.clip import MotionClip

        first_clip = MotionClip(
            fps=bank.fps,
            joint_names=bank.joint_names,
            body_names=bank.body_names,
            qpos=first.qpos,
            qvel=first.qvel,
            body_pos=first.body_pos,
            body_quat=first.body_quat,
            body_lin_vel=first.body_lin_vel,
            body_ang_vel=first.body_ang_vel,
            foot_contacts=first.foot_contacts,
        )
        return cls(
            model=model,
            motion=first_clip,
            actuator_config=asset_config.actuator_config,
            config=env_config,
            reference_bank=bank,
            embodiment=embodiment,
            observation_assembler=assembler,
        )

    @property
    def task(self):
        return "sonic_transfer"

    @property
    def max_episode_steps(self) -> int:
        return max(
            1,
            math.ceil(
                (self.reference_bank.max_clip_frames - 1)
                / self.reference_bank.fps
                / self.control_timestep
            ),
        )

    @property
    def sampling_bin_count(self) -> int:
        return self.reference_bank.num_clips

    @property
    def observation_layout(self) -> ObservationLayout:
        return ObservationLayout(
            (
                ("sonic_future_reference", self.observation_assembler.reference_size),
                ("gravity_direction", 3),
                ("base_angular_velocity", 3),
                ("joint_position", self.action_size),
                ("joint_velocity", self.action_size),
                ("last_action", self.action_size),
            )
        )

    @property
    def critic_observation_layout(self) -> ObservationLayout:
        bodies = len(self.reference_bank.body_names)
        return ObservationLayout(
            (
                ("reference_joint_window", 580),
                ("anchor_position_error", 3),
                ("anchor_orientation_error_6d", 6),
                ("body_position", 3 * bodies),
                ("body_orientation_6d", 6 * bodies),
                ("base_linear_velocity", 3),
                ("base_angular_velocity", 3),
                ("joint_position", self.action_size),
                ("joint_velocity", self.action_size),
                ("last_action", self.action_size),
            )
        )

    @property
    def action_layout(self) -> ActionLayout:
        return ActionLayout(tuple(self.model.actuator_joint_names))

    def unaligned_reference(self, time: jax.Array) -> MotionSample:
        return self.reference_bank.sample_global_frame(time)

    def advance_reference_time(self, time: jax.Array) -> jax.Array:
        # Frame units make control increments exact even after hours of data.
        return self.reference_bank.advance_global_frame(
            time, self.reference_bank.fps * self.control_timestep
        )

    def reference_time_seconds(self, time: jax.Array) -> jax.Array:
        return time / self.reference_bank.fps

    def bounded_reference_time(self, time: jax.Array) -> jax.Array:
        global_frame = jnp.maximum(jnp.asarray(time), 0.0)
        clip_id = jnp.clip(
            jnp.searchsorted(
                self.reference_bank.time_offsets, global_frame, side="right"
            )
            - 1,
            0,
            self.reference_bank.num_clips - 1,
        )
        local_frame = jnp.clip(
            global_frame - self.reference_bank.time_offsets[clip_id],
            0.0,
            self.reference_bank.lengths[clip_id] - 1,
        )
        return self.reference_bank.time_offsets[clip_id] + local_frame

    def _reference_time_to_sampling_bin(self, reference_time: jax.Array) -> jax.Array:
        frame = jnp.maximum(reference_time, 0.0)
        return jnp.clip(
            jnp.searchsorted(self.reference_bank.time_offsets, frame, side="right") - 1,
            0,
            self.reference_bank.num_clips - 1,
        ).astype(jnp.int32)

    def _sampling_probabilities(self, failure_ema: jax.Array) -> jax.Array:
        return self.reference_bank.clip_sampling_probabilities(
            failure_ema,
            split="train",
            uniform_ratio=self.config.adaptive_sampling.uniform_ratio,
        )

    def _sample_reference_time(
        self, rng: jax.Array, sampling_failure_ema: jax.Array
    ) -> jax.Array:
        clip_key, frame_key = jax.random.split(rng)
        selected = self.reference_bank.sample_clip(
            clip_key,
            sampling_failure_ema,
            split="train",
            uniform_ratio=self.config.adaptive_sampling.uniform_ratio,
        )
        max_frame = jnp.maximum(self.reference_bank.lengths[selected] - 1, 1)
        local_frame = jax.random.uniform(frame_key, (), maxval=max_frame)
        return self.reference_bank.time_offsets[selected] + local_frame

    def observation(
        self,
        state: MotionTrackingState,
        rng: jax.Array | None = None,
        *,
        _feedback: jax.Array | None = None,
    ) -> jax.Array:
        if _feedback is None:
            _feedback = self._delayed_actor_feedback(state)
        qpos, qvel, _, anchor_quat = self._unpack_actor_feedback(_feedback)
        global_frame = self.bounded_reference_time(
            self.advance_reference_time(state.reference_time)
        )
        clip_id = jnp.clip(
            jnp.searchsorted(
                self.reference_bank.time_offsets, global_frame, side="right"
            )
            - 1,
            0,
            self.reference_bank.num_clips - 1,
        )
        local_frame = global_frame - self.reference_bank.time_offsets[clip_id]
        reference = self.observation_assembler.reference(
            self.reference_bank,
            clip_id,
            local_frame,
            anchor_quat,
            state.alignment_yaw,
        )
        gravity_key = reference_key = None
        if rng is not None and self.config.observation_noise.enabled:
            reference_key, gravity_key = jax.random.split(rng)
            magnitude = self.config.observation_noise.anchor_orientation
            reference = reference.reshape(self.observation_assembler.future_frames, 64)
            reference = reference.at[:, 58:64].add(
                jax.random.uniform(
                    reference_key,
                    (self.observation_assembler.future_frames, 6),
                    minval=-magnitude,
                    maxval=magnitude,
                )
            )
            reference = reference.reshape(-1)
        observation = self.observation_assembler.step(
            reference,
            root_quaternion=qpos[3:7],
            root_angular_velocity=qvel[3:6],
            logical_joint_position=(
                qpos[jnp.asarray(self.model.policy_qpos_indices)]
                - jnp.asarray(self.model.default_joint_target)
                - state.joint_default_offset
            ),
            logical_joint_velocity=qvel[jnp.asarray(self.model.policy_qvel_indices)],
            previous_action=state.last_action,
        )
        if gravity_key is not None:
            magnitude = self.config.observation_noise.anchor_orientation
            start = self.observation_assembler.reference_size
            observation = observation.at[start : start + 3].add(
                jax.random.uniform(
                    gravity_key,
                    (3,),
                    minval=-magnitude,
                    maxval=magnitude,
                )
            )
        return observation

    def critic_observation(self, state: MotionTrackingState) -> jax.Array:
        actor = self.observation(state, _feedback=self._actor_feedback(state.data))
        reference = actor[: self.observation_assembler.reference_size].reshape(
            self.observation_assembler.future_frames, 64
        )
        # The first 58 columns recover the release's original blockwise
        # [all positions, all velocities] command despite its non-flat reshape.
        reference_joints = reference[:, :58].reshape(-1)
        terms = self._observation_terms(state, self._actor_feedback(state.data))
        ids = jnp.asarray(self.tracked_body_ids)
        anchor_id = self.anchor_body_id
        anchor_position = state.data.xpos[anchor_id]
        anchor_quat = state.data.xquat[anchor_id]
        body_position = jax.vmap(
            lambda value: quat_rotate_inverse(anchor_quat, value - anchor_position)
        )(state.data.xpos[ids])
        body_orientation = jax.vmap(
            lambda value: rotation_6d(quat_multiply(quat_conjugate(anchor_quat), value))
        )(state.data.xquat[ids])
        return jnp.concatenate(
            (
                reference_joints,
                terms[2],
                terms[3],
                body_position.reshape(-1),
                body_orientation.reshape(-1),
                terms[4],
                terms[5],
                terms[6],
                terms[7],
                terms[8],
            )
        )

    def action_target(self, state: MotionTrackingState, action: jax.Array) -> jax.Array:
        return sonic_action_target(
            action,
            state.data.ctrl,
            state.joint_default_offset,
            model=self.model,
            actuator_config=self.actuator_config,
        )

    def _reward(self, previous, state, world_id=0):
        reward, components = super()._reward(previous, state, world_id)
        # Isaac Lab RewardManager integrates weighted reward rates over step_dt.
        return reward * self.control_timestep, {
            name: value * self.control_timestep for name, value in components.items()
        }

    @property
    def allowed_contact_body_names(self):
        return tuple(
            name
            for name in self.motion.body_names
            if any(part in name for part in ("ankle_roll", "wrist_yaw", "elbow"))
        )

    def _anchor_orientation_terminated(self, reference_quat, robot_quat):
        return (
            quat_distance(reference_quat, robot_quat) ** 2
            > self.config.done.root_orientation_error
        )

    def _foot_acceleration_cost(self, previous, state):
        # Isaac Lab updates joint acceleration at physics frequency, not 50 Hz.
        indices = jnp.asarray(
            [
                index
                for name, index in zip(
                    self.model.actuator_joint_names,
                    self.model.policy_qvel_indices,
                    strict=True,
                )
                if "ankle" in name
            ]
        )
        acceleration = (state.data.qvel[indices] - state.last_substep_qvel[indices]) / (
            self.control_timestep / self.decimation
        )
        return jnp.sum(acceleration**2)

    def initial_task_state(self):
        # Newest first: return, length, validity. Checkpointed with the runner.
        return jnp.zeros((3, 100), dtype=jnp.float32)

    def update_task_state(self, task_state, info):
        done = info["done/any"]
        count = jnp.minimum(jnp.sum(done), task_state.shape[1])
        newest = (
            done.size
            - 1
            - jnp.nonzero(done[::-1], size=task_state.shape[1], fill_value=done.size)[0]
        )
        completed = jnp.stack(
            (
                info["episode_return"][newest],
                info["episode_length"][newest],
                jnp.ones(task_state.shape[1]),
            )
        )
        positions = jnp.arange(task_state.shape[1])
        return jnp.where(
            positions[None, :] < count,
            completed,
            task_state[:, jnp.maximum(positions - count, 0)],
        )

    def task_state_metrics(self, task_state):
        count = jnp.sum(task_state[2])
        means = jnp.sum(task_state[:2] * task_state[2], axis=1) / jnp.maximum(count, 1)
        return {
            "objective/rewards": means[0],
            "objective/length": means[1],
            "objective/completed_episodes": count,
        }

    def _finish_termination_flags(self, state, height, orientation, end_effector):
        flags = super()._finish_termination_flags(
            state, height, orientation, end_effector
        )
        global_frame = jnp.floor(state.reference_time).astype(jnp.int32)
        clip_id = jnp.clip(
            jnp.searchsorted(
                self.reference_bank.time_offsets, global_frame, side="right"
            )
            - 1,
            0,
            self.reference_bank.num_clips - 1,
        )
        local_frame = global_frame - self.reference_bank.time_offsets[clip_id]
        timeout = local_frame >= self.reference_bank.lengths[clip_id] - 1
        return {
            **flags,
            "done/timeout": timeout,
            "done/any": flags["done/terminal"] | timeout,
        }
