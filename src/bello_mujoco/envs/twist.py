"""Standalone TWIST teacher/student task for Bello MJX."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
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
from bello_mujoco.motion.clip import (
    MotionClip,
    MotionSample,
    load_motion_clip,
    sample_motion,
)
from bello_mujoco.motion.twist_dataset import (
    TWIST_FUTURE_OFFSETS,
    PackedReferenceBank,
)
from bello_mujoco.robot import AssetName
from bello_mujoco.robot_tasks import motion_task_definition
from bello_mujoco.sim import (
    BelloModel,
    ModelRandomizationConfig,
    WarpConfig,
    load_bello_model,
)
from bello_mujoco.terrain import FLAT_TERRAIN, TerrainConfig

TwistStage = Literal["teacher", "student"]


@dataclass(frozen=True)
class TwistSamplingConfig:
    """Released per-clip sampling with adaptive hard-motion emphasis."""

    base_distribution: Literal[
        "manifest_weight", "frame_count_times_manifest_weight"
    ] = "manifest_weight"
    curriculum_enabled: bool = True
    curriculum_gamma: float = 0.01
    curriculum_min_difficulty: float = 1.0
    curriculum_max_difficulty: float = 9.0


TWIST_SIMULATION_HZ = 500
TWIST_POLICY_HZ = 50
TWIST_DECIMATION = TWIST_SIMULATION_HZ // TWIST_POLICY_HZ
TWIST_EPISODE_STEPS = 10 * TWIST_POLICY_HZ
TWIST_HISTORY_FRAMES = 11

DEFAULT_KEY_BODY_NAMES = motion_task_definition("full_body").twist_key_body_names


@dataclass(frozen=True)
class TwistObservationContract:
    joint_names: tuple[str, ...]
    key_body_names: tuple[str, ...] = DEFAULT_KEY_BODY_NAMES
    future_offsets: tuple[int, ...] = TWIST_FUTURE_OFFSETS
    history_frames: int = TWIST_HISTORY_FRAMES
    randomization_size: int = 55

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
    def student_reference_frame_size(self) -> int:
        # Height + roll/pitch/yaw + local linear velocity + yaw rate + joints.
        return 1 + 3 + 3 + 1 + self.action_size

    @property
    def teacher_reference_frame_size(self) -> int:
        return self.student_reference_frame_size + 9 * 3

    @property
    def proprioception_frame_size(self) -> int:
        # base angular velocity + roll/pitch + q + qdot + previous action.
        return 3 + 2 + 3 * self.action_size

    @property
    def teacher_observation_size(self) -> int:
        return self.teacher_core_size + self.privileged_information_size

    @property
    def teacher_core_size(self) -> int:
        return len(self.future_offsets) * self.teacher_reference_frame_size + (
            self.proprioception_frame_size
        )

    @property
    def privileged_information_size(self) -> int:
        return 3 + 1 + 9 * 3 + 2 + self.randomization_size

    @property
    def student_observation_size(self) -> int:
        return self.history_frames * (
            self.student_reference_frame_size + self.proprioception_frame_size
        )

    @property
    def critic_observation_size(self) -> int:
        return self.teacher_observation_size

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
                        len(self.future_offsets) * self.teacher_reference_frame_size,
                    ),
                    ("current_proprioception", self.proprioception_frame_size),
                    ("base_linear_velocity", 3),
                    ("root_height", 1),
                    ("current_key_body_positions", 27),
                    ("foot_contacts", 2),
                    ("domain_randomization", self.randomization_size),
                )
            )
        return ObservationLayout(
            (
                (
                    "reference_proprioception_history",
                    self.history_frames
                    * (
                        self.student_reference_frame_size
                        + self.proprioception_frame_size
                    ),
                ),
            )
        )

    @property
    def critic_layout(self) -> ObservationLayout:
        return ObservationLayout(
            (("teacher_privileged_observation", self.teacher_observation_size),)
        )


@dataclass(frozen=True)
class TwistObservationNoiseConfig:
    enabled: bool = True
    base_angular_velocity: float = 0.1
    roll_pitch: float = 0.1
    joint_position: float = 0.01
    joint_velocity: float = 0.1


@dataclass(frozen=True)
class TwistDomainRandomizationConfig:
    enabled: bool = True
    seed: int = 0
    added_base_mass: tuple[float, float] = (-3.0, 3.0)
    added_base_com: tuple[float, float] = (-0.05, 0.05)
    friction: tuple[float, float] = (0.1, 2.0)
    motor_strength: tuple[float, float] = (0.8, 1.2)
    gravity_fraction: tuple[float, float] = (-0.1, 0.1)
    root_push_interval_steps: int = 4 * TWIST_POLICY_HZ
    root_push_velocity_xy: float = 1.0
    end_effector_push_interval_steps: int = 2 * TWIST_POLICY_HZ
    end_effector_push_force: float = 20.0


@dataclass(frozen=True)
class TwistRewardConfig:
    tracking_joint_position: float = 0.6
    tracking_joint_velocity: float = 0.2
    tracking_root_pose: float = 0.6
    tracking_root_velocity: float = 1.0
    tracking_key_body_position: float = 2.0
    foot_slip: float = -0.1
    excessive_foot_contact_force: float = -5.0e-4
    foot_stumble: float = -1.25
    joint_position_limit: float = -5.0
    torque_limit: float = -1.0
    joint_velocity: float = -1.0e-4
    joint_acceleration: float = -5.0e-8
    action_rate: float = -0.01
    foot_air_time: float = 5.0
    base_angular_velocity_xy: float = -0.01
    ankle_joint_velocity: float = -2.0e-4
    ankle_joint_acceleration: float = -1.0e-7


@dataclass(frozen=True)
class TwistDoneConfig:
    root_height_error: float = 0.2
    roll_pitch_limit: float = 1.0
    key_body_position_error: float = 0.7
    root_velocity_limit: float = 5.0
    torso_contact_force: float = 1.0


@dataclass(frozen=True)
class TwistSimulationConfig:
    terrain: TerrainConfig = FLAT_TERRAIN


@dataclass(frozen=True)
class TwistEnvConfig:
    max_episode_steps: int = TWIST_EPISODE_STEPS
    action_clip: float = 5.0
    action_delay_steps: tuple[int, int] = (0, 1)
    action_history_steps: int = 8
    observation_noise: TwistObservationNoiseConfig = TwistObservationNoiseConfig()
    domain_randomization: TwistDomainRandomizationConfig = (
        TwistDomainRandomizationConfig()
    )
    reward: TwistRewardConfig = TwistRewardConfig()
    done: TwistDoneConfig = TwistDoneConfig()
    key_body_names: tuple[str, ...] = DEFAULT_KEY_BODY_NAMES

    def __post_init__(self) -> None:
        if self.max_episode_steps != TWIST_EPISODE_STEPS:
            raise ValueError("TWIST episodes must be exactly 10 seconds at 50 Hz")
        if not math.isfinite(self.action_clip) or self.action_clip <= 0.0:
            raise ValueError("TWIST action_clip must be positive and finite")
        low, high = self.action_delay_steps
        if low < 0 or high < low:
            raise ValueError(
                "action_delay_steps must be an increasing nonnegative range"
            )
        if self.action_delay_steps != (0, 1):
            raise ValueError("TWIST action delay must match the released 0-1 range")
        if self.action_history_steps != 8:
            raise ValueError("TWIST action history must contain exactly eight frames")
        if len(self.key_body_names) != 9:
            raise ValueError("TWIST requires exactly nine key bodies")
        for name in (
            "base_angular_velocity",
            "roll_pitch",
            "joint_position",
            "joint_velocity",
        ):
            value = getattr(self.observation_noise, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"observation_noise.{name} must be nonnegative")
        domain = self.domain_randomization
        ModelRandomizationConfig(
            enabled=domain.enabled,
            seed=domain.seed,
            added_base_mass_range=domain.added_base_mass,
            base_com_offset_range=domain.added_base_com,
            friction_range=domain.friction,
            actuator_strength_range=domain.motor_strength,
            gravity_fraction_range=domain.gravity_fraction,
        )
        for name in ("root_push_interval_steps", "end_effector_push_interval_steps"):
            if getattr(domain, name) <= 0:
                raise ValueError(f"domain_randomization.{name} must be positive")
        for name in ("root_push_velocity_xy", "end_effector_push_force"):
            value = getattr(domain, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"domain_randomization.{name} must be finite and nonnegative"
                )
        for field_name in self.reward.__dataclass_fields__:
            if not math.isfinite(getattr(self.reward, field_name)):
                raise ValueError(f"reward.{field_name} must be finite")
        for field_name in self.done.__dataclass_fields__:
            value = getattr(self.done, field_name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"done.{field_name} must be positive and finite")


@chex.dataclass(frozen=True)
class TwistMotionState:
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


@chex.dataclass(frozen=True)
class TwistState:
    motion_state: TwistMotionState
    observation_history: jax.Array
    action_history: jax.Array
    action_delay_steps: jax.Array
    clip_index: jax.Array
    domain_randomization: jax.Array
    foot_air_time: jax.Array
    last_foot_contact: jax.Array
    total_step: jax.Array
    root_push_phase: jax.Array
    gravity_fraction: jax.Array


@dataclass(frozen=True)
class BelloTwistEnv:
    """TWIST owns task semantics while reusing only generic model/motion storage."""

    model: BelloModel
    motion: MotionClip
    stage: TwistStage
    config: TwistEnvConfig = TwistEnvConfig()
    reference_bank: PackedReferenceBank | None = None
    sampling: TwistSamplingConfig = TwistSamplingConfig()

    def __post_init__(self) -> None:
        if self.stage not in {"teacher", "student"}:
            raise ValueError(f"invalid TWIST stage: {self.stage!r}")
        if self.model.mjx_model is None:
            raise ValueError("TWIST requires an MJX-enabled Bello model")
        if not np.array_equal(
            self.model.policy_ctrl_indices,
            np.arange(self.model.mj_model.nu, dtype=np.int32),
        ):
            raise ValueError(
                "TWIST requires every generated actuator to be policy-controlled"
            )
        if len(self.model.actuator_joint_names) != self.action_size:
            raise ValueError("TWIST actuator and action ordering sizes differ")
        if not math.isclose(
            TWIST_DECIMATION * float(self.model.mj_model.opt.timestep),
            1.0 / TWIST_POLICY_HZ,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("TWIST policy timestep must be 0.02 seconds")
        if self.motion.num_frames <= max(TWIST_FUTURE_OFFSETS):
            raise ValueError("TWIST motion is too short for its future horizon")
        missing = set(self.config.key_body_names) - set(self.motion.body_names)
        if missing:
            raise ValueError(f"TWIST key bodies missing from motion: {sorted(missing)}")
        if self.contract.randomization_size != 5 + 2 * self.action_size:
            raise ValueError("TWIST randomization contract has drifted")
        if self.reference_bank is not None:
            clip_count = len(self.reference_bank.sampling_probabilities)
            if self.reference_bank.lengths.shape != (clip_count,) or (
                self.reference_bank.offsets.shape != (clip_count,)
            ):
                raise ValueError("TWIST reference bank clip metadata has drifted")
            total_frames = int(self.reference_bank.qpos.shape[0])
            _validate_reference_bank_clip_bounds(
                self.reference_bank.offsets,
                self.reference_bank.lengths,
                self.reference_bank.sampling_probabilities,
                total_frames,
            )
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
            if tuple(self.reference_bank.qpos.shape[1:]) != (self.model.mj_model.nq,):
                raise ValueError("TWIST reference bank qpos shape differs from model")

    @classmethod
    def load(
        cls,
        stage: TwistStage,
        asset_config: AssetPrepConfig,
        simulation: TwistSimulationConfig,
        *,
        motion_path: Path,
        config: TwistEnvConfig = TwistEnvConfig(),
        warp: WarpConfig = WarpConfig(),
        batch_size: int = 1,
        reference_bank: PackedReferenceBank | None = None,
        sampling: TwistSamplingConfig = TwistSamplingConfig(),
    ) -> "BelloTwistEnv":
        domain = config.domain_randomization
        model = load_bello_model(
            asset_config,
            warp=warp,
            batch_size=batch_size,
            terrain=simulation.terrain,
            simulation_timestep=1.0 / TWIST_SIMULATION_HZ,
            model_randomization=ModelRandomizationConfig(
                enabled=domain.enabled,
                seed=domain.seed,
                added_base_mass_range=domain.added_base_mass,
                base_com_offset_range=domain.added_base_com,
                friction_range=domain.friction,
                actuator_strength_range=domain.motor_strength,
                gravity_fraction_range=domain.gravity_fraction,
            ),
        )
        motion = load_motion_clip(motion_path, model)
        return cls(model, motion, stage, config, reference_bank, sampling)

    @property
    def contract(self) -> TwistObservationContract:
        return TwistObservationContract(
            joint_names=self.model.actuator_joint_names,
            key_body_names=self.config.key_body_names,
            randomization_size=5 + 2 * self.action_size,
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
    def action_clip(self) -> float:
        return self.config.action_clip

    @property
    def control_timestep(self) -> float:
        return TWIST_DECIMATION * float(self.model.mj_model.opt.timestep)

    @property
    def ppo_stagger_initial_step_count(self) -> bool:
        return False

    def initial_task_state(self) -> jax.Array:
        """Return released TWIST's checkpointed per-motion difficulty state."""

        clip_count = (
            len(self.reference_bank.sampling_probabilities)
            if self.reference_bank is not None
            else 0
        )
        return jnp.full(
            (clip_count,),
            self.sampling.curriculum_max_difficulty,
            dtype=jnp.float32,
        )

    def with_task_state(self, difficulty: jax.Array) -> "BelloTwistEnv":
        """Bind curriculum sampling probabilities without host copies."""

        if self.reference_bank is None or not self.sampling.curriculum_enabled:
            return self
        if difficulty.shape != self.reference_bank.sampling_probabilities.shape:
            raise ValueError("TWIST motion difficulty shape differs from clip index")
        weighted = jnp.asarray(self.reference_bank.sampling_probabilities) * difficulty
        probabilities = weighted / jnp.sum(weighted)
        reference_bank = replace(
            self.reference_bank, sampling_probabilities=probabilities
        )
        return self._unchecked_runtime_copy(reference_bank=reference_bank)

    def update_task_state(
        self, difficulty: jax.Array, info: dict[str, jax.Array]
    ) -> jax.Array:
        """Apply the released completion-rate curriculum after one policy step."""

        if self.reference_bank is None or not self.sampling.curriculum_enabled:
            return difficulty
        clip_indices = info["reference/clip_index"].astype(jnp.int32)
        returned = info["returned_episode"].astype(jnp.float32)
        clip_lengths = jnp.asarray(self.reference_bank.lengths)[clip_indices]
        completion = info["episode_length"].astype(jnp.float32) / jnp.maximum(
            clip_lengths.astype(jnp.float32), 1.0
        )
        completion_sum = (
            jnp.zeros_like(difficulty).at[clip_indices].add(completion * returned)
        )
        completion_count = jnp.zeros_like(difficulty).at[clip_indices].add(returned)
        completion_rate = completion_sum / jnp.maximum(completion_count, 1.0)
        completion_rate = jnp.where(completion_count > 0.0, completion_rate, 0.7)
        factor = jnp.where(
            completion_rate <= 0.5,
            1.0 + self.sampling.curriculum_gamma,
            jnp.where(
                completion_rate >= 0.95,
                1.0 - self.sampling.curriculum_gamma,
                1.0,
            ),
        )
        return jnp.clip(
            difficulty * factor,
            self.sampling.curriculum_min_difficulty,
            self.sampling.curriculum_max_difficulty,
        )

    def task_state_metrics(self, difficulty: jax.Array) -> dict[str, jax.Array]:
        if difficulty.size == 0:
            return {}
        return {
            "curriculum/mean_difficulty": jnp.mean(difficulty),
            "curriculum/min_difficulty": jnp.min(difficulty),
            "curriculum/max_difficulty": jnp.max(difficulty),
        }

    def _unchecked_runtime_copy(
        self,
        *,
        reference_bank: PackedReferenceBank | None,
    ) -> "BelloTwistEnv":
        bound = object.__new__(BelloTwistEnv)
        for name in ("model", "motion", "stage", "config", "sampling"):
            object.__setattr__(bound, name, getattr(self, name))
        object.__setattr__(bound, "reference_bank", reference_bank)
        return bound

    @property
    def key_body_ids(self) -> np.ndarray:
        return np.asarray(
            [self.model.mj_model.body(name).id for name in self.config.key_body_names]
        )

    def reset(self, rng: jax.Array | None = None) -> tuple[TwistState, jax.Array]:
        state = self._reset_state(rng)
        obs_key = rng
        if rng is not None:
            obs_key = jax.random.split(rng, 4)[3]
        return state, self.observation(state, obs_key)

    def _reset_state(self, rng: jax.Array | None = None) -> TwistState:
        clip_key = pose_key = delay_key = obs_key = rng
        if rng is not None:
            clip_key, pose_key, delay_key, obs_key = jax.random.split(rng, 4)
        clip_index, reference_time = self._sample_clip_start(clip_key)
        motion_state = TwistMotionState(
            data=self.model.make_data(),
            last_action=jnp.zeros(self.action_size, dtype=jnp.float32),
            reference_time=reference_time,
            alignment_xy=jnp.zeros(2, dtype=jnp.float32),
            alignment_yaw=jnp.asarray(0.0, dtype=jnp.float32),
            step_count=jnp.asarray(0, dtype=jnp.int32),
            episode_return=jnp.asarray(0.0, dtype=jnp.float32),
            episode_length=jnp.asarray(0, dtype=jnp.int32),
            last_episode_return=jnp.asarray(0.0, dtype=jnp.float32),
            last_episode_length=jnp.asarray(0, dtype=jnp.int32),
        )
        source = self._aligned_reference(motion_state, clip_index, 0)
        desired_xy = jnp.zeros(2, dtype=jnp.float32)
        if pose_key is not None:
            desired_xy = jax.random.uniform(pose_key, (2,), minval=-0.3, maxval=0.3)
        alignment_xy = desired_xy - source.qpos[:2]
        motion_state = motion_state.replace(alignment_xy=alignment_xy)
        reference = self._aligned_reference(motion_state, clip_index, 0)
        qpos = self._reset_qpos(reference.qpos, pose_key)
        qvel = 0.8 * reference.qvel
        data = motion_state.data.replace(
            qpos=qpos,
            qvel=qvel,
            ctrl=self._action_target(jnp.zeros(self.action_size, dtype=jnp.float32)),
        )
        data = mjx.forward(self.model.mjx_model, data)
        motion_state = motion_state.replace(data=data)
        randomization = self._sample_domain_randomization()
        delay = self._sample_action_delay(delay_key)
        action_history = jnp.zeros(
            (self.config.action_history_steps, self.action_size),
            dtype=jnp.float32,
        )
        partial_state = TwistState(
            motion_state=motion_state,
            observation_history=jnp.zeros(
                (
                    TWIST_HISTORY_FRAMES,
                    self.contract.student_reference_frame_size
                    + self.contract.proprioception_frame_size,
                ),
                dtype=jnp.float32,
            ),
            action_history=action_history,
            action_delay_steps=delay,
            clip_index=clip_index,
            domain_randomization=randomization,
            foot_air_time=jnp.zeros(2, dtype=jnp.float32),
            last_foot_contact=jnp.zeros(2, dtype=bool),
            total_step=jnp.asarray(0, dtype=jnp.int32),
            root_push_phase=jnp.asarray(0, dtype=jnp.int32),
            gravity_fraction=jnp.asarray(self.model.gravity_fraction[0]),
        )
        state = partial_state
        if self.stage == "student":
            frame = self._student_frame(partial_state, obs_key)
            state = partial_state.replace(
                observation_history=jnp.repeat(
                    frame[None, :], TWIST_HISTORY_FRAMES, axis=0
                )
            )
        return state

    def _reset_qpos(
        self, reference_qpos: jax.Array, rng: jax.Array | None
    ) -> jax.Array:
        del rng
        return reference_qpos.at[2].add(0.05)

    def reset_batch(
        self, batch_size: int, rng: jax.Array | None = None
    ) -> tuple[TwistState, jax.Array]:
        if rng is None:
            state = jax.vmap(lambda _: self._reset_state())(jnp.arange(batch_size))
        else:
            state = jax.vmap(self._reset_state)(jax.random.split(rng, batch_size))
        data = state.motion_state.data.replace(
            _impl=state.motion_state.data._impl.replace(nworld=batch_size)
        )
        physical_randomization = jnp.asarray(self.model.physical_randomization)
        if physical_randomization.shape[0] == 1:
            physical_randomization = jnp.repeat(
                physical_randomization, batch_size, axis=0
            )
        state = state.replace(domain_randomization=physical_randomization)
        gravity_fraction = jnp.asarray(self.model.gravity_fraction)
        if gravity_fraction.shape[0] == 1:
            gravity_fraction = jnp.repeat(gravity_fraction, batch_size, axis=0)
        state = state.replace(gravity_fraction=gravity_fraction)
        state = state.replace(motion_state=state.motion_state.replace(data=data))
        return state, self.observation_batch(state, rng)

    def initial_step_counts(
        self, batch_size: int, rng: jax.Array | None = None
    ) -> jax.Array:
        del rng
        return jnp.zeros((batch_size,), dtype=jnp.int32)

    def with_initial_step_counts(
        self, state: TwistState, step_count: jax.Array
    ) -> TwistState:
        episode_length = jnp.copy(step_count)
        total_step = jnp.copy(step_count)
        return state.replace(
            motion_state=state.motion_state.replace(
                step_count=step_count, episode_length=episode_length
            ),
            total_step=total_step,
        )

    def step(
        self, state: TwistState, action: jax.Array, rng: jax.Array | None = None
    ) -> tuple[TwistState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        return self._step_reference_bank(state, action, rng)

    def step_batch(
        self,
        state: TwistState,
        action: jax.Array,
        rng: jax.Array | None = None,
    ) -> tuple[TwistState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        return self._step_reference_bank_batch(state, action, rng)

    def observation(self, state: TwistState, rng: jax.Array | None = None) -> jax.Array:
        if self.stage == "teacher":
            return self._teacher_observation(state, rng=rng)
        return state.observation_history.reshape(-1)

    def observation_batch(
        self, state: TwistState, rng: jax.Array | None = None
    ) -> jax.Array:
        if self.stage == "student":
            return state.observation_history.reshape(
                (state.observation_history.shape[0], -1)
            )
        if rng is None:
            core = jax.vmap(self._teacher_observation_core)(state)
        else:
            keys = jax.random.split(rng, state.action_delay_steps.shape[0])
            core = jax.vmap(self._teacher_observation_core)(state, keys)
        contacts = self._foot_contacts_batch(
            state.motion_state, state.action_delay_steps.shape[0]
        )
        return jnp.concatenate((core, contacts, state.domain_randomization), axis=-1)

    def critic_observation(self, state: TwistState) -> jax.Array:
        return self._teacher_observation(state)

    def critic_observation_batch(
        self, state: TwistState, rng: jax.Array | None = None
    ) -> jax.Array:
        del rng
        return jnp.concatenate(
            (
                jax.vmap(self._teacher_observation_core)(state),
                self._foot_contacts_batch(
                    state.motion_state, state.action_delay_steps.shape[0]
                ),
                state.domain_randomization,
            ),
            axis=-1,
        )

    def critic_observation_with_world(
        self, state: TwistState, world_id: jax.Array
    ) -> jax.Array:
        return self._teacher_observation(state, world_id)

    def ppo_action_diagnostics(self, env_action: jax.Array) -> dict[str, jax.Array]:
        flat_action = env_action.reshape((-1, self.action_size))
        targets = jax.vmap(self._action_target)(flat_action)
        policy = jnp.asarray(self.model.policy_ctrl_indices)
        qpos_targets = jax.vmap(
            lambda ctrl: jnp.linalg.solve(
                jnp.asarray(self.model.actuator_coordinate_transform), ctrl
            )[policy]
        )(targets)
        low = jnp.asarray(self.model.joint_target_low)[policy]
        high = jnp.asarray(self.model.joint_target_high)[policy]
        return {
            "target/outside_joint_limit_fraction": jnp.mean(
                ((qpos_targets < low) | (qpos_targets > high)).astype(jnp.float32)
            )
        }

    def _teacher_observation(
        self,
        state: TwistState,
        world_id: jax.Array | int = 0,
        rng: jax.Array | None = None,
    ) -> jax.Array:
        return jnp.concatenate(
            (
                self._teacher_observation_core(state, rng),
                self._foot_contacts(state.motion_state, world_id),
                state.domain_randomization,
            )
        )

    def _teacher_observation_core(
        self, state: TwistState, rng: jax.Array | None = None
    ) -> jax.Array:
        future = jnp.concatenate(
            tuple(
                self._teacher_reference_frame(state, offset)
                for offset in TWIST_FUTURE_OFFSETS
            )
        )
        motion_state = state.motion_state
        root_quat = motion_state.data.qpos[3:7]
        base_linear_velocity = quat_rotate_inverse(
            root_quat, motion_state.data.qvel[:3]
        )
        return jnp.concatenate(
            (
                future,
                self._proprioception(state, rng),
                base_linear_velocity,
                motion_state.data.qpos[2:3],
                self._current_key_body_positions(motion_state),
            )
        )

    def _student_reference_frame(self, state: TwistState, offset: int) -> jax.Array:
        motion_state = state.motion_state
        reference = self._aligned_reference(motion_state, state.clip_index, offset)
        root_quat = reference.qpos[3:7]
        local_velocity = quat_rotate_inverse(root_quat, reference.qvel[:3])
        return jnp.concatenate(
            (
                reference.qpos[2:3],
                _roll_pitch_yaw(root_quat),
                local_velocity,
                reference.qvel[5:6],
                reference.qpos[jnp.asarray(self.model.policy_qpos_indices)],
            )
        )

    def _teacher_reference_frame(self, state: TwistState, offset: int) -> jax.Array:
        motion_state = state.motion_state
        reference = self._aligned_reference(motion_state, state.clip_index, offset)
        root_quat = reference.qpos[3:7]
        key_indices = jnp.asarray(
            [self.motion.body_names.index(name) for name in self.config.key_body_names]
        )
        key_positions = reference.body_pos[key_indices]
        root_position = reference.qpos[:3]
        local_positions = jax.vmap(
            lambda value: quat_rotate_inverse(
                yaw_quat_from_angle(_roll_pitch_yaw(root_quat)[2]),
                value - root_position,
            )
        )(key_positions)
        return jnp.concatenate(
            (
                self._student_reference_frame(state, offset),
                local_positions.reshape(-1),
            )
        )

    def _student_frame(
        self, state: TwistState, rng: jax.Array | None = None
    ) -> jax.Array:
        return jnp.concatenate(
            (self._student_reference_frame(state, 0), self._proprioception(state, rng))
        )

    def _step_reference_bank(
        self, state: TwistState, action: jax.Array, rng: jax.Array | None
    ):
        action = jnp.asarray(action)
        action_history = jnp.concatenate(
            (state.action_history[1:], action[None, :]), axis=0
        )
        applied = jnp.clip(
            jax.lax.dynamic_index_in_dim(
                action_history,
                action_history.shape[0] - 1 - state.action_delay_steps,
                keepdims=False,
            ),
            -self.action_clip,
            self.action_clip,
        )
        data = state.motion_state.data.replace(
            ctrl=self._action_target(
                applied, state.motion_state, state.domain_randomization
            )
        )
        data = jax.lax.fori_loop(
            0,
            TWIST_DECIMATION,
            lambda _, value: mjx.step(self.model.mjx_model, value),
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
        contacts = self._foot_contacts(motion_state, 0).astype(bool)
        foot_air_time = state.foot_air_time + self.control_timestep
        reward, reward_info = self._twist_reward(
            state, motion_state, reference, contacts, foot_air_time, 0
        )
        done_flags = self._bank_termination_flags(
            motion_state, state.clip_index, reference, 0
        )
        reward, reward_info = self._sanitize_reward(reward, reward_info, done_flags)
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
        stepped = TwistState(
            motion_state=motion_state,
            observation_history=state.observation_history,
            action_history=action_history,
            action_delay_steps=state.action_delay_steps,
            clip_index=state.clip_index,
            domain_randomization=state.domain_randomization,
            foot_air_time=jnp.where(contacts, 0.0, foot_air_time),
            last_foot_contact=contacts,
            total_step=state.total_step + 1,
            root_push_phase=state.root_push_phase,
            gravity_fraction=state.gravity_fraction,
        )
        if self.stage == "student":
            frame = self._student_frame(stepped, rng)
            stored_history, observation_history = self._student_history_transition(
                state, stepped, frame
            )
            stepped = stepped.replace(observation_history=stored_history)
        else:
            observation_history = stepped.observation_history
        observation = self.observation(
            stepped.replace(observation_history=observation_history), rng
        )
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
        action = jnp.asarray(action)
        delay_key = push_key = reset_key = obs_key = rng
        if rng is not None:
            delay_key, push_key, reset_key, obs_key = jax.random.split(rng, 4)
        delay = self._sample_action_delay_batch(state, delay_key)
        action_history = jnp.concatenate(
            (state.action_history[:, 1:], action[:, None, :]), axis=1
        )
        indices = jnp.full(
            (batch_size,), action_history.shape[1] - 1 - delay, dtype=jnp.int32
        )
        applied = jax.vmap(
            lambda values, index: jax.lax.dynamic_index_in_dim(
                values, index, keepdims=False
            )
        )(action_history, indices)
        applied = jnp.clip(applied, -self.action_clip, self.action_clip)
        ctrl = jax.vmap(self._action_target)(
            applied, state.motion_state, state.domain_randomization
        )
        data = state.motion_state.data.replace(
            ctrl=ctrl,
            _impl=state.motion_state.data._impl.replace(nworld=batch_size),
        )
        data, push_applied = self._apply_disturbances_batch(data, state, push_key)
        data = jax.lax.fori_loop(
            0,
            TWIST_DECIMATION,
            lambda _, value: mjx.step(self.model.mjx_model, value),
            data,
        )
        data = self._apply_post_step_disturbances_batch(data, state, push_key)
        motion_state = state.motion_state.replace(
            data=data,
            last_action=applied,
            reference_time=(state.motion_state.reference_time + self.control_timestep),
            step_count=state.motion_state.step_count + 1,
            episode_length=state.motion_state.episode_length + 1,
        )
        contact_normal, contact_tangent = self._foot_contact_forces_batch(
            motion_state, batch_size
        )
        contacts = (contact_normal > 5.0).astype(bool)
        foot_air_time = state.foot_air_time + self.control_timestep

        def reward_one(
            previous,
            current,
            clip_index,
            contact,
            air_time,
            twist_state,
            normal_force,
            tangent_force,
        ):
            reference = self._aligned_reference(current, clip_index, 0)
            return self._twist_reward(
                twist_state,
                current,
                reference,
                contact,
                air_time,
                0,
                contact_normal=normal_force,
                contact_tangent=tangent_force,
            )

        reward, reward_info = jax.vmap(reward_one)(
            state.motion_state,
            motion_state,
            state.clip_index,
            contacts,
            foot_air_time,
            state,
            contact_normal,
            contact_tangent,
        )
        references = jax.vmap(
            lambda current, clip: self._aligned_reference(current, clip, 0)
        )(motion_state, state.clip_index)
        torso_contacts = self._termination_torso_contacts_batch(
            motion_state, batch_size
        )
        done_flags = jax.vmap(
            lambda current, clip, reference, torso_contact: (
                self._bank_termination_flags(
                    current,
                    clip,
                    reference,
                    0,
                    torso_contact=torso_contact,
                )
            )
        )(
            motion_state,
            state.clip_index,
            references,
            torso_contacts,
        )
        reward, reward_info = self._sanitize_reward(reward, reward_info, done_flags)
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
        stepped = TwistState(
            motion_state=motion_state,
            observation_history=state.observation_history,
            action_history=action_history,
            action_delay_steps=jnp.full_like(state.action_delay_steps, delay),
            clip_index=state.clip_index,
            domain_randomization=state.domain_randomization,
            foot_air_time=jnp.where(contacts, 0.0, foot_air_time),
            last_foot_contact=contacts,
            total_step=state.total_step + 1,
            root_push_phase=state.root_push_phase,
            gravity_fraction=state.gravity_fraction,
        )
        if self.stage == "student":
            if obs_key is None:
                frames = jax.vmap(self._student_frame)(stepped)
            else:
                keys = jax.random.split(obs_key, batch_size)
                frames = jax.vmap(self._student_frame)(stepped, keys)
            stored_history, observation_history = (
                self._student_history_transition_batch(state, stepped, frames)
            )
            stepped = stepped.replace(observation_history=stored_history)
        else:
            observation_history = stepped.observation_history
        stepped_observation = self.observation_batch(
            stepped.replace(observation_history=observation_history), obs_key
        )
        info = self._bank_info(stepped, stepped_observation, reward_info, done_flags)
        info["push/applied"] = push_applied.astype(jnp.float32)
        reset_state, reset_observation = self.reset_batch(batch_size, reset_key)
        reset_state = reset_state.replace(
            motion_state=reset_state.motion_state.replace(
                last_episode_return=motion_state.last_episode_return,
                last_episode_length=motion_state.last_episode_length,
            ),
            total_step=stepped.total_step,
        )
        next_state = tree_where(done, reset_state, stepped)
        next_observation = jnp.where(
            done[:, None], reset_observation, stepped_observation
        )
        return next_state, next_observation, reward, done, info

    def _action_target(
        self,
        action: jax.Array,
        motion_state=None,
        randomization: jax.Array | None = None,
    ) -> jax.Array:
        """Map the released unbounded Gaussian action to Bello PD targets."""

        policy = jnp.asarray(self.model.policy_ctrl_indices)
        logical = jnp.asarray(self.model.default_joint_target)
        target = logical[policy] + 0.5 * jnp.asarray(action, dtype=jnp.float32)
        del motion_state, randomization
        logical = logical.at[policy].set(target)
        return jnp.matmul(
            jnp.asarray(self.model.actuator_coordinate_transform),
            logical,
            precision=jax.lax.Precision.HIGHEST,
        )

    def _twist_reward(
        self,
        previous: TwistState,
        current,
        reference: MotionSample,
        foot_contact: jax.Array,
        foot_air_time: jax.Array,
        world_id: jax.Array | int,
        *,
        contact_normal: jax.Array | None = None,
        contact_tangent: jax.Array | None = None,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        policy_qpos = jnp.asarray(self.model.policy_qpos_indices)
        policy_qvel = jnp.asarray(self.model.policy_qvel_indices)
        qpos_error = reference.qpos[policy_qpos] - current.data.qpos[policy_qpos]
        qvel_error = reference.qvel[policy_qvel] - current.data.qvel[policy_qvel]
        weights = self._joint_tracking_weights
        root_position_error = reference.qpos[2] - current.data.qpos[2]
        root_orientation_error = _quat_distance(
            current.data.qpos[3:7], reference.qpos[3:7]
        )
        reference_linear = quat_rotate_inverse(reference.qpos[3:7], reference.qvel[:3])
        current_linear = quat_rotate_inverse(
            current.data.qpos[3:7], current.data.qvel[:3]
        )
        reference_angular = reference.qvel[3:6]
        current_angular = current.data.qvel[3:6]
        key_indices = jnp.asarray(
            [self.motion.body_names.index(name) for name in self.config.key_body_names]
        )
        target_key = reference.body_pos[key_indices] - reference.qpos[:3]
        current_key = (
            current.data.xpos[jnp.asarray(self.key_body_ids)] - current.data.qpos[:3]
        )
        target_yaw = yaw_quat_from_angle(_roll_pitch_yaw(reference.qpos[3:7])[2])
        current_yaw = yaw_quat_from_angle(_roll_pitch_yaw(current.data.qpos[3:7])[2])
        target_key = jax.vmap(lambda x: quat_rotate_inverse(target_yaw, x))(target_key)
        current_key = jax.vmap(lambda x: quat_rotate_inverse(current_yaw, x))(
            current_key
        )
        if contact_normal is None or contact_tangent is None:
            contact_normal, contact_tangent = self._foot_contact_forces(
                current, world_id
            )
        foot_velocity = self._body_linear_velocity(
            current, jnp.asarray(self.model.foot_body_ids)
        )
        torque = jnp.abs(
            current.data.actuator_force[jnp.asarray(self.model.policy_ctrl_indices)]
        )
        torque_limit = jnp.asarray(self.model.actuator_peak_force)[
            jnp.asarray(self.model.policy_ctrl_indices)
        ]
        lower = jnp.asarray(self.model.joint_target_low)[
            jnp.asarray(self.model.policy_ctrl_indices)
        ]
        upper = jnp.asarray(self.model.joint_target_high)[
            jnp.asarray(self.model.policy_ctrl_indices)
        ]
        positions = current.data.qpos[policy_qpos]
        velocities = current.data.qvel[policy_qvel]
        previous_velocities = previous.motion_state.data.qvel[policy_qvel]
        acceleration = (velocities - previous_velocities) / self.control_timestep
        first_contact = (foot_air_time > 0.0) & (
            foot_contact | previous.last_foot_contact
        )
        moving = jnp.linalg.norm(reference.qvel[:2]) > 0.05
        air_time_reward = jnp.sum(
            jnp.minimum(foot_air_time - 0.5, 0.0) * first_contact.astype(jnp.float32)
        ) * moving.astype(jnp.float32)
        ankle = jnp.asarray(self._ankle_policy_indices)
        cfg = self.config.reward
        dt = self.control_timestep
        raw = {
            "tracking_joint_position": jnp.exp(
                -0.15 * jnp.sum(weights * qpos_error**2)
            ),
            "tracking_joint_velocity": jnp.exp(
                -0.01 * jnp.sum(weights * qvel_error**2)
            ),
            "tracking_root_pose": jnp.exp(
                -5.0 * (root_position_error**2 + 0.1 * root_orientation_error**2)
            ),
            "tracking_root_velocity": jnp.exp(
                -(
                    jnp.sum((reference_linear - current_linear) ** 2)
                    + 0.5 * jnp.sum((reference_angular - current_angular) ** 2)
                )
            ),
            "tracking_key_body_position": jnp.exp(
                -10.0 * jnp.sum((current_key - target_key) ** 2)
            ),
            "foot_slip": jnp.sum(
                jnp.sqrt(jnp.linalg.norm(foot_velocity[:, :2], axis=-1))
                * foot_contact.astype(jnp.float32)
            ),
            "excessive_foot_contact_force": jnp.sum(
                jax.nn.relu(contact_normal - 100.0)
            ),
            "foot_stumble": jnp.any(contact_tangent > 4.0 * contact_normal).astype(
                jnp.float32
            ),
            "joint_position_limit": jnp.sum(
                jax.nn.relu(lower - positions) + jax.nn.relu(positions - upper)
            ),
            "torque_limit": jnp.sum(
                jax.nn.relu(torque / jnp.maximum(torque_limit, 1.0e-6) - 0.95)
            ),
            "joint_velocity": jnp.sum(velocities**2),
            "joint_acceleration": jnp.sum(acceleration**2),
            "action_rate": jnp.linalg.norm(
                current.last_action - previous.motion_state.last_action
            ),
            "foot_air_time": air_time_reward,
            "base_angular_velocity_xy": jnp.sum(current.data.qvel[3:5] ** 2),
            "ankle_joint_velocity": jnp.sum(velocities[ankle] ** 2),
            "ankle_joint_acceleration": jnp.sum(acceleration[ankle] ** 2),
        }
        components = {
            f"reward/{name}": dt * getattr(cfg, name) * value
            for name, value in raw.items()
        }
        return sum(components.values(), jnp.asarray(0.0, dtype=jnp.float32)), components

    def _sanitize_reward(self, reward, components, done_flags):
        invalid = done_flags["done/nonfinite"] | ~jnp.isfinite(reward)
        sanitized = {
            name: jnp.where(invalid, 0.0, jnp.nan_to_num(value))
            for name, value in components.items()
        }
        return sum(sanitized.values(), jnp.zeros_like(reward)), sanitized

    @property
    def _joint_tracking_weights(self) -> jax.Array:
        values = []
        for name in self.action_layout.names:
            if "knee" in name or "hip_pitch" in name:
                values.append(1.0)
            elif "ankle" in name:
                values.append(0.5)
            elif "waist" in name:
                values.append(0.6)
            elif "elbow_pitch" in name or "wrist" in name:
                values.append(1.0)
            else:
                values.append(0.8)
        return jnp.asarray(values, dtype=jnp.float32)

    @property
    def _ankle_policy_indices(self) -> np.ndarray:
        return np.asarray(
            [i for i, name in enumerate(self.action_layout.names) if "ankle" in name],
            dtype=np.int32,
        )

    def _bank_info(
        self,
        state: TwistState,
        observation: jax.Array,
        reward_info: dict[str, jax.Array],
        done_flags: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        done = done_flags["done/any"]
        info = {
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
        nacon = jnp.ravel(jnp.asarray(state.motion_state.data._impl.nacon))[0]
        capacity = self.model.warp_naconmax
        if capacity is not None:
            batch_size = observation.shape[0] if observation.ndim == 2 else 1
            capacity_fraction = nacon.astype(jnp.float32) / float(capacity)
            info["physics/contact_count_per_env"] = jnp.broadcast_to(
                nacon.astype(jnp.float32) / float(batch_size), done.shape
            )
            info["physics/contact_capacity_fraction"] = jnp.broadcast_to(
                capacity_fraction, done.shape
            )
            info["physics/contact_capacity_overflow"] = jnp.broadcast_to(
                nacon > capacity, done.shape
            )
        return info

    def _bank_termination_flags(
        self,
        state,
        clip_index: jax.Array,
        reference: MotionSample,
        world_id: jax.Array | int,
        *,
        torso_contact: jax.Array | None = None,
    ) -> dict[str, jax.Array]:
        cfg = self.config.done
        height = jnp.abs(state.data.qpos[2] - reference.qpos[2]) > cfg.root_height_error
        roll_pitch = _roll_pitch(state.data.qpos[3:7])
        orientation = jnp.any(jnp.abs(roll_pitch) > cfg.roll_pitch_limit)
        key_indices = jnp.asarray(
            [self.motion.body_names.index(name) for name in self.config.key_body_names]
        )
        pose = _key_body_pose_error_exceeds(
            state.data.xpos[jnp.asarray(self.key_body_ids)],
            state.data.qpos[:3],
            state.data.qpos[3:7],
            reference.body_pos[key_indices],
            reference.qpos[:3],
            reference.qpos[3:7],
            cfg.key_body_position_error,
        )
        root_velocity = jnp.linalg.norm(state.data.qvel[:3]) > cfg.root_velocity_limit
        if torso_contact is None:
            torso_contact = self._body_has_forceful_contact(
                state, self._body_id("torso_link"), cfg.torso_contact_force, world_id
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
        last_frame = (
            jnp.asarray(self.reference_bank.lengths)[clip_index] - 1
            if self.reference_bank is not None
            else jnp.asarray(self.motion.num_frames - 1, dtype=jnp.int32)
        )
        timeout = (state.step_count >= self.config.max_episode_steps) | (
            frame >= last_frame
        )
        terminal = (
            height | orientation | pose | root_velocity | torso_contact | nonfinite
        )
        return {
            "done/height": height,
            "done/orientation": orientation,
            "done/pose": pose,
            "done/root_velocity": root_velocity,
            "done/torso_contact": torso_contact,
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
        maximum_start = jnp.maximum(length - 1, 0)
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
                self.motion.duration,
            )
            return self._aligned_motion_sample(
                time, state.alignment_xy, state.alignment_yaw
            )
        frame = (
            jnp.rint(state.reference_time * TWIST_POLICY_HZ).astype(jnp.int32) + offset
        )
        length = jnp.asarray(self.reference_bank.lengths)[clip_index]
        frame = jnp.mod(frame, length)
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
            .set(_quat_multiply(rotation, qpos[3:7])),
            qvel=qvel.at[:3].set(quat_rotate(rotation, qvel[:3])),
            body_pos=jax.vmap(lambda item: quat_rotate(rotation, item))(body_pos)
            .at[:, :2]
            .add(state.alignment_xy),
            body_quat=jax.vmap(lambda item: _quat_multiply(rotation, item))(body_quat),
            body_lin_vel=jax.vmap(lambda item: quat_rotate(rotation, item))(
                body_lin_vel
            ),
            body_ang_vel=jax.vmap(lambda item: quat_rotate(rotation, item))(
                body_ang_vel
            ),
            foot_contacts=value("foot_contacts"),
        )

    def _aligned_motion_sample(
        self, time: jax.Array, alignment_xy: jax.Array, alignment_yaw: jax.Array
    ) -> MotionSample:
        source = sample_motion(self.motion, time)
        rotation = yaw_quat_from_angle(alignment_yaw)
        root_position = quat_rotate(rotation, source.qpos[:3]).at[:2].add(alignment_xy)
        return MotionSample(
            qpos=source.qpos.at[:3]
            .set(root_position)
            .at[3:7]
            .set(_quat_multiply(rotation, source.qpos[3:7])),
            qvel=source.qvel.at[:3].set(quat_rotate(rotation, source.qvel[:3])),
            body_pos=jax.vmap(lambda item: quat_rotate(rotation, item))(source.body_pos)
            .at[:, :2]
            .add(alignment_xy),
            body_quat=jax.vmap(lambda item: _quat_multiply(rotation, item))(
                source.body_quat
            ),
            body_lin_vel=jax.vmap(lambda item: quat_rotate(rotation, item))(
                source.body_lin_vel
            ),
            body_ang_vel=jax.vmap(lambda item: quat_rotate(rotation, item))(
                source.body_ang_vel
            ),
            foot_contacts=source.foot_contacts,
        )

    def _proprioception(
        self, state: TwistState, rng: jax.Array | None = None
    ) -> jax.Array:
        motion_state = state.motion_state
        root_quat = motion_state.data.qpos[3:7]
        base_angular_velocity = 0.25 * motion_state.data.qvel[3:6]
        roll_pitch = _roll_pitch(root_quat)
        qpos = (
            motion_state.data.qpos[jnp.asarray(self.model.policy_qpos_indices)]
            - jnp.asarray(self.model.default_joint_target)[
                jnp.asarray(self.model.policy_ctrl_indices)
            ]
        )
        qvel = (
            0.05 * motion_state.data.qvel[jnp.asarray(self.model.policy_qvel_indices)]
        )
        if rng is not None and self.config.observation_noise.enabled:
            keys = jax.random.split(rng, 4)
            cfg = self.config.observation_noise
            ramp = jnp.minimum(
                state.total_step.astype(jnp.float32) / float(3000 * 24), 1.0
            )
            base_angular_velocity = _uniform_noise(
                base_angular_velocity, ramp * cfg.base_angular_velocity, keys[0]
            )
            roll_pitch = _uniform_noise(roll_pitch, ramp * cfg.roll_pitch, keys[1])
            qpos = _uniform_noise(qpos, ramp * cfg.joint_position, keys[2])
            qvel = _uniform_noise(qvel, ramp * cfg.joint_velocity, keys[3])
        return jnp.concatenate(
            (base_angular_velocity, roll_pitch, qpos, qvel, state.action_history[-1])
        )

    def _current_key_body_positions(self, state) -> jax.Array:
        root_quat = yaw_quat_from_angle(_roll_pitch_yaw(state.data.qpos[3:7])[2])
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

    def _sample_action_delay_batch(
        self, state: TwistState, rng: jax.Array | None
    ) -> jax.Array:
        del state
        return self._sample_action_delay(rng)

    def _student_history_transition(
        self,
        previous: TwistState,
        current: TwistState,
        frame: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        del current
        history = jnp.concatenate(
            (previous.observation_history[1:], frame[None, :]), axis=0
        )
        return history, history

    def _student_history_transition_batch(
        self,
        previous: TwistState,
        current: TwistState,
        frames: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        del current
        history = jnp.concatenate(
            (previous.observation_history[:, 1:], frames[:, None, :]), axis=1
        )
        return history, history

    def _sample_domain_randomization(self) -> jax.Array:
        return jnp.asarray(self.model.physical_randomization[0])

    def _apply_disturbances_batch(self, data, state: TwistState, rng):
        batch_size = int(state.motion_state.step_count.shape[0])
        cfg = self.config.domain_randomization
        applied = jnp.zeros((batch_size,), dtype=bool)
        if rng is None or not cfg.enabled:
            return data.replace(xfrc_applied=jnp.zeros_like(data.xfrc_applied)), applied
        root_key, force_key = jax.random.split(rng)
        root_due = (state.motion_state.step_count > 0) & (
            state.motion_state.step_count % cfg.root_push_interval_steps == 0
        )
        root_velocity = jax.random.uniform(
            root_key,
            (batch_size, 2),
            minval=-cfg.root_push_velocity_xy,
            maxval=cfg.root_push_velocity_xy,
        )
        qvel = data.qvel.at[:, :2].set(
            jnp.where(root_due[:, None], root_velocity, data.qvel[:, :2])
        )
        force_due = (state.motion_state.step_count > 0) & (
            state.motion_state.step_count % cfg.end_effector_push_interval_steps == 0
        )
        force = jax.random.uniform(
            force_key,
            (batch_size, 3),
            minval=-cfg.end_effector_push_force,
            maxval=cfg.end_effector_push_force,
        )
        wrist_ids = jnp.asarray(
            [self._body_id("l_wrist_link"), self._body_id("r_wrist_link")]
        )
        xfrc = jnp.zeros_like(data.xfrc_applied)
        xfrc = xfrc.at[:, wrist_ids, :3].set(
            jnp.where(force_due[:, None, None], force[:, None, :], 0.0)
        )
        body_mass = jnp.asarray(self.model.mjx_model.body_mass)
        total_mass = jnp.sum(body_mass, axis=-1)
        gravity_force = total_mass * state.gravity_fraction * -9.81
        root_body = self._body_id("pelvis_link")
        wrench_force = jnp.stack(
            (
                jnp.zeros_like(gravity_force),
                jnp.zeros_like(gravity_force),
                gravity_force,
            ),
            axis=-1,
        )
        xfrc = xfrc.at[:, root_body, :3].add(wrench_force)
        return data.replace(qvel=qvel, xfrc_applied=xfrc), root_due | force_due

    def _foot_contact_forces(self, state, world_id):
        contact_world, active, bodies, normal, tangent = self._contact_force_data(state)
        selected = active & (contact_world == jnp.asarray(world_id))
        membership = (
            pair_contains_each(bodies, self.model.foot_body_ids) & selected[:, None]
        )
        return (
            jnp.max(jnp.where(membership, normal[:, None], 0.0), axis=0),
            jnp.max(jnp.where(membership, tangent[:, None], 0.0), axis=0),
        )

    def _foot_contact_forces_batch(self, state, batch_size: int):
        contact_world, active, bodies, normal, tangent = self._contact_force_data(state)
        valid_world = active & (contact_world >= 0) & (contact_world < batch_size)
        safe_world = jnp.clip(contact_world, 0, batch_size - 1)
        membership = pair_contains_each(bodies, self.model.foot_body_ids)
        normal_values = jnp.where(
            membership & valid_world[:, None], normal[:, None], 0.0
        )
        tangent_values = jnp.where(
            membership & valid_world[:, None], tangent[:, None], 0.0
        )
        return (
            _batched_contact_max(safe_world, normal_values, batch_size),
            _batched_contact_max(safe_world, tangent_values, batch_size),
        )

    def _contact_force_data(self, state):
        impl = state.data._impl
        geom = jnp.asarray(impl.contact__geom, dtype=jnp.int32)
        geom = jnp.clip(geom, 0, self.model.mj_model.ngeom - 1)
        active = jnp.arange(geom.shape[0]) < jnp.ravel(jnp.asarray(impl.nacon))[0]
        contact_world = jnp.asarray(impl.contact__worldid, dtype=jnp.int32)
        address = jnp.asarray(impl.contact__efc_address, dtype=jnp.int32)[:, 0]
        forces = jnp.asarray(impl.efc__force, dtype=jnp.float32)
        if forces.ndim == 1:
            forces = forces[None, :]
        force_world = jnp.clip(contact_world, 0, forces.shape[0] - 1)
        force_address = jnp.clip(address, 0, forces.shape[1] - 1)
        normal = jnp.where(
            (contact_world >= 0) & (address >= 0),
            jnp.abs(forces[force_world, force_address]),
            0.0,
        )
        tangent1 = jnp.where(
            (contact_world >= 0) & (address >= 0),
            jnp.abs(forces[force_world, jnp.clip(address + 1, 0, forces.shape[1] - 1)]),
            0.0,
        )
        tangent2 = jnp.where(
            (contact_world >= 0) & (address >= 0),
            jnp.abs(forces[force_world, jnp.clip(address + 2, 0, forces.shape[1] - 1)]),
            0.0,
        )
        bodies = jnp.asarray(self.model.geom_body_ids)[geom]
        tangent = jnp.sqrt(tangent1**2 + tangent2**2)
        return contact_world, active & (address >= 0), bodies, normal, tangent

    def _body_has_forceful_contact(self, state, body_id, threshold, world_id):
        contact_world, active, bodies, normal, _ = self._contact_force_data(state)
        return jnp.any(
            active
            & (contact_world == jnp.asarray(world_id))
            & (normal > threshold)
            & jnp.any(bodies == body_id, axis=-1)
        )

    def _body_has_forceful_contact_batch(
        self, state, body_id: int, threshold: float, batch_size: int
    ) -> jax.Array:
        contact_world, active, bodies, normal, _ = self._contact_force_data(state)
        valid_world = active & (contact_world >= 0) & (contact_world < batch_size)
        safe_world = jnp.clip(contact_world, 0, batch_size - 1)
        values = jnp.where(
            valid_world & jnp.any(bodies == body_id, axis=-1), normal, 0.0
        )
        maximum = _batched_contact_max(safe_world, values, batch_size)
        return maximum > threshold

    def _termination_torso_contacts_batch(self, state, batch_size: int) -> jax.Array:
        return self._body_has_forceful_contact_batch(
            state,
            self._body_id("torso_link"),
            self.config.done.torso_contact_force,
            batch_size,
        )

    def _apply_post_step_disturbances_batch(self, data, state, rng):
        del state, rng
        return data

    def _body_linear_velocity(self, state, body_ids):
        cvel = state.data.cvel[body_ids]
        root_body_ids = jnp.asarray(self.model.mj_model.body_rootid)[body_ids]
        offset = state.data.xpos[body_ids] - state.data.subtree_com[root_body_ids]
        return cvel[:, 3:6] - jnp.cross(offset, cvel[:, :3])

    def _body_id(self, name: str) -> int:
        return self.model.mj_model.body(name).id

    def _foot_contacts(self, state, world_id: jax.Array | int) -> jax.Array:
        normal, _ = self._foot_contact_forces(state, world_id)
        return (normal > 5.0).astype(jnp.float32)

    def _foot_contacts_batch(self, state, batch_size: int) -> jax.Array:
        normal, _ = self._foot_contact_forces_batch(state, batch_size)
        return (normal > 5.0).astype(jnp.float32)


def teacher_observation_from_student_critic(
    critic_observation: jax.Array, contract: TwistObservationContract
) -> jax.Array:
    """Return the exact teacher actor input stored in a student critic input."""

    return critic_observation[..., : contract.teacher_observation_size]


def _validate_reference_bank_clip_bounds(
    offsets: np.ndarray,
    lengths: np.ndarray,
    probabilities: np.ndarray,
    total_frames: int,
) -> None:
    offsets = np.asarray(offsets, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if total_frames <= 0:
        raise ValueError("TWIST reference bank must contain frames")
    if probabilities.shape != offsets.shape:
        raise ValueError("TWIST reference bank sampling metadata has drifted")
    if np.any(offsets < 0) or np.any(lengths <= 0):
        raise ValueError("TWIST reference bank clip metadata must be positive")
    expected_offsets = np.zeros_like(offsets)
    if len(offsets) > 1:
        expected_offsets[1:] = np.cumsum(lengths[:-1], dtype=np.int64)
    if not np.array_equal(offsets, expected_offsets) or (
        int(lengths.sum()) != total_frames
    ):
        raise ValueError("TWIST reference bank clip offsets do not cover its arrays")
    if (
        np.any(~np.isfinite(probabilities))
        or np.any(probabilities <= 0.0)
        or not np.isclose(probabilities.sum(), 1.0)
    ):
        raise ValueError("TWIST reference bank sampling probabilities are invalid")


def _batched_contact_max(
    world_ids: jax.Array, values: jax.Array, batch_size: int
) -> jax.Array:
    """Reduce flattened contact values independently for each simulation world."""

    output_shape = (batch_size, *values.shape[1:])
    return jnp.zeros(output_shape, dtype=values.dtype).at[world_ids].max(values)


def _key_body_pose_error_exceeds(
    current_positions: jax.Array,
    current_root_position: jax.Array,
    current_root_quaternion: jax.Array,
    reference_positions: jax.Array,
    reference_root_position: jax.Array,
    reference_root_quaternion: jax.Array,
    threshold: jax.Array | float,
) -> jax.Array:
    """Released TWIST termination: maximum root-local error of any key body."""

    current_local = jax.vmap(
        lambda position: quat_rotate_inverse(current_root_quaternion, position)
    )(current_positions - current_root_position)
    reference_local = jax.vmap(
        lambda position: quat_rotate_inverse(reference_root_quaternion, position)
    )(reference_positions - reference_root_position)
    per_body_squared_error = jnp.sum((current_local - reference_local) ** 2, axis=-1)
    return jnp.max(per_body_squared_error) > jnp.asarray(threshold) ** 2


def _roll_pitch(quaternion: jax.Array) -> jax.Array:
    return _roll_pitch_yaw(quaternion)[:2]


def _roll_pitch_yaw(quaternion: jax.Array) -> jax.Array:
    w, x, y, z = quaternion
    roll = jnp.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = jnp.arcsin(jnp.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return jnp.asarray((roll, pitch, yaw))


def _quat_distance(left: jax.Array, right: jax.Array) -> jax.Array:
    dot = jnp.clip(jnp.abs(jnp.dot(left, right)), 0.0, 1.0)
    return 2.0 * jnp.arccos(dot)


def _quat_multiply(left: jax.Array, right: jax.Array) -> jax.Array:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return jnp.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )


def _uniform_noise(value: jax.Array, magnitude: float, rng: jax.Array) -> jax.Array:
    magnitude = jnp.asarray(magnitude, dtype=value.dtype)
    return value + jax.random.uniform(
        rng, value.shape, minval=-magnitude, maxval=magnitude
    )


def default_twist_asset_config(asset: AssetName) -> AssetPrepConfig:
    return AssetPrepConfig(
        asset=asset,
        collision_mode="box",
        target_total_mass_kg=None,
        enforce_policy_effort_limits=False,
        actuator_mode="policy",
    )
