"""One-stage TWIST2 whole-body motion tracking for Bello MJX."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp

from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.contracts import ObservationLayout
from bello_mujoco.envs.locomotion import quat_rotate_inverse, yaw_quat_from_angle
from bello_mujoco.envs.twist import (
    BelloTwistEnv,
    DEFAULT_KEY_BODY_NAMES,
    TWIST_EPISODE_STEPS,
    TWIST_FUTURE_OFFSETS,
    TWIST_POLICY_HZ,
    TWIST_SIMULATION_HZ,
    TwistSamplingConfig,
    TwistSimulationConfig,
    TwistState,
    _key_body_pose_error_exceeds,
    _quat_distance,
    _quat_multiply,
    _roll_pitch,
    _roll_pitch_yaw,
    _uniform_noise,
    default_twist_asset_config,
)
from bello_mujoco.motion.clip import MotionSample, load_motion_clip
from bello_mujoco.motion.twist_dataset import PackedReferenceBank
from bello_mujoco.sim import (
    ModelRandomizationConfig,
    WarpConfig,
    load_bello_model,
)

TWIST2_UPSTREAM_REVISION = "d5c7108e9ef82d1b8770e5b692f27a1294f3aa8a"
TWIST2_CURRENT_OFFSETS = (0,)
TWIST2_PAST_FRAMES = 10
TWIST2_NOISE_INCREASING_STEPS = 50_000
TWIST2_NOISE_WARMUP_POLICY_STEPS = TWIST2_NOISE_INCREASING_STEPS * 24
TWIST2_ROOT_PUSH_INTERVAL_STEPS = 4 * TWIST_POLICY_HZ
TWIST2_RAW_ACTION_CLIP = 5.0 / 0.5
TWIST2_RELEASED_MOTION_COUNTS = (
    ("twist1_to_twist2", 12_788),
    ("OMOMO_g1_GMR", 5_841),
    ("AMASS_g1_GMR8", 724),
    ("v1_v2_v3_g1", 73),
)


@dataclass(frozen=True)
class Twist2ParityContract:
    """Executed behavior of the pinned public TWIST2 controller release."""

    upstream_revision: str = TWIST2_UPSTREAM_REVISION
    privileged_offsets: tuple[int, ...] = TWIST_FUTURE_OFFSETS
    current_offsets: tuple[int, ...] = TWIST2_CURRENT_OFFSETS
    past_frames: int = TWIST2_PAST_FRAMES
    corrected_full_current_command: bool = True
    teacher_distillation: bool = False
    raw_action_clip: float = TWIST2_RAW_ACTION_CLIP
    action_delay_steps: tuple[int, int] = (0, 1)
    action_delay_warmup_policy_steps: int = TWIST2_NOISE_WARMUP_POLICY_STEPS
    root_push_interval_policy_steps: int = TWIST2_ROOT_PUSH_INTERVAL_STEPS
    gravity_randomization_executed: bool = False
    actor_normalization_updates: int = 3_000
    critic_observation_normalization: bool = False

    def __post_init__(self) -> None:
        if self.upstream_revision != TWIST2_UPSTREAM_REVISION:
            raise ValueError("TWIST2 parity revision differs from the pinned release")
        if self.privileged_offsets != TWIST_FUTURE_OFFSETS:
            raise ValueError("TWIST2 privileged offsets differ from the release")
        if self.current_offsets != (0,):
            raise ValueError("released TWIST2 uses only the current command offset")
        if self.past_frames != 10:
            raise ValueError("released TWIST2 uses ten past actor frames")
        if not self.corrected_full_current_command:
            raise ValueError(
                "TWIST2 must preserve the complete current command; the upstream "
                "mask bug that drops the final joint is intentionally corrected"
            )
        if self.teacher_distillation:
            raise ValueError("released TWIST2 training disables its teacher path")
        if self.raw_action_clip != TWIST2_RAW_ACTION_CLIP:
            raise ValueError("TWIST2 raw action clip differs from executed source")
        if self.action_delay_steps != (0, 1):
            raise ValueError("TWIST2 action delay differs from executed source")
        if self.action_delay_warmup_policy_steps != TWIST2_NOISE_WARMUP_POLICY_STEPS:
            raise ValueError("TWIST2 action-delay warm-up differs from executed source")
        if self.root_push_interval_policy_steps != TWIST2_ROOT_PUSH_INTERVAL_STEPS:
            raise ValueError("TWIST2 root-push interval differs from executed source")
        if self.gravity_randomization_executed:
            raise ValueError("released TWIST2 never calls its gravity randomizer")
        if self.actor_normalization_updates != 3_000:
            raise ValueError(
                "TWIST2 actor-normalizer duration differs from the release"
            )
        if self.critic_observation_normalization:
            raise ValueError(
                "released TWIST2 critic consumes raw privileged observations"
            )


@dataclass(frozen=True)
class Twist2ObservationContract:
    joint_names: tuple[str, ...]
    key_body_names: tuple[str, ...] = DEFAULT_KEY_BODY_NAMES
    privileged_offsets: tuple[int, ...] = TWIST_FUTURE_OFFSETS
    past_frames: int = TWIST2_PAST_FRAMES
    randomization_size: int = 55

    def __post_init__(self) -> None:
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("TWIST2 joint_names must be nonempty and unique")
        if len(self.key_body_names) != 9 or len(set(self.key_body_names)) != 9:
            raise ValueError("TWIST2 requires exactly nine unique key bodies")
        if self.privileged_offsets != TWIST_FUTURE_OFFSETS:
            raise ValueError("TWIST2 privileged offsets differ from the release")
        if self.past_frames != TWIST2_PAST_FRAMES:
            raise ValueError("TWIST2 requires exactly ten past actor frames")
        if self.randomization_size != 5 + 2 * self.action_size:
            raise ValueError("TWIST2 randomization layout differs from the action set")

    @property
    def action_size(self) -> int:
        return len(self.joint_names)

    @property
    def command_frame_size(self) -> int:
        # Root XY velocity, root height, roll/pitch, yaw rate, and joint targets.
        return 6 + self.action_size

    @property
    def student_reference_frame_size(self) -> int:
        # BelloTwistEnv uses this name when allocating its explicit history.
        return self.command_frame_size

    @property
    def proprioception_frame_size(self) -> int:
        return 5 + 3 * self.action_size

    @property
    def actor_frame_size(self) -> int:
        return self.command_frame_size + self.proprioception_frame_size

    @property
    def actor_observation_size(self) -> int:
        # Current frame, ten past frames, and the released offset-zero branch.
        return (1 + self.past_frames) * self.actor_frame_size + (
            self.command_frame_size
        )

    @property
    def privileged_reference_frame_size(self) -> int:
        return 21 + self.action_size + 9 * 3

    @property
    def private_information_size(self) -> int:
        # Base velocity, root pose, key bodies, contacts, and physical DR values.
        return 3 + 3 + 4 + 27 + 2 + self.randomization_size

    @property
    def critic_observation_size(self) -> int:
        return (
            len(self.privileged_offsets) * self.privileged_reference_frame_size
            + self.proprioception_frame_size
            + self.private_information_size
        )

    @property
    def actor_layout(self) -> ObservationLayout:
        return ObservationLayout(
            (
                ("current_command", self.command_frame_size),
                ("current_proprioception", self.proprioception_frame_size),
                (
                    "past_command_proprioception",
                    self.past_frames * self.actor_frame_size,
                ),
                ("offset_zero_command", self.command_frame_size),
            )
        )

    @property
    def critic_layout(self) -> ObservationLayout:
        return ObservationLayout(
            (
                (
                    "privileged_reference",
                    len(self.privileged_offsets) * self.privileged_reference_frame_size,
                ),
                ("current_proprioception", self.proprioception_frame_size),
                ("base_linear_velocity", 3),
                ("root_position", 3),
                ("root_orientation", 4),
                ("current_key_body_positions", 27),
                ("foot_contacts", 2),
                ("domain_randomization", self.randomization_size),
            )
        )


@dataclass(frozen=True)
class Twist2ObservationNoiseConfig:
    enabled: bool = True
    base_angular_velocity: float = 0.1
    roll_pitch: float = 0.1
    joint_position: float = 0.01
    joint_velocity: float = 0.1
    increasing_steps: int = TWIST2_NOISE_INCREASING_STEPS


@dataclass(frozen=True)
class Twist2DomainRandomizationConfig:
    enabled: bool = True
    seed: int = 0
    added_base_mass: tuple[float, float] = (-3.0, 3.0)
    added_base_com: tuple[float, float] = (-0.05, 0.05)
    friction: tuple[float, float] = (0.1, 2.0)
    motor_strength: tuple[float, float] = (0.8, 1.2)
    root_push_interval_steps: int = TWIST2_ROOT_PUSH_INTERVAL_STEPS
    root_push_velocity_xy: float = 1.0


@dataclass(frozen=True)
class Twist2RewardConfig:
    tracking_joint_dof: float = 2.0
    tracking_joint_vel: float = 0.2
    tracking_root_translation_z: float = 1.0
    tracking_root_rotation: float = 1.0
    tracking_root_linear_vel: float = 1.0
    tracking_root_angular_vel: float = 1.0
    tracking_keybody_pos: float = 2.0
    tracking_keybody_pos_global: float = 2.0
    alive: float = 0.5
    feet_slip: float = -0.1
    feet_contact_forces: float = -5.0e-4
    feet_stumble: float = -1.25
    dof_pos_limits: float = -5.0
    dof_torque_limits: float = -1.0
    dof_vel: float = -1.0e-4
    dof_acc: float = -5.0e-8
    action_rate: float = -0.05
    feet_air_time: float = 5.0
    ang_vel_xy: float = -0.01
    ankle_dof_acc: float = -1.0e-7
    ankle_dof_vel: float = -2.0e-4
    feet_air_time_target: float = 0.5
    max_contact_force: float = 500.0
    soft_torque_limit: float = 0.95


@dataclass(frozen=True)
class Twist2DoneConfig:
    root_height_error: float = 0.3
    roll_pitch_limit: float = 4.0
    key_body_position_error: float = 0.7
    root_velocity_limit: float = 5.0


@dataclass(frozen=True)
class Twist2EnvConfig:
    max_episode_steps: int = TWIST_EPISODE_STEPS
    action_clip: float = TWIST2_RAW_ACTION_CLIP
    action_delay_steps: tuple[int, int] = (0, 1)
    action_history_steps: int = 8
    observation_noise: Twist2ObservationNoiseConfig = Twist2ObservationNoiseConfig()
    domain_randomization: Twist2DomainRandomizationConfig = (
        Twist2DomainRandomizationConfig()
    )
    reward: Twist2RewardConfig = Twist2RewardConfig()
    done: Twist2DoneConfig = Twist2DoneConfig()
    key_body_names: tuple[str, ...] = DEFAULT_KEY_BODY_NAMES

    def __post_init__(self) -> None:
        if self.max_episode_steps != TWIST_EPISODE_STEPS:
            raise ValueError("TWIST2 episodes must be exactly 10 seconds at 50 Hz")
        if self.action_clip != TWIST2_RAW_ACTION_CLIP:
            raise ValueError("TWIST2 action_clip differs from executed source")
        if self.action_delay_steps != (0, 1):
            raise ValueError(
                "released TWIST2 executes only zero- or one-step action delay"
            )
        if self.action_history_steps != 8:
            raise ValueError("released TWIST2 allocates eight action frames")
        if len(self.key_body_names) != 9 or len(set(self.key_body_names)) != 9:
            raise ValueError("TWIST2 requires exactly nine unique key bodies")
        noise = self.observation_noise
        if noise.increasing_steps != TWIST2_NOISE_INCREASING_STEPS:
            raise ValueError("TWIST2 noise curriculum must use 50,000 steps")
        for name in (
            "base_angular_velocity",
            "roll_pitch",
            "joint_position",
            "joint_velocity",
        ):
            value = getattr(noise, name)
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
            gravity_fraction_range=(0.0, 0.0),
        )
        if domain.root_push_interval_steps != TWIST2_ROOT_PUSH_INTERVAL_STEPS:
            raise ValueError("TWIST2 root-push interval differs from executed source")
        if (
            not math.isfinite(domain.root_push_velocity_xy)
            or domain.root_push_velocity_xy < 0.0
        ):
            raise ValueError(
                "domain_randomization.root_push_velocity_xy must be nonnegative"
            )
        for field_name in self.reward.__dataclass_fields__:
            value = getattr(self.reward, field_name)
            if not math.isfinite(value):
                raise ValueError(f"reward.{field_name} must be finite")
        for field_name in self.done.__dataclass_fields__:
            value = getattr(self.done, field_name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"done.{field_name} must be positive and finite")


@dataclass(frozen=True)
class BelloTwist2Env(BelloTwistEnv):
    """Isolated one-stage TWIST2 task using shared Bello simulation storage."""

    config: Twist2EnvConfig = Twist2EnvConfig()

    def __post_init__(self) -> None:
        if self.stage != "student":
            raise ValueError(
                "TWIST2 has one stage and uses the internal student history"
            )
        super().__post_init__()

    @classmethod
    def load(
        cls,
        asset_config: AssetPrepConfig,
        simulation: TwistSimulationConfig,
        *,
        motion_path: Path,
        config: Twist2EnvConfig = Twist2EnvConfig(),
        warp: WarpConfig = WarpConfig(),
        batch_size: int = 1,
        reference_bank: PackedReferenceBank | None = None,
        sampling: TwistSamplingConfig = TwistSamplingConfig(
            curriculum_max_difficulty=10.0
        ),
    ) -> "BelloTwist2Env":
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
                gravity_fraction_range=(0.0, 0.0),
            ),
        )
        motion = load_motion_clip(motion_path, model)
        return cls(model, motion, "student", config, reference_bank, sampling)

    @property
    def contract(self) -> Twist2ObservationContract:
        return Twist2ObservationContract(
            joint_names=self.model.actuator_joint_names,
            key_body_names=self.config.key_body_names,
            randomization_size=5 + 2 * self.action_size,
        )

    @property
    def task(self) -> Literal["twist2"]:
        return "twist2"

    @property
    def observation_layout(self) -> ObservationLayout:
        return self.contract.actor_layout

    @property
    def critic_observation_layout(self) -> ObservationLayout:
        return self.contract.critic_layout

    def _unchecked_runtime_copy(
        self, *, reference_bank: PackedReferenceBank | None
    ) -> "BelloTwist2Env":
        bound = object.__new__(BelloTwist2Env)
        for name in ("model", "motion", "stage", "config", "sampling"):
            object.__setattr__(bound, name, getattr(self, name))
        object.__setattr__(bound, "reference_bank", reference_bank)
        return bound

    def initial_task_state(self) -> jax.Array:
        clip_count = (
            len(self.reference_bank.sampling_probabilities)
            if self.reference_bank is not None
            else 0
        )
        return jnp.full(
            (clip_count,), self.sampling.curriculum_max_difficulty, dtype=jnp.float32
        )

    def update_task_state(
        self, difficulty: jax.Array, info: dict[str, jax.Array]
    ) -> jax.Array:
        if self.reference_bank is None or not self.sampling.curriculum_enabled:
            return difficulty
        clip_indices = info["reference/clip_index"].astype(jnp.int32)
        returned = info["returned_episode"].astype(jnp.float32)
        clip_lengths = jnp.asarray(self.reference_bank.lengths)[clip_indices]
        completion = info["episode_length"].astype(jnp.float32) / jnp.maximum(
            clip_lengths.astype(jnp.float32), 1.0
        )
        totals = jnp.zeros_like(difficulty).at[clip_indices].add(completion * returned)
        counts = jnp.zeros_like(difficulty).at[clip_indices].add(returned)
        rates = jnp.where(counts > 0.0, totals / jnp.maximum(counts, 1.0), 0.7)
        gamma = self.sampling.curriculum_gamma
        factor = jnp.where(
            rates <= 0.5,
            1.0 + gamma,
            jnp.where(
                rates >= 0.99,
                1.0 - 20.0 * gamma,
                jnp.where(rates >= 0.95, 1.0 - gamma, 1.0),
            ),
        )
        return jnp.clip(
            difficulty * factor,
            self.sampling.curriculum_min_difficulty,
            self.sampling.curriculum_max_difficulty,
        )

    def reset(self, rng: jax.Array | None = None) -> tuple[TwistState, jax.Array]:
        state, _ = super().reset(rng)
        return state, self._initial_actor_observation(state)

    def reset_batch(
        self, batch_size: int, rng: jax.Array | None = None
    ) -> tuple[TwistState, jax.Array]:
        state, _ = super().reset_batch(batch_size, rng)
        current = state.observation_history[:, -1]
        command = current[:, : self.contract.command_frame_size]
        empty_history = jnp.zeros_like(state.observation_history[:, :-1]).reshape(
            (batch_size, -1)
        )
        return state, jnp.concatenate((current, empty_history, command), axis=-1)

    def _initial_actor_observation(self, state: TwistState) -> jax.Array:
        current = state.observation_history[-1]
        command = current[: self.contract.command_frame_size]
        empty_history = jnp.zeros_like(state.observation_history[:-1]).reshape(-1)
        return jnp.concatenate((current, empty_history, command))

    def _student_history_transition(
        self,
        previous: TwistState,
        current: TwistState,
        frame: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        shifted, observed = super()._student_history_transition(
            previous, current, frame
        )
        filled = jnp.repeat(frame[None, :], shifted.shape[0], axis=0)
        stored = jnp.where(current.motion_state.episode_length <= 1, filled, shifted)
        return stored, observed

    def _student_history_transition_batch(
        self,
        previous: TwistState,
        current: TwistState,
        frames: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        shifted, observed = super()._student_history_transition_batch(
            previous, current, frames
        )
        filled = jnp.repeat(frames[:, None, :], shifted.shape[1], axis=1)
        first_step = current.motion_state.episode_length <= 1
        stored = jnp.where(first_step[:, None, None], filled, shifted)
        return stored, observed

    def observation(self, state: TwistState, rng: jax.Array | None = None) -> jax.Array:
        del rng
        history = state.observation_history
        current = history[-1]
        command = current[: self.contract.command_frame_size]
        return jnp.concatenate((current, history[:-1].reshape(-1), command))

    def observation_batch(
        self, state: TwistState, rng: jax.Array | None = None
    ) -> jax.Array:
        del rng
        history = state.observation_history
        current = history[:, -1]
        command = current[:, : self.contract.command_frame_size]
        return jnp.concatenate(
            (current, history[:, :-1].reshape((history.shape[0], -1)), command),
            axis=-1,
        )

    def critic_observation(self, state: TwistState) -> jax.Array:
        core = self._critic_observation_core(state)
        return jnp.concatenate(
            (
                core,
                self._foot_contacts(state.motion_state, 0),
                state.domain_randomization,
            )
        )

    def critic_observation_batch(
        self, state: TwistState, rng: jax.Array | None = None
    ) -> jax.Array:
        del rng
        batch_size = state.action_delay_steps.shape[0]
        return jnp.concatenate(
            (
                jax.vmap(self._critic_observation_core)(state),
                self._foot_contacts_batch(state.motion_state, batch_size),
                state.domain_randomization,
            ),
            axis=-1,
        )

    def critic_observation_with_world(
        self, state: TwistState, world_id: jax.Array
    ) -> jax.Array:
        return jnp.concatenate(
            (
                self._critic_observation_core(state),
                self._foot_contacts(state.motion_state, world_id),
                state.domain_randomization,
            )
        )

    def _critic_observation_core(self, state: TwistState) -> jax.Array:
        privileged = jnp.concatenate(
            tuple(
                self._privileged_reference_frame(state, offset)
                for offset in TWIST_FUTURE_OFFSETS
            )
        )
        motion_state = state.motion_state
        root_quat = motion_state.data.qpos[3:7]
        base_linear_velocity = quat_rotate_inverse(
            root_quat, motion_state.data.qvel[:3]
        )
        key_positions = motion_state.data.xpos[jnp.asarray(self.key_body_ids)]
        local_key_positions = jax.vmap(
            lambda value: quat_rotate_inverse(
                root_quat, value - motion_state.data.qpos[:3]
            )
        )(key_positions).reshape(-1)
        return jnp.concatenate(
            (
                privileged,
                self._proprioception(state),
                base_linear_velocity,
                motion_state.data.qpos[:3],
                root_quat,
                local_key_positions,
            )
        )

    def _student_reference_frame(self, state: TwistState, offset: int) -> jax.Array:
        reference = self._aligned_reference(
            state.motion_state, state.clip_index, offset
        )
        root_quat = reference.qpos[3:7]
        local_linear_velocity = quat_rotate_inverse(root_quat, reference.qvel[:3])
        local_angular_velocity = quat_rotate_inverse(root_quat, reference.qvel[3:6])
        return jnp.concatenate(
            (
                local_linear_velocity[:2],
                reference.qpos[2:3],
                _roll_pitch(root_quat),
                local_angular_velocity[2:3],
                reference.qpos[jnp.asarray(self.model.policy_qpos_indices)],
            )
        )

    def _privileged_reference_frame(self, state: TwistState, offset: int) -> jax.Array:
        current = state.motion_state
        reference = self._aligned_reference(current, state.clip_index, offset)
        previous = self._aligned_reference(current, state.clip_index, offset - 1)
        root_quat = reference.qpos[3:7]
        local_linear_velocity = quat_rotate_inverse(root_quat, reference.qvel[:3])
        local_angular_velocity = quat_rotate_inverse(root_quat, reference.qvel[3:6])
        position_delta = quat_rotate_inverse(
            previous.qpos[3:7], reference.qpos[:3] - previous.qpos[:3]
        )
        source_quat_delta = _quat_multiply(
            previous.qpos[3:7], _quat_conjugate(root_quat)
        )
        rotation_delta = quat_rotate_inverse(
            previous.qpos[3:7], _roll_pitch_yaw(source_quat_delta)
        )
        key_indices = jnp.asarray(
            [self.motion.body_names.index(name) for name in self.config.key_body_names]
        )
        local_key_positions = jax.vmap(
            lambda value: quat_rotate_inverse(root_quat, value - reference.qpos[:3])
        )(reference.body_pos[key_indices]).reshape(-1)
        return jnp.concatenate(
            (
                reference.qpos[:3],
                reference.qpos[:3] - current.data.qpos[:3],
                _roll_pitch_yaw(root_quat),
                local_linear_velocity,
                local_angular_velocity,
                position_delta,
                rotation_delta,
                reference.qpos[jnp.asarray(self.model.policy_qpos_indices)],
                local_key_positions,
            )
        )

    def _proprioception(
        self, state: TwistState, rng: jax.Array | None = None
    ) -> jax.Array:
        motion_state = state.motion_state
        root_quat = motion_state.data.qpos[3:7]
        base_angular_velocity = 0.25 * quat_rotate_inverse(
            root_quat, motion_state.data.qvel[3:6]
        )
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
                state.total_step.astype(jnp.float32) / float(cfg.increasing_steps * 24),
                1.0,
            )
            base_angular_velocity = _uniform_noise(
                base_angular_velocity, ramp * cfg.base_angular_velocity, keys[0]
            )
            roll_pitch = _uniform_noise(roll_pitch, ramp * cfg.roll_pitch, keys[1])
            qpos = _uniform_noise(qpos, ramp * cfg.joint_position, keys[2])
            qvel = _uniform_noise(qvel, ramp * cfg.joint_velocity, keys[3])
        qvel = qvel.at[jnp.asarray(self._ankle_policy_indices)].set(0.0)
        return jnp.concatenate(
            (base_angular_velocity, roll_pitch, qpos, qvel, state.action_history[-1])
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
        root_height_error = reference.qpos[2] - current.data.qpos[2]
        root_orientation_error = _quat_distance(
            current.data.qpos[3:7], reference.qpos[3:7]
        )
        reference_linear = quat_rotate_inverse(reference.qpos[3:7], reference.qvel[:3])
        current_linear = quat_rotate_inverse(
            current.data.qpos[3:7], current.data.qvel[:3]
        )
        reference_angular = quat_rotate_inverse(
            reference.qpos[3:7], reference.qvel[3:6]
        )
        current_angular = quat_rotate_inverse(
            current.data.qpos[3:7], current.data.qvel[3:6]
        )
        key_indices = jnp.asarray(
            [self.motion.body_names.index(name) for name in self.config.key_body_names]
        )
        target_global = reference.body_pos[key_indices]
        current_global = current.data.xpos[jnp.asarray(self.key_body_ids)]
        reference_yaw = yaw_quat_from_angle(_roll_pitch_yaw(reference.qpos[3:7])[2])
        current_yaw = yaw_quat_from_angle(_roll_pitch_yaw(current.data.qpos[3:7])[2])
        target_local = jax.vmap(
            lambda value: quat_rotate_inverse(reference_yaw, value - reference.qpos[:3])
        )(target_global)
        current_local = jax.vmap(
            lambda value: quat_rotate_inverse(
                current_yaw, value - current.data.qpos[:3]
            )
        )(current_global)
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
        first_contact = (previous.foot_air_time > 0.0) & (
            foot_contact | previous.last_foot_contact
        )
        moving = jnp.linalg.norm(reference.qvel[:2]) > 0.05
        cfg = self.config.reward
        air_time_reward = jnp.sum(
            jnp.minimum(foot_air_time - cfg.feet_air_time_target, 0.0)
            * first_contact.astype(jnp.float32)
        ) * moving.astype(jnp.float32)
        ankle = jnp.asarray(self._ankle_policy_indices)
        raw = {
            "tracking_joint_dof": jnp.exp(-0.15 * jnp.sum(qpos_error**2)),
            "tracking_joint_vel": jnp.exp(-0.01 * jnp.sum(qvel_error**2)),
            "tracking_root_translation_z": jnp.exp(-5.0 * root_height_error**2),
            "tracking_root_rotation": jnp.exp(-5.0 * root_orientation_error**2),
            "tracking_root_linear_vel": jnp.exp(
                -jnp.sum((reference_linear - current_linear) ** 2)
            ),
            "tracking_root_angular_vel": jnp.exp(
                -jnp.sum((reference_angular - current_angular) ** 2)
            ),
            "tracking_keybody_pos": jnp.exp(
                -10.0 * jnp.sum((current_local - target_local) ** 2)
            ),
            "tracking_keybody_pos_global": jnp.exp(
                -10.0 * jnp.sum((current_global - target_global) ** 2)
            ),
            "alive": jnp.asarray(1.0, dtype=jnp.float32),
            "feet_slip": jnp.sum(
                jnp.sqrt(jnp.linalg.norm(foot_velocity[:, :2], axis=-1))
                * foot_contact.astype(jnp.float32)
            ),
            "feet_contact_forces": jax.nn.relu(
                jnp.linalg.norm(contact_normal) - cfg.max_contact_force
            ),
            "feet_stumble": jnp.any(
                contact_tangent > 4.0 * jnp.abs(contact_normal)
            ).astype(jnp.float32),
            "dof_pos_limits": jnp.sum(
                jax.nn.relu(lower - positions) + jax.nn.relu(positions - upper)
            ),
            "dof_torque_limits": jnp.sum(
                jax.nn.relu(
                    torque / jnp.maximum(torque_limit, 1.0e-6) - cfg.soft_torque_limit
                )
            ),
            "dof_vel": jnp.sum(velocities**2),
            "dof_acc": jnp.sum(acceleration**2),
            "action_rate": jnp.linalg.norm(
                current.last_action - previous.motion_state.last_action
            ),
            "feet_air_time": air_time_reward,
            "ang_vel_xy": jnp.sum(current_angular[:2] ** 2),
            "ankle_dof_acc": jnp.sum(acceleration[ankle] ** 2),
            "ankle_dof_vel": jnp.sum(velocities[ankle] ** 2),
        }
        dt = self.control_timestep
        components = {
            f"reward/{name}": dt * getattr(cfg, name) * value
            for name, value in raw.items()
        }
        return sum(components.values(), jnp.asarray(0.0, dtype=jnp.float32)), components

    def _bank_termination_flags(
        self,
        state,
        clip_index: jax.Array,
        reference: MotionSample,
        world_id: jax.Array | int,
        *,
        torso_contact: jax.Array | None = None,
        tracking_state: TwistState | None = None,
    ) -> dict[str, jax.Array]:
        del world_id, torso_contact, tracking_state
        cfg = self.config.done
        height = jnp.abs(state.data.qpos[2] - reference.qpos[2]) > cfg.root_height_error
        orientation = jnp.any(
            jnp.abs(_roll_pitch(state.data.qpos[3:7])) > cfg.roll_pitch_limit
        )
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
        terminal = height | orientation | pose | root_velocity | nonfinite
        false = jnp.asarray(False)
        return {
            "done/height": height,
            "done/orientation": orientation,
            "done/pose": pose,
            "done/root_velocity": root_velocity,
            "done/torso_contact": false,
            "done/nonfinite": nonfinite,
            "done/timeout": timeout,
            "done/twist2_timeout": state.step_count >= self.config.max_episode_steps,
            "done/terminal": terminal,
            "done/any": terminal | timeout,
        }

    def _sample_action_delay(self, rng: jax.Array | None) -> jax.Array:
        if rng is None:
            return jnp.asarray(0, dtype=jnp.int32)
        return jax.random.randint(rng, (), 0, 2, dtype=jnp.int32)

    def _termination_torso_contacts_batch(self, state, batch_size: int) -> jax.Array:
        del state
        return jnp.zeros((batch_size,), dtype=bool)

    def _sample_action_delay_batch(
        self, state: TwistState, rng: jax.Array | None
    ) -> jax.Array:
        sampled = self._sample_action_delay(rng)
        warmup = self.config.observation_noise.increasing_steps * 24
        return jnp.where(state.total_step[0] <= warmup, 0, sampled).astype(jnp.int32)

    def _apply_disturbances_batch(self, data, state: TwistState, rng):
        batch_size = int(state.motion_state.step_count.shape[0])
        cfg = self.config.domain_randomization
        if rng is None or not cfg.enabled:
            return data.replace(
                xfrc_applied=jnp.zeros_like(data.xfrc_applied)
            ), jnp.zeros((batch_size,), dtype=bool)
        next_total_step = state.total_step[0] + 1
        root_due_scalar = next_total_step % cfg.root_push_interval_steps == 0
        root_due = jnp.full((batch_size,), root_due_scalar, dtype=bool)
        return data.replace(xfrc_applied=jnp.zeros_like(data.xfrc_applied)), root_due

    def _apply_post_step_disturbances_batch(self, data, state: TwistState, rng):
        cfg = self.config.domain_randomization
        if rng is None or not cfg.enabled:
            return data
        batch_size = int(state.motion_state.step_count.shape[0])
        next_total_step = state.total_step[0] + 1
        root_due_scalar = next_total_step % cfg.root_push_interval_steps == 0
        root_due = jnp.full((batch_size,), root_due_scalar, dtype=bool)
        root_velocity = jax.random.uniform(
            rng,
            (batch_size, 2),
            minval=-cfg.root_push_velocity_xy,
            maxval=cfg.root_push_velocity_xy,
        )
        qvel = data.qvel.at[:, :2].set(
            jnp.where(root_due[:, None], root_velocity, data.qvel[:, :2])
        )
        return data.replace(qvel=qvel)


def default_twist2_asset_config() -> AssetPrepConfig:
    return default_twist_asset_config()


def _quat_conjugate(quaternion: jax.Array) -> jax.Array:
    return quaternion * jnp.asarray((1.0, -1.0, -1.0, -1.0))
