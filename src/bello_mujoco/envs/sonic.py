"""SONIC transfer task for Bello using synchronized AMASS references.

The task deliberately owns its observation, reward, termination, reset, and
adaptive-sampling semantics.  It reuses only the vectorized MJX stepping and
contact plumbing from :mod:`bello_mujoco.envs.twist`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal, NamedTuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from bello_mujoco.actuators import joint_position_action_scales
from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.contracts import ObservationLayout
from bello_mujoco.envs.locomotion import (
    quat_rotate,
    quat_rotate_inverse,
    yaw_quat,
    yaw_quat_from_angle,
)
from bello_mujoco.envs.motion_tracking import quat_from_euler_xyz, rotation_6d
from bello_mujoco.envs.twist import (
    BelloTwistEnv,
    TwistMotionState,
    TwistSimulationConfig,
    _quat_distance,
    _quat_multiply,
    _uniform_noise,
    default_twist_asset_config,
)
from bello_mujoco.motion.clip import MotionSample, load_motion_clip
from bello_mujoco.motion.sonic_dataset import (
    SONIC_CRITIC_FUTURE_OFFSETS,
    SONIC_REQUIRED_FUTURE_HORIZON,
    SONIC_SMPL_FUTURE_OFFSETS,
    SonicReferenceBank,
    build_sonic_sampling_bins,
)
from bello_mujoco.sim import ModelRandomizationConfig, WarpConfig, load_bello_model
from bello_mujoco.sonic_policy import (
    BELLO_TO_G1_VECTOR_BASIS,
    SONIC_ACTOR_OBSERVATION_SIZE,
    SONIC_BELLO_ACTION_SIZE,
    SONIC_BELLO_PROPRIO_SIZE,
    SONIC_ROOT_MOTION_COMMAND_SIZE,
    SONIC_SMPL_INPUT_SIZE,
)

SONIC_SIMULATION_HZ = 200
SONIC_POLICY_HZ = 50
SONIC_DECIMATION = SONIC_SIMULATION_HZ // SONIC_POLICY_HZ
SONIC_HISTORY_FRAMES = 10
SONIC_EPISODE_STEPS = 10 * SONIC_POLICY_HZ
SONIC_ACTION_CLIP = 20.0

SONIC_TRACKED_BODY_NAMES = (
    "pelvis_link",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "l_shoulder_roll_link",
    "l_elbow_link",
    "l_wrist_link",
    "r_shoulder_roll_link",
    "r_elbow_link",
    "r_wrist_link",
)
# The released term is named ``tracking_vr_5point_local``, but its pinned
# configuration contains only a head proxy and the two wrists. Bello has a
# real head body, so use that semantic point instead of G1's torso offset.
SONIC_REWARD_POINT_BODY_NAMES = (
    "head_link",
    "l_wrist_link",
    "r_wrist_link",
)
SONIC_END_EFFECTOR_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "l_wrist_link",
    "r_wrist_link",
)
SONIC_ANTI_SHAKE_BODY_NAMES = ("l_wrist_link", "r_wrist_link", "head_link")
SONIC_ALLOWED_CONTACT_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "l_wrist_link",
    "r_wrist_link",
    "l_elbow_link",
    "r_elbow_link",
)


@dataclass(frozen=True)
class SonicObservationContract:
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...] = SONIC_TRACKED_BODY_NAMES
    history_frames: int = SONIC_HISTORY_FRAMES
    smpl_future_offsets: tuple[int, ...] = SONIC_SMPL_FUTURE_OFFSETS
    critic_future_offsets: tuple[int, ...] = SONIC_CRITIC_FUTURE_OFFSETS

    def __post_init__(self) -> None:
        if len(self.joint_names) != SONIC_BELLO_ACTION_SIZE:
            raise ValueError("Bello SONIC requires exactly 25 action joints")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("SONIC action joint names must be unique")
        if self.body_names != SONIC_TRACKED_BODY_NAMES:
            raise ValueError("SONIC tracked-body order differs from the port contract")
        if self.history_frames != SONIC_HISTORY_FRAMES:
            raise ValueError("SONIC requires exactly ten proprioceptive frames")
        if self.smpl_future_offsets != SONIC_SMPL_FUTURE_OFFSETS:
            raise ValueError("SONIC SMPL future offsets differ from the release")
        if self.critic_future_offsets != SONIC_CRITIC_FUTURE_OFFSETS:
            raise ValueError("SONIC critic future offsets differ from the release")

    @property
    def action_size(self) -> int:
        return len(self.joint_names)

    @property
    def required_body_names(self) -> tuple[str, ...]:
        return self.body_names + tuple(
            name
            for name in SONIC_REWARD_POINT_BODY_NAMES
            if name not in self.body_names
        )

    @property
    def history_frame_size(self) -> int:
        # Gravity, base linear/angular velocity, q, qdot, and previous action.
        return 9 + 3 * self.action_size

    @property
    def student_reference_frame_size(self) -> int:
        # Compatibility with the shared history transition; SONIC commands are
        # supplied separately to its frozen SMPL encoder.
        return 0

    @property
    def proprioception_frame_size(self) -> int:
        return self.history_frame_size

    @property
    def actor_observation_size(self) -> int:
        return SONIC_ACTOR_OBSERVATION_SIZE

    @property
    def critic_command_frame_size(self) -> int:
        return 2 * self.action_size

    @property
    def critic_observation_size(self) -> int:
        # Ten future joint-position/velocity frames, anchor delta, current body
        # transforms, and raw ten-frame proprioception without gravity.
        return (
            len(self.critic_future_offsets) * self.critic_command_frame_size
            + 3
            + 6
            + 3 * len(self.body_names)
            + 6 * len(self.body_names)
            + SONIC_BELLO_PROPRIO_SIZE
        )

    @property
    def actor_layout(self) -> ObservationLayout:
        return ObservationLayout(
            (
                ("smpl_future_encoder_input", SONIC_SMPL_INPUT_SIZE),
                ("desired_root_xy_trajectory", SONIC_ROOT_MOTION_COMMAND_SIZE),
                ("projected_gravity_history", 30),
                ("base_angular_velocity_history", 30),
                ("joint_position_history", 250),
                ("joint_velocity_history", 250),
                ("action_history", 250),
            )
        )

    @property
    def critic_layout(self) -> ObservationLayout:
        return ObservationLayout(
            (
                (
                    "future_joint_positions",
                    len(self.critic_future_offsets) * self.action_size,
                ),
                (
                    "future_joint_velocities",
                    len(self.critic_future_offsets) * self.action_size,
                ),
                ("motion_anchor_position_in_robot", 3),
                ("motion_anchor_orientation_in_robot", 6),
                ("robot_body_positions", 3 * len(self.body_names)),
                ("robot_body_orientations", 6 * len(self.body_names)),
                ("base_linear_velocity_history", 30),
                ("base_angular_velocity_history", 30),
                ("joint_position_history", 250),
                ("joint_velocity_history", 250),
                ("action_history", 250),
            )
        )


@dataclass(frozen=True)
class SonicObservationNoiseConfig:
    enabled: bool = True
    ramp_steps: int = 48_000
    smpl_joint_position: float = 0.05
    smpl_root_orientation_6d: float = 0.05
    projected_gravity: float = 0.05
    base_angular_velocity: float = 0.2
    joint_position: float = 0.01
    joint_velocity: float = 0.5


@dataclass(frozen=True)
class SonicDomainRandomizationConfig:
    enabled: bool = True
    seed: int = 0
    joint_default_position: tuple[float, float] = (-0.01, 0.01)
    added_base_mass: tuple[float, float] = (-2.0, 5.0)
    base_com: tuple[float, float] = (-0.05, 0.05)
    friction: tuple[float, float] = (0.3, 1.6)
    motor_strength: tuple[float, float] = (0.9, 1.1)
    push_interval_steps: tuple[int, int] = (4 * SONIC_POLICY_HZ, 6 * SONIC_POLICY_HZ)
    push_warmup_steps: int = 48_000
    push_linear_velocity_xy: float = 0.5
    push_linear_velocity_z: float = 0.2
    push_angular_velocity_roll_pitch: float = 0.52
    push_angular_velocity_yaw: float = 0.78


@dataclass(frozen=True)
class SonicRewardConfig:
    tracking_anchor_position: float = 0.5
    tracking_anchor_orientation: float = 0.5
    tracking_relative_body_position: float = 1.0
    tracking_relative_body_orientation: float = 1.0
    tracking_body_linear_velocity: float = 1.0
    tracking_body_angular_velocity: float = 1.0
    tracking_joint_position: float = 1.0
    tracking_foot_contact: float = 0.25
    action_rate: float = -0.1
    joint_limit: float = -10.0
    target_joint_limit: float = -1.0
    undesired_contacts: float = -0.1
    anti_shake_angular_velocity: float = -0.005
    tracking_five_point_local: float = 2.0
    feet_acceleration: float = -2.5e-6
    anchor_position_std: float = 0.3
    anchor_orientation_std: float = 0.4
    relative_body_position_std: float = 0.3
    relative_body_orientation_std: float = 0.4
    body_linear_velocity_std: float = 1.0
    body_angular_velocity_std: float = 3.14
    joint_position_std: float = 0.35
    five_point_std: float = 0.1
    undesired_contact_threshold: float = 1.0
    anti_shake_threshold: float = 1.5


@dataclass(frozen=True)
class SonicDoneConfig:
    root_height_error: float = 0.15
    root_height_error_when_low: float = 0.75
    low_reference_root_height: float = 0.5
    root_orientation_squared_error: float = 0.2
    end_effector_height_error: float = 0.15
    end_effector_height_error_when_low: float = 0.75
    foot_position_error: float = 0.2


@dataclass(frozen=True)
class SonicSamplingConfig:
    enabled: bool = True
    bin_size: int = 50
    initial_failures: float = 1.0
    uniform_sampling_rate: float = 0.1
    pre_failure_window: int = 200
    failure_rate_max_over_mean: float = 200.0


@dataclass(frozen=True)
class SonicEnvConfig:
    max_episode_steps: int = SONIC_EPISODE_STEPS
    action_clip: float = SONIC_ACTION_CLIP
    action_delay_steps: tuple[int, int] = (0, 0)
    action_history_steps: int = 1
    observation_noise: SonicObservationNoiseConfig = SonicObservationNoiseConfig()
    domain_randomization: SonicDomainRandomizationConfig = (
        SonicDomainRandomizationConfig()
    )
    reward: SonicRewardConfig = SonicRewardConfig()
    done: SonicDoneConfig = SonicDoneConfig()
    sampling: SonicSamplingConfig = SonicSamplingConfig()
    key_body_names: tuple[str, ...] = SONIC_TRACKED_BODY_NAMES

    def __post_init__(self) -> None:
        if self.max_episode_steps != SONIC_EPISODE_STEPS:
            raise ValueError("SONIC episodes must be 10 seconds at 50 Hz")
        if self.action_clip != SONIC_ACTION_CLIP:
            raise ValueError("SONIC action clipping must match the released value 20")
        if self.action_delay_steps != (0, 0) or self.action_history_steps != 1:
            raise ValueError("SONIC transfer does not add action delay")
        if self.key_body_names != SONIC_TRACKED_BODY_NAMES:
            raise ValueError("SONIC key-body order differs from the port contract")
        noise = self.observation_noise
        if isinstance(noise.ramp_steps, bool) or noise.ramp_steps < 0:
            raise ValueError("observation_noise.ramp_steps must be nonnegative")
        for name in noise.__dataclass_fields__:
            if name in {"enabled", "ramp_steps"}:
                continue
            value = getattr(noise, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"observation_noise.{name} must be nonnegative")
        domain = self.domain_randomization
        ModelRandomizationConfig(
            enabled=domain.enabled,
            seed=domain.seed,
            joint_angle_offset_range=domain.joint_default_position,
            joint_angle_offset_scope="policy",
            added_base_mass_range=domain.added_base_mass,
            base_com_offset_range=domain.base_com,
            friction_range=domain.friction,
            actuator_strength_range=domain.motor_strength,
        )
        push_low, push_high = domain.push_interval_steps
        if push_low <= 0 or push_high < push_low:
            raise ValueError("SONIC push interval must be positive and ordered")
        if isinstance(domain.push_warmup_steps, bool) or domain.push_warmup_steps < 0:
            raise ValueError("SONIC push warmup must be nonnegative")
        for name in (
            "push_linear_velocity_xy",
            "push_linear_velocity_z",
            "push_angular_velocity_roll_pitch",
            "push_angular_velocity_yaw",
        ):
            value = getattr(domain, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"domain_randomization.{name} must be nonnegative")
        reward = self.reward
        for name in reward.__dataclass_fields__:
            if not math.isfinite(getattr(reward, name)):
                raise ValueError(f"reward.{name} must be finite")
        for name in (
            "anchor_position_std",
            "anchor_orientation_std",
            "relative_body_position_std",
            "relative_body_orientation_std",
            "body_linear_velocity_std",
            "body_angular_velocity_std",
            "joint_position_std",
            "five_point_std",
            "undesired_contact_threshold",
            "anti_shake_threshold",
        ):
            if getattr(reward, name) <= 0.0:
                raise ValueError(f"reward.{name} must be positive")
        for name in self.done.__dataclass_fields__:
            value = getattr(self.done, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"done.{name} must be positive and finite")
        sampling = self.sampling
        if sampling.bin_size <= 0 or sampling.pre_failure_window < 0:
            raise ValueError("SONIC sampling bin/window sizes are invalid")
        if sampling.initial_failures <= 0.0:
            raise ValueError("initial_failures must be positive")
        if not 0.0 <= sampling.uniform_sampling_rate <= 1.0:
            raise ValueError("uniform_sampling_rate must be in [0, 1]")
        if sampling.failure_rate_max_over_mean < 1.0:
            raise ValueError("failure_rate_max_over_mean must be at least one")


class SonicSamplingState(NamedTuple):
    failures: jax.Array
    attempts: jax.Array


@chex.dataclass(frozen=True)
class SonicState:
    motion_state: TwistMotionState
    observation_history: jax.Array
    critic_history: jax.Array
    action_history: jax.Array
    action_delay_steps: jax.Array
    clip_index: jax.Array
    domain_randomization: jax.Array
    foot_air_time: jax.Array
    last_foot_contact: jax.Array
    total_step: jax.Array
    gravity_fraction: jax.Array
    running_reference_root_height: jax.Array
    next_push_step: jax.Array


@dataclass(frozen=True)
class BelloSonicEnv(BelloTwistEnv):
    """Morphology-adapted transfer environment for the released SONIC actor."""

    config: SonicEnvConfig = SonicEnvConfig()
    reference_bank: SonicReferenceBank | None = None
    sampling: SonicSamplingConfig = SonicSamplingConfig()
    sampling_bin_cdf: jax.Array | None = None
    action_scale: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.stage != "student":
            raise ValueError("SONIC has one transfer-learning stage")
        if self.sampling != self.config.sampling:
            raise ValueError(
                "SONIC sampler configuration must match the environment contract"
            )
        if self.model.mjx_model is None:
            raise ValueError("SONIC requires an MJX-enabled Bello model")
        if self.reference_bank is None:
            raise ValueError("SONIC requires a synchronized reference bank")
        if not np.array_equal(
            self.model.policy_ctrl_indices,
            np.arange(self.model.mj_model.nu, dtype=np.int32),
        ):
            raise ValueError("SONIC requires every generated actuator as an action")
        if self.action_size != SONIC_BELLO_ACTION_SIZE:
            raise ValueError("SONIC transfer requires Bello's 25-action embodiment")
        action_scale = np.asarray(self.action_scale)
        if action_scale.shape != (self.action_size,) or not np.all(
            np.isfinite(action_scale) & (action_scale > 0.0)
        ):
            raise ValueError("SONIC requires one positive action scale per Bello joint")
        if self.reference_bank.joint_names != self.action_layout.names:
            raise ValueError("SONIC dataset action order differs from the Bello model")
        if (
            self.reference_bank.calibration.bello_joint_names
            != self.action_layout.names
        ):
            raise ValueError("SONIC calibration order differs from the Bello model")
        if self.reference_bank.qpos.shape[1:] != (self.model.mj_model.nq,):
            raise ValueError("SONIC reference qpos size differs from the Bello model")
        if self.reference_bank.qvel.shape[1:] != (self.model.mj_model.nv,):
            raise ValueError("SONIC reference qvel size differs from the Bello model")
        required_bodies = set(self.contract.required_body_names)
        missing_reference = required_bodies - set(self.reference_bank.body_names)
        if missing_reference:
            raise ValueError(
                f"SONIC dataset is missing bodies: {sorted(missing_reference)}"
            )
        model_bodies = {
            self.model.mj_model.body(index).name
            for index in range(self.model.mj_model.nbody)
        }
        missing_model = required_bodies - model_bodies
        if missing_model:
            raise ValueError(f"Bello model is missing bodies: {sorted(missing_model)}")
        frames = self.reference_bank.qpos.shape[0]
        expected_first_dimensions = (
            self.reference_bank.qvel,
            self.reference_bank.body_pos,
            self.reference_bank.body_quat_wxyz,
            self.reference_bank.body_lin_vel,
            self.reference_bank.body_ang_vel,
            self.reference_bank.foot_contacts,
            self.reference_bank.smpl_joint_pos,
            self.reference_bank.smpl_root_quat_wxyz,
            self.reference_bank.smpl_root_translation,
            self.reference_bank.g1_joint_pos,
        )
        if any(value.shape[0] != frames for value in expected_first_dimensions):
            raise ValueError("SONIC synchronized frame arrays have drifted")
        if self.reference_bank.smpl_joint_pos.shape[1:] != (24, 3):
            raise ValueError("SONIC SMPL joint array must have shape (frames, 24, 3)")
        if self.reference_bank.g1_joint_pos.shape[1:] != (29,):
            raise ValueError("SONIC G1 joint array must have shape (frames, 29)")
        if self.reference_bank.valid_clip_count != len(self.reference_bank.lengths):
            raise ValueError("SONIC reference-bank valid clip count is invalid")
        valid_lengths = np.asarray(self.reference_bank.lengths)
        if np.any(valid_lengths <= SONIC_REQUIRED_FUTURE_HORIZON):
            raise ValueError("SONIC clips do not cover the required future horizon")
        probabilities = np.asarray(self.reference_bank.sampling_probabilities)
        if probabilities.shape != np.asarray(self.reference_bank.lengths).shape:
            raise ValueError("SONIC clip sampling-probability shape has drifted")
        if (
            np.any(probabilities < 0.0)
            or not np.any(probabilities > 0.0)
            or not np.isclose(np.sum(probabilities), 1.0)
        ):
            raise ValueError("SONIC clip sampling probabilities are invalid")
        if not math.isclose(
            self.control_timestep, 1.0 / SONIC_POLICY_HZ, abs_tol=1e-12
        ):
            raise ValueError("SONIC policy timestep must be 0.02 seconds")
        if self.reference_bank.sampling_bin_size != self.sampling.bin_size:
            raise ValueError("SONIC sampling-bin size differs from the environment")
        bins = build_sonic_sampling_bins(
            valid_lengths,
            bin_size=self.sampling.bin_size,
        )
        for name, value in zip(
            (
                "sampling_bin_clip_indices",
                "sampling_bin_starts",
                "sampling_bin_widths",
                "sampling_bin_valid",
                "sampling_clip_bin_offsets",
            ),
            bins,
            strict=True,
        ):
            configured = np.asarray(getattr(self.reference_bank, name))
            if not np.array_equal(configured, value):
                raise ValueError(f"SONIC {name} differs from the reference bank")

    @classmethod
    def load(
        cls,
        asset_config: AssetPrepConfig,
        simulation: TwistSimulationConfig,
        *,
        motion_path: Path,
        reference_bank: SonicReferenceBank,
        config: SonicEnvConfig = SonicEnvConfig(),
        warp: WarpConfig = WarpConfig(),
        batch_size: int = 1,
    ) -> "BelloSonicEnv":
        domain = config.domain_randomization
        model = load_bello_model(
            asset_config,
            warp=warp,
            batch_size=batch_size,
            terrain=simulation.terrain,
            simulation_timestep=1.0 / SONIC_SIMULATION_HZ,
            model_randomization=ModelRandomizationConfig(
                enabled=domain.enabled,
                seed=domain.seed,
                joint_angle_offset_range=domain.joint_default_position,
                joint_angle_offset_scope="policy",
                added_base_mass_range=domain.added_base_mass,
                base_com_offset_range=domain.base_com,
                friction_range=domain.friction,
                actuator_strength_range=domain.motor_strength,
            ),
        )
        motion = load_motion_clip(motion_path, model)
        env = cls(
            model,
            motion,
            "student",
            config,
            reference_bank,
            config.sampling,
            action_scale=joint_position_action_scales(
                asset_config.actuator_config,
                tuple(
                    model.actuator_joint_names[index]
                    for index in model.policy_ctrl_indices
                ),
            ),
        )
        return (
            env.with_task_state(env.initial_task_state())
            if config.sampling.enabled
            else env
        )

    @property
    def contract(self) -> SonicObservationContract:
        return SonicObservationContract(self.model.actuator_joint_names)

    @property
    def task(self) -> Literal["sonic"]:
        return "sonic"

    @property
    def observation_layout(self) -> ObservationLayout:
        return self.contract.actor_layout

    @property
    def critic_observation_layout(self) -> ObservationLayout:
        return self.contract.critic_layout

    @property
    def physics_steps_per_control(self) -> int:
        return SONIC_DECIMATION

    @property
    def control_timestep(self) -> float:
        return SONIC_DECIMATION * float(self.model.mj_model.opt.timestep)

    def _unchecked_runtime_copy(
        self,
        *,
        reference_bank: SonicReferenceBank | None,
        sampling_bin_cdf: jax.Array | None = None,
    ) -> "BelloSonicEnv":
        bound = object.__new__(BelloSonicEnv)
        for name in (
            "model",
            "motion",
            "stage",
            "config",
            "sampling",
            "action_scale",
        ):
            object.__setattr__(bound, name, getattr(self, name))
        object.__setattr__(bound, "reference_bank", reference_bank)
        object.__setattr__(bound, "sampling_bin_cdf", sampling_bin_cdf)
        return bound

    @property
    def _sampling_bins(
        self,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        if self.reference_bank is None:
            raise RuntimeError("SONIC sampling bins require a reference bank")
        values = (
            self.reference_bank.sampling_bin_clip_indices,
            self.reference_bank.sampling_bin_starts,
            self.reference_bank.sampling_bin_widths,
            self.reference_bank.sampling_bin_valid,
            self.reference_bank.sampling_clip_bin_offsets,
        )
        return values

    def initial_task_state(self) -> SonicSamplingState:
        bins = len(self._sampling_bins[0])
        initial = jnp.full((bins,), self.sampling.initial_failures, dtype=jnp.float32)
        return SonicSamplingState(initial, jnp.ones_like(initial))

    def with_task_state(self, state: SonicSamplingState) -> "BelloSonicEnv":
        if not self.sampling.enabled:
            return self
        bin_clips, _, bin_widths, bin_valid, _ = self._sampling_bins
        if (
            state.failures.shape != bin_clips.shape
            or state.attempts.shape != bin_clips.shape
        ):
            raise ValueError("SONIC adaptive-sampling state shape has drifted")
        assert self.reference_bank is not None
        valid = jnp.asarray(bin_valid, dtype=jnp.float32)
        rates = state.failures / jnp.maximum(state.attempts, 1.0)
        mean = jnp.maximum(jnp.sum(rates * valid) / jnp.sum(valid), 1.0e-6)
        rates = jnp.minimum(rates, mean * self.sampling.failure_rate_max_over_mean)
        clip_mass = jnp.asarray(self.reference_bank.sampling_probabilities)[
            jnp.asarray(bin_clips)
        ]
        clip_lengths = jnp.asarray(self.reference_bank.lengths)[jnp.asarray(bin_clips)]
        valid_starts = clip_lengths - SONIC_REQUIRED_FUTURE_HORIZON
        base = clip_mass * jnp.asarray(bin_widths) / valid_starts * valid
        failure_based = rates * valid
        failure_based /= jnp.sum(failure_based)
        uniform = valid / jnp.sum(valid)
        mixture = (
            base * (1.0 - self.sampling.uniform_sampling_rate) * failure_based
            + base * self.sampling.uniform_sampling_rate * uniform
        )
        mixture /= jnp.sum(mixture)
        cdf = jnp.cumsum(mixture)
        cdf = cdf.at[-1].set(1.0)
        return self._unchecked_runtime_copy(
            reference_bank=self.reference_bank,
            sampling_bin_cdf=cdf,
        )

    def update_task_state(
        self, state: SonicSamplingState, info: dict[str, jax.Array]
    ) -> SonicSamplingState:
        if not self.sampling.enabled:
            return state
        _, _, bin_widths, _, clip_bin_offsets = self._sampling_bins
        failed = info["done/terminal"].astype(bool)
        clip = info["reference/clip_index"].astype(jnp.int32)
        current_frame = info["reference/frame"].astype(jnp.int32)
        local_bins = current_frame // self.sampling.bin_size
        bin_offsets = jnp.asarray(clip_bin_offsets)[clip]
        bin_counts = (
            jnp.asarray(clip_bin_offsets)[clip + 1]
            - jnp.asarray(clip_bin_offsets)[clip]
        )
        valid = (local_bins >= 0) & (local_bins < bin_counts)
        indices = bin_offsets + jnp.maximum(local_bins, 0)
        indices = jnp.clip(indices, 0, state.failures.size - 1)
        widths = jnp.asarray(bin_widths)[indices]
        attempts = (
            jnp.zeros_like(state.attempts)
            .at[indices]
            .add(valid.astype(jnp.float32) / widths)
        )
        failures = (
            jnp.zeros_like(state.failures)
            .at[indices]
            .add((valid & failed).astype(jnp.float32))
        )
        return SonicSamplingState(
            failures=state.failures + failures,
            attempts=state.attempts + attempts,
        )

    def task_state_metrics(self, state: SonicSamplingState) -> dict[str, jax.Array]:
        valid = jnp.asarray(self._sampling_bins[3], dtype=jnp.float32)
        rates = state.failures / jnp.maximum(state.attempts, 1.0)
        return {
            "sampling/failure_rate_mean": jnp.sum(rates * valid) / jnp.sum(valid),
            "sampling/failure_rate_max": jnp.max(
                jnp.where(valid.astype(bool), rates, -jnp.inf)
            ),
            "sampling/attempts": jnp.sum(state.attempts * valid),
        }

    def _sample_clip_start(self, rng: jax.Array | None) -> tuple[jax.Array, jax.Array]:
        bin_clips, starts, widths, bin_valid, _ = self._sampling_bins
        cdf = self.sampling_bin_cdf
        if cdf is None:
            assert self.reference_bank is not None
            clip_mass = jnp.asarray(self.reference_bank.sampling_probabilities)[
                jnp.asarray(bin_clips)
            ]
            clip_lengths = jnp.asarray(self.reference_bank.lengths)[
                jnp.asarray(bin_clips)
            ]
            valid_starts = clip_lengths - SONIC_REQUIRED_FUTURE_HORIZON
            probabilities = (
                clip_mass
                * jnp.asarray(widths)
                / valid_starts
                * jnp.asarray(bin_valid, dtype=jnp.float32)
            )
            probabilities /= jnp.sum(probabilities)
            cdf = jnp.cumsum(probabilities)
            cdf = cdf.at[-1].set(1.0)
        bin_key = frame_key = offset_key = rng
        if rng is not None:
            bin_key, frame_key, offset_key = jax.random.split(rng, 3)
        if bin_key is None:
            bin_index = jnp.searchsorted(cdf, 0.0, side="right").astype(jnp.int32)
            in_bin = jnp.asarray(0, dtype=jnp.int32)
        else:
            uniform = jax.random.uniform(bin_key)
            bin_index = jnp.searchsorted(cdf, uniform, side="right").astype(jnp.int32)
            bin_index = jnp.minimum(bin_index, len(bin_clips) - 1)
            in_bin = jnp.floor(
                jax.random.uniform(frame_key) * jnp.asarray(widths)[bin_index]
            ).astype(jnp.int32)
        clip = jnp.asarray(bin_clips)[bin_index]
        frame = jnp.asarray(starts)[bin_index] + in_bin
        if (
            offset_key is not None
            and self.sampling.enabled
            and self.sampling.pre_failure_window
        ):
            frame -= jax.random.randint(
                offset_key,
                (),
                0,
                self.sampling.pre_failure_window,
                dtype=jnp.int32,
            )
            frame = jnp.maximum(frame, 0)
        return clip, frame.astype(jnp.float32) / SONIC_POLICY_HZ

    def reset(self, rng: jax.Array | None = None) -> tuple[SonicState, jax.Array]:
        state = self._reset_state(rng)
        return state, self.observation(state, rng)

    def _reset_state(self, rng: jax.Array | None = None) -> SonicState:
        keys = (None,) * 6 if rng is None else tuple(jax.random.split(rng, 6))
        clip_key, pose_key, velocity_key, joint_key, push_key, observation_key = keys
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
        motion_state = motion_state.replace(alignment_xy=-source.qpos[:2])
        reference = self._aligned_reference(motion_state, clip_index, 0)
        position_delta = jnp.zeros(3, dtype=jnp.float32)
        rotation_delta = jnp.asarray((1.0, 0.0, 0.0, 0.0), dtype=jnp.float32)
        if pose_key is not None:
            position_key, rotation_key = jax.random.split(pose_key)
            position_delta = jax.random.uniform(
                position_key,
                (3,),
                minval=jnp.asarray((-0.05, -0.05, -0.01)),
                maxval=jnp.asarray((0.05, 0.05, 0.01)),
            )
            euler = jax.random.uniform(
                rotation_key,
                (3,),
                minval=jnp.asarray((-0.1, -0.1, -0.2)),
                maxval=jnp.asarray((0.1, 0.1, 0.2)),
            )
            rotation_delta = quat_from_euler_xyz(euler)
        qpos = reference.qpos.at[:3].add(position_delta)
        qpos = qpos.at[3:7].set(_quat_multiply(reference.qpos[3:7], rotation_delta))
        if joint_key is not None:
            qpos = qpos.at[jnp.asarray(self.model.policy_qpos_indices)].add(
                jax.random.uniform(
                    joint_key, (self.action_size,), minval=-0.1, maxval=0.1
                )
            )
        qvel = reference.qvel
        if velocity_key is not None:
            linear_key, angular_key = jax.random.split(velocity_key)
            qvel = qvel.at[:3].add(
                jax.random.uniform(
                    linear_key,
                    (3,),
                    minval=jnp.asarray((-0.5, -0.5, -0.2)),
                    maxval=jnp.asarray((0.5, 0.5, 0.2)),
                )
            )
            qvel = qvel.at[3:6].add(
                jax.random.uniform(
                    angular_key,
                    (3,),
                    minval=jnp.asarray((-0.52, -0.52, -0.78)),
                    maxval=jnp.asarray((0.52, 0.52, 0.78)),
                )
            )
        qvel = qvel.at[jnp.asarray(self.model.policy_qvel_indices)].set(0.0)
        data = motion_state.data.replace(
            qpos=qpos,
            qvel=qvel,
            ctrl=self._action_target(jnp.zeros(self.action_size, dtype=jnp.float32)),
        )
        data = mjx.forward(self.model.mjx_model, data)
        motion_state = motion_state.replace(data=data)
        interval_low, interval_high = (
            self.config.domain_randomization.push_interval_steps
        )
        next_push = jnp.asarray(interval_high, dtype=jnp.int32)
        if push_key is not None:
            next_push = jax.random.randint(
                push_key, (), interval_low, interval_high + 1, dtype=jnp.int32
            )
        partial = SonicState(
            motion_state=motion_state,
            observation_history=jnp.zeros(
                (SONIC_HISTORY_FRAMES, self.contract.history_frame_size),
                dtype=jnp.float32,
            ),
            critic_history=jnp.zeros(
                (SONIC_HISTORY_FRAMES, self.contract.history_frame_size),
                dtype=jnp.float32,
            ),
            action_history=jnp.zeros((1, self.action_size), dtype=jnp.float32),
            action_delay_steps=jnp.asarray(0, dtype=jnp.int32),
            clip_index=clip_index,
            domain_randomization=self._sample_domain_randomization(),
            foot_air_time=jnp.zeros(2, dtype=jnp.float32),
            last_foot_contact=jnp.zeros(2, dtype=bool),
            total_step=jnp.asarray(0, dtype=jnp.int32),
            gravity_fraction=jnp.asarray(self.model.gravity_fraction[0]),
            running_reference_root_height=reference.qpos[2],
            next_push_step=next_push,
        )
        clean_frame = self._clean_student_frame(partial)
        actor_frame = self._student_frame(partial, observation_key)
        return partial.replace(
            observation_history=jnp.repeat(
                actor_frame[None, :], SONIC_HISTORY_FRAMES, axis=0
            ),
            critic_history=jnp.repeat(
                clean_frame[None, :], SONIC_HISTORY_FRAMES, axis=0
            ),
        )

    def _student_frame(
        self, state: SonicState, rng: jax.Array | None = None
    ) -> jax.Array:
        frame = self._clean_student_frame(state)
        if rng is None or not self.config.observation_noise.enabled:
            return frame
        keys = jax.random.split(jax.random.fold_in(rng, 0x50524F50), 4)
        cfg = self.config.observation_noise
        ramp = jnp.minimum(
            state.total_step.astype(jnp.float32) / max(cfg.ramp_steps, 1),
            1.0,
        )
        gravity = _uniform_noise(frame[:3], ramp * cfg.projected_gravity, keys[0])
        angular = _uniform_noise(frame[6:9], ramp * cfg.base_angular_velocity, keys[1])
        qpos_start = 9
        qvel_start = qpos_start + self.action_size
        action_start = qvel_start + self.action_size
        qpos = _uniform_noise(
            frame[qpos_start:qvel_start], ramp * cfg.joint_position, keys[2]
        )
        qvel = _uniform_noise(
            frame[qvel_start:action_start], ramp * cfg.joint_velocity, keys[3]
        )
        return (
            frame.at[:3]
            .set(gravity)
            .at[6:9]
            .set(angular)
            .at[qpos_start:qvel_start]
            .set(qpos)
            .at[qvel_start:action_start]
            .set(qvel)
        )

    def _clean_student_frame(self, state: SonicState) -> jax.Array:
        data = state.motion_state.data
        root_quat = data.qpos[3:7]
        projected_gravity = quat_rotate_inverse(
            root_quat, jnp.asarray((0.0, 0.0, -1.0), dtype=jnp.float32)
        )
        base_linear_velocity = quat_rotate_inverse(root_quat, data.qvel[:3])
        base_angular_velocity = quat_rotate_inverse(root_quat, data.qvel[3:6])
        policy = jnp.asarray(self.model.policy_ctrl_indices)
        qpos = (
            data.qpos[jnp.asarray(self.model.policy_qpos_indices)]
            - jnp.asarray(self.model.default_joint_target)[policy]
            - state.domain_randomization
        )
        qvel = data.qvel[jnp.asarray(self.model.policy_qvel_indices)]
        return jnp.concatenate(
            (
                projected_gravity,
                base_linear_velocity,
                base_angular_velocity,
                qpos,
                qvel,
                state.motion_state.last_action,
            )
        )

    def observation(self, state: SonicState, rng: jax.Array | None = None) -> jax.Array:
        smpl = self._smpl_encoder_input(state, rng)
        root_motion = self._root_motion_command(state)
        proprioception = self._actor_proprioception(state.observation_history, rng)
        observation = jnp.concatenate((smpl, root_motion, proprioception))
        if observation.shape != (SONIC_ACTOR_OBSERVATION_SIZE,):
            raise RuntimeError("SONIC actor observation shape has drifted")
        return observation

    def observation_batch(
        self, state: SonicState, rng: jax.Array | None = None
    ) -> jax.Array:
        if rng is None:
            return jax.vmap(self.observation)(state)
        keys = jax.random.split(rng, state.clip_index.shape[0])
        return jax.vmap(self.observation)(state, keys)

    def _actor_proprioception(
        self, history: jax.Array, rng: jax.Array | None
    ) -> jax.Array:
        del rng
        gravity = history[:, :3]
        angular = history[:, 6:9]
        qpos = history[:, 9 : 9 + self.action_size]
        qvel = history[:, 9 + self.action_size : 9 + 2 * self.action_size]
        action = history[:, 9 + 2 * self.action_size :]
        value = jnp.concatenate(
            (
                gravity.reshape(-1),
                angular.reshape(-1),
                qpos.reshape(-1),
                qvel.reshape(-1),
                action.reshape(-1),
            )
        )
        if value.shape != (SONIC_BELLO_PROPRIO_SIZE,):
            raise RuntimeError("SONIC Bello proprioception shape has drifted")
        return value

    def _root_motion_command(self, state: SonicState) -> jax.Array:
        """Encode the short desired XY path in the current robot root frame."""

        current_reference = self._aligned_reference(
            state.motion_state,
            state.clip_index,
            SONIC_SMPL_FUTURE_OFFSETS[0],
        )
        current_root_quat = state.motion_state.data.qpos[3:7]
        value = jnp.concatenate(
            tuple(
                quat_rotate_inverse(
                    current_root_quat,
                    self._aligned_reference(
                        state.motion_state,
                        state.clip_index,
                        offset,
                    ).qpos[:3]
                    - current_reference.qpos[:3],
                )[:2]
                for offset in SONIC_SMPL_FUTURE_OFFSETS
            )
        )
        if value.shape != (SONIC_ROOT_MOTION_COMMAND_SIZE,):
            raise RuntimeError("SONIC root-motion command shape has drifted")
        return value

    def _smpl_encoder_input(
        self, state: SonicState, rng: jax.Array | None = None
    ) -> jax.Array:
        assert self.reference_bank is not None
        basis = jnp.asarray(BELLO_TO_G1_VECTOR_BASIS)
        current_rotation = _quat_rotation_matrix(state.motion_state.data.qpos[3:7])
        alignment_rotation = _quat_rotation_matrix(
            yaw_quat_from_angle(state.motion_state.alignment_yaw)
        )
        joint_noise = orientation_noise = None
        if rng is not None and self.config.observation_noise.enabled:
            noise_key = jax.random.fold_in(rng, 0x534D504C)
            joint_key, orientation_key = jax.random.split(noise_key)
            ramp = jnp.minimum(
                state.total_step.astype(jnp.float32)
                / max(self.config.observation_noise.ramp_steps, 1),
                1.0,
            )
            joint_noise = _uniform_noise(
                jnp.zeros((len(SONIC_SMPL_FUTURE_OFFSETS), 24, 3)),
                ramp * self.config.observation_noise.smpl_joint_position,
                joint_key,
            )
            orientation_noise = _uniform_noise(
                jnp.zeros((len(SONIC_SMPL_FUTURE_OFFSETS), 6)),
                ramp * self.config.observation_noise.smpl_root_orientation_6d,
                orientation_key,
            )

        def frame(frame_index: int, offset: int) -> jax.Array:
            absolute = self._absolute_reference_frame(
                state.motion_state, state.clip_index, offset
            )
            joints = jnp.asarray(self.reference_bank.smpl_joint_pos)[absolute]
            root_quat = jnp.asarray(self.reference_bank.smpl_root_quat_wxyz)[absolute]
            relative_matrix = (
                basis
                @ current_rotation.T
                @ alignment_rotation
                @ _quat_rotation_matrix(root_quat)
            )
            orientation = relative_matrix[:, :2].reshape(-1)
            if joint_noise is not None and orientation_noise is not None:
                joints = joints + joint_noise[frame_index]
                orientation = orientation + orientation_noise[frame_index]
            wrists = jnp.asarray(self.reference_bank.g1_joint_pos)[absolute, 23:29]
            return jnp.concatenate((joints.reshape(-1), orientation, wrists))

        value = jnp.concatenate(
            tuple(
                frame(frame_index, offset)
                for frame_index, offset in enumerate(SONIC_SMPL_FUTURE_OFFSETS)
            )
        )
        if value.shape != (SONIC_SMPL_INPUT_SIZE,):
            raise RuntimeError("SONIC SMPL encoder input shape has drifted")
        return value

    def critic_observation(self, state: SonicState) -> jax.Array:
        current = state.motion_state
        future_references = tuple(
            self._aligned_reference(current, state.clip_index, offset)
            for offset in SONIC_CRITIC_FUTURE_OFFSETS
        )
        policy_qpos = jnp.asarray(self.model.policy_qpos_indices)
        policy_qvel = jnp.asarray(self.model.policy_qvel_indices)
        future_joint_positions = jnp.concatenate(
            tuple(reference.qpos[policy_qpos] for reference in future_references)
        )
        future_joint_velocities = jnp.concatenate(
            tuple(reference.qvel[policy_qvel] for reference in future_references)
        )
        reference = self._aligned_reference(current, state.clip_index, 0)
        root_quat = current.data.qpos[3:7]
        anchor_position = quat_rotate_inverse(
            root_quat, reference.qpos[:3] - current.data.qpos[:3]
        )
        anchor_orientation = rotation_6d(
            _quat_multiply(_quat_conjugate(root_quat), reference.qpos[3:7])
        )
        body_ids = jnp.asarray(self.key_body_ids)
        body_positions = jax.vmap(
            lambda value: quat_rotate_inverse(root_quat, value - current.data.qpos[:3])
        )(current.data.xpos[body_ids]).reshape(-1)
        body_orientations = jax.vmap(
            lambda value: rotation_6d(_quat_multiply(_quat_conjugate(root_quat), value))
        )(current.data.xquat[body_ids]).reshape(-1)
        history = state.critic_history
        raw_history = jnp.concatenate(
            (
                history[:, 3:6].reshape(-1),
                history[:, 6:9].reshape(-1),
                history[:, 9 : 9 + self.action_size].reshape(-1),
                history[:, 9 + self.action_size : 9 + 2 * self.action_size].reshape(-1),
                history[:, 9 + 2 * self.action_size :].reshape(-1),
            )
        )
        value = jnp.concatenate(
            (
                future_joint_positions,
                future_joint_velocities,
                anchor_position,
                anchor_orientation,
                body_positions,
                body_orientations,
                raw_history,
            )
        )
        if value.shape != (self.contract.critic_observation_size,):
            raise RuntimeError("SONIC critic observation shape has drifted")
        return value

    def critic_observation_batch(
        self, state: SonicState, rng: jax.Array | None = None
    ) -> jax.Array:
        del rng
        return jax.vmap(self.critic_observation)(state)

    def critic_observation_with_world(
        self, state: SonicState, world_id: jax.Array
    ) -> jax.Array:
        del world_id
        return self.critic_observation(state)

    def _twist_reward(
        self,
        previous: SonicState,
        current,
        reference: MotionSample,
        foot_contact: jax.Array,
        foot_air_time: jax.Array,
        world_id: jax.Array | int,
        *,
        contact_normal: jax.Array | None = None,
        contact_tangent: jax.Array | None = None,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        del foot_air_time, contact_normal, contact_tangent
        cfg = self.config.reward
        body_ids = jnp.asarray(self.key_body_ids)
        reference_indices = jnp.asarray(
            [self.motion.body_names.index(name) for name in self.contract.body_names]
        )
        reference_positions = reference.body_pos[reference_indices]
        current_positions = current.data.xpos[body_ids]
        reference_quats = reference.body_quat[reference_indices]
        current_quats = current.data.xquat[body_ids]
        reference_relative_positions, reference_relative_quats = (
            _heading_aligned_reference_pose(
                reference.qpos[:3],
                reference.qpos[3:7],
                reference_positions,
                reference_quats,
                current.data.qpos[:3],
                current.data.qpos[3:7],
            )
        )
        body_orientation_error = jax.vmap(_quat_distance)(
            reference_relative_quats, current_quats
        )
        body_linear_velocity = self._body_linear_velocity(current, body_ids)
        body_angular_velocity = current.data.cvel[body_ids, :3]
        five_model_ids = jnp.asarray(
            [self._body_id(name) for name in SONIC_REWARD_POINT_BODY_NAMES]
        )
        five_reference_ids = jnp.asarray(
            [
                self.motion.body_names.index(name)
                for name in SONIC_REWARD_POINT_BODY_NAMES
            ]
        )
        reference_five = jax.vmap(
            lambda value: quat_rotate_inverse(
                reference.qpos[3:7], value - reference.qpos[:3]
            )
        )(reference.body_pos[five_reference_ids])
        current_five = jax.vmap(
            lambda value: quat_rotate_inverse(
                current.data.qpos[3:7], value - current.data.qpos[:3]
            )
        )(current.data.xpos[five_model_ids])
        policy_qpos = jnp.asarray(self.model.policy_qpos_indices)
        policy_qvel = jnp.asarray(self.model.policy_qvel_indices)
        positions = current.data.qpos[policy_qpos]
        lower = jnp.asarray(self.model.joint_target_low)[
            jnp.asarray(self.model.policy_ctrl_indices)
        ]
        upper = jnp.asarray(self.model.joint_target_high)[
            jnp.asarray(self.model.policy_ctrl_indices)
        ]
        acceleration = (
            current.data.qvel[policy_qvel]
            - previous.motion_state.data.qvel[policy_qvel]
        ) / self.control_timestep
        ankle = jnp.asarray(
            [
                index
                for index, name in enumerate(self.action_layout.names)
                if "ankle" in name
            ]
        )
        anti_shake_ids = jnp.asarray(
            [self._body_id(name) for name in SONIC_ANTI_SHAKE_BODY_NAMES]
        )
        anti_shake_speed = jnp.linalg.norm(
            current.data.cvel[anti_shake_ids, :3], axis=-1
        )
        foot_velocity = self._body_linear_velocity(
            current, jnp.asarray(self.model.foot_body_ids)
        )
        foot_slip = jnp.mean(
            jnp.linalg.norm(foot_velocity[:, :2], axis=-1)
            * foot_contact.astype(jnp.float32)
        )
        requested_target = self._unbounded_policy_target(
            current.last_action,
            previous.domain_randomization,
        )
        target_limit_excess = (
            jax.nn.relu(lower - requested_target)
            + jax.nn.relu(requested_target - upper)
        ) / jnp.asarray(self.action_scale, dtype=jnp.float32)
        raw = {
            "tracking_anchor_position": jnp.exp(
                -jnp.sum((reference.qpos[:3] - current.data.qpos[:3]) ** 2)
                / cfg.anchor_position_std**2
            ),
            "tracking_anchor_orientation": jnp.exp(
                -(_quat_distance(reference.qpos[3:7], current.data.qpos[3:7]) ** 2)
                / cfg.anchor_orientation_std**2
            ),
            "tracking_relative_body_position": jnp.exp(
                -jnp.mean(
                    jnp.sum(
                        (reference_relative_positions - current_positions) ** 2,
                        axis=-1,
                    )
                )
                / cfg.relative_body_position_std**2
            ),
            "tracking_relative_body_orientation": jnp.exp(
                -jnp.mean(body_orientation_error**2)
                / cfg.relative_body_orientation_std**2
            ),
            "tracking_body_linear_velocity": jnp.exp(
                -jnp.mean(
                    jnp.sum(
                        (
                            reference.body_lin_vel[reference_indices]
                            - body_linear_velocity
                        )
                        ** 2,
                        axis=-1,
                    )
                )
                / cfg.body_linear_velocity_std**2
            ),
            "tracking_body_angular_velocity": jnp.exp(
                -jnp.mean(
                    jnp.sum(
                        (
                            reference.body_ang_vel[reference_indices]
                            - body_angular_velocity
                        )
                        ** 2,
                        axis=-1,
                    )
                )
                / cfg.body_angular_velocity_std**2
            ),
            "tracking_joint_position": jnp.exp(
                -jnp.mean((reference.qpos[policy_qpos] - positions) ** 2)
                / cfg.joint_position_std**2
            ),
            "tracking_foot_contact": 1.0
            - jnp.mean(
                jnp.abs(
                    foot_contact.astype(jnp.float32)
                    - reference.foot_contacts.astype(jnp.float32)
                )
            ),
            "action_rate": jnp.sum(
                (current.last_action - previous.motion_state.last_action) ** 2
            ),
            "joint_limit": jnp.sum(
                jax.nn.relu(lower - positions) + jax.nn.relu(positions - upper)
            ),
            "target_joint_limit": jnp.mean(target_limit_excess**2),
            "undesired_contacts": self._undesired_contact(current, world_id).astype(
                jnp.float32
            ),
            "anti_shake_angular_velocity": jnp.mean(
                jax.nn.relu(anti_shake_speed - cfg.anti_shake_threshold) ** 2
            ),
            "tracking_five_point_local": jnp.exp(
                -jnp.mean(jnp.sum((reference_five - current_five) ** 2, axis=-1))
                / cfg.five_point_std**2
            ),
            "feet_acceleration": jnp.sum(acceleration[ankle] ** 2),
        }
        components = {
            f"reward/{name}": self.control_timestep * getattr(cfg, name) * value
            for name, value in raw.items()
        }
        metrics = {
            "tracking/root_position_error_m": jnp.linalg.norm(
                reference.qpos[:3] - current.data.qpos[:3]
            ),
            "tracking/root_orientation_error_rad": _quat_distance(
                reference.qpos[3:7], current.data.qpos[3:7]
            ),
            "tracking/joint_position_rms_rad": jnp.sqrt(
                jnp.mean((reference.qpos[policy_qpos] - positions) ** 2)
            ),
            "tracking/foot_slip_mean_mps": foot_slip,
            "target/joint_limit_excess_rms_action": jnp.sqrt(
                jnp.mean(target_limit_excess**2)
            ),
        }
        return (
            sum(components.values(), jnp.asarray(0.0, dtype=jnp.float32)),
            {**components, **metrics},
        )

    def _sanitize_reward(self, reward, components, done_flags):
        invalid = done_flags["done/nonfinite"] | ~jnp.isfinite(reward)
        sanitized = {
            name: jnp.where(invalid, 0.0, jnp.nan_to_num(value))
            for name, value in components.items()
        }
        reward_values = (
            value for name, value in sanitized.items() if name.startswith("reward/")
        )
        return sum(reward_values, jnp.zeros_like(reward)), sanitized

    def _bank_termination_flags(
        self,
        state,
        clip_index: jax.Array,
        reference: MotionSample,
        world_id: jax.Array | int,
        *,
        torso_contact: jax.Array | None = None,
        tracking_state: SonicState | None = None,
    ) -> dict[str, jax.Array]:
        del world_id, torso_contact
        if tracking_state is None:
            raise ValueError("SONIC termination requires its full tracking state")
        cfg = self.config.done
        running_reference_root_height = (
            0.1 * reference.qpos[2] + 0.9 * tracking_state.running_reference_root_height
        )
        low_reference = running_reference_root_height < cfg.low_reference_root_height
        root_threshold = jnp.where(
            low_reference, cfg.root_height_error_when_low, cfg.root_height_error
        )
        root_height = jnp.abs(reference.qpos[2] - state.data.qpos[2]) > root_threshold
        orientation = (
            _quat_distance(reference.qpos[3:7], state.data.qpos[3:7]) ** 2
            > cfg.root_orientation_squared_error
        )
        reference_end_ids = jnp.asarray(
            [
                self.motion.body_names.index(name)
                for name in SONIC_END_EFFECTOR_BODY_NAMES
            ]
        )
        model_end_ids = jnp.asarray(
            [self._body_id(name) for name in SONIC_END_EFFECTOR_BODY_NAMES]
        )
        shifted_reference, _ = _heading_aligned_reference_pose(
            reference.qpos[:3],
            reference.qpos[3:7],
            reference.body_pos[reference_end_ids],
            reference.body_quat[reference_end_ids],
            state.data.qpos[:3],
            state.data.qpos[3:7],
        )
        end_threshold = jnp.where(
            low_reference,
            cfg.end_effector_height_error_when_low,
            cfg.end_effector_height_error,
        )
        end_effector = jnp.any(
            jnp.abs(shifted_reference[:, 2] - state.data.xpos[model_end_ids, 2])
            > end_threshold
        )
        foot_reference = shifted_reference[:2]
        foot_current = state.data.xpos[model_end_ids[:2]]
        foot_position = jnp.any(
            jnp.linalg.norm(foot_reference - foot_current, axis=-1)
            > cfg.foot_position_error
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
        frame = jnp.rint(state.reference_time * SONIC_POLICY_HZ).astype(jnp.int32)
        assert self.reference_bank is not None
        last_frame = jnp.asarray(self.reference_bank.lengths)[clip_index] - 1
        timeout = (state.step_count >= self.config.max_episode_steps) | (
            frame >= last_frame
        )
        terminal = root_height | orientation | end_effector | foot_position | nonfinite
        false = jnp.asarray(False)
        return {
            "done/root_height": root_height,
            "done/orientation": orientation,
            "done/end_effector_height": end_effector,
            "done/foot_position": foot_position,
            "done/nonfinite": nonfinite,
            "done/timeout": timeout,
            "done/sonic_timeout": state.step_count >= self.config.max_episode_steps,
            "done/terminal": terminal,
            "done/any": terminal | timeout,
            "done/height": root_height,
            "done/pose": end_effector | foot_position,
            "done/root_velocity": false,
            "done/torso_contact": false,
        }

    def _bank_info(self, state, observation, reward_info, done_flags):
        info = super()._bank_info(state, observation, reward_info, done_flags)
        info["reference/frame"] = jnp.rint(
            state.motion_state.reference_time * SONIC_POLICY_HZ
        ).astype(jnp.float32)
        info["total_step"] = state.total_step.astype(jnp.float32)
        return info

    def _absolute_reference_frame(
        self, state, clip_index: jax.Array, offset: int
    ) -> jax.Array:
        assert self.reference_bank is not None
        frame = jnp.rint(state.reference_time * SONIC_POLICY_HZ).astype(jnp.int32)
        length = jnp.asarray(self.reference_bank.lengths)[clip_index]
        frame = jnp.clip(frame + offset, 0, length - 1)
        return jnp.asarray(self.reference_bank.offsets)[clip_index] + frame

    def _aligned_reference(
        self, state, clip_index: jax.Array, offset: int
    ) -> MotionSample:
        assert self.reference_bank is not None
        absolute = self._absolute_reference_frame(state, clip_index, offset)

        def value(name: str) -> jax.Array:
            return jnp.asarray(getattr(self.reference_bank, name))[absolute]

        qpos = value("qpos")
        qvel = value("qvel")
        body_pos = value("body_pos")
        body_quat = value("body_quat_wxyz")
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
                value("body_lin_vel")
            ),
            body_ang_vel=jax.vmap(lambda item: quat_rotate(rotation, item))(
                value("body_ang_vel")
            ),
            foot_contacts=value("foot_contacts"),
        )

    def _undesired_contact(self, state, world_id: jax.Array | int) -> jax.Array:
        contact_world, active, bodies, normal, _ = self._contact_force_data(state)
        allowed = jnp.asarray(
            [self._body_id(name) for name in SONIC_ALLOWED_CONTACT_BODY_NAMES]
        )
        robot = jnp.asarray(self.model.robot_geom_ids)
        geom_body_ids = jnp.asarray(self.model.geom_body_ids)[robot]
        robot_contact = jnp.any(bodies[..., None] == geom_body_ids, axis=(-1, -2))
        allowed_contact = jnp.any(bodies[..., None] == allowed, axis=(-1, -2))
        return jnp.any(
            active
            & (contact_world == jnp.asarray(world_id))
            & robot_contact
            & ~allowed_contact
            & (normal > self.config.reward.undesired_contact_threshold)
        )

    def _sample_action_delay(self, rng: jax.Array | None) -> jax.Array:
        del rng
        return jnp.asarray(0, dtype=jnp.int32)

    def _sample_action_delay_batch(
        self, state: SonicState, rng: jax.Array | None
    ) -> jax.Array:
        del state, rng
        return jnp.asarray(0, dtype=jnp.int32)

    def _sample_domain_randomization(self) -> jax.Array:
        """Return the per-world default-joint offset used by SONIC actions."""

        return jnp.asarray(self.model.joint_angle_offset[0], dtype=jnp.float32)

    def reset_batch(
        self, batch_size: int, rng: jax.Array | None = None
    ) -> tuple[SonicState, jax.Array]:
        state, observation = super().reset_batch(batch_size, rng)
        offsets = jnp.asarray(self.model.joint_angle_offset, dtype=jnp.float32)
        if offsets.shape[0] == 1:
            offsets = jnp.repeat(offsets, batch_size, axis=0)
        if offsets.shape != (batch_size, self.action_size):
            raise ValueError("SONIC default-joint randomization shape has drifted")
        state = state.replace(domain_randomization=offsets)
        controls = jax.vmap(self._action_target)(
            jnp.zeros((batch_size, self.action_size), dtype=jnp.float32),
            state.motion_state,
            offsets,
        )
        state = state.replace(
            motion_state=state.motion_state.replace(
                data=state.motion_state.data.replace(ctrl=controls)
            )
        )
        clean_frames = jax.vmap(self._clean_student_frame)(state)
        if rng is None:
            actor_frames = clean_frames
        else:
            history_key = jax.random.fold_in(rng, 0x534F4E49)
            actor_frames = jax.vmap(self._student_frame)(
                state, jax.random.split(history_key, batch_size)
            )
        state = state.replace(
            observation_history=jnp.repeat(
                actor_frames[:, None, :], SONIC_HISTORY_FRAMES, axis=1
            ),
            critic_history=jnp.repeat(
                clean_frames[:, None, :], SONIC_HISTORY_FRAMES, axis=1
            ),
        )
        observation = self.observation_batch(state, rng)
        return state, observation

    def ppo_action_diagnostics(self, env_action: jax.Array) -> dict[str, jax.Array]:
        flat_action = env_action.reshape((-1, self.action_size))
        requested = jax.vmap(self._unbounded_policy_target)(flat_action)
        policy = jnp.asarray(self.model.policy_ctrl_indices)
        low = jnp.asarray(self.model.joint_target_low)[policy]
        high = jnp.asarray(self.model.joint_target_high)[policy]
        outside = (requested < low) | (requested > high)
        diagnostics = {
            "target/outside_joint_limit_fraction": jnp.mean(outside.astype(jnp.float32))
        }
        diagnostics.update(
            {
                f"target/outside_joint_limit/{name}_fraction": jnp.mean(
                    outside[:, index].astype(jnp.float32)
                )
                for index, name in enumerate(self.action_layout.names)
            }
        )
        return diagnostics

    def _unbounded_policy_target(
        self,
        action: jax.Array,
        randomization: jax.Array | None = None,
    ) -> jax.Array:
        policy = jnp.asarray(self.model.policy_ctrl_indices)
        offset = (
            jnp.asarray(self.model.joint_angle_offset[0], dtype=jnp.float32)
            if randomization is None
            else jnp.asarray(randomization, dtype=jnp.float32)
        )
        if offset.shape != (self.action_size,):
            raise ValueError("SONIC default-joint offset shape has drifted")
        return (
            jnp.asarray(self.model.default_joint_target)[policy]
            + offset
            + jnp.asarray(self.action_scale, dtype=jnp.float32)
            * jnp.asarray(action, dtype=jnp.float32)
        )

    def _action_target(
        self,
        action: jax.Array,
        motion_state=None,
        randomization: jax.Array | None = None,
    ) -> jax.Array:
        """Apply scale-aware SONIC actions and bound every requested joint target."""

        del motion_state
        policy = jnp.asarray(self.model.policy_ctrl_indices)
        logical = jnp.asarray(self.model.default_joint_target)
        target = jnp.clip(
            self._unbounded_policy_target(action, randomization),
            jnp.asarray(self.model.joint_target_low)[policy],
            jnp.asarray(self.model.joint_target_high)[policy],
        )
        logical = logical.at[policy].set(target)
        return jnp.matmul(
            jnp.asarray(self.model.actuator_coordinate_transform),
            logical,
            precision=jax.lax.Precision.HIGHEST,
        )

    def _termination_torso_contacts_batch(self, state, batch_size: int) -> jax.Array:
        del state
        return jnp.zeros((batch_size,), dtype=bool)

    def _apply_disturbances_batch(self, data, state: SonicState, rng):
        batch_size = int(state.motion_state.step_count.shape[0])
        cfg = self.config.domain_randomization
        if rng is None or not cfg.enabled:
            return data.replace(
                xfrc_applied=jnp.zeros_like(data.xfrc_applied)
            ), jnp.zeros((batch_size,), dtype=bool)
        due = (state.total_step + 1 >= state.next_push_step) & (
            state.total_step + 1 >= cfg.push_warmup_steps
        )
        linear_key, angular_key = jax.random.split(rng)
        linear = jax.random.uniform(
            linear_key,
            (batch_size, 3),
            minval=jnp.asarray(
                (
                    -cfg.push_linear_velocity_xy,
                    -cfg.push_linear_velocity_xy,
                    -cfg.push_linear_velocity_z,
                )
            ),
            maxval=jnp.asarray(
                (
                    cfg.push_linear_velocity_xy,
                    cfg.push_linear_velocity_xy,
                    cfg.push_linear_velocity_z,
                )
            ),
        )
        angular = jax.random.uniform(
            angular_key,
            (batch_size, 3),
            minval=jnp.asarray(
                (
                    -cfg.push_angular_velocity_roll_pitch,
                    -cfg.push_angular_velocity_roll_pitch,
                    -cfg.push_angular_velocity_yaw,
                )
            ),
            maxval=jnp.asarray(
                (
                    cfg.push_angular_velocity_roll_pitch,
                    cfg.push_angular_velocity_roll_pitch,
                    cfg.push_angular_velocity_yaw,
                )
            ),
        )
        qvel = data.qvel.at[:, :3].set(
            jnp.where(due[:, None], linear, data.qvel[:, :3])
        )
        qvel = qvel.at[:, 3:6].set(jnp.where(due[:, None], angular, data.qvel[:, 3:6]))
        return data.replace(
            qvel=qvel, xfrc_applied=jnp.zeros_like(data.xfrc_applied)
        ), due

    def _post_step_state(self, state: SonicState, push_applied: jax.Array, rng):
        reference_height = (
            self._aligned_reference(state.motion_state, state.clip_index, 0).qpos[2]
            if state.clip_index.ndim == 0
            else jax.vmap(
                lambda motion_state, clip_index: self._aligned_reference(
                    motion_state, clip_index, 0
                ).qpos[2]
            )(state.motion_state, state.clip_index)
        )
        clean_frame = (
            self._clean_student_frame(state)
            if state.clip_index.ndim == 0
            else jax.vmap(self._clean_student_frame)(state)
        )
        critic_history = (
            jnp.concatenate((state.critic_history[1:], clean_frame[None, :]), axis=0)
            if state.clip_index.ndim == 0
            else jnp.concatenate(
                (state.critic_history[:, 1:], clean_frame[:, None, :]), axis=1
            )
        )
        state = state.replace(
            running_reference_root_height=(
                0.1 * reference_height + 0.9 * state.running_reference_root_height
            ),
            critic_history=critic_history,
        )
        if rng is None or not self.config.domain_randomization.enabled:
            return state
        interval_key = jax.random.split(rng, 3)[2]
        low, high = self.config.domain_randomization.push_interval_steps
        intervals = jax.random.randint(
            interval_key, push_applied.shape, low, high + 1, dtype=jnp.int32
        )
        next_push = jnp.where(
            push_applied, state.total_step + intervals, state.next_push_step
        )
        return state.replace(next_push_step=next_push)

    def _post_reset_state(
        self, reset_state: SonicState, previous_state: SonicState
    ) -> SonicState:
        return reset_state.replace(
            next_push_step=previous_state.total_step + reset_state.next_push_step
        )


def default_sonic_asset_config() -> AssetPrepConfig:
    return default_twist_asset_config()


def _build_sampling_bins(
    lengths: np.ndarray, bin_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    clip_indices: list[int] = []
    starts: list[int] = []
    widths: list[int] = []
    clip_bin_offsets = [0]
    for clip_index, length in enumerate(np.asarray(lengths).tolist()):
        valid_frames = int(length) - SONIC_REQUIRED_FUTURE_HORIZON
        for start in range(0, valid_frames, bin_size):
            clip_indices.append(clip_index)
            starts.append(start)
            widths.append(min(bin_size, valid_frames - start))
        clip_bin_offsets.append(len(starts))
    if not starts:
        raise ValueError("SONIC reference bank has no valid sampling bins")
    return (
        np.asarray(clip_indices, dtype=np.int32),
        np.asarray(starts, dtype=np.int32),
        np.asarray(widths, dtype=np.int32),
        np.asarray(clip_bin_offsets, dtype=np.int32),
    )


def _quat_conjugate(quaternion: jax.Array) -> jax.Array:
    return quaternion * jnp.asarray((1.0, -1.0, -1.0, -1.0))


def _quat_rotation_matrix(quaternion: jax.Array) -> jax.Array:
    w, x, y, z = quaternion / jnp.maximum(jnp.linalg.norm(quaternion), 1.0e-8)
    return jnp.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        )
    )


def _heading_aligned_reference_pose(
    reference_root_position: jax.Array,
    reference_root_quaternion: jax.Array,
    reference_positions: jax.Array,
    reference_quaternions: jax.Array,
    robot_root_position: jax.Array,
    robot_root_quaternion: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Match SONIC's reference pose to robot XY and heading only."""

    heading_delta = yaw_quat(
        _quat_multiply(
            robot_root_quaternion, _quat_conjugate(reference_root_quaternion)
        )
    )
    origin = robot_root_position.at[2].set(reference_root_position[2])
    positions = origin + jax.vmap(
        lambda value: quat_rotate(heading_delta, value - reference_root_position)
    )(reference_positions)
    quaternions = jax.vmap(lambda value: _quat_multiply(heading_delta, value))(
        reference_quaternions
    )
    return positions, quaternions
