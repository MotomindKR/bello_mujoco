"""Corpus motion collection sharing the canonical Mini physics and feedback path."""

from dataclasses import dataclass, replace
import math

import jax
import jax.numpy as jnp

from bello_mujoco.contracts import ActionLayout, ObservationLayout
from bello_mujoco.envs.motion_tracking import BelloMotionTrackingEnv
from bello_mujoco.motion.clip import MotionClip
from bello_mujoco.robot_tasks import motion_task_definition
from bello_mujoco.sim import WarpConfig, load_model
from bello_mujoco.sonic.reference import ReferenceBank, load_reference_bank
from bello_mujoco.supermimic import rotations as rot
from bello_mujoco.supermimic.config import ControlConfig, ObservationConfig
from bello_mujoco.supermimic.control import ActionMapping
from bello_mujoco.supermimic.feedback import (
    actor_sample,
    advance_feedback,
    measured_feedback,
)
from bello_mujoco.supermimic.kinematics import Kinematics
from bello_mujoco.supermimic.mdp import TrackingConfig, TrackingMDP, critic_features
from bello_mujoco.supermimic.observations import ObservationCodec
from bello_mujoco.supermimic.sampling import ReturnSampler, SamplingConfig
from bello_mujoco.supermimic.state import BodyState, PhysicalState


def validate_frame_clock(bank, control_timestep):
    step = bank.fps * control_timestep
    if not math.log2(step).is_integer():
        raise ValueError(
            "SuperMimic clips must use a power-of-two multiple of the control rate"
        )
    if bank.clock_end >= 2**24 * min(1.0, step):
        raise ValueError("SuperMimic corpus exceeds exact float32 frame-clock capacity")


def align_reference(state, xy, yaw):
    orientation = rot.exp(
        jnp.stack((jnp.zeros_like(yaw), jnp.zeros_like(yaw), yaw), axis=-1)
    )
    shift = jnp.concatenate((xy, jnp.zeros_like(xy[..., :1])), axis=-1)
    return replace(
        state,
        position=rot.rotate(orientation, state.position) + shift,
        quaternion=rot.multiply(orientation, state.quaternion),
        linear_velocity=rot.rotate(orientation, state.linear_velocity),
        angular_velocity=rot.rotate(orientation, state.angular_velocity),
    )


@dataclass(frozen=True)
class BelloSuperMimicEnv(BelloMotionTrackingEnv):
    reference_bank: ReferenceBank | None = None
    kinematics: Kinematics | None = None
    codec: ObservationCodec | None = None
    mapping: ActionMapping | None = None
    mdp: TrackingMDP | None = None
    sampler: ReturnSampler | None = None
    reset_probabilities: jax.Array | None = None

    def __post_init__(self):
        super().__post_init__()
        if any(
            getattr(self, name) is None
            for name in (
                "reference_bank",
                "kinematics",
                "codec",
                "mapping",
                "mdp",
                "sampler",
                "reset_probabilities",
            )
        ):
            raise ValueError(
                "SuperMimic requires corpus, kinematics, codec, mapping, MDP, and sampler"
            )
        if (
            self.model.robot.asset != "bello_mini"
            or self.config.anchor_body_name != "pelvis_link"
        ):
            raise ValueError("SuperMimic requires the canonical Mini pelvis reference")
        if (
            self.reference_bank.body_names
            != motion_task_definition("bello_mini").beyondmimic_tracked_body_names
        ):
            raise ValueError(
                "SuperMimic requires the canonical Mini tracked-body order"
            )
        if self.reference_bank.joint_names != self.model.actuator_joint_names:
            raise ValueError("SuperMimic corpus joint order differs from the model")
        validate_frame_clock(self.reference_bank, self.control_timestep)
        if self.config.push.enabled or self.config.adaptive_sampling.enabled:
            raise ValueError(
                "SuperMimic uses explicit context and its return sampler; disable pushes and legacy adaptive sampling"
            )
        cfg = self.config.model_randomization
        if cfg.enabled and any(
            getattr(cfg, name) != bounds
            for name, bounds in (
                ("joint_origin_range", (0.0, 0.0)),
                ("joint_inertia_scale_range", (1.0, 1.0)),
                ("torso_mass_scale_range", (1.0, 1.0)),
                ("added_base_mass_range", (0.0, 0.0)),
                ("base_com_offset_range", (0.0, 0.0)),
                ("actuator_strength_range", (1.0, 1.0)),
                ("gravity_fraction_range", (0.0, 0.0)),
            )
        ):
            raise ValueError(
                "SuperMimic context currently supports friction, calibration, and torso COM randomization"
            )
        if self.config.observation_mode != "proprioceptive":
            raise ValueError("SuperMimic actor uses proprioceptive feedback")
        if self.config.command_delay_seconds != 0:
            raise ValueError(
                "SuperMimic raw-action dynamics requires zero command delay"
            )
        if self.model.mj_model.actuator_ctrllimited.any():
            raise ValueError("SuperMimic requires unprojected PD actuator setpoints")

    @classmethod
    def load_corpus(
        cls,
        asset_config,
        env_config,
        paths,
        *,
        evaluation_names=(),
        observation_config=ObservationConfig(),
        tracking_config=TrackingConfig(),
        sampling_config=SamplingConfig(),
        warp=WarpConfig(),
        batch_size=1,
        use_viewer_model=False,
        load_mjx=True,
    ):
        if load_mjx and env_config.observation_low_pass_filter.enabled:
            raise ValueError(
                "filtered SuperMimic MJX-Warp training is not validated; use the unfiltered reference profile"
            )
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
        bank = load_reference_bank(paths, model, evaluation_names=evaluation_names)
        kin = Kinematics.from_model(
            model.mj_model,
            model.actuator_joint_names,
            bank.body_names,
            root_body_name=env_config.anchor_body_name,
        )
        codec = ObservationCodec(
            observation_config, jnp.asarray(model.default_joint_target)
        )
        control = ControlConfig(
            float(model.mj_model.opt.timestep),
            env_config.decimation,
            env_config.command_delay_seconds,
            env_config.action_clip,
        )
        mapping = ActionMapping.from_model(model, asset_config.actuator_config)
        mdp = TrackingMDP(
            kin,
            jnp.asarray(model.target_limits.joint[:, 0]),
            jnp.asarray(model.target_limits.joint[:, 1]),
            control.timestep,
            tracking_config,
        )
        first = bank.sample_frames(jnp.asarray(0), jnp.arange(int(bank.lengths[0])))
        motion = MotionClip(
            bank.fps,
            bank.joint_names,
            bank.body_names,
            first.qpos,
            first.qvel,
            first.body_pos,
            first.body_quat,
            first.body_lin_vel,
            first.body_ang_vel,
            first.foot_contacts,
        )
        sampler = ReturnSampler.from_bank(bank, sampling_config)
        probabilities = sampler.probabilities(sampler.initial_state(), 1.0)
        return cls(
            model,
            motion,
            asset_config.actuator_config,
            env_config,
            bank,
            kin,
            codec,
            mapping,
            mdp,
            sampler,
            probabilities,
        )

    @property
    def task(self):
        return "supermimic"

    @property
    def observation_layout(self):
        n = self.action_size
        parts = (
            ("joint_position", n),
            ("joint_velocity", n),
            ("base_angular_velocity", 3),
            ("gravity_direction", 3),
            ("last_action", n),
        )
        if self.codec.config.reference_orientation:
            parts += (("reference_orientation_6d", 6),)
        return ObservationLayout(parts)

    @property
    def action_layout(self):
        return ActionLayout(tuple(self.model.actuator_joint_names))

    @property
    def max_episode_steps(self):
        return max(
            1,
            math.ceil(
                (self.reference_bank.max_clip_frames - 1)
                / (self.reference_bank.fps * self.control_timestep)
            ),
        )

    @property
    def critic_observation_layout(self):
        n = self.action_size
        return ObservationLayout(
            (
                ("invariant_physical_state", 13 + 2 * n),
                ("joint_position_error", n),
                ("body_position_error", 3 * len(self.kinematics.body_names)),
                ("root_error", 9),
                ("physical_context", n + 5),
                ("future_reference", self.codec.goal_size),
            )
        )

    def critic_observation(self, state, world_id=0):
        return critic_features(
            self.physical_state(state),
            self.reference_state(
                state.reference_time, state.alignment_xy, state.alignment_yaw
            ),
            self.goal(state.reference_time),
            self.context(state, world_id),
            self.kinematics,
        )

    def critic_observation_batch(self, state, rng=None):
        del rng
        return jax.vmap(self.critic_observation)(
            state, jnp.arange(state.last_action.shape[0])
        )

    @property
    def sampling_bin_count(self):
        return len(self.sampler.starts)

    def initial_sampling_failure_ema(self):
        return jnp.zeros(0, jnp.float32)

    def _update_sampling_failure_ema(self, state, terminal):
        return state.sampling_failure_ema[0]

    def _sample_reference_time(self, key, _failure_ema):
        return self.sampler.sample(key, self.reset_probabilities)

    def unaligned_reference(self, frame):
        return self.reference_bank.sample_global_frame(frame)

    def bounded_reference_time(self, frame):
        bank = self.reference_bank
        clip = self.clip_id(frame)
        return jnp.clip(
            frame,
            bank.time_offsets[clip],
            bank.time_offsets[clip] + bank.lengths[clip] - 1,
        )

    def clip_id(self, frame):
        return jnp.clip(
            jnp.searchsorted(self.reference_bank.time_offsets, frame, side="right") - 1,
            0,
            self.reference_bank.num_clips - 1,
        )

    def advance_reference_time(self, frame):
        return self.reference_bank.advance_global_frame(
            frame, self.reference_bank.fps * self.control_timestep
        )

    def reference_time_seconds(self, frame):
        return frame / self.reference_bank.fps

    def reference_state(self, frame, xy, yaw):
        reference = self.reference_bank.sample_global_frame(frame)
        return align_reference(
            self.kinematics.from_qpos(reference.qpos, reference.qvel), xy, yaw
        )

    def reference_bodies(self, frame, xy, yaw):
        """Cached body targets on the validated collection frame clock."""
        if self.reference_bank.fps * self.control_timestep < 1:
            # Interpolating joint angles requires FK after interpolation.
            return self.kinematics.forward(self.reference_state(frame, xy, yaw))
        sample = self.reference_bank.sample_global_frame(frame)
        orientation = rot.exp(jnp.array([0.0, 0.0, yaw]))
        shift = jnp.concatenate((xy, jnp.zeros(1)))
        return BodyState(
            rot.rotate(orientation, sample.body_pos) + shift,
            rot.multiply(orientation, sample.body_quat),
            rot.rotate(orientation, sample.body_lin_vel),
            rot.rotate(orientation, sample.body_ang_vel),
        )

    def goal(self, frame):
        bank = self.reference_bank
        clip = self.clip_id(frame)
        offsets = (
            jnp.arange(1, self.codec.config.future_frames + 1)
            * bank.fps
            * self.control_timestep
        )
        samples = bank.window(clip, frame - bank.time_offsets[clip], offsets)
        future = jax.vmap(self.kinematics.from_qpos)(samples.qpos, samples.qvel)
        current = self.reference_state(frame, jnp.zeros(2), jnp.asarray(0.0))
        return self.codec.goal(current, future)

    def action_target(self, state, action):
        return self.mapping.command(action, state.joint_default_offset).accepted

    def prepare_action(self, action):
        return jnp.asarray(action)

    def _initial_ctrl(self, joint_default_offset, qpos):
        del joint_default_offset
        return self._logical_target_to_ctrl(qpos[self.model.actuator_qpos_indices])

    def _perturb_root(self, qpos, qvel, key):
        keys = jax.random.split(key, 4)

        def noise(rng, magnitude):
            bound = jnp.asarray(magnitude)
            return jax.random.uniform(rng, (3,), minval=-bound, maxval=bound)

        cfg = self.config.reset
        state = self.kinematics.from_qpos(qpos, qvel)
        state = replace(
            state,
            position=state.position + noise(keys[0], cfg.root_position_noise),
            quaternion=rot.multiply(
                rot.exp(noise(keys[1], cfg.root_rotation_noise)), state.quaternion
            ),
            linear_velocity=state.linear_velocity
            + noise(keys[2], cfg.root_linear_velocity_noise),
            angular_velocity=state.angular_velocity
            + noise(keys[3], cfg.root_angular_velocity_noise),
        )
        qpos, qvel = self.kinematics.to_qpos(state, qpos, qvel)
        target = self._initial_ctrl(jnp.zeros(self.action_size), qpos)
        joints = jnp.matmul(
            jnp.asarray(self.model.target_limits.joint_from_motor),
            target,
            precision=jax.lax.Precision.HIGHEST,
        )
        return qpos.at[self.model.actuator_qpos_indices].set(joints), qvel

    def physical_state(self, state):
        return self.kinematics.from_qpos(state.data.qpos, state.data.qvel)

    def physics_metrics(self, data):
        impl = data._impl
        return {
            "physics/max_constraints": jnp.max(impl.nefc),
            "physics/contacts": jnp.max(impl.nacon),
            "invalid/constraint_capacity": jnp.any(impl.nefc > impl.njmax).astype(
                jnp.float32
            ),
            "invalid/contact_capacity": jnp.any(impl.nacon > impl.naconmax).astype(
                jnp.float32
            ),
        }

    def _actor_feedback(self, data):
        return self.kinematics.from_qpos(data.qpos, data.qvel).flatten()

    def _noisy_actor_feedback(self, data, rng):
        state = self.kinematics.from_qpos(data.qpos, data.qvel)
        return measured_feedback(
            state,
            rng,
            self.config.observation_noise,
            self.config.observation_low_pass_filter,
        ).flatten()

    def _advance_actor_feedback_history(self, history, data, rng=None):
        physical = self.kinematics.from_qpos(data.qpos, data.qvel)
        memory = PhysicalState.unflatten(history, self.action_size)
        return advance_feedback(
            memory,
            physical,
            rng,
            noise=self.config.observation_noise,
            filtering=self.config.observation_low_pass_filter,
            timestep=self.control_timestep,
        ).flatten()

    def observation(self, state, rng=None, *, _feedback=None):
        packed = self._delayed_actor_feedback(state) if _feedback is None else _feedback
        feedback = PhysicalState.unflatten(packed, self.action_size)
        reference = self.reference_state(
            state.reference_time, state.alignment_xy, state.alignment_yaw
        )
        return actor_sample(
            self.codec,
            feedback,
            state.last_action,
            reference.quaternion,
            state.joint_default_offset,
            rng,
            noise=self.config.observation_noise,
            filtering=self.config.observation_low_pass_filter,
        )

    def _reward(self, previous, state, world_id=0):
        reference = self.reference_state(
            state.reference_time, state.alignment_xy, state.alignment_yaw
        )
        return self.mdp.reward(
            self.physical_state(state),
            reference,
            state.last_action,
            previous.last_action,
        )

    def termination_flags(self, state):
        physical = self.physical_state(state)
        reference = self.reference_state(
            state.reference_time, state.alignment_xy, state.alignment_yaw
        )
        height, tilt = self.mdp.errors(physical, reference)
        clip = self.clip_id(state.reference_time)
        end = (
            self.reference_bank.time_offsets[clip]
            + self.reference_bank.lengths[clip]
            - 1
        )
        nonfinite = ~jnp.all(jnp.isfinite(physical.flatten()))
        terminal = self.mdp.terminal(physical, reference)
        timeout = state.reference_time >= end
        return {
            "done/height": height > self.mdp.config.height_error,
            "done/orientation": tilt > self.mdp.config.tilt_error,
            "done/end_effector": jnp.asarray(False),
            "done/nonfinite": nonfinite,
            "done/terminal": terminal,
            "done/timeout": timeout,
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
            **reward_info,
            **done_flags,
        }

    def _sanitize_nonfinite_reward(self, reward, reward_info, done_flags):
        # Tracking diagnostics are not additive reward components.
        invalid = done_flags["done/nonfinite"] | ~jnp.isfinite(reward)
        return jnp.where(invalid, 0.0, reward), {
            name: jnp.where(invalid, 0.0, value) for name, value in reward_info.items()
        }

    def context(self, state, world_id):
        n = self.action_size
        if self.config.model_randomization.enabled:
            friction = jnp.asarray(self.model.physical_randomization)[world_id, 4:5]
            com = jnp.asarray(self.model.torso_com_offset)[world_id]
        else:
            terrain = self.model.terrain_geom_ids
            if not len(terrain):
                raise ValueError("SuperMimic context requires a ground plane")
            friction = jnp.asarray(
                [float(self.model.mj_model.geom_friction[terrain[0], 0])]
            )
            com = jnp.zeros(3)
        return jnp.concatenate(
            (
                state.joint_default_offset[:n],
                friction,
                com,
                jnp.asarray(state.actor_feedback_delay_steps, jnp.float32)[None],
            )
        )
