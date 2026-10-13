"""Shared native MuJoCo rendering and evaluation math."""

from __future__ import annotations

from dataclasses import dataclass, fields
from io import BytesIO
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from bello_mujoco.evaluation.walking import (
    WalkingAcceptanceConfig,
    WalkingDiagnostics,
    native_walking_sample,
)
from bello_mujoco.evaluation.video import (
    CameraConfig,
    open_rollout_video_resources,
    write_video_frame,
)
from bello_mujoco.envs.foot_geometry import (
    sole_support_positions,
    heightfield_surface_normal,
)
from bello_mujoco.envs.observations import noisy_feedback
from bello_mujoco.envs.actions import limit_motor_step
from bello_mujoco.envs.locomotion import BelloLocomotionEnv
from bello_mujoco.envs.locomotion_state import BelloState
from bello_mujoco.envs.gait import gait_frequency, resolve_gait_command_transition
from bello_mujoco.envs.rewards import command_is_moving
from bello_mujoco.evaluation.settling import (
    SettlingDiagnostics,
    annotate_standing_frame,
)
from bello_mujoco.policy import (
    apply_actor,
    init_actor_critic,
    init_observation_stack,
    policy_observation_size,
    sample_actor_action,
    stacked_observation,
    update_observation_stack,
)
from bello_mujoco.spatial import quat_rotate_inverse_wxyz, quat_rotate_wxyz
from bello_mujoco.terrain import terrain_surface_height
from bello_mujoco.evaluation.software import run_software_evaluation

if TYPE_CHECKING:
    from bello_mujoco.training.ppo_types import ObservationNormState
    from bello_mujoco.training.ppo_config import PPOConfig

RolloutStats: TypeAlias = dict[str, float | int]
DONE_FLAG_KEYS = (
    "done/any",
    "done/terminal",
    "done/base_height",
    "done/bad_orientation",
    "done/timeout",
    "done/nonfinite",
    "done/unstable",
)


@dataclass(frozen=True)
class NativeDataImplementation:
    cacc: jax.Array
    cinert: jax.Array
    cfrc_int: jax.Array
    actuator_velocity: jax.Array
    efc__force: jax.Array
    nacon: jax.Array
    contact__geom: jax.Array
    contact__pos: jax.Array
    contact__frame: jax.Array
    contact__worldid: jax.Array
    contact__efc_address: jax.Array
    contact__dim: jax.Array


@dataclass(frozen=True)
class NativeDataView:
    """JAX-array view of native MuJoCo data used by shared task functions."""

    qpos: jax.Array
    qvel: jax.Array
    qacc: jax.Array
    ctrl: jax.Array
    xpos: jax.Array
    xquat: jax.Array
    cvel: jax.Array
    subtree_com: jax.Array
    geom_xpos: jax.Array
    geom_xmat: jax.Array
    actuator_length: jax.Array
    actuator_force: jax.Array
    qfrc_constraint: jax.Array
    _impl: NativeDataImplementation


class NativeObservationPipeline:
    """Preserve training feedback filtering, noise, and delay across native steps."""

    def __init__(self, env: BelloLocomotionEnv, *, seed: int) -> None:
        self.env = env
        self._key = jax.random.PRNGKey(seed)
        self.reset()

    def reset(self) -> None:
        self.torso_command = jnp.zeros(3, dtype=jnp.float32)
        self._history: jax.Array | None = None
        self._step: int | None = None
        self._delay = jnp.asarray(0, dtype=jnp.int32)
        self._observation_key = self._key

    def observe(
        self,
        mj_data: mujoco.MjData,
        *,
        command: np.ndarray,
        last_action: np.ndarray,
        step_count: int,
        gait_phase: float | jax.Array | None = None,
        standing_transition_pending: int | jax.Array = 0,
        torso_command=(0.0, 0.0, 0.0),
    ) -> jax.Array:
        env = self.env
        env.command_config.torso.validate_target(torso_command)
        if not env.command_config.enabled and any(torso_command):
            raise ValueError("torso targets require command conditioning")
        if self._step != step_count:
            expected = 0 if self._step is None else self._step + 1
            if step_count != expected:
                raise ValueError(
                    f"native feedback expected step {expected}, got {step_count}; "
                    "reset feedback when resetting the episode"
                )
        if self._step is not None and self._step != step_count:
            from bello_mujoco.envs.torso import slew_torso_command

            self.torso_command = slew_torso_command(
                self.torso_command,
                jnp.asarray(torso_command),
                env.command_config.torso.slew_rate,
                env.control_timestep,
            )
        state = mujoco_eval_state(
            env,
            mj_data,
            command=command,
            torso_command=self.torso_command,
            last_action=last_action,
            foot_contact_debounce_time=jnp.zeros(env.model.foot_body_ids.size),
            last_foot_contact=jnp.zeros(env.model.foot_body_ids.size),
            step_count=step_count,
            episode_return=jnp.asarray(0.0),
            episode_track_lin_vel_xy=jnp.asarray(0.0),
            episode_length=jnp.asarray(0),
            gait_phase=gait_phase,
            standing_transition_pending=standing_transition_pending,
        )
        if self._step != step_count:
            self._key, feedback_key, self._observation_key, delay_key = (
                jax.random.split(self._key, 4)
            )
            if self._history is None:
                feedback = env._actor_feedback(state)
                if env.config.observation_low_pass_filter.noise_before_filter:
                    feedback = noisy_feedback(
                        feedback, env.config.observation.noise, feedback_key
                    )
                self._history = jnp.repeat(
                    feedback[None, :], env.maximum_observation_delay_steps + 1, axis=0
                )
                self._delay = env._sample_observation_delay(delay_key)
            else:
                self._history = env._advance_actor_feedback_history(
                    self._history, state, feedback_key
                )
            self._step = step_count
        return env.observation(
            state.replace(
                actor_feedback_history=self._history,
                actor_feedback_delay_steps=self._delay,
            ),
            self._observation_key,
        )


def render_policy_rollout_isolated(
    env: BelloLocomotionEnv,
    policy,
    *,
    steps: int,
    **kwargs,
) -> RolloutStats:
    if env.model.mjx_model is not None:
        raise ValueError("software evaluation requires a native-only rendering model")
    # Do not serialize cached JAX properties or device arrays from the trainer.
    env_fields = {field.name: getattr(env, field.name) for field in fields(env)}
    weights = BytesIO()
    eqx.tree_serialise_leaves(weights, policy)
    return run_software_evaluation(
        _render_host_policy,
        env_fields,
        weights.getvalue(),
        jax.device_get(dict(kwargs, steps=steps)),
        timeout_seconds=300.0 + steps,
    )


def _render_host_policy(env_fields: dict, weights: bytes, kwargs: dict) -> RolloutStats:
    env = BelloLocomotionEnv(**env_fields)
    config = kwargs["ppo_config"].policy
    policy = init_actor_critic(
        jax.random.PRNGKey(0),
        policy_observation_size(env.observation_size, config),
        env.action_size,
        config,
        critic_obs_size=policy_observation_size(env.critic_observation_size, config),
    )
    policy = eqx.tree_deserialise_leaves(BytesIO(weights), policy)
    return render_policy_rollout(env, policy, **kwargs)


def render_policy_rollout(
    env: BelloLocomotionEnv,
    policy,
    *,
    output_path: Path,
    command: tuple[float, float, float],
    steps: int,
    width: int,
    height: int,
    fps: int,
    stop_on_done: bool,
    camera_config: CameraConfig,
    ppo_config: PPOConfig,
    observation_norm_state: ObservationNormState,
    obs_stack_size: int = 1,
    stochastic_actions: bool = False,
    action_seed: int = 0,
    push_at_step: int | None = None,
    push_velocity: tuple[float, float] = (0.0, 0.0),
    record_standing: bool = False,
    walking_acceptance: WalkingAcceptanceConfig = WalkingAcceptanceConfig(),
    torso_command: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> RolloutStats:
    env.command_config.torso.validate_target(torso_command)
    if push_at_step is not None and not 0 < push_at_step < steps:
        raise ValueError("push_at_step must be inside the rollout")
    if len(push_velocity) != 2 or not np.all(np.isfinite(push_velocity)):
        raise ValueError("push_velocity must contain two finite values")
    if record_standing and (np.any(np.asarray(command) != 0) or not stop_on_done):
        raise ValueError(
            "standing diagnostics require zero command and first-failure stopping"
        )
    if walking_acceptance.enabled and (
        record_standing or not stop_on_done or np.linalg.norm(command[:2]) < 0.2
    ):
        raise ValueError(
            "walking acceptance requires translation >= 0.2 m/s and first-failure stopping"
        )
    walking = (
        WalkingDiagnostics(env.control_timestep, walking_acceptance)
        if walking_acceptance.enabled
        else None
    )
    mj_model = env.model.mj_model
    mj_data = mujoco.MjData(mj_model)
    reset_mujoco_data(env, mj_data)

    command_array = np.asarray(command, dtype=np.float32)
    last_action = np.zeros(env.action_size, dtype=np.float32)
    step_count = 0
    feedback = NativeObservationPipeline(env, seed=env.config.model_randomization.seed)
    obs = feedback.observe(
        mj_data,
        command=command_array,
        torso_command=torso_command,
        last_action=last_action,
        step_count=step_count,
    )
    obs_stack = init_observation_stack(obs, obs_stack_size)
    action_key = jax.random.PRNGKey(action_seed)
    stats = init_rollout_stats(
        steps, stochastic_actions=stochastic_actions, action_seed=action_seed
    )
    stats["observation_seed"] = env.config.model_randomization.seed
    stats["push_applied"] = 0
    stats["push_velocity_x"] = float(push_velocity[0])
    stats["push_velocity_y"] = float(push_velocity[1])
    diagnostics = (
        SettlingDiagnostics(env.control_timestep, push_at_step)
        if record_standing
        else None
    )

    def annotate(frame):
        if diagnostics is None or not diagnostics.samples:
            return frame
        return annotate_standing_frame(
            frame,
            diagnostics.samples[-1],
            None if push_at_step is None else push_at_step * env.control_timestep,
        )

    resources, renderer, camera, writer = open_rollout_video_resources(
        mj_model,
        output_path,
        width=width,
        height=height,
        fps=fps,
        camera_config=camera_config,
    )
    try:
        for step_index in range(steps):
            write_video_frame(
                renderer,
                camera,
                writer,
                mj_model,
                mj_data,
                camera_config,
                root_body_name=env.model.robot.root_body_name,
                transform=annotate,
            )
            action_rng = None
            if stochastic_actions:
                action_key, action_rng = jax.random.split(action_key)
            action = rollout_policy_action(
                policy,
                obs_stack,
                action_rng=action_rng,
                action_clip=env.action_clip,
                ppo_config=ppo_config,
                observation_norm_state=observation_norm_state,
            )
            action = np.asarray(jax.device_get(action), dtype=np.float32)
            if push_at_step is not None and step_index == push_at_step:
                root_qvel = int(env.model.root_qvel_adr)
                mj_data.qvel[root_qvel : root_qvel + 2] += push_velocity
                stats["push_applied"] = 1
            previous_ctrl = mj_data.ctrl.copy()
            requested_ctrl = mujoco_action_to_ctrl(env, action)
            requested_ctrl = limit_motor_step(
                jnp.asarray(requested_ctrl),
                jnp.asarray(previous_ctrl),
                env.model.target_limits,
            )
            for substep in range(env.decimation):
                mj_data.ctrl[:] = env.delayed_ctrl(
                    previous_ctrl, requested_ctrl, substep
                )
                mujoco.mj_step(mj_model, mj_data)
            last_action = action
            step_count += 1
            done_flags = mujoco_done_flags(env, mj_data, step_count)
            if diagnostics is not None:
                # Diagnostic duration, not the training timeout, is the horizon.
                done_flags["done/timeout"] = False
                done_flags["done/any"] = done_flags["done/terminal"]
                diagnostics.record(
                    mujoco_standing_sample(env, mj_data, done_flags["done/terminal"])
                )
            if walking is not None:
                walking.record(
                    native_walking_sample(
                        env, mj_data, command_array, done_flags["done/any"]
                    )
                )
            done = bool(done_flags["done/any"])
            if done:
                record_rollout_done(
                    stats,
                    step_index=step_index + 1,
                    episode_length=step_count,
                    done_flags=done_flags,
                    control_timestep=env.control_timestep,
                )
            if done and stop_on_done:
                obs = feedback.observe(
                    mj_data,
                    command=command_array,
                    torso_command=torso_command,
                    last_action=last_action,
                    step_count=step_count,
                )
                obs_stack = update_observation_stack(obs_stack, obs, jnp.asarray(True))
                write_video_frame(
                    renderer,
                    camera,
                    writer,
                    mj_model,
                    mj_data,
                    camera_config,
                    root_body_name=env.model.robot.root_body_name,
                    transform=annotate,
                )
                break
            if done:
                reset_mujoco_data(env, mj_data)
                last_action = np.zeros(env.action_size, dtype=np.float32)
                step_count = 0
                feedback.reset()
                obs = feedback.observe(
                    mj_data,
                    command=command_array,
                    torso_command=torso_command,
                    last_action=last_action,
                    step_count=step_count,
                )
                obs_stack = init_observation_stack(obs, obs_stack_size)
                continue
            obs = feedback.observe(
                mj_data,
                command=command_array,
                torso_command=torso_command,
                last_action=last_action,
                step_count=step_count,
            )
            obs_stack = update_observation_stack(obs_stack, obs, jnp.asarray(False))
    finally:
        resources.close()
        if diagnostics is not None:
            with output_path.with_suffix(".jsonl").open("w") as trace:
                for sample in diagnostics.samples:
                    trace.write(json.dumps(sample, allow_nan=False) + "\n")
        if walking is not None:
            with output_path.with_suffix(".walking.jsonl").open("w") as trace:
                for sample in walking.samples:
                    trace.write(json.dumps(sample, allow_nan=False) + "\n")
    if walking is not None:
        stats.update(walking.summary())
    stats["survived_rollout"] = int(stats["completed_episodes"] == 0)
    if diagnostics is not None:
        stats.update(diagnostics.summary())
    if push_at_step is not None:
        first_done = int(stats["first_done_step"])
        end = first_done if first_done >= 0 else steps
        stats["post_push_seconds"] = max(0, end - push_at_step) * env.control_timestep
    return finalize_rollout_stats(stats, current_episode_length=step_count)


def mujoco_standing_sample(
    env, data: mujoco.MjData, terminal: bool
) -> dict[str, float]:
    model = env.model.mj_model
    frame = model.body(env._reward_frame_link("track_lin_vel_xy")).id
    rotation = data.xmat[frame].reshape(3, 3)
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    yaw_rotation = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1],
        ]
    )
    forward = yaw_rotation @ np.asarray(env.model.robot.locomotion_forward_axis)
    left = np.cross([0, 0, 1], forward)
    geometry = env._sole_reward_geometry
    feet = np.asarray(env.model.foot_body_ids)
    centers = data.xpos[feet] + np.einsum(
        "nij,nj->ni", data.xmat[feet].reshape(-1, 3, 3), geometry.centers
    )
    delta = centers[0] - centers[1]
    headings = []
    command_forward = rotation @ np.asarray(env.model.robot.locomotion_forward_axis)
    command_forward = command_forward[:2] / max(
        np.linalg.norm(command_forward[:2]), 1e-6
    )
    for name, axis in env.model.robot.foot_forward_axes:
        vector = data.xmat[model.body(name).id].reshape(3, 3) @ np.asarray(axis)
        vector = vector[:2] / max(np.linalg.norm(vector[:2]), 1e-6)
        headings.append(
            abs(
                math.atan2(
                    command_forward[0] * vector[1] - command_forward[1] * vector[0],
                    command_forward @ vector,
                )
            )
        )
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_XBODY,
        model.body(env._reward_link("track_lin_vel_xy")).id,
        velocity,
        0,
    )
    speed = float(np.linalg.norm(velocity[3:5]))
    torso = model.body(env._reward_link("flat_orientation")).id
    gravity = data.xmat[torso].reshape(3, 3).T @ np.array([0, 0, -1])
    contacts = mujoco_foot_contact(env, data)
    indices = env.model.policy_ctrl_indices
    saturation = (
        np.abs(data.actuator_force[indices])
        >= 0.99 * np.asarray(env.model.actuator_peak_force)[indices]
    )
    return {
        "speed": speed,
        "yaw_rate": float(
            mujoco_link_ang_vel(env, data, env._reward_link("track_ang_vel_z"))[2]
        ),
        "tilt": math.acos(float(np.clip(-gravity[2], -1, 1))),
        "foreaft_error": abs(float(delta @ forward)),
        "width_error": abs(float(delta @ left) - geometry.nominal_width),
        "heading_error": max(headings),
        "left_contact": float(contacts[0]),
        "right_contact": float(contacts[1]),
        "x": float(data.xpos[frame, 0]),
        "y": float(data.xpos[frame, 1]),
        "yaw": yaw,
        "torque_saturation": float(np.mean(saturation)),
        "self_collision_contacts": float(mujoco_self_collision(env, data)),
        "terminal": float(terminal),
    }


def rollout_policy_action(
    policy,
    obs_stack: jax.Array,
    *,
    action_rng: jax.Array | None = None,
    action_clip: float,
    ppo_config: PPOConfig,
    observation_norm_state: ObservationNormState,
) -> jax.Array:
    from bello_mujoco.training.normalization import normalize_observation

    obs = normalize_observation(
        stacked_observation(obs_stack),
        observation_norm_state,
        ppo_config,
    )
    mean = apply_actor(policy, obs)
    action = mean
    if action_rng is not None:
        action = sample_actor_action(policy, obs, action_rng)
    return jnp.clip(action, -action_clip, action_clip)


def init_rollout_stats(
    steps: int, *, stochastic_actions: bool, action_seed: int
) -> RolloutStats:
    stats: RolloutStats = {
        "steps": int(steps),
        "stochastic_actions": int(stochastic_actions),
        "action_seed": int(action_seed),
        "completed_episodes": 0,
        "first_done_step": -1,
        "first_done_seconds": -1.0,
        "last_done_step": -1,
        "mean_completed_episode_length": 0.0,
        "max_completed_episode_length": 0,
        "current_episode_length": 0,
        "_completed_episode_length_sum": 0.0,
    }
    for key in DONE_FLAG_KEYS:
        stats[key] = 0
    return stats


def record_rollout_done(
    stats: RolloutStats,
    *,
    step_index: int,
    episode_length: int,
    done_flags,
    control_timestep: float,
) -> None:
    stats["completed_episodes"] = int(stats["completed_episodes"]) + 1
    stats["last_done_step"] = int(step_index)
    if int(stats["first_done_step"]) < 0:
        stats["first_done_step"] = int(step_index)
        stats["first_done_seconds"] = float(step_index * control_timestep)
    stats["_completed_episode_length_sum"] = float(
        stats["_completed_episode_length_sum"]
    ) + float(episode_length)
    stats["max_completed_episode_length"] = max(
        int(stats["max_completed_episode_length"]), int(episode_length)
    )
    for key in DONE_FLAG_KEYS:
        value = done_flags.get(key, False)
        stats[key] = int(stats[key]) + int(bool(np.asarray(value)))


def finalize_rollout_stats(
    stats: RolloutStats, *, current_episode_length: int
) -> RolloutStats:
    completed = int(stats["completed_episodes"])
    length_sum = float(stats.pop("_completed_episode_length_sum"))
    stats["mean_completed_episode_length"] = (
        length_sum / completed if completed else 0.0
    )
    stats["current_episode_length"] = int(current_episode_length)
    return stats


def mujoco_policy_observation(
    env: BelloLocomotionEnv,
    mj_data: mujoco.MjData,
    *,
    command: np.ndarray,
    last_action: np.ndarray,
    step_count: int,
    last_action_delta: np.ndarray | None = None,
    gait_phase: float | jax.Array | None = None,
) -> jax.Array:
    """Instantaneous noise-free snapshot; rollouts use NativeObservationPipeline."""
    state = mujoco_eval_state(
        env,
        mj_data,
        command=command,
        last_action=last_action,
        last_action_delta=last_action_delta,
        foot_contact_debounce_time=jnp.zeros(
            env.model.foot_body_ids.size, dtype=jnp.float32
        ),
        last_foot_contact=jnp.zeros(env.model.foot_body_ids.size, dtype=jnp.float32),
        step_count=step_count,
        episode_return=jnp.asarray(0.0, dtype=jnp.float32),
        episode_track_lin_vel_xy=jnp.asarray(0.0, dtype=jnp.float32),
        episode_length=jnp.asarray(0, dtype=jnp.int32),
        gait_phase=gait_phase,
    )
    return env.observation(state)


def reset_mujoco_data(env: BelloLocomotionEnv, mj_data: mujoco.MjData) -> None:
    mujoco.mj_resetData(env.model.mj_model, mj_data)
    mj_data.qpos[:] = env.model.default_qpos
    mj_data.qvel[:] = env.model.default_qvel
    mj_data.ctrl[:] = mujoco_action_to_ctrl(
        env, np.zeros(env.action_size, dtype=np.float32)
    )
    mujoco.mj_forward(env.model.mj_model, mj_data)
    if env.model.foot_body_ids.size:
        clearance_adjustment = env.config.spawn.clearance - float(
            np.min(mujoco_foot_height(env, mj_data))
        )
        mj_data.qpos[env.model.root_qpos_adr + 2] += clearance_adjustment
        mujoco.mj_forward(env.model.mj_model, mj_data)


def mujoco_action_to_ctrl(env: BelloLocomotionEnv, action: np.ndarray) -> np.ndarray:
    return np.asarray(
        jax.device_get(env.action_to_ctrl(jnp.asarray(action, dtype=jnp.float32))),
        dtype=np.float32,
    )


def mujoco_requested_actuator_force(
    env: BelloLocomotionEnv, mj_data: mujoco.MjData
) -> np.ndarray:
    """Return native MuJoCo's unclipped position-servo force request."""

    model = env.model.mj_model
    return (
        model.actuator_gainprm[:, 0] * mj_data.ctrl
        + model.actuator_biasprm[:, 1] * mj_data.actuator_length
        + model.actuator_biasprm[:, 2] * mj_data.actuator_velocity
    ).astype(np.float32)


def mujoco_done_flags(
    env: BelloLocomotionEnv, mj_data: mujoco.MjData, step_count: int
) -> dict[str, bool]:
    zeros = jnp.zeros(env.model.foot_body_ids.size, dtype=jnp.float32)
    state = mujoco_eval_state(
        env,
        mj_data,
        command=np.zeros(3, dtype=np.float32),
        last_action=np.zeros(env.action_size, dtype=np.float32),
        foot_contact_debounce_time=zeros,
        last_foot_contact=zeros,
        step_count=step_count,
        episode_return=jnp.asarray(0.0, dtype=jnp.float32),
        episode_track_lin_vel_xy=jnp.asarray(0.0, dtype=jnp.float32),
        episode_length=jnp.asarray(step_count, dtype=jnp.int32),
    )
    return {
        name: bool(jax.device_get(value))
        for name, value in env.done_flags(state).items()
    }


def mujoco_link_quat(
    env: BelloLocomotionEnv, mj_data: mujoco.MjData, link_name: str
) -> np.ndarray:
    if link_name == env.model.robot.root_body_name:
        root_qpos = int(env.model.root_qpos_adr)
        return np.asarray(mj_data.qpos[root_qpos + 3 : root_qpos + 7], dtype=np.float32)
    body_id = mujoco.mj_name2id(env.model.mj_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
    if body_id < 0:
        raise ValueError(f"configured link/body not found: {link_name!r}")
    return np.asarray(mj_data.xquat[body_id], dtype=np.float32)


def mujoco_link_clearance(
    env: BelloLocomotionEnv, mj_data: mujoco.MjData, link_name: str
) -> float:
    if link_name == env.model.robot.root_body_name:
        root_qpos = int(env.model.root_qpos_adr)
        position = np.asarray(mj_data.qpos[root_qpos : root_qpos + 3])
    else:
        body_id = mujoco.mj_name2id(
            env.model.mj_model, mujoco.mjtObj.mjOBJ_BODY, link_name
        )
        if body_id < 0:
            raise ValueError(f"configured link/body not found: {link_name!r}")
        position = np.asarray(mj_data.xpos[body_id])

    if env.terrain_config.mode == "none":
        terrain_height = env.terrain_config.pos[2]
    else:
        terrain_height = terrain_surface_height(
            env.model.mj_model,
            int(env.model.terrain_geom_ids[0]),
            position[None, :2],
        )[0]
    return float(position[2] - terrain_height)


def mujoco_link_lin_vel(
    env: BelloLocomotionEnv, mj_data: mujoco.MjData, link_name: str
) -> np.ndarray:
    if link_name == env.model.robot.root_body_name:
        root_qpos = int(env.model.root_qpos_adr)
        root_qvel = int(env.model.root_qvel_adr)
        return quat_rotate_inverse_wxyz(
            np.asarray(mj_data.qpos[root_qpos + 3 : root_qpos + 7], dtype=np.float32),
            np.asarray(mj_data.qvel[root_qvel : root_qvel + 3], dtype=np.float32),
        )
    body_id = mujoco.mj_name2id(env.model.mj_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
    if body_id < 0:
        raise ValueError(f"configured link/body not found: {link_name!r}")
    velocity = np.empty(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        env.model.mj_model,
        mj_data,
        mujoco.mjtObj.mjOBJ_XBODY,
        body_id,
        velocity,
        1,
    )
    return np.asarray(velocity[3:6], dtype=np.float32)


def mujoco_link_ang_vel(
    env: BelloLocomotionEnv, mj_data: mujoco.MjData, link_name: str
) -> np.ndarray:
    if link_name == env.model.robot.root_body_name:
        root_qvel = int(env.model.root_qvel_adr)
        return np.asarray(mj_data.qvel[root_qvel + 3 : root_qvel + 6], dtype=np.float32)
    body_id = mujoco.mj_name2id(env.model.mj_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
    if body_id < 0:
        raise ValueError(f"configured link/body not found: {link_name!r}")
    velocity = np.empty(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        env.model.mj_model,
        mj_data,
        mujoco.mjtObj.mjOBJ_XBODY,
        body_id,
        velocity,
        1,
    )
    return np.asarray(velocity[:3], dtype=np.float32)


def mujoco_eval_state(
    env: BelloLocomotionEnv,
    mj_data: mujoco.MjData,
    *,
    command: np.ndarray,
    last_action: np.ndarray,
    foot_contact_debounce_time: jax.Array,
    last_foot_contact: jax.Array,
    step_count: int,
    episode_return: jax.Array,
    episode_track_lin_vel_xy: jax.Array,
    episode_length: jax.Array,
    gait_phase: float | jax.Array | None = None,
    standing_transition_pending: int | jax.Array = 0,
    standing_reward_blend: float | jax.Array = 1.0,
    last_action_delta: np.ndarray | None = None,
    tracking_ang_vel_z: float | jax.Array | None = None,
    torso_command=(0.0, 0.0, 0.0),
) -> BelloState:
    if gait_phase is None:
        moving = command_is_moving(
            jnp.asarray(command, dtype=jnp.float32), env.command_config
        )
        _, initial_phase, _ = resolve_gait_command_transition(
            jnp.zeros(3),
            jnp.asarray(command),
            jnp.asarray(0.0),
            jnp.asarray(0.0),
            jnp.asarray(0),
            env.command_config,
            env.config.gait,
            jnp.asarray(True),
            -env._locomotion_right_axis()[:2],
        )
        gait_phase = jnp.where(
            moving,
            jnp.mod(
                initial_phase
                + step_count
                * gait_frequency(jnp.asarray(command), env.config.gait)
                * env.control_timestep,
                1.0,
            ),
            0.0,
        )
    if last_action_delta is None:
        last_action_delta = np.zeros(env.action_size, dtype=np.float32)
    if tracking_ang_vel_z is None:
        tracking_ang_vel_z = mujoco_link_ang_vel(
            env, mj_data, env._reward_link("track_ang_vel_z")
        )[2]
    data = NativeDataView(
        qpos=jnp.asarray(np.asarray(mj_data.qpos, dtype=np.float32).copy()),
        qvel=jnp.asarray(np.asarray(mj_data.qvel, dtype=np.float32).copy()),
        qacc=jnp.asarray(np.asarray(mj_data.qacc, dtype=np.float32).copy()),
        ctrl=jnp.asarray(np.asarray(mj_data.ctrl, dtype=np.float32).copy()),
        xpos=jnp.asarray(np.asarray(mj_data.xpos, dtype=np.float32).copy()),
        xquat=jnp.asarray(np.asarray(mj_data.xquat, dtype=np.float32).copy()),
        cvel=jnp.asarray(np.asarray(mj_data.cvel, dtype=np.float32).copy()),
        subtree_com=jnp.asarray(
            np.asarray(mj_data.subtree_com, dtype=np.float32).copy()
        ),
        geom_xpos=jnp.asarray(np.asarray(mj_data.geom_xpos, dtype=np.float32).copy()),
        geom_xmat=jnp.asarray(
            np.asarray(mj_data.geom_xmat, dtype=np.float32).reshape((-1, 3, 3)).copy()
        ),
        actuator_length=jnp.asarray(
            np.asarray(mj_data.actuator_length, dtype=np.float32).copy()
        ),
        actuator_force=jnp.asarray(
            np.asarray(mj_data.actuator_force, dtype=np.float32).copy()
        ),
        qfrc_constraint=jnp.asarray(
            np.asarray(mj_data.qfrc_constraint, dtype=np.float32).copy()
        ),
        _impl=NativeDataImplementation(
            nacon=jnp.asarray([mj_data.ncon], dtype=jnp.int32),
            contact__geom=jnp.asarray(np.asarray(mj_data.contact.geom).copy()),
            contact__pos=jnp.asarray(
                np.asarray(mj_data.contact.pos).copy(), dtype=jnp.float32
            ),
            contact__frame=jnp.asarray(
                np.asarray(mj_data.contact.frame).reshape(-1, 3, 3).copy(),
                dtype=jnp.float32,
            ),
            contact__worldid=jnp.zeros(mj_data.ncon, dtype=jnp.int32),
            contact__efc_address=jnp.asarray(
                np.asarray(mj_data.contact.efc_address[:, None]).copy()
            ),
            contact__dim=jnp.asarray(np.asarray(mj_data.contact.dim).copy()),
            cinert=jnp.asarray(np.asarray(mj_data.cinert, dtype=np.float32).copy()),
            cacc=jnp.asarray(np.asarray(mj_data.cacc, dtype=np.float32).copy()),
            cfrc_int=jnp.asarray(np.asarray(mj_data.cfrc_int, dtype=np.float32).copy()),
            actuator_velocity=jnp.asarray(
                np.asarray(mj_data.actuator_velocity, dtype=np.float32).copy()
            ),
            efc__force=jnp.asarray(
                np.asarray(mj_data.efc_force, dtype=np.float32).copy()
            ),
        ),
    )
    state = BelloState(
        data=data,
        last_action=jnp.asarray(last_action, dtype=jnp.float32),
        last_action_delta=jnp.asarray(last_action_delta, dtype=jnp.float32),
        tracking_ang_vel_z=jnp.asarray(tracking_ang_vel_z, dtype=jnp.float32),
        actor_feedback_history=jnp.zeros(
            (
                env.maximum_observation_delay_steps + 1,
                env.actor_feedback_size,
            ),
            dtype=jnp.float32,
        ),
        actor_feedback_delay_steps=jnp.asarray(0, dtype=jnp.int32),
        foot_contact_debounce_time=foot_contact_debounce_time,
        last_foot_contact=last_foot_contact,
        command=jnp.asarray(command, dtype=jnp.float32),
        command_target=jnp.asarray(command, dtype=jnp.float32),
        torso_command=jnp.asarray(torso_command, dtype=jnp.float32),
        torso_command_target=jnp.asarray(torso_command, dtype=jnp.float32),
        gait_phase=jnp.asarray(gait_phase, dtype=jnp.float32),
        standing_transition_pending=jnp.asarray(
            standing_transition_pending, dtype=jnp.int32
        ),
        next_start_left=jnp.asarray(True),
        standing_reward_blend=jnp.asarray(standing_reward_blend, dtype=jnp.float32),
        command_resample_step_count=jnp.asarray(step_count, dtype=jnp.int32),
        step_count=jnp.asarray(step_count, dtype=jnp.int32),
        push_steps_remaining=jnp.asarray(-1, dtype=jnp.int32),
        episode_return=episode_return,
        episode_track_lin_vel_xy=episode_track_lin_vel_xy,
        episode_length=episode_length,
        last_episode_return=jnp.asarray(0.0, dtype=jnp.float32),
        last_episode_track_lin_vel_xy=jnp.asarray(0.0, dtype=jnp.float32),
        last_episode_length=jnp.asarray(0, dtype=jnp.int32),
    )
    actor_feedback = env._actor_feedback(state)
    return state.replace(
        actor_feedback_history=jnp.repeat(
            actor_feedback[None, :],
            env.maximum_observation_delay_steps + 1,
            axis=0,
        )
    )


def mujoco_eval_info(
    env: BelloLocomotionEnv,
    mj_data: mujoco.MjData,
    *,
    command: np.ndarray | None = None,
) -> dict[str, jax.Array]:
    info = {
        "base_height": jnp.asarray(
            mujoco_link_clearance(env, mj_data, env.root_link_name),
            dtype=jnp.float32,
        )
    }
    if command is not None:
        info["command"] = jnp.asarray(command, dtype=jnp.float32)
    return info


def mujoco_contact_terms(
    env: BelloLocomotionEnv, mj_data: mujoco.MjData
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    foot_contact, foot_normal_force = mujoco_foot_contact_and_normal_force(env, mj_data)
    return (
        jnp.asarray(foot_contact, dtype=jnp.float32),
        jnp.asarray(foot_normal_force, dtype=jnp.float32),
        jnp.asarray(mujoco_foot_linear_velocity(env, mj_data), dtype=jnp.float32),
        jnp.asarray(mujoco_foot_angular_speed(env, mj_data), dtype=jnp.float32),
        jnp.asarray(mujoco_undesired_contact(env, mj_data), dtype=jnp.float32),
        jnp.asarray(mujoco_self_collision(env, mj_data), dtype=jnp.float32),
    )


def mujoco_foot_contact(env: BelloLocomotionEnv, mj_data: mujoco.MjData) -> np.ndarray:
    foot_contact, _ = mujoco_foot_contact_and_normal_force(env, mj_data)
    return foot_contact


def mujoco_foot_contact_and_normal_force(
    env: BelloLocomotionEnv, mj_data: mujoco.MjData
) -> tuple[np.ndarray, np.ndarray]:
    foot_contact = np.zeros(env.model.foot_body_ids.size, dtype=np.float32)
    foot_normal_force = np.zeros_like(foot_contact)
    if env.model.terrain_geom_ids.size == 0:
        return foot_contact, foot_normal_force
    force_threshold = float(env.config.contact.foot_contact_force_threshold)
    terrain_geom_ids = set(int(geom_id) for geom_id in env.model.terrain_geom_ids)
    foot_geom_ids = set(int(geom_id) for geom_id in env.model.foot_geom_ids)
    foot_body_ids = np.asarray(env.model.foot_body_ids, dtype=np.int32)
    foot_body_index = {
        int(body_id): foot_index for foot_index, body_id in enumerate(foot_body_ids)
    }
    geom_body_ids = np.asarray(env.model.geom_body_ids, dtype=np.int32)
    for contact_index in range(int(mj_data.ncon)):
        contact = mj_data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if geom1 in terrain_geom_ids:
            other_geom = geom2
        elif geom2 in terrain_geom_ids:
            other_geom = geom1
        else:
            continue
        if other_geom not in foot_geom_ids:
            continue
        normal_force = mujoco_contact_normal_force(
            env.model.mj_model, mj_data, contact_index
        )
        foot_index = foot_body_index.get(int(geom_body_ids[other_geom]))
        if foot_index is not None:
            foot_normal_force[foot_index] += normal_force
            if normal_force > force_threshold:
                foot_contact[foot_index] = 1.0
    return foot_contact, foot_normal_force


def mujoco_foot_height(env: BelloLocomotionEnv, mj_data: mujoco.MjData) -> np.ndarray:
    if env.model.foot_body_ids.size == 0:
        return np.zeros((0,), dtype=np.float32)
    foot_position = mujoco_foot_sole_positions(env, mj_data)
    if env.terrain_config.mode == "none":
        terrain_height = np.full(
            foot_position.shape[0], env.terrain_config.pos[2], dtype=np.float32
        )
    else:
        terrain_height = terrain_surface_height(
            env.model.mj_model,
            int(env.model.terrain_geom_ids[0]),
            foot_position[:, :2],
        )
    return foot_position[:, 2] - terrain_height


def mujoco_foot_sole_positions(
    env: BelloLocomotionEnv, mj_data: mujoco.MjData
) -> np.ndarray:
    ids = np.asarray(env.model.foot_geom_ids, dtype=np.int32)
    position = np.asarray(mj_data.geom_xpos[ids], dtype=np.float32)

    def terrain_height(xy):
        if env.terrain_config.mode == "none":
            return np.full(xy.shape[:-1], env.terrain_config.pos[2], dtype=np.float32)
        return terrain_surface_height(
            env.model.mj_model, int(env.model.terrain_geom_ids[0]), xy
        )

    return sole_support_positions(
        geom_position=position,
        geom_rotation=np.asarray(mj_data.geom_xmat[ids], dtype=np.float32).reshape(
            (-1, 3, 3)
        ),
        geom_size=np.asarray(env.model.mj_model.geom_size[ids], dtype=np.float32),
        geom_type=np.asarray(env.model.mj_model.geom_type[ids]),
        geom_body_ids=np.asarray(env.model.geom_body_ids)[ids],
        foot_body_ids=np.asarray(env.model.foot_body_ids),
        terrain_up=mujoco_terrain_surface_normal(env, position[:, :2]),
        terrain_height=terrain_height,
        array_module=np,
    )


def mujoco_terrain_surface_normal(
    env: BelloLocomotionEnv, world_xy: np.ndarray
) -> np.ndarray:
    world_xy = np.asarray(world_xy, dtype=np.float32)
    output_shape = world_xy.shape[:-1] + (3,)
    if env.terrain_config.mode == "none":
        return np.broadcast_to(
            np.asarray([0.0, 0.0, 1.0], dtype=np.float32), output_shape
        )
    if env.terrain_config.mode == "plane":
        terrain_geom_id = int(env.model.terrain_geom_ids[0])
        terrain_up = quat_rotate_wxyz(
            np.asarray(env.model.mj_model.geom_quat[terrain_geom_id], dtype=np.float32),
            np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        )
        return np.broadcast_to(terrain_up, output_shape)

    terrain_geom_id = int(env.model.terrain_geom_ids[0])
    return heightfield_surface_normal(
        world_xy,
        lambda xy: terrain_surface_height(env.model.mj_model, terrain_geom_id, xy),
        array_module=np,
    )


def mujoco_foot_linear_velocity(
    env: BelloLocomotionEnv, mj_data: mujoco.MjData
) -> np.ndarray:
    if env.model.foot_body_ids.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    foot_world_vel = []
    for body_id in np.asarray(env.model.foot_body_ids, dtype=np.int32):
        velocity = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            env.model.mj_model,
            mj_data,
            mujoco.mjtObj.mjOBJ_XBODY,
            int(body_id),
            velocity,
            0,
        )
        foot_world_vel.append(velocity[3:6])
    return np.asarray(foot_world_vel, dtype=np.float32)


def mujoco_foot_angular_speed(
    env: BelloLocomotionEnv, mj_data: mujoco.MjData
) -> np.ndarray:
    if env.model.foot_body_ids.size == 0:
        return np.zeros((0,), dtype=np.float32)
    foot_body_ids = np.asarray(env.model.foot_body_ids, dtype=np.int32)
    foot_ang_vel = np.asarray(mj_data.cvel[foot_body_ids, :3], dtype=np.float32)
    return np.linalg.norm(foot_ang_vel, axis=1).astype(np.float32)


def mujoco_undesired_contact(
    env: BelloLocomotionEnv, mj_data: mujoco.MjData
) -> np.float32:
    if env.model.terrain_geom_ids.size == 0:
        return np.float32(0.0)
    terrain_geom_ids = set(int(geom_id) for geom_id in env.model.terrain_geom_ids)
    nonfoot_geom_ids = set(int(geom_id) for geom_id in env.model.nonfoot_robot_geom_ids)
    nonfoot_body_ids = set(int(body_id) for body_id in env.model.nonfoot_robot_body_ids)
    geom_body_ids = np.asarray(env.model.geom_body_ids, dtype=np.int32)
    force_threshold = float(env.config.contact.undesired_contact_force_threshold)
    contacted_body_ids: set[int] = set()
    for contact_index in range(int(mj_data.ncon)):
        contact = mj_data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if geom1 in terrain_geom_ids:
            other_geom = geom2
        elif geom2 in terrain_geom_ids:
            other_geom = geom1
        else:
            continue
        if other_geom not in nonfoot_geom_ids:
            continue
        normal_force = mujoco_contact_normal_force(
            env.model.mj_model, mj_data, contact_index
        )
        if normal_force <= force_threshold:
            continue
        body_id = int(geom_body_ids[other_geom])
        if body_id in nonfoot_body_ids:
            contacted_body_ids.add(body_id)
    return np.float32(len(contacted_body_ids))


def mujoco_self_collision(
    env: BelloLocomotionEnv, mj_data: mujoco.MjData
) -> np.float32:
    robot_geom_ids = set(int(geom_id) for geom_id in env.model.robot_geom_ids)
    if not robot_geom_ids:
        return np.float32(0.0)
    force_threshold = float(env.config.contact.self_collision_force_threshold)
    count = 0
    for contact_index in range(int(mj_data.ncon)):
        contact = mj_data.contact[contact_index]
        if int(contact.geom1) not in robot_geom_ids:
            continue
        if int(contact.geom2) not in robot_geom_ids:
            continue
        normal_force = mujoco_contact_normal_force(
            env.model.mj_model, mj_data, contact_index
        )
        if normal_force > force_threshold:
            count += 1
    return np.float32(count)


def mujoco_contact_normal_force(
    mj_model: mujoco.MjModel, mj_data: mujoco.MjData, contact_index: int
) -> float:
    contact_force = np.zeros(6, dtype=np.float64)
    mujoco.mj_contactForce(mj_model, mj_data, contact_index, contact_force)
    return abs(float(contact_force[0]))
