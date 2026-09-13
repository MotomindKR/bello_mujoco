"""Native motion checkpoint, reference, and worker evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import jax
import mujoco
import numpy as np

from bello_mujoco.envs.motion_tracking import (
    BelloMotionTrackingEnv,
    MotionTrackingState,
    quat_distance,
)
from bello_mujoco.policy import (
    apply_actor,
    init_observation_stack,
    stacked_observation,
    update_observation_stack,
)
from bello_mujoco.training.ppo_types import ObservationNormState
from bello_mujoco.training.ppo_config import PPOConfig
from bello_mujoco.training.normalization import normalize_observation
from bello_mujoco.evaluation.video import (
    CameraConfig,
    open_rollout_video_resources,
    write_video_frame,
)
from bello_mujoco.motion.clip import (
    load_motion_clip,
    motion_tracking_asset_config,
    sample_motion,
)
from bello_mujoco.robot import AssetName
from bello_mujoco.motion.native import NativeContactData, NativeMotionData
from bello_mujoco.motion.runtime import load_motion_policy_runtime
from bello_mujoco.sim import load_model
from bello_mujoco.terrain import FLAT_TERRAIN
from bello_mujoco.evaluation.software import (
    run_software_evaluation,
    validate_software_renderer,
)


class MotionRolloutStats(TypedDict):
    rendered_steps: int
    episode_return: float
    terminal: bool
    timeout: bool
    completion_fraction: float
    root_position_rms_m: float
    root_orientation_rms_deg: float
    joint_position_rms_rad: float


def render_checkpoint_with_stats(
    checkpoint: Path,
    output_path: Path,
    *,
    steps: int | None = None,
    stop_on_terminal: bool = True,
    width: int = 640,
    height: int = 480,
    fps: int = 50,
    end_effector_payload_mass_kg: float | None = None,
) -> MotionRolloutStats:
    runtime = load_motion_policy_runtime(
        checkpoint,
        end_effector_payload_mass_kg=end_effector_payload_mass_kg,
        use_viewer_model=True,
    )
    metadata = runtime.metadata
    return render_motion_policy(
        runtime.env,
        runtime.policy,
        output_path,
        steps=runtime.env.max_episode_steps if steps is None else steps,
        stop_on_terminal=stop_on_terminal,
        width=width,
        height=height,
        fps=fps,
        obs_stack_size=metadata.ppo.policy.obs_stack_size,
        ppo_config=metadata.ppo,
        observation_norm_state=runtime.normalization.observation,
    )


def render_motion_policy(
    env: BelloMotionTrackingEnv,
    policy,
    output_path: Path,
    *,
    steps: int,
    stop_on_terminal: bool = True,
    width: int,
    height: int,
    fps: int,
    obs_stack_size: int,
    ppo_config: PPOConfig,
    observation_norm_state: ObservationNormState,
) -> MotionRolloutStats:
    if steps <= 0:
        raise ValueError("steps must be positive")
    mj_model = env.model.mj_model
    mj_data = mujoco.MjData(mj_model)
    state, observation = reset_native_motion_rollout(env, mj_data)
    observation_stack = init_observation_stack(observation, obs_stack_size)
    camera_config = CameraConfig(
        mode="follow", distance=3.0, azimuth=135.0, elevation=-16.0
    )
    resources, renderer, camera, writer = open_rollout_video_resources(
        mj_model,
        output_path,
        width=width,
        height=height,
        fps=fps,
        camera_config=camera_config,
    )
    episode_return = 0.0
    rendered_steps = 0
    terminal = False
    timeout = False
    root_position_errors = []
    root_orientation_errors = []
    joint_position_errors = []
    try:
        for _ in range(steps):
            reference = env.aligned_reference(
                state.reference_time, state.alignment_xy, state.alignment_yaw
            )
            policy_qpos = jax.numpy.asarray(env.model.policy_qpos_indices)
            actual_position = state.data.xpos[env.anchor_body_id]
            actual_orientation = state.data.xquat[env.anchor_body_id]
            target_position = reference.body_pos[env.anchor_body_index]
            target_orientation = reference.body_quat[env.anchor_body_index]
            root_position_errors.append(
                float(
                    jax.device_get(
                        jax.numpy.linalg.norm(actual_position - target_position)
                    )
                )
            )
            root_orientation_errors.append(
                float(
                    jax.device_get(
                        quat_distance(actual_orientation, target_orientation)
                    )
                )
            )
            joint_position_errors.append(
                np.asarray(
                    jax.device_get(
                        state.data.qpos[policy_qpos] - reference.qpos[policy_qpos]
                    )
                )
            )
            write_video_frame(
                renderer,
                camera,
                writer,
                mj_model,
                mj_data,
                camera_config,
                root_body_name=env.model.robot.root_body_name,
            )
            policy_observation = normalize_observation(
                stacked_observation(observation_stack),
                observation_norm_state,
                ppo_config,
            )
            action = apply_actor(policy, policy_observation)
            state, observation, reward, done, info = native_motion_step(
                env, mj_data, state, action
            )
            episode_return += float(jax.device_get(reward))
            rendered_steps += 1
            observation_stack = update_observation_stack(
                observation_stack, observation, done
            )
            if bool(jax.device_get(done)):
                step_terminal = bool(jax.device_get(info["done/terminal"]))
                step_timeout = bool(jax.device_get(info["done/timeout"]))
                nonfinite = bool(jax.device_get(info["done/nonfinite"]))
                terminal = terminal or step_terminal
                timeout = timeout or step_timeout
                if step_timeout or nonfinite or (stop_on_terminal and step_terminal):
                    break
    finally:
        resources.close()
    return MotionRolloutStats(
        rendered_steps=rendered_steps,
        episode_return=episode_return,
        terminal=terminal,
        timeout=timeout,
        completion_fraction=min(rendered_steps / env.max_episode_steps, 1.0),
        root_position_rms_m=float(np.sqrt(np.mean(np.square(root_position_errors)))),
        root_orientation_rms_deg=float(
            np.degrees(np.sqrt(np.mean(np.square(root_orientation_errors))))
        ),
        joint_position_rms_rad=float(
            np.sqrt(np.mean(np.square(joint_position_errors)))
        ),
    )


def reset_native_motion_rollout(
    env: BelloMotionTrackingEnv, mj_data: mujoco.MjData
) -> tuple[MotionTrackingState, jax.Array]:
    reference_time = jax.numpy.asarray(0.0, dtype=jax.numpy.float32)
    source = env.aligned_reference(
        reference_time,
        jax.numpy.zeros(2, dtype=jax.numpy.float32),
        jax.numpy.asarray(0.0, dtype=jax.numpy.float32),
    )
    alignment_xy = -source.qpos[:2]
    alignment_yaw = jax.numpy.asarray(0.0, dtype=jax.numpy.float32)
    reference = env.aligned_reference(reference_time, alignment_xy, alignment_yaw)
    joint_default_offset = jax.numpy.zeros(env.action_size, dtype=jax.numpy.float32)
    mj_data.qpos[:] = np.asarray(reference.qpos)
    mj_data.qvel[:] = np.asarray(reference.qvel)
    mj_data.ctrl[:] = np.asarray(
        env._initial_ctrl(joint_default_offset), dtype=np.float64
    )
    mujoco.mj_forward(env.model.mj_model, mj_data)
    state = native_motion_state(
        env,
        mj_data,
        last_action=np.zeros(env.action_size, dtype=np.float32),
        reference_time=reference_time,
        alignment_xy=alignment_xy,
        alignment_yaw=alignment_yaw,
        joint_default_offset=joint_default_offset,
        step_count=0,
        episode_return=0.0,
    )
    return state, env.observation(state)


def native_motion_step(
    env: BelloMotionTrackingEnv,
    mj_data: mujoco.MjData,
    state: MotionTrackingState,
    action: jax.Array,
) -> tuple[MotionTrackingState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
    action = jax.numpy.clip(
        jax.numpy.asarray(action), -env.action_clip, env.action_clip
    )
    mj_data.ctrl[:] = np.asarray(env.action_target(state, action), dtype=np.float64)
    for _ in range(env.decimation):
        mujoco.mj_step(env.model.mj_model, mj_data)
    next_reference_time = state.reference_time + env.control_timestep
    stepped = native_motion_state(
        env,
        mj_data,
        last_action=np.asarray(action, dtype=np.float32),
        reference_time=next_reference_time,
        alignment_xy=state.alignment_xy,
        alignment_yaw=state.alignment_yaw,
        joint_default_offset=state.joint_default_offset,
        step_count=int(state.step_count) + 1,
        episode_return=float(state.episode_return),
    )
    reward, reward_info = env._reward(
        state, stepped, jax.numpy.asarray(0, dtype=jax.numpy.int32)
    )
    done_flags = env.termination_flags(stepped)
    reward, reward_info = env._sanitize_nonfinite_reward(
        reward, reward_info, done_flags
    )
    episode_return = state.episode_return + reward
    stepped = stepped.replace(episode_return=episode_return)
    observation = env.observation(stepped)
    return (
        stepped,
        observation,
        reward,
        done_flags["done/any"],
        {
            **done_flags,
            **reward_info,
        },
    )


def native_motion_state(
    env: BelloMotionTrackingEnv,
    mj_data: mujoco.MjData,
    *,
    last_action: np.ndarray,
    reference_time: jax.Array,
    alignment_xy: jax.Array,
    alignment_yaw: jax.Array,
    joint_default_offset: jax.Array,
    step_count: int,
    episode_return: float,
) -> MotionTrackingState:
    ncon = int(mj_data.ncon)
    nefc = int(mj_data.nefc)
    contact_geom = np.asarray(mj_data.contact.geom[:ncon], dtype=np.int32).copy()
    contact_address = np.asarray(
        mj_data.contact.efc_address[:ncon], dtype=np.int32
    ).copy()
    efc_force = np.zeros(max(nefc, 1), dtype=np.float32)
    if nefc:
        efc_force[:nefc] = np.asarray(mj_data.efc_force[:nefc], dtype=np.float32)
    impl = NativeContactData(
        contact__geom=jax.numpy.asarray(contact_geom),
        nacon=jax.numpy.asarray([ncon], dtype=jax.numpy.int32),
        contact__worldid=jax.numpy.zeros(ncon, dtype=jax.numpy.int32),
        contact__efc_address=jax.numpy.asarray(contact_address[:, None]),
        efc__force=jax.numpy.asarray(efc_force),
    )
    data = NativeMotionData(
        qpos=jax.numpy.asarray(
            np.asarray(mj_data.qpos).copy(), dtype=jax.numpy.float32
        ),
        qvel=jax.numpy.asarray(
            np.asarray(mj_data.qvel).copy(), dtype=jax.numpy.float32
        ),
        ctrl=jax.numpy.asarray(
            np.asarray(mj_data.ctrl).copy(), dtype=jax.numpy.float32
        ),
        xpos=jax.numpy.asarray(
            np.asarray(mj_data.xpos).copy(), dtype=jax.numpy.float32
        ),
        xquat=jax.numpy.asarray(
            np.asarray(mj_data.xquat).copy(), dtype=jax.numpy.float32
        ),
        cvel=jax.numpy.asarray(
            np.asarray(mj_data.cvel).copy(), dtype=jax.numpy.float32
        ),
        subtree_com=jax.numpy.asarray(
            np.asarray(mj_data.subtree_com).copy(), dtype=jax.numpy.float32
        ),
        actuator_force=jax.numpy.asarray(
            np.asarray(mj_data.actuator_force).copy(), dtype=jax.numpy.float32
        ),
        _impl=impl,
    )
    actor_feedback = env._actor_feedback(data)
    return MotionTrackingState(
        data=data,
        last_action=jax.numpy.asarray(last_action, dtype=jax.numpy.float32),
        actor_feedback_history=jax.numpy.repeat(
            actor_feedback[None, :],
            env.maximum_observation_delay_steps + 1,
            axis=0,
        ),
        actor_feedback_delay_steps=jax.numpy.asarray(0, dtype=jax.numpy.int32),
        reference_time=reference_time,
        alignment_xy=alignment_xy,
        alignment_yaw=alignment_yaw,
        joint_default_offset=joint_default_offset,
        next_push_step=jax.numpy.asarray(0, dtype=jax.numpy.int32),
        sampling_failure_ema=jax.numpy.zeros(
            env.sampling_bin_count, dtype=jax.numpy.float32
        ),
        step_count=jax.numpy.asarray(step_count, dtype=jax.numpy.int32),
        episode_return=jax.numpy.asarray(episode_return, dtype=jax.numpy.float32),
        episode_length=jax.numpy.asarray(step_count, dtype=jax.numpy.int32),
        last_episode_return=jax.numpy.asarray(0.0, dtype=jax.numpy.float32),
        last_episode_length=jax.numpy.asarray(0, dtype=jax.numpy.int32),
    )


def render_reference(
    motion_path: Path,
    output_path: Path,
    *,
    asset: AssetName,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    end_effector_payload_mass_kg: float = 0.0,
) -> Path:
    validate_software_renderer()
    model = load_model(
        motion_tracking_asset_config(
            asset=asset, end_effector_payload_mass_kg=end_effector_payload_mass_kg
        ),
        terrain=FLAT_TERRAIN,
        use_viewer_model=True,
        load_mjx=False,
    )
    motion = load_motion_clip(motion_path, model)
    mj_model = model.mj_model
    mj_data = mujoco.MjData(mj_model)
    camera_config = CameraConfig(
        mode="follow", distance=3.0, azimuth=135.0, elevation=-16.0
    )
    resources, renderer, camera, writer = open_rollout_video_resources(
        mj_model,
        output_path,
        width=width,
        height=height,
        fps=fps,
        camera_config=camera_config,
    )
    frame_count = int(np.floor(motion.duration * fps)) + 1
    try:
        for frame in range(frame_count):
            reference = sample_motion(motion, min(frame / fps, motion.duration))
            mj_data.qpos[:] = np.asarray(reference.qpos)
            mj_data.qvel[:] = np.asarray(reference.qvel)
            mujoco.mj_forward(mj_model, mj_data)
            write_video_frame(
                renderer,
                camera,
                writer,
                mj_model,
                mj_data,
                camera_config,
                root_body_name=model.robot.root_body_name,
            )
    finally:
        resources.close()
    return output_path


@dataclass(frozen=True)
class MotionEvaluationRequest:
    checkpoint: Path
    output_path: Path
    steps: int
    width: int
    height: int
    fps: int


def evaluate_motion_video(request: MotionEvaluationRequest) -> MotionRolloutStats:
    return run_software_evaluation(
        render_checkpoint_with_stats,
        request.checkpoint,
        request.output_path,
        steps=request.steps,
        width=request.width,
        height=request.height,
        fps=request.fps,
    )
