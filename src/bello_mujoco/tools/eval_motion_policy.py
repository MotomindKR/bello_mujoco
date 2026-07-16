"""Render a deterministic rollout from a motion-tracking PPO checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import jax
import mujoco
import numpy as np

from bello_mujoco.checkpoint import restore_training_checkpoint
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
from bello_mujoco.train.ppo import (
    ObservationNormState,
    PPOConfig,
    init_observation_norm_state,
    normalize_observation,
)
from bello_mujoco.render import (
    CameraConfig,
    make_camera,
    update_camera,
)
from bello_mujoco.video import Mp4Writer


class MotionRolloutStats(TypedDict):
    rendered_steps: int
    episode_return: float
    terminal: bool
    timeout: bool
    completion_fraction: float
    root_position_rms_m: float
    root_orientation_rms_deg: float
    joint_position_rms_rad: float


@dataclass(frozen=True)
class _NativeContactData:
    contact__geom: jax.Array
    nacon: jax.Array
    contact__worldid: jax.Array
    contact__efc_address: jax.Array
    efc__force: jax.Array


@dataclass(frozen=True)
class _NativeMotionData:
    qpos: jax.Array
    qvel: jax.Array
    ctrl: jax.Array
    xpos: jax.Array
    xquat: jax.Array
    cvel: jax.Array
    subtree_com: jax.Array
    actuator_force: jax.Array
    _impl: _NativeContactData


def render_checkpoint(
    checkpoint: Path,
    output_path: Path,
    *,
    steps: int | None = None,
    stop_on_terminal: bool = True,
    width: int = 640,
    height: int = 480,
    fps: int = 50,
) -> Path:
    render_checkpoint_with_stats(
        checkpoint,
        output_path,
        steps=steps,
        stop_on_terminal=stop_on_terminal,
        width=width,
        height=height,
        fps=fps,
    )
    return output_path


def render_checkpoint_with_stats(
    checkpoint: Path,
    output_path: Path,
    *,
    steps: int | None = None,
    stop_on_terminal: bool = True,
    width: int = 640,
    height: int = 480,
    fps: int = 50,
) -> MotionRolloutStats:
    train_state, metadata = restore_training_checkpoint(checkpoint)
    if metadata.task != "motion_tracking" or metadata.motion_env is None:
        raise ValueError(f"checkpoint task is {metadata.task!r}, not motion_tracking")
    env = BelloMotionTrackingEnv.load_native(
        metadata.asset,
        metadata.motion_env,
        use_viewer_model=True,
    )
    return render_motion_policy(
        env,
        train_state.policy,
        output_path,
        steps=steps or env.max_episode_steps,
        stop_on_terminal=stop_on_terminal,
        width=width,
        height=height,
        fps=fps,
        obs_stack_size=metadata.ppo.policy.obs_stack_size,
        ppo_config=metadata.ppo,
        observation_norm_state=observation_norm_state_from_metadata(
            metadata.observation_norm, metadata.obs_size
        ),
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
    ppo_config: PPOConfig | None = None,
    observation_norm_state: ObservationNormState | None = None,
) -> MotionRolloutStats:
    mj_model = env.model.mj_model
    mj_data = mujoco.MjData(mj_model)
    state, observation = reset_native_motion_rollout(env, mj_data)
    observation_stack = init_observation_stack(observation, obs_stack_size)
    ppo_config = PPOConfig() if ppo_config is None else ppo_config
    if observation_norm_state is None:
        observation_norm_state = init_observation_norm_state(
            int(observation_stack.size)
        )
    renderer = mujoco.Renderer(mj_model, width=width, height=height)
    camera_config = CameraConfig(
        mode="follow", distance=3.0, azimuth=135.0, elevation=-16.0
    )
    camera = make_camera(mj_model, camera_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = Mp4Writer(output_path, width=width, height=height, fps=fps)
    episode_return = 0.0
    rendered_steps = 0
    terminal = False
    timeout = False
    root_position_errors = []
    root_orientation_errors = []
    joint_position_errors = []
    try:
        for _ in range(steps):
            reference = env._aligned_reference(
                state.reference_time, state.alignment_xy, state.alignment_yaw
            )
            root = env.model.root_qpos_adr
            policy_qpos = jax.numpy.asarray(env.model.policy_qpos_indices)
            if env.config.formulation == "beyondmimic":
                actual_position = state.data.xpos[env.anchor_body_id]
                actual_orientation = state.data.xquat[env.anchor_body_id]
                target_position = reference.body_pos[env.anchor_body_index]
                target_orientation = reference.body_quat[env.anchor_body_index]
            else:
                actual_position = state.data.qpos[root : root + 3]
                actual_orientation = state.data.qpos[root + 3 : root + 7]
                target_position = reference.qpos[root : root + 3]
                target_orientation = reference.qpos[root + 3 : root + 7]
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
            update_camera(camera, mj_model, mj_data, camera_config)
            renderer.update_scene(mj_data, camera=camera)
            writer.append_data(renderer.render())
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
        writer.close()
        renderer.close()
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
    source = env._aligned_reference(
        reference_time,
        jax.numpy.zeros(2, dtype=jax.numpy.float32),
        jax.numpy.asarray(0.0, dtype=jax.numpy.float32),
    )
    alignment_xy = -source.qpos[:2]
    alignment_yaw = jax.numpy.asarray(0.0, dtype=jax.numpy.float32)
    reference = env._aligned_reference(reference_time, alignment_xy, alignment_yaw)
    joint_default_offset = jax.numpy.zeros(env.action_size, dtype=jax.numpy.float32)
    mj_data.qpos[:] = np.asarray(reference.qpos)
    mj_data.qvel[:] = np.asarray(reference.qvel)
    mj_data.ctrl[:] = np.asarray(
        env._initial_ctrl(reference, joint_default_offset), dtype=np.float64
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
    mj_data.ctrl[:] = np.asarray(env._action_to_ctrl(state, action), dtype=np.float64)
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
    reward, reward_info = env._reward(state, stepped)
    done_flags = env._done_flags(stepped)
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
    impl = _NativeContactData(
        contact__geom=jax.numpy.asarray(contact_geom),
        nacon=jax.numpy.asarray([ncon], dtype=jax.numpy.int32),
        contact__worldid=jax.numpy.zeros(ncon, dtype=jax.numpy.int32),
        contact__efc_address=jax.numpy.asarray(contact_address[:, None]),
        efc__force=jax.numpy.asarray(efc_force),
    )
    data = _NativeMotionData(
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
    return MotionTrackingState(
        data=data,
        last_action=jax.numpy.asarray(last_action, dtype=jax.numpy.float32),
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


def observation_norm_state_from_metadata(metadata, size: int) -> ObservationNormState:
    if metadata is None:
        return init_observation_norm_state(size)
    mean = jax.numpy.asarray(metadata.mean, dtype=jax.numpy.float32)
    var = jax.numpy.asarray(metadata.var, dtype=jax.numpy.float32)
    if mean.shape != (size,) or var.shape != (size,):
        raise ValueError("checkpoint actor observation normalization shape mismatch")
    return ObservationNormState(
        mean=mean,
        var=var,
        count=jax.numpy.asarray(metadata.count, dtype=jax.numpy.float32),
    )


def entrypoint() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument(
        "--full-clip",
        action="store_true",
        help="continue the rollout after early-termination flags until the motion ends",
    )
    args = parser.parse_args()
    print(
        render_checkpoint(
            args.checkpoint,
            args.output,
            steps=args.steps,
            stop_on_terminal=not args.full_clip,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
    )


if __name__ == "__main__":
    entrypoint()
