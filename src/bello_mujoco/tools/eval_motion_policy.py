"""Render a deterministic rollout from a motion-tracking PPO checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TypedDict

import jax
import mujoco
import numpy as np

from bello_mujoco.checkpoint import restore_training_checkpoint
from bello_mujoco.envs.motion_tracking import BelloMotionTrackingEnv, quat_distance
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
    sync_mj_data,
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


def render_checkpoint(
    checkpoint: Path,
    output_path: Path,
    *,
    steps: int | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 50,
) -> Path:
    render_checkpoint_with_stats(
        checkpoint,
        output_path,
        steps=steps,
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
    width: int = 640,
    height: int = 480,
    fps: int = 50,
) -> MotionRolloutStats:
    train_state, metadata = restore_training_checkpoint(checkpoint)
    if metadata.task != "motion_tracking" or metadata.motion_env is None:
        raise ValueError(f"checkpoint task is {metadata.task!r}, not motion_tracking")
    env = BelloMotionTrackingEnv.load(
        metadata.asset,
        metadata.motion_env,
        warp=metadata.warp,
        batch_size=1,
        use_viewer_model=True,
    )
    return render_motion_policy(
        env,
        train_state.policy,
        output_path,
        steps=steps or env.max_episode_steps,
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
    width: int,
    height: int,
    fps: int,
    obs_stack_size: int,
    ppo_config: PPOConfig | None = None,
    observation_norm_state: ObservationNormState | None = None,
) -> MotionRolloutStats:
    state, observation = env.reset(reference_time=jax.numpy.asarray(0.0))
    observation_stack = init_observation_stack(observation, obs_stack_size)
    ppo_config = PPOConfig() if ppo_config is None else ppo_config
    if observation_norm_state is None:
        observation_norm_state = init_observation_norm_state(
            int(observation_stack.size)
        )
    mj_model = env.model.mj_model
    mj_data = mujoco.MjData(mj_model)
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
            sync_mj_data(mj_model, mj_data, state)
            update_camera(camera, mj_model, mj_data, camera_config)
            renderer.update_scene(mj_data, camera=camera)
            writer.append_data(renderer.render())
            policy_observation = normalize_observation(
                stacked_observation(observation_stack),
                observation_norm_state,
                ppo_config,
            )
            action = apply_actor(policy, policy_observation)
            state, observation, reward, done, info = env.step(state, action)
            episode_return += float(jax.device_get(reward))
            rendered_steps += 1
            observation_stack = update_observation_stack(
                observation_stack, observation, done
            )
            if bool(jax.device_get(done)):
                terminal = bool(jax.device_get(info["done/terminal"]))
                timeout = bool(jax.device_get(info["done/timeout"]))
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
    args = parser.parse_args()
    print(
        render_checkpoint(
            args.checkpoint,
            args.output,
            steps=args.steps,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
    )


if __name__ == "__main__":
    entrypoint()
