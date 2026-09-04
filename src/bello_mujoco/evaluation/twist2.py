"""Deterministic held-out evaluation and video for typed TWIST2 policies."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import jax
import mujoco
import numpy as np

from bello_mujoco.envs.twist2 import BelloTwist2Env
from bello_mujoco.evaluation.render import (
    CameraConfig,
    open_rollout_video_resources,
    update_camera,
)
from bello_mujoco.evaluation.twist import _evaluate_clip
from bello_mujoco.motion.twist_dataset import materialize_packed_clip
from bello_mujoco.policy import apply_actor
from bello_mujoco.training.checkpoint import (
    Twist2CheckpointMetadata,
    restore_evaluation_checkpoint,
)
from bello_mujoco.training.ppo import normalize_observation


@dataclass(frozen=True)
class Twist2EvaluationConfig:
    output_path: Path = Path("outputs/evaluation/twist2.json")
    video_path: Path | None = None
    runtime_clip_directory: Path = Path("outputs/twist2_runtime_clips/evaluation")
    stop_on_terminal: bool = True
    video_width: int = 640
    video_height: int = 480
    video_fps: int = 50
    video_max_steps: int = 500
    camera: CameraConfig = CameraConfig()


def evaluate_twist2_checkpoint(
    checkpoint_path: Path,
    config: Twist2EvaluationConfig = Twist2EvaluationConfig(),
    *,
    allow_git_commit_mismatch: bool = False,
) -> dict[str, Any]:
    train_state, metadata, normalization = restore_evaluation_checkpoint(
        checkpoint_path,
        allow_git_commit_mismatch=allow_git_commit_mismatch,
    )
    if not isinstance(metadata, Twist2CheckpointMetadata):
        raise ValueError("TWIST2 evaluation requires a typed twist2 checkpoint")
    index_path = metadata.dataset.index_path.expanduser().resolve()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index["dataset_fingerprint"] != metadata.dataset.dataset_fingerprint:
        raise ValueError("evaluation dataset fingerprint differs from checkpoint")
    if index["robot_fingerprint"] != metadata.robot_fingerprint:
        raise ValueError("evaluation robot fingerprint differs from checkpoint")
    heldout = sorted(
        (clip for clip in index["clips"] if clip["split"] == "evaluation"),
        key=lambda clip: (clip["source_id"], clip["sequence_id"]),
    )
    if not heldout:
        raise ValueError("TWIST2 evaluation dataset contains no held-out clips")
    clip_results = []
    video_clip = None
    for clip in heldout:
        runtime_path = _runtime_clip_path(config, metadata, clip)
        materialize_packed_clip(
            index_path,
            runtime_path,
            split="evaluation",
            source_id=clip["source_id"],
            sequence_id=clip["sequence_id"],
        )
        env = BelloTwist2Env.load(
            metadata.asset,
            metadata.simulation,
            motion_path=runtime_path,
            config=metadata.twist2_env,
            warp=metadata.warp,
            batch_size=1,
            sampling=metadata.sampling,
        )
        result = _evaluate_clip(
            env,
            train_state.policy,
            normalization.observation,
            metadata.ppo,
            stop_on_terminal=config.stop_on_terminal,
        )
        clip_results.append({**clip, **result})
        if config.video_path is not None and video_clip is None:
            video_clip = (env, clip)
    video = None
    if video_clip is not None and config.video_path is not None:
        env, clip = video_clip
        steps = min(
            config.video_max_steps,
            env.config.max_episode_steps,
            max(1, int(np.ceil(env.motion.duration / env.control_timestep))),
        )
        video_path = config.video_path.expanduser().resolve()
        _render_twist2_rollout(
            env,
            train_state.policy,
            normalization.observation,
            metadata.ppo,
            video_path,
            steps=steps,
            config=config,
        )
        video = {
            "path": str(video_path),
            "source_id": clip["source_id"],
            "sequence_id": clip["sequence_id"],
            "steps": steps,
        }
    report = _aggregate_results(metadata, clip_results, video)
    output_path = config.output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _runtime_clip_path(config, metadata, clip) -> Path:
    return (
        config.runtime_clip_directory
        / metadata.dataset.dataset_fingerprint.removeprefix("sha256:")
        / f"{clip['source_id']}__{clip['sequence_id']}.npz"
    )


def _render_twist2_rollout(
    env: BelloTwist2Env,
    policy,
    observation_norm_state,
    ppo_config,
    output_path: Path,
    *,
    steps: int,
    config: Twist2EvaluationConfig,
) -> None:
    state, observation = env.reset()
    mj_model = env.model.mj_model
    mj_data = mujoco.MjData(mj_model)
    resources, renderer, camera, writer = open_rollout_video_resources(
        mj_model,
        output_path,
        width=config.video_width,
        height=config.video_height,
        fps=config.video_fps,
        camera_config=config.camera,
    )
    try:
        for _ in range(steps):
            mj_data.qpos[:] = np.asarray(
                jax.device_get(state.motion_state.data.qpos), dtype=np.float64
            )
            mj_data.qvel[:] = np.asarray(
                jax.device_get(state.motion_state.data.qvel), dtype=np.float64
            )
            mujoco.mj_forward(mj_model, mj_data)
            update_camera(
                camera,
                mj_model,
                mj_data,
                config.camera,
                root_body_name=env.model.robot.root_body_name,
            )
            renderer.update_scene(mj_data, camera=camera)
            writer.append_data(renderer.render())
            policy_observation = normalize_observation(
                observation, observation_norm_state, ppo_config
            )
            action = apply_actor(policy, policy_observation)
            state, observation, _, done, info = env.step(state, action)
            if bool(done) and (
                bool(info["done/timeout"])
                or (bool(info["done/terminal"]) and config.stop_on_terminal)
            ):
                break
    finally:
        resources.close()


def _aggregate_results(metadata, clips, video) -> dict[str, Any]:
    metric_names = (
        "episode_return",
        "root_position_rms_m",
        "root_orientation_rms_deg",
        "joint_position_rms_rad",
        "foot_slip_mean_mps",
    )
    global_metrics = {
        name: float(np.mean([clip[name] for clip in clips])) for name in metric_names
    }
    sources = {}
    for source_id in sorted({clip["source_id"] for clip in clips}):
        selected = [clip for clip in clips if clip["source_id"] == source_id]
        sources[source_id] = {
            "clips": len(selected),
            **{
                name: float(np.mean([clip[name] for clip in selected]))
                for name in metric_names
            },
        }
    return {
        "task": "twist2",
        "dataset_fingerprint": metadata.dataset.dataset_fingerprint,
        "robot_fingerprint": metadata.robot_fingerprint,
        "upstream_revision": metadata.parity_contract.upstream_revision,
        "global": global_metrics,
        "per_source": sources,
        "per_clip": clips,
        "video": video,
    }
