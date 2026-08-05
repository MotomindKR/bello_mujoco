"""Deterministic held-out MJX evaluation for TWIST students."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from bello_mujoco.envs.twist import (
    BelloTwistEnv,
    _quat_distance,
)
from bello_mujoco.motion.twist_dataset import materialize_packed_clip
from bello_mujoco.policy import apply_actor
from bello_mujoco.training.checkpoint import (
    TwistStudentCheckpointMetadata,
    restore_evaluation_checkpoint,
)
from bello_mujoco.training.ppo import normalize_observation


@dataclass(frozen=True)
class TwistEvaluationConfig:
    output_path: Path = Path("outputs/evaluation/twist_student.json")
    runtime_clip_directory: Path = Path("outputs/twist_runtime_clips/evaluation")
    stop_on_terminal: bool = True


def evaluate_twist_student_checkpoint(
    checkpoint_path: Path,
    config: TwistEvaluationConfig = TwistEvaluationConfig(),
    *,
    allow_git_commit_mismatch: bool = False,
) -> dict[str, Any]:
    train_state, metadata, normalization = restore_evaluation_checkpoint(
        checkpoint_path,
        allow_git_commit_mismatch=allow_git_commit_mismatch,
    )
    if not isinstance(metadata, TwistStudentCheckpointMetadata):
        raise ValueError(
            "public TWIST evaluation accepts only a typed twist_student checkpoint"
        )
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
        raise ValueError("TWIST evaluation dataset contains no held-out clips")
    clip_results = []
    for clip in heldout:
        runtime_path = (
            config.runtime_clip_directory
            / metadata.dataset.dataset_fingerprint.removeprefix("sha256:")
            / f"{clip['source_id']}__{clip['sequence_id']}.npz"
        )
        materialize_packed_clip(
            index_path,
            runtime_path,
            split="evaluation",
            source_id=clip["source_id"],
            sequence_id=clip["sequence_id"],
        )
        env = BelloTwistEnv.load(
            "student",
            metadata.asset,
            metadata.simulation,
            motion_path=runtime_path,
            config=metadata.twist_env,
            warp=metadata.warp,
            batch_size=1,
        )
        result = _evaluate_clip(
            env,
            train_state.policy,
            normalization.observation,
            metadata.ppo,
            stop_on_terminal=config.stop_on_terminal,
        )
        clip_results.append({**clip, **result})
    report = _aggregate_results(metadata, clip_results)
    output_path = config.output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _evaluate_clip(
    env: BelloTwistEnv,
    policy,
    observation_norm_state,
    ppo_config,
    *,
    stop_on_terminal: bool,
) -> dict[str, Any]:
    state, observation = env.reset()
    root_errors = []
    orientation_errors = []
    joint_errors = []
    foot_slips = []
    rewards = []
    reward_terms: dict[str, list[float]] = {}
    termination_counts: dict[str, int] = {}
    terminal = False
    timeout = False
    motion_steps = max(1, int(np.ceil(env.motion.duration / env.control_timestep)))
    max_steps = min(env.config.max_episode_steps, motion_steps)
    for _ in range(max_steps):
        motion_state = state.motion_state
        reference = env._aligned_motion_sample(
            motion_state.reference_time,
            motion_state.alignment_xy,
            motion_state.alignment_yaw,
        )
        root_errors.append(
            float(jnp.linalg.norm(motion_state.data.qpos[:3] - reference.qpos[:3]))
        )
        orientation_errors.append(
            float(_quat_distance(motion_state.data.qpos[3:7], reference.qpos[3:7]))
        )
        indices = jnp.asarray(env.model.policy_qpos_indices)
        joint_errors.append(
            np.asarray(motion_state.data.qpos[indices] - reference.qpos[indices])
        )
        policy_observation = normalize_observation(
            observation, observation_norm_state, ppo_config
        )
        action = apply_actor(policy, policy_observation)
        state, observation, reward, done, info = env.step(state, action)
        motion_state = state.motion_state
        rewards.append(float(reward))
        contacts = np.asarray(env._foot_contacts(motion_state, 0), dtype=bool)
        foot_velocity = np.asarray(
            env._body_linear_velocity(
                motion_state, jnp.asarray(env.model.foot_body_ids)
            )
        )
        foot_slips.append(
            float(
                np.mean(
                    np.linalg.norm(foot_velocity, axis=-1) * contacts.astype(np.float32)
                )
            )
        )
        for name, value in info.items():
            if name.startswith("reward/"):
                reward_terms.setdefault(name, []).append(float(value))
            if name.startswith("done/") and bool(value):
                termination_counts[name] = termination_counts.get(name, 0) + 1
        if bool(done):
            terminal = bool(info["done/terminal"])
            timeout = bool(info["done/timeout"])
            if timeout or (terminal and stop_on_terminal):
                break
    return {
        "steps": len(rewards),
        "episode_return": float(sum(rewards)),
        "root_position_rms_m": _rms(root_errors),
        "root_orientation_rms_deg": float(np.degrees(_rms(orientation_errors))),
        "joint_position_rms_rad": _rms(np.asarray(joint_errors)),
        "foot_slip_mean_mps": float(np.mean(foot_slips)) if foot_slips else 0.0,
        "terminal": terminal,
        "timeout": timeout,
        "termination_counts": termination_counts,
        "reward_terms": {
            name: float(np.mean(values)) for name, values in reward_terms.items()
        },
    }


def _aggregate_results(metadata, clips: list[dict[str, Any]]) -> dict[str, Any]:
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
        "task": "twist_student",
        "dataset_fingerprint": metadata.dataset.dataset_fingerprint,
        "robot_fingerprint": metadata.robot_fingerprint,
        "teacher_fingerprint": metadata.teacher_fingerprint,
        "global": global_metrics,
        "per_source": sources,
        "per_clip": clips,
    }


def _rms(values) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array)))) if array.size else 0.0
