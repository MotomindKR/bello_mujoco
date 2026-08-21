"""Task runtime reconstruction shared by motion evaluation and deployment."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import equinox as eqx

from bello_mujoco.assets_pipeline.prepare import prepare_asset
from bello_mujoco.envs.motion_tracking import BelloMotionTrackingEnv
from bello_mujoco.policy import policy_observation_size
from bello_mujoco.repository import resolve_repository_path
from bello_mujoco.training.checkpoint import (
    CheckpointNormalizationState,
    MotionCheckpointMetadata,
    restore_evaluation_checkpoint,
    validate_motion_clip_contract,
)
from bello_mujoco.training.ppo import critic_observation_size


@dataclass(frozen=True)
class MotionPolicyRuntime:
    env: BelloMotionTrackingEnv
    policy: eqx.Module
    metadata: MotionCheckpointMetadata
    normalization: CheckpointNormalizationState
    motion_path: Path


def load_motion_policy_runtime(
    checkpoint_path: Path,
    *,
    motion_path: Path | None = None,
    end_effector_payload_mass_kg: float | None = None,
    use_viewer_model: bool,
    allow_git_commit_mismatch: bool = False,
) -> MotionPolicyRuntime:
    train_state, metadata, normalization = restore_evaluation_checkpoint(
        checkpoint_path,
        allow_git_commit_mismatch=allow_git_commit_mismatch,
    )
    if not isinstance(metadata, MotionCheckpointMetadata):
        raise ValueError(f"checkpoint task is {metadata.task!r}, not 'motion_tracking'")
    resolved_motion_path = (
        resolve_repository_path(metadata.motion.path)
        if motion_path is None
        else Path(motion_path)
    )
    env_config = replace(metadata.motion_env, motion_path=resolved_motion_path)
    asset = metadata.asset
    if end_effector_payload_mass_kg is not None:
        asset = replace(
            asset,
            end_effector_payload_mass_kg=end_effector_payload_mass_kg,
        )
    prepared = prepare_asset(asset)
    env = BelloMotionTrackingEnv.load_prepared_native(
        prepared,
        env_config,
        actuator_config=asset.actuator_config,
        use_viewer_model=use_viewer_model,
    )
    validate_motion_checkpoint_runtime(env, metadata)
    validate_motion_clip_contract(resolved_motion_path, env.motion, metadata.motion)
    return MotionPolicyRuntime(
        env=env,
        policy=train_state.policy,
        metadata=metadata,
        normalization=normalization,
        motion_path=resolved_motion_path,
    )


def validate_motion_checkpoint_runtime(
    env: BelloMotionTrackingEnv,
    metadata: MotionCheckpointMetadata,
) -> None:
    expected_obs_size = policy_observation_size(
        env.observation_size, metadata.ppo.policy
    )
    expected_critic_obs_size = policy_observation_size(
        critic_observation_size(env), metadata.ppo.policy
    )
    if expected_obs_size != metadata.obs_size:
        raise ValueError(
            "runtime config produced policy observation_size="
            f"{expected_obs_size}, but checkpoint expects {metadata.obs_size}"
        )
    if expected_critic_obs_size != metadata.critic_obs_size:
        raise ValueError(
            "runtime config produced critic observation_size="
            f"{expected_critic_obs_size}, but checkpoint expects "
            f"{metadata.critic_obs_size}"
        )
    if env.action_size != metadata.action_size:
        raise ValueError(
            "runtime config produced action_size="
            f"{env.action_size}, but checkpoint expects {metadata.action_size}"
        )
    if env.observation_layout != metadata.actor_observation_layout:
        raise ValueError("checkpoint actor observation layout does not match runtime")
    if env.critic_observation_layout != metadata.critic_observation_layout:
        raise ValueError("checkpoint critic observation layout does not match runtime")
    if env.action_layout != metadata.action_layout:
        raise ValueError("checkpoint action layout does not match runtime")
