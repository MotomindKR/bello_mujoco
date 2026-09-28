"""Task runtime reconstruction shared by locomotion evaluation and deployment."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import equinox as eqx

from bello_mujoco.models.compiler import compile_robot
from bello_mujoco.envs.locomotion import BelloLocomotionEnv
from bello_mujoco.policy import policy_observation_size
from bello_mujoco.training.checkpoint import (
    CheckpointNormalizationState,
    LocomotionCheckpointMetadata,
    restore_evaluation_checkpoint,
)


@dataclass(frozen=True)
class LocomotionPolicyRuntime:
    env: BelloLocomotionEnv
    policy: eqx.Module
    metadata: LocomotionCheckpointMetadata
    normalization: CheckpointNormalizationState
    urdf_path: Path


def load_locomotion_policy_runtime(
    checkpoint_path: Path,
    *,
    use_viewer_model: bool,
) -> LocomotionPolicyRuntime:
    train_state, metadata, normalization = restore_evaluation_checkpoint(
        checkpoint_path,
    )
    if not isinstance(metadata, LocomotionCheckpointMetadata):
        raise ValueError(f"checkpoint task is {metadata.task!r}, not 'locomotion'")
    prepared = compile_robot(metadata.asset)
    env = BelloLocomotionEnv.from_compiled(
        prepared,
        warp=metadata.warp,
        batch_size=1,
        env_config=replace(
            metadata.env,
            arm_pose_path=(
                Path(checkpoint_path).resolve() / "standing.npz"
                if metadata.env.arm_pose_path is not None
                else None
            ),
            model_randomization=(
                replace(
                    metadata.env.model_randomization,
                    enabled=False,
                    payload_distribution="fixed",
                )
                if metadata.asset.barbell is not None
                else metadata.env.model_randomization
            ),
        ),
        actuator_config=metadata.asset.actuator_config,
        imu_body_name=metadata.asset.imu_config.body_name,
        reward_config=metadata.reward,
        command_config=metadata.command,
        done_config=metadata.done,
        use_viewer_model=use_viewer_model,
        load_mjx=False,
    )
    validate_locomotion_checkpoint_shape(env, metadata)
    return LocomotionPolicyRuntime(
        env=env,
        policy=train_state.policy,
        metadata=metadata,
        normalization=normalization,
        urdf_path=prepared.urdf_path,
    )


def validate_locomotion_checkpoint_shape(
    env: BelloLocomotionEnv,
    metadata: LocomotionCheckpointMetadata,
) -> None:
    expected_obs_size = policy_observation_size(
        env.observation_size, metadata.ppo.policy
    )
    expected_critic_obs_size = policy_observation_size(
        env.critic_observation_size, metadata.ppo.policy
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
