from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bello_mujoco.assets_pipeline.prepare import prepare_asset
from bello_mujoco.checkpoint import (
    CheckpointMetadata,
    restore_evaluation_checkpoint,
)
from bello_mujoco.envs.locomotion import BelloLocomotionEnv
from bello_mujoco.policy import policy_observation_size
from bello_mujoco.rerun_eval import GamepadConfig, run_rerun_evaluation
from bello_mujoco.train.ppo import critic_observation_size


@dataclass(frozen=True)
class EvalPolicyConfig:
    command: tuple[float, float, float] = (0.0, 0.0, 0.0)
    steps: int | None = None
    stop_on_done: bool = False
    stochastic_actions: bool = False
    action_seed: int = 0
    gamepad: GamepadConfig = GamepadConfig()


def main(
    checkpoint_path: Path,
    eval_config: EvalPolicyConfig = EvalPolicyConfig(),
) -> None:
    validate_eval_config(eval_config)
    train_state, metadata, normalization = restore_evaluation_checkpoint(
        checkpoint_path
    )
    if metadata.task != "locomotion":
        raise ValueError(
            f"checkpoint task is {metadata.task!r}, not 'locomotion'; use "
            "`bello-eval task=motion` for motion checkpoints"
        )
    if any(
        value is None
        for value in (metadata.env, metadata.reward, metadata.command, metadata.done)
    ):
        raise ValueError("locomotion checkpoint is missing its environment contract")
    prepared = prepare_asset(metadata.asset)
    env = BelloLocomotionEnv.load_prepared(
        prepared,
        warp=metadata.warp,
        batch_size=1,
        env_config=metadata.env,
        actuator_config=metadata.asset.actuator_config,
        reward_config=metadata.reward,
        command_config=metadata.command,
        done_config=metadata.done,
        use_viewer_model=True,
        load_mjx=False,
    )
    validate_checkpoint_shape(env, metadata)
    run_rerun_evaluation(
        env,
        train_state.policy,
        normalization=normalization.observation,
        ppo_config=metadata.ppo,
        urdf_path=prepared.urdf_path,
        command=eval_config.command,
        gamepad_config=eval_config.gamepad,
        steps=eval_config.steps,
        stop_on_done=eval_config.stop_on_done,
        obs_stack_size=metadata.ppo.policy.obs_stack_size,
        stochastic_actions=eval_config.stochastic_actions,
        action_seed=eval_config.action_seed,
    )


def validate_eval_config(config: EvalPolicyConfig) -> None:
    if config.steps is not None and config.steps <= 0:
        raise ValueError(f"eval_config.steps must be positive, got {config.steps}")
    if len(config.command) != 3:
        raise ValueError(
            f"eval_config.command must have three values, got {config.command}"
        )
    if config.action_seed < 0:
        raise ValueError(
            f"eval_config.action_seed must be nonnegative, got {config.action_seed}"
        )
    if config.gamepad.controller_index < 0:
        raise ValueError(
            "eval_config.gamepad.controller_index must be nonnegative, got "
            f"{config.gamepad.controller_index}"
        )
    if not 0.0 <= config.gamepad.deadzone < 1.0:
        raise ValueError(
            "eval_config.gamepad.deadzone must be in [0, 1), got "
            f"{config.gamepad.deadzone}"
        )


def validate_checkpoint_shape(
    env: BelloLocomotionEnv,
    metadata: CheckpointMetadata,
) -> None:
    expected_obs_size = policy_observation_size(
        env.observation_size, metadata.ppo.policy
    )
    expected_critic_obs_size = policy_observation_size(
        critic_observation_size(env), metadata.ppo.policy
    )
    if expected_obs_size != metadata.obs_size:
        raise ValueError(
            "eval runtime config produced policy observation_size="
            f"{expected_obs_size}, but checkpoint expects {metadata.obs_size}"
        )
    if expected_critic_obs_size != metadata.critic_obs_size:
        raise ValueError(
            "eval runtime config produced critic observation_size="
            f"{expected_critic_obs_size}, but checkpoint expects "
            f"{metadata.critic_obs_size}"
        )
    if env.action_size != metadata.action_size:
        raise ValueError(
            "eval runtime config produced action_size="
            f"{env.action_size}, but checkpoint expects {metadata.action_size}"
        )
    if env.observation_layout != metadata.actor_observation_layout:
        raise ValueError("checkpoint actor observation layout does not match runtime")
    if env.critic_observation_layout != metadata.critic_observation_layout:
        raise ValueError("checkpoint critic observation layout does not match runtime")
    if env.action_layout != metadata.action_layout:
        raise ValueError("checkpoint action layout does not match runtime")
