from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Literal

import hydra
from omegaconf import DictConfig

from bello_mujoco.assets_pipeline.prepare import prepare_asset
from bello_mujoco.checkpoint import CheckpointMetadata, restore_training_checkpoint
from bello_mujoco.envs.locomotion import BelloLocomotionEnv
from bello_mujoco.hydra_config import config_path, structured_config
from bello_mujoco.policy import policy_observation_size
from bello_mujoco.rerun_eval import GamepadConfig, run_rerun_evaluation
from bello_mujoco.train.ppo import critic_observation_size


@dataclass(frozen=True)
class EvalPolicyConfig:
    command: tuple[float, float, float] = (0.0, 0.0, 0.0)
    steps: int | None = None
    stop_on_done: bool = False
    simulation_backend: Literal["mujoco", "mjx"] = "mujoco"
    stochastic_actions: bool = False
    action_seed: int = 0
    gamepad: GamepadConfig = GamepadConfig()


@dataclass(frozen=True)
class EvalToolConfig:
    checkpoint_path: Path | None = None
    eval_config: EvalPolicyConfig = EvalPolicyConfig()


def main(
    checkpoint_path: Path,
    eval_config: EvalPolicyConfig = EvalPolicyConfig(),
) -> None:
    validate_eval_config(eval_config)
    train_state, metadata = restore_training_checkpoint(checkpoint_path)
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
        load_mjx=eval_config.simulation_backend == "mjx",
    )
    validate_checkpoint_shape(env, metadata)
    run_rerun_evaluation(
        env,
        train_state.policy,
        urdf_path=prepared.urdf_path,
        command=eval_config.command,
        gamepad_config=eval_config.gamepad,
        steps=eval_config.steps,
        stop_on_done=eval_config.stop_on_done,
        simulation_backend=eval_config.simulation_backend,
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
    if config.simulation_backend not in {"mujoco", "mjx"}:
        raise ValueError(
            "eval_config.simulation_backend must be 'mujoco' or 'mjx', got "
            f"{config.simulation_backend!r}"
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


def entrypoint() -> None:
    sys.argv[1:] = [expand_checkpoint_override(arg) for arg in sys.argv[1:]]
    hydra_entrypoint()


def expand_checkpoint_override(argument: str) -> str:
    prefix = "checkpoint_path="
    if not argument.startswith(prefix):
        return argument
    value = argument.removeprefix(prefix)
    if not value.startswith("~"):
        return argument
    return prefix + str(Path(value).expanduser())


@hydra.main(
    version_base=None,
    config_path=config_path("eval_policy"),
    config_name="default",
)
def hydra_entrypoint(config: DictConfig) -> None:
    cfg = structured_config(EvalToolConfig, config)
    if cfg.checkpoint_path is None:
        raise ValueError(
            "checkpoint_path must be set, e.g. checkpoint_path=checkpoints/ppo_seed_0"
        )
    main(cfg.checkpoint_path.expanduser(), cfg.eval_config)


if __name__ == "__main__":
    entrypoint()
