from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
import os
from pathlib import Path
import sys
from typing import Literal

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import hydra
from omegaconf import DictConfig

from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig, prepare_asset
from bello_mujoco.checkpoint import CheckpointMetadata, restore_training_checkpoint
from bello_mujoco.envs.locomotion import BelloEnvConfig, BelloLocomotionEnv, DoneConfig
from bello_mujoco.envs.rewards import CommandConfig, RewardConfig
from bello_mujoco.hydra_config import config_path, structured_config
from bello_mujoco.policy import policy_observation_size
from bello_mujoco.rerun_eval import GamepadConfig, run_rerun_evaluation
from bello_mujoco.train.ppo import critic_observation_size
from bello_mujoco.sim import WarpConfig

MISSING = object()


@dataclass(frozen=True)
class EvalRuntimeConfig:
    source: Literal["checkpoint", "override"] = "checkpoint"
    asset_config: AssetPrepConfig = AssetPrepConfig()
    env_config: BelloEnvConfig = BelloEnvConfig()
    reward_config: RewardConfig = RewardConfig()
    command_config: CommandConfig = CommandConfig()
    done_config: DoneConfig = DoneConfig()
    warp: WarpConfig = WarpConfig()


@dataclass(frozen=True)
class EvalPolicyConfig:
    command: tuple[float, float, float] = (0.0, 0.0, 0.0)
    steps: int | None = None
    stop_on_done: bool = False
    stochastic_actions: bool = False
    action_seed: int = 0
    gamepad: GamepadConfig = GamepadConfig()
    runtime: EvalRuntimeConfig = EvalRuntimeConfig()


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
    asset_config, env_config, reward_config, command_config, done_config, warp = (
        eval_runtime_configs(metadata, eval_config.runtime)
    )
    prepared = prepare_asset(asset_config)
    env = BelloLocomotionEnv.load_prepared(
        prepared,
        warp=warp,
        batch_size=1,
        env_config=env_config,
        actuator_config=asset_config.actuator_config,
        reward_config=reward_config,
        command_config=command_config,
        done_config=done_config,
        use_viewer_model=True,
        load_mjx=False,
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
        obs_stack_size=metadata.ppo.policy.obs_stack_size,
        stochastic_actions=eval_config.stochastic_actions,
        action_seed=eval_config.action_seed,
    )


def eval_runtime_configs(
    metadata: CheckpointMetadata,
    config: EvalRuntimeConfig,
) -> tuple[
    AssetPrepConfig,
    BelloEnvConfig,
    RewardConfig,
    CommandConfig,
    DoneConfig,
    WarpConfig,
]:
    if config.source == "checkpoint":
        return (
            metadata.asset,
            metadata.env,
            metadata.reward,
            metadata.command,
            metadata.done,
            metadata.warp,
        )
    if config.source == "override":
        return (
            merge_dataclass_defaults(
                metadata.asset, config.asset_config, AssetPrepConfig()
            ),
            merge_dataclass_defaults(metadata.env, config.env_config, BelloEnvConfig()),
            merge_dataclass_defaults(
                metadata.reward, config.reward_config, RewardConfig()
            ),
            merge_dataclass_defaults(
                metadata.command, config.command_config, CommandConfig()
            ),
            merge_dataclass_defaults(metadata.done, config.done_config, DoneConfig()),
            merge_dataclass_defaults(metadata.warp, config.warp, WarpConfig()),
        )
    raise ValueError(f"unknown eval runtime config source: {config.source}")


def merge_dataclass_defaults(base, override, default):
    return merge_default_value(base, override, default)


def merge_default_value(base, override, default):
    if isinstance(base, Mapping) and isinstance(override, Mapping):
        return merge_default_mapping(base, override, default)
    if not (is_dataclass(base) and is_dataclass(override) and is_dataclass(default)):
        return base if override == default else override
    updates = {}
    for field in fields(override):
        override_value = getattr(override, field.name)
        default_value = getattr(default, field.name)
        base_value = getattr(base, field.name)
        updates[field.name] = merge_default_value(
            base_value, override_value, default_value
        )
    return replace(base, **updates)


def merge_default_mapping(base: Mapping, override: Mapping, default) -> dict:
    default = default if isinstance(default, Mapping) else {}
    merged = {}
    keys = [*base.keys(), *(key for key in override.keys() if key not in base)]
    for key in keys:
        base_value = base.get(key, MISSING)
        override_value = override.get(key, MISSING)
        default_value = default.get(key, MISSING)
        if override_value is MISSING:
            merged[key] = base_value
        elif base_value is MISSING:
            merged[key] = override_value
        elif default_value is MISSING:
            merged[key] = override_value
        else:
            merged[key] = merge_default_value(base_value, override_value, default_value)
    return merged


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
    metadata_critic_obs_size = (
        getattr(metadata, "critic_obs_size", None) or metadata.obs_size
    )
    if expected_critic_obs_size != metadata_critic_obs_size:
        raise ValueError(
            "eval runtime config produced critic observation_size="
            f"{expected_critic_obs_size}, but checkpoint expects "
            f"{metadata_critic_obs_size}"
        )
    if env.action_size != metadata.action_size:
        raise ValueError(
            "eval runtime config produced action_size="
            f"{env.action_size}, but checkpoint expects {metadata.action_size}"
        )


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
