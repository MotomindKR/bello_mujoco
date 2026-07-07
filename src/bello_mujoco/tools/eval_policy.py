from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Literal

import tyro

from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.checkpoint import CheckpointMetadata, restore_training_checkpoint
from bello_mujoco.envs.locomotion import BelloEnvConfig, BelloLocomotionEnv, DoneConfig
from bello_mujoco.envs.rewards import CommandConfig, RewardConfig
from bello_mujoco.render import CameraConfig, RewardOverlayConfig, launch_policy_viewer
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
    realtime: bool = True
    stop_on_done: bool = False
    camera_config: CameraConfig = CameraConfig()
    reward_overlay: RewardOverlayConfig = RewardOverlayConfig()
    runtime: EvalRuntimeConfig = EvalRuntimeConfig()


def main(
    checkpoint_path: tyro.conf.Positional[Path],
    eval_config: EvalPolicyConfig = EvalPolicyConfig(),
) -> None:
    validate_eval_config(eval_config)
    train_state, metadata = restore_training_checkpoint(checkpoint_path)
    asset_config, env_config, reward_config, command_config, done_config, warp = (
        eval_runtime_configs(metadata, eval_config.runtime)
    )
    env = BelloLocomotionEnv.load(
        asset_config,
        warp=warp,
        batch_size=1,
        env_config=env_config,
        reward_config=reward_config,
        command_config=command_config,
        done_config=done_config,
    )
    validate_checkpoint_shape(env, metadata)
    launch_policy_viewer(
        env,
        train_state.policy,
        command=eval_config.command,
        steps=eval_config.steps,
        realtime=eval_config.realtime,
        stop_on_done=eval_config.stop_on_done,
        camera_config=eval_config.camera_config,
        reward_overlay=eval_config.reward_overlay,
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
    if config.camera_config.distance <= 0:
        raise ValueError(
            "eval_config.camera_config.distance must be positive, got "
            f"{config.camera_config.distance}"
        )
    if config.reward_overlay.precision < 0:
        raise ValueError(
            "eval_config.reward_overlay.precision must be nonnegative, got "
            f"{config.reward_overlay.precision}"
        )
    if config.reward_overlay.max_reward_terms < 0:
        raise ValueError(
            "eval_config.reward_overlay.max_reward_terms must be nonnegative, got "
            f"{config.reward_overlay.max_reward_terms}"
        )


def validate_checkpoint_shape(
    env: BelloLocomotionEnv,
    metadata: CheckpointMetadata,
) -> None:
    if env.observation_size != metadata.obs_size:
        raise ValueError(
            "eval runtime config produced observation_size="
            f"{env.observation_size}, but checkpoint expects {metadata.obs_size}"
        )
    if env.action_size != metadata.action_size:
        raise ValueError(
            "eval runtime config produced action_size="
            f"{env.action_size}, but checkpoint expects {metadata.action_size}"
        )


def entrypoint() -> None:
    tyro.cli(main)


if __name__ == "__main__":
    entrypoint()
