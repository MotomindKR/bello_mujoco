"""Orbax checkpoint helpers for PPO training and evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import equinox as eqx
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.envs.locomotion import BelloEnvConfig, DoneConfig
from bello_mujoco.envs.rewards import CommandConfig, RewardConfig
from bello_mujoco.sim import WarpConfig
from bello_mujoco.tracking import to_builtin
from bello_mujoco.train.ppo import (
    PPOConfig,
    TrainState,
    ValueNormState,
    init_train_state,
)

METADATA_FILE = "metadata.json"
STATE_DIR = "state"


@dataclass(frozen=True)
class CheckpointConfig:
    """Final training checkpoint settings."""

    enabled: bool = False
    directory: Path = Path("checkpoints")
    name: str | None = None
    upload_wandb_artifact: bool = True
    wandb_artifact_name: str | None = None
    wandb_artifact_type: str = "checkpoint"


@dataclass(frozen=True)
class ValueNormMetadata:
    mean: float
    var: float
    count: float


@dataclass(frozen=True)
class CheckpointMetadata:
    ppo: PPOConfig
    asset: AssetPrepConfig
    reward: RewardConfig
    command: CommandConfig
    done: DoneConfig
    env: BelloEnvConfig
    warp: WarpConfig
    obs_size: int
    action_size: int
    critic_obs_size: int | None = None
    value_norm: ValueNormMetadata | None = None


def checkpoint_path(config: CheckpointConfig, ppo_config: PPOConfig) -> Path:
    name = timestamped_checkpoint_name(config, ppo_config)
    return config.directory / name


def timestamped_checkpoint_name(
    config: CheckpointConfig,
    ppo_config: PPOConfig,
    *,
    timestamp: datetime | None = None,
) -> str:
    base_name = config.name or f"ppo_seed_{ppo_config.seed}"
    checkpoint_time = timestamp or datetime.now()
    suffix = checkpoint_time.strftime("%Y%m%d_%H%M%S_%f")
    return f"{base_name}_{suffix}"


def save_training_checkpoint(
    path: Path,
    train_state: TrainState,
    metadata: CheckpointMetadata,
) -> None:
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(
            f"checkpoint path already exists: {path}. Checkpoint paths are "
            "timestamped; choose a new path instead of overwriting."
        )
    path.mkdir(parents=True, exist_ok=True)
    params, _ = eqx.partition(train_state.policy, eqx.is_array)
    item = {
        "policy": params,
        "opt_state": train_state.opt_state,
        "update": train_state.update,
    }
    ocp.PyTreeCheckpointer().save(path / STATE_DIR, item, force=False)
    (path / METADATA_FILE).write_text(
        json.dumps(to_builtin(metadata), indent=2, sort_keys=True)
    )


def restore_training_checkpoint(path: Path) -> tuple[TrainState, CheckpointMetadata]:
    path = Path(path).resolve()
    metadata = load_checkpoint_metadata(path)
    template_state, _ = init_train_state(
        jax.random.PRNGKey(metadata.ppo.seed),
        metadata.obs_size,
        metadata.action_size,
        metadata.ppo,
        critic_obs_size=metadata.critic_obs_size,
    )
    params_template, static_policy = eqx.partition(template_state.policy, eqx.is_array)
    item_template = {
        "policy": params_template,
        "opt_state": template_state.opt_state,
        "update": jnp.asarray(0, dtype=jnp.int32),
    }
    restore_args = ocp.checkpoint_utils.construct_restore_args(item_template)
    restored = ocp.PyTreeCheckpointer().restore(
        path / STATE_DIR,
        item=item_template,
        restore_args=restore_args,
    )
    train_state = TrainState(
        policy=eqx.combine(restored["policy"], static_policy),
        opt_state=restored["opt_state"],
        update=restored["update"],
    )
    return train_state, metadata


def load_checkpoint_metadata(path: Path) -> CheckpointMetadata:
    data = json.loads((Path(path).resolve() / METADATA_FILE).read_text())
    return CheckpointMetadata(
        ppo=dataclass_from_dict(PPOConfig, data["ppo"]),
        asset=dataclass_from_dict(AssetPrepConfig, data["asset"]),
        reward=dataclass_from_dict(RewardConfig, data["reward"]),
        command=dataclass_from_dict(CommandConfig, data["command"]),
        done=dataclass_from_dict(DoneConfig, data["done"]),
        env=dataclass_from_dict(BelloEnvConfig, data["env"]),
        warp=dataclass_from_dict(WarpConfig, data["warp"]),
        obs_size=int(data["obs_size"]),
        action_size=int(data["action_size"]),
        critic_obs_size=(
            int(data["critic_obs_size"])
            if data.get("critic_obs_size") is not None
            else None
        ),
        value_norm=(
            dataclass_from_dict(ValueNormMetadata, data["value_norm"])
            if data.get("value_norm") is not None
            else None
        ),
    )


def value_norm_metadata(state: ValueNormState | None) -> ValueNormMetadata | None:
    if state is None:
        return None
    return ValueNormMetadata(
        mean=float(jax.device_get(state.mean)),
        var=float(jax.device_get(state.var)),
        count=float(jax.device_get(state.count)),
    )


def dataclass_from_dict(
    cls,
    data: dict[str, Any],
    *,
    strict: bool = False,
    path: str | None = None,
):
    if not isinstance(data, dict):
        raise TypeError(f"{path or cls.__name__} must be a mapping")
    field_names = {field.name for field in fields(cls)}
    if strict:
        unknown = sorted(set(data) - field_names)
        if unknown:
            location = path or cls.__name__
            names = ", ".join(unknown)
            raise ValueError(f"unknown config key(s) for {location}: {names}")
    kwargs = {}
    type_hints = get_type_hints(cls)
    for field in fields(cls):
        if field.name not in data:
            continue
        field_path = field.name if path is None else f"{path}.{field.name}"
        kwargs[field.name] = coerce_value(
            type_hints.get(field.name, field.type),
            data[field.name],
            strict=strict,
            path=field_path,
        )
    return cls(**kwargs)


def coerce_value(annotation, value, *, strict: bool = False, path: str | None = None):
    if value is None:
        return None
    if annotation is Path or annotation == "Path":
        return Path(value)
    if isinstance(annotation, type) and is_dataclass(annotation):
        return dataclass_from_dict(annotation, value, strict=strict, path=path)
    if "Path" in str(annotation) and "None" in str(annotation):
        return Path(value)
    origin = get_origin(annotation)
    if origin is dict:
        key_type, value_type = get_args(annotation)
        return {
            coerce_value(key_type, item_key, strict=strict, path=path): coerce_value(
                value_type,
                item_value,
                strict=strict,
                path=f"{path}.{item_key}" if path is not None else str(item_key),
            )
            for item_key, item_value in value.items()
        }
    if origin is tuple:
        args = get_args(annotation)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(
                coerce_value(args[0], item, strict=strict, path=path) for item in value
            )
        if args:
            return tuple(
                coerce_value(item_type, item, strict=strict, path=path)
                for item_type, item in zip(args, value, strict=False)
            )
        return tuple(value)
    if origin is not None and type(None) in get_args(annotation):
        non_none = [arg for arg in get_args(annotation) if arg is not type(None)]
        if non_none:
            return coerce_value(non_none[0], value, strict=strict, path=path)
    if str(annotation).startswith("tuple["):
        return tuple(value)
    return value
