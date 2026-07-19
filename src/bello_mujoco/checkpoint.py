"""Orbax checkpoint helpers for PPO training and evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp

from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.contracts import ActionLayout, ObservationLayout
from bello_mujoco.envs.locomotion import (
    BelloEnvConfig,
    BelloLocomotionEnv,
    DoneConfig,
)
from bello_mujoco.envs.rewards import CommandConfig, RewardConfig
from bello_mujoco.sim import WarpConfig
from bello_mujoco.tracking import to_builtin
from bello_mujoco.train.ppo import (
    PPOConfig,
    RunnerState,
    TrainState,
    init_runner_state,
    init_train_state,
)

METADATA_FILE = "metadata.json"
STATE_DIR = "state"
CONTINUATION_DIR = "continuation"
CHECKPOINT_SCHEMA_VERSION = 3
CHECKPOINT_KIND = "training_resume"

_CHECKPOINT_DOCUMENT_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "update",
        "config_fingerprint",
        "train_layout_fingerprint",
        "continuation_layout_fingerprint",
        "metadata",
    }
)
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class CheckpointConfig:
    """Periodic and final training checkpoint settings."""

    enabled: bool = False
    directory: Path = Path("checkpoints")
    name: str | None = None
    every_updates: int | None = None
    resume_path: Path | None = None
    upload_wandb_artifact: bool = True
    wandb_artifact_name: str | None = None
    wandb_artifact_type: str = "checkpoint"


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
    critic_obs_size: int
    action_size: int
    actor_observation_layout: ObservationLayout
    critic_observation_layout: ObservationLayout
    action_layout: ActionLayout


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
    runner_state: RunnerState,
    metadata: CheckpointMetadata,
) -> None:
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(
            f"checkpoint path already exists: {path}. Checkpoint paths are "
            "timestamped; choose a new path instead of overwriting."
        )
    train_item = _train_checkpoint_item(runner_state)
    continuation_item = _continuation_checkpoint_item(runner_state)
    document = _checkpoint_document(metadata, train_item, continuation_item)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(tempfile.mkdtemp(prefix=f".{path.name}.tmp-", dir=path.parent))
    try:
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(
            staging_path / STATE_DIR,
            _orbax_storage_item(train_item),
            force=False,
        )
        checkpointer.save(
            staging_path / CONTINUATION_DIR,
            _orbax_storage_item(continuation_item),
            force=False,
        )
        metadata_path = staging_path / METADATA_FILE
        metadata_path.write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _sync_file(metadata_path)
        _sync_directory(staging_path)
        if path.exists():
            raise FileExistsError(
                f"checkpoint path was created while saving: {path}. Refusing "
                "to replace it."
            )
        staging_path.rename(path)
        _sync_directory(path.parent)
    except BaseException:
        shutil.rmtree(staging_path, ignore_errors=True)
        raise


def restore_training_checkpoint(path: Path) -> tuple[TrainState, CheckpointMetadata]:
    path = Path(path).resolve()
    metadata, saved_update, train_fingerprint, _ = _load_checkpoint_document(path)
    template_state, _ = init_train_state(
        jax.random.PRNGKey(metadata.ppo.seed),
        metadata.obs_size,
        metadata.action_size,
        metadata.ppo,
        critic_obs_size=metadata.critic_obs_size,
    )
    _, static_policy = eqx.partition(template_state.policy, eqx.is_array)
    item_template = _train_state_checkpoint_item(template_state)
    restored = _restore_checkpoint_item(
        path / STATE_DIR,
        item_template,
        train_fingerprint,
        label="training state",
    )
    _validate_restored_update(restored["update"], saved_update, metadata.ppo)
    train_state = TrainState(
        policy=eqx.combine(restored["policy"], static_policy),
        opt_state=restored["opt_state"],
        update=restored["update"],
    )
    return train_state, metadata


def restore_runner_checkpoint(
    path: Path,
    env: BelloLocomotionEnv,
) -> tuple[
    RunnerState,
    eqx.Module,
    optax.GradientTransformation,
    CheckpointMetadata,
]:
    path = Path(path).resolve()
    metadata, saved_update, train_fingerprint, continuation_fingerprint = (
        _load_checkpoint_document(path)
    )
    _validate_environment_layouts(env, metadata)
    template, static_policy, optimizer = init_runner_state(
        jax.random.PRNGKey(metadata.ppo.seed), env, metadata.ppo
    )
    train_item = _restore_checkpoint_item(
        path / STATE_DIR,
        _train_checkpoint_item(template),
        train_fingerprint,
        label="training state",
    )
    continuation_item = _restore_checkpoint_item(
        path / CONTINUATION_DIR,
        _continuation_checkpoint_item(template),
        continuation_fingerprint,
        label="continuation state",
    )
    _validate_restored_update(train_item["update"], saved_update, metadata.ppo)
    runner_state = RunnerState(
        params=train_item["policy"],
        opt_state=train_item["opt_state"],
        update=train_item["update"],
        env_state=continuation_item["env_state"],
        obs_stack=continuation_item["obs_stack"],
        critic_obs_stack=continuation_item["critic_obs_stack"],
        rng=continuation_item["rng"],
        learning_rate=continuation_item["learning_rate"],
        value_norm_state=continuation_item["value_norm_state"],
    )
    return runner_state, static_policy, optimizer, metadata


def load_checkpoint_metadata(path: Path) -> CheckpointMetadata:
    metadata, _, _, _ = _load_checkpoint_document(Path(path).resolve())
    return metadata


def _train_state_checkpoint_item(train_state: TrainState) -> dict[str, Any]:
    params, _ = eqx.partition(train_state.policy, eqx.is_array)
    return {
        "policy": params,
        "opt_state": train_state.opt_state,
        "update": train_state.update,
    }


def _train_checkpoint_item(runner_state: RunnerState) -> dict[str, Any]:
    return {
        "policy": runner_state.params,
        "opt_state": runner_state.opt_state,
        "update": runner_state.update,
    }


def _continuation_checkpoint_item(runner_state: RunnerState) -> dict[str, Any]:
    return {
        "env_state": runner_state.env_state,
        "obs_stack": runner_state.obs_stack,
        "critic_obs_stack": runner_state.critic_obs_stack,
        "rng": runner_state.rng,
        "learning_rate": runner_state.learning_rate,
        "value_norm_state": runner_state.value_norm_state,
    }


def _checkpoint_document(
    metadata: CheckpointMetadata,
    train_item: dict[str, Any],
    continuation_item: dict[str, Any],
) -> dict[str, Any]:
    metadata_payload = to_builtin(metadata)
    resolved_metadata = dataclass_from_dict(
        CheckpointMetadata,
        metadata_payload,
        strict=True,
        require_all_fields=True,
        path="metadata",
    )
    _validate_metadata_layouts(resolved_metadata)
    update = _validate_update(train_item["update"], resolved_metadata.ppo)
    canonical_payload = to_builtin(resolved_metadata)
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": CHECKPOINT_KIND,
        "update": update,
        "config_fingerprint": _fingerprint(canonical_payload),
        "train_layout_fingerprint": _layout_fingerprint(train_item),
        "continuation_layout_fingerprint": _layout_fingerprint(continuation_item),
        "metadata": canonical_payload,
    }


def _load_checkpoint_document(
    path: Path,
) -> tuple[CheckpointMetadata, int, str, str]:
    metadata_path = Path(path).resolve() / METADATA_FILE
    try:
        document = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"checkpoint metadata is not valid JSON: {metadata_path}"
        ) from error
    if not isinstance(document, dict):
        raise TypeError(f"checkpoint metadata root must be a mapping: {metadata_path}")
    if "schema_version" not in document:
        raise ValueError(
            "unversioned checkpoint metadata is not supported; expected checkpoint "
            f"schema version {CHECKPOINT_SCHEMA_VERSION}"
        )
    schema_version = document["schema_version"]
    if type(schema_version) is not int or schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported checkpoint schema version {schema_version!r}; expected "
            f"{CHECKPOINT_SCHEMA_VERSION}. Legacy checkpoints are not supported."
        )
    _require_exact_fields(
        document,
        _CHECKPOINT_DOCUMENT_FIELDS,
        path="checkpoint metadata envelope",
    )
    if document["kind"] != CHECKPOINT_KIND:
        raise ValueError(
            f"unsupported checkpoint kind {document['kind']!r}; expected "
            f"{CHECKPOINT_KIND!r}"
        )
    config_fingerprint = _validated_fingerprint(
        document["config_fingerprint"], "config_fingerprint"
    )
    train_fingerprint = _validated_fingerprint(
        document["train_layout_fingerprint"], "train_layout_fingerprint"
    )
    continuation_fingerprint = _validated_fingerprint(
        document["continuation_layout_fingerprint"],
        "continuation_layout_fingerprint",
    )
    metadata = dataclass_from_dict(
        CheckpointMetadata,
        document["metadata"],
        strict=True,
        require_all_fields=True,
        path="metadata",
    )
    _validate_metadata_layouts(metadata)
    canonical_payload = to_builtin(metadata)
    if _fingerprint(canonical_payload) != config_fingerprint:
        raise ValueError(
            "checkpoint config fingerprint does not match its resolved metadata; "
            "the checkpoint is corrupt or has been modified"
        )
    update = document["update"]
    if type(update) is not int:
        raise ValueError("checkpoint update must be an integer")
    _validate_update(update, metadata.ppo)
    return metadata, update, train_fingerprint, continuation_fingerprint


def _validate_metadata_layouts(metadata: CheckpointMetadata) -> None:
    stack_size = metadata.ppo.policy.obs_stack_size
    expected_actor_size = metadata.actor_observation_layout.size * stack_size
    if metadata.obs_size != expected_actor_size:
        raise ValueError(
            "checkpoint actor observation layout resolves to "
            f"{expected_actor_size} stacked values, but obs_size is "
            f"{metadata.obs_size}"
        )
    expected_critic_size = metadata.critic_observation_layout.size * stack_size
    if metadata.critic_obs_size != expected_critic_size:
        raise ValueError(
            "checkpoint critic observation layout resolves to "
            f"{expected_critic_size} stacked values, but critic_obs_size is "
            f"{metadata.critic_obs_size}"
        )
    if metadata.action_size != metadata.action_layout.size:
        raise ValueError(
            "checkpoint action layout contains "
            f"{metadata.action_layout.size} actions, but action_size is "
            f"{metadata.action_size}"
        )


def _validate_environment_layouts(
    env: BelloLocomotionEnv, metadata: CheckpointMetadata
) -> None:
    stack_size = metadata.ppo.policy.obs_stack_size
    if env.observation_layout != metadata.actor_observation_layout:
        raise ValueError(
            "resume environment actor observation layout does not match the "
            "checkpoint metadata"
        )
    if env.critic_observation_layout != metadata.critic_observation_layout:
        raise ValueError(
            "resume environment critic observation layout does not match the "
            "checkpoint metadata"
        )
    if env.action_layout != metadata.action_layout:
        raise ValueError(
            "resume environment action layout does not match the checkpoint metadata"
        )
    actor_size = env.observation_size * stack_size
    if actor_size != metadata.obs_size:
        raise ValueError(
            f"resume environment resolves to actor obs_size {actor_size}, but the "
            f"checkpoint records {metadata.obs_size}"
        )
    critic_size = env.critic_observation_size * stack_size
    if critic_size != metadata.critic_obs_size:
        raise ValueError(
            "resume environment resolves to critic_obs_size "
            f"{critic_size}, but the checkpoint records {metadata.critic_obs_size}"
        )
    if env.action_size != metadata.action_size:
        raise ValueError(
            f"resume environment has action_size {env.action_size}, but the "
            f"checkpoint records {metadata.action_size}"
        )


def _restore_checkpoint_item(
    path: Path,
    template: dict[str, Any],
    saved_fingerprint: str,
    *,
    label: str,
) -> dict[str, Any]:
    if _layout_fingerprint(template) != saved_fingerprint:
        raise ValueError(
            f"checkpoint {label} layout does not match the layout resolved from "
            "its metadata and environment; the checkpoint is incompatible"
        )
    storage_template = _orbax_storage_item(template)
    restore_args = ocp.checkpoint_utils.construct_restore_args(storage_template)
    restored_storage = ocp.PyTreeCheckpointer().restore(
        path,
        item=storage_template,
        restore_args=restore_args,
    )
    restored = _restore_empty_arrays(restored_storage, template)
    if _layout_fingerprint(restored) != saved_fingerprint:
        raise ValueError(
            f"restored checkpoint {label} layout does not match its recorded "
            "fingerprint; the checkpoint is corrupt or incompatible"
        )
    return restored


def _orbax_storage_item(item: Any) -> Any:
    """Encode empty arrays, which Orbax PyTreeCheckpointer cannot serialize."""
    return jax.tree_util.tree_map(
        lambda value: jnp.zeros((1,), dtype=value.dtype) if value.size == 0 else value,
        item,
    )


def _restore_empty_arrays(storage_item: Any, template: Any) -> Any:
    """Recover encoded empty arrays with their exact template shapes."""
    return jax.tree_util.tree_map(
        lambda restored, expected: expected if expected.size == 0 else restored,
        storage_item,
        template,
    )


def _validate_update(update: Any, config: PPOConfig) -> int:
    if type(update) is int:
        value = update
    else:
        shape = getattr(update, "shape", None)
        dtype = getattr(update, "dtype", None)
        if shape != () or dtype is None or not jnp.issubdtype(dtype, jnp.integer):
            raise ValueError("checkpoint update must be a scalar integer")
        value = int(jax.device_get(update))
    if value < 0 or value > config.num_updates:
        raise ValueError(
            f"checkpoint update {value} is outside the valid range "
            f"[0, {config.num_updates}]"
        )
    return value


def _validate_restored_update(
    update: Any, saved_update: int, config: PPOConfig
) -> None:
    restored_update = _validate_update(update, config)
    if restored_update != saved_update:
        raise ValueError(
            f"restored checkpoint update {restored_update} does not match recorded "
            f"update {saved_update}; the checkpoint is corrupt"
        )


def _require_exact_fields(
    data: dict[str, Any], expected: frozenset[str], *, path: str
) -> None:
    actual = set(data)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise ValueError(f"missing required field(s) for {path}: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"unknown field(s) for {path}: {', '.join(unknown)}")


def _validated_fingerprint(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{name} must have the form 'sha256:<64 lowercase hex digits>'"
        )
    return value


def _fingerprint(value: Any) -> str:
    canonical_json = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"sha256:{hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()}"


def _layout_fingerprint(tree: Any) -> str:
    leaves = []
    for key_path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if shape is None or dtype is None:
            raise TypeError(
                "checkpoint state may only contain array leaves, got "
                f"{type(value).__qualname__} at {jax.tree_util.keystr(key_path)}"
            )
        leaves.append(
            {
                "path": jax.tree_util.keystr(key_path),
                "shape": [int(size) for size in shape],
                "dtype": str(jnp.dtype(dtype)),
            }
        )
    return _fingerprint({"array_leaves": leaves})


def _sync_file(path: Path) -> None:
    with path.open("rb") as file:
        os.fsync(file.fileno())


def _sync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            pass
    finally:
        os.close(descriptor)


def dataclass_from_dict(
    cls,
    data: dict[str, Any],
    *,
    strict: bool = False,
    require_all_fields: bool = False,
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
    if require_all_fields:
        missing = sorted(field_names - set(data))
        if missing:
            location = path or cls.__name__
            names = ", ".join(missing)
            raise ValueError(f"missing required config key(s) for {location}: {names}")
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
            require_all_fields=require_all_fields,
            path=field_path,
        )
    return cls(**kwargs)


def coerce_value(
    annotation,
    value,
    *,
    strict: bool = False,
    require_all_fields: bool = False,
    path: str | None = None,
):
    if value is None:
        return None
    if annotation is Path or annotation == "Path":
        return Path(value)
    if isinstance(annotation, type) and is_dataclass(annotation):
        return dataclass_from_dict(
            annotation,
            value,
            strict=strict,
            require_all_fields=require_all_fields,
            path=path,
        )
    if "Path" in str(annotation) and "None" in str(annotation):
        return Path(value)
    origin = get_origin(annotation)
    if origin is dict:
        key_type, value_type = get_args(annotation)
        return {
            coerce_value(
                key_type,
                item_key,
                strict=strict,
                require_all_fields=require_all_fields,
                path=path,
            ): coerce_value(
                value_type,
                item_value,
                strict=strict,
                require_all_fields=require_all_fields,
                path=f"{path}.{item_key}" if path is not None else str(item_key),
            )
            for item_key, item_value in value.items()
        }
    if origin is tuple:
        args = get_args(annotation)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(
                coerce_value(
                    args[0],
                    item,
                    strict=strict,
                    require_all_fields=require_all_fields,
                    path=path,
                )
                for item in value
            )
        if args:
            return tuple(
                coerce_value(
                    item_type,
                    item,
                    strict=strict,
                    require_all_fields=require_all_fields,
                    path=path,
                )
                for item_type, item in zip(args, value, strict=False)
            )
        return tuple(value)
    if origin is not None and type(None) in get_args(annotation):
        non_none = [arg for arg in get_args(annotation) if arg is not type(None)]
        if non_none:
            return coerce_value(
                non_none[0],
                value,
                strict=strict,
                require_all_fields=require_all_fields,
                path=path,
            )
    if str(annotation).startswith("tuple["):
        return tuple(value)
    return value
