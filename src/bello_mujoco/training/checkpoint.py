"""Orbax checkpoint helpers for PPO training and evaluation."""

from __future__ import annotations

import json
import hashlib
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from bello_mujoco.models.compiler import ModelConfig
from bello_mujoco.contracts import ActionLayout, ObservationLayout
from bello_mujoco.configuration import coerce_value as coerce_value
from bello_mujoco.configuration import dataclass_from_dict
from bello_mujoco.envs.locomotion_config import BelloEnvConfig, DoneConfig
from bello_mujoco.envs.motion_tracking import MotionTrackingEnvConfig
from bello_mujoco.motion.clip import MotionClip, save_motion_clip
from bello_mujoco.envs.rewards import CommandConfig, RewardConfig
from bello_mujoco.sim import WarpConfig
from bello_mujoco.training.ppo_types import (
    ObservationNormState,
    PPOEnvironment,
    RunnerState,
    TrainState,
    ValueNormState,
)
from bello_mujoco.training.ppo_config import PPOConfig
from bello_mujoco.training.normalization import (
    init_observation_norm_state,
    init_value_norm_state,
)
from bello_mujoco.training.ppo import init_runner_state, init_train_state
from bello_mujoco.repository import capture_repository_commit, require_repository_commit
from bello_mujoco.training.tracking import to_builtin

METADATA_FILE = "metadata.json"
STATE_DIR = "state"
CONTINUATION_DIR = "continuation"
MOTION_FILE = "motion.npz"

_CHECKPOINT_DOCUMENT_FIELDS = frozenset(
    {
        "git_commit",
        "update",
        "metadata",
    }
)


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
class _CheckpointMetadataBase:
    ppo: PPOConfig
    asset: ModelConfig
    warp: WarpConfig
    obs_size: int
    critic_obs_size: int
    action_size: int
    actor_observation_layout: ObservationLayout
    critic_observation_layout: ObservationLayout
    action_layout: ActionLayout


@dataclass(frozen=True)
class LocomotionCheckpointMetadata(_CheckpointMetadataBase):
    reward: RewardConfig
    command: CommandConfig
    done: DoneConfig
    env: BelloEnvConfig
    task: Literal["locomotion"] = "locomotion"


@dataclass(frozen=True)
class MotionCheckpointMetadata(_CheckpointMetadataBase):
    motion_env: MotionTrackingEnvConfig
    task: Literal["motion_tracking"] = "motion_tracking"


CheckpointMetadata: TypeAlias = LocomotionCheckpointMetadata | MotionCheckpointMetadata


class CheckpointNormalizationState(NamedTuple):
    """Running statistics needed to reproduce policy and value inputs."""

    value: ValueNormState
    observation: ObservationNormState
    critic_observation: ObservationNormState


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


def validate_checkpoint_name(name: str | None) -> None:
    if name is None:
        return
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(
            f"checkpoint_config.name must be a nonempty path component, got {name!r}"
        )


def validate_checkpoint_options(config: CheckpointConfig) -> None:
    validate_checkpoint_name(config.name)


def save_training_checkpoint(
    path: Path,
    runner_state: RunnerState,
    metadata: CheckpointMetadata,
    git_commit: str,
    *,
    motion: MotionClip | None = None,
) -> None:
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(
            f"checkpoint path already exists: {path}. Checkpoint paths are "
            "timestamped; choose a new path instead of overwriting."
        )
    if isinstance(metadata, MotionCheckpointMetadata) != (motion is not None):
        raise ValueError("motion checkpoint requires its in-memory reference clip")
    train_item = _train_checkpoint_item(runner_state)
    continuation_item = _continuation_checkpoint_item(runner_state)
    document = _checkpoint_document(
        metadata,
        train_item,
        continuation_item,
        git_commit,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(tempfile.mkdtemp(prefix=f".{path.name}.tmp-", dir=path.parent))
    checkpointer = ocp.PyTreeCheckpointer()
    try:
        if (
            isinstance(metadata, LocomotionCheckpointMetadata)
            and metadata.env.arm_pose_path is not None
        ):
            if (
                hashlib.sha256(metadata.env.arm_pose_path.read_bytes()).hexdigest()
                != metadata.env.arm_pose_sha256
            ):
                raise ValueError("arm reference changed since training initialization")
            shutil.copyfile(metadata.env.arm_pose_path, staging_path / "standing.npz")
            document["metadata"]["env"]["arm_pose_path"] = "standing.npz"
            _sync_file(staging_path / "standing.npz")
        if motion is not None:
            save_motion_clip(staging_path / MOTION_FILE, motion)
            _sync_file(staging_path / MOTION_FILE)
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
        _require_unchanged_git_commit(git_commit)
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
    finally:
        checkpointer.close()


def restore_evaluation_checkpoint(
    path: Path,
) -> tuple[TrainState, CheckpointMetadata, CheckpointNormalizationState]:
    """Restore a policy plus the exact running statistics used by its actor."""

    path = Path(path).resolve()
    metadata, saved_update = _load_checkpoint_document(
        path,
    )
    template_state, _ = init_train_state(
        jax.random.PRNGKey(metadata.ppo.seed),
        metadata.obs_size,
        metadata.action_size,
        metadata.ppo,
        critic_obs_size=metadata.critic_obs_size,
    )
    _, static_policy = eqx.partition(template_state.policy, eqx.is_array)
    restored = _restore_checkpoint_item(
        path / STATE_DIR,
        _train_state_checkpoint_item(template_state, metadata),
    )
    _validate_restored_normalization_state(restored, metadata)
    _validate_restored_update(restored["update"], saved_update, metadata.ppo)
    train_state = TrainState(
        policy=eqx.combine(restored["policy"], static_policy),
        opt_state=restored["opt_state"],
        update=restored["update"],
    )
    normalization = CheckpointNormalizationState(
        value=restored["value_norm_state"],
        observation=restored["observation_norm_state"],
        critic_observation=restored["critic_observation_norm_state"],
    )
    return train_state, metadata, normalization


def restore_runner_checkpoint(
    path: Path,
    env: PPOEnvironment,
    expected_metadata: CheckpointMetadata,
) -> tuple[
    RunnerState,
    eqx.Module,
    optax.GradientTransformation,
    CheckpointMetadata,
]:
    path = Path(path).resolve()
    recorded_metadata, saved_update = _load_checkpoint_document(
        path,
    )
    effective_metadata = resumed_checkpoint_metadata(
        recorded_metadata,
        expected_metadata,
        saved_update=saved_update,
    )
    _validate_environment_layouts(env, effective_metadata)
    template, static_policy, optimizer = init_runner_state(
        jax.random.PRNGKey(effective_metadata.ppo.seed),
        env,
        effective_metadata.ppo,
    )
    train_item = _restore_checkpoint_item(
        path / STATE_DIR,
        _train_checkpoint_item(template),
    )
    continuation_item = _restore_checkpoint_item(
        path / CONTINUATION_DIR,
        _continuation_checkpoint_item(template),
    )
    _validate_restored_normalization_state(train_item, recorded_metadata)
    _validate_restored_update(
        train_item["update"], saved_update, effective_metadata.ppo
    )
    runner_state = RunnerState(
        params=train_item["policy"],
        opt_state=train_item["opt_state"],
        update=train_item["update"],
        env_state=continuation_item["env_state"],
        obs_stack=continuation_item["obs_stack"],
        critic_obs_stack=continuation_item["critic_obs_stack"],
        rng=continuation_item["rng"],
        learning_rate=continuation_item["learning_rate"],
        value_norm_state=train_item["value_norm_state"],
        observation_norm_state=train_item["observation_norm_state"],
        critic_observation_norm_state=train_item["critic_observation_norm_state"],
        task_state=continuation_item["task_state"],
    )
    return runner_state, static_policy, optimizer, effective_metadata


def load_checkpoint_metadata(
    path: Path,
) -> CheckpointMetadata:
    metadata, _ = _load_checkpoint_document(
        Path(path).resolve(),
    )
    return metadata


def load_checkpoint_update(
    path: Path,
) -> int:
    """Return the saved update without restoring arrays."""

    _, saved_update = _load_checkpoint_document(
        Path(path).resolve(),
    )
    return saved_update


def resumed_checkpoint_metadata(
    recorded: CheckpointMetadata,
    requested: CheckpointMetadata,
    *,
    saved_update: int,
) -> CheckpointMetadata:
    """Validate a resume request and apply its operationally mutable fields."""

    recorded_payload = _resume_contract_payload(
        recorded,
    )
    requested_payload = _resume_contract_payload(
        requested,
    )
    if recorded_payload != requested_payload:
        mismatches = sorted(
            key
            for key in set(recorded_payload) | set(requested_payload)
            if recorded_payload.get(key) != requested_payload.get(key)
        )
        raise ValueError(
            "resume checkpoint contract differs from the requested run for "
            + ", ".join(mismatches)
        )
    if requested.ppo.num_updates < saved_update:
        raise ValueError(
            "requested training budget ends before the saved checkpoint update: "
            f"{requested.ppo.num_updates} < {saved_update}"
        )
    return requested


def _train_state_checkpoint_item(
    train_state: TrainState, metadata: CheckpointMetadata
) -> dict[str, Any]:
    params, _ = eqx.partition(train_state.policy, eqx.is_array)
    return {
        "policy": params,
        "opt_state": train_state.opt_state,
        "update": train_state.update,
        "value_norm_state": init_value_norm_state(),
        "observation_norm_state": init_observation_norm_state(metadata.obs_size),
        "critic_observation_norm_state": init_observation_norm_state(
            metadata.critic_obs_size
        ),
    }


def _train_checkpoint_item(runner_state: RunnerState) -> dict[str, Any]:
    return {
        "policy": runner_state.params,
        "opt_state": runner_state.opt_state,
        "update": runner_state.update,
        "value_norm_state": runner_state.value_norm_state,
        "observation_norm_state": runner_state.observation_norm_state,
        "critic_observation_norm_state": (runner_state.critic_observation_norm_state),
    }


def _continuation_checkpoint_item(runner_state: RunnerState) -> dict[str, Any]:
    item = {
        "env_state": runner_state.env_state,
        "obs_stack": runner_state.obs_stack,
        "critic_obs_stack": runner_state.critic_obs_stack,
        "rng": runner_state.rng,
        "learning_rate": runner_state.learning_rate,
    }
    item["task_state"] = runner_state.task_state
    return item


def _checkpoint_document(
    metadata: CheckpointMetadata,
    train_item: dict[str, Any],
    continuation_item: dict[str, Any],
    git_commit: str,
) -> dict[str, Any]:
    metadata_payload = to_builtin(metadata)
    resolved_metadata = _metadata_from_dict(
        metadata_payload,
        strict=True,
        require_all_fields=True,
        path="metadata",
    )
    _validate_metadata_layouts(resolved_metadata)
    _require_unchanged_git_commit(git_commit)
    update = _validate_update(train_item["update"], resolved_metadata.ppo)
    canonical_payload = to_builtin(resolved_metadata)
    return {
        "git_commit": git_commit,
        "update": update,
        "metadata": canonical_payload,
    }


def _require_unchanged_git_commit(git_commit: str) -> None:
    current_commit = capture_repository_commit()
    if current_commit != git_commit:
        raise RuntimeError(
            "Git commit changed after training initialization: "
            f"started at {git_commit}, current commit is {current_commit}"
        )


def _resume_contract_payload(
    metadata: CheckpointMetadata,
) -> dict[str, Any]:
    """Return training semantics, excluding budget and persisted running state."""

    payload = to_builtin(metadata)
    payload["ppo"].pop("total_timesteps")
    if (
        isinstance(metadata, LocomotionCheckpointMetadata)
        and metadata.env.arm_pose_path is not None
    ):
        payload["env"]["arm_pose_path"] = "standing.npz"
    return payload


def _load_checkpoint_document(
    path: Path,
) -> tuple[CheckpointMetadata, int]:
    metadata_path = Path(path).resolve() / METADATA_FILE
    try:
        document = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"checkpoint metadata is not valid JSON: {metadata_path}"
        ) from error
    if not isinstance(document, dict):
        raise TypeError(f"checkpoint metadata root must be a mapping: {metadata_path}")
    if "git_commit" not in document:
        raise ValueError(
            "checkpoint predates embedded Git provenance; use the archived source "
            "revision that created it"
        )
    git_commit = document["git_commit"]
    if not isinstance(git_commit, str):
        raise ValueError("git_commit must be a string")
    require_repository_commit(
        git_commit,
    )
    _require_exact_fields(
        document,
        _CHECKPOINT_DOCUMENT_FIELDS,
        path="checkpoint metadata envelope",
    )
    metadata_payload = document["metadata"]
    metadata = _metadata_from_dict(
        metadata_payload,
        strict=True,
        require_all_fields=True,
        path="metadata",
    )
    _validate_metadata_layouts(metadata)
    update = document["update"]
    if type(update) is not int:
        raise ValueError("checkpoint update must be an integer")
    _validate_update(update, metadata.ppo)
    return metadata, update


def _metadata_from_dict(
    value: Any,
    *,
    strict: bool,
    require_all_fields: bool,
    path: str,
) -> CheckpointMetadata:
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be a mapping")
    task = value.get("task")
    if task == "locomotion":
        metadata_type = LocomotionCheckpointMetadata
    elif task == "motion_tracking":
        metadata_type = MotionCheckpointMetadata
    else:
        raise ValueError(
            f"{path}.task must be locomotion or motion_tracking, got {task!r}"
        )
    return dataclass_from_dict(
        metadata_type,
        value,
        strict=strict,
        require_all_fields=require_all_fields,
        path=path,
    )


def _validate_metadata_layouts(metadata: CheckpointMetadata) -> None:
    if isinstance(metadata, MotionCheckpointMetadata):
        if metadata.motion_env.motion_path != Path(MOTION_FILE):
            raise ValueError("motion checkpoints must use their bundled motion.npz")
    elif not isinstance(metadata, LocomotionCheckpointMetadata):
        raise TypeError(f"unsupported checkpoint metadata type {type(metadata)!r}")
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


def _validate_running_moments(
    mean: tuple[float, ...],
    var: tuple[float, ...],
    count: float,
    *,
    name: str,
    expected_size: int = 1,
) -> None:
    if len(mean) != expected_size or len(var) != expected_size:
        raise ValueError(
            f"checkpoint {name} shape mismatch: expected {expected_size} values"
        )
    if not math.isfinite(count) or count < 0.0:
        raise ValueError(f"checkpoint {name} count must be finite and nonnegative")
    if not all(math.isfinite(value) for value in mean):
        raise ValueError(f"checkpoint {name} mean must be finite")
    if not all(math.isfinite(value) and value >= 0.0 for value in var):
        raise ValueError(f"checkpoint {name} variance must be finite and nonnegative")


def _validate_restored_normalization_state(
    restored: dict[str, Any], metadata: CheckpointMetadata
) -> None:
    for key, size in (
        ("value_norm_state", 1),
        ("observation_norm_state", metadata.obs_size),
        ("critic_observation_norm_state", metadata.critic_obs_size),
    ):
        state = restored[key]
        _validate_running_moments(
            tuple(np.asarray(state.mean).reshape(-1)),
            tuple(np.asarray(state.var).reshape(-1)),
            float(state.count),
            name=key,
            expected_size=size,
        )


def _validate_environment_layouts(env, metadata: CheckpointMetadata) -> None:
    env_task = env.task
    if env_task != metadata.task:
        raise ValueError(
            f"checkpoint task {metadata.task!r} does not match environment task "
            f"{env_task!r}"
        )
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


def _restore_checkpoint_item(path: Path, template: dict[str, Any]) -> dict[str, Any]:
    storage_template = _orbax_storage_item(template)
    restore_args = ocp.checkpoint_utils.construct_restore_args(storage_template)
    checkpointer = ocp.PyTreeCheckpointer()
    try:
        restored_storage = checkpointer.restore(
            path, item=storage_template, restore_args=restore_args
        )
    finally:
        checkpointer.close()
    restored = _restore_empty_arrays(restored_storage, template)
    actual_leaves, actual_tree = jax.tree_util.tree_flatten(restored)
    expected_leaves, expected_tree = jax.tree_util.tree_flatten(template)
    if actual_tree != expected_tree or any(
        actual.shape != expected.shape or actual.dtype != expected.dtype
        for actual, expected in zip(actual_leaves, expected_leaves, strict=True)
    ):
        raise ValueError(
            f"checkpoint array structure, shape, or dtype mismatch: {path}"
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


def _sync_file(path: Path) -> None:
    with path.open("rb") as file:
        os.fsync(file.fileno())


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
