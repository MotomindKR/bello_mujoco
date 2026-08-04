"""Orbax checkpoint helpers for PPO training and evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.contracts import ActionLayout, ObservationLayout
from bello_mujoco.configuration import coerce_value as coerce_value
from bello_mujoco.configuration import dataclass_from_dict
from bello_mujoco.envs.locomotion import (
    BelloEnvConfig,
    DoneConfig,
)
from bello_mujoco.envs.motion_tracking import (
    MotionObservationLatencyConfig,
    MotionTrackingEnvConfig,
)
from bello_mujoco.motion.clip import MOTION_FORMAT_VERSION, MotionClip
from bello_mujoco.envs.rewards import CommandConfig, RewardConfig
from bello_mujoco.sim import WarpConfig
from bello_mujoco.training.ppo import (
    ObservationNormState,
    PPOConfig,
    PPOEnvironment,
    RunnerState,
    TrainState,
    ValueNormState,
    init_observation_norm_state,
    init_runner_state,
    init_train_state,
    init_value_norm_state,
)
from bello_mujoco.repository import (
    capture_repository_commit,
    require_repository_commit,
)
from bello_mujoco.training.tracking import to_builtin

METADATA_FILE = "metadata.json"
STATE_DIR = "state"
CONTINUATION_DIR = "continuation"

_CHECKPOINT_DOCUMENT_FIELDS = frozenset(
    {
        "git_commit",
        "update",
        "metadata_sha256",
        "train_layout_sha256",
        "continuation_layout_sha256",
        "metadata",
    }
)
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_HEX_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class CheckpointConfig:
    """Periodic and final training checkpoint settings."""

    enabled: bool = False
    directory: Path = Path("checkpoints")
    name: str | None = None
    every_updates: int | None = None
    resume_path: Path | None = None
    allow_git_commit_mismatch: bool = False
    upload_wandb_artifact: bool = True
    wandb_artifact_name: str | None = None
    wandb_artifact_type: str = "checkpoint"


@dataclass(frozen=True)
class MotionClipContract:
    """Content identity and named schema for a motion checkpoint."""

    path: Path
    content_sha256: str
    source_sha256: str
    format_version: int
    fps: float
    num_frames: int
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    qpos_size: int
    qvel_size: int


@dataclass(frozen=True)
class ValueNormMetadata:
    mean: float
    var: float
    count: float


@dataclass(frozen=True)
class ObservationNormMetadata:
    mean: tuple[float, ...]
    var: tuple[float, ...]
    count: float


@dataclass(frozen=True)
class _CheckpointMetadataBase:
    ppo: PPOConfig
    asset: AssetPrepConfig
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
    motion: MotionClipContract
    task: Literal["motion_tracking"] = "motion_tracking"
    value_norm: ValueNormMetadata | None = None
    observation_norm: ObservationNormMetadata | None = None
    critic_observation_norm: ObservationNormMetadata | None = None
    motion_sampling_failure_ema: tuple[float, ...] | None = None


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
    if config.allow_git_commit_mismatch and config.resume_path is None:
        raise ValueError(
            "checkpoint_config.allow_git_commit_mismatch requires resume_path"
        )


def save_training_checkpoint(
    path: Path,
    runner_state: RunnerState,
    metadata: CheckpointMetadata,
    git_commit: str,
) -> None:
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(
            f"checkpoint path already exists: {path}. Checkpoint paths are "
            "timestamped; choose a new path instead of overwriting."
        )
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


def restore_evaluation_checkpoint(
    path: Path,
    *,
    allow_git_commit_mismatch: bool = False,
) -> tuple[TrainState, CheckpointMetadata, CheckpointNormalizationState]:
    """Restore a policy plus the exact running statistics used by its actor."""

    path = Path(path).resolve()
    metadata, saved_update, train_fingerprint, _ = _load_checkpoint_document(
        path,
        allow_git_commit_mismatch=allow_git_commit_mismatch,
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
        train_fingerprint,
        label="evaluation state",
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
    *,
    allow_git_commit_mismatch: bool = False,
) -> tuple[
    RunnerState,
    eqx.Module,
    optax.GradientTransformation,
    CheckpointMetadata,
]:
    path = Path(path).resolve()
    recorded_metadata, saved_update, train_fingerprint, continuation_fingerprint = (
        _load_checkpoint_document(
            path,
            allow_git_commit_mismatch=allow_git_commit_mismatch,
        )
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
        train_fingerprint,
        label="training state",
    )
    continuation_item = _restore_checkpoint_item(
        path / CONTINUATION_DIR,
        _continuation_checkpoint_item(template),
        continuation_fingerprint,
        label="continuation state",
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
    )
    _validate_restored_motion_runner_state(runner_state, recorded_metadata)
    return runner_state, static_policy, optimizer, effective_metadata


def load_checkpoint_metadata(
    path: Path,
    *,
    allow_git_commit_mismatch: bool = False,
) -> CheckpointMetadata:
    metadata, _, _, _ = _load_checkpoint_document(
        Path(path).resolve(),
        allow_git_commit_mismatch=allow_git_commit_mismatch,
    )
    return metadata


def resumed_checkpoint_metadata(
    recorded: CheckpointMetadata,
    requested: CheckpointMetadata,
    *,
    saved_update: int,
) -> CheckpointMetadata:
    """Validate a resume request and apply its operationally mutable fields."""

    recorded_payload = _resume_contract_payload(recorded)
    requested_payload = _resume_contract_payload(requested)
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
    return {
        "env_state": runner_state.env_state,
        "obs_stack": runner_state.obs_stack,
        "critic_obs_stack": runner_state.critic_obs_stack,
        "rng": runner_state.rng,
        "learning_rate": runner_state.learning_rate,
    }


def _checkpoint_document(
    metadata: CheckpointMetadata,
    train_item: dict[str, Any],
    continuation_item: dict[str, Any],
    git_commit: str,
) -> dict[str, Any]:
    if metadata.asset.output_root is not None:
        raise ValueError(
            "checkpoint asset.output_root is an operational cache location and "
            "must be normalized to null before saving"
        )
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
        "metadata_sha256": _fingerprint(canonical_payload),
        "train_layout_sha256": _layout_fingerprint(train_item),
        "continuation_layout_sha256": _layout_fingerprint(continuation_item),
        "metadata": canonical_payload,
    }


def _require_unchanged_git_commit(git_commit: str) -> None:
    current_commit = capture_repository_commit()
    if current_commit != git_commit:
        raise RuntimeError(
            "Git commit changed after training initialization: "
            f"started at {git_commit}, current commit is {current_commit}"
        )


def _resume_contract_payload(metadata: CheckpointMetadata) -> dict[str, Any]:
    """Return training semantics, excluding budget and persisted running state."""

    payload = to_builtin(metadata)
    payload["ppo"].pop("total_timesteps")
    for key in (
        "value_norm",
        "observation_norm",
        "critic_observation_norm",
        "motion_sampling_failure_ema",
    ):
        payload.pop(key, None)
    motion_env = payload.get("motion_env")
    if motion_env is not None:
        motion_env.pop("motion_path")
    motion = payload.get("motion")
    if motion is not None:
        motion.pop("path")
    return payload


def _load_checkpoint_document(
    path: Path,
    *,
    allow_git_commit_mismatch: bool = False,
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
        allow_mismatch=allow_git_commit_mismatch,
    )
    _require_exact_fields(
        document,
        _CHECKPOINT_DOCUMENT_FIELDS,
        path="checkpoint metadata envelope",
    )
    metadata_sha256 = _validated_fingerprint(
        document["metadata_sha256"], "metadata_sha256"
    )
    train_fingerprint = _validated_fingerprint(
        document["train_layout_sha256"], "train_layout_sha256"
    )
    continuation_fingerprint = _validated_fingerprint(
        document["continuation_layout_sha256"],
        "continuation_layout_sha256",
    )
    metadata_payload = document["metadata"]
    if _fingerprint(metadata_payload) != metadata_sha256:
        raise ValueError(
            "checkpoint metadata fingerprint does not match its contents; "
            "the checkpoint is corrupt or has been modified"
        )
    metadata = _metadata_from_dict(
        _with_checkpoint_metadata_defaults(metadata_payload),
        strict=True,
        require_all_fields=True,
        path="metadata",
    )
    _validate_metadata_layouts(metadata)
    update = document["update"]
    if type(update) is not int:
        raise ValueError("checkpoint update must be an integer")
    _validate_update(update, metadata.ppo)
    return metadata, update, train_fingerprint, continuation_fingerprint


def _with_checkpoint_metadata_defaults(value: Any) -> Any:
    if not isinstance(value, dict) or value.get("task") != "motion_tracking":
        return value
    motion_env = value.get("motion_env")
    if not isinstance(motion_env, dict) or "observation_latency" in motion_env:
        return value
    normalized = dict(value)
    normalized_motion_env = dict(motion_env)
    normalized_motion_env["observation_latency"] = to_builtin(
        MotionObservationLatencyConfig()
    )
    normalized["motion_env"] = normalized_motion_env
    return normalized


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
            f"{path}.task must be 'locomotion' or 'motion_tracking', got {task!r}"
        )
    return dataclass_from_dict(
        metadata_type,
        value,
        strict=strict,
        require_all_fields=require_all_fields,
        path=path,
    )


def _validate_metadata_layouts(metadata: CheckpointMetadata) -> None:
    if metadata.asset.output_root is not None:
        raise ValueError(
            "checkpoint metadata asset.output_root must be null; generated asset "
            "cache locations are not part of the training contract"
        )
    if isinstance(metadata, MotionCheckpointMetadata):
        if any(
            value is None
            for value in (
                metadata.value_norm,
                metadata.observation_norm,
                metadata.critic_observation_norm,
                metadata.motion_sampling_failure_ema,
            )
        ):
            raise ValueError(
                "motion-tracking checkpoint metadata requires normalization and "
                "adaptive-sampler state snapshots"
            )
        if metadata.motion_env.motion_path != metadata.motion.path:
            raise ValueError(
                "motion environment path must match the checkpoint motion contract"
            )
        _validated_fingerprint(
            metadata.motion.content_sha256, "metadata.motion.content_sha256"
        )
        if _HEX_SHA256_PATTERN.fullmatch(metadata.motion.source_sha256) is None:
            raise ValueError(
                "metadata.motion.source_sha256 must contain 64 lowercase hex digits"
            )
        if (
            type(metadata.motion.format_version) is not int
            or metadata.motion.format_version != MOTION_FORMAT_VERSION
        ):
            raise ValueError(
                "unsupported checkpoint motion format version "
                f"{metadata.motion.format_version}; expected {MOTION_FORMAT_VERSION}"
            )
        if (
            type(metadata.motion.num_frames) is not int
            or metadata.motion.num_frames < 2
        ):
            raise ValueError("checkpoint motion must contain at least two frames")
        if not math.isfinite(metadata.motion.fps) or metadata.motion.fps <= 0.0:
            raise ValueError("checkpoint motion fps must be positive and finite")
        if (
            type(metadata.motion.qpos_size) is not int
            or type(metadata.motion.qvel_size) is not int
            or metadata.motion.qpos_size <= 0
            or metadata.motion.qvel_size <= 0
        ):
            raise ValueError("checkpoint motion state dimensions must be positive")
        if not metadata.motion.joint_names or not metadata.motion.body_names:
            raise ValueError("checkpoint motion requires named joint and body layouts")
        if len(set(metadata.motion.joint_names)) != len(metadata.motion.joint_names):
            raise ValueError("checkpoint motion joint names must be unique")
        if len(set(metadata.motion.body_names)) != len(metadata.motion.body_names):
            raise ValueError("checkpoint motion body names must be unique")
        if metadata.motion.qpos_size != 7 + len(metadata.motion.joint_names):
            raise ValueError(
                "checkpoint motion qpos size does not match its named joint schema"
            )
        if metadata.motion.qvel_size != 6 + len(metadata.motion.joint_names):
            raise ValueError(
                "checkpoint motion qvel size does not match its named joint schema"
            )
        if metadata.action_layout.names != metadata.motion.joint_names:
            raise ValueError(
                "checkpoint action layout does not match the motion joint order"
            )
        _validate_motion_state_metadata(metadata)
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


def _validate_motion_state_metadata(metadata: MotionCheckpointMetadata) -> None:
    value = metadata.value_norm
    actor = metadata.observation_norm
    critic = metadata.critic_observation_norm
    failure_ema = metadata.motion_sampling_failure_ema
    if value is None or actor is None or critic is None or failure_ema is None:
        raise ValueError("motion checkpoint state metadata is incomplete")
    _validate_running_moments(
        (value.mean,), (value.var,), value.count, name="value normalization"
    )
    _validate_running_moments(
        actor.mean,
        actor.var,
        actor.count,
        name="actor observation normalization",
        expected_size=metadata.obs_size,
    )
    _validate_running_moments(
        critic.mean,
        critic.var,
        critic.count,
        name="critic observation normalization",
        expected_size=metadata.critic_obs_size,
    )
    if not failure_ema or not all(
        math.isfinite(value) and value >= 0.0 for value in failure_ema
    ):
        raise ValueError(
            "motion adaptive-sampler failure EMA must be a nonempty finite "
            "nonnegative vector"
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
    if not isinstance(metadata, MotionCheckpointMetadata):
        return
    value = metadata.value_norm
    actor = metadata.observation_norm
    critic = metadata.critic_observation_norm
    if value is None or actor is None or critic is None:
        raise ValueError("motion checkpoint normalization metadata is incomplete")
    expected = {
        "value_norm_state": ValueNormState(
            mean=jnp.asarray(value.mean, dtype=jnp.float32),
            var=jnp.asarray(value.var, dtype=jnp.float32),
            count=jnp.asarray(value.count, dtype=jnp.float32),
        ),
        "observation_norm_state": ObservationNormState(
            mean=jnp.asarray(actor.mean, dtype=jnp.float32),
            var=jnp.asarray(actor.var, dtype=jnp.float32),
            count=jnp.asarray(actor.count, dtype=jnp.float32),
        ),
        "critic_observation_norm_state": ObservationNormState(
            mean=jnp.asarray(critic.mean, dtype=jnp.float32),
            var=jnp.asarray(critic.var, dtype=jnp.float32),
            count=jnp.asarray(critic.count, dtype=jnp.float32),
        ),
    }
    for name, expected_state in expected.items():
        actual_leaves = jax.tree_util.tree_leaves(restored[name])
        expected_leaves = jax.tree_util.tree_leaves(expected_state)
        if any(
            not np.array_equal(
                np.asarray(jax.device_get(actual)),
                np.asarray(jax.device_get(expected_value)),
            )
            for actual, expected_value in zip(
                actual_leaves, expected_leaves, strict=True
            )
        ):
            raise ValueError(
                "restored checkpoint normalization state does not match its "
                f"metadata snapshot: {name}"
            )


def _validate_restored_motion_runner_state(
    runner_state: RunnerState, metadata: CheckpointMetadata
) -> None:
    if not isinstance(metadata, MotionCheckpointMetadata):
        return
    expected = metadata.motion_sampling_failure_ema
    if expected is None:
        raise ValueError("motion checkpoint adaptive-sampler metadata is missing")
    actual = np.asarray(
        jax.device_get(runner_state.env_state.sampling_failure_ema), dtype=np.float32
    )
    expected_array = np.asarray(expected, dtype=np.float32)
    if actual.ndim == 2:
        expected_array = np.broadcast_to(expected_array, actual.shape)
    if not np.array_equal(actual, expected_array):
        raise ValueError(
            "restored adaptive-sampler state does not match its metadata snapshot"
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


def value_norm_metadata(state: ValueNormState) -> ValueNormMetadata:
    return ValueNormMetadata(
        mean=float(jax.device_get(state.mean)),
        var=float(jax.device_get(state.var)),
        count=float(jax.device_get(state.count)),
    )


def observation_norm_metadata(state: ObservationNormState) -> ObservationNormMetadata:
    return ObservationNormMetadata(
        mean=tuple(float(value) for value in np.asarray(jax.device_get(state.mean))),
        var=tuple(float(value) for value in np.asarray(jax.device_get(state.var))),
        count=float(jax.device_get(state.count)),
    )


def motion_metadata_with_runner_state(
    metadata: MotionCheckpointMetadata, runner_state: RunnerState
) -> MotionCheckpointMetadata:
    """Attach auditable state snapshots while Orbax retains exact continuation."""

    failure_ema = np.asarray(
        jax.device_get(runner_state.env_state.sampling_failure_ema), dtype=np.float32
    )
    if failure_ema.ndim == 2:
        if not np.array_equal(
            failure_ema, np.broadcast_to(failure_ema[0], failure_ema.shape)
        ):
            raise ValueError("adaptive sampler state must be shared across all worlds")
        failure_ema = failure_ema[0]
    if failure_ema.ndim != 1:
        raise ValueError("adaptive sampler state must be a vector")
    return replace(
        metadata,
        value_norm=value_norm_metadata(runner_state.value_norm_state),
        observation_norm=observation_norm_metadata(runner_state.observation_norm_state),
        critic_observation_norm=observation_norm_metadata(
            runner_state.critic_observation_norm_state
        ),
        motion_sampling_failure_ema=tuple(float(value) for value in failure_ema),
    )


def motion_clip_contract(path: Path, clip: MotionClip) -> MotionClipContract:
    """Capture the exact motion bytes and schema used by a training run."""

    path = Path(path)
    return MotionClipContract(
        path=path,
        content_sha256=_file_fingerprint(path),
        source_sha256=clip.source_sha256,
        format_version=MOTION_FORMAT_VERSION,
        fps=float(clip.fps),
        num_frames=clip.num_frames,
        joint_names=clip.joint_names,
        body_names=clip.body_names,
        qpos_size=int(clip.qpos.shape[1]),
        qvel_size=int(clip.qvel.shape[1]),
    )


def validate_motion_clip_contract(
    path: Path,
    clip: MotionClip,
    expected: MotionClipContract,
) -> MotionClipContract:
    """Accept relocation only when every content and schema field still matches."""

    actual = motion_clip_contract(path, clip)
    comparable_expected = replace(expected, path=Path(path))
    if actual != comparable_expected:
        mismatches = [
            field.name
            for field in fields(MotionClipContract)
            if field.name != "path"
            and getattr(actual, field.name) != getattr(expected, field.name)
        ]
        raise ValueError(
            "motion clip does not match checkpoint content/schema: "
            + ", ".join(mismatches)
        )
    return actual


def _file_fingerprint(path: Path) -> str:
    try:
        content = Path(path).read_bytes()
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"checkpoint runtime input is missing: {Path(path).resolve()}"
        ) from error
    return _content_fingerprint(content)


def _content_fingerprint(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


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
