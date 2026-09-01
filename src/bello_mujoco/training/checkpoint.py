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
from bello_mujoco.policy import ActorCritic, partition_policy_parameters
from bello_mujoco.contracts import ActionLayout, ObservationLayout
from bello_mujoco.configuration import coerce_value as coerce_value
from bello_mujoco.configuration import dataclass_from_dict
from bello_mujoco.envs.locomotion import (
    BelloEnvConfig,
    DoneConfig,
    ObservationLatencyConfig,
)
from bello_mujoco.envs.motion_tracking import (
    MotionObservationLatencyConfig,
    MotionTrackingEnvConfig,
)
from bello_mujoco.envs.twist import (
    TwistEnvConfig,
    TwistObservationContract,
    TwistSamplingConfig,
    TwistSimulationConfig,
)
from bello_mujoco.envs.twist2 import (
    Twist2EnvConfig,
    Twist2ObservationContract,
    Twist2ParityContract,
)
from bello_mujoco.envs.sonic import SonicEnvConfig, SonicObservationContract
from bello_mujoco.motion.clip import MOTION_FORMAT_VERSION, MotionClip
from bello_mujoco.motion.sonic_dataset import SonicJointCalibration
from bello_mujoco.motion.twist2_dataset import Twist2DatasetAudit
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
from bello_mujoco.training.sonic_pretraining import SonicPretrainingConfig
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
    warm_start_path: Path | None = None
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


@dataclass(frozen=True)
class TwistDatasetContract:
    index_path: Path
    schema_version: int
    dataset_fingerprint: str
    robot_fingerprint: str
    gmr_revision: str
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]


@dataclass(frozen=True)
class TwistKLSchedule:
    direction: Literal["student||teacher"] = "student||teacher"
    start: float = 0.1
    end: float = 0.01
    anneal_updates: int = 60_000


@dataclass(frozen=True)
class TwistTeacherCheckpointMetadata(_CheckpointMetadataBase):
    simulation: TwistSimulationConfig
    twist_env: TwistEnvConfig
    sampling: TwistSamplingConfig
    observation_contract: TwistObservationContract
    dataset: TwistDatasetContract
    robot_fingerprint: str
    source_revision: str
    stage: Literal["teacher"] = "teacher"
    task: Literal["twist_teacher"] = "twist_teacher"
    value_norm: ValueNormMetadata | None = None
    observation_norm: ObservationNormMetadata | None = None
    critic_observation_norm: ObservationNormMetadata | None = None


@dataclass(frozen=True)
class TwistStudentCheckpointMetadata(_CheckpointMetadataBase):
    simulation: TwistSimulationConfig
    twist_env: TwistEnvConfig
    sampling: TwistSamplingConfig
    observation_contract: TwistObservationContract
    dataset: TwistDatasetContract
    robot_fingerprint: str
    source_revision: str
    teacher_fingerprint: str
    kl_schedule: TwistKLSchedule = TwistKLSchedule()
    stage: Literal["student"] = "student"
    task: Literal["twist_student"] = "twist_student"
    value_norm: ValueNormMetadata | None = None
    observation_norm: ObservationNormMetadata | None = None
    critic_observation_norm: ObservationNormMetadata | None = None


@dataclass(frozen=True)
class Twist2CheckpointMetadata(_CheckpointMetadataBase):
    simulation: TwistSimulationConfig
    twist2_env: Twist2EnvConfig
    sampling: TwistSamplingConfig
    observation_contract: Twist2ObservationContract
    parity_contract: Twist2ParityContract
    dataset: TwistDatasetContract
    dataset_audit: Twist2DatasetAudit
    robot_fingerprint: str
    source_revision: str
    task: Literal["twist2"] = "twist2"
    value_norm: ValueNormMetadata | None = None
    observation_norm: ObservationNormMetadata | None = None
    critic_observation_norm: ObservationNormMetadata | None = None


@dataclass(frozen=True)
class SonicDatasetContract:
    index_path: Path
    schema_version: int
    dataset_fingerprint: str
    robot_fingerprint: str
    bello_gmr_revision: str
    bello_gmr_config_sha256: str
    g1_gmr_revision: str
    g1_config_sha256: str
    g1_robot_sha256: str
    joint_names: tuple[str, ...]
    qpos_joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    calibration: SonicJointCalibration


@dataclass(frozen=True)
class SonicCheckpointMetadata(_CheckpointMetadataBase):
    simulation: TwistSimulationConfig
    sonic_env: SonicEnvConfig
    observation_contract: SonicObservationContract
    dataset: SonicDatasetContract
    pretraining: SonicPretrainingConfig
    release_checkpoint_sha256: str
    release_artifact_sha256: str
    source_revision: str
    task: Literal["sonic"] = "sonic"
    value_norm: ValueNormMetadata | None = None
    observation_norm: ObservationNormMetadata | None = None
    critic_observation_norm: ObservationNormMetadata | None = None


TwistCheckpointMetadata: TypeAlias = (
    TwistTeacherCheckpointMetadata | TwistStudentCheckpointMetadata
)
CheckpointMetadata: TypeAlias = (
    LocomotionCheckpointMetadata
    | MotionCheckpointMetadata
    | TwistTeacherCheckpointMetadata
    | TwistStudentCheckpointMetadata
    | Twist2CheckpointMetadata
    | SonicCheckpointMetadata
)


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
    if config.resume_path is not None and config.warm_start_path is not None:
        raise ValueError("resume_path and warm_start_path are mutually exclusive")
    if (
        config.allow_git_commit_mismatch
        and config.resume_path is None
        and config.warm_start_path is None
    ):
        raise ValueError(
            "checkpoint_config.allow_git_commit_mismatch requires resume_path or "
            "warm_start_path"
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
    checkpointer = ocp.PyTreeCheckpointer()
    try:
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
    *,
    allow_git_commit_mismatch: bool = False,
    policy: ActorCritic | None = None,
) -> tuple[TrainState, CheckpointMetadata, CheckpointNormalizationState]:
    """Restore a policy plus the exact running statistics used by its actor."""

    path = Path(path).resolve()
    metadata, saved_update, train_fingerprint, _ = _load_checkpoint_document(
        path,
        allow_git_commit_mismatch=allow_git_commit_mismatch,
        allow_retired_asset_schema=allow_git_commit_mismatch,
    )
    template_state, _ = init_train_state(
        jax.random.PRNGKey(metadata.ppo.seed),
        metadata.obs_size,
        metadata.action_size,
        metadata.ppo,
        critic_obs_size=metadata.critic_obs_size,
        policy=policy,
    )
    _, static_policy = partition_policy_parameters(template_state.policy)
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
    policy: ActorCritic | None = None,
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
        allow_source_revision_mismatch=allow_git_commit_mismatch,
    )
    _validate_environment_layouts(env, effective_metadata)
    template, static_policy, optimizer = init_runner_state(
        jax.random.PRNGKey(effective_metadata.ppo.seed),
        env,
        effective_metadata.ppo,
        policy=policy,
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
        task_state=continuation_item.get("task_state", template.task_state),
    )
    _validate_restored_motion_runner_state(runner_state, recorded_metadata)
    return runner_state, static_policy, optimizer, effective_metadata


def warm_start_runner_checkpoint(
    path: Path,
    env: PPOEnvironment,
    fresh_runner: RunnerState,
    expected_metadata: CheckpointMetadata,
    *,
    allow_git_commit_mismatch: bool = False,
    allow_sonic_env_mismatch: bool = False,
) -> tuple[RunnerState, CheckpointMetadata]:
    """Initialize policy and normalization from a compatible prior run.

    Unlike resume, warm start intentionally keeps the newly initialized optimizer,
    update counter, environment state, sampler state, RNG, and learning rate. The
    source checkpoint may use a different dataset fingerprint, but its robot,
    retargeter, observation/action layouts, and policy architecture must remain
    identical. SONIC adaptation stages may explicitly waive environment-reward
    and augmentation equality while retaining those structural checks.
    """

    path = Path(path).resolve()
    recorded_metadata, saved_update, train_fingerprint, _ = _load_checkpoint_document(
        path,
        allow_git_commit_mismatch=allow_git_commit_mismatch,
    )
    validate_warm_start_metadata(
        recorded_metadata,
        expected_metadata,
        allow_source_revision_mismatch=allow_git_commit_mismatch,
        allow_sonic_env_mismatch=allow_sonic_env_mismatch,
    )
    _validate_environment_layouts(env, expected_metadata)
    restored = _restore_checkpoint_item(
        path / STATE_DIR,
        _train_checkpoint_item(fresh_runner),
        train_fingerprint,
        label="warm-start state",
    )
    _validate_restored_normalization_state(restored, recorded_metadata)
    _validate_restored_update(restored["update"], saved_update, recorded_metadata.ppo)
    return (
        fresh_runner._replace(
            params=restored["policy"],
            value_norm_state=restored["value_norm_state"],
            observation_norm_state=restored["observation_norm_state"],
            critic_observation_norm_state=(restored["critic_observation_norm_state"]),
        ),
        recorded_metadata,
    )


def load_checkpoint_metadata(
    path: Path,
    *,
    allow_git_commit_mismatch: bool = False,
) -> CheckpointMetadata:
    metadata, _, _, _ = _load_checkpoint_document(
        Path(path).resolve(),
        allow_git_commit_mismatch=allow_git_commit_mismatch,
        allow_retired_asset_schema=allow_git_commit_mismatch,
    )
    return metadata


def load_checkpoint_update(
    path: Path,
    *,
    allow_git_commit_mismatch: bool = False,
) -> int:
    """Return the integrity-checked saved update without restoring arrays."""

    _, saved_update, _, _ = _load_checkpoint_document(
        Path(path).resolve(),
        allow_git_commit_mismatch=allow_git_commit_mismatch,
    )
    return saved_update


def resumed_checkpoint_metadata(
    recorded: CheckpointMetadata,
    requested: CheckpointMetadata,
    *,
    saved_update: int,
    allow_source_revision_mismatch: bool = False,
) -> CheckpointMetadata:
    """Validate a resume request and apply its operationally mutable fields."""

    recorded_payload = _resume_contract_payload(
        recorded,
        omit_source_revision=allow_source_revision_mismatch,
    )
    requested_payload = _resume_contract_payload(
        requested,
        omit_source_revision=allow_source_revision_mismatch,
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


def validate_warm_start_metadata(
    recorded: CheckpointMetadata,
    requested: CheckpointMetadata,
    *,
    allow_source_revision_mismatch: bool = False,
    allow_sonic_env_mismatch: bool = False,
) -> None:
    """Require transfer-compatible task semantics while allowing new data."""

    recorded_payload = _warm_start_contract_payload(
        recorded,
        omit_source_revision=allow_source_revision_mismatch,
    )
    requested_payload = _warm_start_contract_payload(
        requested,
        omit_source_revision=allow_source_revision_mismatch,
    )
    if allow_sonic_env_mismatch:
        if (
            recorded_payload.get("task") != "sonic"
            or requested_payload.get("task") != "sonic"
        ):
            raise ValueError("SONIC environment mismatch waiver requires SONIC inputs")
        recorded_payload.pop("sonic_env", None)
        requested_payload.pop("sonic_env", None)
    if recorded_payload != requested_payload:
        mismatches = sorted(
            key
            for key in set(recorded_payload) | set(requested_payload)
            if recorded_payload.get(key) != requested_payload.get(key)
        )
        raise ValueError(
            "warm-start checkpoint contract differs from the requested run for "
            + ", ".join(mismatches)
        )


def _train_state_checkpoint_item(
    train_state: TrainState, metadata: CheckpointMetadata
) -> dict[str, Any]:
    params, _ = partition_policy_parameters(train_state.policy)
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
    task_leaves = jax.tree_util.tree_leaves(runner_state.task_state)
    if any(getattr(value, "size", 0) > 0 for value in task_leaves):
        item["task_state"] = runner_state.task_state
    return item


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


def _resume_contract_payload(
    metadata: CheckpointMetadata, *, omit_source_revision: bool = False
) -> dict[str, Any]:
    """Return training semantics, excluding budget and persisted running state."""

    payload = to_builtin(metadata)
    payload["ppo"].pop("total_timesteps")
    if omit_source_revision:
        payload.pop("source_revision", None)
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
    dataset = payload.get("dataset")
    if dataset is not None:
        dataset.pop("index_path")
    return payload


def _warm_start_contract_payload(
    metadata: CheckpointMetadata, *, omit_source_revision: bool
) -> dict[str, Any]:
    payload = _resume_contract_payload(
        metadata,
        omit_source_revision=omit_source_revision,
    )
    payload["ppo"] = {"policy": payload["ppo"]["policy"]}
    dataset = payload.get("dataset")
    if dataset is not None:
        dataset.pop("dataset_fingerprint", None)
        if payload.get("task") == "sonic":
            dataset.pop("calibration", None)
    payload.pop("dataset_audit", None)
    return payload


def _load_checkpoint_document(
    path: Path,
    *,
    allow_git_commit_mismatch: bool = False,
    allow_retired_asset_schema: bool = False,
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
        _with_checkpoint_metadata_defaults(
            metadata_payload,
            allow_retired_asset_schema=allow_retired_asset_schema,
        ),
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


def _with_checkpoint_metadata_defaults(
    value: Any,
    *,
    allow_retired_asset_schema: bool = False,
) -> Any:
    if not isinstance(value, dict):
        return value
    if allow_retired_asset_schema:
        value = _without_retired_asset_schema(value)
    ppo = value.get("ppo")
    if isinstance(ppo, dict):
        normalized = dict(value)
        normalized_ppo = dict(ppo)
        normalized_ppo.setdefault("critic_learning_rate", None)
        normalized_ppo.setdefault("symmetry_loss_coef", 0.0)
        if value.get("task") in {"locomotion", "motion_tracking"}:
            legacy_ppo_defaults = {
                "adam_epsilon": 1.0e-5,
                "value_loss_scale": 0.5,
                "normalize_critic_observations": True,
                "observation_norm_update_mode": "online",
                "observation_norm_update_updates": None,
                "observation_norm_epsilon_mode": "variance_floor",
            }
            for name, default in legacy_ppo_defaults.items():
                normalized_ppo.setdefault(name, default)
            policy = ppo.get("policy")
            if isinstance(policy, dict):
                normalized_policy = dict(policy)
                normalized_policy.setdefault("architecture", "mlp")
                normalized_policy.setdefault("layer_norm_final_hidden", False)
                normalized_policy.setdefault("sonic_lora_rank", 32)
                normalized_policy.setdefault("sonic_lora_alpha", 32.0)
                normalized_policy.setdefault("sonic_train_log_std", True)
                normalized_ppo["policy"] = normalized_policy
        normalized["ppo"] = normalized_ppo
        value = normalized
    asset = value.get("asset")
    if isinstance(asset, dict) and "end_effector_payload_mass_kg" not in asset:
        normalized = dict(value)
        normalized_asset = dict(asset)
        normalized_asset["end_effector_payload_mass_kg"] = 0.0
        normalized["asset"] = normalized_asset
        value = normalized
    if value.get("task") == "locomotion":
        normalized = dict(value)
        command = value.get("command")
        if isinstance(command, dict):
            normalized_command = dict(command)
            normalized_command.setdefault("standing_transitions_enabled", False)
            normalized["command"] = normalized_command
        env = value.get("env")
        if isinstance(env, dict) and "gait" not in env:
            raise ValueError(
                "locomotion checkpoint predates the gait-phase observation contract; "
                "use the archived source revision or train a new policy"
            )
        if isinstance(env, dict):
            normalized_env = dict(env)
            if "observation_latency" not in normalized_env:
                normalized_env["observation_latency"] = to_builtin(
                    ObservationLatencyConfig()
                )
            model_randomization = env.get("model_randomization")
            if isinstance(model_randomization, dict):
                normalized_randomization = dict(model_randomization)
                legacy_randomization_defaults = {
                    "added_base_mass_range": (0.0, 0.0),
                    "base_com_offset_range": (0.0, 0.0),
                    "body_mass_scale_range": (1.0, 1.0),
                    "body_mass_scale_names": (),
                    "body_com_offset_range_xyz": (
                        (0.0, 0.0),
                        (0.0, 0.0),
                        (0.0, 0.0),
                    ),
                    "body_com_offset_name": "pelvis_link",
                    "friction_range": (1.0, 1.0),
                    "actuator_strength_range": (1.0, 1.0),
                    "gravity_fraction_range": (0.0, 0.0),
                }
                for name, default in legacy_randomization_defaults.items():
                    if name not in normalized_randomization:
                        normalized_randomization[name] = default
                normalized_env["model_randomization"] = normalized_randomization
            normalized["env"] = normalized_env
        reward = value.get("reward")
        if isinstance(reward, dict):
            normalized_reward = dict(reward)
            weights = reward.get("weights")
            if isinstance(weights, dict):
                normalized_weights = dict(weights)
                for name in (
                    "action_acceleration",
                    "joint_force_peak",
                    "touchdown_impact",
                    "stand_still",
                    "stand_contact",
                    "stand_foot_heading",
                ):
                    if name not in normalized_weights:
                        normalized_weights[name] = 0.0
                normalized_reward["weights"] = normalized_weights
            if "smoothness" not in normalized_reward:
                normalized_reward["smoothness"] = {
                    "ankle_action_weight": 1.0,
                    "waist_action_weight": 1.0,
                }
            tracking = reward.get("tracking")
            if isinstance(tracking, dict) and "ang_vel_z_lpf_cutoff_hz" not in tracking:
                normalized_tracking = dict(tracking)
                normalized_tracking["ang_vel_z_lpf_cutoff_hz"] = 5.0
                normalized_reward["tracking"] = normalized_tracking
            standing = reward.get("standing")
            if isinstance(standing, dict):
                normalized_standing = dict(standing)
                normalized_standing.setdefault(
                    "foot_heading_deadband", math.radians(5.0)
                )
                normalized_standing.setdefault("foot_heading_scale", math.radians(15.0))
                normalized_reward["standing"] = normalized_standing
            gait = reward.get("gait")
            if isinstance(gait, dict):
                normalized_gait = dict(gait)
                legacy_gait_defaults = {
                    "joint_force_soft_limit_body_weights": 1.0,
                    "joint_force_top_k": 12,
                    "joint_force_apply_while_standing": False,
                    "touchdown_force_soft_limit_body_weights": 1.0,
                    "touchdown_vertical_speed_scale": 0.3,
                }
                for name, default in legacy_gait_defaults.items():
                    if name not in normalized_gait:
                        normalized_gait[name] = default
                normalized_reward["gait"] = normalized_gait
            normalized["reward"] = normalized_reward
        return normalized
    if value.get("task") == "sonic":
        sonic_env = value.get("sonic_env")
        if isinstance(sonic_env, dict):
            normalized = dict(value)
            normalized_sonic_env = dict(sonic_env)
            normalized_sonic_env.pop("balanced_idle_probability", None)
            sampling = sonic_env.get("sampling")
            if isinstance(sampling, dict):
                normalized_sampling = dict(sampling)
                normalized_sampling.setdefault("adaptive_warmup_steps", 48_000)
                normalized_sampling.setdefault("adaptive_ramp_steps", 192_000)
                normalized_sonic_env["sampling"] = normalized_sampling
            normalized["sonic_env"] = normalized_sonic_env
            return normalized
    if value.get("task") != "motion_tracking":
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


def _without_retired_asset_schema(value: dict[str, Any]) -> dict[str, Any]:
    asset = value.get("asset")
    if not isinstance(asset, dict):
        return value
    normalized_asset = dict(asset)
    changed = False
    if "end_effector_sphere_mode" in normalized_asset:
        legacy_mode = normalized_asset.pop("end_effector_sphere_mode")
        if legacy_mode not in {"none", "both"}:
            raise ValueError(
                "metadata.asset.end_effector_sphere_mode must be 'none' or "
                f"'both', got {legacy_mode!r}"
            )
        changed = True
    availability = normalized_asset.get("joint_availability")
    if isinstance(availability, dict):
        normalized_availability = dict(availability)
        for name in ("waist_roll", "waist_pitch", "neck_yaw", "head_pitch"):
            if name not in normalized_availability:
                continue
            legacy_enabled = normalized_availability.pop(name)
            if legacy_enabled is not False:
                raise ValueError(
                    f"metadata.asset.joint_availability.{name} must be false "
                    "for the canonical fixed joint"
                )
            changed = True
        normalized_asset["joint_availability"] = normalized_availability
    if not changed:
        return value
    normalized = dict(value)
    normalized["asset"] = normalized_asset
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
    elif task == "twist_teacher":
        metadata_type = TwistTeacherCheckpointMetadata
    elif task == "twist_student":
        metadata_type = TwistStudentCheckpointMetadata
    elif task == "twist2":
        metadata_type = Twist2CheckpointMetadata
    elif task == "sonic":
        metadata_type = SonicCheckpointMetadata
    else:
        raise ValueError(
            f"{path}.task must be 'locomotion', 'motion_tracking', "
            "'twist_teacher', 'twist_student', 'twist2', or 'sonic', "
            f"got {task!r}"
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
    elif isinstance(
        metadata, (TwistTeacherCheckpointMetadata, TwistStudentCheckpointMetadata)
    ):
        _validate_twist_metadata(metadata)
    elif isinstance(metadata, Twist2CheckpointMetadata):
        _validate_twist2_metadata(metadata)
    elif isinstance(metadata, SonicCheckpointMetadata):
        _validate_sonic_metadata(metadata)
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


def _validate_twist_metadata(metadata: TwistCheckpointMetadata) -> None:
    for name, value in (
        ("dataset_fingerprint", metadata.dataset.dataset_fingerprint),
        ("dataset.robot_fingerprint", metadata.dataset.robot_fingerprint),
        ("robot_fingerprint", metadata.robot_fingerprint),
    ):
        _validated_fingerprint(value, f"metadata.{name}")
    if metadata.dataset.robot_fingerprint != metadata.robot_fingerprint:
        raise ValueError("TWIST dataset and checkpoint robot fingerprints differ")
    if not re.fullmatch(r"[0-9a-f]{40}", metadata.dataset.gmr_revision):
        raise ValueError("TWIST dataset GMR revision must be a 40-character commit")
    if not re.fullmatch(r"[0-9a-f]{40}", metadata.source_revision):
        raise ValueError("TWIST source revision must be a 40-character commit")
    if metadata.dataset.schema_version <= 0:
        raise ValueError("TWIST dataset schema version must be positive")
    if metadata.dataset.joint_names != metadata.action_layout.names:
        raise ValueError("TWIST dataset joint order does not match checkpoint actions")
    if metadata.observation_contract.joint_names != metadata.action_layout.names:
        raise ValueError(
            "TWIST observation contract joint order does not match actions"
        )
    missing = set(metadata.observation_contract.key_body_names) - set(
        metadata.dataset.body_names
    )
    if missing:
        raise ValueError(f"TWIST dataset is missing key bodies: {sorted(missing)}")
    expected_architecture = f"twist_{metadata.stage}"
    if metadata.ppo.policy.architecture != expected_architecture:
        raise ValueError(
            f"TWIST {metadata.stage} checkpoint requires policy architecture "
            f"{expected_architecture!r}"
        )
    if metadata.actor_observation_layout != metadata.observation_contract.actor_layout(
        metadata.stage
    ):
        raise ValueError("TWIST actor observation layout differs from its contract")
    if (
        metadata.critic_observation_layout
        != metadata.observation_contract.critic_layout
    ):
        raise ValueError("TWIST critic observation layout differs from its contract")
    if isinstance(metadata, TwistStudentCheckpointMetadata):
        _validated_fingerprint(
            metadata.teacher_fingerprint, "metadata.teacher_fingerprint"
        )
        if metadata.kl_schedule.direction != "student||teacher":
            raise ValueError("TWIST student KL direction must be student||teacher")
        if (
            metadata.kl_schedule.start != 0.1
            or metadata.kl_schedule.end != 0.01
            or metadata.kl_schedule.anneal_updates != 60_000
        ):
            raise ValueError("TWIST student KL schedule differs from the release")
    normalization = (
        metadata.value_norm,
        metadata.observation_norm,
        metadata.critic_observation_norm,
    )
    if any(value is None for value in normalization):
        raise ValueError("TWIST checkpoint normalization snapshots are required")
    assert metadata.value_norm is not None
    assert metadata.observation_norm is not None
    assert metadata.critic_observation_norm is not None
    _validate_running_moments(
        (metadata.value_norm.mean,),
        (metadata.value_norm.var,),
        metadata.value_norm.count,
        name="TWIST value normalization",
    )
    _validate_running_moments(
        metadata.observation_norm.mean,
        metadata.observation_norm.var,
        metadata.observation_norm.count,
        name="TWIST actor observation normalization",
        expected_size=metadata.obs_size,
    )
    _validate_running_moments(
        metadata.critic_observation_norm.mean,
        metadata.critic_observation_norm.var,
        metadata.critic_observation_norm.count,
        name="TWIST critic observation normalization",
        expected_size=metadata.critic_obs_size,
    )


def _validate_twist2_metadata(metadata: Twist2CheckpointMetadata) -> None:
    for name, value in (
        ("dataset_fingerprint", metadata.dataset.dataset_fingerprint),
        ("dataset.robot_fingerprint", metadata.dataset.robot_fingerprint),
        ("robot_fingerprint", metadata.robot_fingerprint),
    ):
        _validated_fingerprint(value, f"metadata.{name}")
    if metadata.dataset.robot_fingerprint != metadata.robot_fingerprint:
        raise ValueError("TWIST2 dataset and checkpoint robot fingerprints differ")
    if not re.fullmatch(r"[0-9a-f]{40}", metadata.dataset.gmr_revision):
        raise ValueError("TWIST2 dataset GMR revision must be a 40-character commit")
    if not re.fullmatch(r"[0-9a-f]{40}", metadata.source_revision):
        raise ValueError("TWIST2 source revision must be a 40-character commit")
    if metadata.dataset.schema_version <= 0:
        raise ValueError("TWIST2 dataset schema version must be positive")
    if metadata.dataset.joint_names != metadata.action_layout.names:
        raise ValueError("TWIST2 dataset joint order does not match checkpoint actions")
    if metadata.observation_contract.joint_names != metadata.action_layout.names:
        raise ValueError("TWIST2 observation joint order does not match actions")
    missing = set(metadata.observation_contract.key_body_names) - set(
        metadata.dataset.body_names
    )
    if missing:
        raise ValueError(f"TWIST2 dataset is missing key bodies: {sorted(missing)}")
    if metadata.ppo.policy.architecture != "twist2":
        raise ValueError("TWIST2 checkpoint requires policy architecture 'twist2'")
    if metadata.actor_observation_layout != metadata.observation_contract.actor_layout:
        raise ValueError("TWIST2 actor observation layout differs from its contract")
    if (
        metadata.critic_observation_layout
        != metadata.observation_contract.critic_layout
    ):
        raise ValueError("TWIST2 critic observation layout differs from its contract")
    if metadata.parity_contract != Twist2ParityContract():
        raise ValueError("TWIST2 parity contract differs from the maintained behavior")
    normalization = (
        metadata.value_norm,
        metadata.observation_norm,
        metadata.critic_observation_norm,
    )
    if any(value is None for value in normalization):
        raise ValueError("TWIST2 checkpoint normalization snapshots are required")
    assert metadata.value_norm is not None
    assert metadata.observation_norm is not None
    assert metadata.critic_observation_norm is not None
    _validate_running_moments(
        (metadata.value_norm.mean,),
        (metadata.value_norm.var,),
        metadata.value_norm.count,
        name="TWIST2 value normalization",
    )
    _validate_running_moments(
        metadata.observation_norm.mean,
        metadata.observation_norm.var,
        metadata.observation_norm.count,
        name="TWIST2 actor observation normalization",
        expected_size=metadata.obs_size,
    )
    _validate_running_moments(
        metadata.critic_observation_norm.mean,
        metadata.critic_observation_norm.var,
        metadata.critic_observation_norm.count,
        name="TWIST2 critic observation normalization",
        expected_size=metadata.critic_obs_size,
    )


def _validate_sonic_metadata(metadata: SonicCheckpointMetadata) -> None:
    for name, value in (
        ("dataset_fingerprint", metadata.dataset.dataset_fingerprint),
        ("robot_fingerprint", metadata.dataset.robot_fingerprint),
        ("bello_gmr_config_sha256", metadata.dataset.bello_gmr_config_sha256),
        ("release_checkpoint_sha256", metadata.release_checkpoint_sha256),
        ("release_artifact_sha256", metadata.release_artifact_sha256),
        ("g1_config_sha256", metadata.dataset.g1_config_sha256),
        ("g1_robot_sha256", metadata.dataset.g1_robot_sha256),
    ):
        _validated_fingerprint(value, f"metadata.{name}")
    for name, revision in (
        ("bello_gmr_revision", metadata.dataset.bello_gmr_revision),
        ("g1_gmr_revision", metadata.dataset.g1_gmr_revision),
        ("source_revision", metadata.source_revision),
    ):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"SONIC {name} must be a 40-character Git revision")
    if metadata.dataset.schema_version <= 0:
        raise ValueError("SONIC dataset schema version must be positive")
    if metadata.dataset.joint_names != metadata.action_layout.names:
        raise ValueError("SONIC dataset joint order does not match actions")
    if (
        len(metadata.dataset.qpos_joint_names) != len(metadata.dataset.joint_names)
        or len(set(metadata.dataset.qpos_joint_names))
        != len(metadata.dataset.qpos_joint_names)
        or set(metadata.dataset.qpos_joint_names) != set(metadata.dataset.joint_names)
    ):
        raise ValueError("SONIC dataset qpos joint order is not a joint permutation")
    if metadata.dataset.calibration.bello_joint_names != metadata.action_layout.names:
        raise ValueError("SONIC calibration joint order does not match actions")
    if metadata.observation_contract.joint_names != metadata.action_layout.names:
        raise ValueError("SONIC observation joint order does not match actions")
    missing = set(metadata.observation_contract.required_body_names) - set(
        metadata.dataset.body_names
    )
    if missing:
        raise ValueError(f"SONIC dataset is missing tracked bodies: {sorted(missing)}")
    if metadata.ppo.policy.architecture != "sonic":
        raise ValueError("SONIC checkpoint requires policy architecture 'sonic'")
    if metadata.ppo.policy.obs_stack_size != 1:
        raise ValueError("SONIC owns its ten-frame history and cannot be PPO-stacked")
    if metadata.actor_observation_layout != metadata.observation_contract.actor_layout:
        raise ValueError("SONIC actor observation layout differs from its contract")
    if (
        metadata.critic_observation_layout
        != metadata.observation_contract.critic_layout
    ):
        raise ValueError("SONIC critic observation layout differs from its contract")
    normalization = (
        metadata.value_norm,
        metadata.observation_norm,
        metadata.critic_observation_norm,
    )
    if any(value is None for value in normalization):
        raise ValueError("SONIC checkpoint normalization snapshots are required")
    assert metadata.value_norm is not None
    assert metadata.observation_norm is not None
    assert metadata.critic_observation_norm is not None
    _validate_running_moments(
        (metadata.value_norm.mean,),
        (metadata.value_norm.var,),
        metadata.value_norm.count,
        name="SONIC value normalization",
    )
    _validate_running_moments(
        metadata.observation_norm.mean,
        metadata.observation_norm.var,
        metadata.observation_norm.count,
        name="SONIC actor observation normalization",
        expected_size=metadata.obs_size,
    )
    _validate_running_moments(
        metadata.critic_observation_norm.mean,
        metadata.critic_observation_norm.var,
        metadata.critic_observation_norm.count,
        name="SONIC critic observation normalization",
        expected_size=metadata.critic_obs_size,
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
    if not isinstance(
        metadata,
        (
            MotionCheckpointMetadata,
            TwistTeacherCheckpointMetadata,
            TwistStudentCheckpointMetadata,
            Twist2CheckpointMetadata,
            SonicCheckpointMetadata,
        ),
    ):
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
    checkpointer = ocp.PyTreeCheckpointer()
    try:
        restored_storage = checkpointer.restore(
            path,
            item=storage_template,
            restore_args=restore_args,
        )
    finally:
        checkpointer.close()
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


def twist_metadata_with_runner_state(
    metadata: TwistCheckpointMetadata | Twist2CheckpointMetadata,
    runner_state: RunnerState,
) -> TwistCheckpointMetadata | Twist2CheckpointMetadata:
    """Attach normalization state to a typed TWIST checkpoint."""

    return replace(
        metadata,
        value_norm=value_norm_metadata(runner_state.value_norm_state),
        observation_norm=observation_norm_metadata(runner_state.observation_norm_state),
        critic_observation_norm=observation_norm_metadata(
            runner_state.critic_observation_norm_state
        ),
    )


def sonic_metadata_with_runner_state(
    metadata: SonicCheckpointMetadata, runner_state: RunnerState
) -> SonicCheckpointMetadata:
    """Attach PPO normalization state to a typed SONIC checkpoint."""

    return replace(
        metadata,
        value_norm=value_norm_metadata(runner_state.value_norm_state),
        observation_norm=observation_norm_metadata(runner_state.observation_norm_state),
        critic_observation_norm=observation_norm_metadata(
            runner_state.critic_observation_norm_state
        ),
    )


def checkpoint_metadata_fingerprint(metadata: CheckpointMetadata) -> str:
    """Return the canonical metadata identity used to bind a student teacher."""

    return _fingerprint(to_builtin(metadata))


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
