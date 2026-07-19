"""Orbax checkpoint helpers for PPO training and evaluation."""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, distribution
from importlib.resources import files
import json
import os
import re
import shutil
import tempfile
import tomllib
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from bello_mujoco.assets_pipeline.prepare import (
    AssetPrepConfig,
    PreparedAsset,
    prepare_asset,
    source_asset_fingerprint,
)
from bello_mujoco.contracts import ActionLayout, ObservationLayout
from bello_mujoco.envs.locomotion import (
    BelloEnvConfig,
    BelloLocomotionEnv,
    DoneConfig,
)
from bello_mujoco.envs.rewards import CommandConfig, RewardConfig
from bello_mujoco.sim import WarpConfig
from bello_mujoco.terrain import ROBOT_ONLY_TERRAIN, TerrainConfig, runtime_mjcf_path
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
CHECKPOINT_SCHEMA_VERSION = 5
CHECKPOINT_KIND = "training_resume"

_CHECKPOINT_DOCUMENT_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "update",
        "config_fingerprint",
        "train_layout_fingerprint",
        "continuation_layout_fingerprint",
        "runtime_fingerprints",
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


@dataclass(frozen=True)
class RuntimeFingerprints:
    """Immutable inputs needed to interpret a policy checkpoint."""

    source_assets: str
    prepared_assets: str
    simulation_scene: str
    viewer_scene: str
    implementation: str
    dependency_lock: str
    active_distributions: str


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
    runtime_fingerprints: RuntimeFingerprints,
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
        runtime_fingerprints,
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
    }


def _checkpoint_document(
    metadata: CheckpointMetadata,
    train_item: dict[str, Any],
    continuation_item: dict[str, Any],
    runtime_fingerprints: RuntimeFingerprints,
) -> dict[str, Any]:
    if metadata.asset.output_root is not None:
        raise ValueError(
            "checkpoint asset.output_root is an operational cache location and "
            "must be normalized to null before saving"
        )
    metadata_payload = to_builtin(metadata)
    resolved_metadata = dataclass_from_dict(
        CheckpointMetadata,
        metadata_payload,
        strict=True,
        require_all_fields=True,
        path="metadata",
    )
    _validate_metadata_layouts(resolved_metadata)
    current_runtime_fingerprints = capture_runtime_fingerprints(resolved_metadata)
    if current_runtime_fingerprints != runtime_fingerprints:
        mismatches = [
            field.name
            for field in fields(RuntimeFingerprints)
            if getattr(current_runtime_fingerprints, field.name)
            != getattr(runtime_fingerprints, field.name)
        ]
        raise RuntimeError(
            "checkpoint runtime inputs changed after training initialization: "
            f"{', '.join(mismatches)}"
        )
    update = _validate_update(train_item["update"], resolved_metadata.ppo)
    canonical_payload = to_builtin(resolved_metadata)
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": CHECKPOINT_KIND,
        "update": update,
        "config_fingerprint": _fingerprint(canonical_payload),
        "train_layout_fingerprint": _layout_fingerprint(train_item),
        "continuation_layout_fingerprint": _layout_fingerprint(continuation_item),
        "runtime_fingerprints": to_builtin(runtime_fingerprints),
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
    recorded_runtime = dataclass_from_dict(
        RuntimeFingerprints,
        document["runtime_fingerprints"],
        strict=True,
        require_all_fields=True,
        path="runtime_fingerprints",
    )
    for field in fields(RuntimeFingerprints):
        _validated_fingerprint(
            getattr(recorded_runtime, field.name),
            f"runtime_fingerprints.{field.name}",
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
    current_runtime = capture_runtime_fingerprints(metadata)
    if current_runtime != recorded_runtime:
        mismatches = [
            field.name
            for field in fields(RuntimeFingerprints)
            if getattr(current_runtime, field.name)
            != getattr(recorded_runtime, field.name)
        ]
        raise ValueError(
            "checkpoint runtime fingerprint mismatch for "
            f"{', '.join(mismatches)}; evaluate or resume with the exact source, "
            "generated assets, implementation, uv.lock, and installed runtime "
            "versions that created it"
        )
    update = document["update"]
    if type(update) is not int:
        raise ValueError("checkpoint update must be an integer")
    _validate_update(update, metadata.ppo)
    return metadata, update, train_fingerprint, continuation_fingerprint


def _validate_metadata_layouts(metadata: CheckpointMetadata) -> None:
    if metadata.asset.output_root is not None:
        raise ValueError(
            "checkpoint metadata asset.output_root must be null; generated asset "
            "cache locations are not part of the training contract"
        )
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


def normalized_asset_config(config: AssetPrepConfig) -> AssetPrepConfig:
    """Remove the host-specific prepared-asset cache location."""

    return replace(config, output_root=None)


def capture_runtime_fingerprints(
    metadata: CheckpointMetadata,
) -> RuntimeFingerprints:
    return capture_asset_runtime_fingerprints(metadata.asset, metadata.env.terrain)


def capture_asset_runtime_fingerprints(
    asset_config: AssetPrepConfig,
    terrain: TerrainConfig = ROBOT_ONLY_TERRAIN,
) -> RuntimeFingerprints:
    prepared = prepare_asset(asset_config)
    return RuntimeFingerprints(
        source_assets=f"sha256:{source_asset_fingerprint()}",
        prepared_assets=_prepared_asset_closure_fingerprint(prepared),
        simulation_scene=_file_fingerprint(
            runtime_mjcf_path(prepared.mjcf_path, terrain)
        ),
        viewer_scene=_file_fingerprint(
            runtime_mjcf_path(prepared.viewer_mjcf_path, terrain)
        ),
        implementation=_implementation_fingerprint(),
        dependency_lock=_content_fingerprint(_runtime_input_bytes("uv.lock")),
        active_distributions=_active_distributions_fingerprint(),
    )


def _active_distributions_fingerprint() -> str:
    pairs = _active_runtime_distribution_pairs()
    locked_versions = _locked_distribution_versions()
    mismatches = []
    for pair in pairs:
        name, version = pair.rsplit("==", 1)
        locked_version = locked_versions.get(name)
        if locked_version != version:
            expected = "missing" if locked_version is None else locked_version
            mismatches.append(f"{pair} (lock: {expected})")
    if mismatches:
        raise RuntimeError(
            "active runtime distributions do not match uv.lock: "
            + ", ".join(mismatches)
        )
    return _fingerprint(pairs)


def _active_runtime_distribution_pairs() -> tuple[str, ...]:
    """Resolve installed runtime dependencies, including requested extras."""

    pending = deque([(canonicalize_name("bello-mujoco"), frozenset())])
    processed_extras: dict[str, frozenset[str]] = {}
    versions: dict[str, str] = {}
    marker_environment = default_environment()
    while pending:
        requested_name, requested_extras = pending.popleft()
        previous_extras = processed_extras.get(requested_name, frozenset())
        if requested_name in processed_extras and requested_extras <= previous_extras:
            continue
        active_extras = previous_extras | requested_extras
        processed_extras[requested_name] = active_extras
        try:
            installed = distribution(requested_name)
        except PackageNotFoundError as error:
            raise RuntimeError(
                f"required runtime distribution is not installed: {requested_name}"
            ) from error
        name = canonicalize_name(installed.metadata["Name"] or requested_name)
        existing_version = versions.setdefault(name, installed.version)
        if existing_version != installed.version:
            raise RuntimeError(
                f"multiple active versions found for {name}: "
                f"{existing_version}, {installed.version}"
            )
        marker_extras = ("", *sorted(active_extras))
        for raw_requirement in installed.requires or ():
            requirement = Requirement(raw_requirement)
            if requirement.marker is not None and not any(
                requirement.marker.evaluate(marker_environment | {"extra": extra})
                for extra in marker_extras
            ):
                continue
            pending.append(
                (canonicalize_name(requirement.name), frozenset(requirement.extras))
            )
    return tuple(f"{name}=={versions[name]}" for name in sorted(versions))


def _locked_distribution_versions() -> dict[str, str]:
    try:
        lock = tomllib.loads(_runtime_input_bytes("uv.lock").decode("utf-8"))
        packages = lock["package"]
    except (KeyError, TypeError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError("bundled uv.lock is malformed") from error
    if not isinstance(packages, list):
        raise RuntimeError("bundled uv.lock package table must be a list")
    versions: dict[str, str] = {}
    for package in packages:
        if not isinstance(package, dict):
            raise RuntimeError("bundled uv.lock contains a malformed package entry")
        try:
            name = canonicalize_name(package["name"])
            version = str(package["version"])
        except (KeyError, TypeError) as error:
            raise RuntimeError(
                "bundled uv.lock contains a malformed package entry"
            ) from error
        previous = versions.setdefault(name, version)
        if previous != version:
            raise RuntimeError(
                f"bundled uv.lock contains multiple versions for {name}: "
                f"{previous}, {version}"
            )
    return versions


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


def _prepared_asset_closure_fingerprint(prepared: PreparedAsset) -> str:
    """Hash generated robot descriptions and every local file they reference."""

    documents = (
        Path(prepared.urdf_path).resolve(),
        Path(prepared.mjcf_path).resolve(),
        Path(prepared.viewer_mjcf_path).resolve(),
    )
    asset_root = documents[0].parents[1]
    if any(document.parents[1] != asset_root for document in documents[1:]):
        raise ValueError("prepared robot descriptions do not share an asset root")

    closure = set(documents)
    xml_queue = list(documents)
    parsed_xml: set[Path] = set()
    while xml_queue:
        document = xml_queue.pop()
        if document in parsed_xml:
            continue
        parsed_xml.add(document)
        try:
            root = ET.parse(document).getroot()
        except (FileNotFoundError, ET.ParseError) as error:
            raise ValueError(
                f"could not parse prepared robot description {document}"
            ) from error
        for element in root.iter():
            tag = element.tag.rsplit("}", 1)[-1]
            reference = None
            if tag == "mesh":
                reference = element.get("filename") or element.get("file")
            elif tag in {"hfield", "include", "skin", "texture"}:
                reference = element.get("file")
            if not reference:
                continue
            if "://" in reference:
                raise ValueError(
                    "prepared robot descriptions must use local file references, "
                    f"got {reference!r} in {document}"
                )
            dependency = (document.parent / reference).resolve()
            try:
                dependency.relative_to(asset_root)
            except ValueError as error:
                raise ValueError(
                    "prepared robot dependency escapes its generated asset root: "
                    f"{reference!r} in {document}"
                ) from error
            if not dependency.is_file():
                raise FileNotFoundError(
                    f"prepared robot dependency is missing: {dependency}"
                )
            closure.add(dependency)
            if tag == "include":
                xml_queue.append(dependency)

    hasher = hashlib.sha256()
    for path in sorted(
        closure, key=lambda item: item.relative_to(asset_root).as_posix()
    ):
        logical_path = path.relative_to(asset_root).as_posix()
        hasher.update(logical_path.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return f"sha256:{hasher.hexdigest()}"


def _implementation_fingerprint() -> str:
    package_root = Path(__file__).resolve().parent
    hasher = hashlib.sha256()
    hasher.update(f"checkpoint-schema:{CHECKPOINT_SCHEMA_VERSION}\0".encode())
    for path in sorted(package_root.rglob("*.py")):
        hasher.update(path.relative_to(package_root).as_posix().encode())
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    hasher.update(b"pyproject.toml\0")
    hasher.update(_runtime_input_bytes("pyproject.toml"))
    return f"sha256:{hasher.hexdigest()}"


def _runtime_input_bytes(name: str) -> bytes:
    package_file = Path(__file__).resolve()
    source_root = package_file.parents[2]
    source_package_file = source_root / "src" / "bello_mujoco" / "checkpoint.py"
    if source_package_file.is_file() and source_package_file.resolve() == package_file:
        source_input = source_root / name
        if not source_input.is_file():
            raise FileNotFoundError(
                f"checkpoint runtime input is missing: {source_input}"
            )
        return source_input.read_bytes()

    resource = files("bello_mujoco").joinpath("_runtime", name)
    try:
        return resource.read_bytes()
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"installed checkpoint runtime input is missing: bello_mujoco/_runtime/{name}"
        ) from error


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
            if len(value) != len(args):
                location = path or "tuple"
                raise ValueError(
                    f"{location} must contain exactly {len(args)} values, got "
                    f"{len(value)}"
                )
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
