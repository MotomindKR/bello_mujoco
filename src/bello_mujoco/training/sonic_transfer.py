"""Train the imported SONIC policy on a retargeted Bello Mini corpus."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import traceback

import equinox as eqx
import jax

from bello_mujoco.envs.motion_tracking import MotionTrackingEnvConfig
from bello_mujoco.envs.sonic_transfer import BelloSonicTransferEnv
from bello_mujoco.models.compiler import ModelConfig
from bello_mujoco.repository import (
    capture_repository_commit,
    require_checkpoint_output_directory,
    require_repository_commit,
)
from bello_mujoco.sim import WarpConfig
from bello_mujoco.sonic.checkpoint import import_sonic_checkpoint
from bello_mujoco.sonic.corpus import MANIFEST, file_sha256, verify_artifact
from bello_mujoco.training.checkpoint import CheckpointConfig, checkpoint_path
from bello_mujoco.training.ppo import init_runner_state_with_policy, update_runner_state
from bello_mujoco.training.ppo_config import PPOConfig, validate_ppo_config
from bello_mujoco.training.runtime import training_updates
from bello_mujoco.training.tracking import (
    WandbConfig,
    close_run,
    create_wandb_run,
    log_config,
    track_checkpoint_artifact,
    to_builtin,
)

_PINNED_ARTIFACT_PATTERN = re.compile(r".+:(?:v[0-9]+|[0-9a-f]{32,})\Z")


@dataclass(frozen=True)
class SonicDatasetConfig:
    directory: Path
    evaluation_clips: tuple[str, ...] = ()
    artifact: str | None = None
    max_array_bytes: int = 4 * 1024**3


@dataclass(frozen=True)
class SonicPretrainedConfig:
    checkpoint: Path
    revision: str


@dataclass(frozen=True)
class SonicTransferTrainingRunConfig:
    asset_config: ModelConfig
    dataset: SonicDatasetConfig
    pretrained: SonicPretrainedConfig
    config: PPOConfig = PPOConfig()
    env_config: MotionTrackingEnvConfig = MotionTrackingEnvConfig()
    checkpoint_config: CheckpointConfig = CheckpointConfig()
    wandb_config: WandbConfig = WandbConfig()
    warp: WarpConfig = WarpConfig()


def compile_sonic_update(env, config, static_policy, optimizer):
    """Pass corpus buffers as device arguments, never multi-GB HLO constants."""

    @jax.jit
    def update(state, bank):
        dynamic_env = replace(env, reference_bank=bank)
        return update_runner_state(state, static_policy, optimizer, dynamic_env, config)

    return partial(update, bank=env.reference_bank)


def run_training(run_config: SonicTransferTrainingRunConfig) -> None:
    validate_ppo_config(run_config.config)
    if not re.fullmatch(r"[0-9a-f]{40}", run_config.pretrained.revision):
        raise ValueError("SONIC training requires an immutable pretrained revision")
    if run_config.dataset.artifact is None:
        raise ValueError(
            "SONIC training requires the immutable W&B dataset artifact reference"
        )
    if _PINNED_ARTIFACT_PATTERN.fullmatch(run_config.dataset.artifact) is None:
        raise ValueError(
            "SONIC dataset artifact must include an immutable version or digest"
        )
    paths = tuple(sorted(run_config.dataset.directory.glob("*.npz")))
    if not paths:
        raise FileNotFoundError(
            f"SONIC dataset has no .npz clips: {run_config.dataset.directory}"
        )
    git_commit = capture_repository_commit()
    checkpoint_config = replace(
        run_config.checkpoint_config,
        directory=require_checkpoint_output_directory(
            run_config.checkpoint_config.directory
        ),
    )
    run_config = replace(run_config, checkpoint_config=checkpoint_config)
    checkpoint_root = checkpoint_path(checkpoint_config, run_config.config)
    run = None
    succeeded = False
    try:
        run = create_wandb_run(run_config.wandb_config)
        if run is not None:
            run.define_metric("update")
            run.define_metric("train/*", step_metric="update")
        dataset_digest = verify_artifact(
            run_config.dataset.directory,
            run_config.dataset.artifact,
            run_config.dataset.evaluation_clips,
            run=run,
        )
        corpus_bytes = sum(
            entry["array_bytes"]
            for entry in json.loads(
                (run_config.dataset.directory / MANIFEST).read_text()
            )["clips"]
        )
        if (
            run_config.dataset.max_array_bytes <= 0
            or corpus_bytes > run_config.dataset.max_array_bytes
        ):
            raise ValueError(
                f"corpus arrays need {corpus_bytes} bytes, exceeding dataset.max_array_bytes={run_config.dataset.max_array_bytes}; curate a smaller corpus or explicitly budget more device memory"
            )
        env = BelloSonicTransferEnv.load_corpus(
            run_config.asset_config,
            run_config.env_config,
            paths,
            evaluation_names=run_config.dataset.evaluation_clips,
            warp=run_config.warp,
            batch_size=run_config.config.num_envs,
        )
        imported = import_sonic_checkpoint(
            run_config.pretrained.checkpoint,
            revision=run_config.pretrained.revision,
            adapter=env.embodiment,
            key=jax.random.PRNGKey(run_config.config.seed),
        )
        log_config(
            run,
            {
                "task": "sonic_transfer",
                "config": run_config,
                "git_commit": git_commit,
                "pretrained_sha256": imported.sha256,
                "dataset_artifact": run_config.dataset.artifact,
                "dataset_manifest_sha256": dataset_digest,
                "clips": env.reference_bank.clip_names,
            },
        )
        initialized = init_runner_state_with_policy(
            jax.random.PRNGKey(run_config.config.seed),
            env,
            run_config.config,
            imported.policy,
        )
        if checkpoint_config.resume_path is not None:
            initialized = _restore_checkpoint(
                checkpoint_config.resume_path,
                initialized,
                run_config,
                git_commit=git_commit,
                pretrained_sha256=imported.sha256,
                clip_names=env.reference_bank.clip_names,
                dataset_digest=dataset_digest,
            )
        runner_state = initialized[0]
        update_fn = compile_sonic_update(
            env, run_config.config, initialized[1], initialized[2]
        )
        for update in training_updates(
            env, run_config.config, initialized, run, update_fn=update_fn
        ):
            runner_state, metrics, _ = update
            update_number = int(jax.device_get(runner_state.update))
            if (
                checkpoint_config.enabled
                and checkpoint_config.every_updates is not None
                and update_number % checkpoint_config.every_updates == 0
                and update_number < run_config.config.num_updates
            ):
                scheduled = checkpoint_root / f"update_{update_number:06d}"
                _save_checkpoint(
                    scheduled,
                    runner_state,
                    initialized[1],
                    run_config,
                    git_commit=git_commit,
                    pretrained_sha256=imported.sha256,
                    clip_names=env.reference_bank.clip_names,
                    dataset_digest=dataset_digest,
                )
                _upload_checkpoint(run, scheduled, checkpoint_config)
        root = checkpoint_root / "final"
        _save_checkpoint(
            root,
            runner_state,
            initialized[1],
            run_config,
            git_commit=git_commit,
            pretrained_sha256=imported.sha256,
            clip_names=env.reference_bank.clip_names,
            dataset_digest=dataset_digest,
        )
        _upload_checkpoint(run, root, checkpoint_config)
        print(metrics)
        print({"checkpoint_path": str(root)})
        succeeded = True
    except Exception:
        traceback.print_exc()
        raise
    finally:
        close_run(run, exit_code=0 if succeeded else 1)


def _save_checkpoint(
    path: Path,
    runner_state,
    static_policy,
    run_config: SonicTransferTrainingRunConfig,
    *,
    git_commit: str,
    pretrained_sha256: str,
    clip_names: tuple[str, ...],
    dataset_digest: str | None = None,
) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{path.name}.tmp-", dir=path.parent))
    document = {
        "task": "sonic_transfer",
        "update": int(jax.device_get(runner_state.update)),
        "git_commit": git_commit,
        "pretrained_revision": run_config.pretrained.revision,
        "pretrained_sha256": pretrained_sha256,
        "dataset_artifact": run_config.dataset.artifact,
        "dataset_manifest_sha256": dataset_digest,
        "clips": clip_names,
        "resolved_config": to_builtin(run_config),
    }
    try:
        if dataset_digest is not None:
            source_manifest = run_config.dataset.directory / MANIFEST
            if file_sha256(source_manifest) != dataset_digest:
                raise ValueError("corpus manifest changed during training")
            shutil.copyfile(source_manifest, staging / MANIFEST)
        state_path = staging / "state.eqx"
        eqx.tree_serialise_leaves(state_path, runner_state)
        policy_path = staging / "policy.eqx"
        eqx.tree_serialise_leaves(
            policy_path,
            eqx.combine(runner_state.params, static_policy),
        )
        metadata_path = staging / "metadata.json"
        metadata_path.write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        for file_path in staging.iterdir():
            with file_path.open("rb") as stream:
                os.fsync(stream.fileno())
        descriptor = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        require_repository_commit(git_commit)
        staging.rename(path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _restore_checkpoint(
    path: Path,
    initialized,
    run_config: SonicTransferTrainingRunConfig,
    *,
    git_commit: str,
    pretrained_sha256: str,
    clip_names: tuple[str, ...],
    dataset_digest: str | None = None,
):
    path = Path(path).expanduser().resolve()
    try:
        document = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("SONIC checkpoint metadata is not valid JSON") from error
    expected = {
        "task": "sonic_transfer",
        "git_commit": git_commit,
        "pretrained_revision": run_config.pretrained.revision,
        "pretrained_sha256": pretrained_sha256,
        "dataset_artifact": run_config.dataset.artifact,
        "dataset_manifest_sha256": dataset_digest,
        "clips": list(clip_names),
    }
    mismatches = sorted(
        key for key, value in expected.items() if document.get(key) != value
    )
    if mismatches:
        raise ValueError(
            "SONIC resume checkpoint differs from the requested run for "
            + ", ".join(mismatches)
        )
    recorded_contract = _resume_contract(document.get("resolved_config"))
    requested_contract = _resume_contract(to_builtin(run_config))
    if recorded_contract != requested_contract:
        changed = sorted(
            key
            for key in set(recorded_contract) | set(requested_contract)
            if recorded_contract.get(key) != requested_contract.get(key)
        )
        raise ValueError(
            "SONIC resume training semantics changed for " + ", ".join(changed)
        )
    runner_state, static_policy, optimizer = initialized
    restored = eqx.tree_deserialise_leaves(path / "state.eqx", runner_state)
    update = int(jax.device_get(restored.update))
    if not 0 <= update < run_config.config.num_updates:
        raise ValueError(
            "SONIC resume update must be inside the requested unfinished budget"
        )
    return jax.device_put(restored), static_policy, optimizer


def _resume_contract(value) -> dict:
    if not isinstance(value, dict):
        raise TypeError("SONIC resolved_config must be a mapping")
    value = json.loads(json.dumps(value))
    value.pop("checkpoint_config", None)
    value.pop("wandb_config", None)
    value["config"].pop("total_timesteps", None)
    value["dataset"].pop("directory", None)
    value["pretrained"].pop("checkpoint", None)
    return value


def _upload_checkpoint(run, path: Path, config: CheckpointConfig) -> None:
    if run is None or not config.upload_wandb_artifact:
        return
    track_checkpoint_artifact(
        run,
        path,
        name=config.wandb_artifact_name or path.parent.name,
        artifact_type=config.wandb_artifact_type,
    )
