"""Atomic full-state continuation and actor-only evaluation checkpoints."""

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import tempfile

import equinox as eqx
import jax
import numpy as np

from bello_mujoco.repository import require_repository_commit
from bello_mujoco.sonic.corpus import MANIFEST, file_sha256
from bello_mujoco.training.tracking import to_builtin


_TRANSFER_CHUNK_BYTES = 16 * 1024 * 1024


def _serialise_leaf(stream, value):
    """Keep Equinox's NPY stream format without materializing a replay leaf on the host."""
    if (
        not isinstance(value, (jax.Array, np.ndarray))
        or value.nbytes <= _TRANSFER_CHUNK_BYTES
    ):
        eqx.default_serialise_filter_spec(stream, value)
        return
    dtype = np.dtype(value.dtype)
    if dtype.kind not in "biufc":
        eqx.default_serialise_filter_spec(stream, value)
        return
    np.lib.format.write_array_header_1_0(
        stream,
        {
            "descr": np.lib.format.dtype_to_descr(dtype),
            "fortran_order": False,
            "shape": value.shape,
        },
    )
    row_bytes = value.nbytes // value.shape[0]
    rows = max(1, _TRANSFER_CHUNK_BYTES // row_bytes)
    for start in range(0, value.shape[0], rows):
        chunk = np.ascontiguousarray(jax.device_get(value[start : start + rows]))
        stream.write(memoryview(chunk).cast("B"))


def resume_contract(run_config):
    value = to_builtin(run_config)
    value.pop("checkpoint_config")
    value.pop("wandb_config")
    value.pop("checkpoint_evaluation")
    value.pop("max_training_seconds")
    value["config"].pop("total_timesteps")
    value["dataset"].pop("directory")
    return value


def save_checkpoint(path, runner, run_config, *, git_commit, dataset_digest):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if file_sha256(run_config.dataset.directory / MANIFEST) != dataset_digest:
        raise ValueError("corpus manifest changed during training")
    require_repository_commit(git_commit)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
    document = {
        "task": "supermimic",
        "git_commit": git_commit,
        "dataset_manifest_sha256": dataset_digest,
        "update": int(runner.update),
        "vector_steps": int(runner.collector.vector_steps),
        "actor_updates": int(runner.actor_updates),
        "resolved_config": to_builtin(run_config),
    }
    try:
        eqx.tree_serialise_leaves(
            staging / "state.eqx", runner, filter_spec=_serialise_leaf
        )
        learner = runner.learner
        eqx.tree_serialise_leaves(
            staging / "actor.eqx",
            (learner.actor, learner.observation_norm, learner.goal_norm),
        )
        eqx.tree_serialise_leaves(
            staging / "dynamics.eqx",
            (learner.model, learner.input_norm, learner.output_norm),
        )
        shutil.copyfile(run_config.dataset.directory / MANIFEST, staging / MANIFEST)
        document["state_sha256"] = file_sha256(staging / "state.eqx")
        document["actor_sha256"] = file_sha256(staging / "actor.eqx")
        document["dynamics_sha256"] = file_sha256(staging / "dynamics.eqx")
        (staging / "metadata.json").write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        for child in staging.iterdir():
            with child.open("rb") as stream:
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


def checkpoint_metadata(path):
    path = Path(path)
    document = json.loads((path / "metadata.json").read_text())
    required = {
        "task",
        "git_commit",
        "dataset_manifest_sha256",
        "update",
        "vector_steps",
        "actor_updates",
        "resolved_config",
        "state_sha256",
        "actor_sha256",
        "dynamics_sha256",
    }
    if set(document) != required or document["task"] != "supermimic":
        raise ValueError("incompatible SuperMimic checkpoint metadata")
    for name in ("update", "vector_steps", "actor_updates"):
        if type(document[name]) is not int or document[name] < 0:
            raise ValueError(f"invalid checkpoint {name}")
    if file_sha256(path / MANIFEST) != document["dataset_manifest_sha256"]:
        raise ValueError("checkpoint corpus manifest checksum mismatch")
    require_repository_commit(document["git_commit"])
    return document


def restore_checkpoint(path, template, run_config, *, git_commit, dataset_digest):
    from bello_mujoco.configuration import dataclass_from_dict
    from bello_mujoco.training.supermimic import SuperMimicTrainingRunConfig

    path = Path(path)
    document = checkpoint_metadata(path)
    saved = dataclass_from_dict(
        SuperMimicTrainingRunConfig, document["resolved_config"], strict=True
    )
    if (
        document["git_commit"] != git_commit
        or document["dataset_manifest_sha256"] != dataset_digest
    ):
        raise ValueError("checkpoint source or corpus does not match this training run")
    if resume_contract(saved) != resume_contract(run_config):
        raise ValueError("SuperMimic resume configuration changed")
    if file_sha256(path / "state.eqx") != document["state_sha256"]:
        raise ValueError("checkpoint state checksum mismatch")
    restored = eqx.tree_deserialise_leaves(path / "state.eqx", template)
    if (
        int(restored.update) != document["update"]
        or int(restored.collector.vector_steps) != document["vector_steps"]
        or int(restored.actor_updates) != document["actor_updates"]
    ):
        raise ValueError("checkpoint counters disagree with metadata")
    if (
        int(restored.collector.vector_steps) * run_config.config.num_envs
        >= run_config.config.total_timesteps
    ):
        raise ValueError("resume requires an unfinished training budget")
    return restored


def _evaluation_environment(
    path, *, dataset_directory, reference_clips, use_viewer_model=False
):
    from bello_mujoco.configuration import dataclass_from_dict
    from bello_mujoco.training.supermimic import SuperMimicTrainingRunConfig
    from bello_mujoco.training.supermimic import load_environment
    from bello_mujoco.sonic.corpus import verify_corpus

    path = Path(path)
    document = checkpoint_metadata(path)
    config = dataclass_from_dict(
        SuperMimicTrainingRunConfig, document["resolved_config"], strict=True
    )
    directory = Path(dataset_directory).expanduser().resolve()
    digest = verify_corpus(directory, config.dataset.evaluation_clips)
    if digest != document["dataset_manifest_sha256"]:
        raise ValueError("evaluation corpus differs from checkpoint")
    manifest = json.loads((directory / MANIFEST).read_text())
    available = {Path(entry["file"]).stem for entry in manifest["clips"]}
    if (
        not reference_clips
        or len(set(reference_clips)) != len(reference_clips)
        or any(name not in available for name in reference_clips)
    ):
        raise ValueError(
            "reference_clips must select unique clips from the recorded corpus"
        )
    config = replace(
        config,
        dataset=replace(config.dataset, directory=directory),
        environment=replace(
            config.environment,
            randomization=replace(config.environment.randomization, enabled=False),
            noise=replace(config.environment.noise, enabled=False),
            latency=replace(config.environment.latency, enabled=False),
            reset=replace(config.environment.reset, random_start=False),
        ),
    )
    env = load_environment(
        config,
        load_mjx=False,
        paths=tuple(directory / f"{name}.npz" for name in reference_clips),
        evaluation_names=(),
        use_viewer_model=use_viewer_model,
    )
    return document, config, env


def load_actor(path, *, dataset_directory, reference_clips, use_viewer_model=False):
    from bello_mujoco.supermimic.model import Actor, Moments

    path = Path(path)
    document, config, env = _evaluation_environment(
        path,
        dataset_directory=dataset_directory,
        reference_clips=reference_clips,
        use_viewer_model=use_viewer_model,
    )
    codec = env.codec
    template = (
        Actor(
            codec.sample_size,
            config.observations.history_frames,
            codec.goal_size,
            env.action_size,
            config=config.networks,
            key=jax.random.PRNGKey(0),
        ),
        Moments.empty(codec.sample_size),
        Moments.empty(codec.goal_size),
    )
    if file_sha256(path / "actor.eqx") != document["actor_sha256"]:
        raise ValueError("checkpoint actor checksum mismatch")
    actor, obs_norm, goal_norm = eqx.tree_deserialise_leaves(
        path / "actor.eqx", template
    )
    return env, actor, obs_norm, goal_norm, config


def load_dynamics(path, *, dataset_directory, reference_clips):
    """Load the diagnostic model snapshot without replay or optimizer allocation."""
    from bello_mujoco.supermimic.model import Moments, WorldModel

    path = Path(path)
    document, config, env = _evaluation_environment(
        path, dataset_directory=dataset_directory, reference_clips=reference_clips
    )
    if file_sha256(path / "dynamics.eqx") != document["dynamics_sha256"]:
        raise ValueError("checkpoint dynamics checksum mismatch")
    n, context_size = env.action_size, env.action_size + 5
    model = WorldModel(
        n, context_size, config=config.networks, key=jax.random.PRNGKey(0)
    )
    template = (
        model,
        Moments.empty(model.input_size),
        Moments.empty(12 + 2 * n),
    )
    model, input_norm, output_norm = eqx.tree_deserialise_leaves(
        path / "dynamics.eqx", template
    )
    return env, model, input_norm, output_norm, config
