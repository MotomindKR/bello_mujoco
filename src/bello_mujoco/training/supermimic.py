"""Alternate Mini simulation, learned dynamics fitting, and analytic policy updates."""

from dataclasses import dataclass, field, replace
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from bello_mujoco.envs.supermimic import BelloSuperMimicEnv
from bello_mujoco.models.compiler import ModelConfig
from bello_mujoco.repository import (
    capture_repository_commit,
    require_checkpoint_output_directory,
)
from bello_mujoco.sim import WarpConfig
from bello_mujoco.sonic.corpus import MANIFEST, verify_artifact, verify_corpus
from bello_mujoco.supermimic.collection import collect_step, initialize_collection
from bello_mujoco.supermimic.config import (
    EnvironmentConfig,
    LearningConfig,
    ObservationConfig,
)
from bello_mujoco.supermimic.mdp import TrackingConfig
from bello_mujoco.supermimic.model import NetworkConfig
from bello_mujoco.supermimic.optimization import (
    initialize_learner,
    update_actor_critic,
    update_world_model,
)
from bello_mujoco.supermimic.sampling import SamplingConfig
from bello_mujoco.training.checkpoint import (
    CheckpointConfig,
    checkpoint_path,
    validate_checkpoint_options,
)
from bello_mujoco.training.tracking import (
    WandbConfig,
    close_run,
    create_wandb_run,
    log_config,
    track_checkpoint_artifact,
    track_metric_snapshot,
)


@dataclass(frozen=True)
class DatasetConfig:
    directory: Path
    evaluation_clips: tuple[str, ...] = ()
    artifact: str | None = None
    max_array_bytes: int = 4 * 1024**3


@dataclass(frozen=True)
class SuperMimicTrainingRunConfig:
    asset_config: ModelConfig
    dataset: DatasetConfig
    max_training_seconds: float | None = None
    config: LearningConfig = field(default_factory=LearningConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    observations: ObservationConfig = field(default_factory=ObservationConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    networks: NetworkConfig = field(default_factory=NetworkConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    warp: WarpConfig = field(default_factory=WarpConfig)
    checkpoint_config: CheckpointConfig = field(default_factory=CheckpointConfig)
    wandb_config: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self):
        if self.max_training_seconds is not None and (
            not math.isfinite(self.max_training_seconds)
            or self.max_training_seconds <= 0
        ):
            raise ValueError("max_training_seconds must be positive and finite")
        maximum_history = max(
            self.observations.history_frames,
            self.environment.latency.delay_steps[1] + 1,
        )
        if self.config.replay_steps < maximum_history:
            raise ValueError(
                "replay_steps cannot reconstruct observation and feedback history"
            )
        if self.config.total_timesteps // self.config.num_envs > 2**31 - 1:
            raise ValueError("vector-step counter exceeds int32 range")
        if self.dataset.max_array_bytes <= 0:
            raise ValueError("dataset.max_array_bytes must be positive")
        validate_checkpoint_options(self.checkpoint_config)
        interval = self.checkpoint_config.every_updates
        if interval is not None and (type(interval) is not int or interval <= 0):
            raise ValueError(
                "checkpoint every_updates must be a positive integer or null"
            )


def load_environment(
    run_config, *, load_mjx, paths=None, evaluation_names=None, use_viewer_model=False
):
    if paths is None:
        paths = tuple(sorted(run_config.dataset.directory.glob("*.npz")))
    return BelloSuperMimicEnv.load_corpus(
        run_config.asset_config,
        run_config.environment.motion_config(),
        paths,
        evaluation_names=run_config.dataset.evaluation_clips
        if evaluation_names is None
        else evaluation_names,
        observation_config=run_config.observations,
        tracking_config=run_config.tracking,
        sampling_config=run_config.sampling,
        warp=run_config.warp,
        batch_size=run_config.config.num_envs if load_mjx else 1,
        load_mjx=load_mjx,
        use_viewer_model=use_viewer_model,
    )


def compile_updates(env, config):
    def collect(runner, bank):
        return collect_step(runner, replace(env, reference_bank=bank), config)

    def fit_model(runner):
        def batch(carry, _):
            learner, key = carry
            key, sample_key = jax.random.split(key)
            window = runner.replay.sample_windows(
                sample_key, config.model_batch_size, config.window_length
            )
            learner, metrics = update_world_model(
                learner, window, config, runner.update
            )
            return (learner, key), metrics

        (learner, key), metrics = jax.lax.scan(
            batch, (runner.learner, runner.key), None, length=config.model_batches
        )
        return replace(runner, learner=learner, key=key), jax.tree.map(
            jnp.mean, metrics
        )

    def improve(runner, bank):
        key, start_key, update_key = jax.random.split(runner.key, 3)
        starts = runner.replay.sample_starts(
            start_key,
            config.imagined_batch_size,
            history_frames=env.codec.config.history_frames,
            feedback_frames=env.maximum_observation_delay_steps + 1,
            horizon=config.imagination_horizon,
        )
        dynamic_env = replace(env, reference_bank=bank)

        def update():
            learner, metrics = update_actor_critic(
                runner.learner, starts, dynamic_env, config, runner.update, update_key
            )
            return learner, {**metrics, "imagination/skipped": jnp.asarray(0.0)}

        def skip():
            metrics = {
                name: jnp.asarray(0.0)
                for name in (
                    "actor/loss",
                    "critic/loss",
                    "actor/gradient_norm",
                    "imagination/reward",
                    "invalid/actor_critic_update",
                )
            }
            return runner.learner, {**metrics, "imagination/skipped": jnp.asarray(1.0)}

        learner, metrics = jax.lax.cond(starts.valid, update, skip)
        return replace(
            runner,
            learner=learner,
            key=key,
            actor_updates=runner.actor_updates
            + (1.0 - metrics["imagination/skipped"]).astype(jnp.int32),
        ), metrics

    return jax.jit(collect), jax.jit(fit_model), jax.jit(improve)


def require_valid_metrics(metrics):
    for name, value in jax.device_get(metrics).items():
        scalar = float(value)
        if not math.isfinite(scalar) or (name.startswith("invalid/") and scalar > 0):
            raise RuntimeError(
                f"SuperMimic rejected {name}={scalar}; no further optimization was run"
            )


def run_training(run_config):
    from bello_mujoco.supermimic.checkpoint import restore_checkpoint, save_checkpoint

    started = time.monotonic()
    commit = capture_repository_commit()
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError(
            "SuperMimic collection requires CUDA/MJX-Warp; use nix develop .#training on a Linux CUDA host"
        )
    cfg = run_config.config
    available = cfg.total_timesteps // cfg.num_envs - cfg.warmup_steps
    if math.ceil(available * cfg.update_to_data_ratio) <= cfg.model_warmup_updates:
        raise ValueError("training budget must extend beyond model_warmup_updates")
    directory = run_config.dataset.directory.expanduser().resolve()
    run_config = replace(
        run_config,
        dataset=replace(run_config.dataset, directory=directory),
        checkpoint_config=replace(
            run_config.checkpoint_config,
            directory=require_checkpoint_output_directory(
                run_config.checkpoint_config.directory
            ),
        ),
    )
    digest = verify_corpus(directory, run_config.dataset.evaluation_clips)
    document = json.loads((directory / MANIFEST).read_text())
    array_bytes = sum(entry["array_bytes"] for entry in document["clips"])
    if array_bytes > run_config.dataset.max_array_bytes:
        raise ValueError(
            f"corpus needs {array_bytes} bytes; exceeds dataset.max_array_bytes"
        )
    run = None
    succeeded = False
    try:
        run = create_wandb_run(run_config.wandb_config)
        if run_config.dataset.artifact is not None:
            verified = verify_artifact(
                directory,
                run_config.dataset.artifact,
                run_config.dataset.evaluation_clips,
                run=run,
            )
            if digest != verified:
                raise ValueError(
                    "dataset manifest changed during artifact verification"
                )
        env = load_environment(run_config, load_mjx=True)
        bank = env.reference_bank
        train_lengths = jnp.where(bank.split_ids == 0, bank.lengths - 1, 0)
        if (
            float(jnp.max(train_lengths))
            < cfg.imagination_horizon * bank.fps * env.control_timestep
        ):
            raise ValueError("training clips are too short for imagination_horizon")
        init_key, collect_key = jax.random.split(jax.random.PRNGKey(cfg.seed))
        learner = initialize_learner(
            env, cfg, run_config.networks, env.action_size + 5, init_key
        )
        runner = initialize_collection(env, cfg, learner, collect_key)
        if run_config.checkpoint_config.resume_path is not None:
            runner = restore_checkpoint(
                run_config.checkpoint_config.resume_path,
                runner,
                run_config,
                git_commit=commit,
                dataset_digest=digest,
            )
        collect, fit, improve = compile_updates(env, cfg)
        log_config(
            run,
            {
                "task": "supermimic",
                "config": run_config,
                "git_commit": commit,
                "dataset_manifest_sha256": digest,
                "corpus_array_bytes": array_bytes,
            },
        )
        root = checkpoint_path(run_config.checkpoint_config, cfg)
        remaining = cfg.total_timesteps // cfg.num_envs
        vector_steps = int(jax.device_get(runner.collector.vector_steps))
        while vector_steps < cfg.warmup_steps:
            runner, metrics = collect(runner, bank)
            require_valid_metrics(metrics)
            vector_steps += 1
        while vector_steps < remaining:
            update = int(jax.device_get(runner.update))
            if (
                run_config.max_training_seconds is not None
                and time.monotonic() - started >= run_config.max_training_seconds
            ):
                print(
                    json.dumps(
                        {
                            "training_stop": "wall_time_budget",
                            "update": update,
                            "env_steps": vector_steps * cfg.num_envs,
                        }
                    ),
                    flush=True,
                )
                break
            before = math.floor(update / cfg.update_to_data_ratio)
            after = math.floor((update + 1) / cfg.update_to_data_ratio)
            metrics = {}
            for _ in range(min(after - before, remaining - vector_steps)):
                runner, metrics = collect(runner, bank)
                require_valid_metrics(metrics)
                vector_steps += 1
            runner, model_metrics = fit(runner)
            require_valid_metrics(model_metrics)
            metrics = {**metrics, **model_metrics}
            if update >= cfg.model_warmup_updates:
                runner, actor_metrics = improve(runner, bank)
                require_valid_metrics(actor_metrics)
                metrics.update(actor_metrics)
            runner = replace(runner, update=runner.update + 1)
            update += 1
            metrics = jax.block_until_ready(metrics)
            track_metric_snapshot(
                run,
                metrics,
                env_steps=vector_steps * cfg.num_envs,
                update=update,
                context={"phase": "train"},
            )
            if update == 1 or update % 50 == 0:
                print(
                    json.dumps(
                        {
                            "update": update,
                            "env_steps": vector_steps * cfg.num_envs,
                            **{name: float(value) for name, value in metrics.items()},
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            checkpoint = run_config.checkpoint_config
            if (
                checkpoint.enabled
                and checkpoint.every_updates
                and update % checkpoint.every_updates == 0
            ):
                path = root / f"update_{update:06d}"
                save_checkpoint(
                    path, runner, run_config, git_commit=commit, dataset_digest=digest
                )
                if checkpoint.upload_wandb_artifact:
                    track_checkpoint_artifact(
                        run,
                        path,
                        name=checkpoint.wandb_artifact_name or root.name,
                        artifact_type=checkpoint.wandb_artifact_type,
                    )
        if int(jax.device_get(runner.actor_updates)) == 0:
            raise RuntimeError(
                "training budget completed without an actor update; increase the budget or reduce model_warmup_updates"
            )
        if run_config.checkpoint_config.enabled:
            final = root / "final"
            save_checkpoint(
                final, runner, run_config, git_commit=commit, dataset_digest=digest
            )
            if run_config.checkpoint_config.upload_wandb_artifact:
                track_checkpoint_artifact(
                    run,
                    final,
                    name=run_config.checkpoint_config.wandb_artifact_name or root.name,
                    artifact_type=run_config.checkpoint_config.wandb_artifact_type,
                )
        succeeded = True
    finally:
        close_run(run, exit_code=0 if succeeded else 1)
