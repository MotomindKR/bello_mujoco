"""Task-neutral host-side helpers for PPO training workflows."""

from __future__ import annotations

from typing import Protocol, TypeAlias
import os
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from tqdm.auto import tqdm
from bello_mujoco.training.ppo import update_runner_state
from bello_mujoco.training.ppo_config import PPOConfig
from bello_mujoco.training.ppo_types import PPOEnvironment
from bello_mujoco.training.tracking import track_metric_snapshot

from bello_mujoco.training.ppo_types import RunnerState

RunnerInitialization: TypeAlias = tuple[
    RunnerState,
    eqx.Module,
    optax.GradientTransformation,
]


class VideoSchedule(Protocol):
    enabled: bool
    every_updates: int
    render_final: bool


def training_stop_requested() -> bool:
    """The Vast supervisor requests a checkpoint before its hard deadline."""
    directory = os.environ.get("BELLO_JOB_DIR")
    return bool(directory) and (Path(directory) / "stop_requested").is_file()


def training_updates(
    env: PPOEnvironment,
    config: PPOConfig,
    initial_runner: RunnerInitialization,
    run,
    *,
    update_fn=None,
):
    """Yield synchronized, validated updates after recording training metrics."""
    runner_state, static_policy, optimizer = initial_runner
    start_update = int(jax.device_get(runner_state.update))
    if not 0 <= start_update < config.num_updates:
        raise ValueError(
            "runner update must be within the unfinished training range, got "
            f"update={start_update}, num_updates={config.num_updates}"
        )

    if update_fn is None:
        update_fn = jax.jit(
            lambda state: update_runner_state(
                state, static_policy, optimizer, env, config
            )
        )

    for _ in tqdm(
        range(start_update, config.num_updates),
        desc="PPO updates",
        unit="update",
        dynamic_ncols=True,
        initial=start_update,
        total=config.num_updates,
    ):
        runner_state, metrics = update_fn(runner_state)
        metrics = jax.block_until_ready(metrics)
        update = int(jax.device_get(runner_state.update))
        raise_on_invalid_training_metrics(metrics, update)
        env_steps = update * config.num_envs * config.num_steps
        track_metric_snapshot(
            run,
            metrics,
            env_steps=env_steps,
            update=update,
            context={"phase": "train"},
        )
        yield runner_state, metrics, env_steps
        if training_stop_requested():
            if run is not None:
                run.summary["training/stop_reason"] = "budget_deadline"
                run.summary["training/completed_target"] = update >= config.num_updates
            print(
                f"Graceful training stop after update {update}; saving final checkpoint.",
                flush=True,
            )
            return


def should_render_training_video(
    update: int,
    num_updates: int,
    config: VideoSchedule,
) -> bool:
    if not config.enabled or update <= 0 or training_stop_requested():
        return False
    if config.render_final and update == num_updates:
        return True
    return update % config.every_updates == 0


def raise_on_invalid_training_metrics(metrics: dict, update: int) -> None:
    contaminated: list[tuple[str, str]] = []
    for name, value in metrics.items():
        prefix = next(
            (
                prefix
                for prefix in ("nonfinite/", "invalid/")
                if name.startswith(prefix)
            ),
            None,
        )
        if prefix is None or value is None:
            continue
        if float(jax.device_get(value)) > 0.0:
            contaminated.append((prefix.removesuffix("/"), name.removeprefix(prefix)))
    if contaminated:
        kind, source = contaminated[0]
        raise RuntimeError(
            f"PPO encountered {kind} {source} at update {update}; "
            "the poisoned optimizer step was skipped."
        )
    invalid_update = metrics.get("invalid_update")
    if invalid_update is not None and float(jax.device_get(invalid_update)) > 0.0:
        raise RuntimeError(
            f"PPO rejected an implausible update at update {update}; "
            "the poisoned optimizer step was skipped."
        )
    nonfinite_update = metrics.get("nonfinite_update")
    if nonfinite_update is not None and float(jax.device_get(nonfinite_update)) > 0.0:
        raise RuntimeError(
            "PPO produced a nonfinite update at update "
            f"{update}; skipped the poisoned optimizer step. "
            "Lower learning_rate/init_log_std or inspect rewards/observations."
        )
    checked_keys = (
        "loss",
        "actor_loss",
        "value_loss",
        "value_loss_raw",
        "entropy",
        "approx_kl",
        "policy_clip_fraction",
        "action_std_mean",
        "action_std_min",
        "action_std_max",
    )
    for name in checked_keys:
        value = metrics.get(name)
        if value is None:
            continue
        if not bool(jnp.all(jnp.isfinite(jax.device_get(value)))):
            raise RuntimeError(
                f"PPO metric {name!r} became nonfinite at update {update}; "
                "stopping training instead of continuing with invalid policy state."
            )
