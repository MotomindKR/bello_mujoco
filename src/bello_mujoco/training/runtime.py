"""Task-neutral host-side helpers for PPO training workflows."""

from __future__ import annotations

from typing import Protocol, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from tqdm.auto import tqdm

from bello_mujoco.training.ppo import RunnerState

RunnerInitialization: TypeAlias = tuple[
    RunnerState,
    eqx.Module,
    optax.GradientTransformation,
]


class VideoSchedule(Protocol):
    enabled: bool
    every_updates: int
    render_final: bool


def training_update_range(start_update: int, num_updates: int):
    return tqdm(
        range(start_update, num_updates),
        desc="PPO updates",
        unit="update",
        dynamic_ncols=True,
        initial=start_update,
        total=num_updates,
    )


def should_render_training_video(
    update: int,
    num_updates: int,
    config: VideoSchedule,
) -> bool:
    if not config.enabled or update <= 0:
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
