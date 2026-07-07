"""Host-side experiment tracking helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np


@dataclass(frozen=True)
class AimConfig:
    """Aim run configuration for command-line training."""

    enabled: bool = False
    repo: str = ".aim"
    experiment: str = "bello-ppo"
    run_name: str | None = None
    log_system_params: bool = True
    capture_terminal_logs: bool = False


def create_aim_run(config: AimConfig):
    if not config.enabled:
        return None
    from aim import Run

    run = Run(
        repo=config.repo,
        experiment=config.experiment,
        log_system_params=config.log_system_params,
        capture_terminal_logs=config.capture_terminal_logs,
    )
    if config.run_name is not None:
        run.name = config.run_name
    return run


def log_config(run, config: Mapping[str, Any]) -> None:
    if run is None:
        return
    for key, value in config.items():
        run.set(key, to_builtin(value), strict=False)


def track_metric_history(
    run,
    metrics: Mapping[str, Any],
    *,
    samples_per_update: int,
    context: Mapping[str, Any] | None = None,
) -> None:
    if run is None:
        return
    host_metrics = jax.device_get(metrics)
    update_count = metric_update_count(host_metrics)
    for update in range(update_count):
        env_steps = (update + 1) * samples_per_update
        run.track(update + 1, name="update", step=env_steps, context=context)
        for name, values in host_metrics.items():
            array = np.asarray(values)
            value = array if array.ndim == 0 else array[update]
            run.track(float(value), name=name, step=env_steps, context=context)


def metric_update_count(metrics: Mapping[str, Any]) -> int:
    for values in metrics.values():
        array = np.asarray(values)
        if array.ndim > 0:
            return int(array.shape[0])
    return 1


def close_run(run) -> None:
    if run is not None:
        run.close()


def to_builtin(value: Any) -> Any:
    if is_dataclass(value):
        return to_builtin(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value
