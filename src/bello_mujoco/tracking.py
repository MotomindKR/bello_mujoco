"""Host-side experiment tracking helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any
from typing import Literal

import jax
import numpy as np


@dataclass(frozen=True)
class WandbConfig:
    """Weights & Biases run configuration for command-line training."""

    enabled: bool = False
    project: str = "bello-mujoco"
    entity: str | None = None
    group: str | None = None
    job_type: str = "train"
    run_name: str | None = None
    mode: Literal["online", "offline", "disabled"] = "online"
    dir: str = ".wandb"
    tags: tuple[str, ...] = ()


def create_wandb_run(config: WandbConfig):
    if not config.enabled:
        return None
    import wandb

    Path(config.dir).mkdir(parents=True, exist_ok=True)
    return wandb.init(
        project=config.project,
        entity=config.entity,
        group=config.group,
        job_type=config.job_type,
        name=config.run_name,
        mode=config.mode,
        dir=config.dir,
        tags=list(config.tags),
    )


def log_config(run, config: Mapping[str, Any]) -> None:
    if run is None:
        return
    run.config.update(to_builtin(config), allow_val_change=True)


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
        payload: dict[str, float | int] = {"update": update + 1}
        for name, values in host_metrics.items():
            array = np.asarray(values)
            value = array if array.ndim == 0 else array[update]
            payload[name] = float(value)
        if context:
            prefix = "/".join(str(value) for value in context.values())
            payload = {
                key if key == "update" else f"{prefix}/{key}": value
                for key, value in payload.items()
            }
        run.log(payload, step=env_steps)


def track_video(
    run,
    path: Path,
    *,
    name: str,
    step: int,
    fps: int = 30,
    caption: str | None = None,
) -> None:
    if run is None:
        return
    import wandb

    run.log(
        {name: wandb.Video(str(path), fps=fps, format=path.suffix.lstrip("."), caption=caption)},
        step=step,
    )


def metric_update_count(metrics: Mapping[str, Any]) -> int:
    for values in metrics.values():
        array = np.asarray(values)
        if array.ndim > 0:
            return int(array.shape[0])
    return 1


def close_run(run) -> None:
    if run is not None:
        run.finish()


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
