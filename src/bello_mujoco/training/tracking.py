"""Host-side experiment tracking helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from typing import Literal

import jax
import numpy as np

if TYPE_CHECKING:
    from wandb import Artifact


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
        settings=wandb.Settings(start_method="thread"),
    )


def log_config(run, config: Mapping[str, Any]) -> None:
    if run is None:
        return
    run.config.update(to_builtin(config), allow_val_change=True)


def track_metric_snapshot(
    run,
    metrics: Mapping[str, Any],
    *,
    env_steps: int,
    update: int,
    context: Mapping[str, Any] | None = None,
) -> None:
    if run is None:
        return
    payload = metric_payload(
        jax.device_get(metrics),
        update=update,
        context=context,
    )
    run.log(payload, step=env_steps)


def metric_payload(
    metrics: Mapping[str, Any],
    *,
    update: int,
    context: Mapping[str, Any] | None = None,
) -> dict[str, float | int]:
    payload: dict[str, float | int] = {"update": update}
    for name, value in metrics.items():
        payload[name] = float(np.asarray(value))
    if context:
        prefix = "/".join(str(value) for value in context.values())
        payload = {
            key if key == "update" else f"{prefix}/{key}": value
            for key, value in payload.items()
        }
    return payload


def track_video(
    run,
    path: Path,
    *,
    name: str,
    step: int,
    caption: str | None = None,
) -> None:
    if run is None:
        return
    import wandb

    run.log(
        {name: wandb.Video(str(path), format=path.suffix.lstrip("."), caption=caption)},
        step=step,
    )


def track_checkpoint_artifact(
    run,
    path: Path,
    *,
    name: str,
    artifact_type: str = "checkpoint",
) -> Artifact | None:
    if run is None:
        return
    import wandb

    artifact = wandb.Artifact(name=name, type=artifact_type)
    artifact.add_dir(str(path))
    return run.log_artifact(artifact)


def close_run(run, *, exit_code: int = 0) -> None:
    if run is not None:
        run.finish(exit_code=exit_code)


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
