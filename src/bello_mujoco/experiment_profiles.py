"""Canonical training profiles exposed by the public CLI."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal

from omegaconf import DictConfig, OmegaConf

from bello_mujoco.assets import project_data_path
from bello_mujoco.hydra_config import structured_config
from bello_mujoco.tools.train_motion_ppo import (
    MotionSelectionConfig,
    MotionTrainingRunConfig,
)
from bello_mujoco.tools.train_ppo import TrainingRunConfig

ExperimentName = Literal["smoke", "locomotion", "hardware_locomotion", "motion"]


def locomotion_training_profile(name: ExperimentName) -> TrainingRunConfig:
    if name not in {"smoke", "locomotion", "hardware_locomotion"}:
        raise ValueError(f"{name!r} is not a locomotion experiment")
    paths = [_config("train/default.yaml")]
    if name in {"locomotion", "hardware_locomotion"}:
        paths.append(_config("profiles/train/locomotion.yaml"))
    if name == "hardware_locomotion":
        paths.append(_config("profiles/train/hardware_locomotion.yaml"))
    if name == "smoke":
        paths.append(_config("profiles/train/smoke.yaml"))
    return structured_config(TrainingRunConfig, _merged_profile(paths))


def motion_training_profile(
    motion_path: Path | None = None,
) -> MotionTrainingRunConfig:
    config = structured_config(
        MotionTrainingRunConfig,
        _merged_profile(
            (
                _config("train_motion/default.yaml"),
                _config("profiles/train/motion.yaml"),
            )
        ),
    )
    if motion_path is None:
        return config
    resolved = motion_path.expanduser().resolve()
    return replace(
        config,
        motion=MotionSelectionConfig(
            directory=resolved.parent,
            sequence=resolved.stem,
        ),
    )


def profile_config_path(name: ExperimentName) -> Path:
    mapping = {
        "smoke": _config("profiles/train/smoke.yaml"),
        "locomotion": _config("profiles/train/locomotion.yaml"),
        "hardware_locomotion": _config("profiles/train/hardware_locomotion.yaml"),
        "motion": _config("profiles/train/motion.yaml"),
    }
    return mapping[name]


def _config(relative_path: str) -> Path:
    return project_data_path("configs") / relative_path


def _merged_profile(paths: tuple[Path, ...] | list[Path]) -> DictConfig:
    configs = []
    for path in paths:
        config = OmegaConf.load(path)
        if not isinstance(config, DictConfig):
            raise TypeError(f"training profile {path} must be a mapping")
        for internal_key in ("defaults", "hydra"):
            if internal_key in config:
                del config[internal_key]
        configs.append(config)
    merged = OmegaConf.merge(*configs)
    OmegaConf.resolve(merged)
    return merged
