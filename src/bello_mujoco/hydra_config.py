from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from bello_mujoco.checkpoint import dataclass_from_dict

T = TypeVar("T")

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


def config_path(group: str) -> str:
    return str(CONFIG_ROOT / group)


def structured_config(cls: type[T], config: DictConfig) -> T:
    data = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if not isinstance(data, dict):
        raise TypeError(f"Hydra config for {cls.__name__} must be a mapping")
    return dataclass_from_dict(cls, data)


def clear_hydra() -> None:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
