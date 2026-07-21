from __future__ import annotations

from typing import TypeVar

from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from bello_mujoco.assets import project_data_path
from bello_mujoco.config_decode import dataclass_from_dict

T = TypeVar("T")

CONFIG_ROOT = project_data_path("configs")


def config_path(group: str) -> str:
    return str(CONFIG_ROOT / group)


def structured_config(cls: type[T], config: DictConfig) -> T:
    data = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if not isinstance(data, dict):
        raise TypeError(f"Hydra config for {cls.__name__} must be a mapping")
    return dataclass_from_dict(cls, data, strict=True)


def clear_hydra() -> None:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
