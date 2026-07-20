from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

from dataclasses import dataclass
from pathlib import Path
import sys

import hydra
from omegaconf import DictConfig

from bello_mujoco.hardware_locomotion import (
    HardwareLocomotionConfig,
    run_hardware_locomotion,
)
from bello_mujoco.hydra_config import config_path, structured_config


@dataclass(frozen=True)
class HardwareLocomotionToolConfig:
    checkpoint_path: Path | None = None
    hardware: HardwareLocomotionConfig = HardwareLocomotionConfig()


def main(
    checkpoint_path: Path,
    hardware: HardwareLocomotionConfig = HardwareLocomotionConfig(),
) -> None:
    run_hardware_locomotion(checkpoint_path.expanduser(), hardware)


def entrypoint() -> None:
    sys.argv[1:] = [expand_checkpoint_override(argument) for argument in sys.argv[1:]]
    hydra_entrypoint()


def expand_checkpoint_override(argument: str) -> str:
    prefix = "checkpoint_path="
    if not argument.startswith(prefix):
        return argument
    value = argument.removeprefix(prefix)
    if not value.startswith("~"):
        return argument
    return prefix + str(Path(value).expanduser())


@hydra.main(
    version_base=None,
    config_path=config_path("run_locomotion_hardware"),
    config_name="default",
)
def hydra_entrypoint(config: DictConfig) -> None:
    parsed = structured_config(HardwareLocomotionToolConfig, config)
    if parsed.checkpoint_path is None:
        raise ValueError(
            "checkpoint_path must be set, e.g. "
            "checkpoint_path=checkpoints/hardware_locomotion/final"
        )
    main(parsed.checkpoint_path, parsed.hardware)


if __name__ == "__main__":
    entrypoint()
