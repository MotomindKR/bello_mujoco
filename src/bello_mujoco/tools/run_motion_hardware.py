from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

from dataclasses import dataclass
from pathlib import Path
import sys

import hydra
from omegaconf import DictConfig

from bello_mujoco.hardware_motion import HardwareMotionConfig, run_hardware_motion
from bello_mujoco.hydra_config import config_path, structured_config


@dataclass(frozen=True)
class HardwareMotionToolConfig:
    checkpoint_path: Path | None = None
    hardware: HardwareMotionConfig = HardwareMotionConfig()


def main(
    checkpoint_path: Path,
    hardware: HardwareMotionConfig = HardwareMotionConfig(),
) -> None:
    run_hardware_motion(checkpoint_path.expanduser(), hardware)


def entrypoint() -> None:
    sys.argv[1:] = [expand_path_override(argument) for argument in sys.argv[1:]]
    hydra_entrypoint()


def expand_path_override(argument: str) -> str:
    prefixes = ("checkpoint_path=", "hardware.motion_path=")
    for prefix in prefixes:
        if argument.startswith(prefix):
            value = argument.removeprefix(prefix)
            if value.startswith("~"):
                return prefix + str(Path(value).expanduser())
    return argument


@hydra.main(
    version_base=None,
    config_path=config_path("run_motion_hardware"),
    config_name="default",
)
def hydra_entrypoint(config: DictConfig) -> None:
    parsed = structured_config(HardwareMotionToolConfig, config)
    if parsed.checkpoint_path is None:
        raise ValueError(
            "checkpoint_path must be set, e.g. "
            "checkpoint_path=checkpoints/asap_balance/final"
        )
    main(parsed.checkpoint_path, parsed.hardware)


if __name__ == "__main__":
    entrypoint()
