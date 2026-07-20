#!/usr/bin/env python3
"""Run a fingerprint-pinned locomotion checkpoint with live gamepad commands."""

from __future__ import annotations

from dataclasses import dataclass, fields
import importlib.util
from pathlib import Path
import sys

import tyro

import bello_mujoco.hardware_locomotion as hardware
from bello_mujoco.rerun_eval import GamepadConfig


@dataclass(frozen=True)
class GamepadHardwareConfig:
    checkpoint_path: Path
    endpoint: str = "http://127.0.0.1:50051"
    arm: bool = False
    operator_confirmation: str = ""
    controller_index: int = 0
    deadzone: float = 0.1
    ramp_seconds: float = 10.0
    max_duration_seconds: float = 10.0
    max_initial_position_error_radians: float = 0.75
    command_validity_seconds: float = 0.08
    acknowledgement_timeout_seconds: float = 0.08
    max_loop_lag_seconds: float = 0.10


def _runtime_hardware_module():
    current_fields = {field.name for field in fields(hardware.HardwareLocomotionConfig)}
    if "gamepad" in current_fields:
        return hardware
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "bello_mujoco"
        / "hardware_locomotion.py"
    )
    spec = importlib.util.spec_from_file_location(
        "bello_gamepad_hardware_runtime", source
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load gamepad hardware runtime from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(config: GamepadHardwareConfig) -> None:
    runtime = _runtime_hardware_module()
    gamepad_config = GamepadConfig(
        enabled=True,
        controller_index=config.controller_index,
        deadzone=config.deadzone,
    )
    common = {
        "endpoint": config.endpoint,
        "command": (0.0, 0.0, 0.0),
        "arm": config.arm,
        "operator_confirmation": config.operator_confirmation,
        "ramp_seconds": config.ramp_seconds,
        "max_duration_seconds": config.max_duration_seconds,
        "max_initial_position_error_radians": (
            config.max_initial_position_error_radians
        ),
        "command_validity_seconds": config.command_validity_seconds,
        "acknowledgement_timeout_seconds": (config.acknowledgement_timeout_seconds),
        "max_loop_lag_seconds": config.max_loop_lag_seconds,
    }
    runtime.run_hardware_locomotion(
        config.checkpoint_path.expanduser(),
        runtime.HardwareLocomotionConfig(gamepad=gamepad_config, **common),
    )


if __name__ == "__main__":
    main(tyro.cli(GamepadHardwareConfig))
