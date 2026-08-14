from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import subprocess
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from bello_mujoco.cli import (
    EmulatorCommandConfig,
    EvalCommandConfig,
    HardwareCommandConfig,
    MotionCommandConfig,
    PrepareAssetsCommandConfig,
    TrainCommandConfig,
    VastTrainCommandConfig,
    _reject_explicit_fields,
)
from bello_mujoco.configuration import (
    CONFIG_ROOT,
    command_config,
    coerce_value,
    locomotion_training_profile,
    motion_training_profile,
    structured_config,
)

COMMAND_SCHEMAS = {
    "prepare_assets": PrepareAssetsCommandConfig,
    "train": TrainCommandConfig,
    "eval": EvalCommandConfig,
    "motion": MotionCommandConfig,
    "vast_train": VastTrainCommandConfig,
    "emulator": EmulatorCommandConfig,
    "hardware": HardwareCommandConfig,
}


def test_importing_command_surface_does_not_import_jax() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import bello_mujoco.cli; "
                "assert 'jax' not in sys.modules; "
                "assert 'bello_mujoco.evaluation.locomotion' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_hardware_entrypoint_forces_cpu_backend() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os, sys; "
                "os.environ['JAX_PLATFORMS'] = 'gpu'; "
                "os.environ['JAX_PLATFORM_NAME'] = 'gpu'; "
                "sys.argv = ['bello-run-hardware']; "
                "import bello_mujoco.cli as cli; "
                "cli._hardware_hydra = lambda: None; "
                "cli.hardware_entrypoint(); "
                "import jax; "
                "assert os.environ['JAX_PLATFORMS'] == 'cpu'; "
                "assert os.environ['JAX_PLATFORM_NAME'] == 'cpu'; "
                "assert jax.default_backend() == 'cpu'"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(("name", "schema"), COMMAND_SCHEMAS.items())
def test_command_yaml_is_complete(name: str, schema: type) -> None:
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_ROOT / "commands"),
    ):
        config = compose(config_name=name)

    command_config(schema, config)
    raw = OmegaConf.to_container(config, resolve=True)
    assert isinstance(raw, dict)
    for field in fields(schema):
        incomplete = dict(raw)
        incomplete.pop(field.name)
        with pytest.raises(ValueError, match=field.name):
            command_config(schema, OmegaConf.create(incomplete))


@pytest.mark.parametrize(
    "value",
    (
        "can0=vcan0",
        {"can0": "vcan0"},
        1,
    ),
)
def test_tuple_config_rejects_non_sequences(value: object) -> None:
    with pytest.raises(TypeError, match="bus_interfaces must be a list or tuple"):
        coerce_value(tuple[str, ...], value, path="bus_interfaces")


def test_mapping_config_rejects_non_mapping() -> None:
    with pytest.raises(TypeError, match="joints must be a mapping"):
        coerce_value(dict[str, int], [("left_hip", 1)], path="joints")


def test_irrelevant_explicit_override_is_rejected_even_at_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["bello-eval", "task=motion", "command=[0.0,0.0,0.0]"],
    )

    with pytest.raises(ValueError, match="task=motion does not accept: command"):
        _reject_explicit_fields(("command",), context="task=motion")


def test_supported_training_profiles_load_and_smoke_is_local() -> None:
    smoke = locomotion_training_profile("smoke")
    locomotion = locomotion_training_profile("locomotion")
    hardware = locomotion_training_profile("hardware_locomotion")
    motion = motion_training_profile()

    assert all(
        profile.config.num_updates > 0
        for profile in (smoke, locomotion, hardware, motion)
    )
    assert smoke.config.num_envs == 2
    assert smoke.config.num_updates == 1
    assert smoke.checkpoint_config.enabled is False
    assert smoke.checkpoint_config.allow_git_commit_mismatch is False
    assert smoke.wandb_config.enabled is False
    assert smoke.video_config.enabled is False
    assert locomotion.checkpoint_config.every_updates == 512
    assert hardware.checkpoint_config.every_updates == 512
    assert hardware.config.total_timesteps == 8_000_000_000
    assert hardware.video_config.command == (0.0, 0.5, 0.0)
    assert locomotion.env_config.observation_latency.enabled
    assert locomotion.env_config.observation_latency.delay_steps == (0, 2)
    assert hardware.env_config.observation_latency.enabled
    assert hardware.env_config.observation_latency.delay_steps == (0, 2)
    assert motion.checkpoint_config.every_updates == 500


def test_motion_path_override_is_explicit(tmp_path: Path) -> None:
    path = tmp_path / "turn.npz"
    profile = motion_training_profile(path)

    assert profile.motion.directory == tmp_path
    assert profile.motion.sequence == "turn"


def test_public_command_configs_are_strict() -> None:
    for raw, rejected in (
        ({"experiment": "smoke", "typo": True}, "typo"),
        ({"experiment": "invalid"}, "experiment"),
        ({"experiment": None}, "experiment"),
    ):
        try:
            structured_config(TrainCommandConfig, OmegaConf.create(raw))
        except (TypeError, ValueError) as error:
            assert rejected in str(error)
        else:
            raise AssertionError(f"invalid public command config was accepted: {raw}")

    for config_type, raw in (
        (HardwareCommandConfig, {"arm": 1}),
        (HardwareCommandConfig, {"duration_seconds": True}),
        (HardwareCommandConfig, {"remote_staleness_ms": True}),
        (EvalCommandConfig, {"allow_git_commit_mismatch": 1}),
        (TrainCommandConfig, {"allow_git_commit_mismatch": 1}),
    ):
        try:
            structured_config(config_type, OmegaConf.create(raw))
        except TypeError:
            pass
        else:
            raise AssertionError(f"invalid public command type was accepted: {raw}")
