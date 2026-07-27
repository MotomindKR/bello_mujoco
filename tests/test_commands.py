from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from bello_mujoco.cli import HardwareCommandConfig, TrainCommandConfig
from bello_mujoco.configuration import (
    locomotion_training_profile,
    motion_training_profile,
    structured_config,
)


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
    assert smoke.wandb_config.enabled is False
    assert smoke.video_config.enabled is False


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

    for raw in ({"arm": 1}, {"duration_seconds": True}):
        try:
            structured_config(HardwareCommandConfig, OmegaConf.create(raw))
        except TypeError:
            pass
        else:
            raise AssertionError(f"invalid public command type was accepted: {raw}")
