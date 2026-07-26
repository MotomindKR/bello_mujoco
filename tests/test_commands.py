from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from omegaconf import OmegaConf

from bello_mujoco.assets_pipeline.prepare import canonical_physical_asset_config
from bello_mujoco.cli import (
    HardwareCommandConfig,
    TrainCommandConfig,
)
from bello_mujoco.configuration import (
    config_path,
    locomotion_training_profile,
    motion_training_profile,
    structured_config,
)


def test_supported_training_profiles_preserve_canonical_contracts() -> None:
    smoke = locomotion_training_profile("smoke")
    locomotion = locomotion_training_profile("locomotion")
    hardware = locomotion_training_profile("hardware_locomotion")

    assert smoke.config.num_envs == 2
    assert smoke.wandb_config.enabled is False
    assert locomotion.config.policy.hidden_dims == (512, 256, 128)
    assert locomotion.reward_config.weights.torque_l1 == -1.2
    assert locomotion.reward_config.weights.torque_peak == -12.0
    assert locomotion.reward_config.weights.stand_foot_motion == -1.0
    assert hardware.asset_config.target_total_mass_kg == 60.0
    assert hardware.asset_config.joint_availability.disabled_joint_names == {
        "waist_roll_joint",
        "waist_pitch_joint",
        "neck_yaw_joint",
        "head_pitch_joint",
    }
    assert hardware.config.gamma == 0.995
    assert hardware.config.gae_lambda == 0.98


def test_motion_profile_is_bounded_beyondmimic() -> None:
    profile = motion_training_profile()

    assert profile.env_config.formulation == "beyondmimic_joint_limits"
    assert profile.env_config.action_clip == 1.0
    assert profile.config.policy.action_distribution == "tanh_gaussian"
    assert profile.asset_config.target_total_mass_kg == 60.0


def test_motion_path_override_is_explicit(tmp_path: Path) -> None:
    path = tmp_path / "turn.npz"
    profile = motion_training_profile(path)

    assert profile.motion.directory == tmp_path
    assert profile.motion.sequence == "turn"


def test_physical_profile_is_single_60kg_sphere_hand_morphology() -> None:
    profile = canonical_physical_asset_config(imu_enabled=True)

    assert profile.asset == "full_body"
    assert profile.collision_mode == "box"
    assert profile.end_effector_sphere_mode == "both"
    assert profile.target_total_mass_kg == 60.0
    assert profile.imu_config.enabled
    assert profile.joint_availability.waist_yaw
    assert len(profile.joint_availability.disabled_joint_names) == 4


def test_public_command_configs_reject_unknown_keys() -> None:
    raw = OmegaConf.create({"experiment": "smoke", "legacy_flag": True})

    try:
        structured_config(TrainCommandConfig, raw)
    except ValueError as error:
        assert "legacy_flag" in str(error)
    else:
        raise AssertionError("unknown public command key was accepted")


def test_hardware_command_surface_contains_only_operator_essentials() -> None:
    config = HardwareCommandConfig()
    indefinite = replace(config, duration_seconds=None)

    assert indefinite.duration_seconds is None
    assert not hasattr(config, "max_target_step_radians")
    assert not hasattr(config, "max_tilt_radians")
    assert not hasattr(config, "max_loop_lag_seconds")


def test_all_public_hydra_documents_are_bundled() -> None:
    root = Path(config_path("commands"))
    assert {path.stem for path in root.glob("*.yaml")} == {
        "prepare_assets",
        "train",
        "eval",
        "motion",
        "vast_train",
        "robstride",
        "hardware",
    }
