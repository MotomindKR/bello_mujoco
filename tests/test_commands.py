from __future__ import annotations

from dataclasses import fields, replace
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
    TwistDataCommandConfig,
    VastTrainCommandConfig,
    _hardware_policy_log_path,
    _reject_explicit_fields,
    _warp_overrides,
)
from bello_mujoco.configuration import (
    CONFIG_ROOT,
    command_config,
    coerce_value,
    locomotion_training_profile,
    motion_training_profile,
    profile_config_relative_path,
    structured_config,
    twist2_training_profile,
    twist_training_profile,
)

COMMAND_SCHEMAS = {
    "prepare_assets": PrepareAssetsCommandConfig,
    "train": TrainCommandConfig,
    "eval": EvalCommandConfig,
    "motion": MotionCommandConfig,
    "twist_data": TwistDataCommandConfig,
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


def test_training_entrypoint_uses_memory_sharing_jax_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bello_mujoco.cli as cli

    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    monkeypatch.delenv("TF_GPU_ALLOCATOR", raising=False)
    monkeypatch.setattr(cli, "_train_hydra", lambda: None)

    cli.train_entrypoint()

    assert cli.os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert cli.os.environ["TF_GPU_ALLOCATOR"] == "cuda_malloc_async"


def test_training_entrypoint_preserves_explicit_allocator_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bello_mujoco.cli as cli

    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    monkeypatch.setenv("TF_GPU_ALLOCATOR", "platform")
    monkeypatch.setattr(cli, "_train_hydra", lambda: None)

    cli.train_entrypoint()

    assert cli.os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"
    assert cli.os.environ["TF_GPU_ALLOCATOR"] == "platform"


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


def test_hardware_elbow_yaw_offsets_are_flat_overrides() -> None:
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_ROOT / "commands"),
    ):
        config = compose(
            config_name="hardware",
            overrides=(
                "left_elbow_yaw_offset_radians=1.57079632679",
                "right_elbow_yaw_offset_radians=-1.57079632679",
            ),
        )

    command = command_config(HardwareCommandConfig, config)
    assert command.left_elbow_yaw_offset_radians == pytest.approx(1.57079632679)
    assert command.right_elbow_yaw_offset_radians == pytest.approx(-1.57079632679)


def test_hardware_upper_body_tracking_is_an_explicit_boolean() -> None:
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_ROOT / "commands"),
    ):
        default = command_config(
            HardwareCommandConfig,
            compose(config_name="hardware"),
        )
        disabled = command_config(
            HardwareCommandConfig,
            compose(
                config_name="hardware",
                overrides=("upper_body_tracking_enabled=false",),
            ),
        )

    assert default.upper_body_tracking_enabled
    assert default.rtmw_endpoint == "127.0.0.1:50055"
    assert default.rtmw_transport_latency_budget_ms == pytest.approx(10.0)
    assert default.rtmw_human_height_m == pytest.approx(1.75)
    assert not disabled.upper_body_tracking_enabled


def test_hardware_policy_log_is_disabled_by_null() -> None:
    assert _hardware_policy_log_path(None) is None


def test_hardware_policy_log_expands_explicit_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    assert _hardware_policy_log_path(Path("~/policy.jsonl")) == (
        tmp_path / "policy.jsonl"
    )


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
    bello = locomotion_training_profile("bello_locomotion")
    mini = locomotion_training_profile("bello_mini_locomotion")
    motion = motion_training_profile()
    twist_teacher = twist_training_profile("teacher")
    twist_student = twist_training_profile("student")
    twist2 = twist2_training_profile()

    assert all(
        profile.config.num_updates > 0
        for profile in (
            smoke,
            bello,
            mini,
            motion,
            twist_teacher,
            twist_student,
            twist2,
        )
    )
    assert smoke.config.num_envs == 2
    assert smoke.config.num_updates == 1
    assert smoke.checkpoint_config.enabled is False
    assert smoke.checkpoint_config.allow_git_commit_mismatch is False
    assert smoke.wandb_config.enabled is False
    assert smoke.video_config.enabled is False
    assert smoke.asset_config.asset == "bello"
    assert bello.asset_config.asset == "bello"
    assert mini.asset_config.asset == "bello_mini"
    assert motion.asset_config.asset == "bello"
    assert twist_teacher.asset_config.asset == "bello"
    assert twist_student.asset_config.asset == "bello"
    assert twist2.asset_config.asset == "bello"
    assert motion.asset_config.actuator_mode == "policy"
    assert twist_teacher.asset_config.actuator_mode == "policy"
    assert twist2.asset_config.actuator_mode == "policy"
    assert bello.config == mini.config
    assert bello.config.total_timesteps == 8_000_000_000
    assert bello.config.gamma == pytest.approx(0.995)
    assert bello.config.gae_lambda == pytest.approx(0.98)
    assert bello.config.symmetry_loss_coef == pytest.approx(0.0)
    assert bello.config.learning_rate == pytest.approx(1.0e-4)
    assert bello.config.min_learning_rate == pytest.approx(3.0e-5)
    assert bello.config.adaptive_kl_statistic == "mean"
    assert bello.checkpoint_config.every_updates == 512
    assert mini.checkpoint_config.every_updates == 512
    assert not bello.asset_config.enforce_policy_effort_limits
    assert not mini.asset_config.enforce_policy_effort_limits
    assert len(bello.asset_config.actuator_config.policy_joint_names) == 13
    assert len(mini.asset_config.actuator_config.policy_joint_names) == 14
    assert len(mini.asset_config.actuator_config.hold_joint_names) == 14
    assert "waist_roll_joint" in mini.asset_config.actuator_config.policy_joint_names
    assert bello.reward_config.weights == mini.reward_config.weights
    assert bello.reward_config.tracking == mini.reward_config.tracking
    assert bello.reward_config.standing == mini.reward_config.standing
    assert bello.reward_config.gait == mini.reward_config.gait
    assert bello.reward_config.smoothness == mini.reward_config.smoothness
    assert bello.reward_config.weights.healthy_survival == pytest.approx(1.5)
    assert bello.reward_config.weights.touchdown_impact == pytest.approx(-8.0)
    assert bello.reward_config.weights.action_acceleration == pytest.approx(0.0)
    assert bello.reward_config.weights.stand_upright == pytest.approx(-8.0)
    assert bello.reward_config.standing.foot_heading_deadband == pytest.approx(
        0.0872664626
    )
    assert bello.command_config == mini.command_config
    assert bello.command_config.lin_vel_x == (-1.0, 1.0)
    assert bello.command_config.lin_vel_y == (-1.0, 1.0)
    assert bello.command_config.ang_vel_yaw == (-1.0, 1.0)
    assert bello.command_config.standing_probability == pytest.approx(0.25)
    assert bello.command_config.standing_transitions_enabled
    assert bello.env_config.random_push.enabled
    assert not bello.reward_config.gait.joint_force_apply_while_standing
    assert bello.reward_config.gait.touchdown_force_soft_limit_body_weights == (
        pytest.approx(1.0)
    )
    assert bello.reward_config.smoothness.ankle_action_weight == pytest.approx(1.0)
    assert bello.reward_config.smoothness.waist_action_weight == pytest.approx(1.0)
    assert bello.reward_config.limits_pose.hip_lateral_command_index == 0
    assert mini.reward_config.limits_pose.hip_lateral_command_index == 0
    assert bello.reward_config.links.track_lin_vel_xy_frame is None
    assert mini.reward_config.links.track_lin_vel_xy_frame is None
    for profile in (bello, mini):
        assert profile.reward_config.links.flat_orientation == "torso_link"
        assert profile.reward_config.links.base_height == "pelvis_link"
        assert profile.done_config.bad_orientation.link == "torso_link"
        assert profile.done_config.base_height.link == "pelvis_link"
    assert bello.reward_config.limits_pose.base_height_target == pytest.approx(0.86)
    assert mini.reward_config.limits_pose.base_height_target == pytest.approx(0.88)
    assert bello.done_config.base_height.min_height == pytest.approx(0.55)
    assert mini.done_config.base_height.min_height == pytest.approx(0.56)
    assert bello.env_config.gait.frequency_hz == pytest.approx(0.9)
    assert mini.env_config.gait.frequency_hz == pytest.approx(0.89)
    assert mini.env_config.spawn == replace(
        bello.env_config.spawn,
        joint_pos_scale_range=(1.0, 1.0),
    )
    assert (
        replace(
            mini.env_config,
            gait=bello.env_config.gait,
            spawn=bello.env_config.spawn,
        )
        == bello.env_config
    )
    assert bello.video_config.command == mini.video_config.command == (0.0, 0.5, 0.0)
    assert bello.env_config.observation_latency.enabled
    assert bello.env_config.observation_latency.delay_steps == (0, 2)
    assert bello.env_config.observation_low_pass_filter.enabled
    assert bello.env_config.observation_low_pass_filter.cutoff_hz == pytest.approx(10.0)
    assert bello.env_config.observation_low_pass_filter.noise_before_filter
    assert bello.env_config.observation_low_pass_filter.residual_noise_scale == (
        pytest.approx(0.1)
    )
    assert bello.env_config.torso_com_curriculum.enabled
    assert bello.env_config.torso_com_curriculum.ramp_steps == 3_000_000
    assert bello.env_config.model_randomization.torso_mass_scale_range == (
        0.8,
        1.2,
    )
    assert bello.env_config.model_randomization.torso_com_offset_range == (
        -0.1,
        0.1,
    )
    assert bello.checkpoint_config.name == "bello_locomotion_seed0"
    assert mini.checkpoint_config.name == "bello_mini_locomotion_seed0"
    assert motion.checkpoint_config.every_updates == 500
    assert twist_teacher.config.num_updates == 30_002
    assert twist_teacher.config.policy.architecture == "twist_teacher"
    assert twist_teacher.warp.contact_budget_per_env == 32
    assert twist_student.config.num_updates == 200_000
    assert twist_student.config.policy.architecture == "twist_student"
    assert twist2.config.num_updates == 30_001
    assert twist2.config.policy.architecture == "twist2"
    assert twist2.config.normalize_observations
    assert not twist2.config.normalize_critic_observations
    assert twist2.config.observation_norm_update_mode == "rollout"
    assert twist2.config.observation_norm_update_updates == 3_000
    assert twist2.config.observation_norm_epsilon_mode == "std_offset"
    assert twist2.sampling.base_distribution == "manifest_weight"
    assert twist2.sampling.curriculum_max_difficulty == 10.0


def test_retired_generic_locomotion_experiments_have_no_aliases() -> None:
    for name in ("locomotion", "hardware_locomotion"):
        with pytest.raises(ValueError, match="not a locomotion experiment"):
            locomotion_training_profile(name)  # type: ignore[arg-type]


def test_robot_locomotion_profile_paths_are_public_and_explicit() -> None:
    assert profile_config_relative_path("bello_locomotion") == Path(
        "configs/profiles/train/bello_locomotion.yaml"
    )
    assert profile_config_relative_path("bello_mini_locomotion") == Path(
        "configs/profiles/train/bello_mini_locomotion.yaml"
    )


def test_contact_budget_command_override_is_explicit_and_positive() -> None:
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_ROOT / "commands"),
    ):
        raw = compose(config_name="train", overrides=["contact_budget_per_env=64"])
    command = command_config(TrainCommandConfig, raw)
    warp = twist_training_profile("teacher").warp

    assert _warp_overrides(warp, command).contact_budget_per_env == 64
    with pytest.raises(ValueError, match="must be positive"):
        _warp_overrides(warp, replace(command, contact_budget_per_env=0))


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
