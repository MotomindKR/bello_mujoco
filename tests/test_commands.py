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
    HandsCommandConfig,
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
    "hands": HandsCommandConfig,
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


def test_hands_entrypoint_forces_cpu_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bello_mujoco.cli as cli

    monkeypatch.setenv("JAX_PLATFORMS", "gpu")
    monkeypatch.setenv("JAX_PLATFORM_NAME", "gpu")
    monkeypatch.setattr(cli, "_hands_hydra", lambda: None)

    cli.hands_entrypoint()

    assert cli.os.environ["JAX_PLATFORMS"] == "cpu"
    assert cli.os.environ["JAX_PLATFORM_NAME"] == "cpu"


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


def test_hardware_joystick_limits_are_flat_overrides() -> None:
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_ROOT / "commands"),
    ):
        config = compose(
            config_name="hardware",
            overrides=(
                "joystick_command_scale=0.4",
                "joystick_linear_slew_rate_mps2=0.8",
                "joystick_yaw_slew_rate_rps2=1.2",
            ),
        )

    command = command_config(HardwareCommandConfig, config)
    assert command.joystick_command_scale == pytest.approx(0.4)
    assert command.joystick_linear_slew_rate_mps2 == pytest.approx(0.8)
    assert command.joystick_yaw_slew_rate_rps2 == pytest.approx(1.2)


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
    assert default.rtmw_reference_qualification_frames == 1
    assert default.rtmw_min_joint_confidence == pytest.approx(0.0)
    assert default.rtmw_min_rotation_confidence == pytest.approx(0.0)
    assert default.rtmw_max_source_interval_seconds == pytest.approx(0.70)
    assert default.rtmw_human_height_m == pytest.approx(1.75)
    assert default.smplx_human_height_m == pytest.approx(1.66)
    assert default.rtmw_gmr_task_profile == "live_upper_body"
    assert default.rtmw_max_required_rotation_residual_degrees == pytest.approx(35.0)
    assert not disabled.upper_body_tracking_enabled


def test_hands_command_defaults_are_disarmed_classic_can() -> None:
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_ROOT / "commands"),
    ):
        command = command_config(
            HandsCommandConfig,
            compose(config_name="hands"),
        )

    assert command.protocol == "linker_o6"
    assert command.can_interface == "can1"
    assert command.left_hand_enabled and command.right_hand_enabled
    assert command.control_rate_hz == pytest.approx(5.0)
    assert command.can_tx_spacing_seconds == pytest.approx(0.003)
    assert command.min_hand_confidence == pytest.approx(0.05)
    assert command.smoothing_time_constant_ms == pytest.approx(100.0)
    assert command.flexion_open_threshold == pytest.approx(0.35)
    assert command.flexion_close_threshold == pytest.approx(0.65)
    assert command.left_open_positions == (255, 255, 255, 255, 255, 255)
    assert command.left_closed_positions == (0, 0, 0, 0, 0, 0)
    assert not command.arm
    assert not command.calibration_confirmed


def test_hands_command_wires_rtmw_mapping_and_l6_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bello_mujoco.cli as cli
    from bello_mujoco.hardware import hand_control, rtmw_hands

    captured: dict[str, object] = {}

    class FakeTransport:
        def __init__(self, config: object) -> None:
            captured["can"] = config

    def fake_run(config: object, transport_factory: object) -> None:
        captured["rtmw"] = config
        captured["transport"] = transport_factory(config)  # type: ignore[operator]

    monkeypatch.setattr(hand_control, "LinkerHandsCanTransport", FakeTransport)
    monkeypatch.setattr(rtmw_hands, "run_rtmw_hands", fake_run)
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_ROOT / "commands"),
    ):
        config = compose(
            config_name="hands",
            overrides=(
                "protocol=linker_l6",
                "right_hand_enabled=false",
                "arm=true",
                "calibration_confirmed=true",
                "duration_seconds=2.5",
                "flexion_open_threshold=0.25",
                "flexion_close_threshold=0.75",
                "left_open_positions=[250,240,230,220,210,200]",
                "left_closed_positions=[10,20,30,40,50,60]",
            ),
        )

    cli._hands_hydra.__wrapped__(config)

    can = captured["can"]
    assert isinstance(can, hand_control.LinkerHandsCanConfig)
    assert can.model == "L6"
    assert can.interface == "can1"
    assert can.left_enabled and not can.right_enabled
    runtime = captured["rtmw"]
    assert isinstance(runtime, rtmw_hands.RtmwHandsConfig)
    assert runtime.max_duration_seconds == pytest.approx(2.5)
    assert runtime.left_enabled and not runtime.right_enabled
    assert runtime.arm and runtime.calibration_confirmed
    assert runtime.flexion_open_threshold == pytest.approx(0.25)
    assert runtime.flexion_close_threshold == pytest.approx(0.75)
    assert runtime.left_calibration.open_positions == (
        250,
        240,
        230,
        220,
        210,
        200,
    )
    assert runtime.left_calibration.closed_positions == (10, 20, 30, 40, 50, 60)
    assert runtime.geometry.thumb_adduction_range_degrees == pytest.approx(60.0)


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
    assert mini.asset_config.imu_config.body_name == "pelvis_link"
    assert "waist_roll_joint" in mini.asset_config.actuator_config.policy_joint_names
    assert mini.reward_config.weights == replace(
        bello.reward_config.weights,
        healthy_survival=2.25,
        moving_lin_vel_xy_error=-1.0,
        moving_ang_vel_z_error=-1.0,
        stand_hip_roll_continuous_torque_excess=-8.0,
        stand_waist_deviation=-4.0,
        gait_contact=-2.5,
        gait_swing_clearance=-1.0,
    )
    assert bello.reward_config.tracking == mini.reward_config.tracking
    assert bello.reward_config.standing == mini.reward_config.standing
    assert mini.reward_config.gait == replace(
        bello.reward_config.gait,
        swing_clearance_target=0.075,
    )
    assert bello.reward_config.smoothness == mini.reward_config.smoothness
    assert bello.reward_config.weights.healthy_survival == pytest.approx(1.5)
    assert mini.reward_config.weights.healthy_survival == pytest.approx(2.25)
    assert bello.reward_config.weights.touchdown_impact == pytest.approx(-8.0)
    assert mini.reward_config.weights.touchdown_impact == pytest.approx(-8.0)
    assert bello.reward_config.weights.gait_contact == pytest.approx(-1.0)
    assert bello.reward_config.weights.gait_swing_clearance == pytest.approx(-0.5)
    assert bello.reward_config.weights.moving_lin_vel_xy_error == pytest.approx(0.0)
    assert mini.reward_config.weights.moving_lin_vel_xy_error == pytest.approx(-1.0)
    assert bello.reward_config.weights.moving_ang_vel_z_error == pytest.approx(0.0)
    assert mini.reward_config.weights.moving_ang_vel_z_error == pytest.approx(-1.0)
    assert (
        mini.reward_config.weights.stand_hip_roll_continuous_torque_excess
        == pytest.approx(-8.0)
    )
    assert (
        mini.reward_config.standing.hip_roll_continuous_torque_limit
        == pytest.approx(10.0)
    )
    assert mini.reward_config.weights.stand_waist_deviation == pytest.approx(-4.0)
    assert mini.reward_config.weights.gait_contact == pytest.approx(-2.5)
    assert mini.reward_config.weights.gait_swing_clearance == pytest.approx(-1.0)
    assert mini.reward_config.gait.swing_clearance_target == pytest.approx(0.075)
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
    assert bello.env_config.random_push.lin_vel_x == (-2.0, 2.0)
    assert bello.env_config.random_push.lin_vel_y == (-2.0, 2.0)
    assert mini.env_config.random_push.lin_vel_x == (-0.5, 0.5)
    assert mini.env_config.random_push.lin_vel_y == (-0.5, 0.5)
    assert not bello.reward_config.gait.joint_force_apply_while_standing
    assert bello.reward_config.gait.touchdown_force_soft_limit_body_weights == (
        pytest.approx(1.0)
    )
    assert bello.reward_config.smoothness.ankle_action_weight == pytest.approx(1.0)
    assert bello.reward_config.smoothness.waist_action_weight == pytest.approx(1.0)
    assert bello.reward_config.limits_pose.hip_deviation_command_indices == (1,)
    assert mini.reward_config.limits_pose.hip_deviation_command_indices == (0,)
    assert bello.reward_config.limits_pose.knee_deviation_apply_while_moving
    assert not mini.reward_config.limits_pose.knee_deviation_apply_while_moving
    assert bello.reward_config.links.track_lin_vel_xy_frame is None
    assert mini.reward_config.links.track_lin_vel_xy_frame is None
    for profile in (bello, mini):
        assert profile.reward_config.links.flat_orientation == "torso_link"
        assert profile.reward_config.links.base_height == "pelvis_link"
        assert profile.done_config.bad_orientation.link == "torso_link"
        assert profile.done_config.base_height.link == "pelvis_link"
    assert bello.reward_config.limits_pose.base_height_target == pytest.approx(0.86)
    assert mini.reward_config.limits_pose.base_height_target == pytest.approx(0.87)
    assert bello.done_config.base_height.min_height == pytest.approx(0.55)
    assert mini.done_config.base_height.min_height == pytest.approx(0.56)
    assert bello.env_config.gait.frequency_hz == pytest.approx(0.9)
    assert mini.env_config.gait.frequency_hz == pytest.approx(0.90)
    assert bello.warp.contact_budget_per_env == 40
    assert mini.warp.contact_budget_per_env == 40
    assert mini.env_config.spawn == replace(
        bello.env_config.spawn,
        joint_pos_scale_range=(1.0, 1.0),
    )
    assert (
        replace(
            mini.env_config,
            gait=bello.env_config.gait,
            random_push=bello.env_config.random_push,
            spawn=bello.env_config.spawn,
        )
        == bello.env_config
    )
    assert bello.video_config.command == (0.0, 0.5, 0.0)
    assert mini.video_config.command == (0.5, 0.0, 0.0)
    assert bello.env_config.observation_latency.enabled
    assert bello.env_config.observation_latency.delay_steps == (0, 2)
    assert bello.env_config.observation_low_pass_filter.enabled
    assert bello.env_config.observation_low_pass_filter.cutoff_hz == pytest.approx(10.0)
    assert bello.env_config.observation_low_pass_filter.noise_before_filter
    assert bello.env_config.observation_low_pass_filter.residual_noise_scale == (
        pytest.approx(0.1)
    )
    assert bello.env_config.torso_com_curriculum.enabled
    assert bello.env_config.torso_com_curriculum.ramp_steps == 488_281
    assert bello.env_config.model_randomization.torso_mass_scale_range == (
        0.8,
        1.2,
    )
    assert bello.env_config.model_randomization.torso_com_offset_range == (
        -0.15,
        0.15,
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
    assert profile_config_relative_path("bello_mini_standing") == Path(
        "configs/profiles/train/bello_mini_standing.yaml"
    )


def test_bello_mini_standing_profile_keeps_locomotion_plumbing() -> None:
    locomotion = locomotion_training_profile("bello_mini_locomotion")
    standing = locomotion_training_profile("bello_mini_standing")

    assert standing.config == locomotion.config
    assert standing.asset_config == locomotion.asset_config
    assert standing.env_config == locomotion.env_config
    assert standing.done_config == locomotion.done_config
    assert standing.warp == locomotion.warp
    assert standing.command_config.enabled
    assert standing.command_config.lin_vel_x == (0.0, 0.0)
    assert standing.command_config.lin_vel_y == (0.0, 0.0)
    assert standing.command_config.ang_vel_yaw == (0.0, 0.0)
    assert standing.command_config.standing_probability == pytest.approx(1.0)
    assert not standing.command_config.standing_transitions_enabled
    assert standing.reward_config.weights == replace(
        locomotion.reward_config.weights,
        moving_lin_vel_xy_error=0.0,
        moving_ang_vel_z_error=0.0,
        joint_force_peak=0.0,
        gait_contact=0.0,
        gait_stance_slip=0.0,
        gait_swing_clearance=0.0,
    )
    assert standing.reward_config.tracking == locomotion.reward_config.tracking
    assert standing.reward_config.standing == locomotion.reward_config.standing
    assert standing.checkpoint_config.name == "bello_mini_standing_seed0"
    assert standing.checkpoint_config.wandb_artifact_name == (
        "bello_mini_standing_seed0"
    )
    assert standing.wandb_config.group == "bello-mini-standing"
    assert standing.video_config.command == (0.0, 0.0, 0.0)


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
        (HardwareCommandConfig, {"joystick_command_scale": True}),
        (HardwareCommandConfig, {"joystick_linear_slew_rate_mps2": True}),
        (HardwareCommandConfig, {"joystick_yaw_slew_rate_rps2": True}),
        (HandsCommandConfig, {"arm": 1}),
        (HandsCommandConfig, {"calibration_confirmed": 1}),
        (HandsCommandConfig, {"control_rate_hz": True}),
        (HandsCommandConfig, {"max_reference_age_ms": True}),
        (EvalCommandConfig, {"allow_git_commit_mismatch": 1}),
        (TrainCommandConfig, {"allow_git_commit_mismatch": 1}),
    ):
        try:
            structured_config(config_type, OmegaConf.create(raw))
        except TypeError:
            pass
        else:
            raise AssertionError(f"invalid public command type was accepted: {raw}")
