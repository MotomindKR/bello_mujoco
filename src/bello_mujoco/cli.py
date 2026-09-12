"""Small Hydra command surface for supported Bello workflows."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Literal

import hydra
from omegaconf import DictConfig

from bello_mujoco.configuration import (
    ExperimentName,
    command_config,
    config_path,
    locomotion_training_profile,
    motion_training_profile,
)
from bello_mujoco.robot import AssetName

if TYPE_CHECKING:
    from bello_mujoco.training.tracking import WandbConfig


@dataclass(frozen=True)
class TrainCommandConfig:
    experiment: ExperimentName
    seed: int | None
    total_timesteps: int | None
    num_envs: int | None
    learning_rate: float | None
    contact_budget_per_env: int | None
    motion_path: Path | None
    end_effector_payload_mass_kg: float | None
    resume_path: Path | None
    checkpoint_directory: Path | None
    checkpoint_name: str | None
    wandb_enabled: bool | None
    run_name: str | None
    video_enabled: bool | None


@dataclass(frozen=True)
class EvalCommandConfig:
    task: Literal["locomotion", "motion"]
    checkpoint_path: Path | None
    end_effector_payload_mass_kg: float | None
    output_path: Path
    video_path: Path | None
    command: tuple[float, float, float]
    steps: int | None
    stop_on_done: bool
    stochastic_actions: bool
    action_seed: int
    gamepad_enabled: bool


@dataclass(frozen=True)
class VastTrainCommandConfig:
    experiment: ExperimentName
    motion_path: Path | None
    total_timesteps: int | None
    learning_rate: float | None
    end_effector_payload_mass_kg: float | None
    max_price: float | None
    branch: str | None
    run_name: str | None
    duration_hours: float
    ssh_timeout_seconds: int
    skip_confirmation: bool
    dry_run: bool
    preferred_countries: list[str]


@dataclass(frozen=True)
class EmulatorCommandConfig:
    asset: AssetName
    bus_interfaces: tuple[str, ...]
    command_latency_seconds: float
    motor_kp_scale: float
    motor_kd_scale: float
    video_path: Path | None
    end_effector_payload_mass_kg: float


@dataclass(frozen=True)
class HardwareCommandConfig:
    task: Literal["locomotion", "motion"]
    checkpoint_path: Path | None
    standing_checkpoint_path: Path | None
    endpoint: str
    remote_endpoint: str | None
    remote_staleness_ms: int
    joystick_command_scale: float
    end_effector_payload_mass_kg: float | None
    duration_seconds: float | None
    ramp_seconds: float | None
    gain_scale: float
    max_linear_acceleration: float | None
    policy_transition_seconds: float | None
    max_motion_tilt_radians: float | None
    status_interval_seconds: float | None
    policy_log_path: Path | None
    arm: bool
    allow_provisional_hardware: bool
    upper_body_tracking_enabled: bool
    rtmw_endpoint: str
    rtmw_tls_root_certificate_path: Path | None
    rtmw_tls_server_name_override: str | None
    rtmw_connect_timeout_seconds: float
    rtmw_rpc_timeout_seconds: float
    rtmw_reference_qualification_frames: int
    rtmw_min_joint_confidence: float
    rtmw_min_rotation_confidence: float
    rtmw_max_required_rotation_residual_degrees: float
    rtmw_max_pipeline_latency_ms: float
    rtmw_transport_latency_budget_ms: float
    rtmw_max_source_interval_seconds: float
    rtmw_camera_to_world_quaternion_xyzw: tuple[float, float, float, float]
    rtmw_align_initial_heading: bool
    rtmw_align_initial_floor: bool
    rtmw_human_height_m: float
    rtmw_gmr_task_profile: str
    smplx_neutral_model_path: Path | None
    smplx_human_height_m: float


@dataclass(frozen=True)
class HandsCommandConfig:
    protocol: Literal["linker_o6", "linker_l6"]
    can_interface: str
    left_hand_enabled: bool
    right_hand_enabled: bool
    can_probe_timeout_seconds: float
    can_feedback_poll_interval_seconds: float
    can_feedback_timeout_seconds: float
    can_tx_spacing_seconds: float
    rtmw_endpoint: str
    rtmw_tls_root_certificate_path: Path | None
    rtmw_tls_server_name_override: str | None
    rtmw_connect_timeout_seconds: float
    rtmw_rpc_timeout_seconds: float
    rtmw_human_height_m: float
    min_hand_confidence: float
    max_pipeline_latency_ms: float
    transport_latency_budget_ms: float
    max_reference_age_ms: int
    max_source_interval_seconds: float
    control_rate_hz: float
    smoothing_time_constant_ms: float
    max_raw_rate_per_second: float
    flexion_open_threshold: float
    flexion_close_threshold: float
    finger_full_flex_degrees: float
    thumb_full_flex_degrees: float
    thumb_adduction_range_degrees: float
    minimum_segment_length_m: float
    left_open_positions: tuple[int, int, int, int, int, int]
    left_closed_positions: tuple[int, int, int, int, int, int]
    right_open_positions: tuple[int, int, int, int, int, int]
    right_closed_positions: tuple[int, int, int, int, int, int]
    duration_seconds: float | None
    arm: bool
    calibration_confirmed: bool


def train_entrypoint() -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    _train_hydra()


@hydra.main(
    version_base=None,
    config_path=config_path("commands"),
    config_name="train",
)
def _train_hydra(config: DictConfig) -> None:
    command = command_config(TrainCommandConfig, config)
    if command.resume_path is not None and command.motion_path is not None:
        raise ValueError(
            "resume uses the checkpoint's bundled motion; do not set motion_path"
        )
    if command.experiment == "motion":
        from bello_mujoco.training.motion import (
            run_training as run_motion_training,
        )

        run_config = motion_training_profile(command.motion_path)
        run_config = replace(
            run_config,
            config=_ppo_overrides(run_config.config, command),
            warp=_warp_overrides(run_config.warp, command),
            asset_config=replace(
                run_config.asset_config,
                end_effector_payload_mass_kg=_coalesce(
                    command.end_effector_payload_mass_kg,
                    run_config.asset_config.end_effector_payload_mass_kg,
                ),
            ),
            checkpoint_config=_checkpoint_overrides(
                run_config.checkpoint_config, command
            ),
            wandb_config=_wandb_overrides(run_config.wandb_config, command),
            video_config=replace(
                run_config.video_config,
                enabled=_coalesce(
                    command.video_enabled, run_config.video_config.enabled
                ),
            ),
        )
        run_motion_training(run_config)
        return
    if (
        command.motion_path is not None
        or command.end_effector_payload_mass_kg is not None
    ):
        raise ValueError(
            "motion_path and end_effector_payload_mass_kg are only valid for "
            "the motion experiment"
        )
    _reject_explicit_fields(
        (
            "motion_path",
            "end_effector_payload_mass_kg",
        ),
        context=f"experiment={command.experiment}",
    )
    from bello_mujoco.training.locomotion import (
        run_training as run_locomotion_training,
    )

    run_config = locomotion_training_profile(command.experiment)
    run_config = replace(
        run_config,
        config=_ppo_overrides(run_config.config, command),
        warp=_warp_overrides(run_config.warp, command),
        checkpoint_config=_checkpoint_overrides(run_config.checkpoint_config, command),
        wandb_config=_wandb_overrides(run_config.wandb_config, command),
        video_config=replace(
            run_config.video_config,
            enabled=_coalesce(command.video_enabled, run_config.video_config.enabled),
        ),
    )
    run_locomotion_training(run_config)


def eval_entrypoint() -> None:
    from bello_mujoco.evaluation.software import configure_software_rendering

    configure_software_rendering()
    _expand_path_overrides(("checkpoint_path=", "output_path=", "video_path="))
    _eval_hydra()


def export_entrypoint() -> None:
    os.environ["JAX_PLATFORMS"] = "cpu"
    _expand_path_overrides(("checkpoint_path=", "output_path="))
    _export_hydra()


@hydra.main(
    version_base=None, config_path=config_path("commands"), config_name="export"
)
def _export_hydra(config: DictConfig) -> None:
    from bello_mujoco.deployment import ExportConfig, export_checkpoint

    print(export_checkpoint(command_config(ExportConfig, config)))


@hydra.main(
    version_base=None,
    config_path=config_path("commands"),
    config_name="eval",
)
def _eval_hydra(config: DictConfig) -> None:
    command = command_config(EvalCommandConfig, config)
    checkpoint = _required_path(command.checkpoint_path, "checkpoint_path")
    if command.task == "motion":
        _reject_explicit_fields(
            (
                "command",
                "stochastic_actions",
                "action_seed",
                "gamepad_enabled",
                "video_path",
            ),
            context="task=motion",
        )
        from bello_mujoco.evaluation.motion import render_checkpoint_with_stats

        stats = render_checkpoint_with_stats(
            checkpoint,
            command.output_path.expanduser(),
            steps=command.steps,
            stop_on_terminal=command.stop_on_done,
            end_effector_payload_mass_kg=command.end_effector_payload_mass_kg,
        )
        print(json.dumps(stats, indent=2, sort_keys=True))
        return
    _reject_explicit_fields(
        ("output_path", "video_path", "end_effector_payload_mass_kg"),
        context="task=locomotion",
    )
    if command.gamepad_enabled:
        _reject_explicit_fields(("command",), context="gamepad_enabled=true")
    if not command.stochastic_actions:
        _reject_explicit_fields(
            ("action_seed",),
            context="stochastic_actions=false",
        )
    from bello_mujoco.evaluation.locomotion import (
        EvalPolicyConfig,
        evaluate_locomotion_checkpoint,
    )

    evaluate_locomotion_checkpoint(
        checkpoint,
        EvalPolicyConfig(
            command=command.command,
            steps=command.steps,
            stop_on_done=command.stop_on_done,
            stochastic_actions=command.stochastic_actions,
            action_seed=command.action_seed,
            gamepad=replace(
                EvalPolicyConfig().gamepad,
                enabled=command.gamepad_enabled,
            ),
        ),
    )


def vast_train_entrypoint() -> None:
    _expand_path_overrides(("motion_path=",))
    _vast_train_hydra()


@hydra.main(
    version_base=None,
    config_path=config_path("commands"),
    config_name="vast_train",
)
def _vast_train_hydra(config: DictConfig) -> None:
    command = command_config(VastTrainCommandConfig, config)
    if not math.isfinite(command.duration_hours) or command.duration_hours <= 0:
        raise ValueError("duration_hours must be finite and positive")
    if command.experiment == "motion":
        if command.motion_path is None:
            raise ValueError(
                "motion_path must select a repository-tracked clip for Vast motion "
                "training"
            )
    else:
        if (
            command.motion_path is not None
            or command.end_effector_payload_mass_kg is not None
        ):
            raise ValueError(
                "motion_path and end_effector_payload_mass_kg are only valid for "
                "the motion experiment"
            )
        _reject_explicit_fields(
            ("motion_path", "end_effector_payload_mass_kg"),
            context=f"experiment={command.experiment}",
        )
    from bello_mujoco.models import repo_root
    from bello_mujoco.training.vast import (
        VastTrainConfig,
        launch_pipeline,
        safe_exception_message,
    )

    root = repo_root()
    internal = VastTrainConfig(
        experiment=command.experiment,
        motion_path=command.motion_path,
        total_timesteps=command.total_timesteps,
        learning_rate=command.learning_rate,
        end_effector_payload_mass_kg=command.end_effector_payload_mass_kg,
        max_price=command.max_price,
        branch=command.branch,
        run_name=command.run_name,
        skip_confirmation=command.skip_confirmation,
        dry_run=command.dry_run,
        preferred_countries=tuple(command.preferred_countries),
        ssh_timeout_seconds=command.ssh_timeout_seconds,
        job_timeout_seconds=round(command.duration_hours * 3600),
        progress_timeout_seconds=min(
            1800, max(60, round(command.duration_hours * 3600))
        ),
    )
    try:
        status = launch_pipeline(root, internal)
    except KeyboardInterrupt:
        print("error: interrupted; instance teardown was attempted", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"error: {safe_exception_message(exc)}", file=sys.stderr)
        raise SystemExit(1) from None
    raise SystemExit(status)


def emulator_entrypoint() -> None:
    _expand_path_overrides(("video_path=",))
    _emulator_hydra()


@hydra.main(
    version_base=None,
    config_path=config_path("commands"),
    config_name="emulator",
)
def _emulator_hydra(config: DictConfig) -> None:
    command = command_config(EmulatorCommandConfig, config)
    from bello_mujoco.robstride import emulator

    emulator.run_emulator(
        emulator.EmulatorConfig(
            bus_interfaces=command.bus_interfaces,
            command_latency_seconds=command.command_latency_seconds,
            motor_kp_scale=command.motor_kp_scale,
            motor_kd_scale=command.motor_kd_scale,
            video_path=command.video_path,
            asset_config=replace(
                emulator.default_emulator_asset_config(command.asset),
                end_effector_payload_mass_kg=command.end_effector_payload_mass_kg,
            ),
        )
    )


def reference_entrypoint() -> None:
    os.environ["JAX_PLATFORMS"] = "cpu"
    _expand_path_overrides(
        ("rtmw.neutral_model_path=", "rtmw.tls_root_certificate_path=")
    )
    _reference_hydra()


@hydra.main(
    version_base=None, config_path=config_path("commands"), config_name="reference"
)
def _reference_hydra(config: DictConfig) -> None:
    from bello_mujoco.hardware.reference_producer import ProducerConfig, run

    run(command_config(ProducerConfig, config))


def hands_entrypoint() -> None:
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    _expand_path_overrides(("rtmw_tls_root_certificate_path=",))
    _hands_hydra()


@hydra.main(
    version_base=None,
    config_path=config_path("commands"),
    config_name="hands",
)
def _hands_hydra(config: DictConfig) -> None:
    command = command_config(HandsCommandConfig, config)
    from bello_mujoco.hardware.hand_control import (
        LinkerHandsCanConfig,
        LinkerHandsCanTransport,
    )
    from bello_mujoco.hardware.rtmw_hands import (
        HandGeometryConfig,
        HandRawCalibration,
        RtmwHandsConfig,
        run_rtmw_hands,
    )

    can_config = LinkerHandsCanConfig(
        interface=command.can_interface,
        model="O6" if command.protocol == "linker_o6" else "L6",
        left_enabled=command.left_hand_enabled,
        right_enabled=command.right_hand_enabled,
        probe_timeout_seconds=command.can_probe_timeout_seconds,
        feedback_poll_interval_seconds=(command.can_feedback_poll_interval_seconds),
        feedback_timeout_seconds=command.can_feedback_timeout_seconds,
        tx_spacing_seconds=command.can_tx_spacing_seconds,
    )
    hands_config = RtmwHandsConfig(
        endpoint=command.rtmw_endpoint,
        tls_root_certificate_path=command.rtmw_tls_root_certificate_path,
        tls_server_name_override=command.rtmw_tls_server_name_override,
        connect_timeout_seconds=command.rtmw_connect_timeout_seconds,
        rpc_timeout_seconds=command.rtmw_rpc_timeout_seconds,
        human_height_m=command.rtmw_human_height_m,
        min_confidence=command.min_hand_confidence,
        max_pipeline_latency_ms=command.max_pipeline_latency_ms,
        transport_latency_budget_ms=command.transport_latency_budget_ms,
        max_reference_age_ms=command.max_reference_age_ms,
        max_source_interval_seconds=command.max_source_interval_seconds,
        control_rate_hz=command.control_rate_hz,
        smoothing_time_constant_ms=command.smoothing_time_constant_ms,
        max_raw_rate_per_second=command.max_raw_rate_per_second,
        flexion_open_threshold=command.flexion_open_threshold,
        flexion_close_threshold=command.flexion_close_threshold,
        max_duration_seconds=command.duration_seconds,
        left_enabled=command.left_hand_enabled,
        right_enabled=command.right_hand_enabled,
        arm=command.arm,
        calibration_confirmed=command.calibration_confirmed,
        left_calibration=HandRawCalibration(
            open_positions=command.left_open_positions,
            closed_positions=command.left_closed_positions,
        ),
        right_calibration=HandRawCalibration(
            open_positions=command.right_open_positions,
            closed_positions=command.right_closed_positions,
        ),
        geometry=HandGeometryConfig(
            finger_full_flex_degrees=command.finger_full_flex_degrees,
            thumb_full_flex_degrees=command.thumb_full_flex_degrees,
            thumb_adduction_range_degrees=(command.thumb_adduction_range_degrees),
            minimum_segment_length_m=command.minimum_segment_length_m,
        ),
    )

    def transport_factory(_config: RtmwHandsConfig) -> LinkerHandsCanTransport:
        return LinkerHandsCanTransport(can_config)

    run_rtmw_hands(hands_config, transport_factory)
    if not command.arm:
        print(
            "RTMW hand preflight passed; SocketCAN was not opened and no hand "
            "command was sent."
        )


def hardware_entrypoint() -> None:
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    _expand_path_overrides(
        (
            "checkpoint_path=",
            "standing_checkpoint_path=",
            "policy_log_path=",
            "rtmw_tls_root_certificate_path=",
            "smplx_neutral_model_path=",
        )
    )
    _hardware_hydra()


@hydra.main(
    version_base=None,
    config_path=config_path("commands"),
    config_name="hardware",
)
def _hardware_hydra(config: DictConfig) -> None:
    command = command_config(HardwareCommandConfig, config)
    checkpoint = _required_path(command.checkpoint_path, "checkpoint_path")
    policy_log_path = _hardware_policy_log_path(command.policy_log_path)
    rtmw_fields = (
        "rtmw_endpoint",
        "rtmw_tls_root_certificate_path",
        "rtmw_tls_server_name_override",
        "rtmw_connect_timeout_seconds",
        "rtmw_rpc_timeout_seconds",
        "rtmw_reference_qualification_frames",
        "rtmw_min_joint_confidence",
        "rtmw_min_rotation_confidence",
        "rtmw_max_required_rotation_residual_degrees",
        "rtmw_max_pipeline_latency_ms",
        "rtmw_transport_latency_budget_ms",
        "rtmw_max_source_interval_seconds",
        "rtmw_camera_to_world_quaternion_xyzw",
        "rtmw_align_initial_heading",
        "rtmw_align_initial_floor",
        "rtmw_human_height_m",
        "rtmw_gmr_task_profile",
    )
    smplx_fields = ("smplx_neutral_model_path", "smplx_human_height_m")
    locomotion_joystick_fields = ("joystick_command_scale",)
    locomotion_upper_body_fields = ("upper_body_tracking_enabled",)
    if command.task == "motion":
        _reject_explicit_fields(
            (
                "remote_endpoint",
                "remote_staleness_ms",
                *locomotion_joystick_fields,
                *rtmw_fields,
                *smplx_fields,
                *locomotion_upper_body_fields,
                "allow_provisional_hardware",
            ),
            context="task=motion",
        )
        standing_checkpoint = _required_path(
            command.standing_checkpoint_path,
            "standing_checkpoint_path",
        )
        from bello_mujoco.hardware.motion import (
            HardwareMotionConfig,
            run_hardware_motion,
        )

        base = HardwareMotionConfig()
        run_hardware_motion(
            checkpoint,
            standing_checkpoint,
            replace(
                base,
                endpoint=command.endpoint,
                end_effector_payload_mass_kg=command.end_effector_payload_mass_kg,
                max_duration_seconds=command.duration_seconds,
                ramp_seconds=_coalesce(command.ramp_seconds, base.ramp_seconds),
                gain_scale=command.gain_scale,
                max_linear_acceleration=_coalesce(
                    command.max_linear_acceleration,
                    base.max_linear_acceleration,
                ),
                policy_transition_seconds=_coalesce(
                    command.policy_transition_seconds,
                    base.policy_transition_seconds,
                ),
                max_motion_tilt_radians=_coalesce(
                    command.max_motion_tilt_radians,
                    base.max_motion_tilt_radians,
                ),
                status_interval_seconds=command.status_interval_seconds,
                policy_log_path=policy_log_path,
                arm=command.arm,
            ),
        )
        return
    if (
        command.standing_checkpoint_path is not None
        or command.end_effector_payload_mass_kg is not None
    ):
        raise ValueError(
            "standing_checkpoint_path and "
            "end_effector_payload_mass_kg are only valid for task=motion"
        )
    _reject_explicit_fields(
        (
            "standing_checkpoint_path",
            "end_effector_payload_mass_kg",
            "policy_transition_seconds",
            "max_motion_tilt_radians",
        ),
        context="task=locomotion",
    )
    if command.remote_endpoint is None or not command.remote_endpoint.strip():
        raise ValueError("remote_endpoint is required for task=locomotion")
    if not command.upper_body_tracking_enabled:
        _reject_explicit_fields(
            (
                *rtmw_fields,
                *smplx_fields,
            ),
            context="upper_body_tracking_enabled=false",
        )
    from bello_mujoco.hardware.locomotion import (
        HardwareLocomotionConfig,
        run_hardware_locomotion,
    )

    base = HardwareLocomotionConfig()
    run_hardware_locomotion(
        checkpoint,
        replace(
            base,
            endpoint=command.endpoint,
            remote=replace(
                base.remote,
                endpoint=command.remote_endpoint,
                staleness_ms=command.remote_staleness_ms,
            ),
            joystick_command_scale=command.joystick_command_scale,
            max_duration_seconds=command.duration_seconds,
            ramp_seconds=_coalesce(command.ramp_seconds, base.ramp_seconds),
            gain_scale=command.gain_scale,
            max_linear_acceleration=_coalesce(
                command.max_linear_acceleration,
                base.max_linear_acceleration,
            ),
            status_interval_seconds=command.status_interval_seconds,
            policy_log_path=policy_log_path,
            arm=command.arm,
            allow_provisional_hardware=command.allow_provisional_hardware,
            upper_body_tracking_enabled=command.upper_body_tracking_enabled,
            rtmw_endpoint=command.rtmw_endpoint,
            rtmw_tls_root_certificate_path=command.rtmw_tls_root_certificate_path,
            rtmw_tls_server_name_override=command.rtmw_tls_server_name_override,
            rtmw_connect_timeout_seconds=command.rtmw_connect_timeout_seconds,
            rtmw_rpc_timeout_seconds=command.rtmw_rpc_timeout_seconds,
            rtmw_reference_qualification_frames=(
                command.rtmw_reference_qualification_frames
            ),
            rtmw_min_joint_confidence=command.rtmw_min_joint_confidence,
            rtmw_min_rotation_confidence=command.rtmw_min_rotation_confidence,
            rtmw_max_required_rotation_residual_degrees=(
                command.rtmw_max_required_rotation_residual_degrees
            ),
            rtmw_max_pipeline_latency_ms=command.rtmw_max_pipeline_latency_ms,
            rtmw_transport_latency_budget_ms=(command.rtmw_transport_latency_budget_ms),
            rtmw_max_source_interval_seconds=(command.rtmw_max_source_interval_seconds),
            rtmw_camera_to_world_quaternion_xyzw=(
                command.rtmw_camera_to_world_quaternion_xyzw
            ),
            rtmw_align_initial_heading=command.rtmw_align_initial_heading,
            rtmw_align_initial_floor=command.rtmw_align_initial_floor,
            smplx_neutral_model_path=(
                _required_path(
                    command.smplx_neutral_model_path,
                    "smplx_neutral_model_path",
                )
                if command.upper_body_tracking_enabled
                else None
            ),
            smplx_human_height_m=command.smplx_human_height_m,
            rtmw_human_height_m=command.rtmw_human_height_m,
            rtmw_gmr_task_profile=command.rtmw_gmr_task_profile,
            upper_body_follow=replace(
                base.upper_body_follow,
            ),
        ),
    )


def _hardware_policy_log_path(configured: Path | None) -> Path | None:
    return None if configured is None else configured.expanduser()


def _ppo_overrides(config, command: TrainCommandConfig):
    values = {}
    for name in ("seed", "total_timesteps", "num_envs", "learning_rate"):
        value = getattr(command, name)
        if value is not None:
            values[name] = value
    return replace(config, **values)


def _checkpoint_overrides(config, command: TrainCommandConfig):
    values = {}
    if command.resume_path is not None:
        values["resume_path"] = command.resume_path.expanduser()
    if command.checkpoint_directory is not None:
        values["directory"] = command.checkpoint_directory.expanduser()
    if command.checkpoint_name is not None:
        values["name"] = command.checkpoint_name
    return replace(config, **values)


def _warp_overrides(config, command: TrainCommandConfig):
    value = command.contact_budget_per_env
    if value is None:
        return config
    if value <= 0:
        raise ValueError("contact_budget_per_env must be positive")
    return replace(config, contact_budget_per_env=value)


def _wandb_overrides(
    config: WandbConfig,
    command: TrainCommandConfig,
) -> WandbConfig:
    values = {}
    if command.wandb_enabled is not None:
        values["enabled"] = command.wandb_enabled
    if command.run_name is not None:
        values["run_name"] = command.run_name
    return replace(config, **values)


def _required_path(value: Path | None, name: str) -> Path:
    if value is None:
        raise ValueError(f"{name} must be set")
    return value.expanduser()


def _reject_explicit_fields(
    names: tuple[str, ...],
    *,
    context: str,
) -> None:
    explicit = _explicit_override_fields()
    invalid = [name for name in names if name in explicit]
    if invalid:
        raise ValueError(f"{context} does not accept: {', '.join(invalid)}")


def _explicit_override_fields() -> frozenset[str]:
    from hydra.core.hydra_config import HydraConfig

    overrides = (
        HydraConfig.get().overrides.task if HydraConfig.initialized() else sys.argv[1:]
    )
    names: set[str] = set()
    for override in overrides:
        key = override.partition("=")[0].lstrip("+~")
        if key and not key.startswith("-"):
            names.add(key)
    return frozenset(names)


def _coalesce(value, default):
    return default if value is None else value


def _expand_path_overrides(prefixes: tuple[str, ...]) -> None:
    expanded = []
    for argument in sys.argv[1:]:
        for prefix in prefixes:
            if argument.startswith(prefix):
                value = argument.removeprefix(prefix)
                if value.startswith("~"):
                    argument = prefix + str(Path(value).expanduser())
                break
        expanded.append(argument)
    sys.argv[1:] = expanded
