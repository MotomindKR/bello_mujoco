"""Small Hydra command surface for supported Bello workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
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
    sonic_training_profile,
    twist2_training_profile,
    twist_training_profile,
)

if TYPE_CHECKING:
    from bello_mujoco.training.tracking import WandbConfig


@dataclass(frozen=True)
class PrepareAssetsCommandConfig:
    profile: Literal["physical", "visualizer"]
    output_root: Path | None


@dataclass(frozen=True)
class TrainCommandConfig:
    experiment: ExperimentName
    seed: int | None
    total_timesteps: int | None
    num_envs: int | None
    learning_rate: float | None
    contact_budget_per_env: int | None
    motion_path: Path | None
    dataset_path: Path | None
    teacher_checkpoint_path: Path | None
    sonic_artifact_path: Path | None
    end_effector_payload_mass_kg: float | None
    resume_path: Path | None
    warm_start_path: Path | None
    allow_git_commit_mismatch: bool
    checkpoint_directory: Path | None
    checkpoint_name: str | None
    checkpoint_enabled: bool | None
    checkpoint_every_updates: int | None
    wandb_enabled: bool | None
    run_name: str | None
    video_enabled: bool | None
    sonic_pretraining_enabled: bool | None


@dataclass(frozen=True)
class EvalCommandConfig:
    task: Literal["locomotion", "motion", "twist_student", "twist2", "sonic"]
    checkpoint_path: Path | None
    sonic_artifact_path: Path | None
    allow_git_commit_mismatch: bool
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
class MotionCommandConfig:
    operation: Literal["import", "import_dataset", "inspect", "render"]
    source: Path | None
    output: Path | None
    jobs: int
    overwrite: bool
    reject_self_collisions: bool
    width: int
    height: int
    fps: int
    require_complete: bool


@dataclass(frozen=True)
class TwistDataCommandConfig:
    operation: Literal["scan_amass", "validate", "build"]
    manifest: Path | None
    source_root: Path | None
    dataset_release: str | None
    license_id: str | None
    split_seed: int
    sampling_weight: float
    overwrite: bool
    body_model_root: Path | None
    retargeted_root: Path | None
    output_root: Path | None
    robot_model_path: Path
    gmr_root: Path
    quality_config: Path
    curation_config: Path


@dataclass(frozen=True)
class SonicDataCommandConfig:
    operation: Literal["retarget_g1", "build", "upgrade_v5", "upgrade_v6"]
    manifest: Path | None
    input_index: Path | None
    body_model_root: Path | None
    bello_retargeted_root: Path | None
    g1_retargeted_root: Path | None
    normalized_root: Path | None
    output_root: Path | None
    gmr_python: Path | None
    jobs: int
    overwrite: bool
    keep_normalized: bool
    limit: int | None
    quality_config: Path
    calibration_config: Path
    sanitation_config: Path

    def __post_init__(self) -> None:
        if self.operation not in {"retarget_g1", "build", "upgrade_v5", "upgrade_v6"}:
            raise ValueError(
                "SONIC data operation must be retarget_g1, build, upgrade_v5, or "
                "upgrade_v6"
            )
        if (
            isinstance(self.jobs, bool)
            or not isinstance(self.jobs, int)
            or self.jobs < 1
        ):
            raise ValueError("SONIC data jobs must be a positive integer")
        if self.limit is not None and (
            isinstance(self.limit, bool)
            or not isinstance(self.limit, int)
            or self.limit < 1
        ):
            raise ValueError("SONIC data limit must be positive when provided")


@dataclass(frozen=True)
class VastTrainCommandConfig:
    experiment: Literal[
        "smoke",
        "locomotion",
        "hardware_locomotion",
        "motion",
        "twist_teacher",
        "twist_student",
        "twist2",
        "sonic",
    ]
    motion_path: Path | None
    dataset_path: Path | None
    teacher_checkpoint_path: Path | None
    sonic_artifact_path: Path | None
    resume_path: Path | None
    warm_start_checkpoint_path: Path | None
    allow_git_commit_mismatch: bool
    total_timesteps: int | None
    num_envs: int | None
    checkpoint_every_updates: int | None
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
    min_upload_mib_per_second: float
    route_probe_mib: int


@dataclass(frozen=True)
class EmulatorCommandConfig:
    bus_interfaces: tuple[str, ...]
    command_latency_seconds: float
    end_effector_payload_mass_kg: float


@dataclass(frozen=True)
class HardwareCommandConfig:
    task: Literal["locomotion", "motion", "twist2", "sonic"]
    checkpoint_path: Path | None
    standing_checkpoint_path: Path | None
    sonic_artifact_path: Path | None
    allow_git_commit_mismatch: bool
    endpoint: str
    remote_endpoint: str | None
    remote_staleness_ms: int
    motion_path: Path | None
    end_effector_payload_mass_kg: float | None
    duration_seconds: float | None
    ramp_seconds: float | None
    gain_scale: float
    max_linear_acceleration: float | None
    policy_transition_seconds: float | None
    max_motion_tilt_radians: float | None
    action_lpf_cutoff_hz: float | None
    status_interval_seconds: float | None
    policy_log_path: Path | None
    arm: bool
    reference_mode: Literal["clip", "live"]
    fastsam_endpoint: str
    fastsam_tls_root_certificate_path: Path | None
    fastsam_tls_server_name_override: str | None
    fastsam_connect_timeout_seconds: float
    fastsam_rpc_timeout_seconds: float
    fastsam_reference_ready_timeout_seconds: float
    fastsam_reference_qualification_frames: int
    fastsam_min_tracking_confidence: float
    fastsam_min_joint_confidence: float
    fastsam_max_pipeline_latency_ms: float
    fastsam_transport_latency_budget_ms: float
    fastsam_camera_to_world_quaternion_xyzw: tuple[float, float, float, float]
    fastsam_align_initial_heading: bool
    fastsam_align_initial_floor: bool
    smplx_neutral_model_path: Path | None
    smplx_human_height_m: float
    left_elbow_yaw_offset_radians: float
    right_elbow_yaw_offset_radians: float
    gemx_endpoint: str
    gemx_connect_timeout_seconds: float
    gemx_rpc_timeout_seconds: float
    gemx_reference_ready_timeout_seconds: float
    gemx_reference_qualification_frames: int
    gemx_min_tracking_confidence: float
    gemx_min_joint_confidence: float
    gemx_max_pipeline_latency_ms: float
    gemx_transport_latency_budget_ms: float
    gemx_camera_to_world_quaternion_xyzw: tuple[float, float, float, float]
    gemx_align_initial_heading: bool
    gemx_align_initial_floor: bool


def prepare_assets_entrypoint() -> None:
    _prepare_assets_hydra()


@hydra.main(
    version_base=None,
    config_path=config_path("commands"),
    config_name="prepare_assets",
)
def _prepare_assets_hydra(config: DictConfig) -> None:
    command = command_config(PrepareAssetsCommandConfig, config)
    from bello_mujoco.assets_pipeline.prepare import (
        canonical_physical_asset_config,
        prepare_asset,
    )

    asset = canonical_physical_asset_config()
    if command.profile == "visualizer":
        asset = replace(
            asset,
            output_root=Path("assets/generated/visualizer"),
            visual_mesh_face_budget=20_000,
        )
    if command.output_root is not None:
        asset = replace(asset, output_root=command.output_root.expanduser())
    prepared = prepare_asset(asset)
    print(
        f"{prepared.name}: {prepared.collision_mesh_count} collision meshes, "
        f"{prepared.visual_mesh_count} visual meshes -> {prepared.mjcf_path} "
        f"(viewer: {prepared.viewer_mjcf_path})"
    )


def train_entrypoint() -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    _expand_path_overrides(
        (
            "motion_path=",
            "dataset_path=",
            "teacher_checkpoint_path=",
            "sonic_artifact_path=",
            "resume_path=",
            "checkpoint_directory=",
        )
    )
    _train_hydra()


@hydra.main(
    version_base=None,
    config_path=config_path("commands"),
    config_name="train",
)
def _train_hydra(config: DictConfig) -> None:
    command = command_config(TrainCommandConfig, config)
    if command.experiment == "sonic":
        _reject_explicit_fields(
            (
                "motion_path",
                "teacher_checkpoint_path",
                "video_enabled",
                "warm_start_path",
                "end_effector_payload_mass_kg",
            ),
            context="experiment=sonic",
        )
        from bello_mujoco.training.sonic import run_training as run_sonic_training

        run_config = sonic_training_profile(
            dataset_index=command.dataset_path,
            release_artifact_directory=command.sonic_artifact_path,
        )
        run_config = replace(
            run_config,
            config=_ppo_overrides(run_config.config, command),
            warp=_warp_overrides(run_config.warp, command),
            checkpoint_config=_checkpoint_overrides(
                run_config.checkpoint_config, command
            ),
            wandb_config=_wandb_overrides(run_config.wandb_config, command),
            pretraining=replace(
                run_config.pretraining,
                enabled=_coalesce(
                    command.sonic_pretraining_enabled,
                    run_config.pretraining.enabled,
                ),
            ),
        )
        run_sonic_training(run_config)
        return
    _reject_explicit_fields(
        ("sonic_pretraining_enabled",),
        context=f"experiment={command.experiment}",
    )
    if command.experiment == "twist2":
        _reject_explicit_fields(
            (
                "motion_path",
                "teacher_checkpoint_path",
                "sonic_artifact_path",
                "video_enabled",
                "end_effector_payload_mass_kg",
            ),
            context="experiment=twist2",
        )
        from bello_mujoco.training.twist2 import run_training as run_twist2_training

        run_config = twist2_training_profile(dataset_index=command.dataset_path)
        run_config = replace(
            run_config,
            config=_ppo_overrides(run_config.config, command),
            warp=_warp_overrides(run_config.warp, command),
            checkpoint_config=_checkpoint_overrides(
                run_config.checkpoint_config, command
            ),
            wandb_config=_wandb_overrides(run_config.wandb_config, command),
        )
        run_twist2_training(run_config)
        return
    if command.experiment in {"twist_teacher", "twist_student"}:
        _reject_explicit_fields(
            (
                "motion_path",
                "sonic_artifact_path",
                "video_enabled",
                "warm_start_path",
                "end_effector_payload_mass_kg",
            ),
            context=f"experiment={command.experiment}",
        )
        stage = command.experiment.removeprefix("twist_")
        if stage == "teacher":
            _reject_explicit_fields(
                ("teacher_checkpoint_path",),
                context="experiment=twist_teacher",
            )
        from bello_mujoco.training.twist import run_training as run_twist_training

        run_config = twist_training_profile(
            stage,
            dataset_index=command.dataset_path,
            teacher_checkpoint_path=command.teacher_checkpoint_path,
        )
        run_config = replace(
            run_config,
            config=_ppo_overrides(run_config.config, command),
            warp=_warp_overrides(run_config.warp, command),
            checkpoint_config=_checkpoint_overrides(
                run_config.checkpoint_config, command
            ),
            wandb_config=_wandb_overrides(run_config.wandb_config, command),
        )
        run_twist_training(run_config)
        return
    if command.experiment == "motion":
        _reject_explicit_fields(
            (
                "dataset_path",
                "teacher_checkpoint_path",
                "sonic_artifact_path",
                "warm_start_path",
            ),
            context="experiment=motion",
        )
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
            "dataset_path",
            "teacher_checkpoint_path",
            "sonic_artifact_path",
            "warm_start_path",
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
    _expand_path_overrides(
        (
            "checkpoint_path=",
            "sonic_artifact_path=",
            "output_path=",
            "video_path=",
        )
    )
    _eval_hydra()


@hydra.main(
    version_base=None,
    config_path=config_path("commands"),
    config_name="eval",
)
def _eval_hydra(config: DictConfig) -> None:
    command = command_config(EvalCommandConfig, config)
    checkpoint = _required_path(command.checkpoint_path, "checkpoint_path")
    if command.task == "sonic":
        _reject_explicit_fields(
            (
                "command",
                "stochastic_actions",
                "action_seed",
                "gamepad_enabled",
                "stop_on_done",
            ),
            context="task=sonic",
        )
        from bello_mujoco.evaluation.sonic import (
            SonicEvaluationConfig,
            evaluate_sonic_checkpoint,
        )

        base = SonicEvaluationConfig()
        report = evaluate_sonic_checkpoint(
            checkpoint,
            replace(
                base,
                release_artifact_directory=_coalesce(
                    command.sonic_artifact_path,
                    base.release_artifact_directory,
                ),
                output_path=(
                    command.output_path
                    if "output_path" in _explicit_override_fields()
                    else base.output_path
                ),
                video_path=command.video_path,
                video_max_steps=_coalesce(command.steps, base.video_max_steps),
            ),
            allow_git_commit_mismatch=command.allow_git_commit_mismatch,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if command.task == "twist2":
        _reject_explicit_fields(
            (
                "command",
                "stochastic_actions",
                "action_seed",
                "gamepad_enabled",
                "stop_on_done",
                "sonic_artifact_path",
            ),
            context="task=twist2",
        )
        from bello_mujoco.evaluation.twist2 import (
            Twist2EvaluationConfig,
            evaluate_twist2_checkpoint,
        )

        report = evaluate_twist2_checkpoint(
            checkpoint,
            Twist2EvaluationConfig(
                output_path=(
                    command.output_path
                    if "output_path" in _explicit_override_fields()
                    else Path("outputs/evaluation/twist2.json")
                ),
                video_path=(
                    command.video_path
                    if "video_path" in _explicit_override_fields()
                    else Path("outputs/evaluation/twist2.mp4")
                ),
                stop_on_terminal=True,
                video_max_steps=_coalesce(command.steps, 500),
            ),
            allow_git_commit_mismatch=command.allow_git_commit_mismatch,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if command.task == "twist_student":
        _reject_explicit_fields(
            (
                "command",
                "steps",
                "stochastic_actions",
                "action_seed",
                "gamepad_enabled",
                "video_path",
                "sonic_artifact_path",
            ),
            context="task=twist_student",
        )
        from bello_mujoco.evaluation.twist import (
            TwistEvaluationConfig,
            evaluate_twist_student_checkpoint,
        )

        report = evaluate_twist_student_checkpoint(
            checkpoint,
            TwistEvaluationConfig(
                output_path=(
                    command.output_path
                    if "output_path" in _explicit_override_fields()
                    else Path("outputs/evaluation/twist_student.json")
                ),
                stop_on_terminal=command.stop_on_done,
            ),
            allow_git_commit_mismatch=command.allow_git_commit_mismatch,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if command.task == "motion":
        _reject_explicit_fields(
            (
                "command",
                "stochastic_actions",
                "action_seed",
                "gamepad_enabled",
                "video_path",
                "sonic_artifact_path",
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
            allow_git_commit_mismatch=command.allow_git_commit_mismatch,
        )
        print(json.dumps(stats, indent=2, sort_keys=True))
        return
    _reject_explicit_fields(
        (
            "output_path",
            "video_path",
            "sonic_artifact_path",
            "end_effector_payload_mass_kg",
        ),
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
            allow_git_commit_mismatch=command.allow_git_commit_mismatch,
        ),
    )


def motion_entrypoint() -> None:
    _expand_path_overrides(("source=", "output="))
    _motion_hydra()


def twist_data_entrypoint() -> None:
    _expand_path_overrides(
        (
            "manifest=",
            "source_root=",
            "body_model_root=",
            "retargeted_root=",
            "output_root=",
            "robot_model_path=",
            "gmr_root=",
            "quality_config=",
            "curation_config=",
        )
    )
    _twist_data_hydra()


def sonic_data_entrypoint() -> None:
    _expand_path_overrides(
        (
            "manifest=",
            "body_model_root=",
            "bello_retargeted_root=",
            "g1_retargeted_root=",
            "normalized_root=",
            "output_root=",
            "input_index=",
            "gmr_python=",
            "quality_config=",
            "calibration_config=",
            "sanitation_config=",
        )
    )
    _sonic_data_hydra()


@hydra.main(
    version_base=None,
    config_path=config_path("commands"),
    config_name="sonic_data",
)
def _sonic_data_hydra(config: DictConfig) -> None:
    command = command_config(SonicDataCommandConfig, config)
    output_root = _required_path(command.output_root, "output_root")
    if command.operation == "upgrade_v6":
        _reject_explicit_fields(
            (
                "manifest",
                "body_model_root",
                "bello_retargeted_root",
                "g1_retargeted_root",
                "normalized_root",
                "gmr_python",
                "jobs",
                "overwrite",
                "keep_normalized",
                "limit",
                "quality_config",
                "calibration_config",
                "sanitation_config",
            ),
            context="operation=upgrade_v6",
        )
        input_index = _required_path(command.input_index, "input_index")
        from bello_mujoco.motion.sonic_dataset import upgrade_sonic_dataset_v6

        upgraded = upgrade_sonic_dataset_v6(input_index, output_root)
        print(json.dumps(upgraded, indent=2, sort_keys=True))
        return
    if command.operation == "upgrade_v5":
        _reject_explicit_fields(
            (
                "manifest",
                "body_model_root",
                "g1_retargeted_root",
                "normalized_root",
                "gmr_python",
                "jobs",
                "overwrite",
                "keep_normalized",
                "limit",
                "quality_config",
                "calibration_config",
                "sanitation_config",
            ),
            context="operation=upgrade_v5",
        )
        input_index = _required_path(command.input_index, "input_index")
        bello_root = _required_path(
            command.bello_retargeted_root, "bello_retargeted_root"
        )
        from bello_mujoco.motion.sonic_dataset import upgrade_sonic_dataset_v5

        def progress(stage: str, current: int, total: int) -> None:
            print(f"SONIC dataset {stage}: {current}/{total}", file=sys.stderr)

        upgraded = upgrade_sonic_dataset_v5(
            input_index,
            bello_root,
            output_root,
            progress=progress,
        )
        print(json.dumps(upgraded, indent=2, sort_keys=True))
        return
    _reject_explicit_fields(("input_index",), context=f"operation={command.operation}")
    manifest = _required_path(command.manifest, "manifest")
    if command.operation == "retarget_g1":
        _reject_explicit_fields(
            (
                "bello_retargeted_root",
                "g1_retargeted_root",
                "quality_config",
                "calibration_config",
                "sanitation_config",
            ),
            context="operation=retarget_g1",
        )
        body_model_root = _required_path(command.body_model_root, "body_model_root")
        normalized_root = _required_path(command.normalized_root, "normalized_root")
        gmr_python = _required_path(command.gmr_python, "gmr_python")
        from bello_mujoco.motion.sonic_retarget import (
            batch_retarget_sonic_manifest,
        )

        def progress(current, total, decision) -> None:
            print(
                f"retargeted {current}/{total}: {decision.source_id}/"
                f"{decision.sequence_id} [{decision.status}]",
                file=sys.stderr,
            )

        report = batch_retarget_sonic_manifest(
            manifest,
            body_model_root=body_model_root,
            output_root=output_root,
            normalized_root=normalized_root,
            gmr_python=gmr_python,
            jobs=command.jobs,
            overwrite=command.overwrite,
            keep_normalized=command.keep_normalized,
            limit=command.limit,
            progress=progress,
        )
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
        if report.failed:
            raise RuntimeError(f"{report.failed} SONIC G1 retarget(s) failed")
        return
    _reject_explicit_fields(
        (
            "body_model_root",
            "normalized_root",
            "gmr_python",
            "jobs",
            "overwrite",
            "keep_normalized",
            "limit",
        ),
        context="operation=build",
    )
    bello_root = _required_path(command.bello_retargeted_root, "bello_retargeted_root")
    g1_root = _required_path(command.g1_retargeted_root, "g1_retargeted_root")
    from bello_mujoco.motion.sonic_dataset import (
        build_sonic_dataset,
        load_sonic_calibration_config,
        load_sonic_sanitation_config,
    )
    from bello_mujoco.motion.twist_dataset import load_quality_config

    def progress(stage: str, current: int, total: int) -> None:
        print(f"SONIC dataset {stage}: {current}/{total}", file=sys.stderr)

    index = build_sonic_dataset(
        manifest,
        bello_root,
        g1_root,
        output_root,
        quality=load_quality_config(command.quality_config),
        calibration=load_sonic_calibration_config(command.calibration_config),
        sanitation=load_sonic_sanitation_config(command.sanitation_config),
        progress=progress,
    )
    print(json.dumps(index, indent=2, sort_keys=True))


@hydra.main(
    version_base=None,
    config_path=config_path("commands"),
    config_name="twist_data",
)
def _twist_data_hydra(config: DictConfig) -> None:
    command = command_config(TwistDataCommandConfig, config)
    manifest_path = _required_path(command.manifest, "manifest")
    from bello_mujoco.motion.twist_dataset import (
        assign_splits,
        build_packed_dataset,
        load_quality_config,
        load_source_curation_config,
        load_source_manifest,
        portable_manifest_fingerprint,
        scan_amass_manifest,
    )

    if command.operation == "scan_amass":
        _reject_explicit_fields(
            (
                "body_model_root",
                "retargeted_root",
                "output_root",
                "robot_model_path",
                "gmr_root",
                "quality_config",
            ),
            context="operation=scan_amass",
        )
        source_root = _required_path(command.source_root, "source_root")
        if command.dataset_release is None:
            raise ValueError("dataset_release is required for operation=scan_amass")
        if command.license_id is None:
            raise ValueError("license_id is required for operation=scan_amass")

        def progress(current: int, total: int, path: Path) -> None:
            if current == 1 or current % 250 == 0 or current == total:
                print(f"scanned {current}/{total}: {path}", file=sys.stderr)

        manifest = scan_amass_manifest(
            source_root,
            manifest_path,
            dataset_release=command.dataset_release,
            license_id=command.license_id,
            split_seed=command.split_seed,
            sampling_weight=command.sampling_weight,
            curation=load_source_curation_config(command.curation_config),
            overwrite=command.overwrite,
            progress=progress,
        )
        print(
            json.dumps(
                {
                    "manifest": str(manifest_path.resolve()),
                    "sequences": len(manifest.sequences),
                    "subdatasets": sorted(
                        {item.subdataset for item in manifest.sequences}
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    _reject_explicit_fields(
        (
            "source_root",
            "dataset_release",
            "license_id",
            "split_seed",
            "sampling_weight",
            "overwrite",
            "curation_config",
        ),
        context=f"operation={command.operation}",
    )

    if command.operation == "validate":
        _reject_explicit_fields(
            ("retargeted_root", "output_root"),
            context="operation=validate",
        )
        manifest = load_source_manifest(
            manifest_path,
            body_model_root=command.body_model_root,
            verify_files=True,
        )
        assignments = assign_splits(manifest)
        print(
            json.dumps(
                {
                    "sequences": len(assignments),
                    "train": sum(item.split == "train" for item in assignments),
                    "evaluation": sum(
                        item.split == "evaluation" for item in assignments
                    ),
                    "portable_manifest_fingerprint": (
                        portable_manifest_fingerprint(assignments)
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    retargeted_root = _required_path(command.retargeted_root, "retargeted_root")
    output_root = _required_path(command.output_root, "output_root")
    index = build_packed_dataset(
        manifest_path,
        retargeted_root,
        output_root,
        body_model_root=command.body_model_root,
        quality=load_quality_config(command.quality_config),
        robot_model_path=command.robot_model_path,
        gmr_root=command.gmr_root,
    )
    print(json.dumps(index, indent=2, sort_keys=True))


@hydra.main(
    version_base=None,
    config_path=config_path("commands"),
    config_name="motion",
)
def _motion_hydra(config: DictConfig) -> None:
    command = command_config(MotionCommandConfig, config)
    source = _required_path(command.source, "source")
    if command.operation == "import":
        _reject_explicit_fields(
            ("jobs", "overwrite", "width", "height", "fps", "require_complete"),
            context="operation=import",
        )
        from bello_mujoco.motion.import_gmr import import_gmr_motion

        output = _required_path(command.output, "output")
        print(
            import_gmr_motion(
                source,
                output,
                reject_self_collisions=command.reject_self_collisions,
            )
        )
        return
    if command.operation == "import_dataset":
        _reject_explicit_fields(
            (
                "reject_self_collisions",
                "width",
                "height",
                "fps",
                "require_complete",
            ),
            context="operation=import_dataset",
        )
        from bello_mujoco.motion.import_gmr import import_dataset

        output = _required_path(command.output, "output")
        motions, failures = import_dataset(
            source,
            output,
            jobs=command.jobs,
            overwrite=command.overwrite,
        )
        manifest = {
            "format": "bello-motion-dataset-v1",
            "collision_mode": "box",
            "tracked_body_profile": "beyondmimic",
            "motions": motions,
            "failures": failures,
        }
        output.mkdir(parents=True, exist_ok=True)
        manifest_path = output / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(manifest_path)
        if failures:
            raise RuntimeError(f"{len(failures)} motion(s) failed")
        return
    if command.operation == "render":
        _reject_explicit_fields(
            (
                "jobs",
                "overwrite",
                "reject_self_collisions",
                "require_complete",
            ),
            context="operation=render",
        )
        from bello_mujoco.evaluation.motion import render_reference

        output = _required_path(command.output, "output")
        print(
            render_reference(
                source,
                output,
                width=command.width,
                height=command.height,
                fps=command.fps,
            )
        )
        return
    _reject_explicit_fields(
        (
            "jobs",
            "overwrite",
            "reject_self_collisions",
            "width",
            "height",
            "fps",
        ),
        context="operation=inspect",
    )
    from bello_mujoco.motion.feasibility import replay_motion_reference

    report = replay_motion_reference(source)
    payload = json.dumps(asdict(report), indent=2, sort_keys=True)
    if command.output is not None:
        command.output.parent.mkdir(parents=True, exist_ok=True)
        command.output.write_text(payload + "\n")
    print(payload)
    if command.require_complete and not report.completed:
        raise RuntimeError("motion reference did not complete feasibility replay")


def vast_train_entrypoint() -> None:
    _expand_path_overrides(
        (
            "motion_path=",
            "dataset_path=",
            "teacher_checkpoint_path=",
            "sonic_artifact_path=",
            "resume_path=",
            "warm_start_checkpoint_path=",
        )
    )
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
    if command.experiment in {"twist_teacher", "twist_student", "twist2", "sonic"}:
        if command.dataset_path is None:
            raise ValueError("dataset_path is required for Vast dataset training")
    else:
        if command.dataset_path is not None:
            raise ValueError(
                "dataset_path is only valid for a TWIST, TWIST2, or SONIC experiment"
            )
        _reject_explicit_fields(
            ("dataset_path",), context=f"experiment={command.experiment}"
        )
    if command.experiment == "twist_student":
        if command.teacher_checkpoint_path is None:
            raise ValueError(
                "teacher_checkpoint_path is required for Vast TWIST student training"
            )
    else:
        if command.teacher_checkpoint_path is not None:
            raise ValueError(
                "teacher_checkpoint_path is only valid for experiment=twist_student"
            )
        _reject_explicit_fields(
            ("teacher_checkpoint_path",), context=f"experiment={command.experiment}"
        )
    if command.experiment == "sonic":
        if command.sonic_artifact_path is None:
            raise ValueError("sonic_artifact_path is required for Vast SONIC training")
        if command.resume_path is None and command.allow_git_commit_mismatch:
            raise ValueError(
                "allow_git_commit_mismatch requires resume_path for Vast SONIC training"
            )
    else:
        if command.sonic_artifact_path is not None:
            raise ValueError("sonic_artifact_path is only valid for experiment=sonic")
        if command.resume_path is not None:
            raise ValueError("resume_path is only valid for experiment=sonic")
        _reject_explicit_fields(
            ("sonic_artifact_path", "resume_path"),
            context=f"experiment={command.experiment}",
        )
    if command.experiment != "twist2":
        if command.warm_start_checkpoint_path is not None:
            raise ValueError(
                "warm_start_checkpoint_path is only valid for experiment=twist2"
            )
        _reject_explicit_fields(
            ("warm_start_checkpoint_path",),
            context=f"experiment={command.experiment}",
        )
    if command.experiment not in {"twist_student", "twist2", "sonic"}:
        _reject_explicit_fields(
            ("allow_git_commit_mismatch",),
            context=f"experiment={command.experiment}",
        )
    from bello_mujoco.assets_pipeline import repo_root
    from bello_mujoco.training.vast import (
        VastTrainConfig,
        launch_pipeline,
        safe_exception_message,
    )

    root = repo_root()
    internal = VastTrainConfig(
        experiment=command.experiment,
        motion_path=command.motion_path,
        dataset_path=command.dataset_path,
        teacher_checkpoint_path=command.teacher_checkpoint_path,
        sonic_artifact_path=command.sonic_artifact_path,
        resume_path=command.resume_path,
        warm_start_checkpoint_path=command.warm_start_checkpoint_path,
        allow_git_commit_mismatch=command.allow_git_commit_mismatch,
        total_timesteps=command.total_timesteps,
        num_envs=command.num_envs,
        checkpoint_every_updates=command.checkpoint_every_updates,
        learning_rate=command.learning_rate,
        end_effector_payload_mass_kg=command.end_effector_payload_mass_kg,
        max_price=command.max_price,
        branch=command.branch,
        run_name=command.run_name,
        skip_confirmation=command.skip_confirmation,
        dry_run=command.dry_run,
        preferred_countries=tuple(command.preferred_countries),
        min_upload_mib_per_second=command.min_upload_mib_per_second,
        route_probe_mib=command.route_probe_mib,
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
            asset_config=replace(
                emulator.default_emulator_asset_config(),
                end_effector_payload_mass_kg=command.end_effector_payload_mass_kg,
            ),
        )
    )


def hardware_entrypoint() -> None:
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    _expand_path_overrides(
        (
            "checkpoint_path=",
            "standing_checkpoint_path=",
            "sonic_artifact_path=",
            "motion_path=",
            "policy_log_path=",
            "fastsam_tls_root_certificate_path=",
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
    fastsam_fields = (
        "fastsam_endpoint",
        "fastsam_tls_root_certificate_path",
        "fastsam_tls_server_name_override",
        "fastsam_connect_timeout_seconds",
        "fastsam_rpc_timeout_seconds",
        "fastsam_reference_ready_timeout_seconds",
        "fastsam_reference_qualification_frames",
        "fastsam_min_tracking_confidence",
        "fastsam_min_joint_confidence",
        "fastsam_max_pipeline_latency_ms",
        "fastsam_transport_latency_budget_ms",
        "fastsam_camera_to_world_quaternion_xyzw",
        "fastsam_align_initial_heading",
        "fastsam_align_initial_floor",
    )
    human_model_fields = ("smplx_neutral_model_path", "smplx_human_height_m")
    gemx_fields = (
        "gemx_endpoint",
        "gemx_connect_timeout_seconds",
        "gemx_rpc_timeout_seconds",
        "gemx_reference_ready_timeout_seconds",
        "gemx_reference_qualification_frames",
        "gemx_min_tracking_confidence",
        "gemx_min_joint_confidence",
        "gemx_max_pipeline_latency_ms",
        "gemx_transport_latency_budget_ms",
        "gemx_camera_to_world_quaternion_xyzw",
        "gemx_align_initial_heading",
        "gemx_align_initial_floor",
    )
    sonic_fields = ("sonic_artifact_path", "reference_mode", *gemx_fields)
    locomotion_upper_body_fields = (
        "left_elbow_yaw_offset_radians",
        "right_elbow_yaw_offset_radians",
    )
    if command.task == "sonic":
        _reject_explicit_fields(
            (
                "remote_endpoint",
                "remote_staleness_ms",
                "standing_checkpoint_path",
                "end_effector_payload_mass_kg",
                "action_lpf_cutoff_hz",
                *fastsam_fields,
                *locomotion_upper_body_fields,
            ),
            context="task=sonic",
        )
        if command.standing_checkpoint_path is not None:
            raise ValueError("standing_checkpoint_path is not valid for task=sonic")
        if command.reference_mode == "clip":
            _reject_explicit_fields(
                (*gemx_fields, *human_model_fields),
                context="task=sonic reference_mode=clip",
            )
            if command.motion_path is None:
                raise ValueError(
                    "motion_path is required for task=sonic reference_mode=clip"
                )
            if command.smplx_neutral_model_path is not None:
                raise ValueError(
                    "smplx_neutral_model_path is not valid for SONIC clip playback"
                )
            neutral_model_path = None
        else:
            if command.motion_path is not None:
                raise ValueError(
                    "motion_path is not valid for task=sonic reference_mode=live"
                )
            neutral_model_path = _required_path(
                command.smplx_neutral_model_path,
                "smplx_neutral_model_path",
            )
        from bello_mujoco.hardware.sonic import (
            HardwareSonicConfig,
            run_hardware_sonic,
        )

        base = HardwareSonicConfig()
        run_hardware_sonic(
            checkpoint,
            replace(
                base,
                endpoint=command.endpoint,
                reference_mode=command.reference_mode,
                release_artifact_directory=_required_path(
                    command.sonic_artifact_path, "sonic_artifact_path"
                ),
                motion_path=command.motion_path,
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
                allow_git_commit_mismatch=command.allow_git_commit_mismatch,
                gemx_endpoint=command.gemx_endpoint,
                gemx_connect_timeout_seconds=(command.gemx_connect_timeout_seconds),
                gemx_rpc_timeout_seconds=command.gemx_rpc_timeout_seconds,
                gemx_reference_ready_timeout_seconds=(
                    command.gemx_reference_ready_timeout_seconds
                ),
                gemx_reference_qualification_frames=(
                    command.gemx_reference_qualification_frames
                ),
                gemx_min_tracking_confidence=command.gemx_min_tracking_confidence,
                gemx_min_joint_confidence=command.gemx_min_joint_confidence,
                gemx_max_pipeline_latency_ms=command.gemx_max_pipeline_latency_ms,
                gemx_transport_latency_budget_ms=(
                    command.gemx_transport_latency_budget_ms
                ),
                gemx_camera_to_world_quaternion_xyzw=(
                    command.gemx_camera_to_world_quaternion_xyzw
                ),
                gemx_align_initial_heading=command.gemx_align_initial_heading,
                gemx_align_initial_floor=command.gemx_align_initial_floor,
                smplx_neutral_model_path=neutral_model_path,
                smplx_human_height_m=command.smplx_human_height_m,
            ),
        )
        return
    if command.task == "twist2":
        _reject_explicit_fields(
            (
                "remote_endpoint",
                "remote_staleness_ms",
                "motion_path",
                "standing_checkpoint_path",
                *sonic_fields,
                *locomotion_upper_body_fields,
            ),
            context="task=twist2",
        )
        if (
            command.motion_path is not None
            or command.standing_checkpoint_path is not None
        ):
            raise ValueError(
                "motion_path and standing_checkpoint_path are not valid for task=twist2"
            )
        from bello_mujoco.hardware.twist2 import (
            HardwareTwist2Config,
            run_hardware_twist2,
        )

        base = HardwareTwist2Config()
        run_hardware_twist2(
            checkpoint,
            replace(
                base,
                endpoint=command.endpoint,
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
                allow_git_commit_mismatch=command.allow_git_commit_mismatch,
                fastsam_endpoint=command.fastsam_endpoint,
                fastsam_tls_root_certificate_path=(
                    command.fastsam_tls_root_certificate_path
                ),
                fastsam_tls_server_name_override=(
                    command.fastsam_tls_server_name_override
                ),
                fastsam_connect_timeout_seconds=(
                    command.fastsam_connect_timeout_seconds
                ),
                fastsam_rpc_timeout_seconds=command.fastsam_rpc_timeout_seconds,
                fastsam_reference_ready_timeout_seconds=(
                    command.fastsam_reference_ready_timeout_seconds
                ),
                fastsam_reference_qualification_frames=(
                    command.fastsam_reference_qualification_frames
                ),
                fastsam_min_tracking_confidence=(
                    command.fastsam_min_tracking_confidence
                ),
                fastsam_min_joint_confidence=command.fastsam_min_joint_confidence,
                fastsam_max_pipeline_latency_ms=(
                    command.fastsam_max_pipeline_latency_ms
                ),
                fastsam_transport_latency_budget_ms=(
                    command.fastsam_transport_latency_budget_ms
                ),
                fastsam_camera_to_world_quaternion_xyzw=(
                    command.fastsam_camera_to_world_quaternion_xyzw
                ),
                fastsam_align_initial_heading=command.fastsam_align_initial_heading,
                fastsam_align_initial_floor=command.fastsam_align_initial_floor,
                smplx_neutral_model_path=_required_path(
                    command.smplx_neutral_model_path,
                    "smplx_neutral_model_path",
                ),
                smplx_human_height_m=command.smplx_human_height_m,
            ),
        )
        return
    if command.task == "motion":
        _reject_explicit_fields(
            (
                "remote_endpoint",
                "remote_staleness_ms",
                *fastsam_fields,
                *human_model_fields,
                *sonic_fields,
                *locomotion_upper_body_fields,
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
                motion_path=command.motion_path,
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
                action_lpf_cutoff_hz=_coalesce(
                    command.action_lpf_cutoff_hz,
                    base.action_lpf_cutoff_hz,
                ),
                status_interval_seconds=command.status_interval_seconds,
                policy_log_path=policy_log_path,
                arm=command.arm,
                allow_git_commit_mismatch=command.allow_git_commit_mismatch,
            ),
        )
        return
    if (
        command.motion_path is not None
        or command.standing_checkpoint_path is not None
        or command.end_effector_payload_mass_kg is not None
    ):
        raise ValueError(
            "motion_path, standing_checkpoint_path, and "
            "end_effector_payload_mass_kg are only valid for task=motion"
        )
    _reject_explicit_fields(
        (
            "motion_path",
            "standing_checkpoint_path",
            "end_effector_payload_mass_kg",
            "policy_transition_seconds",
            "max_motion_tilt_radians",
            "action_lpf_cutoff_hz",
            *sonic_fields,
        ),
        context="task=locomotion",
    )
    if command.remote_endpoint is None or not command.remote_endpoint.strip():
        raise ValueError("remote_endpoint is required for task=locomotion")
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
            allow_git_commit_mismatch=command.allow_git_commit_mismatch,
            fastsam_endpoint=command.fastsam_endpoint,
            fastsam_tls_root_certificate_path=(
                command.fastsam_tls_root_certificate_path
            ),
            fastsam_tls_server_name_override=(command.fastsam_tls_server_name_override),
            fastsam_connect_timeout_seconds=(command.fastsam_connect_timeout_seconds),
            fastsam_rpc_timeout_seconds=command.fastsam_rpc_timeout_seconds,
            fastsam_reference_ready_timeout_seconds=(
                command.fastsam_reference_ready_timeout_seconds
            ),
            fastsam_reference_qualification_frames=(
                command.fastsam_reference_qualification_frames
            ),
            fastsam_min_tracking_confidence=(command.fastsam_min_tracking_confidence),
            fastsam_min_joint_confidence=command.fastsam_min_joint_confidence,
            fastsam_max_pipeline_latency_ms=(command.fastsam_max_pipeline_latency_ms),
            fastsam_transport_latency_budget_ms=(
                command.fastsam_transport_latency_budget_ms
            ),
            fastsam_camera_to_world_quaternion_xyzw=(
                command.fastsam_camera_to_world_quaternion_xyzw
            ),
            fastsam_align_initial_heading=command.fastsam_align_initial_heading,
            fastsam_align_initial_floor=command.fastsam_align_initial_floor,
            smplx_neutral_model_path=_required_path(
                command.smplx_neutral_model_path,
                "smplx_neutral_model_path",
            ),
            smplx_human_height_m=command.smplx_human_height_m,
            upper_body_follow=replace(
                base.upper_body_follow,
                left_elbow_yaw_offset_radians=(command.left_elbow_yaw_offset_radians),
                right_elbow_yaw_offset_radians=(command.right_elbow_yaw_offset_radians),
            ),
        ),
    )


def _hardware_policy_log_path(configured: Path | None) -> Path:
    if configured is not None:
        return configured.expanduser()
    from hydra.core.hydra_config import HydraConfig

    return Path(HydraConfig.get().runtime.output_dir) / "policy_steps.jsonl"


def _ppo_overrides(config, command: TrainCommandConfig):
    values = {}
    for name in ("seed", "total_timesteps", "num_envs", "learning_rate"):
        value = getattr(command, name)
        if value is not None:
            values[name] = value
    return replace(config, **values)


def _checkpoint_overrides(config, command: TrainCommandConfig):
    values = {
        "allow_git_commit_mismatch": command.allow_git_commit_mismatch,
    }
    if command.checkpoint_enabled is not None:
        values["enabled"] = command.checkpoint_enabled
    if command.resume_path is not None:
        values["resume_path"] = command.resume_path.expanduser()
    if command.warm_start_path is not None:
        values["warm_start_path"] = command.warm_start_path.expanduser()
    if command.checkpoint_directory is not None:
        values["directory"] = command.checkpoint_directory.expanduser()
    if command.checkpoint_name is not None:
        values["name"] = command.checkpoint_name
    if command.checkpoint_every_updates is not None:
        values["every_updates"] = command.checkpoint_every_updates
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
