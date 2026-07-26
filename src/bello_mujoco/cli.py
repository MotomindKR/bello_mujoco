"""Small Hydra command surface for supported Bello workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import sys
from typing import Literal

import hydra
from omegaconf import DictConfig

from bello_mujoco.assets_pipeline import repo_root
from bello_mujoco.assets_pipeline.prepare import (
    canonical_physical_asset_config,
    prepare_asset,
)
from bello_mujoco.configuration import (
    ExperimentName,
    config_path,
    locomotion_training_profile,
    motion_training_profile,
    profile_config_path,
    structured_config,
)
from bello_mujoco.evaluation.locomotion import (
    EvalPolicyConfig,
    evaluate_locomotion_checkpoint,
)
from bello_mujoco.evaluation.motion import (
    render_checkpoint_with_stats,
    render_reference,
)
from bello_mujoco.hardware.locomotion import (
    HardwareLocomotionConfig,
    run_hardware_locomotion,
)
from bello_mujoco.hardware.motion import HardwareMotionConfig, run_hardware_motion
from bello_mujoco.motion.feasibility import replay_motion_reference
from bello_mujoco.motion.import_gmr import import_dataset, import_gmr_motion
from bello_mujoco.robstride import runner as robstride_runner
from bello_mujoco.training.locomotion import run_training as run_locomotion_training
from bello_mujoco.training.motion import run_training as run_motion_training
from bello_mujoco.training.tracking import WandbConfig
from bello_mujoco.training.vast import (
    VastTrainConfig,
    launch_pipeline,
    safe_exception_message,
)


@dataclass(frozen=True)
class PrepareAssetsCommandConfig:
    profile: Literal["physical", "visualizer"] = "physical"
    output_root: Path | None = None


@dataclass(frozen=True)
class TrainCommandConfig:
    experiment: ExperimentName = "smoke"
    seed: int | None = None
    total_timesteps: int | None = None
    num_envs: int | None = None
    motion_path: Path | None = None
    resume_path: Path | None = None
    checkpoint_directory: Path | None = None
    checkpoint_name: str | None = None
    wandb_enabled: bool | None = None
    run_name: str | None = None
    video_enabled: bool | None = None


@dataclass(frozen=True)
class EvalCommandConfig:
    task: Literal["locomotion", "motion"] = "locomotion"
    checkpoint_path: Path | None = None
    output_path: Path = Path("outputs/evaluation/motion.mp4")
    command: tuple[float, float, float] = (0.0, 0.0, 0.0)
    steps: int | None = None
    stop_on_done: bool = False
    stochastic_actions: bool = False
    action_seed: int = 0
    gamepad_enabled: bool = True


@dataclass(frozen=True)
class MotionCommandConfig:
    operation: Literal["import", "import_dataset", "inspect", "render"] = "inspect"
    source: Path | None = None
    output: Path | None = None
    jobs: int = min(8, os.cpu_count() or 1)
    overwrite: bool = False
    reject_self_collisions: bool = True
    width: int = 640
    height: int = 480
    fps: int = 30
    require_complete: bool = False


@dataclass(frozen=True)
class VastTrainCommandConfig:
    experiment: ExperimentName = "hardware_locomotion"
    motion_path: Path | None = None
    max_price: float | None = None
    branch: str | None = None
    run_name: str | None = None
    duration_hours: float = 6.0
    yes: bool = False
    dry_run: bool = False
    overrides: tuple[str, ...] = ()


@dataclass(frozen=True)
class RobstrideCommandConfig:
    mode: Literal["hil", "prescribed_gains"] = "hil"
    bus_interfaces: tuple[str, ...] = ("can0=vcan0", "can1=vcan1")
    headless: bool = False
    real_time: bool = True
    max_steps: int | None = None
    torso_hold: bool = True
    release_after_seconds: float | None = None


@dataclass(frozen=True)
class HardwareCommandConfig:
    task: Literal["locomotion", "motion"] = "locomotion"
    checkpoint_path: Path | None = None
    endpoint: str = "http://127.0.0.1:50051"
    command: tuple[float, float, float] = (0.0, 0.0, 0.0)
    motion_path: Path | None = None
    duration_seconds: float | None = 10.0
    ramp_seconds: float | None = None
    kp_scale: float = 1.0
    kd_scale: float = 1.0
    gamepad_enabled: bool = True
    wait_for_start_button: bool = True
    arm: bool = False
    operator_confirmation: str = ""


def prepare_assets_entrypoint() -> None:
    _prepare_assets_hydra()


@hydra.main(
    version_base=None,
    config_path=config_path("commands"),
    config_name="prepare_assets",
)
def _prepare_assets_hydra(config: DictConfig) -> None:
    command = structured_config(PrepareAssetsCommandConfig, config)
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
    _train_hydra()


@hydra.main(
    version_base=None,
    config_path=config_path("commands"),
    config_name="train",
)
def _train_hydra(config: DictConfig) -> None:
    command = structured_config(TrainCommandConfig, config)
    if command.experiment == "motion":
        run_config = motion_training_profile(command.motion_path)
        run_config = replace(
            run_config,
            config=_ppo_overrides(run_config.config, command),
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
    run_config = locomotion_training_profile(command.experiment)
    run_config = replace(
        run_config,
        config=_ppo_overrides(run_config.config, command),
        checkpoint_config=_checkpoint_overrides(run_config.checkpoint_config, command),
        wandb_config=_wandb_overrides(run_config.wandb_config, command),
        video_config=replace(
            run_config.video_config,
            enabled=_coalesce(command.video_enabled, run_config.video_config.enabled),
        ),
    )
    run_locomotion_training(run_config)


def eval_entrypoint() -> None:
    _expand_path_overrides(("checkpoint_path=", "output_path="))
    _eval_hydra()


@hydra.main(
    version_base=None,
    config_path=config_path("commands"),
    config_name="eval",
)
def _eval_hydra(config: DictConfig) -> None:
    command = structured_config(EvalCommandConfig, config)
    checkpoint = _required_path(command.checkpoint_path, "checkpoint_path")
    if command.task == "motion":
        stats = render_checkpoint_with_stats(
            checkpoint,
            command.output_path.expanduser(),
            steps=command.steps,
            stop_on_terminal=command.stop_on_done,
        )
        print(json.dumps(stats, indent=2, sort_keys=True))
        return
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


def motion_entrypoint() -> None:
    _expand_path_overrides(("source=", "output="))
    _motion_hydra()


@hydra.main(
    version_base=None,
    config_path=config_path("commands"),
    config_name="motion",
)
def _motion_hydra(config: DictConfig) -> None:
    command = structured_config(MotionCommandConfig, config)
    source = _required_path(command.source, "source")
    if command.operation == "import":
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
    report = replay_motion_reference(source)
    payload = json.dumps(asdict(report), indent=2, sort_keys=True)
    if command.output is not None:
        command.output.parent.mkdir(parents=True, exist_ok=True)
        command.output.write_text(payload + "\n")
    print(payload)
    if command.require_complete and not report.completed:
        raise RuntimeError("motion reference did not complete feasibility replay")


def vast_train_entrypoint() -> None:
    _vast_train_hydra()


@hydra.main(
    version_base=None,
    config_path=config_path("commands"),
    config_name="vast_train",
)
def _vast_train_hydra(config: DictConfig) -> None:
    command = structured_config(VastTrainCommandConfig, config)
    if command.duration_hours <= 0:
        raise ValueError("duration_hours must be positive")
    if command.experiment == "motion":
        if command.motion_path is None:
            raise ValueError(
                "motion_path must select a repository-tracked clip for Vast motion "
                "training"
            )
        motion_overrides = (f"motion_path={command.motion_path.as_posix()}",)
    else:
        if command.motion_path is not None:
            raise ValueError("motion_path is only valid for the motion experiment")
        motion_overrides = ()
    reserved_overrides = {
        override.partition("=")[0].lstrip("+") for override in command.overrides
    } & {"experiment", "motion_path"}
    if reserved_overrides:
        names = ", ".join(sorted(reserved_overrides))
        raise ValueError(f"use the dedicated Vast command field(s) instead: {names}")
    root = repo_root()
    profile_path = profile_config_path(command.experiment).resolve()
    try:
        relative_profile = profile_path.relative_to(root)
    except ValueError as error:
        raise ValueError("training profile must be inside the repository") from error
    internal = VastTrainConfig(
        train_config=relative_profile,
        task="motion-tracking" if command.experiment == "motion" else "locomotion",
        max_price=command.max_price,
        branch=command.branch,
        run_name=command.run_name,
        yes=command.yes,
        dry_run=command.dry_run,
        job_timeout_seconds=round(command.duration_hours * 3600),
        progress_timeout_seconds=min(
            1800, max(60, round(command.duration_hours * 3600))
        ),
        hydra_overrides=(
            f"experiment={command.experiment}",
            *motion_overrides,
            *command.overrides,
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


def robstride_entrypoint() -> None:
    _robstride_hydra()


@hydra.main(
    version_base=None,
    config_path=config_path("commands"),
    config_name="robstride",
)
def _robstride_hydra(config: DictConfig) -> None:
    command = structured_config(RobstrideCommandConfig, config)
    runner = robstride_runner.RobstrideRunnerConfig(
        bus_interfaces=command.bus_interfaces,
        asset_config=canonical_physical_asset_config(imu_enabled=True),
        enforce_torque_limits=True,
        headless=command.headless,
        real_time=command.real_time,
        max_steps=command.max_steps,
        torso_hold=command.torso_hold,
        release_after_seconds=command.release_after_seconds,
    )
    if command.mode == "prescribed_gains":
        robstride_runner.run_prescribed_gains(runner)
    else:
        robstride_runner.main(runner)


def hardware_entrypoint() -> None:
    _expand_path_overrides(("checkpoint_path=", "motion_path="))
    _hardware_hydra()


@hydra.main(
    version_base=None,
    config_path=config_path("commands"),
    config_name="hardware",
)
def _hardware_hydra(config: DictConfig) -> None:
    command = structured_config(HardwareCommandConfig, config)
    checkpoint = _required_path(command.checkpoint_path, "checkpoint_path")
    if command.task == "motion":
        base = HardwareMotionConfig()
        run_hardware_motion(
            checkpoint,
            replace(
                base,
                endpoint=command.endpoint,
                motion_path=command.motion_path,
                max_duration_seconds=command.duration_seconds,
                ramp_seconds=_coalesce(command.ramp_seconds, base.ramp_seconds),
                kp_scale=command.kp_scale,
                kd_scale=command.kd_scale,
                arm=command.arm,
                operator_confirmation=command.operator_confirmation,
            ),
        )
        return
    base = HardwareLocomotionConfig()
    run_hardware_locomotion(
        checkpoint,
        replace(
            base,
            endpoint=command.endpoint,
            command=command.command,
            max_duration_seconds=command.duration_seconds,
            ramp_seconds=_coalesce(command.ramp_seconds, base.ramp_seconds),
            kp_scale=command.kp_scale,
            kd_scale=command.kd_scale,
            gamepad=replace(base.gamepad, enabled=command.gamepad_enabled),
            wait_for_start_button=command.wait_for_start_button,
            arm=command.arm,
            operator_confirmation=command.operator_confirmation,
        ),
    )


def _ppo_overrides(config, command: TrainCommandConfig):
    values = {}
    for name in ("seed", "total_timesteps", "num_envs"):
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


def _wandb_overrides(config: WandbConfig, command: TrainCommandConfig) -> WandbConfig:
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
