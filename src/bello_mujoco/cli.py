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
    motion_path: Path | None
    end_effector_payload_mass_kg: float | None
    resume_path: Path | None
    allow_git_commit_mismatch: bool
    checkpoint_directory: Path | None
    checkpoint_name: str | None
    wandb_enabled: bool | None
    run_name: str | None
    video_enabled: bool | None


@dataclass(frozen=True)
class EvalCommandConfig:
    task: Literal["locomotion", "motion"]
    checkpoint_path: Path | None
    allow_git_commit_mismatch: bool
    end_effector_payload_mass_kg: float | None
    output_path: Path
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
class VastTrainCommandConfig:
    experiment: ExperimentName
    motion_path: Path | None
    end_effector_payload_mass_kg: float | None
    max_price: float | None
    branch: str | None
    run_name: str | None
    duration_hours: float
    skip_confirmation: bool
    dry_run: bool


@dataclass(frozen=True)
class EmulatorCommandConfig:
    bus_interfaces: tuple[str, ...]
    command_latency_seconds: float


@dataclass(frozen=True)
class HardwareCommandConfig:
    task: Literal["locomotion", "motion"]
    checkpoint_path: Path | None
    standing_checkpoint_path: Path | None
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
    status_interval_seconds: float | None
    policy_log_path: Path | None
    arm: bool


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
    _train_hydra()


@hydra.main(
    version_base=None,
    config_path=config_path("commands"),
    config_name="train",
)
def _train_hydra(config: DictConfig) -> None:
    command = command_config(TrainCommandConfig, config)
    if command.experiment == "motion":
        from bello_mujoco.training.motion import (
            run_training as run_motion_training,
        )

        run_config = motion_training_profile(command.motion_path)
        run_config = replace(
            run_config,
            config=_ppo_overrides(run_config.config, command),
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
        ("motion_path", "end_effector_payload_mass_kg"),
        context=f"experiment={command.experiment}",
    )
    from bello_mujoco.training.locomotion import (
        run_training as run_locomotion_training,
    )

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
    command = command_config(EvalCommandConfig, config)
    checkpoint = _required_path(command.checkpoint_path, "checkpoint_path")
    if command.task == "motion":
        _reject_explicit_fields(
            ("command", "stochastic_actions", "action_seed", "gamepad_enabled"),
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
        ("output_path", "end_effector_payload_mass_kg"),
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
        end_effector_payload_mass_kg=command.end_effector_payload_mass_kg,
        max_price=command.max_price,
        branch=command.branch,
        run_name=command.run_name,
        skip_confirmation=command.skip_confirmation,
        dry_run=command.dry_run,
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
        )
    )


def hardware_entrypoint() -> None:
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    _expand_path_overrides(
        (
            "checkpoint_path=",
            "standing_checkpoint_path=",
            "motion_path=",
            "policy_log_path=",
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
    if command.task == "motion":
        _reject_explicit_fields(
            ("remote_endpoint", "remote_staleness_ms"), context="task=motion"
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
        ),
    )


def _hardware_policy_log_path(configured: Path | None) -> Path:
    if configured is not None:
        return configured.expanduser()
    from hydra.core.hydra_config import HydraConfig

    return Path(HydraConfig.get().runtime.output_dir) / "policy_steps.jsonl"


def _ppo_overrides(config, command: TrainCommandConfig):
    values = {}
    for name in ("seed", "total_timesteps", "num_envs"):
        value = getattr(command, name)
        if value is not None:
            values[name] = value
    return replace(config, **values)


def _checkpoint_overrides(config, command: TrainCommandConfig):
    values = {
        "allow_git_commit_mismatch": command.allow_git_commit_mismatch,
    }
    if command.resume_path is not None:
        values["resume_path"] = command.resume_path.expanduser()
    if command.checkpoint_directory is not None:
        values["directory"] = command.checkpoint_directory.expanduser()
    if command.checkpoint_name is not None:
        values["name"] = command.checkpoint_name
    return replace(config, **values)


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
