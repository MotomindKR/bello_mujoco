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

from bello_mujoco.hardware.pico_grpc import PicoGrpcConfig
from bello_mujoco.training.vast import VastOfferConfig

from bello_mujoco.configuration import (
    ExperimentName,
    is_motion_experiment,
    is_sonic_transfer_experiment,
    is_supermimic_experiment,
    command_config,
    config_path,
    locomotion_training_profile,
    motion_training_profile,
    sonic_transfer_training_profile,
    supermimic_training_profile,
)

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
    motion_dataset: Path | None
    motion_dataset_artifact: str | None
    evaluation_clips: tuple[str, ...]
    pretrained_checkpoint: Path | None
    pretrained_revision: str | None
    end_effector_payload_mass_kg: float | None
    resume_path: Path | None
    checkpoint_directory: Path | None
    checkpoint_name: str | None
    wandb_enabled: bool | None
    run_name: str | None
    video_enabled: bool | None


@dataclass(frozen=True)
class EvalCommandConfig:
    task: Literal["locomotion", "motion", "sonic_transfer", "supermimic"]
    checkpoint_path: Path | None
    motion_dataset: Path | None
    evaluation_clips: tuple[str, ...]
    end_effector_payload_mass_kg: float | None
    output_path: Path
    command: tuple[float, float, float]
    torso_command: tuple[float, float, float]
    steps: int | None
    stop_on_done: bool
    stochastic_actions: bool
    action_seed: int
    gamepad_enabled: bool


@dataclass(frozen=True)
class VastTrainCommandConfig:
    experiment: ExperimentName
    motion_path: Path | None
    motion_dataset_artifact: str | None
    pretrained_revision: str | None
    evaluation_clips: tuple[str, ...]
    total_timesteps: int | None
    num_envs: int | None
    offer: VastOfferConfig
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
class HandsCommandConfig:
    listen: str
    lease_timeout_seconds: float
    protocol: Literal["linker_o6", "linker_l6"]
    left_can_interface: str
    right_can_interface: str
    left_hand_enabled: bool
    right_hand_enabled: bool
    can_probe_timeout_seconds: float
    can_feedback_poll_interval_seconds: float
    can_feedback_timeout_seconds: float
    can_tx_spacing_seconds: float
    xrobot: PicoGrpcConfig
    max_reference_age_ms: int
    control_rate_hz: float
    smoothing_time_constant_ms: float
    max_raw_rate_per_second: float
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
    if is_supermimic_experiment(command.experiment):
        _reject_explicit_fields(
            (
                "motion_path",
                "pretrained_checkpoint",
                "pretrained_revision",
                "end_effector_payload_mass_kg",
                "video_enabled",
            ),
            context="SuperMimic training",
        )
        from bello_mujoco.training.supermimic import run_training

        run_config = supermimic_training_profile(
            _required_path(command.motion_dataset, "motion_dataset"),
            name=command.experiment,
        )
        learning = _ppo_overrides(run_config.config, command)
        run_config = replace(
            run_config,
            config=learning,
            dataset=replace(
                run_config.dataset,
                artifact=command.motion_dataset_artifact,
                evaluation_clips=command.evaluation_clips,
            ),
            environment=replace(
                run_config.environment,
                randomization=replace(
                    run_config.environment.randomization, seed=learning.seed
                ),
            ),
            warp=_warp_overrides(run_config.warp, command),
            checkpoint_config=_checkpoint_overrides(
                run_config.checkpoint_config, command
            ),
            wandb_config=_wandb_overrides(run_config.wandb_config, command),
        )
        run_training(run_config)
        return
    if is_sonic_transfer_experiment(command.experiment):
        dataset = _required_path(command.motion_dataset, "motion_dataset")
        pretrained = _required_path(
            command.pretrained_checkpoint, "pretrained_checkpoint"
        )
        if command.pretrained_revision is None:
            raise ValueError("pretrained_revision is required for sonic_transfer")
        if command.motion_dataset_artifact is None:
            raise ValueError("motion_dataset_artifact is required for sonic_transfer")
        from bello_mujoco.training.sonic_transfer import run_training

        run_config = sonic_transfer_training_profile(
            dataset, pretrained, command.pretrained_revision
        )
        run_config = replace(
            run_config,
            dataset=replace(
                run_config.dataset,
                artifact=command.motion_dataset_artifact,
                evaluation_clips=command.evaluation_clips,
            ),
            config=_ppo_overrides(run_config.config, command),
            warp=_warp_overrides(run_config.warp, command),
            checkpoint_config=_checkpoint_overrides(
                run_config.checkpoint_config, command
            ),
            wandb_config=_wandb_overrides(run_config.wandb_config, command),
        )
        run_training(run_config)
        return
    if is_motion_experiment(command.experiment):
        if any(
            value is not None
            for value in (
                command.motion_dataset,
                command.motion_dataset_artifact,
                command.evaluation_clips or None,
                command.pretrained_checkpoint,
                command.pretrained_revision,
            )
        ):
            raise ValueError(
                "motion_dataset, motion_dataset_artifact, pretrained_checkpoint, "
                "and pretrained_revision are only valid for sonic_transfer"
            )
        if (
            command.experiment == "bello_mini_motion"
            and command.motion_path is None
            and command.resume_path is None
        ):
            raise ValueError("Bello Mini motion requires motion_path or resume_path")
        from bello_mujoco.training.motion import (
            run_training as run_motion_training,
        )

        run_config = motion_training_profile(
            command.motion_path, name=command.experiment
        )
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
        or command.motion_dataset is not None
        or command.motion_dataset_artifact is not None
        or bool(command.evaluation_clips)
        or command.pretrained_checkpoint is not None
        or command.pretrained_revision is not None
        or command.end_effector_payload_mass_kg is not None
    ):
        raise ValueError(
            "motion and SONIC inputs are only valid for their matching experiments"
        )
    _reject_explicit_fields(
        (
            "motion_path",
            "motion_dataset",
            "motion_dataset_artifact",
            "evaluation_clips",
            "pretrained_checkpoint",
            "pretrained_revision",
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
    _expand_path_overrides(("checkpoint_path=", "motion_dataset=", "output_path="))
    _eval_hydra()


def export_entrypoint() -> None:
    os.environ["JAX_PLATFORMS"] = "cpu"
    _expand_path_overrides(("checkpoint_path=", "motion_dataset=", "output_path="))
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
    if command.task == "supermimic":
        _reject_explicit_fields(
            (
                "command",
                "torso_command",
                "stochastic_actions",
                "action_seed",
                "gamepad_enabled",
                "end_effector_payload_mass_kg",
            ),
            context="task=supermimic",
        )
        from bello_mujoco.evaluation.supermimic import render_supermimic_checkpoint

        stats = render_supermimic_checkpoint(
            checkpoint,
            _required_path(command.motion_dataset, "motion_dataset"),
            command.evaluation_clips,
            command.output_path,
            steps=command.steps,
            stop_on_terminal=command.stop_on_done,
        )
        print(json.dumps(stats, indent=2, sort_keys=True))
        return
    if command.task == "sonic_transfer":
        _reject_explicit_fields(
            (
                "command",
                "torso_command",
                "stochastic_actions",
                "action_seed",
                "gamepad_enabled",
                "end_effector_payload_mass_kg",
            ),
            context="task=sonic_transfer",
        )
        dataset = _required_path(command.motion_dataset, "motion_dataset")
        from bello_mujoco.evaluation.sonic_transfer import render_sonic_checkpoint

        stats = render_sonic_checkpoint(
            checkpoint,
            dataset,
            command.evaluation_clips,
            command.output_path,
            steps=command.steps,
            stop_on_terminal=command.stop_on_done,
        )
        print(json.dumps(stats, indent=2, sort_keys=True))
        return
    if command.task == "motion":
        _reject_explicit_fields(
            (
                "command",
                "torso_command",
                "stochastic_actions",
                "action_seed",
                "gamepad_enabled",
                "motion_dataset",
                "evaluation_clips",
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
        (
            "output_path",
            "end_effector_payload_mass_kg",
            "motion_dataset",
            "evaluation_clips",
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
            torso_command=command.torso_command,
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
    if is_motion_experiment(command.experiment):
        if command.motion_path is None:
            raise ValueError(
                "motion_path must select a local .npz clip for Vast motion training"
            )
    elif is_supermimic_experiment(command.experiment):
        if command.motion_dataset_artifact is None:
            raise ValueError("motion_dataset_artifact is required for Vast SuperMimic")
        _reject_explicit_fields(
            ("motion_path", "pretrained_revision", "end_effector_payload_mass_kg"),
            context="Vast SuperMimic training",
        )
    elif is_sonic_transfer_experiment(command.experiment):
        if command.motion_dataset_artifact is None:
            raise ValueError(
                "motion_dataset_artifact is required for Vast sonic_transfer"
            )
        if command.pretrained_revision is None:
            raise ValueError("pretrained_revision is required for Vast sonic_transfer")
        if command.motion_path is not None:
            raise ValueError(
                "sonic_transfer consumes a dataset artifact, not motion_path"
            )
    else:
        if (
            command.motion_path is not None
            or command.motion_dataset_artifact is not None
            or command.pretrained_revision is not None
            or command.end_effector_payload_mass_kg is not None
        ):
            raise ValueError(
                "motion and SONIC inputs are only valid for their matching experiments"
            )
        _reject_explicit_fields(
            (
                "motion_path",
                "motion_dataset_artifact",
                "pretrained_revision",
                "end_effector_payload_mass_kg",
            ),
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
        motion_dataset_artifact=command.motion_dataset_artifact,
        pretrained_revision=command.pretrained_revision,
        evaluation_clips=command.evaluation_clips,
        total_timesteps=command.total_timesteps,
        num_envs=command.num_envs,
        offer=command.offer,
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


def reference_entrypoint() -> None:
    os.environ["JAX_PLATFORMS"] = "cpu"
    _expand_path_overrides(("xrobot.tls_root_certificate_path=",))
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
    _expand_path_overrides(("xrobot.tls_root_certificate_path=",))
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
    from bello_mujoco.hardware.hand_server import HandServerConfig, run_hand_server
    from bello_mujoco.hardware.xrobot_hands import (
        HandGeometryConfig,
        HandRawCalibration,
        XRobotHandsConfig,
        run_xrobot_hands,
    )

    can_config = LinkerHandsCanConfig(
        left_interface=command.left_can_interface,
        right_interface=command.right_can_interface,
        model="O6" if command.protocol == "linker_o6" else "L6",
        left_enabled=command.left_hand_enabled,
        right_enabled=command.right_hand_enabled,
        probe_timeout_seconds=command.can_probe_timeout_seconds,
        feedback_poll_interval_seconds=(command.can_feedback_poll_interval_seconds),
        feedback_timeout_seconds=command.can_feedback_timeout_seconds,
        tx_spacing_seconds=command.can_tx_spacing_seconds,
    )
    hands_config = XRobotHandsConfig(
        xrobot=command.xrobot,
        max_reference_age_ms=command.max_reference_age_ms,
        control_rate_hz=command.control_rate_hz,
        smoothing_time_constant_ms=command.smoothing_time_constant_ms,
        max_raw_rate_per_second=command.max_raw_rate_per_second,
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

    def transport_factory(_config: XRobotHandsConfig) -> LinkerHandsCanTransport:
        return LinkerHandsCanTransport(can_config, require_initial_feedback=False)

    server_config = HandServerConfig(
        listen=command.listen,
        lease_timeout_seconds=command.lease_timeout_seconds,
    )
    if command.arm:
        run_hand_server(hands_config, server_config, transport_factory)
    else:
        run_xrobot_hands(hands_config, transport_factory)
    if not command.arm:
        print(
            "XRoboToolkit hand preflight passed; SocketCAN was not opened and no hand "
            "command was sent."
        )


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


def barbell_eval_entrypoint() -> None:
    os.environ["JAX_PLATFORMS"] = "cpu"
    _barbell_eval_hydra()


@hydra.main(
    version_base=None, config_path=config_path("commands"), config_name="barbell_eval"
)
def _barbell_eval_hydra(config: DictConfig) -> None:
    from bello_mujoco.evaluation.barbell import (
        BarbellEvaluationConfig,
        run_barbell_evaluation,
    )

    result = run_barbell_evaluation(command_config(BarbellEvaluationConfig, config))
    print(json.dumps(result, indent=2))
