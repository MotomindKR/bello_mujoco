"""Hydra configuration decoding and canonical training profiles."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

from omegaconf import DictConfig, OmegaConf

from bello_mujoco.models import project_data_path

if TYPE_CHECKING:
    from bello_mujoco.training.locomotion import TrainingRunConfig
    from bello_mujoco.training.motion import MotionTrainingRunConfig

T = TypeVar("T")
ExperimentName = Literal[
    "smoke",
    "bello_locomotion",
    "bello_mini_locomotion",
    "bello_mini_standing",
    "motion",
    "bello_mini_motion",
    "bello_mini_sonic_transfer",
    "bello_mini_supermimic",
    "bello_mini_supermimic_smoke",
    "bello_mini_supermimic_pilot",
    "bello_mini_supermimic_amass",
    "bello_mini_supermimic_lafan",
]
CONFIG_ROOT = project_data_path("configs")
_PROFILE_CONFIG_PATHS: dict[ExperimentName, Path] = {
    "smoke": Path("profiles/train/smoke.yaml"),
    "bello_locomotion": Path("profiles/train/bello_locomotion.yaml"),
    "bello_mini_locomotion": Path("profiles/train/bello_mini_locomotion.yaml"),
    "bello_mini_standing": Path("profiles/train/bello_mini_standing.yaml"),
    "motion": Path("profiles/train/motion.yaml"),
    "bello_mini_motion": Path("profiles/train/bello_mini_motion.yaml"),
    "bello_mini_sonic_transfer": Path("profiles/train/bello_mini_sonic_transfer.yaml"),
    "bello_mini_supermimic": Path("profiles/train/bello_mini_supermimic.yaml"),
    "bello_mini_supermimic_smoke": Path(
        "profiles/train/bello_mini_supermimic_smoke.yaml"
    ),
    "bello_mini_supermimic_pilot": Path(
        "profiles/train/bello_mini_supermimic_pilot.yaml"
    ),
    "bello_mini_supermimic_amass": Path(
        "profiles/train/bello_mini_supermimic_amass.yaml"
    ),
    "bello_mini_supermimic_lafan": Path(
        "profiles/train/bello_mini_supermimic_lafan.yaml"
    ),
}


def is_motion_experiment(name: ExperimentName) -> bool:
    return name in ("motion", "bello_mini_motion")


def is_sonic_transfer_experiment(name: ExperimentName) -> bool:
    return name == "bello_mini_sonic_transfer"


def is_supermimic_experiment(name: ExperimentName) -> bool:
    return name in (
        "bello_mini_supermimic",
        "bello_mini_supermimic_smoke",
        "bello_mini_supermimic_pilot",
        "bello_mini_supermimic_amass",
        "bello_mini_supermimic_lafan",
    )


def dataclass_from_dict(
    cls,
    data: dict[str, Any],
    *,
    strict: bool = False,
    require_all_fields: bool = False,
    path: str | None = None,
):
    if not isinstance(data, dict):
        raise TypeError(f"{path or cls.__name__} must be a mapping")
    field_names = {field.name for field in fields(cls)}
    if strict:
        unknown = sorted(set(data) - field_names)
        if unknown:
            location = path or cls.__name__
            names = ", ".join(unknown)
            raise ValueError(f"unknown config key(s) for {location}: {names}")
    if require_all_fields:
        missing = sorted(field_names - set(data))
        if missing:
            location = path or cls.__name__
            names = ", ".join(missing)
            raise ValueError(f"missing required config key(s) for {location}: {names}")
    kwargs = {}
    type_hints = get_type_hints(cls)
    for field in fields(cls):
        if field.name not in data:
            continue
        field_path = field.name if path is None else f"{path}.{field.name}"
        kwargs[field.name] = coerce_value(
            type_hints.get(field.name, field.type),
            data[field.name],
            strict=strict,
            require_all_fields=require_all_fields,
            path=field_path,
        )
    return cls(**kwargs)


def coerce_value(
    annotation,
    value,
    *,
    strict: bool = False,
    require_all_fields: bool = False,
    path: str | None = None,
):
    if value is None:
        if annotation is Any or type(None) in get_args(annotation):
            return None
        raise TypeError(f"{path or 'value'} must not be null")
    if annotation is Path:
        if not isinstance(value, (str, Path)):
            raise TypeError(f"{path or 'value'} must be a path")
        return Path(value)
    if isinstance(annotation, type) and is_dataclass(annotation):
        return dataclass_from_dict(
            annotation,
            value,
            strict=strict,
            require_all_fields=require_all_fields,
            path=path,
        )
    origin = get_origin(annotation)
    if origin is Literal:
        allowed = get_args(annotation)
        if not any(
            value == choice and type(value) is type(choice) for choice in allowed
        ):
            location = path or "value"
            choices = ", ".join(repr(choice) for choice in allowed)
            raise ValueError(f"{location} must be one of {choices}, got {value!r}")
        return value
    if origin is dict:
        if not isinstance(value, dict):
            raise TypeError(f"{path or 'value'} must be a mapping")
        key_type, value_type = get_args(annotation)
        return {
            coerce_value(
                key_type,
                item_key,
                strict=strict,
                require_all_fields=require_all_fields,
                path=path,
            ): coerce_value(
                value_type,
                item_value,
                strict=strict,
                require_all_fields=require_all_fields,
                path=f"{path}.{item_key}" if path is not None else str(item_key),
            )
            for item_key, item_value in value.items()
        }
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{path or 'value'} must be a list or tuple")
        args = get_args(annotation)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(
                coerce_value(
                    args[0],
                    item,
                    strict=strict,
                    require_all_fields=require_all_fields,
                    path=path,
                )
                for item in value
            )
        if args:
            if len(value) != len(args):
                location = path or "tuple"
                raise ValueError(
                    f"{location} must contain exactly {len(args)} values, got "
                    f"{len(value)}"
                )
            return tuple(
                coerce_value(
                    item_type,
                    item,
                    strict=strict,
                    require_all_fields=require_all_fields,
                    path=path,
                )
                for item_type, item in zip(args, value, strict=False)
            )
        return tuple(value)
    if origin is not None and type(None) in get_args(annotation):
        non_none = [arg for arg in get_args(annotation) if arg is not type(None)]
        if non_none:
            return coerce_value(
                non_none[0],
                value,
                strict=strict,
                require_all_fields=require_all_fields,
                path=path,
            )
    location = path or "value"
    if annotation is bool:
        if type(value) is not bool:
            raise TypeError(f"{location} must be a boolean")
        return value
    if annotation is int:
        if type(value) is not int:
            raise TypeError(f"{location} must be an integer")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{location} must be a number")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise TypeError(f"{location} must be a string")
        return value
    return value


def config_path(group: str) -> str:
    return str(CONFIG_ROOT / group)


def structured_config(cls: type[T], config: DictConfig) -> T:
    data = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if not isinstance(data, dict):
        raise TypeError(f"Hydra config for {cls.__name__} must be a mapping")
    return dataclass_from_dict(cls, data, strict=True)


def command_config(cls: type[T], config: DictConfig) -> T:
    """Decode a complete public command config without dataclass fallbacks."""
    data = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if not isinstance(data, dict):
        raise TypeError(f"Hydra config for {cls.__name__} must be a mapping")
    return dataclass_from_dict(
        cls,
        data,
        strict=True,
        require_all_fields=True,
    )


def locomotion_training_profile(name: ExperimentName) -> TrainingRunConfig:
    from bello_mujoco.training.locomotion import TrainingRunConfig

    profile_layers = {
        "bello_locomotion": ("bello_locomotion",),
        "bello_mini_locomotion": ("bello_mini_locomotion",),
        "bello_mini_standing": ("bello_mini_locomotion", "bello_mini_standing"),
    }
    if name != "smoke" and name not in profile_layers:
        raise ValueError(f"{name!r} is not a locomotion experiment")
    paths = [_config("train/default.yaml")]
    if name in profile_layers:
        paths.append(_config("train/locomotion.yaml"))
        paths.extend(
            _config(f"profiles/train/{layer}.yaml") for layer in profile_layers[name]
        )
    if name == "smoke":
        paths.append(_config("profiles/train/smoke.yaml"))
    return structured_config(TrainingRunConfig, _merged_profile(paths))


def motion_training_profile(
    motion_path: Path | None = None,
    *,
    name: ExperimentName = "motion",
) -> MotionTrainingRunConfig:
    from bello_mujoco.training.motion import MotionTrainingRunConfig

    if not is_motion_experiment(name):
        raise ValueError(f"{name!r} is not a motion experiment")
    paths = [
        _config("train_motion/default.yaml"),
        _config("profiles/train/motion.yaml"),
    ]
    if name != "motion":
        paths.append(CONFIG_ROOT.parent / profile_config_relative_path(name))
    merged = _merged_profile(paths)
    if motion_path is not None:
        resolved = motion_path.expanduser().resolve()
        merged.motion.directory = str(resolved.parent)
        merged.motion.sequence = resolved.stem
    return structured_config(MotionTrainingRunConfig, merged)


def sonic_transfer_training_profile(
    dataset_directory: Path,
    pretrained_checkpoint: Path,
    pretrained_revision: str,
):
    from bello_mujoco.training.sonic_transfer import SonicTransferTrainingRunConfig

    paths = [
        _config("train_motion/default.yaml"),
        _config("profiles/train/motion.yaml"),
        _config("profiles/train/bello_mini_motion.yaml"),
        _config("profiles/train/bello_mini_sonic_transfer.yaml"),
    ]
    merged = _merged_profile(paths)
    merged.pop("motion")
    merged.pop("video_config")
    merged.dataset.directory = str(dataset_directory.expanduser().resolve())
    merged.pretrained.checkpoint = str(pretrained_checkpoint.expanduser().resolve())
    merged.pretrained.revision = pretrained_revision
    return structured_config(SonicTransferTrainingRunConfig, merged)


def profile_config_relative_path(name: ExperimentName) -> Path:
    return Path("configs") / _PROFILE_CONFIG_PATHS[name]


def supermimic_training_profile(
    dataset_directory: Path, *, name="bello_mini_supermimic"
):
    from bello_mujoco.training.supermimic import SuperMimicTrainingRunConfig

    if not is_supermimic_experiment(name):
        raise ValueError(f"{name!r} is not a SuperMimic experiment")
    mini = OmegaConf.load(_config("profiles/train/bello_mini_motion.yaml"))
    base = OmegaConf.create({"asset_config": mini.asset_config})
    merged = OmegaConf.merge(
        base, OmegaConf.load(_config("profiles/train/bello_mini_supermimic.yaml"))
    )
    if name != "bello_mini_supermimic":
        merged = OmegaConf.merge(
            merged,
            OmegaConf.load(CONFIG_ROOT.parent / profile_config_relative_path(name)),
        )
    merged.dataset.directory = str(dataset_directory.expanduser().resolve())
    return structured_config(SuperMimicTrainingRunConfig, merged)


def _config(relative_path: str) -> Path:
    return project_data_path("configs") / relative_path


def _merged_profile(paths: tuple[Path, ...] | list[Path]) -> DictConfig:
    configs = []
    for path in paths:
        config = OmegaConf.load(path)
        if not isinstance(config, DictConfig):
            raise TypeError(f"training profile {path} must be a mapping")
        configs.append(config)
    merged = OmegaConf.merge(*configs)
    return merged
