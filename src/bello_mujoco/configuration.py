"""Hydra configuration decoding and canonical training profiles."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
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

from bello_mujoco.assets_pipeline import project_data_path

if TYPE_CHECKING:
    from bello_mujoco.training.locomotion import TrainingRunConfig
    from bello_mujoco.training.motion import MotionTrainingRunConfig
    from bello_mujoco.training.twist import TwistTrainingRunConfig
    from bello_mujoco.training.twist2 import Twist2TrainingRunConfig

T = TypeVar("T")
ExperimentName = Literal[
    "smoke",
    "bello_locomotion",
    "bello_mini_locomotion",
    "motion",
    "twist_teacher",
    "twist_student",
    "twist2",
]
CONFIG_ROOT = project_data_path("configs")
_PROFILE_CONFIG_PATHS: dict[ExperimentName, Path] = {
    "smoke": Path("profiles/train/smoke.yaml"),
    "bello_locomotion": Path("profiles/train/bello_locomotion.yaml"),
    "bello_mini_locomotion": Path("profiles/train/bello_mini_locomotion.yaml"),
    "motion": Path("profiles/train/motion.yaml"),
    "twist_teacher": Path("profiles/train/twist_teacher.yaml"),
    "twist_student": Path("profiles/train/twist_student.yaml"),
    "twist2": Path("profiles/train/twist2.yaml"),
}


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

    locomotion_experiments = {"bello_locomotion", "bello_mini_locomotion"}
    if name != "smoke" and name not in locomotion_experiments:
        raise ValueError(f"{name!r} is not a locomotion experiment")
    paths = [_config("train/default.yaml")]
    if name in locomotion_experiments:
        paths.extend(
            (
                _config("train/locomotion.yaml"),
                _config(f"profiles/train/{name}.yaml"),
            )
        )
    if name == "smoke":
        paths.append(_config("profiles/train/smoke.yaml"))
    return structured_config(TrainingRunConfig, _merged_profile(paths))


def motion_training_profile(
    motion_path: Path | None = None,
) -> MotionTrainingRunConfig:
    from bello_mujoco.training.motion import (
        MotionSelectionConfig,
        MotionTrainingRunConfig,
    )

    config = structured_config(
        MotionTrainingRunConfig,
        _merged_profile(
            (
                _config("train_motion/default.yaml"),
                _config("profiles/train/motion.yaml"),
            )
        ),
    )
    if motion_path is None:
        return config
    resolved = motion_path.expanduser().resolve()
    return replace(
        config,
        motion=MotionSelectionConfig(
            directory=resolved.parent,
            sequence=resolved.stem,
        ),
    )


def twist_training_profile(
    stage: Literal["teacher", "student"],
    *,
    dataset_index: Path | None = None,
    teacher_checkpoint_path: Path | None = None,
) -> TwistTrainingRunConfig:
    from bello_mujoco.training.twist import TwistTrainingRunConfig

    config = structured_config(
        TwistTrainingRunConfig,
        _merged_profile(
            (
                _config("train_twist/default.yaml"),
                _config(f"profiles/train/twist_{stage}.yaml"),
            )
        ),
    )
    overrides = {}
    if dataset_index is not None:
        overrides["dataset_index"] = dataset_index.expanduser().resolve()
    if teacher_checkpoint_path is not None:
        overrides["teacher_checkpoint_path"] = (
            teacher_checkpoint_path.expanduser().resolve()
        )
    return replace(config, **overrides)


def twist2_training_profile(
    *, dataset_index: Path | None = None
) -> Twist2TrainingRunConfig:
    from bello_mujoco.training.twist2 import Twist2TrainingRunConfig

    config = structured_config(
        Twist2TrainingRunConfig,
        _merged_profile(
            (
                _config("train_twist2/default.yaml"),
                _config("profiles/train/twist2.yaml"),
            )
        ),
    )
    if dataset_index is None:
        return config
    return replace(config, dataset_index=dataset_index.expanduser().resolve())


def profile_config_relative_path(name: ExperimentName) -> Path:
    return Path("configs") / _PROFILE_CONFIG_PATHS[name]


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
    OmegaConf.resolve(merged)
    return merged
