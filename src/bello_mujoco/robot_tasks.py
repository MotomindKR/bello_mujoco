"""Validated robot-specific contracts for motion tasks."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
import math
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from bello_mujoco.robot import AssetName, RobotDefinition, robot_definition


@dataclass(frozen=True)
class GmrRetargetDefinition:
    source_human: str
    target_robot: str
    ik_config: Path
    source_body_names: tuple[str, ...]
    output_body_names: tuple[str, ...]
    initial_joint_positions: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not self.source_human or not self.target_robot:
            raise ValueError("GMR source and target must not be empty")
        if self.ik_config.is_absolute() or ".." in self.ik_config.parts:
            raise ValueError("GMR ik_config must be a contained relative path")
        for field_name in ("source_body_names", "output_body_names"):
            values = getattr(self, field_name)
            if not values or any(not value for value in values):
                raise ValueError(f"GMR {field_name} must be nonempty")
            if len(set(values)) != len(values):
                raise ValueError(f"GMR {field_name} must contain unique names")
        names = [name for name, _ in self.initial_joint_positions]
        if len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError("GMR initial joint names must be unique and nonempty")
        if any(not math.isfinite(value) for _, value in self.initial_joint_positions):
            raise ValueError("GMR initial joint positions must be finite")

    def config_path(self, gmr_root: Path) -> Path:
        return Path(gmr_root) / self.ik_config


@dataclass(frozen=True)
class MotionTaskDefinition:
    asset: AssetName
    reference_root_body_name: str
    default_tracked_body_names: tuple[str, ...]
    beyondmimic_tracked_body_names: tuple[str, ...]
    default_end_effector_body_names: tuple[str, ...]
    beyondmimic_end_effector_body_names: tuple[str, ...]
    gmr: GmrRetargetDefinition

    def __post_init__(self) -> None:
        for field_name in (
            "default_tracked_body_names",
            "beyondmimic_tracked_body_names",
            "default_end_effector_body_names",
            "beyondmimic_end_effector_body_names",
        ):
            values = getattr(self, field_name)
            if not values or any(not value for value in values):
                raise ValueError(f"motion task {field_name} must be nonempty")
            if len(set(values)) != len(values):
                raise ValueError(f"motion task {field_name} must contain unique names")
        if not self.reference_root_body_name:
            raise ValueError("motion task reference_root_body_name must not be empty")
        for field_name in (
            "default_tracked_body_names",
            "beyondmimic_tracked_body_names",
        ):
            if getattr(self, field_name)[0] != self.reference_root_body_name:
                raise ValueError(
                    f"motion task {field_name} must begin with the reference root body"
                )
        if not set(self.default_end_effector_body_names).issubset(
            self.default_tracked_body_names
        ):
            raise ValueError(
                "default motion end effectors must belong to the default tracked bodies"
            )
        if not set(self.beyondmimic_end_effector_body_names).issubset(
            self.beyondmimic_tracked_body_names
        ):
            raise ValueError(
                "BeyondMimic end effectors must belong to its tracked bodies"
            )

    @property
    def referenced_body_names(self) -> frozenset[str]:
        return frozenset(
            (
                self.reference_root_body_name,
                *self.default_tracked_body_names,
                *self.beyondmimic_tracked_body_names,
                *self.default_end_effector_body_names,
                *self.beyondmimic_end_effector_body_names,
                *self.gmr.output_body_names,
            )
        )


_MOTION_TASK_CONFIG_FILES: dict[AssetName, str] = {
    "bello": "bello_motion.yaml",
    "bello_mini": "bello_mini_motion.yaml",
}
_MOTION_TASK_FIELDS = frozenset(
    {
        "asset",
        "reference_root_body_name",
        "default_tracked_body_names",
        "beyondmimic_tracked_body_names",
        "default_end_effector_body_names",
        "beyondmimic_end_effector_body_names",
        "gmr",
    }
)


@lru_cache(maxsize=None)
def motion_task_definition(asset: AssetName) -> MotionTaskDefinition:
    try:
        filename = _MOTION_TASK_CONFIG_FILES[asset]
    except KeyError as error:
        supported = ", ".join(sorted(_MOTION_TASK_CONFIG_FILES))
        raise ValueError(
            f"robot asset {asset!r} has no motion task profile; expected one of: "
            f"{supported}"
        ) from error
    resource = files("bello_mujoco").joinpath("robots", filename)
    with resource.open("r", encoding="utf-8") as config_file:
        document = OmegaConf.to_container(
            OmegaConf.load(config_file),
            resolve=True,
            throw_on_missing=True,
        )
    if not isinstance(document, dict):
        raise TypeError(f"motion task config {filename} must be a mapping")
    fields = set(document)
    if fields != _MOTION_TASK_FIELDS:
        missing = sorted(_MOTION_TASK_FIELDS - fields)
        unknown = sorted(fields - _MOTION_TASK_FIELDS)
        details = []
        if missing:
            details.append("missing=" + ", ".join(missing))
        if unknown:
            details.append("unknown=" + ", ".join(unknown))
        raise ValueError(
            f"motion task config {filename} fields are invalid: " + "; ".join(details)
        )
    if document["asset"] != asset:
        raise ValueError(
            f"motion task config {filename}.asset must be {asset!r}, got "
            f"{document['asset']!r}"
        )
    definition = MotionTaskDefinition(
        asset=asset,
        reference_root_body_name=_string(
            document["reference_root_body_name"], "reference_root_body_name"
        ),
        default_tracked_body_names=_string_tuple(
            document["default_tracked_body_names"], "default_tracked_body_names"
        ),
        beyondmimic_tracked_body_names=_string_tuple(
            document["beyondmimic_tracked_body_names"],
            "beyondmimic_tracked_body_names",
        ),
        default_end_effector_body_names=_string_tuple(
            document["default_end_effector_body_names"],
            "default_end_effector_body_names",
        ),
        beyondmimic_end_effector_body_names=_string_tuple(
            document["beyondmimic_end_effector_body_names"],
            "beyondmimic_end_effector_body_names",
        ),
        gmr=_gmr_retarget_definition(document["gmr"]),
    )
    _validate_against_robot(definition, robot_definition(asset))
    return definition


def validate_motion_task_source_urdf(
    task: MotionTaskDefinition,
    urdf_path: Path,
) -> None:
    root = ET.parse(urdf_path).getroot()
    link_names = {
        name for link in root.findall("link") if (name := link.get("name")) is not None
    }
    missing_links = sorted(task.referenced_body_names - link_names)
    if missing_links:
        raise ValueError(
            "motion task profile references links absent from the source URDF: "
            + ", ".join(missing_links)
        )


def _validate_against_robot(
    task: MotionTaskDefinition,
    robot: RobotDefinition,
) -> None:
    actuator_names = {actuator.joint_name for actuator in robot.actuators}
    unknown = {name for name, _ in task.gmr.initial_joint_positions} - actuator_names
    if unknown:
        raise ValueError(
            "GMR initial pose references unknown joints: " + ", ".join(sorted(unknown))
        )


def _gmr_retarget_definition(value: Any) -> GmrRetargetDefinition:
    data = _mapping(value, "gmr")
    _require_exact_fields(
        data,
        {
            "source_human",
            "target_robot",
            "ik_config",
            "source_body_names",
            "output_body_names",
            "initial_joint_positions",
        },
        "gmr",
    )
    return GmrRetargetDefinition(
        source_human=_string(data["source_human"], "gmr.source_human"),
        target_robot=_string(data["target_robot"], "gmr.target_robot"),
        ik_config=Path(_string(data["ik_config"], "gmr.ik_config")),
        source_body_names=_string_tuple(
            data["source_body_names"], "gmr.source_body_names"
        ),
        output_body_names=_string_tuple(
            data["output_body_names"], "gmr.output_body_names"
        ),
        initial_joint_positions=tuple(
            (
                _string(name, "gmr.initial_joint_positions"),
                _float(angle, "gmr.initial_joint_positions"),
            )
            for name, angle in _mapping(
                data["initial_joint_positions"], "gmr.initial_joint_positions"
            ).items()
        ),
    )


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise TypeError(f"motion task {path} must be a nonempty trimmed string")
    return value


def _string_tuple(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"motion task {path} must be a list")
    return tuple(_string(item, path) for item in value)


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"motion task {path} must be a mapping")
    return value


def _require_exact_fields(data: dict[str, Any], expected: set[str], path: str) -> None:
    if set(data) == expected:
        return
    missing = sorted(expected - set(data))
    unknown = sorted(set(data) - expected)
    raise ValueError(
        f"motion task {path} fields are invalid: missing={missing}, unknown={unknown}"
    )


def _float(value: Any, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"motion task {path} must be numeric")
    return float(value)
