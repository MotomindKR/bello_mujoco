"""Validated robot-specific contracts for motion tasks."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from bello_mujoco.robot import AssetName, RobotDefinition, robot_definition


@dataclass(frozen=True)
class MotionTaskDefinition:
    asset: AssetName
    legacy_import_joint_names: tuple[str, ...]
    fixed_joint_names: tuple[str, ...]
    reference_root_body_name: str
    default_tracked_body_names: tuple[str, ...]
    beyondmimic_tracked_body_names: tuple[str, ...]
    default_end_effector_body_names: tuple[str, ...]
    beyondmimic_end_effector_body_names: tuple[str, ...]
    twist_key_body_names: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "legacy_import_joint_names",
            "fixed_joint_names",
            "default_tracked_body_names",
            "beyondmimic_tracked_body_names",
            "default_end_effector_body_names",
            "beyondmimic_end_effector_body_names",
            "twist_key_body_names",
        ):
            values = getattr(self, field_name)
            if not values or any(not value for value in values):
                raise ValueError(f"motion task {field_name} must be nonempty")
            if len(set(values)) != len(values):
                raise ValueError(f"motion task {field_name} must contain unique names")
        if not self.reference_root_body_name:
            raise ValueError("motion task reference_root_body_name must not be empty")
        if not set(self.fixed_joint_names) & set(self.legacy_import_joint_names):
            raise ValueError(
                "motion task fixed_joint_names must include a legacy import joint"
            )
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
        if len(self.twist_key_body_names) != 9:
            raise ValueError("TWIST requires exactly nine key bodies")

    @property
    def imported_actuator_joint_names(self) -> tuple[str, ...]:
        fixed = set(self.fixed_joint_names)
        return tuple(
            name for name in self.legacy_import_joint_names if name not in fixed
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
                *self.twist_key_body_names,
            )
        )


_MOTION_TASK_CONFIG_FILES: dict[AssetName, str] = {
    "full_body": "full_body_motion.yaml",
}
_MOTION_TASK_FIELDS = frozenset(
    {
        "asset",
        "legacy_import_joint_names",
        "fixed_joint_names",
        "reference_root_body_name",
        "default_tracked_body_names",
        "beyondmimic_tracked_body_names",
        "default_end_effector_body_names",
        "beyondmimic_end_effector_body_names",
        "twist_key_body_names",
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
        legacy_import_joint_names=_string_tuple(
            document["legacy_import_joint_names"], "legacy_import_joint_names"
        ),
        fixed_joint_names=_string_tuple(
            document["fixed_joint_names"], "fixed_joint_names"
        ),
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
        twist_key_body_names=_string_tuple(
            document["twist_key_body_names"], "twist_key_body_names"
        ),
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
    joint_names = {
        name
        for joint in root.findall("joint")
        if (name := joint.get("name")) is not None
    }
    missing_joints = sorted(set(task.legacy_import_joint_names) - joint_names)
    if missing_joints:
        raise ValueError(
            "motion task profile references joints absent from the source URDF: "
            + ", ".join(missing_joints)
        )


def _validate_against_robot(
    task: MotionTaskDefinition,
    robot: RobotDefinition,
) -> None:
    actuator_names = tuple(actuator.joint_name for actuator in robot.actuators)
    if set(task.imported_actuator_joint_names) != set(actuator_names):
        missing = sorted(set(actuator_names) - set(task.imported_actuator_joint_names))
        unknown = sorted(set(task.imported_actuator_joint_names) - set(actuator_names))
        raise ValueError(
            "motion task imported actuator joints differ from the robot definition: "
            f"missing={missing}, unknown={unknown}"
        )


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise TypeError(f"motion task {path} must be a nonempty trimmed string")
    return value


def _string_tuple(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"motion task {path} must be a list")
    return tuple(_string(item, path) for item in value)
