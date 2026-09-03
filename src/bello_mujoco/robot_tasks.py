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
class GmrLegDefinition:
    name: str
    source_hip_body_name: str
    source_knee_body_name: str
    source_foot_body_name: str
    target_hip_body_name: str
    target_knee_body_name: str
    target_foot_body_name: str
    target_knee_joint_name: str

    @property
    def source_body_names(self) -> tuple[str, str, str]:
        return (
            self.source_hip_body_name,
            self.source_knee_body_name,
            self.source_foot_body_name,
        )

    @property
    def target_body_names(self) -> tuple[str, str, str]:
        return (
            self.target_hip_body_name,
            self.target_knee_body_name,
            self.target_foot_body_name,
        )


@dataclass(frozen=True)
class GmrKneePoleDefinition:
    reference_flexion_radians: float
    sample_minimum_flexion_radians: float
    sample_minimum_distance_meters: float
    smoothing_frames: int
    legs: tuple[GmrLegDefinition, GmrLegDefinition]

    def __post_init__(self) -> None:
        values = (
            self.reference_flexion_radians,
            self.sample_minimum_flexion_radians,
            self.sample_minimum_distance_meters,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("GMR knee-pole values must be positive and finite")
        if self.reference_flexion_radians >= math.pi:
            raise ValueError("GMR knee-pole reference flexion must be smaller than pi")
        if self.smoothing_frames < 1 or self.smoothing_frames % 2 == 0:
            raise ValueError("GMR knee-pole smoothing_frames must be positive and odd")
        names = tuple(leg.name for leg in self.legs)
        if len(set(names)) != 2:
            raise ValueError("GMR legs must have two unique names")
        if any(
            not name
            for leg in self.legs
            for name in (
                leg.name,
                *leg.source_body_names,
                *leg.target_body_names,
                leg.target_knee_joint_name,
            )
        ):
            raise ValueError("GMR leg names must not be empty")


@dataclass(frozen=True)
class GmrRetargetDefinition:
    source_human: str
    target_robot: str
    ik_config: Path
    task_profile: str
    source_body_names: tuple[str, ...]
    source_foot_body_names: tuple[str, str]
    output_body_names: tuple[str, ...]
    standing_default_joint_names: tuple[str, ...]
    knee_pole: GmrKneePoleDefinition

    def __post_init__(self) -> None:
        if not self.source_human or not self.target_robot or not self.task_profile:
            raise ValueError("GMR source, target, and task profile must not be empty")
        if self.ik_config.is_absolute() or ".." in self.ik_config.parts:
            raise ValueError("GMR ik_config must be a contained relative path")
        for field_name in (
            "source_body_names",
            "source_foot_body_names",
            "output_body_names",
            "standing_default_joint_names",
        ):
            values = getattr(self, field_name)
            if not values or any(not value for value in values):
                raise ValueError(f"GMR {field_name} must be nonempty")
            if len(set(values)) != len(values):
                raise ValueError(f"GMR {field_name} must contain unique names")
        source_names = set(self.source_body_names)
        if not set(self.source_foot_body_names).issubset(source_names):
            raise ValueError("GMR source feet must belong to source_body_names")
        source_leg_names = {
            name for leg in self.knee_pole.legs for name in leg.source_body_names
        }
        if not source_leg_names.issubset(source_names):
            raise ValueError("GMR source leg bodies must belong to source_body_names")
        target_leg_names = {
            name for leg in self.knee_pole.legs for name in leg.target_body_names
        }
        if not target_leg_names.issubset(self.output_body_names):
            raise ValueError("GMR target leg bodies must belong to output_body_names")
        if (
            tuple(leg.source_foot_body_name for leg in self.knee_pole.legs)
            != self.source_foot_body_names
        ):
            raise ValueError("GMR source feet must follow the configured leg order")

    def config_path(self, gmr_root: Path) -> Path:
        return Path(gmr_root) / self.ik_config


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
    gmr: GmrRetargetDefinition

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
                *self.gmr.output_body_names,
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
    unknown_standing_joints = sorted(
        set(task.gmr.standing_default_joint_names) - set(actuator_names)
    )
    if unknown_standing_joints:
        raise ValueError(
            "GMR standing defaults reference unknown robot joints: "
            + ", ".join(unknown_standing_joints)
        )
    unknown_knee_joints = sorted(
        {leg.target_knee_joint_name for leg in task.gmr.knee_pole.legs}
        - set(actuator_names)
    )
    if unknown_knee_joints:
        raise ValueError(
            "GMR knee-pole diagnostics reference unknown robot joints: "
            + ", ".join(unknown_knee_joints)
        )


def _gmr_retarget_definition(value: Any) -> GmrRetargetDefinition:
    data = _mapping(value, "gmr")
    expected = {
        "source_human",
        "target_robot",
        "ik_config",
        "task_profile",
        "source_body_names",
        "source_foot_body_names",
        "output_body_names",
        "standing_default_joint_names",
        "knee_pole",
    }
    _require_exact_fields(data, expected, "gmr")
    return GmrRetargetDefinition(
        source_human=_string(data["source_human"], "gmr.source_human"),
        target_robot=_string(data["target_robot"], "gmr.target_robot"),
        ik_config=Path(_string(data["ik_config"], "gmr.ik_config")),
        task_profile=_string(data["task_profile"], "gmr.task_profile"),
        source_body_names=_string_tuple(
            data["source_body_names"], "gmr.source_body_names"
        ),
        source_foot_body_names=_sized_string_tuple(
            data["source_foot_body_names"], 2, "gmr.source_foot_body_names"
        ),
        output_body_names=_string_tuple(
            data["output_body_names"], "gmr.output_body_names"
        ),
        standing_default_joint_names=_string_tuple(
            data["standing_default_joint_names"],
            "gmr.standing_default_joint_names",
        ),
        knee_pole=_gmr_knee_pole_definition(data["knee_pole"]),
    )


def _gmr_knee_pole_definition(value: Any) -> GmrKneePoleDefinition:
    data = _mapping(value, "gmr.knee_pole")
    expected = {
        "reference_flexion_radians",
        "sample_minimum_flexion_radians",
        "sample_minimum_distance_meters",
        "smoothing_frames",
        "legs",
    }
    _require_exact_fields(data, expected, "gmr.knee_pole")
    legs = tuple(
        _gmr_leg_definition(item, index)
        for index, item in enumerate(_list(data["legs"], "gmr.knee_pole.legs"))
    )
    if len(legs) != 2:
        raise ValueError("GMR knee_pole.legs must contain exactly two legs")
    return GmrKneePoleDefinition(
        reference_flexion_radians=_float(
            data["reference_flexion_radians"],
            "gmr.knee_pole.reference_flexion_radians",
        ),
        sample_minimum_flexion_radians=_float(
            data["sample_minimum_flexion_radians"],
            "gmr.knee_pole.sample_minimum_flexion_radians",
        ),
        sample_minimum_distance_meters=_float(
            data["sample_minimum_distance_meters"],
            "gmr.knee_pole.sample_minimum_distance_meters",
        ),
        smoothing_frames=_integer(
            data["smoothing_frames"], "gmr.knee_pole.smoothing_frames"
        ),
        legs=legs,  # type: ignore[arg-type]
    )


def _gmr_leg_definition(value: Any, index: int) -> GmrLegDefinition:
    path = f"gmr.knee_pole.legs[{index}]"
    data = _mapping(value, path)
    expected = {
        "name",
        "source_hip_body_name",
        "source_knee_body_name",
        "source_foot_body_name",
        "target_hip_body_name",
        "target_knee_body_name",
        "target_foot_body_name",
        "target_knee_joint_name",
    }
    _require_exact_fields(data, expected, path)
    return GmrLegDefinition(
        name=_string(data["name"], f"{path}.name"),
        source_hip_body_name=_string(
            data["source_hip_body_name"], f"{path}.source_hip_body_name"
        ),
        source_knee_body_name=_string(
            data["source_knee_body_name"], f"{path}.source_knee_body_name"
        ),
        source_foot_body_name=_string(
            data["source_foot_body_name"], f"{path}.source_foot_body_name"
        ),
        target_hip_body_name=_string(
            data["target_hip_body_name"], f"{path}.target_hip_body_name"
        ),
        target_knee_body_name=_string(
            data["target_knee_body_name"], f"{path}.target_knee_body_name"
        ),
        target_foot_body_name=_string(
            data["target_foot_body_name"], f"{path}.target_foot_body_name"
        ),
        target_knee_joint_name=_string(
            data["target_knee_joint_name"], f"{path}.target_knee_joint_name"
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


def _sized_string_tuple(value: Any, size: int, path: str) -> tuple[str, str]:
    result = _string_tuple(value, path)
    if len(result) != size:
        raise ValueError(f"motion task {path} must contain exactly {size} values")
    return result  # type: ignore[return-value]


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"motion task {path} must be a mapping")
    return value


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"motion task {path} must be a list")
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


def _integer(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"motion task {path} must be an integer")
    return value
