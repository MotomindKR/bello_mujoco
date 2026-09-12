"""Validated static definitions for supported robot assets."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf

from bello_mujoco.models import robot_descriptions_root

AssetName = Literal["bello", "bello_mini"]


@dataclass(frozen=True)
class PayloadAttachment:
    link_name: str
    radius_m: float
    center_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not self.link_name:
            raise ValueError("payload attachment link_name must not be empty")
        if not math.isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise ValueError("payload attachment radius_m must be positive and finite")
        if len(self.center_m) != 3 or not all(
            math.isfinite(value) for value in self.center_m
        ):
            raise ValueError(
                "payload attachment center_m must contain three finite values"
            )


@dataclass(frozen=True)
class CoupledActuator:
    tendon_name: str
    logical_joint_name: str
    joint_coefficients: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not self.tendon_name or not self.logical_joint_name:
            raise ValueError("coupled actuator names must not be empty")
        joint_names = tuple(name for name, _ in self.joint_coefficients)
        if not joint_names or any(not name for name in joint_names):
            raise ValueError("coupled actuator joint names must not be empty")
        if len(set(joint_names)) != len(joint_names):
            raise ValueError("coupled actuator joint names must be unique")
        if self.logical_joint_name not in joint_names:
            raise ValueError(
                "coupled actuator logical_joint_name must appear in joint_coefficients"
            )
        if any(
            not math.isfinite(coefficient) or coefficient == 0.0
            for _, coefficient in self.joint_coefficients
        ):
            raise ValueError("coupled actuator coefficients must be finite and nonzero")


@dataclass(frozen=True)
class CoupledActuatorGroup:
    name: str
    actuators: tuple[CoupledActuator, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("coupled actuator group name must not be empty")
        if not self.actuators:
            raise ValueError("coupled actuator group must contain actuators")
        tendon_names = tuple(actuator.tendon_name for actuator in self.actuators)
        logical_names = tuple(
            actuator.logical_joint_name for actuator in self.actuators
        )
        if len(set(tendon_names)) != len(tendon_names):
            raise ValueError(
                "coupled actuator tendon names must be unique within a group"
            )
        if len(set(logical_names)) != len(logical_names):
            raise ValueError(
                "coupled actuator logical joints must be unique within a group"
            )
        coefficient_joint_sets = {
            tuple(name for name, _ in actuator.joint_coefficients)
            for actuator in self.actuators
        }
        if len(coefficient_joint_sets) != 1:
            raise ValueError(
                "coupled actuators in a group must use the same ordered joints"
            )

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.actuators[0].joint_coefficients)


@dataclass(frozen=True)
class JointActuatorDefinition:
    joint_name: str
    mode: Literal["policy", "hold"]
    default_pos: float
    soft_limit: tuple[float, float] | None
    gains: tuple[float, float]
    effort_limit: float | None

    def __post_init__(self) -> None:
        if not self.joint_name:
            raise ValueError("actuator joint_name must not be empty")
        if self.mode not in {"policy", "hold"}:
            raise ValueError(f"invalid actuator mode {self.mode!r}")
        if not math.isfinite(self.default_pos):
            raise ValueError("actuator default_pos must be finite")
        if self.soft_limit is not None and (
            len(self.soft_limit) != 2
            or not all(math.isfinite(value) for value in self.soft_limit)
            or self.soft_limit[0] >= self.soft_limit[1]
        ):
            raise ValueError("actuator soft_limit must be finite and ordered")
        if self.soft_limit is not None and not (
            self.soft_limit[0] <= self.default_pos <= self.soft_limit[1]
        ):
            raise ValueError("actuator default_pos must fall within soft_limit")
        if len(self.gains) != 2 or not all(
            math.isfinite(value) for value in self.gains
        ):
            raise ValueError("actuator gains must contain two finite values")
        if self.effort_limit is not None and (
            not math.isfinite(self.effort_limit) or self.effort_limit <= 0.0
        ):
            raise ValueError("actuator effort_limit must be positive and finite")


@dataclass(frozen=True)
class FootSupportGeometry:
    body_name: str
    geom_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.body_name:
            raise ValueError("foot support body_name must not be empty")
        if not self.geom_names or any(not name for name in self.geom_names):
            raise ValueError("foot support geom_names must be nonempty")
        if len(set(self.geom_names)) != len(self.geom_names):
            raise ValueError("foot support geom_names must be unique")


@dataclass(frozen=True)
class RobotDefinition:
    asset: AssetName
    source_urdf: Path
    hardware_description: Path | None
    physical_hardware_ready: bool
    motor_model_catalog: Path | None
    root_body_name: str
    root_joint_name: str
    default_root_position: tuple[float, float, float]
    default_root_quaternion: tuple[float, float, float, float]
    default_imu_body_name: str
    locomotion_command_frame_body_name: str
    locomotion_forward_axis: tuple[float, float, float]
    foot_body_names: tuple[str, str]
    foot_support_geometries: tuple[FootSupportGeometry, FootSupportGeometry]
    foot_forward_axes: tuple[
        tuple[str, tuple[float, float, float]],
        tuple[str, tuple[float, float, float]],
    ]
    randomization_base_body_name: str
    randomization_torso_body_name: str
    randomization_central_body_names: tuple[str, ...]
    actuators: tuple[JointActuatorDefinition, ...]
    payload_attachments: tuple[PayloadAttachment, ...] = ()
    coupled_actuator_groups: tuple[CoupledActuatorGroup, ...] = ()

    def __post_init__(self) -> None:
        paths = (
            self.source_urdf,
            *(
                path
                for path in (self.hardware_description, self.motor_model_catalog)
                if path is not None
            ),
        )
        if any(path.is_absolute() or ".." in path.parts for path in paths):
            raise ValueError("robot definition paths must be relative and contained")
        if not self.source_urdf.name:
            raise ValueError("robot definition requires source paths")
        if self.physical_hardware_ready and (
            self.hardware_description is None or self.motor_model_catalog is None
        ):
            raise ValueError(
                "physical_hardware_ready requires a hardware description and motor "
                "model catalog"
            )
        names = (
            self.root_body_name,
            self.root_joint_name,
            self.default_imu_body_name,
            self.locomotion_command_frame_body_name,
            self.randomization_base_body_name,
            self.randomization_torso_body_name,
        )
        if any(not name for name in names):
            raise ValueError("robot definition names must not be empty")
        if len(self.default_root_position) != 3 or not all(
            math.isfinite(value) for value in self.default_root_position
        ):
            raise ValueError("default_root_position must contain three finite values")
        if len(self.default_root_quaternion) != 4 or not all(
            math.isfinite(value) for value in self.default_root_quaternion
        ):
            raise ValueError("default_root_quaternion must contain four finite values")
        quaternion_norm = math.sqrt(
            sum(value * value for value in self.default_root_quaternion)
        )
        if not math.isclose(quaternion_norm, 1.0, abs_tol=1.0e-6):
            raise ValueError("default_root_quaternion must be normalized")
        if len(self.locomotion_forward_axis) != 3 or not all(
            math.isfinite(value) for value in self.locomotion_forward_axis
        ):
            raise ValueError("locomotion_forward_axis must contain three finite values")
        if not math.isclose(self.locomotion_forward_axis[2], 0.0, abs_tol=1.0e-6):
            raise ValueError("locomotion_forward_axis must be horizontal")
        if not math.isclose(
            math.sqrt(sum(value * value for value in self.locomotion_forward_axis)),
            1.0,
            abs_tol=1.0e-6,
        ):
            raise ValueError("locomotion_forward_axis must be normalized")
        if len(set(self.foot_body_names)) != 2 or any(
            not name for name in self.foot_body_names
        ):
            raise ValueError("foot_body_names must contain two unique names")
        support_body_names = tuple(
            item.body_name for item in self.foot_support_geometries
        )
        if support_body_names != self.foot_body_names:
            raise ValueError(
                "foot_support_geometries must follow the declared foot_body_names order"
            )
        support_geom_names = tuple(
            name for item in self.foot_support_geometries for name in item.geom_names
        )
        if len(set(support_geom_names)) != len(support_geom_names):
            raise ValueError("foot support geometry names must be globally unique")
        forward_body_names = tuple(name for name, _ in self.foot_forward_axes)
        if forward_body_names != self.foot_body_names:
            raise ValueError(
                "foot_forward_axes must follow the declared foot_body_names order"
            )
        for _, axis in self.foot_forward_axes:
            if len(axis) != 3 or not all(math.isfinite(value) for value in axis):
                raise ValueError("foot forward axes must contain three finite values")
            if not math.isclose(
                math.sqrt(sum(value * value for value in axis)),
                1.0,
                abs_tol=1.0e-6,
            ):
                raise ValueError("foot forward axes must be normalized")
        if not self.randomization_central_body_names or len(
            set(self.randomization_central_body_names)
        ) != len(self.randomization_central_body_names):
            raise ValueError(
                "randomization_central_body_names must be nonempty and unique"
            )
        if (
            self.randomization_base_body_name
            not in self.randomization_central_body_names
        ):
            raise ValueError(
                "randomization base body must be one of the central bodies"
            )
        if (
            self.randomization_torso_body_name
            not in self.randomization_central_body_names
        ):
            raise ValueError(
                "randomization torso body must be one of the central bodies"
            )
        actuator_joint_names = tuple(actuator.joint_name for actuator in self.actuators)
        if not actuator_joint_names or len(set(actuator_joint_names)) != len(
            actuator_joint_names
        ):
            raise ValueError("actuator joint names must be nonempty and unique")
        payload_link_names = tuple(
            attachment.link_name for attachment in self.payload_attachments
        )
        if len(set(payload_link_names)) != len(payload_link_names):
            raise ValueError("payload attachment links must be unique")
        group_names = tuple(group.name for group in self.coupled_actuator_groups)
        if len(set(group_names)) != len(group_names):
            raise ValueError("coupled actuator group names must be unique")
        tendon_names = tuple(
            actuator.tendon_name
            for group in self.coupled_actuator_groups
            for actuator in group.actuators
        )
        logical_names = tuple(
            actuator.logical_joint_name
            for group in self.coupled_actuator_groups
            for actuator in group.actuators
        )
        if len(set(tendon_names)) != len(tendon_names):
            raise ValueError("coupled actuator tendon names must be globally unique")
        if len(set(logical_names)) != len(logical_names):
            raise ValueError("coupled actuator logical joints must be globally unique")
        grouped_joint_names = tuple(
            name for group in self.coupled_actuator_groups for name in group.joint_names
        )
        if len(set(grouped_joint_names)) != len(grouped_joint_names):
            raise ValueError(
                "joints must not belong to more than one coupled actuator group"
            )
        unknown_coupled_joints = {
            name
            for actuator in self.coupled_actuators
            for name, _ in actuator.joint_coefficients
            if name not in actuator_joint_names
        }
        if unknown_coupled_joints:
            raise ValueError(
                "coupled actuators reference unknown configured joints: "
                + ", ".join(sorted(unknown_coupled_joints))
            )
        actuators_by_name = {
            actuator.joint_name: actuator for actuator in self.actuators
        }
        for group in self.coupled_actuator_groups:
            group_actuators = tuple(
                actuators_by_name[name] for name in group.joint_names
            )
            reference = group_actuators[0]
            if any(item.mode != reference.mode for item in group_actuators[1:]):
                raise ValueError(
                    f"coupled actuator group {group.name!r} has mixed modes"
                )
            if any(item.gains != reference.gains for item in group_actuators[1:]):
                raise ValueError(
                    f"coupled actuator group {group.name!r} has mixed gains"
                )
            if any(
                item.effort_limit != reference.effort_limit
                for item in group_actuators[1:]
            ):
                raise ValueError(
                    f"coupled actuator group {group.name!r} has mixed effort limits"
                )

    @property
    def coupled_actuators(self) -> tuple[CoupledActuator, ...]:
        return tuple(
            actuator
            for group in self.coupled_actuator_groups
            for actuator in group.actuators
        )

    def source_urdf_path(self, description_root: Path) -> Path:
        return description_root / self.source_urdf

    def hardware_description_path(self, description_root: Path) -> Path:
        if self.hardware_description is None:
            raise ValueError(f"robot asset {self.asset!r} has no hardware description")
        return description_root / self.hardware_description

    def motor_model_catalog_path(self, description_root: Path) -> Path:
        if self.motor_model_catalog is None:
            raise ValueError(f"robot asset {self.asset!r} has no motor model catalog")
        return description_root / self.motor_model_catalog


def validate_robot_source_urdf(
    definition: RobotDefinition,
    urdf_path: Path,
) -> None:
    root = ET.parse(urdf_path).getroot()
    link_names = {
        name for link in root.findall("link") if (name := link.get("name")) is not None
    }
    required_link_names = {
        definition.default_imu_body_name,
        definition.locomotion_command_frame_body_name,
        *definition.foot_body_names,
        definition.randomization_base_body_name,
        definition.randomization_torso_body_name,
        *definition.randomization_central_body_names,
        *(attachment.link_name for attachment in definition.payload_attachments),
    }
    missing_links = sorted(required_link_names - link_names)
    if missing_links:
        raise ValueError(
            "robot definition references links absent from the source URDF: "
            + ", ".join(missing_links)
        )

    links_by_name = {
        name: link
        for link in root.findall("link")
        if (name := link.get("name")) is not None
    }
    for support in definition.foot_support_geometries:
        collision_names = {
            name
            for collision in links_by_name[support.body_name].findall("collision")
            if (name := collision.get("name")) is not None
        }
        missing_geoms = sorted(set(support.geom_names) - collision_names)
        if missing_geoms:
            raise ValueError(
                f"foot support geometries for {support.body_name!r} are absent from "
                "the source URDF link: " + ", ".join(missing_geoms)
            )

    joint_names = {
        name
        for joint in root.findall("joint")
        if (name := joint.get("name")) is not None
    }
    configured_joint_names = {actuator.joint_name for actuator in definition.actuators}
    missing_joints = sorted(configured_joint_names - joint_names)
    if missing_joints:
        raise ValueError(
            "robot definition references joints absent from the source URDF: "
            + ", ".join(missing_joints)
        )


_ROBOT_CONFIG_FILES: dict[AssetName, str] = {
    "bello": "bello.yaml",
    "bello_mini": "bello_mini.yaml",
}
_ROBOT_CONFIG_FIELDS = frozenset(
    {
        "asset",
        "source_urdf",
        "hardware_description",
        "physical_hardware_ready",
        "motor_model_catalog",
        "root_body_name",
        "root_joint_name",
        "default_root_position",
        "default_root_quaternion",
        "default_imu_body_name",
        "locomotion_command_frame_body_name",
        "locomotion_forward_axis",
        "foot_body_names",
        "foot_support_geometries",
        "foot_forward_axes",
        "randomization_base_body_name",
        "randomization_torso_body_name",
        "randomization_central_body_names",
        "actuators",
        "payload_attachments",
    }
)


@lru_cache(maxsize=None)
def robot_definition(asset: AssetName) -> RobotDefinition:
    try:
        filename = _ROBOT_CONFIG_FILES[asset]
    except KeyError as error:
        supported = ", ".join(sorted(_ROBOT_CONFIG_FILES))
        raise ValueError(
            f"unsupported robot asset {asset!r}; expected one of: {supported}"
        ) from error
    resource = files("bello_mujoco").joinpath("robots", filename)
    with resource.open("r", encoding="utf-8") as config_file:
        document = OmegaConf.to_container(
            OmegaConf.load(config_file),
            resolve=True,
            throw_on_missing=True,
        )
    data = _mapping(document, path=f"robot config {filename}")
    _require_exact_fields(data, _ROBOT_CONFIG_FIELDS, path=f"robot config {filename}")
    if data["asset"] != asset:
        raise ValueError(
            f"robot config {filename}.asset must be {asset!r}, got {data['asset']!r}"
        )
    return RobotDefinition(
        asset=asset,
        source_urdf=Path(_string(data["source_urdf"], "source_urdf")),
        hardware_description=_optional_path(
            data["hardware_description"], "hardware_description"
        ),
        physical_hardware_ready=_bool(
            data["physical_hardware_ready"], "physical_hardware_ready"
        ),
        motor_model_catalog=_optional_path(
            data["motor_model_catalog"], "motor_model_catalog"
        ),
        root_body_name=_string(data["root_body_name"], "root_body_name"),
        root_joint_name=_string(data["root_joint_name"], "root_joint_name"),
        default_root_position=_float_tuple(
            data["default_root_position"], 3, "default_root_position"
        ),
        default_root_quaternion=_float_tuple(
            data["default_root_quaternion"], 4, "default_root_quaternion"
        ),
        default_imu_body_name=_string(
            data["default_imu_body_name"], "default_imu_body_name"
        ),
        locomotion_command_frame_body_name=_string(
            data["locomotion_command_frame_body_name"],
            "locomotion_command_frame_body_name",
        ),
        locomotion_forward_axis=_float_tuple(
            data["locomotion_forward_axis"], 3, "locomotion_forward_axis"
        ),
        foot_body_names=_string_tuple(data["foot_body_names"], 2, "foot_body_names"),
        foot_support_geometries=_foot_support_geometries(
            data["foot_support_geometries"]
        ),
        foot_forward_axes=_foot_forward_axes(data["foot_forward_axes"]),
        randomization_base_body_name=_string(
            data["randomization_base_body_name"],
            "randomization_base_body_name",
        ),
        randomization_torso_body_name=_string(
            data["randomization_torso_body_name"],
            "randomization_torso_body_name",
        ),
        randomization_central_body_names=_string_tuple(
            data["randomization_central_body_names"],
            None,
            "randomization_central_body_names",
        ),
        actuators=tuple(
            _joint_actuator(value, index)
            for index, value in enumerate(_list(data["actuators"], "actuators"))
        ),
        payload_attachments=tuple(
            _payload_attachment(value, index)
            for index, value in enumerate(
                _list(data["payload_attachments"], "payload_attachments")
            )
        ),
        coupled_actuator_groups=urdf_coupled_actuators(
            robot_descriptions_root() / _string(data["source_urdf"], "source_urdf"),
            tuple(
                value["joint_name"] for value in _list(data["actuators"], "actuators")
            ),
        ),
    )


def _mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be a mapping")
    return value


def _optional_path(value: Any, path: str) -> Path | None:
    if value is None:
        return None
    return Path(_string(value, path))


def _bool(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{path} must be a boolean")
    return value


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{path} must be a list")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{path} must be a nonempty string")
    return value


def _float_tuple(value: Any, size: int, path: str) -> tuple[float, ...]:
    values = _list(value, path)
    if len(values) != size or any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in values
    ):
        raise TypeError(f"{path} must contain exactly {size} numbers")
    return tuple(float(item) for item in values)


def _string_tuple(
    value: Any,
    size: int | None,
    path: str,
) -> tuple[str, ...]:
    values = _list(value, path)
    if size is not None and len(values) != size:
        raise ValueError(f"{path} must contain exactly {size} names")
    return tuple(_string(item, path) for item in values)


def _require_exact_fields(
    value: dict[str, Any],
    expected: frozenset[str],
    *,
    path: str,
) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{path} has unknown fields: {', '.join(unknown)}")


def _foot_forward_axes(
    value: Any,
) -> tuple[
    tuple[str, tuple[float, float, float]],
    tuple[str, tuple[float, float, float]],
]:
    items = _list(value, "foot_forward_axes")
    if len(items) != 2:
        raise ValueError("foot_forward_axes must contain exactly two entries")
    result = []
    for index, item in enumerate(items):
        path = f"foot_forward_axes[{index}]"
        data = _mapping(item, path=path)
        _require_exact_fields(
            data,
            frozenset({"body_name", "axis"}),
            path=path,
        )
        result.append(
            (
                _string(data["body_name"], f"{path}.body_name"),
                _float_tuple(data["axis"], 3, f"{path}.axis"),
            )
        )
    return tuple(result)


def _foot_support_geometries(
    value: Any,
) -> tuple[FootSupportGeometry, FootSupportGeometry]:
    items = _list(value, "foot_support_geometries")
    if len(items) != 2:
        raise ValueError("foot_support_geometries must contain exactly two entries")
    result = []
    for index, item in enumerate(items):
        path = f"foot_support_geometries[{index}]"
        data = _mapping(item, path=path)
        _require_exact_fields(
            data,
            frozenset({"body_name", "geom_names"}),
            path=path,
        )
        result.append(
            FootSupportGeometry(
                body_name=_string(data["body_name"], f"{path}.body_name"),
                geom_names=_string_tuple(
                    data["geom_names"], None, f"{path}.geom_names"
                ),
            )
        )
    return tuple(result)


def _payload_attachment(value: Any, index: int) -> PayloadAttachment:
    path = f"payload_attachments[{index}]"
    data = _mapping(value, path=path)
    _require_exact_fields(
        data,
        frozenset({"link_name", "radius_m", "center_m"}),
        path=path,
    )
    radius = data["radius_m"]
    if isinstance(radius, bool) or not isinstance(radius, (int, float)):
        raise TypeError(f"{path}.radius_m must be a number")
    return PayloadAttachment(
        link_name=_string(data["link_name"], f"{path}.link_name"),
        radius_m=float(radius),
        center_m=_float_tuple(data["center_m"], 3, f"{path}.center_m"),
    )


def _joint_actuator(value: Any, index: int) -> JointActuatorDefinition:
    path = f"actuators[{index}]"
    data = _mapping(value, path=path)
    _require_exact_fields(
        data,
        frozenset(
            {
                "joint_name",
                "mode",
                "default_pos",
                "soft_limit",
                "gains",
                "effort_limit",
            }
        ),
        path=path,
    )
    mode = _string(data["mode"], f"{path}.mode")
    if mode not in {"policy", "hold"}:
        raise ValueError(f"{path}.mode must be 'policy' or 'hold'")
    default_pos = _optional_float(data["default_pos"], f"{path}.default_pos")
    if default_pos is None:
        raise TypeError(f"{path}.default_pos must be a number")
    soft_limit = (
        None
        if data["soft_limit"] is None
        else _float_tuple(data["soft_limit"], 2, f"{path}.soft_limit")
    )
    return JointActuatorDefinition(
        joint_name=_string(data["joint_name"], f"{path}.joint_name"),
        mode=mode,
        default_pos=default_pos,
        soft_limit=soft_limit,
        gains=_float_tuple(data["gains"], 2, f"{path}.gains"),
        effort_limit=_optional_float(data["effort_limit"], f"{path}.effort_limit"),
    )


def _optional_float(value: Any, path: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be a number or null")
    return float(value)


def urdf_coupled_actuators(
    path: Path, joint_order: tuple[str, ...]
) -> tuple[CoupledActuatorGroup, ...]:
    """Translate URDF differential reductions into native fixed-tendon rows."""
    groups = []
    for transmission in ET.parse(path).getroot().findall("transmission"):
        kind = transmission.findtext("type", default="")
        if kind != "transmission_interface/DifferentialTransmission":
            raise ValueError(f"unsupported URDF transmission type {kind!r}")
        members = {}
        for tag in ("actuator", "joint"):
            for index in (1, 2):
                role = f"{tag}{index}"
                member = transmission.find(f"{tag}[@role='{role}']")
                if member is None or not member.get("name"):
                    raise ValueError(f"URDF transmission is missing {role}")
                reduction = float(member.findtext("mechanical_reduction", default="1"))
                if not math.isfinite(reduction) or reduction == 0:
                    raise ValueError(f"URDF transmission has invalid {role} reduction")
                if float(member.findtext("offset", default="0")) != 0:
                    raise ValueError(
                        "native differential tendons require zero URDF joint offsets"
                    )
                members[role] = (member.attrib["name"], reduction)
        j1, r1 = members["joint1"]
        j2, r2 = members["joint2"]
        ordered = tuple(joint for joint in joint_order if joint in (j1, j2))
        if len(ordered) != 2:
            raise ValueError("URDF transmission must reference two configured joints")
        actuators = []
        for index, sign in ((1, 1), (2, -1)):
            motor, reduction = members[f"actuator{index}"]
            row = {j1: reduction * r1, j2: sign * reduction * r2}
            actuators.append(
                CoupledActuator(
                    tendon_name=motor,
                    logical_joint_name=ordered[index - 1],
                    joint_coefficients=tuple((joint, row[joint]) for joint in ordered),
                )
            )
        groups.append(
            CoupledActuatorGroup(
                name=transmission.get("name", ""), actuators=tuple(actuators)
            )
        )
    return tuple(groups)
