"""Compile canonical descriptions into process-local native MuJoCo models."""

from __future__ import annotations

import atexit
from contextlib import ExitStack
from dataclasses import dataclass, field
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

from bello_mujoco.actuators import ActuatorConfig, ActuatorMode, FrozenDict
from bello_mujoco.models import robot_descriptions_root
from bello_mujoco.models.urdf import ImuSensorConfig, compile_urdf_to_mjcf
from bello_mujoco.robot import AssetName, robot_definition, validate_robot_source_urdf

_WORKSPACES = ExitStack()
atexit.register(_WORKSPACES.close)


@dataclass(frozen=True)
class ModelConfig:
    asset: AssetName = "bello"
    target_total_mass_kg: float | None = None
    end_effector_payload_mass_kg: float = 0.0
    neighbor_exclusion_depth: int = 2
    enforce_policy_effort_limits: bool = True
    imu_config: ImuSensorConfig = ImuSensorConfig()
    actuator_mode: ActuatorMode | None = None
    actuator_config: ActuatorConfig | None = None
    joint_soft_limits: dict[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        definition = robot_definition(self.asset)
        limits = FrozenDict(
            {name: tuple(bounds) for name, bounds in self.joint_soft_limits.items()}
        )
        object.__setattr__(self, "joint_soft_limits", limits)
        derived = ActuatorConfig.for_asset(
            self.asset, mode=self.actuator_mode
        ).with_soft_limits(limits)
        configured = (
            derived
            if self.actuator_config is None
            else self.actuator_config.with_soft_limits(limits)
        )
        if self.actuator_mode is not None and configured != derived:
            raise ValueError(
                "actuator_config must match actuator_mode when both are set"
            )
        object.__setattr__(self, "actuator_config", configured)
        expected = {actuator.joint_name for actuator in definition.actuators}
        actual = set(self.actuator_config.joints)
        if expected != actual:
            raise ValueError(
                f"actuator joints differ from robot: missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
            )
        if self.target_total_mass_kg is not None and (
            not math.isfinite(self.target_total_mass_kg)
            or self.target_total_mass_kg <= 0
        ):
            raise ValueError("target_total_mass_kg must be positive and finite")
        if (
            not math.isfinite(self.end_effector_payload_mass_kg)
            or self.end_effector_payload_mass_kg < 0
        ):
            raise ValueError(
                "end_effector_payload_mass_kg must be finite and nonnegative"
            )
        for actuator in definition.actuators:
            if (
                actuator.effort_limit is not None
                and self.actuator_config.joints[actuator.joint_name].effort_limit
                is None
            ):
                raise ValueError(
                    f"actuator overrides must preserve effort limits: {actuator.joint_name}"
                )


def physical_model_config(
    *, asset: AssetName, imu_enabled: bool = False
) -> ModelConfig:
    return ModelConfig(
        asset=asset,
        imu_config=ImuSensorConfig(
            enabled=imu_enabled, body_name=robot_definition(asset).default_imu_body_name
        ),
    )


@dataclass(frozen=True)
class CompiledRobot:
    name: AssetName
    urdf_path: Path
    mjcf_path: Path
    viewer_mjcf_path: Path


def compile_robot(config: ModelConfig) -> CompiledRobot:
    definition = robot_definition(config.asset)
    source = definition.source_urdf_path(robot_descriptions_root())
    validate_robot_source_urdf(definition, source)
    joint_order = actuated_urdf_joint_names(source)
    validate_actuator_joint_contract(config.actuator_config, joint_order)
    workspace = Path(
        _WORKSPACES.enter_context(TemporaryDirectory(prefix=f"bello-{config.asset}-"))
    )
    physics = workspace / "physics.xml"
    visual = workspace / "visual.xml"
    compile_urdf_to_mjcf(
        source,
        physics,
        visual,
        neighbor_exclusion_depth=config.neighbor_exclusion_depth,
        actuator_config=config.actuator_config,
        actuated_joint_order=joint_order,
        enforce_policy_effort_limits=config.enforce_policy_effort_limits,
        imu_config=config.imu_config,
        robot=definition,
        target_total_mass_kg=config.target_total_mass_kg,
        end_effector_payload_mass_kg=config.end_effector_payload_mass_kg,
    )
    return CompiledRobot(config.asset, source, physics, visual)


def actuated_urdf_joint_names(urdf_path: Path) -> tuple[str, ...]:
    return tuple(
        joint.attrib["name"]
        for joint in ET.parse(urdf_path).getroot().findall("joint")
        if joint.get("type") != "fixed"
    )


def validate_actuator_joint_contract(
    config: ActuatorConfig, joint_order: tuple[str, ...]
) -> None:
    configured, actuated = set(config.joints), set(joint_order)
    if configured != actuated:
        raise ValueError(
            f"actuator config differs from URDF: missing={sorted(actuated - configured)}, unknown={sorted(configured - actuated)}"
        )
