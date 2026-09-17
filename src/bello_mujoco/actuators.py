"""Configurable Bello joint actuator parameters."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Literal

from bello_mujoco.robot import AssetName, robot_definition

ActuatorMode = Literal["policy", "hold"]
VALID_ACTUATOR_MODES = {"policy", "hold"}


class FrozenDict(dict):
    def _raise_frozen(self, *args, **kwargs) -> None:
        raise TypeError("FrozenDict cannot be mutated")

    __setitem__ = _raise_frozen
    __delitem__ = _raise_frozen
    clear = _raise_frozen
    pop = _raise_frozen
    popitem = _raise_frozen
    setdefault = _raise_frozen
    update = _raise_frozen


@dataclass(frozen=True)
class JointActuatorConfig:
    mode: ActuatorMode = "hold"
    default_pos: float = 0.0
    soft_limit: tuple[float, float] | None = None
    gains: tuple[float, float] = (40.0, 2.0)
    effort_limit: float | None = None


def default_joint_configs(
    asset: AssetName,
) -> dict[str, JointActuatorConfig]:
    return {
        actuator.joint_name: JointActuatorConfig(
            mode=actuator.mode,
            default_pos=actuator.default_pos,
            soft_limit=actuator.soft_limit,
            gains=actuator.gains,
            effort_limit=actuator.effort_limit,
        )
        for actuator in robot_definition(asset).actuators
    }


@dataclass(frozen=True)
class ActuatorConfig:
    joints: dict[str, JointActuatorConfig]

    @classmethod
    def for_asset(
        cls,
        asset: AssetName,
        *,
        mode: ActuatorMode | None = None,
    ) -> ActuatorConfig:
        joints = default_joint_configs(asset)
        if mode is not None:
            joints = {
                name: JointActuatorConfig(
                    mode=mode,
                    default_pos=config.default_pos,
                    soft_limit=config.soft_limit,
                    gains=config.gains,
                    effort_limit=config.effort_limit,
                )
                for name, config in joints.items()
            }
        return cls(joints=joints)

    def __post_init__(self) -> None:
        for name, config in self.joints.items():
            if config.mode not in VALID_ACTUATOR_MODES:
                raise ValueError(
                    f"actuator config for joint {name!r} has invalid mode "
                    f"{config.mode!r}; expected one of {sorted(VALID_ACTUATOR_MODES)}"
                )
            if len(config.gains) != 2:
                raise ValueError(
                    f"actuator config for joint {name!r} must provide [kp, kv] gains"
                )
            if not all(math.isfinite(gain) for gain in config.gains):
                raise ValueError(
                    f"actuator config for joint {name!r} has nonfinite gains "
                    f"{config.gains}"
                )
            if config.soft_limit is not None and (
                len(config.soft_limit) != 2
                or config.soft_limit[0] >= config.soft_limit[1]
            ):
                raise ValueError(
                    f"actuator config for joint {name!r} has invalid soft_limit "
                    f"{config.soft_limit}; expected lower < upper"
                )
            if config.soft_limit is not None and not all(
                math.isfinite(limit) for limit in config.soft_limit
            ):
                raise ValueError(
                    f"actuator config for joint {name!r} has nonfinite soft_limit "
                    f"{config.soft_limit}"
                )
            if config.soft_limit is not None and not (
                config.soft_limit[0] <= config.default_pos <= config.soft_limit[1]
            ):
                raise ValueError(f"default_pos must fall within soft_limit: {name}")
            if config.effort_limit is not None and (
                not math.isfinite(config.effort_limit) or config.effort_limit <= 0
            ):
                raise ValueError(
                    f"actuator config for joint {name!r} has invalid effort_limit "
                    f"{config.effort_limit}; expected a positive finite value"
                )
        object.__setattr__(self, "joints", FrozenDict(self.joints))

    def with_soft_limits(
        self, limits: dict[str, tuple[float, float]]
    ) -> ActuatorConfig:
        unknown = set(limits) - set(self.joints)
        if unknown:
            raise ValueError(f"unknown joints in soft limits: {sorted(unknown)}")
        return ActuatorConfig(
            joints={
                name: replace(config, soft_limit=limits[name])
                if name in limits
                else config
                for name, config in self.joints.items()
            }
        )

    @property
    def by_name(self) -> dict[str, JointActuatorConfig]:
        return dict(self.joints)

    @property
    def default_joint_pos_by_name(self) -> dict[str, float]:
        return {name: config.default_pos for name, config in self.joints.items()}

    @property
    def policy_joint_names(self) -> set[str]:
        return {name for name, config in self.joints.items() if config.mode == "policy"}

    @property
    def hold_joint_names(self) -> set[str]:
        return {name for name, config in self.joints.items() if config.mode == "hold"}
