"""Configurable Bello joint actuator parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal

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
    mode: ActuatorMode = "policy"
    default_pos: float = 0.0
    action_scale: float | None = None
    soft_limit: tuple[float, float] | None = None
    gains: tuple[float, float] = (40.0, 2.0)


def default_joint_configs() -> dict[str, JointActuatorConfig]:
    return {
        "left_hip_pitch_joint": JointActuatorConfig(
            mode="policy",
            default_pos=-0.325,
            action_scale=0.35,
            soft_limit=(-1.41372, 1.41372),
            gains=(180.0, 5.0),
        ),
        "right_hip_pitch_joint": JointActuatorConfig(
            mode="policy",
            default_pos=0.325,
            action_scale=0.35,
            soft_limit=(-1.41372, 1.41372),
            gains=(180.0, 5.0),
        ),
        "left_hip_roll_joint": JointActuatorConfig(
            mode="policy",
            default_pos=0.0,
            action_scale=0.25,
            soft_limit=(-1.41372, 1.41372),
            gains=(120.0, 5.0),
        ),
        "right_hip_roll_joint": JointActuatorConfig(
            mode="policy",
            default_pos=0.0,
            action_scale=0.25,
            soft_limit=(-1.41372, 1.41372),
            gains=(120.0, 5.0),
        ),
        "left_hip_yaw_joint": JointActuatorConfig(
            mode="policy",
            default_pos=0.125,
            action_scale=0.25,
            soft_limit=(-1.41372, 1.41372),
            gains=(120.0, 5.0),
        ),
        "right_hip_yaw_joint": JointActuatorConfig(
            mode="policy",
            default_pos=-0.125,
            action_scale=0.25,
            soft_limit=(-1.41372, 1.41372),
            gains=(120.0, 5.0),
        ),
        "left_knee_joint": JointActuatorConfig(
            mode="policy",
            default_pos=0.45,
            action_scale=0.45,
            soft_limit=(0.10472, 1.98968),
            gains=(200.0, 6.0),
        ),
        "right_knee_joint": JointActuatorConfig(
            mode="policy",
            default_pos=0.45,
            action_scale=0.45,
            soft_limit=(0.10472, 1.98968),
            gains=(200.0, 6.0),
        ),
        "left_ankle_pitch_joint": JointActuatorConfig(
            mode="policy",
            default_pos=0.30,
            action_scale=0.20,
            soft_limit=(-0.44506, 0.96866),
            gains=(40.0, 2.0),
        ),
        "right_ankle_pitch_joint": JointActuatorConfig(
            mode="policy",
            default_pos=0.30,
            action_scale=0.20,
            soft_limit=(-0.44506, 0.96866),
            gains=(40.0, 2.0),
        ),
        "left_ankle_roll_joint": JointActuatorConfig(
            mode="policy",
            default_pos=0.0,
            action_scale=0.05,
            soft_limit=(-0.07857, 0.07857),
            gains=(40.0, 2.0),
        ),
        "right_ankle_roll_joint": JointActuatorConfig(
            mode="policy",
            default_pos=0.0,
            action_scale=0.05,
            soft_limit=(-0.07857, 0.07857),
            gains=(40.0, 2.0),
        ),
        "waist_yaw_joint": JointActuatorConfig(gains=(180.0, 12.0)),
        "waist_roll_joint": JointActuatorConfig(gains=(240.0, 20.0)),
        "waist_pitch_joint": JointActuatorConfig(gains=(240.0, 20.0)),
        "neck_yaw_joint": JointActuatorConfig(gains=(5.5, 0.5)),
        "head_pitch_joint": JointActuatorConfig(gains=(5.5, 0.5)),
        "right_shoulder_pitch_joint": JointActuatorConfig(gains=(70.0, 2.0)),
        "right_shoulder_roll_joint": JointActuatorConfig(gains=(70.0, 2.0)),
        "right_shoulder_yaw_joint": JointActuatorConfig(gains=(20.0, 1.0)),
        "right_elbow_pitch_joint": JointActuatorConfig(gains=(40.0, 2.0)),
        "right_elbow_yaw_joint": JointActuatorConfig(gains=(20.0, 1.0)),
        "right_wrist_pitch_joint": JointActuatorConfig(gains=(10.0, 1.0)),
        "right_wrist_yaw_joint": JointActuatorConfig(gains=(10.0, 1.0)),
        "left_shoulder_pitch_joint": JointActuatorConfig(gains=(70.0, 2.0)),
        "left_shoulder_roll_joint": JointActuatorConfig(gains=(70.0, 2.0)),
        "left_shoulder_yaw_joint": JointActuatorConfig(gains=(20.0, 1.0)),
        "left_elbow_pitch_joint": JointActuatorConfig(gains=(40.0, 2.0)),
        "left_elbow_yaw_joint": JointActuatorConfig(gains=(20.0, 1.0)),
        "left_wrist_pitch_joint": JointActuatorConfig(gains=(10.0, 1.0)),
        "left_wrist_yaw_joint": JointActuatorConfig(gains=(10.0, 1.0)),
        "right_finger_right_slide_joint": JointActuatorConfig(gains=(100.0, 5.0)),
        "right_finger_left_slide_joint": JointActuatorConfig(gains=(100.0, 5.0)),
        "left_finger_right_slide_joint": JointActuatorConfig(gains=(100.0, 5.0)),
        "left_finger_left_slide_joint": JointActuatorConfig(gains=(100.0, 5.0)),
    }


@dataclass(frozen=True)
class ActuatorConfig:
    joints: dict[str, JointActuatorConfig] = field(
        default_factory=default_joint_configs
    )
    default_action_scale_ratio: float = 0.25

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.default_action_scale_ratio)
            or self.default_action_scale_ratio <= 0
        ):
            raise ValueError(
                "default_action_scale_ratio must be positive, got "
                f"{self.default_action_scale_ratio}"
            )
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
            if config.action_scale is not None and (
                not math.isfinite(config.action_scale) or config.action_scale <= 0
            ):
                raise ValueError(
                    f"actuator config for joint {name!r} has invalid action_scale "
                    f"{config.action_scale}; expected a positive finite value"
                )
            if (
                config.soft_limit is not None
                and config.soft_limit[0] >= config.soft_limit[1]
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
        object.__setattr__(self, "joints", FrozenDict(self.joints))

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
