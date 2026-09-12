import pytest

from bello_mujoco.actuators import ActuatorConfig, JointActuatorConfig
from bello_mujoco.models.compiler import ModelConfig


def test_actuator_config_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="invalid mode"):
        ActuatorConfig(
            joints={
                "joint": JointActuatorConfig(
                    mode="floating",  # type: ignore[arg-type]
                )
            }
        )


def test_asset_config_requires_robot_effort_limits_in_joint_overrides() -> None:
    defaults = ActuatorConfig.for_asset("bello")
    joints = dict(defaults.joints)
    joints["left_hip_pitch_joint"] = JointActuatorConfig()

    with pytest.raises(ValueError, match="must preserve effort limits"):
        ModelConfig(actuator_config=ActuatorConfig(joints=joints))


def test_asset_config_rejects_conflicting_mode_and_joint_map() -> None:
    with pytest.raises(ValueError, match="must match actuator_mode"):
        ModelConfig(
            asset="bello",
            actuator_mode="policy",
            actuator_config=ActuatorConfig.for_asset("bello"),
        )


def test_standalone_actuator_config_has_no_hidden_robot_contract() -> None:
    config = ActuatorConfig(joints={"joint": JointActuatorConfig()})

    assert config.joints["joint"].effort_limit is None
