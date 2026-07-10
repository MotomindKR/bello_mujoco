import pytest

from bello_mujoco.actuators import ActuatorConfig, JointActuatorConfig


def test_actuator_config_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="invalid mode"):
        ActuatorConfig(
            joints={
                "joint": JointActuatorConfig(
                    mode="floating",  # type: ignore[arg-type]
                )
            }
        )


def test_actuator_config_requires_effort_limit_for_known_bello_joints() -> None:
    with pytest.raises(ValueError, match="must provide effort_limit"):
        ActuatorConfig(joints={"left_hip_pitch_joint": JointActuatorConfig()})
