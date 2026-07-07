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
