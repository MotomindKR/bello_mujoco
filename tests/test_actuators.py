import pytest

from bello_mujoco.actuators import ActuatorConfig, JointActuatorConfig


def test_public_actuator_defaults_use_canonical_policy_partition() -> None:
    config = ActuatorConfig()

    assert len(config.policy_joint_names) == 13
    assert config.policy_joint_names == {
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
    }
    assert len(config.hold_joint_names) == 22


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
