import pytest

from bello_mujoco.actuators import ActuatorConfig, JointActuatorConfig
from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig


def test_public_actuator_defaults_use_canonical_policy_partition() -> None:
    config = ActuatorConfig.for_asset("full_body")

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
    assert len(config.hold_joint_names) == 12


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
    defaults = ActuatorConfig.for_asset("full_body")
    joints = dict(defaults.joints)
    joints["left_hip_pitch_joint"] = JointActuatorConfig()

    with pytest.raises(ValueError, match="must preserve effort limits"):
        AssetPrepConfig(actuator_config=ActuatorConfig(joints=joints))


def test_asset_config_rejects_conflicting_mode_and_joint_map() -> None:
    with pytest.raises(ValueError, match="must match actuator_mode"):
        AssetPrepConfig(
            asset="full_body",
            actuator_mode="policy",
            actuator_config=ActuatorConfig.for_asset("full_body"),
        )


def test_standalone_actuator_config_has_no_hidden_robot_contract() -> None:
    config = ActuatorConfig(joints={"joint": JointActuatorConfig()})

    assert config.joints["joint"].effort_limit is None
