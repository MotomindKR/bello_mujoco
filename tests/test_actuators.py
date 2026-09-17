import pytest
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from bello_mujoco.actuators import ActuatorConfig, JointActuatorConfig
from bello_mujoco.models.compiler import ModelConfig
from bello_mujoco.models.urdf import configured_joint_range


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


def test_task_soft_limits_preserve_other_actuator_properties_and_input():
    original = ActuatorConfig(
        joints={"hinge": JointActuatorConfig(default_pos=0.2, gains=(7.0, 1.0))}
    )
    limited = original.with_soft_limits({"hinge": (-0.4, 0.8)})
    assert original.joints["hinge"].soft_limit is None
    assert limited.joints["hinge"].soft_limit == (-0.4, 0.8)
    assert limited.joints["hinge"].gains == original.joints["hinge"].gains
    assert limited.joints["hinge"].default_pos == original.joints["hinge"].default_pos
    with pytest.raises(ValueError, match="unknown joints"):
        original.with_soft_limits({"missing": (-1.0, 1.0)})


@pytest.mark.parametrize(
    "bounds", [(0.5, 1.0), (1.0, -1.0), (0.0,), (0.0, float("nan"))]
)
def test_task_soft_limits_reject_invalid_ranges_or_excluded_nominal(bounds):
    config = ActuatorConfig(joints={"hinge": JointActuatorConfig()})
    with pytest.raises(ValueError):
        config.with_soft_limits({"hinge": bounds})


def test_compiler_does_not_expand_physical_range_or_modify_joint_limits():
    root = ET.fromstring(
        '<mujoco><worldbody><body><joint name="hinge" range="-2 2"/></body></worldbody></mujoco>'
    )
    config = ActuatorConfig(joints={"hinge": JointActuatorConfig()})
    assert configured_joint_range(root, "hinge", config) == (-2.0, 2.0)
    assert configured_joint_range(
        root, "hinge", config.with_soft_limits({"hinge": (-0.5, 1.0)})
    ) == (-0.5, 1.0)
    assert root.find(".//joint").get("range") == "-2 2"
    with pytest.raises(ValueError, match="physical joint range"):
        configured_joint_range(
            root, "hinge", config.with_soft_limits({"hinge": (-3.0, 1.0)})
        )


def test_model_task_limits_decode_freeze_and_round_trip(monkeypatch):
    from dataclasses import asdict
    from bello_mujoco.configuration import dataclass_from_dict
    from bello_mujoco.models import compiler

    monkeypatch.setattr(
        compiler,
        "robot_definition",
        lambda _: SimpleNamespace(
            actuators=[SimpleNamespace(joint_name="hinge", effort_limit=None)]
        ),
    )
    monkeypatch.setattr(
        ActuatorConfig,
        "for_asset",
        classmethod(
            lambda cls, asset, mode=None: cls(
                joints={"hinge": JointActuatorConfig(mode=mode or "hold")}
            )
        ),
    )
    config = dataclass_from_dict(
        ModelConfig,
        {"actuator_mode": "policy", "joint_soft_limits": {"hinge": [-0.4, 0.8]}},
        strict=True,
    )
    assert config.actuator_config.joints["hinge"].mode == "policy"
    assert config.actuator_config.joints["hinge"].soft_limit == (-0.4, 0.8)
    assert dataclass_from_dict(ModelConfig, asdict(config), strict=True) == config
    with pytest.raises(TypeError):
        config.joint_soft_limits["hinge"] = (-1.0, 1.0)
