import xml.etree.ElementTree as ET
from types import SimpleNamespace

import pytest

from bello_mujoco.assets_pipeline.mesh import replace_end_effectors_with_spheres
from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig, prepare_asset
from bello_mujoco.envs.locomotion import BelloLocomotionEnv


def test_end_effector_sphere_replacement_prunes_selected_subtree() -> None:
    root = ET.Element("robot", {"name": "test"})
    for link_name in ("r_wrist_link", "r_eef_link", "r_gripper_link"):
        ET.SubElement(root, "link", {"name": link_name})
    revolute = ET.SubElement(
        root, "joint", {"name": "right_wrist_yaw_joint", "type": "revolute"}
    )
    ET.SubElement(revolute, "origin", {"xyz": "5e-06 -0.0275 -0.102", "rpy": "0 0 0"})
    ET.SubElement(revolute, "parent", {"link": "r_wrist_link"})
    ET.SubElement(revolute, "child", {"link": "r_eef_link"})
    fixed = ET.SubElement(
        root, "joint", {"name": "right_gripper_joint", "type": "fixed"}
    )
    ET.SubElement(fixed, "parent", {"link": "r_eef_link"})
    ET.SubElement(fixed, "child", {"link": "r_gripper_link"})

    replace_end_effectors_with_spheres(root, "right")

    links = {link.get("name") for link in root.findall("link")}
    joints = {joint.get("name") for joint in root.findall("joint")}
    sphere_joint = root.find("./joint[@name='r_end_effector_sphere_fixed']")
    sphere_link = root.find("./link[@name='r_end_effector_sphere_link']")

    assert "r_wrist_link" in links
    assert "r_eef_link" not in links
    assert "r_gripper_link" not in links
    assert "right_wrist_yaw_joint" not in joints
    assert "right_gripper_joint" not in joints
    assert sphere_joint is not None
    assert sphere_joint.find("parent").attrib["link"] == "r_wrist_link"
    assert sphere_joint.find("origin").attrib["xyz"] == "5e-06 -0.0275 -0.122"
    assert sphere_link is not None
    assert sphere_link.find("./collision/geometry/sphere").attrib["radius"] == "0.0325"


def test_observation_size_tracks_reduced_actuator_count() -> None:
    env = BelloLocomotionEnv(SimpleNamespace(mj_model=SimpleNamespace(nu=29)))

    assert env.action_size == 29
    assert env.observation_size == 96


def test_lower_body_rejects_end_effector_sphere_mode() -> None:
    config = AssetPrepConfig(asset="down_body", end_effector_sphere_mode="both")

    with pytest.raises(ValueError, match="requires arm joints"):
        prepare_asset(config)
