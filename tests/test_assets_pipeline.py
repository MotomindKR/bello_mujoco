import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bello_mujoco.assets_pipeline.mesh import replace_end_effectors_with_spheres
from bello_mujoco.assets_pipeline.prepare import (
    AssetPrepConfig,
    PreparedAsset,
    prepare_asset,
    read_prepared_asset_manifest,
    resolved_output_root,
    write_prepared_asset_manifest,
)
from bello_mujoco.envs.locomotion import (
    BaseHeightDoneConfig,
    BelloLocomotionEnv,
    DoneConfig,
    RollPitchDoneConfig,
)
from bello_mujoco.envs.rewards import (
    AngVelXYRewardConfig,
    BaseHeightRewardConfig,
    FlatOrientationRewardConfig,
    LinVelZRewardConfig,
    RewardConfig,
    TrackAngVelZRewardConfig,
    TrackLinVelXYRewardConfig,
)


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
    model = SimpleNamespace(
        mj_model=SimpleNamespace(nu=29, nq=40, nv=39, nbody=4, ngeom=5),
        policy_ctrl_indices=np.arange(6, dtype=np.int32),
        hold_ctrl_indices=np.asarray([], dtype=np.int32),
        actuator_qpos_indices=np.arange(29, dtype=np.int32),
        actuator_qvel_indices=np.arange(29, dtype=np.int32),
        policy_qpos_indices=np.arange(6, dtype=np.int32),
        policy_qvel_indices=np.arange(6, dtype=np.int32),
        policy_action_scale=np.ones(6, dtype=np.float32),
        foot_body_ids=np.asarray([2, 3], dtype=np.int32),
        foot_geom_ids=np.asarray([2, 3], dtype=np.int32),
        terrain_geom_ids=np.asarray([0], dtype=np.int32),
        nonfoot_robot_geom_ids=np.asarray([1, 4], dtype=np.int32),
        default_qpos=np.zeros(40, dtype=np.float32),
        default_qvel=np.zeros(39, dtype=np.float32),
        default_ctrl=np.zeros(29, dtype=np.float32),
    )
    env = BelloLocomotionEnv(
        model,
        reward_config=RewardConfig(
            track_lin_vel_xy=TrackLinVelXYRewardConfig(enabled=False),
            track_ang_vel_z=TrackAngVelZRewardConfig(enabled=False),
            lin_vel_z=LinVelZRewardConfig(enabled=False),
            ang_vel_xy=AngVelXYRewardConfig(enabled=False),
            flat_orientation=FlatOrientationRewardConfig(enabled=False),
            base_height=BaseHeightRewardConfig(enabled=False),
        ),
        done_config=DoneConfig(
            base_height=BaseHeightDoneConfig(enabled=False),
            roll_pitch=RollPitchDoneConfig(enabled=False),
        ),
    )

    assert env.action_size == 6
    assert env.observation_size == 102


def test_lower_body_rejects_end_effector_sphere_mode() -> None:
    config = AssetPrepConfig(asset="lower_body", end_effector_sphere_mode="both")

    with pytest.raises(ValueError, match="requires arm joints"):
        prepare_asset(config)


def test_default_output_root_is_deterministic_tmp_cache() -> None:
    config = AssetPrepConfig(asset="full_body")

    assert resolved_output_root(config, "full_body") == resolved_output_root(
        config, "full_body"
    )
    assert resolved_output_root(config, "full_body").parent == Path(
        "/tmp/bello-mujoco-assets"
    )


def test_explicit_output_root_overrides_tmp_cache() -> None:
    root = Path("/tmp/custom-bello-assets")

    assert (
        resolved_output_root(
            AssetPrepConfig(asset="full_body", output_root=root), "full_body"
        )
        == root
    )


def test_cached_manifest_requires_referenced_mesh_files(tmp_path: Path) -> None:
    asset_dir = tmp_path / "full_body"
    mesh_dir = asset_dir / "meshes"
    mjcf_dir = asset_dir / "mjcf"
    urdf_dir = asset_dir / "urdf"
    mesh_dir.mkdir(parents=True)
    mjcf_dir.mkdir(parents=True)
    urdf_dir.mkdir(parents=True)
    urdf_path = urdf_dir / "bello_full_body.urdf"
    mjcf_path = mjcf_dir / "bello_full_body.xml"
    viewer_mjcf_path = mjcf_dir / "bello_full_body_viewer.xml"
    manifest_path = asset_dir / "asset_manifest.json"
    urdf_path.write_text("<robot />")
    mjcf_path.write_text(
        '<mujoco><asset><mesh name="body" file="../meshes/body.stl" /></asset></mujoco>'
    )
    viewer_mjcf_path.write_text("<mujoco><asset /></mujoco>")
    write_prepared_asset_manifest(
        PreparedAsset("full_body", urdf_path, mjcf_path, viewer_mjcf_path, 1, 1),
        manifest_path,
    )
    assert (
        read_prepared_asset_manifest(
            "full_body", urdf_path, mjcf_path, viewer_mjcf_path, manifest_path
        )
        is None
    )
