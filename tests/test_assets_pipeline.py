import xml.etree.ElementTree as ET
import tempfile
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import trimesh

from bello_mujoco.actuators import ActuatorConfig, JointActuatorConfig
from bello_mujoco.assets_pipeline.mesh import replace_end_effectors_with_spheres
from bello_mujoco.assets_pipeline.mjcf import compile_urdf_to_mjcf
from bello_mujoco.assets_pipeline.prepare import (
    AssetPrepConfig,
    PreparedAsset,
    asset_config_hash,
    prepare_asset,
    read_prepared_asset_manifest,
    resolved_output_root,
    write_prepared_asset_manifest,
)
from bello_mujoco.envs.locomotion import (
    BadOrientationDoneConfig,
    BaseHeightDoneConfig,
    BelloLocomotionEnv,
    DoneConfig,
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
        hold_ctrl_indices=np.arange(6, 29, dtype=np.int32),
        actuator_qpos_indices=np.arange(29, dtype=np.int32),
        actuator_qvel_indices=np.arange(29, dtype=np.int32),
        policy_qpos_indices=np.arange(6, dtype=np.int32),
        policy_qvel_indices=np.arange(6, dtype=np.int32),
        policy_action_scale=np.ones(6, dtype=np.float32),
        foot_body_ids=np.asarray([2, 3], dtype=np.int32),
        robot_geom_ids=np.asarray([1, 2, 3, 4], dtype=np.int32),
        foot_geom_ids=np.asarray([2, 3], dtype=np.int32),
        terrain_geom_ids=np.asarray([0], dtype=np.int32),
        nonfoot_robot_geom_ids=np.asarray([1, 4], dtype=np.int32),
        nonfoot_robot_body_ids=np.asarray([1], dtype=np.int32),
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
            bad_orientation=BadOrientationDoneConfig(enabled=False),
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
    assert resolved_output_root(config, "full_body").parent == (
        Path(tempfile.gettempdir()) / "bello-mujoco-assets"
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
        "cache-a",
    )
    assert (
        read_prepared_asset_manifest(
            "full_body",
            urdf_path,
            mjcf_path,
            viewer_mjcf_path,
            manifest_path,
            "cache-a",
        )
        is None
    )


def test_cached_manifest_requires_matching_cache_key(tmp_path: Path) -> None:
    asset_dir = tmp_path / "full_body"
    mjcf_dir = asset_dir / "mjcf"
    urdf_dir = asset_dir / "urdf"
    mjcf_dir.mkdir(parents=True)
    urdf_dir.mkdir(parents=True)
    urdf_path = urdf_dir / "bello_full_body.urdf"
    mjcf_path = mjcf_dir / "bello_full_body.xml"
    viewer_mjcf_path = mjcf_dir / "bello_full_body_viewer.xml"
    manifest_path = asset_dir / "asset_manifest.json"
    urdf_path.write_text("<robot />")
    mjcf_path.write_text("<mujoco><asset /></mujoco>")
    viewer_mjcf_path.write_text("<mujoco><asset /></mujoco>")
    write_prepared_asset_manifest(
        PreparedAsset("full_body", urdf_path, mjcf_path, viewer_mjcf_path, 1, 1),
        manifest_path,
        "cache-a",
    )

    assert (
        read_prepared_asset_manifest(
            "full_body",
            urdf_path,
            mjcf_path,
            viewer_mjcf_path,
            manifest_path,
            "cache-b",
        )
        is None
    )
    assert (
        read_prepared_asset_manifest(
            "full_body",
            urdf_path,
            mjcf_path,
            viewer_mjcf_path,
            manifest_path,
            "cache-a",
        )
        is not None
    )


def test_asset_cache_key_tracks_prep_config() -> None:
    base = AssetPrepConfig(asset="full_body", mesh_face_budget=1000)
    changed = AssetPrepConfig(asset="full_body", mesh_face_budget=500)

    assert asset_config_hash(base, "full_body") != asset_config_hash(
        changed, "full_body"
    )


def test_generated_actuators_leave_control_unlimited_and_limit_force_from_effort(
    full_body_asset: PreparedAsset,
) -> None:
    model = mujoco.MjModel.from_xml_path(str(full_body_asset.mjcf_path))

    assert not np.any(model.actuator_ctrllimited)
    assert np.all(model.actuator_ctrlrange[:, 0] < model.actuator_ctrlrange[:, 1])
    assert np.all(model.actuator_forcelimited)
    assert np.all(model.actuator_forcerange[:, 0] < 0.0)
    assert np.all(model.actuator_forcerange[:, 1] > 0.0)


def test_actuator_force_observation_is_clamped_by_force_range(
    full_body_asset: PreparedAsset,
) -> None:
    model = mujoco.MjModel.from_xml_path(str(full_body_asset.mjcf_path))
    data = mujoco.MjData(model)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "neck_yaw_joint")
    actuator_id = next(
        actuator_id
        for actuator_id, joint_pair in enumerate(model.actuator_trnid)
        if int(joint_pair[0]) == joint_id
    )
    qpos_id = int(model.jnt_qposadr[joint_id])
    dof_id = int(model.jnt_dofadr[joint_id])

    data.qpos[qpos_id] = -1.0
    data.ctrl[actuator_id] = 1.0
    mujoco.mj_forward(model, data)

    assert data.actuator_force[actuator_id] == pytest.approx(5.5)
    assert data.qfrc_actuator[dof_id] == pytest.approx(5.5)


def test_foot_collision_uses_urdf_primitives(full_body_asset: PreparedAsset) -> None:
    root = ET.parse(full_body_asset.urdf_path).getroot()
    model = mujoco.MjModel.from_xml_path(str(full_body_asset.mjcf_path))
    expected = {
        "box_1": (
            mujoco.mjtGeom.mjGEOM_BOX,
            np.asarray([0.053, 0.085035, 0.025]),
        ),
        "box_2": (
            mujoco.mjtGeom.mjGEOM_BOX,
            np.asarray([0.013, 0.02, 0.025]),
        ),
        "box_3": (
            mujoco.mjtGeom.mjGEOM_BOX,
            np.asarray([0.053, 0.020975, 0.025]),
        ),
        "cylinder_1": (
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            np.asarray([0.04, 0.025, 0.0]),
        ),
        "cylinder_2": (
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            np.asarray([0.04, 0.025, 0.0]),
        ),
    }
    expected_rpy = {
        "left": {
            "box_1": np.asarray([1.57078922439, -0.000114262698, np.pi]),
            "box_2": np.asarray([1.78022873463, -0.000114262698, np.pi]),
            "box_3": np.asarray([1.78022873463, -0.000114262698, np.pi]),
            "cylinder_1": np.asarray([1.78022873463, -0.000114262698, np.pi]),
            "cylinder_2": np.asarray([1.78022873463, -0.000114262698, np.pi]),
        },
        "right": {
            "box_1": np.asarray([1.57078924875, 0.0, 0.0]),
            "box_2": np.asarray([1.78022875899, 0.0, 0.0]),
            "box_3": np.asarray([1.78022875899, 0.0, 0.0]),
            "cylinder_1": np.asarray([1.78022875899, 0.0, 0.0]),
            "cylinder_2": np.asarray([1.78022875899, 0.0, 0.0]),
        },
    }

    for side in ("left", "right"):
        assert (
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"{side}_ankle_roll_link_collision",
            )
            == -1
        )
        for suffix, (geom_type, size) in expected.items():
            geom_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"{side}_ankle_roll_link_collision_{suffix}",
            )
            assert geom_id >= 0
            assert model.geom_type[geom_id] == geom_type
            assert np.allclose(model.geom_size[geom_id], size)
            origin = root.find(
                f"./link[@name='{side}_ankle_roll_link']"
                f"/collision[@name='{side}_ankle_roll_link_collision_{suffix}']"
                "/origin"
            )
            assert origin is not None
            rpy = np.fromstring(origin.attrib["rpy"], sep=" ")
            assert np.allclose(rpy, expected_rpy[side][suffix])


def test_box_viewer_model_makes_collision_geoms_transparent(
    tmp_path: Path,
) -> None:
    mesh_dir = tmp_path / "meshes"
    visual_mesh_dir = tmp_path / "visual_meshes"
    urdf_dir = tmp_path / "urdf"
    mjcf_dir = tmp_path / "mjcf"
    mesh_dir.mkdir()
    visual_mesh_dir.mkdir()
    urdf_dir.mkdir()
    mjcf_dir.mkdir()
    trimesh.creation.box(extents=(0.2, 0.1, 0.05)).export(
        mesh_dir / "base_collision.stl"
    )
    trimesh.creation.box(extents=(0.3, 0.2, 0.08)).export(
        visual_mesh_dir / "base_visual.stl"
    )
    urdf_path = urdf_dir / "robot.urdf"
    urdf_path.write_text(
        """
<robot name="box_visual_test">
  <link name="base">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <visual name="base_visual">
      <geometry>
        <mesh filename="../visual_meshes/base_visual.stl"/>
      </geometry>
    </visual>
    <collision name="base_collision">
      <geometry>
        <mesh filename="../meshes/base_collision.stl"/>
      </geometry>
    </collision>
  </link>
</robot>
""".strip()
    )
    mjcf_path = mjcf_dir / "robot_viewer.xml"

    compile_urdf_to_mjcf(
        urdf_path,
        mjcf_path,
        collision_mode="box",
        include_visuals=True,
        actuated_joint_order=(),
    )

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    collision_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "base_collision"
    )
    visual_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "base_visual")

    assert collision_geom_id >= 0
    assert visual_geom_id >= 0
    assert model.geom_type[collision_geom_id] == mujoco.mjtGeom.mjGEOM_BOX
    assert model.geom_rgba[collision_geom_id, 3] == pytest.approx(0.0)
    assert model.geom_type[visual_geom_id] == mujoco.mjtGeom.mjGEOM_MESH
    assert model.geom_contype[visual_geom_id] == 0
    assert model.geom_conaffinity[visual_geom_id] == 0
    assert model.geom_rgba[visual_geom_id, 3] == pytest.approx(1.0)


def test_actuator_config_does_not_expose_mutable_joint_mapping() -> None:
    config = ActuatorConfig(joints={"joint": JointActuatorConfig(default_pos=0.1)})
    by_name = config.by_name
    by_name["joint"] = JointActuatorConfig(default_pos=0.2)

    assert config.joints["joint"].default_pos == 0.1
    with pytest.raises(TypeError, match="FrozenDict"):
        config.joints["joint"] = JointActuatorConfig(default_pos=0.3)
