import xml.etree.ElementTree as ET
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import pytest
from hydra import compose, initialize_config_dir

import bello_mujoco.assets_pipeline.prepare as prepare_module
from bello_mujoco.actuators import ActuatorConfig, JointActuatorConfig
from bello_mujoco.assets_pipeline.mesh import (
    JointAvailabilityConfig,
    fix_disabled_joints,
    preserve_fixed_link_bodies,
    replace_end_effectors_with_spheres,
)
from bello_mujoco.assets_pipeline.prepare import (
    AssetPrepConfig,
    PreparedAsset,
    asset_config_hash,
    prepare_asset,
    read_prepared_asset_manifest,
    resolved_output_root,
    validate_actuator_joint_contract,
    write_prepared_asset_manifest,
)
from bello_mujoco.hydra_config import config_path, structured_config


def test_end_effector_sphere_replacement_prunes_both_subtrees() -> None:
    root = ET.Element("robot", {"name": "test"})
    for side in ("r", "l"):
        for suffix in ("wrist_link", "eef_link", "gripper_link"):
            ET.SubElement(root, "link", {"name": f"{side}_{suffix}"})
    for side, name in (("r", "right"), ("l", "left")):
        revolute = ET.SubElement(
            root, "joint", {"name": f"{name}_wrist_yaw_joint", "type": "revolute"}
        )
        ET.SubElement(revolute, "origin", {"xyz": "0 0 -0.102", "rpy": "0 0 0"})
        ET.SubElement(revolute, "parent", {"link": f"{side}_wrist_link"})
        ET.SubElement(revolute, "child", {"link": f"{side}_eef_link"})
        ET.SubElement(revolute, "axis", {"xyz": "0 0 1"})
        fixed = ET.SubElement(
            root, "joint", {"name": f"{name}_gripper_joint", "type": "fixed"}
        )
        ET.SubElement(fixed, "parent", {"link": f"{side}_eef_link"})
        ET.SubElement(fixed, "child", {"link": f"{side}_gripper_link"})

    replace_end_effectors_with_spheres(root, "both")

    for side, name in (("r", "right"), ("l", "left")):
        links = {link.get("name") for link in root.findall("link")}
        joints = {joint.get("name") for joint in root.findall("joint")}
        sphere_joint = root.find(f"./joint[@name='{side}_end_effector_sphere_fixed']")
        sphere = root.find(f"./link[@name='{side}_end_effector_sphere_link']")

        assert f"{side}_wrist_link" in links
        assert f"{side}_eef_link" not in links
        assert f"{side}_gripper_link" not in links
        assert f"{name}_wrist_yaw_joint" not in joints
        assert f"{name}_gripper_joint" not in joints
        assert sphere_joint is not None
        assert sphere_joint.find("parent").get("link") == f"{side}_wrist_link"
        assert sphere_joint.find("origin").get("xyz") == "0 0 -0.122"
        assert sphere is not None
        assert sphere.find("./collision/geometry/sphere").get("radius") == "0.0325"


def test_end_effector_sphere_replacement_rejects_removed_modes() -> None:
    with pytest.raises(ValueError, match="must be 'none' or 'both'"):
        replace_end_effectors_with_spheres(ET.Element("robot"), "right")


def test_mujoco_urdf_preserves_fixed_link_bodies() -> None:
    root = ET.Element("robot", {"name": "test"})

    preserve_fixed_link_bodies(root)

    compiler = root.find("./mujoco/compiler")
    assert compiler is not None
    assert compiler.attrib["fusestatic"] == "false"


def test_prepared_full_body_preserves_pelvis_body(
    full_body_asset: PreparedAsset,
) -> None:
    model = mujoco.MjModel.from_xml_path(str(full_body_asset.mjcf_path))

    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link") >= 0
    sensor_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SENSOR,
        "bello_acceleration_stage_sensor",
    )
    assert sensor_id >= 0
    assert model.sensor_type[sensor_id] == mujoco.mjtSensor.mjSENS_ACCELEROMETER


def test_prepared_both_sphere_mode_removes_wrist_subtrees(tmp_path: Path) -> None:
    prepared = prepare_asset(
        AssetPrepConfig(
            output_root=tmp_path,
            end_effector_sphere_mode="both",
        )
    )
    model = mujoco.MjModel.from_xml_path(str(prepared.mjcf_path))

    assert model.nu == 29
    for side, prefix in (("left", "l"), ("right", "r")):
        assert (
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"{side}_wrist_yaw_joint",
            )
            == -1
        )
        assert (
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                f"{prefix}_end_effector_sphere_link",
            )
            >= 0
        )


def test_morphology_pruning_removes_only_absent_actuator_configs() -> None:
    config = AssetPrepConfig(
        end_effector_sphere_mode="both",
        joint_availability=JointAvailabilityConfig(waist_roll=False),
    )

    assert len(config.actuator_config.joints) == 28
    assert "right_wrist_yaw_joint" not in config.actuator_config.joints
    assert "right_finger_right_slide_joint" not in config.actuator_config.joints
    assert "waist_roll_joint" not in config.actuator_config.joints
    assert "waist_yaw_joint" in config.actuator_config.joints


def test_actuator_contract_rejects_missing_and_unknown_joint_configs() -> None:
    joint = JointActuatorConfig(effort_limit=1.0)
    with pytest.raises(ValueError, match="missing actuator config"):
        validate_actuator_joint_contract(ActuatorConfig(joints={}), ("left",))
    with pytest.raises(ValueError, match="absent from the prepared morphology"):
        validate_actuator_joint_contract(
            ActuatorConfig(joints={"left": joint, "typo": joint}),
            ("left",),
        )


def test_prepared_asset_always_has_ankle_differential_tendons(
    full_body_asset: PreparedAsset,
) -> None:
    model = mujoco.MjModel.from_xml_path(str(full_body_asset.mjcf_path))

    assert model.ntendon == 4
    assert {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, index)
        for index in range(model.ntendon)
    } == {
        "left_ankle_motor_1",
        "left_ankle_motor_2",
        "right_ankle_motor_1",
        "right_ankle_motor_2",
    }

    root = ET.parse(full_body_asset.mjcf_path).getroot()
    expected_coefficients = {
        "left_ankle_motor_1": (-1.0, 1.0),
        "left_ankle_motor_2": (1.0, 1.0),
        "right_ankle_motor_1": (-1.0, 1.0),
        "right_ankle_motor_2": (1.0, 1.0),
    }
    for tendon_name, coefficients in expected_coefficients.items():
        fixed = root.find(f"./tendon/fixed[@name='{tendon_name}']")
        assert fixed is not None
        assert tuple(float(joint.get("coef")) for joint in fixed.findall("joint")) == (
            coefficients
        )


@pytest.mark.integration
def test_visualizer_recipe_produces_loadable_flattened_viewer_mjcf(
    tmp_path: Path,
) -> None:
    with initialize_config_dir(
        version_base=None,
        config_dir=config_path("prepare_assets"),
    ):
        recipe = structured_config(
            AssetPrepConfig,
            compose(config_name="visualizer", overrides=[f"output_root={tmp_path}"]),
        )

    assert recipe.visual_mesh_face_budget == 20_000
    assert recipe.end_effector_sphere_mode == "both"
    prepared = prepare_asset(recipe)
    model = mujoco.MjModel.from_xml_path(str(prepared.viewer_mjcf_path))
    root = ET.parse(prepared.viewer_mjcf_path).getroot()

    assert model.nbody > 1
    assert root.find("include") is None
    assert root.findall("./asset/mesh")
    assert root.findall(".//geom[@type='mesh']")
    assert len(root.findall("./tendon/fixed")) == 4
    for joint in ("left_wrist_yaw_joint", "right_wrist_yaw_joint"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint) == -1
    for body in ("l_end_effector_sphere_link", "r_end_effector_sphere_link"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body) >= 0


def test_prepared_asset_always_actuates_ankle_tendons(
    full_body_asset: PreparedAsset,
) -> None:
    model = mujoco.MjModel.from_xml_path(str(full_body_asset.mjcf_path))
    actuator_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(model.nu)
    }

    for side in ("left", "right"):
        assert f"{side}_ankle_pitch_joint_pos" not in actuator_names
        assert f"{side}_ankle_roll_joint_pos" not in actuator_names
        for motor_index in (1, 2):
            name = f"{side}_ankle_motor_{motor_index}_pos"
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            assert actuator_id >= 0
            assert model.actuator_trntype[actuator_id] == mujoco.mjtTrn.mjTRN_TENDON
            assert model.actuator_forcelimited[actuator_id]
            np.testing.assert_allclose(
                model.actuator_forcerange[actuator_id], [-36.0, 36.0]
            )
            assert model.actuator_gainprm[actuator_id, 0] == pytest.approx(11.0)
            assert model.actuator_biasprm[actuator_id, 2] == pytest.approx(-0.5)


@pytest.mark.parametrize("asset", ["lower_body", "upper_body"])
def test_asset_config_rejects_removed_prepared_variants(asset: str) -> None:
    with pytest.raises(ValueError, match="asset must be 'full_body'"):
        AssetPrepConfig(asset=asset)


def test_asset_config_rejects_removed_foot_mesh_collision_mode() -> None:
    with pytest.raises(ValueError, match="collision_mode must be 'mesh' or 'box'"):
        AssetPrepConfig(collision_mode="foot-mesh")


@pytest.mark.parametrize("mode", ["right", "left", "both_articulated"])
def test_asset_config_rejects_removed_end_effector_modes(mode: str) -> None:
    with pytest.raises(
        ValueError,
        match="end_effector_sphere_mode must be 'none' or 'both'",
    ):
        AssetPrepConfig(end_effector_sphere_mode=mode)


def test_default_output_root_is_deterministic_tmp_cache() -> None:
    config = AssetPrepConfig(asset="full_body")

    assert resolved_output_root(config) == resolved_output_root(config)
    assert resolved_output_root(config).parent == (
        Path(tempfile.gettempdir()) / "bello-mujoco-assets"
    )


def test_explicit_output_root_overrides_tmp_cache() -> None:
    root = Path("/tmp/custom-bello-assets")

    assert resolved_output_root(AssetPrepConfig(output_root=root)) == root


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

    assert asset_config_hash(base) != asset_config_hash(changed)
    fixed_head = AssetPrepConfig(
        asset="full_body",
        mesh_face_budget=1000,
        joint_availability=JointAvailabilityConfig(
            neck_yaw=False,
            head_pitch=False,
        ),
    )
    assert asset_config_hash(base) != asset_config_hash(fixed_head)


def test_generated_actuators_leave_control_unlimited_and_limit_force_from_effort(
    full_body_asset: PreparedAsset,
) -> None:
    model = mujoco.MjModel.from_xml_path(str(full_body_asset.mjcf_path))

    assert not np.any(model.actuator_ctrllimited)
    assert np.all(model.actuator_ctrlrange[:, 0] < model.actuator_ctrlrange[:, 1])
    assert np.all(model.actuator_forcelimited)
    assert np.all(model.actuator_forcerange[:, 0] < 0.0)
    assert np.all(model.actuator_forcerange[:, 1] > 0.0)


def test_disabled_joints_are_fixed_in_the_generated_urdf() -> None:
    root = ET.Element("robot")
    for joint_name in (
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "neck_yaw_joint",
        "head_pitch_joint",
    ):
        joint = ET.SubElement(root, "joint", {"name": joint_name, "type": "revolute"})
        ET.SubElement(joint, "axis", {"xyz": "0 0 1"})
        ET.SubElement(joint, "limit", {"lower": "-1", "upper": "1"})

    fix_disabled_joints(
        root,
        JointAvailabilityConfig(
            waist_yaw=True,
            waist_roll=False,
            waist_pitch=False,
            neck_yaw=False,
            head_pitch=False,
        ),
    )

    assert root.find("./joint[@name='waist_yaw_joint']").get("type") == "revolute"
    for joint_name in (
        "waist_roll_joint",
        "waist_pitch_joint",
        "neck_yaw_joint",
        "head_pitch_joint",
    ):
        joint = root.find(f"./joint[@name='{joint_name}']")
        assert joint.get("type") == "fixed"
        assert joint.find("axis") is None
        assert joint.find("limit") is None


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
    prepared = prepare_asset(
        AssetPrepConfig(output_root=tmp_path, collision_mode="box")
    )
    model = mujoco.MjModel.from_xml_path(str(prepared.viewer_mjcf_path))
    visual = (model.geom_contype == 0) & (model.geom_conaffinity == 0)
    collision = ~visual

    assert np.any(model.geom_type[collision] == mujoco.mjtGeom.mjGEOM_BOX)
    assert np.allclose(model.geom_rgba[collision, 3], 0.0)
    assert np.any(model.geom_type[visual] == mujoco.mjtGeom.mjGEOM_MESH)
    assert np.allclose(model.geom_rgba[visual, 3], 1.0)


def test_clean_asset_builds_are_byte_reproducible_within_one_process(
    tmp_path: Path,
) -> None:
    first = prepare_asset(AssetPrepConfig(output_root=tmp_path / "first"))
    second = prepare_asset(AssetPrepConfig(output_root=tmp_path / "second"))

    for first_path, second_path in (
        (first.urdf_path, second.urdf_path),
        (first.mjcf_path, second.mjcf_path),
        (first.viewer_mjcf_path, second.viewer_mjcf_path),
    ):
        assert first_path.read_bytes() == second_path.read_bytes()


def test_asset_cache_version_invalidates_precanonical_generator_outputs(
    monkeypatch,
) -> None:
    config = AssetPrepConfig()
    current_key = asset_config_hash(config)
    assert prepare_module.ASSET_CACHE_VERSION >= 8

    monkeypatch.setattr(prepare_module, "ASSET_CACHE_VERSION", 7)

    assert asset_config_hash(config) != current_key


def test_actuator_config_does_not_expose_mutable_joint_mapping() -> None:
    config = ActuatorConfig(joints={"joint": JointActuatorConfig(default_pos=0.1)})
    by_name = config.by_name
    by_name["joint"] = JointActuatorConfig(default_pos=0.2)

    assert config.joints["joint"].default_pos == 0.1
    with pytest.raises(TypeError, match="FrozenDict"):
        config.joints["joint"] = JointActuatorConfig(default_pos=0.3)
