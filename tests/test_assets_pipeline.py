import fcntl
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

import bello_mujoco.assets_pipeline.prepare as prepare_module
from bello_mujoco.actuators import ActuatorConfig, JointActuatorConfig
from bello_mujoco.assets_pipeline.mesh import (
    JointAvailabilityConfig,
    fix_disabled_joints,
    preserve_fixed_link_bodies,
)
from bello_mujoco.assets_pipeline.prepare import (
    AssetPrepConfig,
    PreparedAsset,
    asset_config_hash,
    canonical_physical_asset_config,
    prepare_asset,
    read_prepared_asset_manifest,
    resolved_output_root,
    scale_urdf_total_mass,
    validate_actuator_joint_contract,
    write_prepared_asset_manifest,
)

CANONICAL_TOTAL_MASS_KG = 55.0683575411785


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


def test_prepared_asset_has_canonical_fixed_sphere_hands(
    tmp_path: Path,
) -> None:
    prepared = prepare_asset(AssetPrepConfig(output_root=tmp_path))
    model = mujoco.MjModel.from_xml_path(str(prepared.mjcf_path))

    assert model.nu == 25
    assert np.sum(model.body_mass) == pytest.approx(CANONICAL_TOTAL_MASS_KG)
    for side, prefix in (("left", "l"), ("right", "r")):
        assert (
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"{side}_end_effector_sphere_joint",
            )
            == -1
        )
        body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"{prefix}_end_effector_sphere_link",
        )
        assert body_id >= 0
        assert model.body_mass[body_id] == pytest.approx(0.0)


def test_prepared_asset_attaches_equal_payload_mass_to_sphere_hands(
    tmp_path: Path,
) -> None:
    payload_mass = 10.0 * 0.45359237
    prepared = prepare_asset(
        AssetPrepConfig(
            output_root=tmp_path,
            end_effector_payload_mass_kg=payload_mass,
        )
    )
    model = mujoco.MjModel.from_xml_path(str(prepared.mjcf_path))

    assert np.sum(model.body_mass) == pytest.approx(
        CANONICAL_TOTAL_MASS_KG + 2.0 * payload_mass
    )
    expected_inertia = 0.4 * payload_mass * 0.0325**2
    for prefix in ("l", "r"):
        body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"{prefix}_end_effector_sphere_link",
        )
        assert model.body_mass[body_id] == pytest.approx(payload_mass)
        assert model.body_ipos[body_id] == pytest.approx((0.0, 0.0, -0.02))
        assert model.body_inertia[body_id] == pytest.approx(
            np.full(3, expected_inertia), rel=2.0e-6
        )


@pytest.mark.parametrize(
    "payload_mass",
    (-1.0, float("nan"), float("inf")),
)
def test_asset_config_rejects_invalid_end_effector_payload_mass(
    payload_mass: float,
) -> None:
    with pytest.raises(ValueError, match="end_effector_payload_mass_kg"):
        AssetPrepConfig(end_effector_payload_mass_kg=payload_mass)


def test_morphology_pruning_removes_only_absent_actuator_configs() -> None:
    config = AssetPrepConfig(
        joint_availability=JointAvailabilityConfig(waist_yaw=False),
    )

    assert len(config.actuator_config.joints) == 24
    assert "right_wrist_pitch_joint" in config.actuator_config.joints
    assert "right_finger_right_slide_joint" not in config.actuator_config.joints
    assert "waist_roll_joint" not in config.actuator_config.joints
    assert "waist_yaw_joint" not in config.actuator_config.joints


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
    recipe = replace(
        canonical_physical_asset_config(),
        output_root=tmp_path,
        visual_mesh_face_budget=20_000,
    )

    assert recipe.visual_mesh_face_budget == 20_000
    prepared = prepare_asset(recipe)
    model = mujoco.MjModel.from_xml_path(str(prepared.viewer_mjcf_path))
    root = ET.parse(prepared.viewer_mjcf_path).getroot()

    assert model.nbody > 1
    assert root.find("include") is None
    assert root.findall("./asset/mesh")
    assert root.findall(".//geom[@type='mesh']")
    assert len(root.findall("./tendon/fixed")) == 4
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


@pytest.mark.parametrize("target", [0.0, -1.0, float("nan"), float("inf")])
def test_asset_config_rejects_invalid_target_total_mass(target: float) -> None:
    with pytest.raises(ValueError, match="target_total_mass_kg"):
        AssetPrepConfig(target_total_mass_kg=target)


def test_urdf_mass_scaling_preserves_mass_distribution_and_inertia_ratio(
    tmp_path: Path,
) -> None:
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text(
        """<robot name="test">
  <link name="first"><inertial><mass value="2"/><inertia ixx="2" iyy="3" izz="4" ixy="0.1" ixz="0.2" iyz="0.3"/></inertial></link>
  <link name="second"><inertial><mass value="3"/><inertia ixx="5" iyy="6" izz="7" ixy="0.4" ixz="0.5" iyz="0.6"/></inertial></link>
</robot>"""
    )

    scale_urdf_total_mass(urdf_path, 60.0)

    root = ET.parse(urdf_path).getroot()
    masses = [float(element.attrib["value"]) for element in root.findall(".//mass")]
    first_inertia = root.find("./link[@name='first']/inertial/inertia")
    assert first_inertia is not None
    assert sum(masses) == pytest.approx(60.0)
    assert masses == pytest.approx([24.0, 36.0])
    assert float(first_inertia.attrib["ixx"]) == pytest.approx(24.0)
    assert float(first_inertia.attrib["ixy"]) == pytest.approx(1.2)


def test_urdf_mass_scaling_keeps_fixed_massless_links_massless(
    tmp_path: Path,
) -> None:
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text(
        """<robot name="test">
  <link name="base"><inertial><mass value="2"/><inertia ixx="2" iyy="3" izz="4" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="marker"/>
  <joint name="marker_joint" type="fixed"><parent link="base"/><child link="marker"/></joint>
</robot>"""
    )

    scale_urdf_total_mass(urdf_path, 10.0)

    root = ET.parse(urdf_path).getroot()
    mass = root.find("./link[@name='base']/inertial/mass")
    assert mass is not None
    assert float(mass.attrib["value"]) == 10.0
    assert root.find("./link[@name='marker']/inertial") is None


def test_canonical_bilateral_link_inertias_are_symmetric(
    full_body_asset: PreparedAsset,
) -> None:
    root = ET.parse(full_body_asset.urdf_path).getroot()
    pairs = {
        ("left_hip_pitch_link", "right_hip_pitch_link"): (-1, -1, 1),
        ("left_hip_roll_link", "right_hip_roll_link"): (-1, -1, 1),
        ("left_hip_yaw_link", "right_hip_yaw_link"): (-1, 1, 1),
        ("left_knee_link", "right_knee_link"): (-1, 1, 1),
        ("left_ankle_pitch_link", "right_ankle_pitch_link"): (-1, 1, 1),
        ("left_ankle_roll_link", "right_ankle_roll_link"): (-1, 1, 1),
        ("l_shoulder_pitch_link", "r_shoulder_pitch_link"): (-1, -1, 1),
        ("l_shoulder_roll_link", "r_shoulder_roll_link"): (-1, -1, 1),
        ("l_upper_arm_link", "r_upper_arm_link"): (-1, -1, 1),
        ("l_elbow_link", "r_elbow_link"): (-1, -1, 1),
        ("l_lower_arm_link", "r_lower_arm_link"): (-1, -1, 1),
        ("l_wrist_link", "r_wrist_link"): (-1, -1, 1),
    }

    incoming_joint = {
        joint.find("child").attrib["link"]: joint
        for joint in root.findall("joint")
        if joint.find("child") is not None
    }
    world_transforms: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def origin_transform(element: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
        if element is None:
            return np.eye(3), np.zeros(3)
        xyz = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        roll, pitch, yaw = np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
        cx, cy, cz = np.cos((roll, pitch, yaw))
        sx, sy, sz = np.sin((roll, pitch, yaw))
        rotation = np.asarray(
            (
                (cy * cz, cz * sx * sy - cx * sz, sx * sz + cx * cz * sy),
                (cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx),
                (-sy, cy * sx, cx * cy),
            )
        )
        return rotation, xyz

    def world_transform(name: str) -> tuple[np.ndarray, np.ndarray]:
        cached = world_transforms.get(name)
        if cached is not None:
            return cached
        joint = incoming_joint.get(name)
        if joint is None:
            result = (np.eye(3), np.zeros(3))
        else:
            parent = joint.find("parent")
            assert parent is not None
            parent_rotation, parent_position = world_transform(parent.attrib["link"])
            local_rotation, local_position = origin_transform(joint.find("origin"))
            result = (
                parent_rotation @ local_rotation,
                parent_position + parent_rotation @ local_position,
            )
        world_transforms[name] = result
        return result

    def inertial_properties(name: str) -> tuple[float, np.ndarray, np.ndarray]:
        inertial = root.find(f"./link[@name='{name}']/inertial")
        assert inertial is not None
        origin = inertial.find("origin")
        mass = inertial.find("mass")
        inertia = inertial.find("inertia")
        assert origin is not None and mass is not None and inertia is not None
        local_rotation, center = origin_transform(origin)
        values = inertia.attrib
        tensor = np.asarray(
            (
                (float(values["ixx"]), float(values["ixy"]), float(values["ixz"])),
                (float(values["ixy"]), float(values["iyy"]), float(values["iyz"])),
                (float(values["ixz"]), float(values["iyz"]), float(values["izz"])),
            )
        )
        link_rotation, link_position = world_transform(name)
        world_center = link_position + link_rotation @ center
        world_inertia_rotation = link_rotation @ local_rotation
        world_inertia = world_inertia_rotation @ tensor @ world_inertia_rotation.T
        return float(mass.attrib["value"]), world_center, world_inertia

    for (left_name, right_name), signs in pairs.items():
        left_mass, left_center, left_inertia = inertial_properties(left_name)
        right_mass, right_center, right_inertia = inertial_properties(right_name)
        transform = np.diag(signs)

        assert left_mass == pytest.approx(right_mass, rel=0.0, abs=1.0e-15)
        np.testing.assert_allclose(left_center, transform @ right_center, atol=1.0e-15)
        np.testing.assert_allclose(
            left_inertia,
            transform @ right_inertia @ transform,
            atol=1.0e-15,
        )


def test_default_output_root_is_deterministic_bounded_user_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    config = AssetPrepConfig(asset="full_body")

    assert resolved_output_root(config) == resolved_output_root(config)
    assert resolved_output_root(config).parent == tmp_path / "bello-mujoco" / "assets"


def test_explicit_output_root_overrides_tmp_cache() -> None:
    root = Path("/tmp/custom-bello-assets")

    assert resolved_output_root(AssetPrepConfig(output_root=root)) == root


def test_default_asset_cache_prunes_inactive_variants_and_keeps_leases(
    tmp_path: Path,
) -> None:
    limit = prepare_module.DEFAULT_ASSET_CACHE_LIMIT
    variants = [tmp_path / f"{index:016x}" for index in range(limit + 2)]
    for variant in variants:
        variant.mkdir()

    lease_path = tmp_path / f"{variants[0].name}.lease"
    with lease_path.open("w") as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_SH)
        prepare_module.prune_default_asset_cache(tmp_path, keep=variants[-1])

    remaining = {path for path in tmp_path.iterdir() if path.is_dir()}
    assert len(remaining) == limit
    assert variants[0] in remaining
    assert variants[-1] in remaining


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
    mesh_path = mesh_dir / "body.stl"
    mesh_path.write_bytes(b"mesh")
    write_prepared_asset_manifest(
        PreparedAsset("full_body", urdf_path, mjcf_path, viewer_mjcf_path, 1, 1),
        manifest_path,
        "cache-a",
    )
    mesh_path.unlink()
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


def test_cached_manifest_requires_matching_cache_key_and_content(
    tmp_path: Path,
) -> None:
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
    (mjcf_dir / "bello_full_body.scene.runtime.xml").write_text("<mujoco />")
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
    urdf_path.write_text("<robot name='changed' />")
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


def test_asset_cache_key_tracks_prep_config() -> None:
    base = AssetPrepConfig(asset="full_body", mesh_face_budget=1000)
    changed = AssetPrepConfig(asset="full_body", mesh_face_budget=500)

    assert asset_config_hash(base) != asset_config_hash(changed)
    assert asset_config_hash(base) != asset_config_hash(
        AssetPrepConfig(asset="full_body", target_total_mass_kg=60.0)
    )
    assert asset_config_hash(base) != asset_config_hash(
        AssetPrepConfig(asset="full_body", end_effector_payload_mass_kg=4.5359237)
    )
    assert asset_config_hash(base) != asset_config_hash(
        AssetPrepConfig(asset="full_body", enforce_policy_effort_limits=False)
    )
    fixed_waist = AssetPrepConfig(
        asset="full_body",
        mesh_face_budget=1000,
        joint_availability=JointAvailabilityConfig(waist_yaw=False),
    )
    assert asset_config_hash(base) != asset_config_hash(fixed_waist)


def test_generated_actuators_leave_control_unlimited_and_limit_force_from_effort(
    full_body_asset: PreparedAsset,
) -> None:
    model = mujoco.MjModel.from_xml_path(str(full_body_asset.mjcf_path))

    assert not np.any(model.actuator_ctrllimited)
    assert np.all(model.actuator_ctrlrange[:, 0] < model.actuator_ctrlrange[:, 1])
    assert np.all(model.actuator_forcelimited)
    assert np.all(model.actuator_forcerange[:, 0] < 0.0)
    assert np.all(model.actuator_forcerange[:, 1] > 0.0)


def test_policy_effort_limits_can_be_disabled_without_losing_peak_metadata(
    tmp_path: Path,
) -> None:
    prepared = prepare_asset(
        AssetPrepConfig(
            output_root=tmp_path,
            enforce_policy_effort_limits=False,
        )
    )
    model = mujoco.MjModel.from_xml_path(str(prepared.mjcf_path))
    actuator_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(model.nu)
    ]
    policy_indices = np.asarray(
        [index for index, name in enumerate(actuator_names) if name.endswith("_pos")]
    )
    hold_indices = np.asarray(
        [index for index, name in enumerate(actuator_names) if name.endswith("_hold")]
    )

    assert policy_indices.size > 0
    assert hold_indices.size > 0
    assert not np.any(model.actuator_forcelimited[policy_indices])
    assert np.all(model.actuator_forcelimited[hold_indices])
    assert np.all(model.actuator_forcerange[:, 0] < 0.0)
    assert np.all(model.actuator_forcerange[:, 1] > 0.0)


def test_disabled_joints_are_fixed_in_the_generated_urdf() -> None:
    root = ET.Element("robot")
    joint = ET.SubElement(
        root, "joint", {"name": "waist_yaw_joint", "type": "revolute"}
    )
    ET.SubElement(joint, "axis", {"xyz": "0 0 1"})
    ET.SubElement(joint, "limit", {"lower": "-1", "upper": "1"})

    fix_disabled_joints(
        root,
        JointAvailabilityConfig(waist_yaw=False),
    )

    joint = root.find("./joint[@name='waist_yaw_joint']")
    assert joint is not None
    assert joint.get("type") == "fixed"
    assert joint.find("axis") is None
    assert joint.find("limit") is None


def test_actuator_force_observation_is_clamped_by_force_range(
    full_body_asset: PreparedAsset,
) -> None:
    model = mujoco.MjModel.from_xml_path(str(full_body_asset.mjcf_path))
    data = mujoco.MjData(model)
    joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "right_wrist_pitch_joint"
    )
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

    assert data.actuator_force[actuator_id] == pytest.approx(14.0)
    assert data.qfrc_actuator[dof_id] == pytest.approx(14.0)


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


def test_asset_cache_key_tracks_generator_revision(monkeypatch) -> None:
    config = AssetPrepConfig()
    current_key = asset_config_hash(config)

    monkeypatch.setattr(
        prepare_module,
        "asset_generator_fingerprint",
        lambda: "f" * 40,
    )
    revision_key = asset_config_hash(config)
    assert revision_key != current_key


def test_actuator_config_does_not_expose_mutable_joint_mapping() -> None:
    config = ActuatorConfig(joints={"joint": JointActuatorConfig(default_pos=0.1)})
    by_name = config.by_name
    by_name["joint"] = JointActuatorConfig(default_pos=0.2)

    assert config.joints["joint"].default_pos == 0.1
    with pytest.raises(TypeError, match="FrozenDict"):
        config.joints["joint"] = JointActuatorConfig(default_pos=0.3)
