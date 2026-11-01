from types import SimpleNamespace
from dataclasses import replace
import xml.etree.ElementTree as ET

import mujoco
import mink
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from bello_mujoco.models.target_limits import TargetLimits
from bello_mujoco.motion.calibration import (
    CalibratedRetargeter,
    KinematicRetargetConfig,
    SourceContactConfig,
    calibration_frames,
    stationary_contacts,
    source_contact_labels,
)
from bello_mujoco.motion.lafan import LafanPreparationConfig
from bello_mujoco.motion.retarget_limits import PlantedContactTask, rigid_support_geoms
from bello_mujoco.sim import geom_lower_z


def test_source_contact_labels_exclude_flight_motion_and_rolling():
    p = np.zeros((50, 2, 3))
    q = np.zeros((50, 2, 4))
    q[..., 0] = 1
    p[15:30, :, 2] = 0.3
    p[:, 1, 0] = np.arange(50) / 50
    q[35:, 0] = Rotation.from_euler("x", (np.arange(15) * 0.1)[:, None]).as_quat(
        scalar_first=True
    )
    mask = stationary_contacts(p, q, 50)
    assert mask[5:10, 0].all()
    assert not mask[15:30].any()
    assert not mask[:, 1].any()
    assert not mask[36:, 0].any()
    assert not stationary_contacts(p, q, 50, SourceContactConfig(min_seconds=2)).any()


def test_full_thigh_rotation_targets_the_last_hip_hinge():
    table = {
        "left_hip_roll_link": ["left_hip", 10, 5, [0, 0, 0], [1, 0, 0, 0]],
        "knee": ["left_knee", 10, 5, [0, 0, 0], [1, 0, 0, 0]],
    }
    assert calibration_frames(table)[0][0] == "left_hip_yaw_link"
    assert calibration_frames(table)[1][0] == "knee"


def test_stationary_raised_feet_are_not_relabelled_as_grounded():
    positions = np.zeros((20, 2, 3))
    positions[:, :, 2] = 0.4
    quaternions = np.tile([1.0, 0, 0, 0], (20, 2, 1))
    assert not stationary_contacts(positions, quaternions, 50).any()


def test_source_wrists_have_explicit_height_thresholds_and_order():
    q = np.tile([1.0, 0, 0, 0], (30, 1))
    targets = {
        name: (np.tile([0.0, 0, height], (30, 1)), q.copy())
        for name, height in (
            ("left_foot", 1.5),
            ("right_foot", 1.5),
            ("left_wrist", 0.1),
            ("right_wrist", 0.4),
        )
    }
    labels = source_contact_labels(
        targets, 50, KinematicRetargetConfig(support_mode="hands_and_feet")
    )
    assert labels.shape == (30, 4)
    assert labels[:, 2].all()
    assert not labels[:, [0, 1, 3]].any()
    assert not source_contact_labels(targets, 50, KinematicRetargetConfig()).any()
    del targets["left_wrist"]
    with pytest.raises(ValueError, match="missing joints"):
        source_contact_labels(
            targets, 50, KinematicRetargetConfig(support_mode="hands_and_feet")
        )


def test_height_band_excludes_hovering_after_a_low_stationary_support():
    position = np.zeros((60, 1, 3))
    position[:20, 0, 2] = 0.04
    position[20:, 0, 2] = 0.10
    quaternion = np.tile([1.0, 0, 0, 0], (60, 1, 1))
    config = SourceContactConfig(height_m=0.12, height_band_m=0.025)
    labels = stationary_contacts(position, quaternion, 50, config)
    assert labels[3:15].all()
    assert not labels[25:].any()
    position[..., 2] = 0.4
    assert not stationary_contacts(position, quaternion, 50, config).any()


@pytest.mark.parametrize("band", [0.0, -0.01, float("nan"), 0.2])
def test_invalid_source_height_band_is_rejected(band):
    with pytest.raises(ValueError, match="height band"):
        SourceContactConfig(height_m=0.12, height_band_m=band)


@pytest.fixture
def synthetic_calibration(tmp_path):
    native = mujoco.MjModel.from_xml_string("""
        <mujoco><compiler angle="radian"/><worldbody>
        <geom type="plane" size="2 2 .1"/>
        <body name="root"><freejoint/><geom type="sphere" size=".05"/>
        <body name="left" pos="0 .15 -.5"><joint name="a" range="-1 1"/>
        <geom type="sphere" size=".1"/></body>
        <body name="right" pos="0 -.15 -.5"><joint name="b" range="-1 1"/>
        <geom type="sphere" size=".1"/></body>
        </body></worldbody></mujoco>
    """)
    source = tmp_path / "robot.urdf"
    source.write_text(
        '<robot><joint name="a"><limit velocity="8"/></joint>'
        '<joint name="b"><limit velocity="8"/></joint></robot>'
    )
    default = native.qpos0.copy()
    default[2] = 0.8
    model = SimpleNamespace(
        mj_model=native,
        default_qpos=default,
        actuator_qpos_indices=np.array([7, 8]),
        actuator_qvel_indices=np.array([6, 7]),
        actuator_joint_names=("a", "b"),
        actuator_coordinate_transform=np.eye(2),
        target_limits=TargetLimits(
            np.tile([-1.0, 1.0], (2, 1)),
            np.tile([-1.0, 1.0], (2, 1)),
            np.eye(2),
            np.ones(2),
        ),
        robot=SimpleNamespace(
            asset="bello_mini",
            foot_body_names=("left", "right"),
            source_urdf_path=lambda root: source,
        ),
        foot_body_ids=np.array([2, 3]),
        foot_geom_ids=np.array([2, 3]),
        robot_geom_ids=np.array([1, 2, 3]),
        terrain_geom_ids=np.array([0]),
    )
    table = {
        frame: [source, 100, 10, [0, 0, 0], [1, 0, 0, 0]]
        for frame, source in (
            ("root", "pelvis"),
            ("left", "left_foot"),
            ("right", "right_foot"),
        )
    }
    backend = SimpleNamespace(
        human_root_name="pelvis",
        ik_match_table1=table,
        ik_match_table2=table,
        solver="daqp",
        damping=0.5,
    )
    backend.update_targets = lambda targets: setattr(
        backend, "scaled_human_data", targets
    )
    q = np.tile([1.0, 0, 0, 0], (10, 1))
    p = np.zeros((10, 3))
    p[:, 2] = [0.8, 0.8, 0.9, 1.1, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8]
    # Anatomical foot positions intentionally differ from robot link origins.
    targets = {
        "pelvis": (p, q),
        "left_foot": (p + [0, 0.4, -0.2], q),
        "right_foot": (p + [0, -0.4, -0.2], q),
    }
    return CalibratedRetargeter(model, backend, 50), targets


def test_canonical_chain_alignment_preserves_flight_and_does_not_mutate_source(
    synthetic_calibration,
):
    retargeter, targets = synthetic_calibration
    initial = {n: (p.copy(), q.copy()) for n, (p, q) in targets.items()}
    mask = np.tile(
        [True, True, False, False, False, True, True, True, True, True], (2, 1)
    ).T
    poses, report = retargeter.retarget(targets, mask)
    assert report["ground_alignment"] == "source_planted_soles"
    np.testing.assert_allclose(
        poses[:5, 2] - poses[0, 2], [0, 0, 0.1, 0.3, 0.1], atol=1e-5
    )
    assert np.all(np.diff(poses[5:9, 2]) <= 1e-6)
    assert abs(poses[-1, 2] - poses[0, 2]) <= 0.0011
    np.testing.assert_allclose(poses[:, 7:], 0, atol=1e-8)
    for name, (p, q) in targets.items():
        np.testing.assert_array_equal(p, initial[name][0])
        np.testing.assert_array_equal(q, initial[name][1])


def test_no_foot_labels_uses_reported_body_alignment(synthetic_calibration):
    retargeter, targets = synthetic_calibration
    poses, report = retargeter.retarget(targets, np.zeros((10, 2), dtype=bool))
    assert report["ground_alignment"] == "body_lower_envelope_no_foot_labels"
    assert poses[3, 2] - poses[0, 2] == pytest.approx(0.3, abs=1e-5)


@pytest.mark.parametrize("mode", ["fitted_projection", "constrained_neutral"])
def test_first_plant_anchor_is_captured_after_floor_recovery(
    synthetic_calibration, monkeypatch, mode
):
    retargeter, targets = synthetic_calibration
    retargeter.config = replace(retargeter.config, initialization_mode=mode)
    # Later standing frames determine alignment; the initial fit is below the plane.
    targets["pelvis"][0][:, 2] = [0.7, 0.7, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    captured = []
    original = PlantedContactTask.update

    def observe(task, configuration, mask):
        if not captured:
            captured.append(
                min(
                    geom_lower_z(task.model, configuration.data, int(geom))
                    for geom in retargeter.model.foot_geom_ids
                )
            )
        return original(task, configuration, mask)

    monkeypatch.setattr(PlantedContactTask, "update", observe)
    poses, report = retargeter.retarget(targets, np.ones((10, 2), dtype=bool))
    assert len(poses) == 10
    assert report["initialization_mode"] == mode
    assert captured[0] >= retargeter.config.contacts.floor_clearance_m - 1e-6


def test_invalid_contact_clock_rejected(synthetic_calibration):
    retargeter, targets = synthetic_calibration
    with pytest.raises(ValueError, match="frames-by-supports bool"):
        retargeter.retarget(targets, np.zeros((6, 2), dtype=bool))


def test_material_point_jacobian_and_anchor_lifecycle(synthetic_calibration):
    calibrated, _ = synthetic_calibration
    model = calibrated.model
    configuration = mink.Configuration(model.mj_model)
    pose = model.default_qpos.copy()
    pose[3:7] = Rotation.from_euler("xyz", [0.2, -0.1, 0.15]).as_quat(scalar_first=True)
    configuration.update(pose)
    task = PlantedContactTask(
        model, model.foot_body_ids, calibrated.support_geoms, 1000.0, 3, 0.001
    )
    task.update(configuration, [True, False])
    before = task.compute_error(configuration)
    jacobian = task.compute_jacobian(configuration)
    numerical = np.empty_like(jacobian)
    for dof in range(model.mj_model.nv):
        displaced = pose.copy()
        step = np.zeros(model.mj_model.nv)
        step[dof] = 1
        mujoco.mj_integratePos(model.mj_model, displaced, step, 1e-7)
        configuration.update(displaced)
        numerical[:, dof] = (task.compute_error(configuration) - before) / 1e-7
    np.testing.assert_allclose(jacobian, numerical, atol=1e-6)
    assert task.points[task.bodies[0]][3] == 0
    configuration.update(pose)
    task.update(configuration, [True, False])
    assert task.points[task.bodies[0]][3] == 1
    task.update(configuration, [False, False])
    np.testing.assert_array_equal(task.compute_jacobian(configuration), 0)
    np.testing.assert_array_equal(task.compute_error(configuration), 0)


def test_lafan_source_mode_requires_explicit_inputs(tmp_path):
    common = dict(
        source_directory=tmp_path,
        output_directory=tmp_path / "out",
        training_sequences=("train",),
        evaluation_sequences=("test",),
    )
    with pytest.raises(ValueError, match="require motion and body-model"):
        LafanPreparationConfig(**common, human_source="smplx")
    with pytest.raises(ValueError, match="unused"):
        LafanPreparationConfig(**common, smplx_directory=tmp_path)


def test_hand_support_then_flight_preserves_source_excursion(
    synthetic_calibration, tmp_path
):
    original, targets = synthetic_calibration
    xml_path = tmp_path / "four-supports.xml"
    mujoco.mj_saveLastXML(str(xml_path), original.model.mj_model)
    tree = ET.parse(xml_path)
    root_body = tree.getroot().find("worldbody/body")
    for side, y in (("left", 0.15), ("right", -0.15)):
        body = ET.SubElement(root_body, "body", name=side + "_hand", pos=f"0 {y} .5")
        ET.SubElement(body, "geom", type="sphere", size=".1")
    native = mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot()).decode())
    model = SimpleNamespace(**vars(original.model))
    model.mj_model = native
    model.robot_geom_ids = np.arange(1, native.ngeom)
    backend = SimpleNamespace(**vars(original.backend))
    table = dict(backend.ik_match_table1)
    for side in ("left", "right"):
        table[side + "_hand"] = [side + "_wrist", 100, 10, [0, 0, 0], [1, 0, 0, 0]]
    backend.ik_match_table1 = backend.ik_match_table2 = table
    backend.update_targets = lambda values: setattr(
        backend, "scaled_human_data", values
    )
    rotation = Rotation.from_euler("x", np.pi).as_quat(scalar_first=True)
    targets = {
        name: (p.copy(), np.tile(rotation, (len(p), 1)))
        for name, (p, _) in targets.items()
    }
    for side in ("left", "right"):
        targets[side + "_wrist"] = (
            targets["pelvis"][0] + [0, 0, -0.7],
            np.tile(rotation, (10, 1)),
        )
    retargeter = CalibratedRetargeter(
        model, backend, 50, KinematicRetargetConfig(support_mode="hands_and_feet")
    )
    labels = np.zeros((10, 4), dtype=bool)
    labels[:2, 2:] = True
    poses, report = retargeter.retarget(targets, labels)
    assert report["ground_alignment"] == "source_planted_supports"
    assert report["source_planted_frames"] == [0, 0, 2, 2]
    np.testing.assert_allclose(
        poses[:5, 2] - poses[0, 2], [0, 0, 0.1, 0.3, 0.1], atol=1e-5
    )
    assert report["planted_point_z_error_m"]["max"] < 1e-5


def test_plant_geometry_must_belong_to_its_body(synthetic_calibration):
    retargeter, _ = synthetic_calibration
    with pytest.raises(ValueError, match="collision geoms"):
        PlantedContactTask(
            retargeter.model,
            retargeter.model.foot_body_ids,
            tuple(reversed(retargeter.support_geoms)),
            1000.0,
            3,
            0.001,
        )


def test_new_support_anchor_cannot_drop_a_large_gap_in_three_frames(
    synthetic_calibration,
):
    retargeter, _ = synthetic_calibration
    configuration = mink.Configuration(retargeter.model.mj_model)
    configuration.update(retargeter.model.default_qpos)
    task = PlantedContactTask(
        retargeter.model,
        retargeter.support_bodies,
        retargeter.support_geoms,
        1000.0,
        3,
        0.001,
        maximum_lowering_step_m=0.02,
    )
    task.update(configuration, [True, False])
    initial = task.points[task.bodies[0]][1][2]
    for frame in range(1, 5):
        task.update(configuration, [True, False])
        assert initial - task.points[task.bodies[0]][1][2] == pytest.approx(
            frame * 0.02
        )


def test_support_includes_rigid_tool_geometry_but_excludes_articulated_children():
    native = mujoco.MjModel.from_xml_string("""
        <mujoco><worldbody><geom type="plane" size="2 2 .1"/>
        <body name="root" pos="0 0 1"><freejoint/><geom size=".05"/>
          <body name="wrist" pos="0 0 -.5"><joint name="wrist_joint"/>
            <geom name="wrist_geom" size=".04"/>
            <body name="tool" pos="0 0 -.1"><geom name="tool_geom" size=".025"/></body>
            <body name="moving_finger" pos=".2 0 0"><joint/>
              <geom name="finger_geom" size=".02"/>
            </body>
          </body>
        </body></worldbody></mujoco>
    """)
    wrist = native.body("wrist").id
    geoms = rigid_support_geoms(native, wrist, range(native.ngeom))
    assert set(geoms) == {native.geom("wrist_geom").id, native.geom("tool_geom").id}
    model = SimpleNamespace(mj_model=native, terrain_geom_ids=(0,))
    configuration = mink.Configuration(native)
    task = PlantedContactTask(model, (wrist,), (geoms,), 1000, 3, 0.001)
    task.update(configuration, [True])
    local, _, _, _ = task.points[wrist]
    assert local[2] == pytest.approx(-0.125)
    before = task.compute_error(configuration)
    jacobian = task.compute_jacobian(configuration)
    pose = configuration.q.copy()
    numerical = np.empty_like(jacobian)
    for index in range(native.nv):
        displaced = pose.copy()
        step = np.zeros(native.nv)
        step[index] = 1
        mujoco.mj_integratePos(native, displaced, step, 1e-7)
        configuration.update(displaced)
        numerical[:, index] = (task.compute_error(configuration) - before) / 1e-7
    np.testing.assert_allclose(jacobian, numerical, atol=1e-6)
    with pytest.raises(ValueError, match="collision geoms"):
        PlantedContactTask(
            model, (wrist,), ((native.geom("finger_geom").id,),), 1000, 3, 0.001
        )
