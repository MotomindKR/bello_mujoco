from types import SimpleNamespace

import mujoco
import mink
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from bello_mujoco.models.target_limits import TargetLimits
from bello_mujoco.motion.calibration import (
    CalibratedRetargeter,
    SourceContactConfig,
    calibration_frames,
    planted_feet,
)
from bello_mujoco.motion.lafan import LafanPreparationConfig
from bello_mujoco.motion.retarget_limits import PlantedFootTask
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
    mask = planted_feet(p, q, 50)
    assert mask[5:10, 0].all()
    assert not mask[15:30].any()
    assert not mask[:, 1].any()
    assert not mask[36:, 0].any()
    assert not planted_feet(p, q, 50, SourceContactConfig(min_seconds=2)).any()


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
    assert not planted_feet(positions, quaternions, 50).any()


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


def test_first_plant_anchor_is_captured_after_floor_recovery(
    synthetic_calibration, monkeypatch
):
    retargeter, targets = synthetic_calibration
    # Later standing frames determine alignment; the initial fit is below the plane.
    targets["pelvis"][0][:, 2] = [0.7, 0.7, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    captured = []
    original = PlantedFootTask.update

    def observe(task, configuration, mask):
        if not captured:
            captured.append(
                min(
                    geom_lower_z(task.model, configuration.data, int(geom))
                    for geom in retargeter.model.foot_geom_ids
                )
            )
        return original(task, configuration, mask)

    monkeypatch.setattr(PlantedFootTask, "update", observe)
    retargeter.retarget(targets, np.ones((10, 2), dtype=bool))
    assert captured[0] >= retargeter.config.contacts.floor_clearance_m - 1e-6


def test_invalid_contact_clock_rejected(synthetic_calibration):
    retargeter, targets = synthetic_calibration
    with pytest.raises(ValueError, match="frames-by-two bool"):
        retargeter.retarget(targets, np.zeros((6, 2), dtype=bool))


def test_material_point_jacobian_and_anchor_lifecycle(synthetic_calibration):
    calibrated, _ = synthetic_calibration
    model = calibrated.model
    configuration = mink.Configuration(model.mj_model)
    pose = model.default_qpos.copy()
    pose[3:7] = Rotation.from_euler("xyz", [0.2, -0.1, 0.15]).as_quat(scalar_first=True)
    configuration.update(pose)
    task = PlantedFootTask(model, 1000.0, 3, 0.001)
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
