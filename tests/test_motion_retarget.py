from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from bello_mujoco.models.target_limits import TargetLimits
from bello_mujoco.motion import retarget
from bello_mujoco.motion.retarget import clip_from_poses, pose_velocities


@pytest.fixture
def model():
    mj_model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="plane" size="0 0 .1"/>'
        '<body name="root"><freejoint/>'
        '<geom type="sphere" size=".1" pos=".2 0 0"/>'
        '<body name="arm" pos=".3 0 0"><joint name="joint" axis="0 0 1"/>'
        '<geom type="sphere" size=".1" pos=".15 0 0"/>'
        "</body></body></worldbody></mujoco>"
    )
    return SimpleNamespace(
        mj_model=mj_model,
        actuator_qpos_indices=np.array([7]),
        actuator_qvel_indices=np.array([6]),
        actuator_coordinate_transform=np.eye(1),
        actuator_joint_names=("joint",),
        target_limits=TargetLimits(
            np.array([[-2.0, 2.0]]), np.array([[-2.0, 2.0]]), np.eye(1), np.ones(1)
        ),
        robot=SimpleNamespace(asset="synthetic", foot_body_names=("arm",)),
        foot_geom_ids=np.array([2]),
    )


def test_cached_body_twists_are_at_link_origins_with_offset_centres_of_mass(model):
    mj_model = model.mj_model
    poses = np.tile(mj_model.qpos0, (3, 1))
    poses[:, 7] = [0, 0.02, 0.04]
    clip, _ = clip_from_poses(model, ("root", "arm"), poses, 50, 0.03)
    np.testing.assert_allclose(clip.body_lin_vel, 0, atol=1e-7)
    np.testing.assert_allclose(clip.body_ang_vel[:, 1, 2], 1)
    data = mujoco.MjData(mj_model)
    data.qpos[:], data.qvel[:] = clip.qpos[0], clip.qvel[0]
    mujoco.mj_forward(mj_model, data)
    com = np.zeros(6)
    mujoco.mj_objectVelocity(mj_model, data, mujoco.mjtObj.mjOBJ_BODY, 2, com, 0)
    assert abs(com[4]) > 0.1


def test_offline_retarget_aligns_ground_preserving_joint_motion_and_source_clock(
    model, monkeypatch
):
    poses = np.tile(model.mj_model.qpos0, (4, 1))
    poses[:, 2] = [-0.1, 0.2, 0.5, 0.2]
    poses[:, 7] = [0.0, 0.1, -0.1, 0.0]
    calls = []

    def solve(frame):
        calls.append(frame)
        return SimpleNamespace(qpos=poses[frame.sequence].copy())

    monkeypatch.setattr(
        retarget,
        "GmrRetargeter",
        lambda *a, **k: SimpleNamespace(
            profile=SimpleNamespace(source_body_names=("pelvis",)), retarget=solve
        ),
    )
    monkeypatch.setattr(
        retarget,
        "motion_task_definition",
        lambda asset: SimpleNamespace(beyondmimic_tracked_body_names=("root", "arm")),
    )
    position = np.arange(12).reshape(4, 3)
    quaternion = np.tile([1.0, 0.0, 0.0, 0.0], (4, 1))
    clip, report = retarget.retarget_gmr(
        model,
        {"pelvis": (position, quaternion)},
        fps=50,
        human_height_m=1.8,
        contact_height_m=0.03,
    )
    expected = poses.copy()
    expected[:, 2] += 0.2
    np.testing.assert_allclose(clip.qpos, expected)
    np.testing.assert_allclose(
        clip.qvel, pose_velocities(model.mj_model, poses, 50), atol=1e-6
    )
    assert report["ground_alignment_offset_m"] == pytest.approx(0.2)
    assert report["worst_floor_penetration_m"] >= -1e-8
    assert len(calls) == len(poses)
    assert [frame.source_timestamp_ns for frame in calls] == [
        1,
        20000001,
        40000001,
        60000001,
    ]
    np.testing.assert_array_equal(
        [frame.transforms["pelvis"][0] for frame in calls], position
    )


def test_ground_alignment_uses_rotated_collision_surfaces_and_preserves_flight(model):
    poses = np.tile(model.mj_model.qpos0, (3, 1))
    poses[:, 2] = [0.3, 0.9, 0.4]
    poses[:, 3:7] = [np.sqrt(0.5), 0, np.sqrt(0.5), 0]
    aligned, offset = retarget.align_poses_to_ground(model.mj_model, poses)
    # Rotating the arm's offset sphere puts its bottom .55 m below the root.
    assert offset == pytest.approx(0.25)
    np.testing.assert_allclose(np.diff(aligned[:, 2]), np.diff(poses[:, 2]))
    np.testing.assert_array_equal(aligned[:, 3:], poses[:, 3:])
    np.testing.assert_allclose(poses[:, 2], [0.3, 0.9, 0.4])


def test_reference_velocities_use_central_differences_in_the_current_root_frame():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom size=".1"/>'
        '<body><joint name="joint"/><geom size=".1"/></body></body></worldbody></mujoco>'
    )
    poses = np.tile(model.qpos0, (3, 1))
    poses[:, 0] = [0.0, 0.02, 0.08]
    poses[:, 7] = [0.0, 0.04, 0.16]
    quaternions = np.empty((3, 4))
    for index, angles in enumerate(
        ([0.2, -0.3, 0.1], [-0.1, 0.5, 0.2], [0.4, 0.1, -0.2])
    ):
        axis = np.asarray(angles)
        mujoco.mju_axisAngle2Quat(
            quaternions[index], axis / np.linalg.norm(axis), np.linalg.norm(axis)
        )
    poses[:, 3:7] = quaternions
    velocity = pose_velocities(model, poses, 50)
    np.testing.assert_allclose(velocity[:, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(velocity[:, 6], [2.0, 4.0, 6.0])
    # Compare the world angular twist against log(R_right R_left^T) / dt.
    from scipy.spatial.transform import Rotation

    rotations = Rotation.from_quat(quaternions, scalar_first=True)
    for frame, (left, right) in enumerate(((0, 1), (0, 2), (1, 2))):
        expected = (rotations[right] * rotations[left].inv()).as_rotvec() / (
            (right - left) / 50
        )
        np.testing.assert_allclose(
            rotations[frame].apply(velocity[frame, 3:6]), expected, atol=1e-12
        )
