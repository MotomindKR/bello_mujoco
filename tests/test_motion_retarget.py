from types import SimpleNamespace

import mujoco
import mink
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


def test_offline_retarget_preserves_feasible_solver_poses_and_source_clock(
    model, monkeypatch
):
    poses = np.tile(model.mj_model.qpos0, (4, 1))
    poses[:, 2] = [0.1, 0.2, 0.5, 0.2]
    poses[:, 7] = [0.0, 0.1, -0.1, 0.0]
    calls = []

    def solve(frame):
        calls.append(frame)
        return SimpleNamespace(qpos=poses[frame.sequence].copy())

    monkeypatch.setattr(
        retarget,
        "_OfflineGmrRetargeter",
        lambda *a, **k: SimpleNamespace(
            profile=SimpleNamespace(source_body_names=("pelvis",)),
            retarget=solve,
            extra_floor_passes=0,
            max_floor_projection_m=0.0,
            initial_pose_passes=0,
            initial_pose_residual=0.0,
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
    np.testing.assert_allclose(clip.qpos, poses)
    np.testing.assert_allclose(
        clip.qvel, pose_velocities(model.mj_model, poses, 50), atol=1e-6
    )
    assert report["floor_ik_extra_passes"] == 0
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


@pytest.mark.parametrize("dt", [0.002, 0.02])
def test_floor_ik_prevents_penetration_without_grounding_airborne_targets(model, dt):
    mj = model.mj_model
    configuration = mink.Configuration(mj)
    pose = mj.qpos0.copy()
    pose[2] = 0.3
    configuration.update(pose)
    floor = retarget.FloorLimit(mj)
    task = mink.FrameTask("root", "body", position_cost=1.0, orientation_cost=1.0)
    for height, expected in ((-0.5, 0.1), (0.7, 0.7)):
        task.set_target(
            mink.SE3.from_rotation_and_translation(
                mink.SO3.identity(), np.array([0.0, 0.0, height])
            )
        )
        for _ in range(8):
            velocity = mink.solve_ik(configuration, [task], dt, "daqp", limits=[floor])
            configuration.integrate_inplace(velocity, dt)
        if height < 0:
            assert configuration.q[2] > height
        else:
            assert configuration.q[2] == pytest.approx(expected, abs=1e-6)
        assert floor.penetration(configuration.q) >= -1e-8


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


def test_offline_first_pose_converges_before_recording_without_freezing_motion(
    model, monkeypatch
):
    wrapper = object.__new__(retarget._OfflineGmrRetargeter)
    wrapper.model = model.mj_model
    wrapper._initialized = False
    wrapper.pose = model.mj_model.qpos0.copy()
    calls = []

    def solve(self, target):
        self._initialized = True
        self.pose[7] += 0.5 * (target - self.pose[7])
        calls.append(target)
        return self.pose.copy()

    monkeypatch.setattr(retarget._OfflineGmrRetargeter, "_solve_floor", solve)
    first = wrapper._solve(1.0)
    assert first[7] == pytest.approx(1.0, abs=0.01)
    assert len(calls) > 1
    before = len(calls)
    second = wrapper._solve(1.1)
    assert len(calls) == before + 1
    assert second[7] > first[7]
    assert pose_velocities(model.mj_model, np.stack((first, second)), 50)[0, 6] > 0


def test_nonconverging_initial_pose_is_rejected_instead_of_exported(model, monkeypatch):
    wrapper = object.__new__(retarget._OfflineGmrRetargeter)
    wrapper.model = model.mj_model
    wrapper._initialized = False
    wrapper.pose = model.mj_model.qpos0.copy()

    def solve(self, _):
        self._initialized = True
        self.pose[7] += 0.01
        return self.pose.copy()

    monkeypatch.setattr(retarget._OfflineGmrRetargeter, "_solve_floor", solve)
    with pytest.raises(
        retarget.RetargetValidationError, match="initial GMR pose did not converge"
    ):
        wrapper._solve(None)
