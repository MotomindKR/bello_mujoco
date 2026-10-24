from types import SimpleNamespace

import mink
import mujoco
import numpy as np
import pytest

from bello_mujoco.models.target_limits import TargetLimits
from bello_mujoco.motion.lafan import clip_from_poses, project_contacts
from bello_mujoco.motion.retarget_limits import RetargetContactLimit, RetargetJointLimit


def test_cached_body_twists_are_at_link_origins_with_offset_centres_of_mass():
    mj_model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="root"><freejoint/>'
        '<geom type="sphere" size=".1" pos=".2 0 0"/>'
        '<body name="arm" pos=".3 0 0"><joint name="joint" axis="0 0 1"/>'
        '<geom type="sphere" size=".1" pos=".15 0 0"/>'
        "</body></body></worldbody></mujoco>"
    )
    model = SimpleNamespace(
        mj_model=mj_model,
        actuator_qpos_indices=np.array([7]),
        actuator_qvel_indices=np.array([6]),
        actuator_coordinate_transform=np.eye(1),
        actuator_joint_names=("joint",),
        target_limits=TargetLimits(
            np.array([[-2.0, 2.0]]), np.array([[-2.0, 2.0]]), np.eye(1), np.ones(1)
        ),
        robot=SimpleNamespace(foot_body_names=("arm",)),
        foot_geom_ids=np.array([1]),
    )
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


def test_contact_projection_repairs_penetration_without_target_tasks():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="plane" size="1 1 .01"/>'
        '<body pos="0 0 .05"><freejoint/><geom type="sphere" size=".1"/>'
        "</body></worldbody></mujoco>"
    )
    limit = RetargetContactLimit(model)
    backend = SimpleNamespace(
        model=model,
        configuration=mink.Configuration(model),
        solver="daqp",
        ik_limits=[limit],
    )
    pose = project_contacts(backend, limit, iterations=5, tolerance=1e-6)
    assert pose[2] == pytest.approx(0.101)


def test_rate_limit_bounds_total_frame_motion_across_multiple_ik_iterations(tmp_path):
    source = tmp_path / "source.urdf"
    source.write_text(
        '<robot><joint name="joint"><limit velocity="2"/></joint></robot>'
    )
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint name="joint"/><geom size=".1"/></body></worldbody></mujoco>'
    )
    description = SimpleNamespace(
        actuator_qpos_indices=np.array([0]),
        actuator_qvel_indices=np.array([0]),
        actuator_joint_names=("joint",),
        actuator_coordinate_transform=np.eye(1),
        target_limits=TargetLimits(
            np.array([[-1.0, 1.0]]), np.array([[-1.0, 1.0]]), np.eye(1), np.ones(1)
        ),
        robot=SimpleNamespace(source_urdf_path=lambda root: source),
    )
    limit = RetargetJointLimit(description, 0.02)
    limit.previous = np.array([0.0])
    configuration = mink.Configuration(model)
    configuration.update(np.array([0.03]))
    inequality = limit.compute_qp_inequalities(configuration, 0.005)
    assert (inequality.G @ np.array([0.01]) <= inequality.h + 1e-12).all()
    assert not (inequality.G @ np.array([0.02]) <= inequality.h + 1e-12).all()
