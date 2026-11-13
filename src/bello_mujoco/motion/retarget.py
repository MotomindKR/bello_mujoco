"""Stock Mini GMR retargeting and shared motion-clip reconstruction."""

from collections.abc import Mapping
from types import SimpleNamespace

import mujoco
import mink
import numpy as np

from bello_mujoco.envs.rewards import CommandConfig
from bello_mujoco.hardware.gmr_retarget import (
    GmrLiveRetargetConfig,
    GmrRetargeter,
    HumanTargetFrame,
)
from bello_mujoco.motion.clip import MotionClip
from bello_mujoco.robot_tasks import motion_task_definition
from bello_mujoco.sim import geom_lower_z


_INITIAL_POSE_MAX_PASSES = 1000


class RetargetValidationError(ValueError):
    """A retargeted pose violates the robot's physical position envelope."""


def pose_velocities(model, qpos, fps):
    """Central pose differences, with free-joint angular velocities at the current pose."""
    qvel = np.zeros((len(qpos), model.nv))
    free_joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    for frame in range(len(qpos)):
        left, right = max(0, frame - 1), min(len(qpos) - 1, frame + 1)
        mujoco.mj_differentiatePos(
            model, qvel[frame], (right - left) / fps, qpos[left], qpos[right]
        )
        for joint in free_joints:
            p, v = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
            world, inverse, local = np.empty(3), np.empty(4), np.empty(3)
            mujoco.mju_rotVecQuat(
                world, qvel[frame, v + 3 : v + 6], qpos[left, p + 3 : p + 7]
            )
            mujoco.mju_negQuat(inverse, qpos[frame, p + 3 : p + 7])
            mujoco.mju_rotVecQuat(local, world, inverse)
            qvel[frame, v + 3 : v + 6] = local
    return qvel


def clip_from_poses(model, body_names, qpos, fps, contact_height_m):
    """Differentiate poses and cache link-origin FK/twists, never COM twists."""
    mj_model = model.mj_model
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape != (len(qpos), mj_model.nq) or len(qpos) < 2:
        raise ValueError("offline references require at least two complete poses")
    if not np.isfinite(qpos).all():
        raise ValueError("offline references contain nonfinite poses")
    logical = qpos[:, model.actuator_qpos_indices]
    constrained = (
        logical @ model.actuator_coordinate_transform.T @ model.target_limits.matrix.T
    )
    bounds = model.target_limits.bounds
    if np.any(constrained < bounds[:, 0] - 1e-4) or np.any(
        constrained > bounds[:, 1] + 1e-4
    ):
        raise RetargetValidationError("retarget violates canonical motor/joint limits")
    qvel = pose_velocities(mj_model, qpos, fps)
    data = mujoco.MjData(mj_model)
    body_ids = [mj_model.body(name).id for name in body_names]
    foot_ids = [mj_model.body(name).id for name in model.robot.foot_body_names]
    foot_geoms = [
        [int(g) for g in model.foot_geom_ids if mj_model.geom_bodyid[g] == body]
        for body in foot_ids
    ]
    positions, quaternions, linear, angular, contacts = [], [], [], [], []
    worst_floor, worst_self = 0.0, 0.0
    plane = mj_model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE
    for pose, velocity in zip(qpos, qvel, strict=True):
        data.qpos[:], data.qvel[:] = pose, velocity
        mujoco.mj_forward(mj_model, data)
        positions.append(data.xpos[body_ids].copy())
        quaternions.append(data.xquat[body_ids].copy())
        twists = np.empty((len(body_ids), 6))
        for index, body in enumerate(body_ids):
            mujoco.mj_objectVelocity(
                mj_model, data, mujoco.mjtObj.mjOBJ_XBODY, body, twists[index], 0
            )
        angular.append(twists[:, :3].copy())
        linear.append(twists[:, 3:].copy())
        contacts.append(
            [
                min(geom_lower_z(mj_model, data, geom) for geom in geoms)
                <= contact_height_m
                for geoms in foot_geoms
            ]
        )
        for contact in data.contact:
            if any(plane[g] for g in contact.geom):
                worst_floor = min(worst_floor, float(contact.dist))
            else:
                worst_self = min(worst_self, float(contact.dist))
    clip = MotionClip(
        fps=float(fps),
        joint_names=model.actuator_joint_names,
        body_names=tuple(body_names),
        qpos=qpos.astype(np.float32),
        qvel=qvel.astype(np.float32),
        body_pos=np.asarray(positions, dtype=np.float32),
        body_quat=np.asarray(quaternions, dtype=np.float32),
        body_lin_vel=np.asarray(linear, dtype=np.float32),
        body_ang_vel=np.asarray(angular, dtype=np.float32),
        foot_contacts=np.asarray(contacts, dtype=np.float32),
    )
    return clip, {
        "frames": len(qpos),
        "seconds": clip.duration,
        "worst_floor_penetration_m": worst_floor,
        "worst_self_penetration_m": worst_self,
        "peak_joint_speed_rad_s": float(
            np.abs(qvel[:, model.actuator_qvel_indices]).max()
        ),
        "root_height_range_m": [float(qpos[:, 2].min()), float(qpos[:, 2].max())],
        "contact_fraction": np.mean(contacts, axis=0).tolist(),
    }


class FloorLimit(mink.Limit):
    """Linearized nonpenetration in Mink's displacement coordinates, G dq <= h."""

    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        planes = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
        if len(free) != 1 or len(planes) != 1:
            raise RetargetValidationError(
                "offline IK requires one free root and one floor"
            )
        self.plane = int(planes[0])
        self.root_z = int(model.jnt_qposadr[free[0]]) + 2
        root = model.jnt_bodyid[free[0]]
        collides = (
            (model.geom_contype & model.geom_conaffinity[self.plane])
            | (model.geom_conaffinity & model.geom_contype[self.plane])
        ) != 0
        self.geoms = np.flatnonzero(
            collides & (model.body_rootid[model.geom_bodyid] == root)
        )
        self._update(model.qpos0)
        if (
            not len(self.geoms)
            or not np.allclose(
                self.data.geom_xmat[self.plane].reshape(3, 3)[:, 2], [0, 0, 1]
            )
            or model.geom_bodyid[self.plane] != 0
        ):
            raise RetargetValidationError(
                "offline IK requires a fixed horizontal colliding floor"
            )

    def _update(self, qpos):
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def compute_qp_inequalities(self, configuration, dt):
        del dt
        self._update(configuration.q)
        rows, bounds = [], []
        points, jacobian = np.empty(6), np.empty((3, self.model.nv))
        for geom in self.geoms:
            distance = mujoco.mj_geomDistance(
                self.model,
                self.data,
                int(geom),
                self.plane,
                float(self.model.stat.extent),
                points,
            )
            if distance >= self.model.stat.extent:
                continue
            mujoco.mj_jac(
                self.model,
                self.data,
                jacobian,
                None,
                points[:3],
                int(self.model.geom_bodyid[geom]),
            )
            rows.append(-jacobian[2].copy())
            bounds.append(distance)
        return mink.Constraint(
            G=np.asarray(rows).reshape(-1, self.model.nv), h=np.asarray(bounds)
        )

    def penetration(self, qpos):
        self._update(qpos)
        mujoco.mj_collision(self.model, self.data)
        return min(
            (float(c.dist) for c in self.data.contact if self.plane in c.geom),
            default=0.0,
        )


class _OfflineGmrRetargeter(GmrRetargeter):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.floor = FloorLimit(env.model.mj_model)
        self.retargeter.ik_limits.append(self.floor)
        self.extra_floor_passes = 0
        self.max_floor_projection_m = 0.0
        self.initial_pose_passes = 0
        self.initial_pose_residual = 0.0

    def _solve(self, targets):
        initialize = not self._initialized
        pose = self._solve_floor(targets)
        if not initialize:
            return pose
        delta = np.empty(self.model.nv)
        tolerance = np.full(self.model.nv, 0.01)
        tolerance[:3] = 0.001
        tolerance[3:6] = 0.005
        for attempt in range(_INITIAL_POSE_MAX_PASSES):
            previous = pose
            pose = self._solve_floor(targets)
            mujoco.mj_differentiatePos(self.model, delta, 1.0, previous, pose)
            self.initial_pose_passes = attempt + 1
            self.initial_pose_residual = float(np.max(np.abs(delta) / tolerance))
            if self.initial_pose_residual <= 1.0:
                return pose
        raise RetargetValidationError(
            f"initial GMR pose did not converge after {self.initial_pose_passes} passes "
            f"(normalized pose change {self.initial_pose_residual:.6g})"
        )

    def _solve_floor(self, targets):
        pose = super()._solve(targets)
        for attempt in range(self.retargeter.max_iter + 1):
            depth = self.floor.penetration(pose)
            if depth >= -self.floor.model.opt.tolerance:
                break
            if attempt < self.retargeter.max_iter:
                pose = super()._solve(targets)
                self.extra_floor_passes += 1
        lift = max(0.0, -depth)
        self.max_floor_projection_m = max(self.max_floor_projection_m, lift)
        pose[self.floor.root_z] += lift
        self.retargeter.configuration.update(pose)
        return pose


def retarget_gmr(model, targets: Mapping, *, fps, human_height_m, contact_height_m):
    """Run the stock profile with native collision-surface floor constraints."""
    wrapper = _OfflineGmrRetargeter(
        SimpleNamespace(model=model, command_config=CommandConfig()),
        GmrLiveRetargetConfig(human_height_m=human_height_m),
    )
    names = wrapper.profile.source_body_names
    if missing := set(names) - set(targets):
        raise ValueError(f"human source is missing GMR targets: {sorted(missing)}")
    count = len(targets[names[0]][0])
    for name in names:
        position, quaternion = map(np.asarray, targets[name])
        if count < 2 or position.shape != (count, 3) or quaternion.shape != (count, 4):
            raise ValueError(f"invalid clocked human transforms for {name}")
    poses = []
    for frame in range(count):
        human = HumanTargetFrame(
            transforms={
                name: (targets[name][0][frame], targets[name][1][frame])
                for name in names
            },
            sequence=frame,
            source_timestamp_ns=1 + round(frame * 1e9 / fps),
            source_age_ms=0.0,
            received_at=0.0,
        )
        poses.append(wrapper.retarget(human).qpos)
    clip, report = clip_from_poses(
        model,
        motion_task_definition(model.robot.asset).beyondmimic_tracked_body_names,
        poses,
        fps,
        contact_height_m,
    )
    return clip, {
        **report,
        "floor_ik_extra_passes": wrapper.extra_floor_passes,
        "max_floor_projection_m": wrapper.max_floor_projection_m,
        "initial_pose_passes": wrapper.initial_pose_passes,
        "initial_pose_residual": wrapper.initial_pose_residual,
    }
