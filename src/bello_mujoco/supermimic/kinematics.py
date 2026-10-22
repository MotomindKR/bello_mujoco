"""Analytic articulated poses and link-origin twists from compiled MuJoCo arrays."""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from bello_mujoco.supermimic import rotations as rot
from bello_mujoco.supermimic.state import BodyState, PhysicalState


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Kinematics:
    joint_names: tuple[str, ...] = field(metadata={"static": True})
    body_names: tuple[str, ...] = field(metadata={"static": True})
    root_qpos_address: int = field(metadata={"static": True})
    root_qvel_address: int = field(metadata={"static": True})
    root_local_position: jax.Array
    root_local_quaternion: jax.Array
    parent: jax.Array
    body_position: jax.Array
    body_quaternion: jax.Array
    joint_index: jax.Array
    joint_type: jax.Array
    joint_axis: jax.Array
    joint_anchor: jax.Array
    joint_reference: jax.Array
    joint_qpos_address: jax.Array
    joint_qvel_address: jax.Array
    tracked_indices: jax.Array

    @classmethod
    def from_model(cls, model, joint_names, body_names, *, root_body_name):
        joint_names, body_names = tuple(joint_names), tuple(body_names)
        if not body_names or len(set(body_names)) != len(body_names):
            raise ValueError("tracked body names must be nonempty and unique")
        if len(set(joint_names)) != len(joint_names):
            raise ValueError("joint names must be unique")
        anchor_body = model.body(root_body_name).id
        root = anchor_body
        fixed_path = []
        while root != 0 and model.body_jntnum[root] == 0:
            fixed_path.append(root)
            root = int(model.body_parentid[root])
        if root == 0 or model.body_parentid[root] != 0:
            raise ValueError("root must be a free body attached to world")
        if model.body_jntnum[root] != 1:
            raise ValueError("root must have exactly one free joint")
        free = model.body_jntadr[root]
        if model.jnt_type[free] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("root must have a free joint")
        root_position = np.zeros(3)
        root_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        for body in reversed(fixed_path):
            offset = np.empty(3)
            mujoco.mju_rotVecQuat(offset, model.body_pos[body], root_quaternion)
            root_position += offset
            orientation = np.empty(4)
            mujoco.mju_mulQuat(orientation, root_quaternion, model.body_quat[body])
            root_quaternion = orientation
        bodies = [root]
        for body in range(root + 1, model.nbody):
            if model.body_parentid[body] in bodies:
                bodies.append(body)
        indices = {body: i for i, body in enumerate(bodies)}
        tracked = [model.body(name).id for name in body_names]
        if any(body not in indices for body in tracked):
            raise ValueError("tracked body is outside the selected root tree")
        joints = [model.joint(name).id for name in joint_names]
        selected = {joint: i for i, joint in enumerate(joints)}
        actual = {
            joint
            for body in bodies[1:]
            for joint in range(
                model.body_jntadr[body],
                model.body_jntadr[body] + model.body_jntnum[body],
            )
        }
        if actual != set(joints):
            raise ValueError("joint layout must cover every joint in the root tree")
        parent, joint_index, joint_type, axis, anchor, reference = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for body in bodies:
            count = int(model.body_jntnum[body]) if body != root else 0
            if count > 1:
                raise ValueError("kinematics supports at most one joint per body")
            joint = int(model.body_jntadr[body]) if count else None
            kind = int(model.jnt_type[joint]) if count else -1
            if count and kind not in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                raise ValueError("kinematics supports hinge and slide joints")
            parent.append(
                indices[int(model.body_parentid[body])] if body != root else 0
            )
            joint_index.append(selected[joint] if count else -1)
            joint_type.append(kind)
            axis.append(model.jnt_axis[joint] if count else np.zeros(3))
            anchor.append(model.jnt_pos[joint] if count else np.zeros(3))
            reference.append(model.qpos0[model.jnt_qposadr[joint]] if count else 0.0)
        return cls(
            joint_names,
            body_names,
            int(model.jnt_qposadr[free]),
            int(model.jnt_dofadr[free]),
            jnp.asarray(root_position, dtype=jnp.float32),
            jnp.asarray(root_quaternion, dtype=jnp.float32),
            jnp.asarray(parent, dtype=jnp.int32),
            jnp.asarray(model.body_pos[bodies], dtype=jnp.float32),
            jnp.asarray(model.body_quat[bodies], dtype=jnp.float32),
            jnp.asarray(joint_index, dtype=jnp.int32),
            jnp.asarray(joint_type, dtype=jnp.int32),
            jnp.asarray(np.asarray(axis), dtype=jnp.float32),
            jnp.asarray(np.asarray(anchor), dtype=jnp.float32),
            jnp.asarray(reference, dtype=jnp.float32),
            jnp.asarray(model.jnt_qposadr[joints], dtype=jnp.int32),
            jnp.asarray(model.jnt_dofadr[joints], dtype=jnp.int32),
            jnp.asarray([indices[body] for body in tracked], dtype=jnp.int32),
        )

    def from_qpos(self, qpos, qvel):
        p, v = self.root_qpos_address, self.root_qvel_address
        free_quaternion = rot.normalize(qpos[p + 3 : p + 7])
        quaternion = rot.multiply(free_quaternion, self.root_local_quaternion)
        offset = rot.rotate(free_quaternion, self.root_local_position)
        angular = rot.rotate(free_quaternion, qvel[v + 3 : v + 6])
        return PhysicalState(
            qpos[p : p + 3] + offset,
            quaternion,
            qvel[v : v + 3] + jnp.cross(angular, offset),
            angular,
            qpos[self.joint_qpos_address],
            qvel[self.joint_qvel_address],
        )

    def to_qpos(self, state, qpos, qvel):
        p, v = self.root_qpos_address, self.root_qvel_address
        quaternion = rot.multiply(
            state.quaternion, rot.conjugate(self.root_local_quaternion)
        )
        offset = rot.rotate(quaternion, self.root_local_position)
        qpos = qpos.at[p : p + 3].set(state.position - offset)
        qpos = qpos.at[p + 3 : p + 7].set(rot.normalize(quaternion))
        qpos = qpos.at[self.joint_qpos_address].set(state.joint_position)
        qvel = qvel.at[v : v + 3].set(
            state.linear_velocity - jnp.cross(state.angular_velocity, offset)
        )
        qvel = qvel.at[v + 3 : v + 6].set(
            rot.rotate(rot.conjugate(quaternion), state.angular_velocity)
        )
        return qpos, qvel.at[self.joint_qvel_address].set(state.joint_velocity)

    def forward(self, state):
        count = self.parent.shape[0]
        free_quaternion = rot.multiply(
            state.quaternion, rot.conjugate(self.root_local_quaternion)
        )
        offset = rot.rotate(free_quaternion, self.root_local_position)
        position = jnp.zeros((count, 3)).at[0].set(state.position - offset)
        quaternion = jnp.zeros((count, 4)).at[0].set(rot.normalize(free_quaternion))
        linear = (
            jnp.zeros((count, 3))
            .at[0]
            .set(state.linear_velocity - jnp.cross(state.angular_velocity, offset))
        )
        angular = jnp.zeros((count, 3)).at[0].set(state.angular_velocity)
        q = jnp.concatenate((state.joint_position, jnp.zeros(1)))
        qd = jnp.concatenate((state.joint_velocity, jnp.zeros(1)))

        def body_step(i, carry):
            positions, quaternions, velocities, angular_velocities = carry
            parent = self.parent[i]
            p0 = positions[parent] + rot.rotate(
                quaternions[parent], self.body_position[i]
            )
            q0 = rot.multiply(quaternions[parent], self.body_quaternion[i])
            anchor = p0 + rot.rotate(q0, self.joint_anchor[i])
            axis = rot.rotate(q0, self.joint_axis[i])
            delta = q[self.joint_index[i]] - self.joint_reference[i]
            speed = qd[self.joint_index[i]]
            hinge = self.joint_type[i] == int(mujoco.mjtJoint.mjJNT_HINGE)
            slide = self.joint_type[i] == int(mujoco.mjtJoint.mjJNT_SLIDE)
            orientation = rot.multiply(
                q0, rot.exp(self.joint_axis[i] * jnp.where(hinge, delta, 0.0))
            )
            point = jnp.where(
                hinge,
                anchor - rot.rotate(orientation, self.joint_anchor[i]),
                p0 + axis * jnp.where(slide, delta, 0.0),
            )
            omega = angular_velocities[parent] + axis * jnp.where(hinge, speed, 0.0)
            velocity = (
                velocities[parent]
                + jnp.cross(angular_velocities[parent], point - positions[parent])
                + jnp.cross(axis, point - anchor) * jnp.where(hinge, speed, 0.0)
                + axis * jnp.where(slide, speed, 0.0)
            )
            return (
                positions.at[i].set(point),
                quaternions.at[i].set(orientation),
                velocities.at[i].set(velocity),
                angular_velocities.at[i].set(omega),
            )

        result = jax.lax.fori_loop(
            1, count, body_step, (position, quaternion, linear, angular)
        )
        return BodyState(*(value[self.tracked_indices] for value in result))
