"""Runtime validation of the configured nominal robot pose."""

import mujoco
import numpy as np


def validate_robot_collision_geometry(model: mujoco.MjModel) -> None:
    mesh_collision = (
        (model.geom_bodyid != 0)
        & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
        & (model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
    )
    if np.any(mesh_collision):
        names = [
            model.geom(int(index)).name for index in np.flatnonzero(mesh_collision)
        ]
        raise ValueError(
            f"robot collision meshes must be packaged as canonical primitives: {names}"
        )


def validate_home_pose(model: mujoco.MjModel) -> None:
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home < 0:
        raise ValueError("compiled robot is missing its configured home pose")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, home)
    for joint in range(model.njnt):
        if model.jnt_limited[joint] and mujoco.mjtJoint(int(model.jnt_type[joint])) in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            position = data.qpos[model.jnt_qposadr[joint]]
            low, high = model.jnt_range[joint]
            if not np.isfinite(position) or not low <= position <= high:
                raise ValueError(
                    f"home pose: joint {model.joint(joint).name!r} is outside its URDF limits"
                )
    mujoco.mj_forward(model, data)
    collisions = {}
    for contact in data.contact:
        # Floor contact is a spawn/terrain concern, not robot self-collision.
        if (
            contact.dist >= -1e-6
            or 0 in model.geom_bodyid[[contact.geom1, contact.geom2]]
        ):
            continue
        pair = tuple(
            sorted((model.geom(contact.geom1).name, model.geom(contact.geom2).name))
        )
        collisions[pair] = max(collisions.get(pair, 0.0), -float(contact.dist))
    if collisions:
        detail = "; ".join(
            f"{a} / {b}: {depth:.6g} m penetration"
            for (a, b), depth in sorted(collisions.items())
        )
        raise ValueError(
            f"configured home pose self-collides; adjust nominal actuator positions or canonical collision geometry: {detail}"
        )
