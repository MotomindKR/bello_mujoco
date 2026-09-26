"""Torso targets independent of the velocity command and gait clock."""

import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation

from bello_mujoco.envs.geometry import yaw_quat


def neutral_torso_quaternion(model, qpos, torso_id, pelvis_id):
    """Nominal CAD torso orientation in the pelvis's yaw-only frame (WXYZ)."""
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation as CpuRotation

    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    pelvis = CpuRotation.from_quat(np.roll(data.xquat[pelvis_id], -1))
    heading = CpuRotation.from_euler("z", pelvis.as_euler("xyz")[2])
    torso = CpuRotation.from_quat(np.roll(data.xquat[torso_id], -1))
    return np.roll((heading.inv() * torso).as_quat(), 1)


def torso_command_from_quaternions(actual, pelvis, neutral, forward_axis):
    """Invert the training target convention for a CPU-retargeted robot pose."""
    import numpy as np
    from scipy.spatial.transform import Rotation as CpuRotation

    actual, pelvis, neutral = (
        CpuRotation.from_quat(np.roll(q, -1)) for q in (actual, pelvis, neutral)
    )
    heading = CpuRotation.from_euler("z", pelvis.as_euler("xyz")[2])
    basis = CpuRotation.from_euler("z", np.arctan2(forward_axis[1], forward_axis[0]))
    return (basis.inv() * heading.inv() * actual * neutral.inv() * basis).as_euler(
        "xyz"
    )


def slew_torso_command(current, target, rate, dt):
    step = jnp.asarray(rate) * dt
    return current + jnp.clip(target - current, -step, step)


def torso_orientation_error(actual, pelvis, command, neutral, forward_axis):
    """SO(3) distance; angles use anatomical forward/left/up, not CAD axes.

    Quaternions are MuJoCo wxyz. `neutral` is the nominal torso orientation
    expressed in the pelvis's yaw-only frame, preserving the CAD frame offset.
    """

    def rotation(q):
        return Rotation.from_quat(jnp.roll(q, -1))

    forward = jnp.asarray(forward_axis)
    basis = Rotation.from_euler("z", jnp.arctan2(forward[1], forward[0]))
    target = (
        rotation(yaw_quat(pelvis))
        * basis
        * Rotation.from_euler("xyz", command)
        * basis.inv()
        * rotation(neutral)
    )
    return (target.inv() * rotation(actual)).magnitude()
