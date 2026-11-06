import numpy as np
from scipy.spatial.transform import Rotation

from bello_mujoco.hardware.smplx_contract import SMPLX_BODY_JOINT_NAMES
from bello_mujoco.motion.soma import _BODY_JOINTS, soma_local_rotations


def test_soma_body_frames_preserve_global_motion_and_unobserved_fingers():
    names = ("Root", *_BODY_JOINTS, "LeftHandIndex1")
    parents = np.zeros(len(names), dtype=int)
    parents[-1] = names.index("LeftHand")
    reference = np.broadcast_to(np.eye(4), (len(names), 4, 4)).copy()
    reference[1:, :3, :3] = Rotation.random(len(names) - 1, random_state=2).as_matrix()
    motion = Rotation.from_euler("zyx", [[0.3, 0.5, -0.7], [0.6, 0.2, -1.1]])
    targets = {
        name: (np.zeros((2, 3)), motion.as_quat(scalar_first=True))
        for name in SMPLX_BODY_JOINT_NAMES
    }
    local = soma_local_rotations(names, parents, reference, targets)
    world = [np.broadcast_to(np.eye(3), (2, 3, 3))]
    for joint in range(1, len(names)):
        world.append(world[parents[joint]] @ local[:, joint - 1])
    for joint in range(1, len(names)):
        np.testing.assert_allclose(
            world[joint], motion.as_matrix() @ reference[joint, :3, :3], atol=1e-12
        )
    np.testing.assert_allclose(
        local[:, -1],
        np.broadcast_to(
            reference[parents[-1], :3, :3].T @ reference[-1, :3, :3], (2, 3, 3)
        ),
        atol=1e-12,
    )


def test_bones_bind_conversion_preserves_global_motion_and_finger_frames():
    from dataclasses import replace
    from bello_mujoco.motion.bvh import BvhMotion
    from bello_mujoco.motion.soma import soma_bvh_rotations

    names = ("Root", *_BODY_JOINTS, "LeftHandIndex1")
    parents = np.zeros(len(names), dtype=int)
    parents[-1] = names.index("LeftHand")
    rng = np.random.default_rng(4)
    positions = np.repeat(rng.normal(size=(1, len(names), 3)), 2, axis=0)
    quaternions = np.repeat(
        Rotation.random(len(names), random_state=3).as_quat(scalar_first=True)[None],
        2,
        axis=0,
    )
    bind = BvhMotion(names, (-1, *parents[1:]), positions, quaternions, 0.02)
    reference = np.broadcast_to(np.eye(4), (len(names), 4, 4)).copy()
    reference[1:, :3, :3] = Rotation.random(len(names) - 1, random_state=6).as_matrix()
    turn = Rotation.from_euler("y", 35, degrees=True)
    moved = quaternions.copy()
    moved[:, 0] = (turn * Rotation.from_quat(moved[:, 0], scalar_first=True)).as_quat(
        scalar_first=True
    )
    motion = replace(bind, quaternion=moved)
    local, translation, _ = soma_bvh_rotations(motion, bind, names, parents, reference)
    world = [np.broadcast_to(np.eye(3), (2, 3, 3))]
    expected = Rotation.from_euler("x", 90, degrees=True).as_matrix() @ turn.as_matrix()
    for joint in range(1, len(names)):
        world.append(world[parents[joint]] @ local[:, joint - 1])
        np.testing.assert_allclose(
            world[joint],
            np.broadcast_to(expected @ reference[joint, :3, :3], (2, 3, 3)),
            atol=1e-12,
        )
    np.testing.assert_allclose(
        translation, motion.world_transforms()[0][:, names.index("Hips")]
    )
