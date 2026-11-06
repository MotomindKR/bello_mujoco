import numpy as np
from scipy.spatial.transform import Rotation

from bello_mujoco.hardware.smplx_contract import SMPLX_BODY_JOINT_NAMES
from bello_mujoco.motion.soma_lafan import _BODY_JOINTS, soma_local_rotations


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
