from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest

from bello_mujoco.hardware.reference import ReferenceUnavailable
from bello_mujoco.hardware.smplx_contract import SMPLX_JOINT_PARENTS
from bello_mujoco.hardware.smplx_grpc import SmplxGrpcConfig
from bello_mujoco.hardware.smplx_kinematics import SmplxKinematics
from bello_mujoco.hardware.smplx_stream import SmplxTargetGrpcStream


@pytest.fixture
def model_path(tmp_path: Path):
    template = np.zeros((8100, 3))
    for i, parent in enumerate(SMPLX_JOINT_PARENTS):
        template[i] = [0, 1, 0] if parent < 0 else template[parent] + [0, 0.05, 0]
    template[18] = [0.5, 1.5, 0]
    template[20] = [0.75, 1.5, 0]
    template[4] = [0.1, 0.5, 0]
    template[7] = [0.1, 0.1, 0]
    template[25:28] = [[0.8, 1.5, 0], [0.83, 1.5, 0], [0.86, 1.5, 0]]
    for vertex, parent in zip(
        (4933, 5058, 5286, 5169, 5361, 7669, 7794, 8022, 7905, 8079),
        (27, 30, 33, 36, 39, 42, 45, 48, 51, 54),
        strict=True,
    ):
        template[vertex] = template[parent] + [0.02, 0, 0]
    path = tmp_path / "neutral.npz"
    np.savez(
        path,
        v_template=template,
        J_regressor=np.eye(55, len(template)),
        kintree_table=np.array([SMPLX_JOINT_PARENTS, range(55)]),
    )
    return path


def test_elbow_and_knee_rotation_direction(model_path):
    fk = SmplxKinematics(model_path)
    rotations = np.zeros((55, 3))
    rotations[18, 2] = np.pi / 2
    rotations[4, 0] = np.pi / 2
    pose = fk.forward(rotations, np.zeros(3))
    np.testing.assert_allclose(pose.positions[20], [0.5, 1.75, 0], atol=1e-12)
    np.testing.assert_allclose(pose.positions[7], [0.1, 0.5, -0.4], atol=1e-12)
    world = fk.forward(rotations, np.zeros(3), world=True)
    np.testing.assert_allclose(world.positions[20], [0, -0.5, 1.75], atol=1e-12)
    np.testing.assert_allclose(world.positions[7], [0.4, -0.1, 0.5], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(world.quaternions, axis=1), 1)


def test_global_rotation_pivots_at_pelvis_and_translation_is_displacement(model_path):
    fk = SmplxKinematics(model_path)
    rotations = np.zeros((55, 3))
    rotations[0, 1] = np.pi / 2
    pose = fk.forward(rotations, np.array([2, 3, 4]))
    np.testing.assert_allclose(pose.positions[0], [2, 4, 4])
    np.testing.assert_allclose(pose.positions[20], [2, 4.5, 3.25])
    np.testing.assert_allclose(pose.quaternions[20], [2**-0.5, 0, 2**-0.5, 0])


def test_distal_finger_rotation_moves_tip_without_moving_other_hand(model_path):
    fk = SmplxKinematics(model_path)
    rotations = np.zeros((55, 3))
    neutral = fk.forward(rotations, np.zeros(3))
    rotations[27, 2] = np.pi / 2
    bent = fk.forward(rotations, np.zeros(3))
    np.testing.assert_allclose(bent.tips[0], [0.86, 1.52, 0])
    np.testing.assert_allclose(bent.tips[1:], neutral.tips[1:])
    np.testing.assert_allclose(bent.positions, neutral.positions)


def test_model_rejects_noncanonical_hierarchy(model_path):
    with np.load(model_path) as data:
        values = dict(data)
    values["kintree_table"][0, 20] = 17
    np.savez(model_path, **values)
    with pytest.raises(ValueError, match="hierarchy"):
        SmplxKinematics(model_path)


def test_gmr_adapter_filters_invalid_body_and_preserves_wire_timing(model_path):
    stream = SmplxTargetGrpcStream(
        SmplxGrpcConfig("localhost:1", model_path), ("pelvis", "left_wrist")
    )
    frame = SimpleNamespace(
        body_valid=False,
        rotations=np.zeros((55, 3)),
        root_translation=np.array([0, 0.1, 0]),
        sequence=7,
        source_timestamp_ns=234,
        source_age_ms=12,
        received_at=time.monotonic(),
    )
    stream.transport = SimpleNamespace(wait_next=lambda **kwargs: frame)
    with pytest.raises(ReferenceUnavailable, match="body tracking"):
        stream.wait_next(after_sequence=None, timeout_seconds=1)
    frame.body_valid = True
    result = stream.wait_next(after_sequence=None, timeout_seconds=1)
    assert result.sequence == 7 and result.source_timestamp_ns == 234
    assert result.source_age_ms == 12 and result.received_at == frame.received_at
    np.testing.assert_allclose(result.transforms["pelvis"][0], [0, 0, 1.1])
    np.testing.assert_allclose(result.transforms["left_wrist"][0], [0, -0.75, 1.6])
