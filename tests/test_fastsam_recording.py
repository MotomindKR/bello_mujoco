from __future__ import annotations

import numpy as np

from bello_mujoco.fastsam_conditioning import CausalFootConditionedRoot
from bello_mujoco.hardware.smplx_retarget import SMPLX_BODY_JOINT_NAMES


def _standing_positions() -> np.ndarray:
    positions = np.zeros((len(SMPLX_BODY_JOINT_NAMES), 3), dtype=np.float64)
    pelvis = SMPLX_BODY_JOINT_NAMES.index("pelvis")
    left = SMPLX_BODY_JOINT_NAMES.index("left_foot")
    right = SMPLX_BODY_JOINT_NAMES.index("right_foot")
    positions[pelvis, 2] = 0.9
    positions[left] = (-0.05, 0.1, 0.0)
    positions[right] = (0.05, -0.1, 0.0)
    return positions


def test_causal_support_foot_conditioning_removes_shared_camera_drift() -> None:
    conditioner = CausalFootConditionedRoot(SMPLX_BODY_JOINT_NAMES)
    confidence = np.ones(len(SMPLX_BODY_JOINT_NAMES))
    pelvis = SMPLX_BODY_JOINT_NAMES.index("pelvis")
    left = SMPLX_BODY_JOINT_NAMES.index("left_foot")
    outputs = []
    for frame, drift in enumerate(np.linspace(0.0, 0.2, 501)):
        positions = _standing_positions()
        positions[:, 0] += drift
        conditioned, contacts = conditioner.condition(
            positions,
            confidence,
            timestamp_ns=1_000_000_000 + frame * 20_000_000,
        )
        outputs.append(conditioned)
        assert np.all(contacts)
    outputs = np.stack(outputs)

    assert abs(outputs[-1, pelvis, 0] - outputs[0, pelvis, 0]) < 0.005
    assert abs(outputs[-1, left, 0] - outputs[0, left, 0]) < 0.005


def test_causal_support_foot_conditioning_preserves_planted_leg_motion() -> None:
    conditioner = CausalFootConditionedRoot(SMPLX_BODY_JOINT_NAMES)
    confidence = np.ones(len(SMPLX_BODY_JOINT_NAMES))
    pelvis = SMPLX_BODY_JOINT_NAMES.index("pelvis")
    left = SMPLX_BODY_JOINT_NAMES.index("left_foot")
    right = SMPLX_BODY_JOINT_NAMES.index("right_foot")
    outputs = []
    contact_history = []
    for frame, root_x in enumerate(np.linspace(0.0, 0.2, 501)):
        positions = _standing_positions()
        positions[pelvis, 0] = root_x
        positions[left, 0] = 0.0
        positions[right, 0] = root_x + 0.05
        positions[right, 2] = 0.15
        conditioned, contacts = conditioner.condition(
            positions,
            confidence,
            timestamp_ns=1_000_000_000 + frame * 20_000_000,
        )
        outputs.append(conditioned)
        contact_history.append(contacts)
    outputs = np.stack(outputs)
    contact_history = np.stack(contact_history)

    assert outputs[-1, pelvis, 0] > 0.19
    assert abs(outputs[-1, left, 0] - outputs[0, left, 0]) < 0.005
    assert np.all(contact_history[:, 0])
    assert not np.any(contact_history[:, 1])


def test_causal_support_foot_contact_hysteresis_rejects_threshold_chatter() -> None:
    conditioner = CausalFootConditionedRoot(SMPLX_BODY_JOINT_NAMES)
    confidence = np.ones(len(SMPLX_BODY_JOINT_NAMES))
    right = SMPLX_BODY_JOINT_NAMES.index("right_foot")
    history = []
    for frame in range(100):
        positions = _standing_positions()
        positions[right, 2] += 0.055 if frame % 2 else 0.065
        _, contacts = conditioner.condition(
            positions,
            confidence,
            timestamp_ns=1_000_000_000 + frame * 20_000_000,
        )
        history.append(contacts)
    history = np.stack(history)

    switches = np.count_nonzero(history[1:] != history[:-1], axis=0)
    assert np.max(switches) <= 1
