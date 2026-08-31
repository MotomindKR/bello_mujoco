from __future__ import annotations

from types import SimpleNamespace
import time

import numpy as np
import pytest

from bello_mujoco.hardware.gemx import (
    GemxCoordinateAdapter,
    GemxSomaStream,
    _SOMA_NEUTRAL_GLOBAL_WXYZ,
    _SOMA_TO_SMPLX_GMR,
)
from bello_mujoco.hardware.smplx_retarget import SMPLX_GMR_BODY_NAMES
from bello_mujoco.hardware.twist2 import ReferenceUnavailable


SOMA_BODY_NAMES = (
    "Hips",
    "Chest",
    "LeftLeg",
    "LeftShin",
    "LeftToeBase",
    "RightLeg",
    "RightShin",
    "RightToeBase",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightArm",
    "RightForeArm",
    "RightHand",
)


def test_gemx_soma_adapter_covers_the_gmr_body_contract() -> None:
    assert tuple(_SOMA_TO_SMPLX_GMR) == SMPLX_GMR_BODY_NAMES
    assert tuple(_SOMA_TO_SMPLX_GMR.values()) == SOMA_BODY_NAMES


def test_gemx_frame_is_reduced_from_soma_without_claiming_smplx() -> None:
    names = SOMA_BODY_NAMES + tuple(f"unused_{index}" for index in range(63))
    config = SimpleNamespace(
        camera_to_world_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        align_initial_heading=False,
        align_initial_floor=False,
        min_tracking_confidence=0.5,
        min_joint_confidence=0.05,
        full_joint_confidence=0.25,
        max_pipeline_latency_ms=120.0,
        transport_latency_budget_ms=10.0,
    )
    stream = object.__new__(GemxSomaStream)
    stream.config = config
    stream._soma_joint_names = names
    stream._coordinates = GemxCoordinateAdapter(config, names)
    stream._last_frame_id = None
    stream._last_source_timestamp_ns = None
    stream._active_track_id = None
    stream._previous_mapped_positions = None
    stream._previous_mapped_quaternions = None

    class RootConditioner:
        def condition(self, positions, _confidence, *, timestamp_ns):
            assert timestamp_ns in (123_000_000, 223_000_000, 323_000_000)
            return positions, np.asarray((1.0, 0.0))

        def reset(self):
            return None

    stream._root_conditioner = RootConditioner()
    positions = np.zeros((77, 3), dtype=np.float32)
    positions[:, 0] = np.arange(77)
    positions[0] = (0.0, 0.0, 0.9)
    quaternions = np.zeros((77, 4), dtype=np.float32)
    quaternions[:, 3] = 1.0
    for name, rest_wxyz in _SOMA_NEUTRAL_GLOBAL_WXYZ.items():
        quaternions[names.index(name)] = np.asarray(rest_wxyz)[[1, 2, 3, 0]]
    message = SimpleNamespace(
        frame_id=7,
        source_timestamp_ns=123_000_000,
        tracking_state="TRACKING",
        track_id=2,
        joint_positions=positions.reshape(-1),
        root_translation=positions[0],
        global_joint_quaternions=quaternions.reshape(-1),
        joint_confidence=np.ones(77, dtype=np.float32),
        association_confidence=0.9,
        pipeline_latency_ms=20.0,
    )

    frame = stream._parse_frame(message)

    assert frame.sequence == 7
    assert frame.source_age_ms == 30
    assert frame.received_at <= time.monotonic()
    assert tuple(frame.transforms) == SMPLX_GMR_BODY_NAMES
    for body_name, soma_name in _SOMA_TO_SMPLX_GMR.items():
        source_index = names.index(soma_name)
        np.testing.assert_allclose(
            frame.transforms[body_name][0], positions[source_index]
        )
        np.testing.assert_allclose(
            frame.transforms[body_name][1],
            (1.0, 0.0, 0.0, 0.0),
            atol=5.0e-8,
        )

    second_positions = positions.copy()
    left_hand = names.index("LeftHand")
    second_positions[left_hand, 0] += 4.0
    second_confidence = np.ones(77, dtype=np.float32)
    second_confidence[left_hand] = 0.10
    second = SimpleNamespace(
        **{
            **message.__dict__,
            "frame_id": 8,
            "source_timestamp_ns": 223_000_000,
            "joint_positions": second_positions.reshape(-1),
            "root_translation": second_positions[0],
            "joint_confidence": second_confidence,
        }
    )
    interpolated = stream._parse_frame(second)
    alpha = (0.10 - 0.05) / (0.25 - 0.05)
    np.testing.assert_allclose(
        interpolated.transforms["left_wrist"][0],
        positions[left_hand] + alpha * np.asarray((4.0, 0.0, 0.0)),
    )

    held_positions = positions.copy()
    left_toe = names.index("LeftToeBase")
    held_positions[left_toe, 0] += 4.0
    held_confidence = np.ones(77, dtype=np.float32)
    held_confidence[left_toe] = 0.01
    held_message = SimpleNamespace(
        **{
            **message.__dict__,
            "frame_id": 9,
            "source_timestamp_ns": 323_000_000,
            "joint_positions": held_positions.reshape(-1),
            "root_translation": held_positions[0],
            "joint_confidence": held_confidence,
        }
    )
    held = stream._parse_frame(held_message)
    np.testing.assert_allclose(held.transforms["left_foot"][0], positions[left_toe])

    bootstrap = object.__new__(GemxSomaStream)
    bootstrap.config = config
    bootstrap._previous_mapped_positions = None
    bootstrap._previous_mapped_quaternions = None
    bootstrap_confidence = np.ones(len(SOMA_BODY_NAMES))
    bootstrap_confidence[SOMA_BODY_NAMES.index("LeftToeBase")] = 0.01
    with pytest.raises(ReferenceUnavailable, match="LeftToeBase"):
        bootstrap._condition_skeleton(
            np.zeros((len(SOMA_BODY_NAMES), 3)),
            np.repeat(((1.0, 0.0, 0.0, 0.0),), len(SOMA_BODY_NAMES), axis=0),
            bootstrap_confidence,
            SOMA_BODY_NAMES,
        )
