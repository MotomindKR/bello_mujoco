from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import math
from pathlib import Path
import time

import numpy as np
import pytest

from bello_mujoco.contracts import ActionLayout
from bello_mujoco.hardware.runtime import HardwareFrame
from bello_mujoco.hardware.sonic import (
    HardwareSonicConfig,
    HardwareSonicPolicyAdapter,
    SonicClipReferenceSource,
    SonicLiveReferenceSource,
    SonicReferenceWindow,
    validate_hardware_sonic_config,
)
from bello_mujoco.hardware.twist2 import Twist2ReferenceFrame
from bello_mujoco.motion.clip import MotionClip
from bello_mujoco.motion.sonic_dataset import SONIC_CRITIC_FUTURE_OFFSETS
from bello_mujoco.sonic_policy import (
    BELLO_TO_G1_VECTOR_BASIS,
    SONIC_G1_ACTION_SIZE,
    SONIC_G1_DEFAULT_JOINT_POSITION,
)


def _fake_clip_source() -> SonicClipReferenceSource:
    frames = max(SONIC_CRITIC_FUTURE_OFFSETS) + 3
    nq = 32
    nv = 31
    names = ("left_joint", "right_joint")
    default_qpos = np.linspace(-0.2, 0.2, nq)
    default_qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    env = SimpleNamespace(
        action_layout=ActionLayout(names),
        model=SimpleNamespace(
            default_qpos=default_qpos,
            mj_model=SimpleNamespace(nq=nq, nv=nv),
        ),
    )
    qpos = np.arange(frames * nq, dtype=np.float32).reshape(frames, nq)
    qvel = np.arange(frames * nv, dtype=np.float32).reshape(frames, nv)
    clip = MotionClip(
        fps=50.0,
        joint_names=names,
        body_names=(),
        source_sha256="sha256:" + "a" * 64,
        qpos=qpos,
        qvel=qvel,
        body_pos=np.empty((frames, 0, 3), dtype=np.float32),
        body_quat=np.empty((frames, 0, 4), dtype=np.float32),
        body_lin_vel=np.empty((frames, 0, 3), dtype=np.float32),
        body_ang_vel=np.empty((frames, 0, 3), dtype=np.float32),
        foot_contacts=np.empty((frames, 0), dtype=np.float32),
    )
    return SonicClipReferenceSource(
        clip,
        env,
        HardwareSonicConfig(motion_path=Path("clip.npz")),
    )


def test_sonic_clip_uses_canonical_standing_and_end_clamped_future() -> None:
    source = _fake_clip_source()

    standing = source.standing()
    np.testing.assert_allclose(standing.qpos[0], source.env.model.default_qpos)
    np.testing.assert_allclose(standing.qvel, 0.0)
    assert not np.array_equal(standing.qpos[0], np.asarray(source.clip.qpos)[0])

    while source.advance():
        pass
    window = source.latest()
    assert window.sequence == source.clip.num_frames - 1
    np.testing.assert_allclose(
        window.qpos,
        np.repeat(
            np.asarray(source.clip.qpos)[-1][None, :],
            len(SONIC_CRITIC_FUTURE_OFFSETS),
            axis=0,
        ),
    )


def test_sonic_live_prediction_keeps_root_translation(monkeypatch) -> None:
    nq = 10
    nv = 9
    current = np.zeros(nq, dtype=np.float64)
    current[2] = 0.8
    current[3] = 1.0
    reference = Twist2ReferenceFrame(
        qpos=current,
        qvel=np.ones(nv, dtype=np.float64),
        sequence=3,
        source_age_ms=10,
        received_at=time.monotonic(),
    )
    source = SimpleNamespace(standing=lambda: reference, latest=lambda: reference)
    env = SimpleNamespace(
        model=SimpleNamespace(
            root_qpos_adr=0,
            mj_model=object(),
        )
    )
    config = replace(
        HardwareSonicConfig(
            reference_mode="live",
            smplx_neutral_model_path=Path("neutral.npz"),
        ),
        max_reference_age_ms=500,
    )
    live = SonicLiveReferenceSource(source, env, config)

    monkeypatch.setattr(
        "bello_mujoco.hardware.sonic.validate_sonic_reference_frame",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "bello_mujoco.hardware.sonic._clip_reference_joints",
        lambda *_args: None,
    )

    def integrate(_model, qpos, _qvel, interval):
        qpos[:3] += interval

    monkeypatch.setattr("bello_mujoco.hardware.sonic.mujoco.mj_integratePos", integrate)

    predicted = live.latest()
    np.testing.assert_allclose(
        predicted.qpos[:, :3],
        np.repeat(current[None, :3], len(SONIC_CRITIC_FUTURE_OFFSETS), axis=0),
    )
    assert predicted.sequence == reference.sequence


def test_sonic_robot_input_preserves_released_temporal_packing() -> None:
    nq = 32
    nv = 31
    joint_map = np.zeros((SONIC_G1_ACTION_SIZE, 25), dtype=np.float64)
    joint_map[:25] = np.eye(25)
    model = SimpleNamespace(
        root_qpos_adr=0,
        policy_ctrl_indices=np.arange(25, dtype=np.int32),
        policy_qpos_indices=np.arange(7, 32, dtype=np.int32),
        policy_qvel_indices=np.arange(6, 31, dtype=np.int32),
        default_joint_target=np.zeros(25, dtype=np.float64),
    )
    adapter = object.__new__(HardwareSonicPolicyAdapter)
    adapter.env = SimpleNamespace(model=model, g1_from_bello_joint=joint_map)
    adapter.default_joint_target = model.default_joint_target
    adapter.policy_indices = model.policy_ctrl_indices
    adapter.policy_qpos_indices = model.policy_qpos_indices
    adapter.policy_qvel_indices = model.policy_qvel_indices
    adapter._reference_heading = None

    qpos = np.zeros((len(SONIC_CRITIC_FUTURE_OFFSETS), nq), dtype=np.float64)
    qvel = np.zeros((len(SONIC_CRITIC_FUTURE_OFFSETS), nv), dtype=np.float64)
    for frame in range(len(SONIC_CRITIC_FUTURE_OFFSETS)):
        angle = math.pi / 2.0
        qpos[frame, 3:7] = (
            math.cos(angle / 2.0),
            0.0,
            0.0,
            math.sin(angle / 2.0),
        )
        qpos[frame, 7:] = frame + np.arange(25) / 100.0
        qvel[frame, 6:] = 10.0 + frame + np.arange(25) / 100.0
    reference = SonicReferenceWindow(
        qpos=qpos,
        qvel=qvel,
        sequence=4,
        source_age_ms=None,
        received_at=time.monotonic(),
    )
    hardware = HardwareFrame(
        positions=np.zeros(25),
        velocities=np.zeros(25),
        orientation_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
        angular_velocity=np.zeros(3),
        linear_acceleration=np.zeros(3),
        max_motor_age_ms=0,
        imu_age_ms=0,
        max_group_age_ms=0,
        sample_span_ms=0.0,
        state_frame_round_trip_ms=0.0,
        group_states=(),
        group_causes=(),
        group_details=(),
    )

    packed = adapter._robot_encoder_input(hardware, reference)
    positions = np.stack(
        [SONIC_G1_DEFAULT_JOINT_POSITION + joint_map @ frame[7:] for frame in qpos]
    )
    velocities = np.stack([joint_map @ frame[6:] for frame in qvel])
    released_command = np.concatenate(
        (positions.reshape(-1), velocities.reshape(-1))
    ).reshape((len(SONIC_CRITIC_FUTURE_OFFSETS), 2 * SONIC_G1_ACTION_SIZE))
    identity_orientation = (
        np.asarray(BELLO_TO_G1_VECTOR_BASIS)
        @ np.eye(3)
        @ np.asarray(BELLO_TO_G1_VECTOR_BASIS).T
    )
    expected = np.concatenate(
        (
            released_command,
            np.repeat(
                identity_orientation[:, :2].reshape(1, 6),
                len(SONIC_CRITIC_FUTURE_OFFSETS),
                axis=0,
            ),
        ),
        axis=-1,
    ).reshape(-1)
    np.testing.assert_allclose(packed, expected, atol=1.0e-6)


def test_sonic_reference_modes_are_explicitly_validated() -> None:
    validate_hardware_sonic_config(HardwareSonicConfig(motion_path=Path("clip.npz")))
    validate_hardware_sonic_config(
        HardwareSonicConfig(
            reference_mode="live",
            smplx_neutral_model_path=Path("neutral.npz"),
        )
    )
    with pytest.raises(ValueError, match="motion_path is required"):
        validate_hardware_sonic_config(HardwareSonicConfig())
    with pytest.raises(ValueError, match="not valid.*reference_mode=live"):
        validate_hardware_sonic_config(
            HardwareSonicConfig(
                reference_mode="live",
                motion_path=Path("clip.npz"),
                smplx_neutral_model_path=Path("neutral.npz"),
            )
        )
