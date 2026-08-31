from __future__ import annotations

from contextlib import nullcontext
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
    _transition_to_sonic_standing_policy,
    validate_hardware_sonic_config,
    validate_sonic_transition_adapters,
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


def test_sonic_live_prediction_compensates_age_and_keeps_root_translation(
    monkeypatch,
) -> None:
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
        received_at=12.0,
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
    monkeypatch.setattr("bello_mujoco.hardware.twist2.time.monotonic", lambda: 12.04)

    intervals = []
    def integrate(_model, qpos, _qvel, interval):
        intervals.append(interval)
        qpos[:3] += interval
        qpos[-1] += interval

    monkeypatch.setattr("bello_mujoco.hardware.sonic.mujoco.mj_integratePos", integrate)

    predicted = live.latest()
    np.testing.assert_allclose(
        predicted.qpos[:, :3],
        np.repeat(current[None, :3], len(SONIC_CRITIC_FUTURE_OFFSETS), axis=0),
    )
    expected = 0.05 + np.asarray(SONIC_CRITIC_FUTURE_OFFSETS) / 50.0
    np.testing.assert_allclose(intervals, expected)
    np.testing.assert_allclose(predicted.qpos[:, -1], expected)
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


class _FakeStaticControl:
    def __init__(self, previous_target: np.ndarray) -> None:
        self.previous_target = previous_target.copy()
        self.frame = SimpleNamespace(max_motor_age_ms=1, imu_age_ms=2)
        self.targets: list[np.ndarray] = []
        self.drain_count = 0
        self.reset_deadline_count = 0
        self.pace_count = 0

    def read_frame(self, **_kwargs):
        return self.frame

    def send_target(self, target) -> None:
        value = np.asarray(target).copy()
        self.targets.append(value)
        self.previous_target = value

    def drain(self) -> None:
        self.drain_count += 1

    def reset_deadline(self) -> None:
        self.reset_deadline_count += 1

    def pace(self) -> None:
        self.pace_count += 1


def test_sonic_return_to_standing_runs_balance_policy_and_interpolates() -> None:
    control = _FakeStaticControl(np.asarray((0.0, 2.0)))
    target = np.asarray((2.0, 0.0))
    reset_count = 0

    def reset_adapter() -> None:
        nonlocal reset_count
        reset_count += 1

    policy_step = SimpleNamespace(
        physical_target=target,
        observation=np.zeros(2),
        policy_observation=np.zeros(2),
        raw_action=np.zeros(2),
        action=np.zeros(2),
    )
    adapter = SimpleNamespace(
        control_timestep=0.02,
        reset=reset_adapter,
        step=lambda frame, command: policy_step,
    )
    config = replace(
        HardwareSonicConfig(motion_path=Path("clip.npz")),
        policy_transition_seconds=0.04,
    )

    _transition_to_sonic_standing_policy(
        control,
        adapter,
        np.zeros(3, dtype=np.float32),
        config,
        policy_log=None,
        run_started=time.monotonic(),
    )

    np.testing.assert_allclose(control.targets, ((1.0, 1.0), (2.0, 0.0)))
    assert control.adapter is adapter
    assert control.drain_count == 2
    assert control.reset_deadline_count == 1
    assert control.pace_count == 2
    assert reset_count == 1


def test_sonic_live_motion_reenters_after_reference_recovery(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from bello_mujoco.hardware import sonic as sonic_hardware

    frame = HardwareFrame(
        positions=np.zeros(2),
        velocities=np.zeros(2),
        orientation_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
        angular_velocity=np.zeros(3),
        linear_acceleration=np.zeros(3),
        max_motor_age_ms=0,
        imu_age_ms=0,
        max_group_age_ms=0,
        sample_span_ms=0.0,
        state_frame_round_trip_ms=0.0,
        group_states=("RUNNING", "RUNNING"),
        group_causes=("UNSPECIFIED", "UNSPECIFIED"),
        group_details=("", ""),
    )
    transitions: list[str] = []
    sonic_sequences: list[int] = []
    confirmations = 0

    class FakeStandingAdapter:
        control_timestep = 0.02

        def reset(self) -> None:
            return None

        def step(self, _frame, _command):
            values = np.zeros(2)
            return SimpleNamespace(
                physical_target=values,
                observation=values,
                policy_observation=values,
                raw_action=values,
                action=values,
                tilt_radians=0.0,
            )

    class FakeSonicAdapter:
        control_timestep = 0.02

        def reset(self) -> None:
            return None

        def validate_reference(self, _reference) -> None:
            return None

        def step(self, _frame, reference):
            sonic_sequences.append(reference.sequence)
            values = np.zeros(2)
            return SimpleNamespace(
                physical_target=values,
                observation=values,
                policy_observation=values,
                raw_source_action=values,
                action=values,
                command=values,
                action_clipped=False,
                reference_sequence=reference.sequence,
                reference_age_ms=0.0,
            )

    standing_adapter = FakeStandingAdapter()
    sonic_adapter = FakeSonicAdapter()

    def reference(sequence: int) -> SonicReferenceWindow:
        return SonicReferenceWindow(
            qpos=np.zeros((1, 1)),
            qvel=np.zeros((1, 1)),
            sequence=sequence,
            source_age_ms=0,
            received_at=time.monotonic(),
        )

    class FakeReferences:
        mode = "live"
        calls = 0

        def reset(self) -> None:
            self.calls = 0

        def latest(self) -> SonicReferenceWindow:
            self.calls += 1
            if self.calls == 3:
                raise sonic_hardware.ReferenceUnavailable("tracking stalled")
            if self.calls == 6:
                raise KeyboardInterrupt
            return reference(self.calls)

        def advance(self) -> bool:
            return True

    class FakeControl:
        previous_target = np.zeros(2)

        def __init__(self) -> None:
            self._adapter = standing_adapter

        @property
        def adapter(self):
            return self._adapter

        @adapter.setter
        def adapter(self, adapter) -> None:
            self._adapter = adapter
            transitions.append("sonic" if adapter is sonic_adapter else "standing")

        def ramp_to_nominal(self, _positions):
            return frame

        def wait_for_run(
            self, _prompt, *, operator_input, validate_hold
        ) -> bool:
            del operator_input
            validate_hold(frame)
            return True

        def read_frame(self, *, max_tilt_radians=None):
            del max_tilt_radians
            return frame

        def send_target(self, target) -> None:
            self.previous_target = np.asarray(target)

        def pace(self) -> None:
            time.sleep(0.001)

        def reset_deadline(self) -> None:
            return None

        def drain(self) -> None:
            return None

    control = FakeControl()
    monkeypatch.setattr(
        sonic_hardware,
        "armed_control_session",
        lambda *_args, **_kwargs: nullcontext(control),
    )
    monkeypatch.setattr(
        sonic_hardware,
        "require_nominal_pose",
        lambda *_args, **_kwargs: None,
    )

    def confirm_motion(expected, _prompt, hold_step, _operator_input) -> bool:
        nonlocal confirmations
        assert expected == "MOTION"
        assert hold_step()
        confirmations += 1
        return True

    monkeypatch.setattr(
        sonic_hardware,
        "wait_for_operator_confirmation_while_holding",
        confirm_motion,
    )

    sonic_hardware.control_physical_sonic(
        SimpleNamespace(),
        {},
        SimpleNamespace(),
        FakeReferences(),
        frame,
        standing_adapter,
        sonic_adapter,
        HardwareSonicConfig(
            reference_mode="live",
            smplx_neutral_model_path=Path("neutral.npz"),
            max_duration_seconds=0.04,
            policy_transition_seconds=0.001,
            gemx_reference_ready_timeout_seconds=0.01,
            gemx_reference_qualification_frames=1,
        ),
    )

    assert confirmations == 1
    assert sonic_sequences == [1, 2, 4, 5]
    assert transitions == ["sonic", "standing", "sonic", "standing"]
    output = capsys.readouterr().out
    assert "reference became unavailable" in output
    assert "MOTION remains armed" in output
    assert output.count("Transitioning from standing control") == 2


def test_sonic_transition_requires_identical_standing_hardware_contract() -> None:
    common = {
        "physical_coordinate_names": ("left", "right"),
        "nominal_physical_target": np.asarray((0.1, -0.2)),
        "kp": np.asarray((10.0, 20.0)),
        "kd": np.asarray((1.0, 2.0)),
        "control_timestep": 0.02,
    }
    validate_sonic_transition_adapters(
        SimpleNamespace(**common),
        SimpleNamespace(**common),
    )
    incompatible = dict(common)
    incompatible["nominal_physical_target"] = np.asarray((0.1, -0.1))
    with pytest.raises(ValueError, match="different nominal poses"):
        validate_sonic_transition_adapters(
            SimpleNamespace(**common),
            SimpleNamespace(**incompatible),
        )
