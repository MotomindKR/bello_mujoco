from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import math
import queue
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import grpc
import numpy as np
import pytest

from bello_mujoco.hardware import rtmw_hands
from bello_mujoco.hardware.rtmw import (
    RTMW_COORDINATE_SPACE,
    RTMW_REPRESENTATION,
    RTMW_ROTATION_SEMANTICS,
    RTMW_SCHEMA_ID,
    RTMW_SMPLX52_NAMES,
    RTMW_SMPLX52_PARENTS,
)
from bello_mujoco.hardware.rtmw_hands import (
    RTMW_FINGER_TIP_NAMES,
    HandRawCalibration,
    RtmwHandCommandFilter,
    RtmwHandLandmarkMapper,
    RtmwHandTarget,
    RtmwHandsConfig,
    RtmwHandsDiscontinuity,
    RtmwHandsFrame,
    RtmwHandsStream,
    RtmwHandsUnavailable,
    run_rtmw_hands,
)


_END_STREAM = object()


def test_importing_hand_runner_does_not_load_policy_or_simulation_stacks() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import bello_mujoco.hardware.rtmw_hands; "
                "assert 'jax' not in sys.modules; "
                "assert 'mujoco' not in sys.modules; "
                "assert 'torch' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def hands_config(**overrides: object) -> RtmwHandsConfig:
    values: dict[str, object] = {
        "endpoint": "127.0.0.1:50055",
        "min_confidence": 0.3,
        "max_pipeline_latency_ms": 700.0,
        "transport_latency_budget_ms": 10.0,
        "max_reference_age_ms": 1_000,
        "max_source_interval_seconds": 0.7,
        "smoothing_time_constant_ms": 500.0,
        "max_raw_rate_per_second": 255.0,
    }
    values.update(overrides)
    return RtmwHandsConfig(**values)


def neutral_landmarks() -> tuple[np.ndarray, np.ndarray]:
    joints = np.zeros((len(RTMW_SMPLX52_NAMES), 3), dtype=np.float64)
    tips = np.zeros((len(RTMW_FINGER_TIP_NAMES), 3), dtype=np.float64)
    joint_index = {name: index for index, name in enumerate(RTMW_SMPLX52_NAMES)}
    tip_index = {name: index for index, name in enumerate(RTMW_FINGER_TIP_NAMES)}
    lateral = {"index": 0.030, "middle": 0.010, "ring": -0.010, "pinky": -0.030}
    for side, mirror, offset in (("right", 1.0, 0.2), ("left", -1.0, -0.2)):
        wrist = np.asarray((offset, 0.0, 0.0))
        joints[joint_index[f"{side}_wrist"]] = wrist
        for finger, x in lateral.items():
            base = wrist + np.asarray((mirror * x, 0.050, 0.0))
            joints[joint_index[f"{side}_{finger}1"]] = base
            joints[joint_index[f"{side}_{finger}2"]] = base + (0.0, 0.030, 0.0)
            joints[joint_index[f"{side}_{finger}3"]] = base + (0.0, 0.055, 0.0)
            tips[tip_index[f"{side}_{finger}_tip"]] = base + (0.0, 0.080, 0.0)
        thumb_base = wrist + np.asarray((mirror * 0.035, 0.025, 0.0))
        radial = np.asarray((mirror, 0.0, 0.0))
        joints[joint_index[f"{side}_thumb1"]] = thumb_base
        joints[joint_index[f"{side}_thumb2"]] = thumb_base + 0.025 * radial
        joints[joint_index[f"{side}_thumb3"]] = thumb_base + 0.050 * radial
        tips[tip_index[f"{side}_thumb_tip"]] = thumb_base + 0.075 * radial
    return joints, tips


def curled_landmarks() -> tuple[np.ndarray, np.ndarray]:
    joints, tips = neutral_landmarks()
    joint_index = {name: index for index, name in enumerate(RTMW_SMPLX52_NAMES)}
    tip_index = {name: index for index, name in enumerate(RTMW_FINGER_TIP_NAMES)}
    for side, mirror in (("right", 1.0), ("left", -1.0)):
        for finger in ("index", "middle", "ring", "pinky"):
            base = joints[joint_index[f"{side}_{finger}1"]]
            vectors = [
                0.030 * np.asarray((0.0, math.cos(angle), math.sin(angle)))
                for angle in map(math.radians, (30.0, 90.0, 150.0))
            ]
            pip = base + vectors[0]
            dip = pip + vectors[1]
            tip = dip + vectors[2]
            joints[joint_index[f"{side}_{finger}2"]] = pip
            joints[joint_index[f"{side}_{finger}3"]] = dip
            tips[tip_index[f"{side}_{finger}_tip"]] = tip

        thumb_base = joints[joint_index[f"{side}_thumb1"]]
        radial = np.asarray((mirror, 0.0, 0.0))
        palmar = np.asarray((0.0, 0.0, 1.0))
        vectors = [
            0.025 * (math.cos(angle) * radial + math.sin(angle) * palmar)
            for angle in map(math.radians, (20.0, 60.0, 100.0))
        ]
        thumb2 = thumb_base + vectors[0]
        thumb3 = thumb2 + vectors[1]
        thumb_tip = thumb3 + vectors[2]
        joints[joint_index[f"{side}_thumb2"]] = thumb2
        joints[joint_index[f"{side}_thumb3"]] = thumb3
        tips[tip_index[f"{side}_thumb_tip"]] = thumb_tip
    return joints, tips


def map_landmarks(
    mapper: RtmwHandLandmarkMapper,
    side: str,
    joints: np.ndarray,
    tips: np.ndarray,
) -> RtmwHandTarget:
    return mapper.map_side(
        side,  # type: ignore[arg-type]
        joints,
        np.full(len(RTMW_SMPLX52_NAMES), 0.9),
        np.ones(len(RTMW_SMPLX52_NAMES), dtype=bool),
        tips,
        np.full(len(RTMW_FINGER_TIP_NAMES), 0.9),
    )


def rtmw_frame(
    captured_at_ns: int,
    *,
    frame_id: int = 1,
    stream_instance_id: str = "test-stream",
    track_id: str = "test-track",
) -> SimpleNamespace:
    joints, tips = curled_landmarks()
    return SimpleNamespace(
        frame_id=frame_id,
        source_timestamp_ns=captured_at_ns,
        captured_monotonic_ns=captured_at_ns,
        output_monotonic_ns=captured_at_ns + 10_000_000,
        stream_instance_id=stream_instance_id,
        track_id=track_id,
        tracking_state="tracked",
        joint_positions=joints.reshape(-1).tolist(),
        joint_confidence=np.full(len(RTMW_SMPLX52_NAMES), 0.9).tolist(),
        joint_valid=np.ones(len(RTMW_SMPLX52_NAMES), dtype=bool).tolist(),
        finger_tip_positions=tips.reshape(-1).tolist(),
        finger_tip_confidence=np.full(len(RTMW_FINGER_TIP_NAMES), 0.9).tolist(),
        pipeline_latency_ms=10.0,
    )


def stream_info(**overrides: object) -> SimpleNamespace:
    values = {
        "schema_version": 1,
        "stream_instance_id": "test-stream",
        "config_sha256": "1" * 64,
        "joint_names": RTMW_SMPLX52_NAMES,
        "joint_parents": RTMW_SMPLX52_PARENTS,
        "root_index": 0,
        "coordinate_space": RTMW_COORDINATE_SPACE,
        "quaternion_order": "xyzw",
        "rotation_semantics": RTMW_ROTATION_SEMANTICS,
        "position_units": "meters",
        "representation": RTMW_REPRESENTATION,
        "schema_id": RTMW_SCHEMA_ID,
        "depth_provenance": "calibrated_camera+assumed_subject_height",
        "metric_scale": True,
        "global_translation": True,
        "global_depth_is_measured": False,
        "subject_height_m": 1.75,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_neutral_and_curled_hands_are_mirrored_consistently() -> None:
    mapper = RtmwHandLandmarkMapper(hands_config())
    neutral_joints, neutral_tips = neutral_landmarks()
    curled_joints, curled_tips = curled_landmarks()

    neutral_left = map_landmarks(mapper, "left", neutral_joints, neutral_tips)
    neutral_right = map_landmarks(mapper, "right", neutral_joints, neutral_tips)
    curled_left = map_landmarks(mapper, "left", curled_joints, curled_tips)
    curled_right = map_landmarks(mapper, "right", curled_joints, curled_tips)

    assert neutral_left.normalized_positions == pytest.approx((0.0,) * 6)
    assert neutral_right.normalized_positions == pytest.approx((0.0,) * 6)
    assert curled_left.normalized_positions == pytest.approx(
        curled_right.normalized_positions
    )
    assert curled_left.normalized_positions[0] == pytest.approx(100.0 / 120.0)
    assert curled_left.normalized_positions[1] == pytest.approx(0.0)
    assert curled_left.normalized_positions[2:] == pytest.approx((150.0 / 180.0,) * 4)


def test_mapping_is_invariant_to_camera_rigid_transform() -> None:
    mapper = RtmwHandLandmarkMapper(hands_config())
    joints, tips = curled_landmarks()
    expected = map_landmarks(mapper, "right", joints, tips)
    angle = 0.7
    rotation = np.asarray(
        (
            (math.cos(angle), -math.sin(angle), 0.0),
            (math.sin(angle), math.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    translation = np.asarray((0.7, -0.2, 1.3))

    actual = map_landmarks(
        mapper,
        "right",
        joints @ rotation.T + translation,
        tips @ rotation.T + translation,
    )

    assert actual.normalized_positions == pytest.approx(
        expected.normalized_positions, abs=1.0e-12
    )


def test_thumb_adduction_is_palm_local_and_side_correct() -> None:
    mapper = RtmwHandLandmarkMapper(hands_config())
    joints, tips = neutral_landmarks()
    joint_index = {name: index for index, name in enumerate(RTMW_SMPLX52_NAMES)}
    tip_index = {name: index for index, name in enumerate(RTMW_FINGER_TIP_NAMES)}
    for side, mirror in (("right", 1.0), ("left", -1.0)):
        thumb1 = joints[joint_index[f"{side}_thumb1"]]
        direction = np.asarray((mirror * 0.5, math.sqrt(3.0) / 2.0, 0.0))
        joints[joint_index[f"{side}_thumb2"]] = thumb1 + 0.025 * direction
        joints[joint_index[f"{side}_thumb3"]] = thumb1 + 0.050 * direction
        tips[tip_index[f"{side}_thumb_tip"]] = thumb1 + 0.075 * direction

    left = map_landmarks(mapper, "left", joints, tips)
    right = map_landmarks(mapper, "right", joints, tips)

    assert left.normalized_positions[1] == pytest.approx(1.0)
    assert right.normalized_positions[1] == pytest.approx(1.0)


def test_stream_keeps_the_good_hand_when_other_side_confidence_drops() -> None:
    stream = RtmwHandsStream(hands_config())
    stream.stream_instance_id = "test-stream"
    message = rtmw_frame(5_000_000_000)
    left_thumb_tip = RTMW_FINGER_TIP_NAMES.index("left_thumb_tip")
    message.finger_tip_confidence[left_thumb_tip] = 0.1

    frame = stream._parse_frame(message)

    assert frame.left is None
    assert "confidence" in str(frame.left_unavailable_reason)
    assert frame.right is not None


def test_stream_rejects_frames_when_both_hands_are_unavailable() -> None:
    stream = RtmwHandsStream(hands_config())
    stream.stream_instance_id = "test-stream"
    message = rtmw_frame(5_000_000_000)
    message.finger_tip_confidence = [0.0] * len(RTMW_FINGER_TIP_NAMES)

    with pytest.raises(RtmwHandsUnavailable, match="landmarks are unavailable"):
        stream._parse_frame(message)


def test_stream_gates_tracking_latency_and_source_intervals() -> None:
    stream = RtmwHandsStream(hands_config(max_pipeline_latency_ms=20.0))
    stream.stream_instance_id = "test-stream"
    lost = rtmw_frame(5_000_000_000)
    lost.tracking_state = "lost"
    with pytest.raises(RtmwHandsUnavailable, match="tracking state"):
        stream._parse_frame(lost)

    late = rtmw_frame(6_000_000_000)
    late.pipeline_latency_ms = 21.0
    late.output_monotonic_ns = late.captured_monotonic_ns + 21_000_000
    with pytest.raises(RtmwHandsUnavailable, match="too high"):
        stream._parse_frame(late)

    first = rtmw_frame(7_000_000_000)
    stream._parse_frame(first)
    gap = rtmw_frame(7_700_000_001, frame_id=2)
    with pytest.raises(RtmwHandsDiscontinuity, match="source interval"):
        stream._parse_frame(gap)


def test_command_filter_keeps_flexion_binary_with_inverted_calibration() -> None:
    config = hands_config(
        smoothing_time_constant_ms=0.0,
        max_raw_rate_per_second=100.0,
        left_calibration=HandRawCalibration(
            open_positions=(200, 0, 0, 0, 0, 0),
            closed_positions=(0, 255, 255, 255, 255, 255),
        ),
    )
    command_filter = RtmwHandCommandFilter(config)
    opened = RtmwHandTarget("left", (0.0,) * 6, 1.0)
    closed = RtmwHandTarget("left", (1.0,) * 6, 1.0)

    assert command_filter.update(opened, now=1.0) == (200, 0, 0, 0, 0, 0)
    assert command_filter.update(closed, now=1.1) == (0, 10, 255, 255, 255, 255)


def test_command_filter_applies_lpf_only_to_thumb_abduction() -> None:
    config = hands_config(
        smoothing_time_constant_ms=500.0,
        max_raw_rate_per_second=10_000.0,
    )
    command_filter = RtmwHandCommandFilter(config)
    opened = RtmwHandTarget("right", (0.0,) * 6, 1.0)
    closed = RtmwHandTarget("right", (1.0,) * 6, 1.0)

    command_filter.update(opened, now=1.0)
    actual = command_filter.update(closed, now=1.5)

    expected = math.floor(255.0 * math.exp(-1.0) + 0.5)
    assert actual == (0, expected, 0, 0, 0, 0)


def test_command_filter_slew_limit_applies_only_to_thumb_abduction() -> None:
    config = hands_config(
        control_rate_hz=10.0,
        smoothing_time_constant_ms=0.0,
        max_raw_rate_per_second=100.0,
    )
    command_filter = RtmwHandCommandFilter(config)
    command_filter.seed("right", (200,) * 6, now=1.0)
    command_filter.resume("right", now=20.0)

    actual = command_filter.update(
        RtmwHandTarget("right", (1.0,) * 6, 1.0),
        now=20.0,
    )

    assert actual == (0, 190, 0, 0, 0, 0)


def test_command_filter_uses_binary_flexion_hysteresis() -> None:
    config = hands_config(
        smoothing_time_constant_ms=0.0,
        max_raw_rate_per_second=10_000.0,
        flexion_open_threshold=0.35,
        flexion_close_threshold=0.65,
    )
    command_filter = RtmwHandCommandFilter(config)

    first = command_filter.update(
        RtmwHandTarget("left", (0.7, 0.0, 0.7, 0.2, 0.8, 0.4), 1.0),
        now=1.0,
    )
    deadband = command_filter.update(
        RtmwHandTarget("left", (0.5, 0.0, 0.5, 0.5, 0.5, 0.5), 1.0),
        now=1.1,
    )
    opened = command_filter.update(
        RtmwHandTarget("left", (0.3, 0.0, 0.3, 0.3, 0.3, 0.3), 1.0),
        now=1.2,
    )

    assert first == (0, 255, 0, 255, 0, 255)
    assert deadband == first
    assert opened == (255,) * 6


def test_command_filter_reseed_resets_binary_flexion_latches() -> None:
    config = hands_config(
        smoothing_time_constant_ms=0.0,
        max_raw_rate_per_second=10_000.0,
    )
    command_filter = RtmwHandCommandFilter(config)
    closed = RtmwHandTarget("right", (1.0,) * 6, 1.0)
    ambiguous = RtmwHandTarget("right", (0.5,) * 6, 1.0)

    assert command_filter.update(closed, now=1.0) == (0,) * 6
    command_filter.seed("right", (255,) * 6, now=1.1)
    command_filter.resume("right", now=1.2)
    assert command_filter.update(ambiguous, now=1.2) == (255, 128, 255, 255, 255, 255)


@pytest.mark.parametrize(
    "overrides",
    (
        {"flexion_open_threshold": -0.1},
        {"flexion_close_threshold": 1.1},
        {"flexion_open_threshold": 0.5, "flexion_close_threshold": 0.5},
        {"flexion_open_threshold": 0.7, "flexion_close_threshold": 0.6},
    ),
)
def test_config_rejects_invalid_flexion_hysteresis(overrides: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="flexion_"):
        hands_config(**overrides)


def target_frame(*, received_at: float | None = None) -> RtmwHandsFrame:
    left = RtmwHandTarget("left", (0.1,) * 6, 0.9)
    right = RtmwHandTarget("right", (0.2,) * 6, 0.9)
    return RtmwHandsFrame(
        sequence=1,
        source_timestamp_ns=1,
        source_age_ms=10,
        received_at=time.monotonic() if received_at is None else received_at,
        track_id="track",
        left=left,
        right=right,
    )


def test_unarmed_run_receives_without_constructing_transport() -> None:
    stop = threading.Event()

    class FakeStream:
        def __enter__(self) -> FakeStream:
            return self

        def __exit__(self, *_error: object) -> bool:
            return False

        def wait_next(self, **_arguments: object) -> RtmwHandsFrame:
            stop.set()
            return target_frame()

    def forbidden_transport(_config: RtmwHandsConfig):
        raise AssertionError("unarmed validation must not touch CAN transport")

    run_rtmw_hands(
        hands_config(arm=False),
        forbidden_transport,
        stop_event=stop,
        stream_factory=lambda _config: FakeStream(),  # type: ignore[arg-type]
    )


def test_unarmed_duration_expiry_is_not_reported_as_success() -> None:
    class EmptyStream:
        def __enter__(self) -> EmptyStream:
            return self

        def __exit__(self, *_error: object) -> bool:
            return False

        def wait_next(self, **_arguments: object) -> None:
            return None

    with pytest.raises(TimeoutError, match="before an acceptable hand frame"):
        run_rtmw_hands(
            hands_config(max_duration_seconds=0.002),
            lambda _config: (_ for _ in ()).throw(
                AssertionError("unarmed preflight must not construct CAN transport")
            ),
            stream_factory=lambda _config: EmptyStream(),  # type: ignore[arg-type]
        )


def test_armed_config_requires_explicit_calibration_confirmation() -> None:
    with pytest.raises(ValueError, match="calibration_confirmed=true"):
        hands_config(arm=True)


def test_armed_run_sends_nothing_before_first_rtmw_target() -> None:
    stop = threading.Event()
    commands: list[tuple[str, tuple[int, ...]]] = []
    refreshes: list[str] = []

    class EmptyStream:
        def __enter__(self) -> EmptyStream:
            return self

        def __exit__(self, *_error: object) -> bool:
            return False

        def wait_next(self, **_arguments: object) -> None:
            stop.set()
            return None

    class FakeTransport:
        initial_positions = {
            "left": (255, 255, 255, 255, 255, 255),
            "right": (255, 255, 255, 255, 255, 255),
        }

        def set_positions(self, side: str, positions: tuple[int, ...]) -> None:
            commands.append((side, positions))

        def refresh_positions(self, side: str) -> tuple[int, ...]:
            refreshes.append(side)
            return self.initial_positions[side]

    @contextmanager
    def transport(_config: RtmwHandsConfig):
        yield FakeTransport()

    run_rtmw_hands(
        hands_config(arm=True, calibration_confirmed=True),
        transport,  # type: ignore[arg-type]
        stop_event=stop,
        stream_factory=lambda _config: EmptyStream(),  # type: ignore[arg-type]
    )

    assert commands == []
    assert refreshes == []


def test_outage_during_recovery_probe_prevents_obsolete_write() -> None:
    stop = threading.Event()
    commands: list[tuple[int, ...]] = []
    outage_pending = False
    calls = 0
    frame = replace(target_frame(), right=None)

    class FakeStream:
        def __enter__(self) -> FakeStream:
            return self

        def __exit__(self, *_error: object) -> bool:
            return False

        def wait_next(self, **_arguments: object) -> RtmwHandsFrame | None:
            nonlocal calls
            calls += 1
            if calls == 1:
                return frame
            if outage_pending:
                stop.set()
                raise RtmwHandsDiscontinuity("outage published during probe")
            raise AssertionError("the recovery probe must publish an outage")

    class FakeTransport:
        initial_positions = {"left": (255,) * 6}

        def refresh_positions(self, _side: str) -> tuple[int, ...]:
            nonlocal outage_pending
            outage_pending = True
            return (200,) * 6

        def set_positions(self, _side: str, positions: tuple[int, ...]) -> None:
            commands.append(positions)

    @contextmanager
    def transport(_config: RtmwHandsConfig):
        yield FakeTransport()

    run_rtmw_hands(
        hands_config(
            arm=True,
            calibration_confirmed=True,
            right_enabled=False,
        ),
        transport,  # type: ignore[arg-type]
        stop_event=stop,
        stream_factory=lambda _config: FakeStream(),  # type: ignore[arg-type]
    )

    assert commands == []


def test_armed_run_emits_only_binary_flexion_for_live_target() -> None:
    stop = threading.Event()
    commands: list[tuple[int, ...]] = []
    calls = 0
    target = replace(
        target_frame(),
        left=RtmwHandTarget("left", (0.8, 0.5, 0.8, 0.8, 0.8, 0.8), 0.9),
        right=None,
    )

    class FakeStream:
        def __enter__(self) -> FakeStream:
            return self

        def __exit__(self, *_error: object) -> bool:
            return False

        def wait_next(self, *, timeout_seconds: float, **_arguments: object):
            nonlocal calls
            calls += 1
            if calls == 1:
                return target
            if calls == 2:
                return replace(target, sequence=2, received_at=time.monotonic())
            time.sleep(timeout_seconds)
            return None

    class FakeTransport:
        initial_positions = {"left": (255,) * 6}

        def refresh_positions(self, _side: str) -> tuple[int, ...]:
            return (255,) * 6

        def set_positions(self, _side: str, positions: tuple[int, ...]) -> None:
            commands.append(positions)
            stop.set()

    @contextmanager
    def transport(_config: RtmwHandsConfig):
        yield FakeTransport()

    run_rtmw_hands(
        hands_config(
            arm=True,
            calibration_confirmed=True,
            right_enabled=False,
            smoothing_time_constant_ms=0.0,
            max_raw_rate_per_second=1.0e9,
        ),
        transport,  # type: ignore[arg-type]
        stop_event=stop,
        stream_factory=lambda _config: FakeStream(),  # type: ignore[arg-type]
    )

    assert commands == [(0, 128, 0, 0, 0, 0)]


def test_target_expiring_during_recovery_probe_sends_no_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop = threading.Event()
    now = 100.0
    commands: list[tuple[int, ...]] = []
    calls = 0
    frame = replace(target_frame(received_at=now), right=None)

    def monotonic() -> float:
        return now

    monkeypatch.setattr(rtmw_hands, "time", SimpleNamespace(monotonic=monotonic))

    class FakeStream:
        def __enter__(self) -> FakeStream:
            return self

        def __exit__(self, *_error: object) -> bool:
            return False

        def wait_next(self, **_arguments: object) -> RtmwHandsFrame | None:
            nonlocal calls
            calls += 1
            if calls == 1:
                return frame
            stop.set()
            return None

    class FakeTransport:
        initial_positions = {"left": (255,) * 6}

        def refresh_positions(self, _side: str) -> tuple[int, ...]:
            nonlocal now
            now += 2.0
            return (200,) * 6

        def set_positions(self, _side: str, positions: tuple[int, ...]) -> None:
            commands.append(positions)

    @contextmanager
    def transport(_config: RtmwHandsConfig):
        yield FakeTransport()

    run_rtmw_hands(
        hands_config(
            arm=True,
            calibration_confirmed=True,
            right_enabled=False,
        ),
        transport,  # type: ignore[arg-type]
        stop_event=stop,
        stream_factory=lambda _config: FakeStream(),  # type: ignore[arg-type]
    )

    assert commands == []


def test_rtmw_outage_suspends_writes_and_recovery_refreshes_and_reseeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Clock:
        now = 0.0

        def monotonic(self) -> float:
            return self.now

        def sleep(self, seconds: float) -> None:
            self.now += seconds

    time = Clock()
    monkeypatch.setattr(rtmw_hands, "time", time)
    stop = threading.Event()
    commands: list[tuple[int, ...]] = []
    refreshes: list[tuple[int, ...]] = []
    calls = 0
    closed = replace(
        target_frame(),
        received_at=time.monotonic(),
        left=RtmwHandTarget("left", (1.0,) * 6, 0.9),
        right=None,
    )

    class FakeStream:
        def __enter__(self) -> FakeStream:
            return self

        def __exit__(self, *_error: object) -> bool:
            return False

        def wait_next(self, *, timeout_seconds: float, **_arguments: object):
            nonlocal calls
            calls += 1
            if calls == 1:
                return closed
            if calls == 2:
                return replace(closed, sequence=2, received_at=time.monotonic())
            if calls == 3:
                time.sleep(timeout_seconds)
                return None
            if calls == 4:
                time.sleep(timeout_seconds)
                raise RtmwHandsUnavailable("test outage")
            if calls == 5:
                return replace(
                    closed,
                    sequence=3,
                    received_at=time.monotonic(),
                )
            if calls == 6:
                time.sleep(timeout_seconds)
                return None
            if calls == 7:
                return replace(
                    closed,
                    sequence=4,
                    received_at=time.monotonic(),
                )
            time.sleep(timeout_seconds)
            return None

    class FakeTransport:
        initial_positions = {"left": (255,) * 6}

        def __init__(self) -> None:
            self.samples = iter(
                (
                    (255, 200, 255, 255, 255, 255),
                    (255, 50, 255, 255, 255, 255),
                )
            )

        def refresh_positions(self, _side: str) -> tuple[int, ...]:
            sample = next(self.samples)
            refreshes.append(sample)
            return sample

        def set_positions(self, _side: str, positions: tuple[int, ...]) -> None:
            commands.append(positions)
            if len(commands) == 2:
                stop.set()

    @contextmanager
    def transport(_config: RtmwHandsConfig):
        yield FakeTransport()

    run_rtmw_hands(
        hands_config(
            arm=True,
            calibration_confirmed=True,
            right_enabled=False,
            smoothing_time_constant_ms=0.0,
            max_raw_rate_per_second=100.0,
        ),
        transport,  # type: ignore[arg-type]
        stop_event=stop,
        stream_factory=lambda _config: FakeStream(),  # type: ignore[arg-type]
    )

    assert refreshes == [
        (255, 200, 255, 255, 255, 255),
        (255, 50, 255, 255, 255, 255),
    ]
    assert commands == [
        (0, 180, 0, 0, 0, 0),
        (0, 30, 0, 0, 0, 0),
    ]


def test_missing_rtmw_frames_send_no_positions() -> None:
    stop = threading.Event()
    commands: list[tuple[int, ...]] = []

    class EmptyStream:
        def __enter__(self) -> EmptyStream:
            return self

        def __exit__(self, *_error: object) -> bool:
            return False

        def wait_next(self, *, timeout_seconds: float, **_arguments: object) -> None:
            del timeout_seconds
            stop.set()
            return None

    class FakeTransport:
        initial_positions = {"left": (255,) * 6}

        def set_positions(self, _side: str, positions: tuple[int, ...]) -> None:
            commands.append(positions)

        def refresh_positions(self, _side: str) -> tuple[int, ...]:
            raise AssertionError("missing RTMW data must not probe a hand")

    @contextmanager
    def transport(_config: RtmwHandsConfig):
        yield FakeTransport()

    run_rtmw_hands(
        hands_config(
            arm=True,
            calibration_confirmed=True,
            right_enabled=False,
        ),
        transport,  # type: ignore[arg-type]
        stop_event=stop,
        stream_factory=lambda _config: EmptyStream(),  # type: ignore[arg-type]
    )

    assert commands == []


def test_stale_rtmw_target_sends_no_positions() -> None:
    stop = threading.Event()
    commands: list[tuple[int, ...]] = []
    stale = replace(
        target_frame(received_at=time.monotonic() - 2.0),
        left=RtmwHandTarget("left", (1.0,) * 6, 0.9),
        right=None,
    )

    class FakeStream:
        def __enter__(self) -> FakeStream:
            return self

        def __exit__(self, *_error: object) -> bool:
            return False

        def wait_next(self, **_arguments: object) -> RtmwHandsFrame:
            stop.set()
            return stale

    class FakeTransport:
        initial_positions = {"left": (255,) * 6}

        def set_positions(self, _side: str, positions: tuple[int, ...]) -> None:
            commands.append(positions)

        def refresh_positions(self, _side: str) -> tuple[int, ...]:
            raise AssertionError("a stale target must not probe a hand")

    @contextmanager
    def transport(_config: RtmwHandsConfig):
        yield FakeTransport()

    run_rtmw_hands(
        hands_config(
            arm=True,
            calibration_confirmed=True,
            right_enabled=False,
        ),
        transport,  # type: ignore[arg-type]
        stop_event=stop,
        stream_factory=lambda _config: FakeStream(),  # type: ignore[arg-type]
    )

    assert commands == []


def test_side_missing_then_valid_before_tick_requires_probe_and_newer_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop = threading.Event()
    commands: list[tuple[str, tuple[int, ...]]] = []
    refreshes: list[str] = []
    now = 100.0
    calls = 0

    def monotonic() -> float:
        return now

    def frame(sequence: int, *, right_available: bool = True) -> RtmwHandsFrame:
        result = replace(
            target_frame(received_at=now),
            sequence=sequence,
            left=RtmwHandTarget("left", (1.0,) * 6, 0.9),
        )
        if right_available:
            return result
        return replace(
            result,
            right=None,
            right_unavailable_reason="low confidence",
        )

    monkeypatch.setattr(rtmw_hands, "time", SimpleNamespace(monotonic=monotonic))

    class FakeStream:
        def __enter__(self) -> FakeStream:
            return self

        def __exit__(self, *_error: object) -> bool:
            return False

        def wait_next(self, *, timeout_seconds: float, **_arguments: object):
            nonlocal calls, now
            calls += 1
            if calls == 1:
                return frame(1)
            if calls == 2:
                return frame(2)
            if calls == 3:
                now += timeout_seconds
                return None
            if calls == 4:
                return frame(3)
            if calls == 5:
                now += timeout_seconds
                return None
            if calls == 6:
                return frame(4, right_available=False)
            if calls == 7:
                return frame(5)
            if calls == 8:
                now += timeout_seconds
                return None
            if calls == 9:
                return frame(6)
            if calls == 10:
                now += timeout_seconds
                return None
            raise AssertionError("hand runner did not stop after recovered-side write")

    class FakeTransport:
        initial_positions = {"left": (255,) * 6, "right": (255,) * 6}

        def refresh_positions(self, side: str) -> tuple[int, ...]:
            refreshes.append(side)
            return self.initial_positions[side]

        def set_positions(self, side: str, positions: tuple[int, ...]) -> None:
            commands.append((side, positions))
            if len(commands) == 6:
                stop.set()

    @contextmanager
    def transport(_config: RtmwHandsConfig):
        yield FakeTransport()

    run_rtmw_hands(
        hands_config(
            arm=True,
            calibration_confirmed=True,
            smoothing_time_constant_ms=0.0,
            max_raw_rate_per_second=1.0e9,
        ),
        transport,  # type: ignore[arg-type]
        stop_event=stop,
        stream_factory=lambda _config: FakeStream(),  # type: ignore[arg-type]
    )

    assert [side for side, _positions in commands] == [
        "left",
        "left",
        "right",
        "left",
        "left",
        "right",
    ]
    assert refreshes == ["left", "right", "right"]


class FakeRpcError(grpc.RpcError):
    def __init__(self, code: grpc.StatusCode, detail: str) -> None:
        self._code = code
        self._detail = detail

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return self._detail


class FakeCall:
    def __init__(self) -> None:
        self.messages: queue.Queue[object] = queue.Queue()
        self.cancelled = False

    def __iter__(self) -> FakeCall:
        return self

    def __next__(self) -> object:
        payload = self.messages.get()
        if payload is _END_STREAM:
            raise StopIteration
        if isinstance(payload, BaseException):
            raise payload
        return payload

    def send(self, payload: object) -> None:
        self.messages.put(payload)

    def cancel(self) -> None:
        self.cancelled = True
        self.messages.put(FakeRpcError(grpc.StatusCode.CANCELLED, "cancelled"))


class FakeReadyFuture:
    def result(self, timeout: float) -> None:
        del timeout

    def cancel(self) -> bool:
        return True


class FakeChannel:
    def __init__(self, stub: object) -> None:
        self.stub = stub
        self.ready_future = FakeReadyFuture()
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeStub:
    def __init__(self, info: object, call: FakeCall) -> None:
        self.info = info
        self.call = call

    def GetStreamInfo(self, _request: object, *, timeout: float) -> object:
        del timeout
        return self.info

    def GetStatus(self, _request: object, *, timeout: float) -> object:
        del timeout
        return SimpleNamespace(
            terminal_failure="",
            ready=True,
            source_running=True,
            inference_running=True,
        )

    def Subscribe(self, _request: object) -> FakeCall:
        return self.call


def test_armed_stream_enters_while_initial_rtmw_connection_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = RtmwHandsStream(hands_config(arm=True, calibration_confirmed=True))
    attempted = threading.Event()
    monkeypatch.setattr(
        rtmw_hands._rtmw,
        "import_pinned_rtmw_proto",
        lambda: (
            SimpleNamespace(SubscribeRequest=lambda: object()),
            SimpleNamespace(SkeletonServiceStub=lambda channel: channel),
        ),
    )

    def unavailable() -> None:
        attempted.set()
        raise grpc.FutureTimeoutError()

    monkeypatch.setattr(stream, "_connect_transport", unavailable)

    with stream:
        with pytest.raises(RtmwHandsDiscontinuity, match="not connected"):
            stream.wait_next(after_sequence=None, timeout_seconds=1.0)
        assert attempted.wait(timeout=1.0)


def test_stream_publishes_outage_before_reconnected_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = RtmwHandsStream(hands_config())
    first_call = FakeCall()
    second_call = FakeCall()
    first_channel = FakeChannel(FakeStub(stream_info(), first_call))
    second_channel = FakeChannel(
        FakeStub(
            stream_info(stream_instance_id="replacement", config_sha256="2" * 64),
            second_call,
        )
    )
    channels = iter((first_channel, second_channel))
    monkeypatch.setattr(
        rtmw_hands._rtmw,
        "import_pinned_rtmw_proto",
        lambda: (
            SimpleNamespace(SubscribeRequest=lambda: object()),
            SimpleNamespace(SkeletonServiceStub=lambda channel: channel.stub),
        ),
    )
    monkeypatch.setattr(
        grpc, "channel_ready_future", lambda channel: channel.ready_future
    )
    monkeypatch.setattr(stream, "_create_channel", lambda: next(channels))

    with stream:
        first_call.send(rtmw_frame(5_000_000_000))
        first = stream.wait_next(after_sequence=None, timeout_seconds=1.0)
        assert first is not None
        replacement = rtmw_frame(
            6_000_000_000,
            frame_id=1,
            stream_instance_id="replacement",
            track_id="replacement-track",
        )
        second_call.send(replacement)
        first_call.send(FakeRpcError(grpc.StatusCode.UNAVAILABLE, "restart"))

        with pytest.raises(RtmwHandsDiscontinuity, match="reconnecting"):
            stream.wait_next(after_sequence=first.sequence, timeout_seconds=1.0)
        resumed = stream.wait_next(after_sequence=None, timeout_seconds=1.0)

        assert resumed is not None
        assert resumed.track_id == "replacement-track"
        assert stream.stream_instance_id == "replacement"

    assert first_channel.closed
    assert second_channel.closed
