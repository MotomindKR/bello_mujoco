from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import math
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from bello_mujoco.hardware.pico_grpc import PicoGrpcConfig
from bello_mujoco.hardware.xrobot_hand_stream import HandLandmarks
from bello_mujoco.hardware import xrobot_hands
from bello_mujoco.hardware.xrobot_hands import (
    HandRawCalibration,
    XRobotHandCommandFilter,
    XRobotHandLandmarkMapper,
    XRobotHandTarget,
    XRobotHandsConfig,
    XRobotHandsDiscontinuity,
    XRobotHandsFrame,
    XRobotHandsUnavailable,
    run_xrobot_hands,
)


FIXTURE_JOINT_NAMES = tuple(
    name
    for side in ("left", "right")
    for name in (f"{side}_wrist",)
    + tuple(
        f"{side}_{finger}{joint}"
        for finger in ("thumb", "index", "middle", "ring", "pinky")
        for joint in (1, 2, 3)
    )
)
FIXTURE_TIP_NAMES = tuple(
    f"{side}_{finger}_tip"
    for side in ("left", "right")
    for finger in ("thumb", "index", "middle", "ring", "pinky")
)


def native_positions(side, joints, tips):
    positions = np.zeros((26, 3))
    positions[1] = joints[FIXTURE_JOINT_NAMES.index(f"{side}_wrist")]
    for finger, start in (
        ("thumb", 2),
        ("index", 7),
        ("middle", 12),
        ("ring", 17),
        ("pinky", 22),
    ):
        for joint in (1, 2, 3):
            positions[start + joint - 1] = joints[
                FIXTURE_JOINT_NAMES.index(f"{side}_{finger}{joint}")
            ]
        positions[start + 3] = tips[FIXTURE_TIP_NAMES.index(f"{side}_{finger}_tip")]
    return positions


def test_importing_hand_runner_does_not_load_policy_or_simulation_stacks() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import bello_mujoco.hardware.xrobot_hands; assert 'jax' not in sys.modules; assert 'mujoco' not in sys.modules; assert 'torch' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def hands_config(**overrides: object) -> XRobotHandsConfig:
    values: dict[str, object] = {
        "xrobot": PicoGrpcConfig(endpoint="127.0.0.1:50052"),
        "max_reference_age_ms": 1000,
        "smoothing_time_constant_ms": 500.0,
        "max_raw_rate_per_second": 255.0,
    }
    values.update(overrides)
    return XRobotHandsConfig(**values)


def neutral_landmarks() -> tuple[np.ndarray, np.ndarray]:
    joints = np.zeros((len(FIXTURE_JOINT_NAMES), 3), dtype=np.float64)
    tips = np.zeros((len(FIXTURE_TIP_NAMES), 3), dtype=np.float64)
    joint_index = {name: index for index, name in enumerate(FIXTURE_JOINT_NAMES)}
    tip_index = {name: index for index, name in enumerate(FIXTURE_TIP_NAMES)}
    lateral = {"index": 0.03, "middle": 0.01, "ring": -0.01, "pinky": -0.03}
    for side, mirror, offset in (("right", 1.0, 0.2), ("left", -1.0, -0.2)):
        wrist = np.asarray((offset, 0.0, 0.0))
        joints[joint_index[f"{side}_wrist"]] = wrist
        for finger, x in lateral.items():
            base = wrist + np.asarray((mirror * x, 0.05, 0.0))
            joints[joint_index[f"{side}_{finger}1"]] = base
            joints[joint_index[f"{side}_{finger}2"]] = base + (0.0, 0.03, 0.0)
            joints[joint_index[f"{side}_{finger}3"]] = base + (0.0, 0.055, 0.0)
            tips[tip_index[f"{side}_{finger}_tip"]] = base + (0.0, 0.08, 0.0)
        thumb_base = wrist + np.asarray((mirror * 0.035, 0.025, 0.0))
        radial = np.asarray((mirror, 0.0, 0.0))
        joints[joint_index[f"{side}_thumb1"]] = thumb_base
        joints[joint_index[f"{side}_thumb2"]] = thumb_base + 0.025 * radial
        joints[joint_index[f"{side}_thumb3"]] = thumb_base + 0.05 * radial
        tips[tip_index[f"{side}_thumb_tip"]] = thumb_base + 0.075 * radial
    return (joints, tips)


def curled_landmarks() -> tuple[np.ndarray, np.ndarray]:
    joints, tips = neutral_landmarks()
    joint_index = {name: index for index, name in enumerate(FIXTURE_JOINT_NAMES)}
    tip_index = {name: index for index, name in enumerate(FIXTURE_TIP_NAMES)}
    for side, mirror in (("right", 1.0), ("left", -1.0)):
        for finger in ("index", "middle", "ring", "pinky"):
            base = joints[joint_index[f"{side}_{finger}1"]]
            vectors = [
                0.03 * np.asarray((0.0, math.cos(angle), math.sin(angle)))
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
    return (joints, tips)


def map_landmarks(
    mapper: XRobotHandLandmarkMapper, side: str, joints: np.ndarray, tips: np.ndarray
) -> XRobotHandTarget:
    return mapper.map_side(side, native_positions(side, joints, tips))


def test_neutral_and_curled_hands_are_mirrored_consistently() -> None:
    mapper = XRobotHandLandmarkMapper(hands_config())
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


@pytest.mark.parametrize(
    "valid", [(True, False), (False, True), (True, True), (False, False)]
)
def test_stream_maps_each_hand_independently_without_body_tracking(monkeypatch, valid):
    joints, tips = neutral_landmarks()
    packet = SimpleNamespace(
        left=HandLandmarks(native_positions("left", joints, tips), 20)
        if valid[0]
        else None,
        right=HandLandmarks(native_positions("right", joints, tips), 20)
        if valid[1]
        else None,
        sequence=3,
        source_timestamp_ns=1,
        source_age_ms=20,
        received_at=time.monotonic(),
        track_id="calibrated",
    )
    stream = xrobot_hands.XRobotHandsStream(hands_config())
    stream._transport = SimpleNamespace(wait_next=lambda **kwargs: packet)
    if not any(valid):
        with pytest.raises(XRobotHandsUnavailable):
            stream.wait_next(after_sequence=None, timeout_seconds=1)
        return
    frame = stream.wait_next(after_sequence=None, timeout_seconds=1)
    assert (frame.left is not None, frame.right is not None) == valid
    assert frame.source_age_ms == 20 and frame.received_at == packet.received_at


def test_mapping_is_invariant_to_world_rigid_transform() -> None:
    mapper = XRobotHandLandmarkMapper(hands_config())
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
        expected.normalized_positions, abs=1e-12
    )


def test_thumb_adduction_is_palm_local_and_side_correct() -> None:
    mapper = XRobotHandLandmarkMapper(hands_config())
    joints, tips = neutral_landmarks()
    joint_index = {name: index for index, name in enumerate(FIXTURE_JOINT_NAMES)}
    tip_index = {name: index for index, name in enumerate(FIXTURE_TIP_NAMES)}
    for side, mirror in (("right", 1.0), ("left", -1.0)):
        thumb1 = joints[joint_index[f"{side}_thumb1"]]
        direction = np.asarray((mirror * 0.5, math.sqrt(3.0) / 2.0, 0.0))
        joints[joint_index[f"{side}_thumb2"]] = thumb1 + 0.025 * direction
        joints[joint_index[f"{side}_thumb3"]] = thumb1 + 0.05 * direction
        tips[tip_index[f"{side}_thumb_tip"]] = thumb1 + 0.075 * direction
    left = map_landmarks(mapper, "left", joints, tips)
    right = map_landmarks(mapper, "right", joints, tips)
    assert left.normalized_positions[1] == pytest.approx(1.0)
    assert right.normalized_positions[1] == pytest.approx(1.0)


def test_command_filter_limits_all_axes_with_inverted_calibration() -> None:
    config = hands_config(
        smoothing_time_constant_ms=0.0,
        max_raw_rate_per_second=100.0,
        left_calibration=HandRawCalibration(
            open_positions=(200, 0, 0, 0, 0, 0),
            closed_positions=(0, 255, 255, 255, 255, 255),
        ),
    )
    command_filter = XRobotHandCommandFilter(config)
    opened = XRobotHandTarget("left", (0.0,) * 6)
    closed = XRobotHandTarget("left", (1.0,) * 6)
    assert command_filter.update(opened, now=1.0) == (200, 0, 0, 0, 0, 0)
    assert command_filter.update(closed, now=1.1) == (190, 10, 10, 10, 10, 10)


def test_command_filter_applies_lpf_to_all_axes() -> None:
    config = hands_config(
        smoothing_time_constant_ms=500.0, max_raw_rate_per_second=10000.0
    )
    command_filter = XRobotHandCommandFilter(config)
    opened = XRobotHandTarget("right", (0.0,) * 6)
    closed = XRobotHandTarget("right", (1.0,) * 6)
    command_filter.update(opened, now=1.0)
    actual = command_filter.update(closed, now=1.5)
    expected = math.floor(255.0 * math.exp(-1.0) + 0.5)
    assert actual == (expected,) * 6


def test_command_filter_limits_first_recovered_step_on_all_axes() -> None:
    config = hands_config(
        control_rate_hz=10.0,
        smoothing_time_constant_ms=0.0,
        max_raw_rate_per_second=100.0,
    )
    command_filter = XRobotHandCommandFilter(config)
    command_filter.seed("right", (200,) * 6, now=1.0)
    command_filter.resume("right", now=20.0)
    actual = command_filter.update(XRobotHandTarget("right", (1.0,) * 6), now=20.0)
    assert actual == (190,) * 6


def test_command_filter_follows_partial_flexion_without_latching() -> None:
    config = hands_config(
        smoothing_time_constant_ms=0.0,
        max_raw_rate_per_second=10000.0,
    )
    command_filter = XRobotHandCommandFilter(config)
    first = command_filter.update(
        XRobotHandTarget("left", (0.7, 0.0, 0.7, 0.2, 0.8, 0.4)), now=1.0
    )
    halfway = command_filter.update(
        XRobotHandTarget("left", (0.5, 0.0, 0.5, 0.5, 0.5, 0.5)), now=1.1
    )
    opened = command_filter.update(
        XRobotHandTarget("left", (0.3, 0.0, 0.3, 0.3, 0.3, 0.3)), now=1.2
    )
    assert first == (77, 255, 77, 204, 51, 153)
    assert halfway == (128, 255, 128, 128, 128, 128)
    assert opened == (179, 255, 179, 179, 179, 179)


def test_command_filter_reseed_uses_measured_positions_for_every_axis() -> None:
    config = hands_config(smoothing_time_constant_ms=0.0, max_raw_rate_per_second=100.0)
    command_filter = XRobotHandCommandFilter(config)
    closed = XRobotHandTarget("right", (1.0,) * 6)
    halfway = XRobotHandTarget("right", (0.5,) * 6)
    assert command_filter.update(closed, now=1.0) == (0,) * 6
    command_filter.seed("right", (255, 200, 180, 50, 80, 90), now=1.1)
    command_filter.resume("right", now=1.2)
    assert command_filter.update(halfway, now=1.2) == (235, 180, 160, 70, 100, 110)


@pytest.mark.parametrize(
    "side",
    (
        "left",
        "right",
    ),
)
def test_proportional_targets_interpolate_each_calibrated_range(side) -> None:
    calibration = HandRawCalibration(
        open_positions=(200, 0, 250, 10, 90, 120),
        closed_positions=(0, 200, 50, 210, 190, 120),
    )
    command_filter = XRobotHandCommandFilter(
        hands_config(
            smoothing_time_constant_ms=0,
            max_raw_rate_per_second=10000,
            **{f"{side}_calibration": calibration},
        )
    )
    for now, amount, expected in (
        (1.0, 0.0, (200, 0, 250, 10, 90, 120)),
        (2.0, 0.25, (150, 50, 200, 60, 115, 120)),
        (3.0, 0.5, (100, 100, 150, 110, 140, 120)),
        (4.0, 0.75, (50, 150, 100, 160, 165, 120)),
        (5.0, 1.0, (0, 200, 50, 210, 190, 120)),
    ):
        assert (
            command_filter.update(XRobotHandTarget(side, (amount,) * 6), now=now)
            == expected
        )


def target_frame(*, received_at: float | None = None) -> XRobotHandsFrame:
    left = XRobotHandTarget("left", (0.1,) * 6)
    right = XRobotHandTarget("right", (0.2,) * 6)
    return XRobotHandsFrame(
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

        def wait_next(self, **_arguments: object) -> XRobotHandsFrame:
            stop.set()
            return target_frame()

    def forbidden_transport(_config: XRobotHandsConfig):
        raise AssertionError("unarmed validation must not touch CAN transport")

    run_xrobot_hands(
        hands_config(arm=False),
        forbidden_transport,
        stop_event=stop,
        stream_factory=lambda _config: FakeStream(),
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
        run_xrobot_hands(
            hands_config(max_duration_seconds=0.002),
            lambda _config: (_ for _ in ()).throw(
                AssertionError("unarmed preflight must not construct CAN transport")
            ),
            stream_factory=lambda _config: EmptyStream(),
        )


def test_armed_config_requires_explicit_calibration_confirmation() -> None:
    with pytest.raises(ValueError, match="calibration_confirmed=true"):
        hands_config(arm=True)


def test_armed_run_sends_nothing_before_first_smplx_target() -> None:
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
    def transport(_config: XRobotHandsConfig):
        yield FakeTransport()

    run_xrobot_hands(
        hands_config(arm=True, calibration_confirmed=True),
        transport,
        stop_event=stop,
        stream_factory=lambda _config: EmptyStream(),
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

        def wait_next(self, **_arguments: object) -> XRobotHandsFrame | None:
            nonlocal calls
            calls += 1
            if calls == 1:
                return frame
            if outage_pending:
                stop.set()
                raise XRobotHandsDiscontinuity("outage published during probe")
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
    def transport(_config: XRobotHandsConfig):
        yield FakeTransport()

    run_xrobot_hands(
        hands_config(arm=True, calibration_confirmed=True, right_enabled=False),
        transport,
        stop_event=stop,
        stream_factory=lambda _config: FakeStream(),
    )
    assert commands == []


@pytest.mark.parametrize(
    ("rate_limit", "expected"),
    [(1e9, (51, 128, 51, 51, 51, 51)), (100.0, (235,) * 6)],
)
def test_armed_run_follows_proportional_target_with_bounded_recovery(
    monkeypatch: pytest.MonkeyPatch, rate_limit: float, expected: tuple[int, ...]
) -> None:
    stop = threading.Event()
    commands: list[tuple[int, ...]] = []
    calls = 0
    now = 100.0
    monkeypatch.setattr(xrobot_hands, "time", SimpleNamespace(monotonic=lambda: now))
    target = replace(
        target_frame(received_at=now),
        left=XRobotHandTarget("left", (0.8, 0.5, 0.8, 0.8, 0.8, 0.8)),
        right=None,
    )

    class FakeStream:
        def __enter__(self) -> FakeStream:
            return self

        def __exit__(self, *_error: object) -> bool:
            return False

        def wait_next(self, *, timeout_seconds: float, **_arguments: object):
            nonlocal calls, now
            calls += 1
            if calls == 1:
                return target
            if calls == 2:
                now += 3.0  # Waiting for another observation must not allow a jump.
                return replace(target, sequence=2, received_at=now)
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
    def transport(_config: XRobotHandsConfig):
        yield FakeTransport()

    run_xrobot_hands(
        hands_config(
            arm=True,
            calibration_confirmed=True,
            right_enabled=False,
            smoothing_time_constant_ms=0.0,
            max_raw_rate_per_second=rate_limit,
        ),
        transport,
        stop_event=stop,
        stream_factory=lambda _config: FakeStream(),
    )
    assert commands == [expected]


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

    monkeypatch.setattr(xrobot_hands, "time", SimpleNamespace(monotonic=monotonic))

    class FakeStream:
        def __enter__(self) -> FakeStream:
            return self

        def __exit__(self, *_error: object) -> bool:
            return False

        def wait_next(self, **_arguments: object) -> XRobotHandsFrame | None:
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
    def transport(_config: XRobotHandsConfig):
        yield FakeTransport()

    run_xrobot_hands(
        hands_config(arm=True, calibration_confirmed=True, right_enabled=False),
        transport,
        stop_event=stop,
        stream_factory=lambda _config: FakeStream(),
    )
    assert commands == []


def test_smplx_outage_suspends_writes_and_recovery_refreshes_and_reseeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    class Clock:
        now = 0.0

        def monotonic(self) -> float:
            return self.now

        def sleep(self, seconds: float) -> None:
            self.now += seconds

    time = Clock()
    monkeypatch.setattr(xrobot_hands, "time", time)
    stop = threading.Event()
    commands: list[tuple[int, ...]] = []
    refreshes: list[tuple[int, ...]] = []
    calls = 0
    closed = replace(
        target_frame(),
        received_at=time.monotonic(),
        left=XRobotHandTarget("left", (1.0,) * 6),
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
                raise XRobotHandsUnavailable("test outage")
            if calls == 5:
                return replace(closed, sequence=3, received_at=time.monotonic())
            if calls == 6:
                time.sleep(timeout_seconds)
                return None
            if calls == 7:
                return replace(closed, sequence=4, received_at=time.monotonic())
            time.sleep(timeout_seconds)
            return None

    class FakeTransport:
        initial_positions = {"left": (255,) * 6}

        def __init__(self) -> None:
            self.samples = iter(
                ((255, 200, 255, 255, 255, 255), (255, 50, 255, 255, 255, 255))
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
    def transport(_config: XRobotHandsConfig):
        yield FakeTransport()

    run_xrobot_hands(
        hands_config(
            arm=True,
            calibration_confirmed=True,
            right_enabled=False,
            smoothing_time_constant_ms=0.0,
            max_raw_rate_per_second=100.0,
        ),
        transport,
        stop_event=stop,
        stream_factory=lambda _config: FakeStream(),
    )
    assert refreshes == [(255, 200, 255, 255, 255, 255), (255, 50, 255, 255, 255, 255)]
    assert commands == [(235, 180, 235, 235, 235, 235), (235, 30, 235, 235, 235, 235)]


def test_missing_smplx_frames_send_no_positions() -> None:
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
            raise AssertionError("missing SMPL-X data must not probe a hand")

    @contextmanager
    def transport(_config: XRobotHandsConfig):
        yield FakeTransport()

    run_xrobot_hands(
        hands_config(arm=True, calibration_confirmed=True, right_enabled=False),
        transport,
        stop_event=stop,
        stream_factory=lambda _config: EmptyStream(),
    )
    assert commands == []


def test_stale_smplx_target_sends_no_positions() -> None:
    stop = threading.Event()
    commands: list[tuple[int, ...]] = []
    stale = replace(
        target_frame(received_at=time.monotonic() - 2.0),
        left=XRobotHandTarget("left", (1.0,) * 6),
        right=None,
    )

    class FakeStream:
        def __enter__(self) -> FakeStream:
            return self

        def __exit__(self, *_error: object) -> bool:
            return False

        def wait_next(self, **_arguments: object) -> XRobotHandsFrame:
            stop.set()
            return stale

    class FakeTransport:
        initial_positions = {"left": (255,) * 6}

        def set_positions(self, _side: str, positions: tuple[int, ...]) -> None:
            commands.append(positions)

        def refresh_positions(self, _side: str) -> tuple[int, ...]:
            raise AssertionError("a stale target must not probe a hand")

    @contextmanager
    def transport(_config: XRobotHandsConfig):
        yield FakeTransport()

    run_xrobot_hands(
        hands_config(arm=True, calibration_confirmed=True, right_enabled=False),
        transport,
        stop_event=stop,
        stream_factory=lambda _config: FakeStream(),
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

    def frame(sequence: int, *, right_available: bool = True) -> XRobotHandsFrame:
        result = replace(
            target_frame(received_at=now),
            sequence=sequence,
            left=XRobotHandTarget("left", (1.0,) * 6),
        )
        if right_available:
            return result
        return replace(result, right=None, right_unavailable_reason="invalid tracking")

    monkeypatch.setattr(xrobot_hands, "time", SimpleNamespace(monotonic=monotonic))

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
    def transport(_config: XRobotHandsConfig):
        yield FakeTransport()

    run_xrobot_hands(
        hands_config(
            arm=True,
            calibration_confirmed=True,
            smoothing_time_constant_ms=0.0,
            max_raw_rate_per_second=1000000000.0,
        ),
        transport,
        stop_event=stop,
        stream_factory=lambda _config: FakeStream(),
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
