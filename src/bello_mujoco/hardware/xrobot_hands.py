"""XRoboToolkit hand-landmark mapping and transport-independent hand control."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
import logging
import math
import threading
import time
from typing import Callable, ContextManager, Literal, Protocol, TypeAlias

import numpy as np

from bello_mujoco.hardware.pico_grpc import PicoGrpcConfig
from bello_mujoco.hardware.xrobot_hand_stream import (
    HAND_NAMES,
    FINGER_JOINTS,
    FINGER_TIPS,
    XRobotHandsGrpcStream,
)
from bello_mujoco.hardware.reference import ReferenceUnavailable, ReferenceDiscontinuity


HandSide: TypeAlias = Literal["left", "right"]
NormalizedHandPositions: TypeAlias = tuple[float, float, float, float, float, float]
RawHandPositions: TypeAlias = tuple[int, int, int, int, int, int]

HAND_POSITION_ORDER = (
    "thumb_flex",
    "thumb_abduction",
    "index",
    "middle",
    "ring",
    "pinky",
)

_SIDES: tuple[HandSide, HandSide] = ("left", "right")
_FINGERS = ("index", "middle", "ring", "pinky", "thumb")
_LOGGER = logging.getLogger(__name__)


class XRobotHandsUnavailable(RuntimeError):
    """No hand target may be sent until another accepted XRoboToolkit frame arrives."""


class XRobotHandsDiscontinuity(XRobotHandsUnavailable):
    """The XRoboToolkit tracking epoch changed and all smoothing state must be reset."""


class _HandGeometryUnavailable(ValueError):
    """One side lacks the observations needed for a safe command."""


@dataclass(frozen=True, slots=True)
class HandRawCalibration:
    """Raw controller endpoints for open and closed human-hand targets."""

    open_positions: RawHandPositions = (255, 255, 255, 255, 255, 255)
    closed_positions: RawHandPositions = (0, 0, 0, 0, 0, 0)

    def __post_init__(self) -> None:
        for name, values in (
            ("open_positions", self.open_positions),
            ("closed_positions", self.closed_positions),
        ):
            if not isinstance(values, tuple) or len(values) != len(HAND_POSITION_ORDER):
                raise ValueError(f"{name} must contain six raw hand positions")
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 255
                for value in values
            ):
                raise ValueError(f"{name} values must be integers in [0, 255]")

    def raw_target(self, normalized: NormalizedHandPositions) -> np.ndarray:
        values = np.asarray(normalized, dtype=np.float64)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError("normalized hand target must contain six finite values")
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("normalized hand target values must be in [0, 1]")
        opened = np.asarray(self.open_positions, dtype=np.float64)
        closed = np.asarray(self.closed_positions, dtype=np.float64)
        return opened + values * (closed - opened)


@dataclass(frozen=True, slots=True)
class HandGeometryConfig:
    """Scale-free anatomical ranges used to interpret landmark geometry."""

    finger_full_flex_degrees: float = 180.0
    thumb_full_flex_degrees: float = 120.0
    thumb_adduction_range_degrees: float = 60.0
    minimum_segment_length_m: float = 0.004

    def __post_init__(self) -> None:
        for name in (
            "finger_full_flex_degrees",
            "thumb_full_flex_degrees",
            "thumb_adduction_range_degrees",
            "minimum_segment_length_m",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        for name in (
            "finger_full_flex_degrees",
            "thumb_full_flex_degrees",
            "thumb_adduction_range_degrees",
        ):
            if getattr(self, name) > 360.0:
                raise ValueError(f"{name} must not exceed 360 degrees")


@dataclass(frozen=True, slots=True)
class XRobotHandsConfig:
    """Connection, observation, filtering, and arming policy for hand control."""

    xrobot: PicoGrpcConfig
    max_reference_age_ms: int = 500
    control_rate_hz: float = 5.0
    smoothing_time_constant_ms: float = 100.0
    max_raw_rate_per_second: float = 510.0
    max_duration_seconds: float | None = None
    left_enabled: bool = True
    right_enabled: bool = True
    arm: bool = False
    calibration_confirmed: bool = False
    left_calibration: HandRawCalibration = field(default_factory=HandRawCalibration)
    right_calibration: HandRawCalibration = field(default_factory=HandRawCalibration)
    geometry: HandGeometryConfig = field(default_factory=HandGeometryConfig)

    def __post_init__(self) -> None:
        for name in ("control_rate_hz", "max_raw_rate_per_second"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if (
            not math.isfinite(self.smoothing_time_constant_ms)
            or self.smoothing_time_constant_ms < 0.0
        ):
            raise ValueError(
                "XRoboToolkit hands smoothing_time_constant_ms must be finite and nonnegative"
            )
        if self.max_duration_seconds is not None and (
            not math.isfinite(self.max_duration_seconds)
            or self.max_duration_seconds <= 0.0
        ):
            raise ValueError(
                "XRoboToolkit hands max_duration_seconds must be positive and finite"
            )
        if isinstance(self.max_reference_age_ms, bool) or not isinstance(
            self.max_reference_age_ms, int
        ):
            raise TypeError(
                "XRoboToolkit hands max_reference_age_ms must be an integer"
            )
        if self.max_reference_age_ms <= 0:
            raise ValueError("XRoboToolkit hands max_reference_age_ms must be positive")
        if not self.left_enabled and not self.right_enabled:
            raise ValueError("at least one XRoboToolkit hand must be enabled")
        if not isinstance(self.arm, bool):
            raise TypeError("XRoboToolkit hands arm must be a boolean")
        if not isinstance(self.calibration_confirmed, bool):
            raise TypeError(
                "XRoboToolkit hands calibration_confirmed must be a boolean"
            )
        if self.arm and not self.calibration_confirmed:
            raise ValueError(
                "armed hand control requires calibration_confirmed=true after "
                "checking all six open/closed endpoints"
            )

    def calibration(self, side: HandSide) -> HandRawCalibration:
        return self.left_calibration if side == "left" else self.right_calibration

    def enabled(self, side: HandSide) -> bool:
        return self.left_enabled if side == "left" else self.right_enabled


@dataclass(frozen=True, slots=True)
class XRobotHandTarget:
    side: HandSide
    normalized_positions: NormalizedHandPositions


@dataclass(frozen=True, slots=True)
class XRobotHandsFrame:
    sequence: int
    source_timestamp_ns: int
    source_age_ms: int
    received_at: float
    track_id: str
    left: XRobotHandTarget | None
    right: XRobotHandTarget | None
    left_unavailable_reason: str | None = None
    right_unavailable_reason: str | None = None

    def target(self, side: HandSide) -> XRobotHandTarget | None:
        return self.left if side == "left" else self.right

    def unavailable_reason(self, side: HandSide) -> str | None:
        return (
            self.left_unavailable_reason
            if side == "left"
            else self.right_unavailable_reason
        )


class HandPositionTransport(Protocol):
    """Minimal interface supplied by the classic-CAN transport layer."""

    @property
    def initial_positions(self) -> Mapping[HandSide, RawHandPositions]: ...

    def set_positions(self, side: HandSide, positions: RawHandPositions) -> None: ...

    def refresh_positions(self, side: HandSide) -> RawHandPositions: ...


HandTransportFactory: TypeAlias = Callable[
    [XRobotHandsConfig], ContextManager[HandPositionTransport]
]


class XRobotHandLandmarkMapper:
    """Convert global XRoboToolkit hand landmarks into palm-local actuator targets."""

    def __init__(self, config: XRobotHandsConfig) -> None:
        self.config = config
        self._joint_index = {name: index for index, name in enumerate(HAND_NAMES)}

    def map_side(self, side: HandSide, joint_positions: np.ndarray) -> XRobotHandTarget:
        if joint_positions.shape != (26, 3) or not np.isfinite(joint_positions).all():
            raise _HandGeometryUnavailable(
                "hand landmarks must contain 26 finite XYZ positions"
            )
        positions = {f"{side}_wrist": joint_positions[self._joint_index["wrist"]]}
        for finger, names in FINGER_JOINTS.items():
            for number, name in enumerate(names, 1):
                positions[f"{side}_{finger}{number}"] = joint_positions[
                    self._joint_index[name]
                ]
        tips = {
            f"{side}_{finger}_tip": joint_positions[self._joint_index[name]]
            for finger, name in FINGER_TIPS.items()
        }
        radial, distal, palmar = self._palm_basis(side, positions)
        finger_flex = {
            finger: self._finger_flex(
                side,
                finger,
                positions,
                tips[f"{side}_{finger}_tip"],
                distal,
                palmar,
            )
            for finger in ("index", "middle", "ring", "pinky")
        }
        thumb_flex, thumb_adduction = self._thumb_targets(
            side,
            positions,
            tips[f"{side}_thumb_tip"],
            radial,
            distal,
            palmar,
        )
        normalized = (
            thumb_flex,
            thumb_adduction,
            finger_flex["index"],
            finger_flex["middle"],
            finger_flex["ring"],
            finger_flex["pinky"],
        )
        return XRobotHandTarget(
            side=side,
            normalized_positions=normalized,
        )

    def _palm_basis(
        self,
        side: HandSide,
        positions: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        wrist = positions[f"{side}_wrist"]
        index = positions[f"{side}_index1"]
        pinky = positions[f"{side}_pinky1"]
        bases = np.stack([positions[f"{side}_{finger}1"] for finger in _FINGERS[:-1]])
        radial = self._unit(index - pinky, "index-to-pinky palm width")
        distal_vector = np.mean(bases, axis=0) - wrist
        distal_vector = distal_vector - np.dot(distal_vector, radial) * radial
        distal = self._unit(distal_vector, "wrist-to-knuckle palm length")
        handedness = -1.0 if side == "left" else 1.0
        palmar = self._unit(handedness * np.cross(radial, distal), "palm normal")
        return radial, distal, palmar

    def _finger_flex(
        self,
        side: HandSide,
        finger: str,
        positions: dict[str, np.ndarray],
        tip: np.ndarray,
        distal: np.ndarray,
        palmar: np.ndarray,
    ) -> float:
        points = [positions[f"{side}_{finger}{joint}"] for joint in (1, 2, 3)]
        points.append(tip)
        bone_angles = [
            self._plane_angle(points[index + 1] - points[index], distal, palmar)
            for index in range(3)
        ]
        total_flex = max(0.0, bone_angles[0])
        total_flex += max(0.0, bone_angles[1] - bone_angles[0])
        total_flex += max(0.0, bone_angles[2] - bone_angles[1])
        full_flex = math.radians(self.config.geometry.finger_full_flex_degrees)
        return float(np.clip(total_flex / full_flex, 0.0, 1.0))

    def _thumb_targets(
        self,
        side: HandSide,
        positions: dict[str, np.ndarray],
        tip: np.ndarray,
        radial: np.ndarray,
        distal: np.ndarray,
        palmar: np.ndarray,
    ) -> tuple[float, float]:
        points = [positions[f"{side}_thumb{joint}"] for joint in (1, 2, 3)]
        points.append(tip)
        flex_angles = [
            self._plane_angle(points[index + 1] - points[index], radial, palmar)
            for index in range(3)
        ]
        total_flex = max(0.0, flex_angles[0])
        total_flex += max(0.0, flex_angles[1] - flex_angles[0])
        total_flex += max(0.0, flex_angles[2] - flex_angles[1])
        thumb_flex = np.clip(
            total_flex / math.radians(self.config.geometry.thumb_full_flex_degrees),
            0.0,
            1.0,
        )

        proximal = points[1] - points[0]
        palm_projection = proximal - np.dot(proximal, palmar) * palmar
        projected = self._unit(palm_projection, "thumb palm-plane direction")
        adduction_angle = math.atan2(
            float(np.dot(projected, distal)), float(np.dot(projected, radial))
        )
        thumb_adduction = np.clip(
            max(0.0, adduction_angle)
            / math.radians(self.config.geometry.thumb_adduction_range_degrees),
            0.0,
            1.0,
        )
        return float(thumb_flex), float(thumb_adduction)

    def _plane_angle(
        self,
        vector: np.ndarray,
        forward: np.ndarray,
        bend: np.ndarray,
    ) -> float:
        if (
            float(np.linalg.norm(vector))
            < self.config.geometry.minimum_segment_length_m
        ):
            raise _HandGeometryUnavailable("finger segment is too short")
        forward_component = float(np.dot(vector, forward))
        bend_component = float(np.dot(vector, bend))
        if math.hypot(forward_component, bend_component) < (
            self.config.geometry.minimum_segment_length_m
        ):
            raise _HandGeometryUnavailable(
                "finger segment has no usable flexion projection"
            )
        return math.atan2(bend_component, forward_component)

    def _unit(self, vector: np.ndarray, label: str) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if (
            not math.isfinite(norm)
            or norm < self.config.geometry.minimum_segment_length_m
        ):
            raise _HandGeometryUnavailable(f"{label} is degenerate")
        return vector / norm


class XRobotHandCommandFilter:
    """Proportional control with smoothing and slew limits on all six axes."""

    def __init__(self, config: XRobotHandsConfig) -> None:
        self.config = config
        self._state: dict[HandSide, np.ndarray] = {}
        self._updated_at: dict[HandSide, float] = {}

    def seed(
        self,
        side: HandSide,
        positions: Sequence[int],
        *,
        now: float,
    ) -> None:
        """Start smoothing at a position read from the physical hand."""

        if not math.isfinite(now):
            raise ValueError("hand filter timestamp must be finite")
        if len(positions) != 6:
            raise ValueError("initial hand position must contain six values")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 255
            for value in positions
        ):
            raise ValueError("initial hand positions must be integers in [0, 255]")
        measured = np.asarray(positions, dtype=np.float64)
        self._state[side] = measured
        self._updated_at[side] = now

    def resume(self, side: HandSide, *, now: float) -> None:
        """Resume from the last command with one control period of authority."""

        if not math.isfinite(now):
            raise ValueError("hand filter timestamp must be finite")
        if side not in self._state:
            raise RuntimeError(f"hand filter side {side!r} has not been seeded")
        self._updated_at[side] = now - 1.0 / self.config.control_rate_hz

    def update(
        self,
        target: XRobotHandTarget,
        *,
        now: float,
    ) -> RawHandPositions:
        if not math.isfinite(now):
            raise ValueError("hand filter timestamp must be finite")
        side = target.side
        desired = self.config.calibration(side).raw_target(target.normalized_positions)
        previous = self._state.get(side)
        previous_at = self._updated_at.get(side)
        if previous is None or previous_at is None:
            filtered = desired
        else:
            dt = now - previous_at
            if not math.isfinite(dt) or dt <= 0.0:
                raise ValueError("hand filter timestamps must advance")
            tau = self.config.smoothing_time_constant_ms / 1_000.0
            alpha = 1.0 if tau == 0.0 else -math.expm1(-dt / tau)
            low_passed = previous + alpha * (desired - previous)
            maximum_delta = self.config.max_raw_rate_per_second * dt
            filtered = previous + np.clip(
                low_passed - previous, -maximum_delta, maximum_delta
            )
        filtered = np.clip(filtered, 0.0, 255.0)
        self._state[side] = filtered
        self._updated_at[side] = now
        rounded = np.floor(filtered + 0.5).astype(np.uint8)
        return tuple(int(value) for value in rounded)  # type: ignore[return-value]


class XRobotHandsStream:
    """Map each valid optical hand independently of body tracking."""

    def __init__(self, config: XRobotHandsConfig):
        self.config = config
        self._mapper = XRobotHandLandmarkMapper(config)
        self._transport = XRobotHandsGrpcStream(config.xrobot)

    def __enter__(self):
        self._transport.__enter__()
        return self

    def __exit__(self, *exception):
        return self._transport.__exit__(*exception)

    def wait_next(self, *, after_sequence, timeout_seconds):
        try:
            frame = self._transport.wait_next(
                after_sequence=after_sequence, timeout_seconds=timeout_seconds
            )
        except ReferenceDiscontinuity as error:
            raise XRobotHandsDiscontinuity(str(error)) from error
        except ReferenceUnavailable as error:
            raise XRobotHandsUnavailable(str(error)) from error
        if frame is None:
            return None
        targets = {}
        reasons = {}
        for side in _SIDES:
            targets[side] = None
            reasons[side] = None
            if not self.config.enabled(side):
                continue
            hand = getattr(frame, side)
            if hand is None:
                reasons[side] = "Hand tracking is invalid or stale"
                continue
            try:
                targets[side] = self._mapper.map_side(
                    side,
                    hand.positions,
                )
            except _HandGeometryUnavailable as error:
                reasons[side] = str(error)
        if not any(targets.values()):
            raise XRobotHandsUnavailable("No enabled hand has a valid target")
        return XRobotHandsFrame(
            frame.sequence,
            frame.source_timestamp_ns,
            int(frame.source_age_ms),
            frame.received_at,
            frame.track_id,
            targets["left"],
            targets["right"],
            reasons["left"],
            reasons["right"],
        )


def run_xrobot_hands(
    config: XRobotHandsConfig,
    transport_factory: HandTransportFactory,
    *,
    stop_event: threading.Event | None = None,
    stream_factory: Callable[[XRobotHandsConfig], ContextManager[XRobotHandsStream]] = (
        XRobotHandsStream
    ),
) -> None:
    """Follow the latest fresh XRoboToolkit hand pose until stopped.

    Missing, stale, or disconnected XRoboToolkit input suspends position writes for the
    affected side. All six axes follow proportional targets with the configured
    low-pass and slew limits.
    """

    stop = stop_event if stop_event is not None else threading.Event()
    position_filter = XRobotHandCommandFilter(config)
    transport_context: ContextManager[HandPositionTransport | None]
    if config.arm:
        transport_context = transport_factory(config)
    else:
        transport_context = nullcontext(None)
    period = 1.0 / config.control_rate_hz
    latest: XRobotHandsFrame | None = None
    last_sequence: int | None = None
    active_track_id: str | None = None
    side_active = {side: False for side in _SIDES}
    side_primed_after_sequence: dict[HandSide, int | None] = {
        side: None for side in _SIDES
    }

    def suspend_all_sides() -> None:
        for side in _SIDES:
            side_active[side] = False
            side_primed_after_sequence[side] = None

    next_command_at = time.monotonic()
    deadline = (
        None
        if config.max_duration_seconds is None
        else next_command_at + config.max_duration_seconds
    )
    stale = True

    with stream_factory(config) as stream, transport_context as transport:
        if transport is not None:
            initial_positions = transport.initial_positions
            seeded_at = time.monotonic()
            for side in _SIDES:
                if config.enabled(side):
                    if side not in initial_positions:
                        raise RuntimeError(
                            f"hand transport did not probe enabled side {side!r}"
                        )
                    position_filter.seed(side, initial_positions[side], now=seeded_at)
        while not stop.is_set():
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                if not config.arm:
                    raise TimeoutError(
                        "XRoboToolkit hand preflight duration expired before an acceptable "
                        "hand frame arrived"
                    )
                return
            until_command = max(0.0, next_command_at - now)
            timeout = max(0.001, min(period, until_command or period))
            try:
                frame = stream.wait_next(
                    after_sequence=last_sequence,
                    timeout_seconds=timeout,
                )
            except XRobotHandsUnavailable as error:
                if not stale:
                    _LOGGER.warning("XRoboToolkit hand target unavailable: %s", error)
                latest = None
                last_sequence = None
                active_track_id = None
                suspend_all_sides()
                stale = True
            else:
                if frame is not None:
                    if (
                        active_track_id is not None
                        and frame.track_id != active_track_id
                    ):
                        suspend_all_sides()
                    latest = frame
                    last_sequence = frame.sequence
                    active_track_id = frame.track_id
                    for side in _SIDES:
                        if config.enabled(side) and frame.target(side) is None:
                            side_active[side] = False
                            side_primed_after_sequence[side] = None
                    stale = False
                    if not config.arm:
                        return

            now = time.monotonic()
            if now < next_command_at:
                continue
            skipped = max(0, math.floor((now - next_command_at) / period))
            next_command_at += (skipped + 1) * period
            if latest is None:
                continue
            age_ms = latest.source_age_ms + (now - latest.received_at) * 1_000.0
            if age_ms > config.max_reference_age_ms:
                if not stale:
                    _LOGGER.warning(
                        "XRoboToolkit hand target became stale: %.1f ms > %d ms",
                        age_ms,
                        config.max_reference_age_ms,
                    )
                latest = None
                active_track_id = None
                suspend_all_sides()
                stale = True
                continue
            for side in _SIDES:
                if not config.enabled(side):
                    continue
                now = time.monotonic()
                age_ms = latest.source_age_ms + (now - latest.received_at) * 1_000.0
                if age_ms > config.max_reference_age_ms:
                    latest = None
                    active_track_id = None
                    suspend_all_sides()
                    stale = True
                    break
                target = latest.target(side)
                if target is None:
                    side_active[side] = False
                    side_primed_after_sequence[side] = None
                    continue
                if not side_active[side]:
                    primed_after = side_primed_after_sequence[side]
                    if primed_after is not None:
                        if latest.sequence <= primed_after:
                            continue
                        position_filter.resume(side, now=now)
                        side_active[side] = True
                        side_primed_after_sequence[side] = None
                    else:
                        if (
                            transport is None
                        ):  # pragma: no cover - arm=false returns above
                            raise RuntimeError(
                                "armed hand control has no CAN transport"
                            )
                        measured = transport.refresh_positions(side)
                        now = time.monotonic()
                        age_ms = (
                            latest.source_age_ms + (now - latest.received_at) * 1_000.0
                        )
                        if age_ms > config.max_reference_age_ms:
                            latest = None
                            active_track_id = None
                            suspend_all_sides()
                            stale = True
                            break
                        position_filter.seed(side, measured, now=now)
                        side_primed_after_sequence[side] = latest.sequence
                        break
                positions = position_filter.update(target, now=now)
                now = time.monotonic()
                age_ms = latest.source_age_ms + (now - latest.received_at) * 1_000.0
                if age_ms > config.max_reference_age_ms:
                    latest = None
                    active_track_id = None
                    suspend_all_sides()
                    stale = True
                    break
                if transport is not None:
                    transport.set_positions(side, positions)
                now = time.monotonic()

    if not config.arm:
        raise RuntimeError(
            "XRoboToolkit hand preflight stopped before an acceptable frame"
        )


__all__ = (
    "HAND_POSITION_ORDER",
    "HandGeometryConfig",
    "HandPositionTransport",
    "HandRawCalibration",
    "HandSide",
    "RawHandPositions",
    "XRobotHandCommandFilter",
    "XRobotHandLandmarkMapper",
    "XRobotHandTarget",
    "XRobotHandsConfig",
    "XRobotHandsDiscontinuity",
    "XRobotHandsFrame",
    "XRobotHandsStream",
    "XRobotHandsUnavailable",
    "run_xrobot_hands",
)
