"""RTMW hand-landmark mapping and transport-independent hand control."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
import threading
import time
from typing import Any, Callable, ContextManager, Literal, Protocol, TypeAlias

import grpc
from google.protobuf import empty_pb2
import numpy as np

from bello_mujoco.hardware import rtmw_contract as _rtmw


HandSide: TypeAlias = Literal["left", "right"]
NormalizedHandPositions: TypeAlias = tuple[float, float, float, float, float, float]
RawHandPositions: TypeAlias = tuple[int, int, int, int, int, int]

RTMW_FINGER_TIP_NAMES = _rtmw.RTMW_FINGER_TIP_NAMES
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
_FLEXION_AXES = (0, 2, 3, 4, 5)
_RETRYABLE_GRPC_CODES = frozenset(
    (
        grpc.StatusCode.CANCELLED,
        grpc.StatusCode.DEADLINE_EXCEEDED,
        grpc.StatusCode.UNAVAILABLE,
    )
)
_RECONNECT_INITIAL_BACKOFF_SECONDS = 0.1
_RECONNECT_MAX_BACKOFF_SECONDS = 1.0
_LOGGER = logging.getLogger(__name__)


class RtmwHandsUnavailable(RuntimeError):
    """No hand target may be sent until another accepted RTMW frame arrives."""


class RtmwHandsDiscontinuity(RtmwHandsUnavailable):
    """The RTMW tracking epoch changed and all smoothing state must be reset."""


class _RtmwServiceUnavailable(RuntimeError):
    """The RTMW producer is reachable but not ready to stream."""


class _RtmwStreamStopped(RuntimeError):
    """Internal signal used to interrupt transport setup during cleanup."""


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
class RtmwHandsConfig:
    """Connection, observation, filtering, and arming policy for hand control."""

    endpoint: str = "127.0.0.1:50055"
    tls_root_certificate_path: Path | None = None
    tls_server_name_override: str | None = None
    connect_timeout_seconds: float = 5.0
    rpc_timeout_seconds: float = 2.0
    human_height_m: float = 1.75
    min_confidence: float = 0.05
    max_pipeline_latency_ms: float = 250.0
    transport_latency_budget_ms: float = 10.0
    max_reference_age_ms: int = 1_000
    max_source_interval_seconds: float = 0.70
    control_rate_hz: float = 5.0
    smoothing_time_constant_ms: float = 100.0
    max_raw_rate_per_second: float = 510.0
    flexion_open_threshold: float = 0.35
    flexion_close_threshold: float = 0.65
    max_duration_seconds: float | None = None
    left_enabled: bool = True
    right_enabled: bool = True
    arm: bool = False
    calibration_confirmed: bool = False
    left_calibration: HandRawCalibration = field(default_factory=HandRawCalibration)
    right_calibration: HandRawCalibration = field(default_factory=HandRawCalibration)
    geometry: HandGeometryConfig = field(default_factory=HandGeometryConfig)

    def __post_init__(self) -> None:
        if not self.endpoint or "://" in self.endpoint:
            raise ValueError("RTMW endpoint must be a gRPC host:port target")
        for name in (
            "connect_timeout_seconds",
            "rpc_timeout_seconds",
            "human_height_m",
            "max_pipeline_latency_ms",
            "max_source_interval_seconds",
            "control_rate_hz",
            "max_raw_rate_per_second",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"RTMW hands {name} must be positive and finite")
        if (
            not math.isfinite(self.smoothing_time_constant_ms)
            or self.smoothing_time_constant_ms < 0.0
        ):
            raise ValueError(
                "RTMW hands smoothing_time_constant_ms must be finite and nonnegative"
            )
        for name in ("flexion_open_threshold", "flexion_close_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"RTMW hands {name} must be finite and in [0, 1]")
        if self.flexion_open_threshold >= self.flexion_close_threshold:
            raise ValueError(
                "RTMW hands flexion_open_threshold must be less than "
                "flexion_close_threshold"
            )
        if self.max_duration_seconds is not None and (
            not math.isfinite(self.max_duration_seconds)
            or self.max_duration_seconds <= 0.0
        ):
            raise ValueError(
                "RTMW hands max_duration_seconds must be positive and finite"
            )
        if (
            not math.isfinite(self.min_confidence)
            or not 0.0 <= self.min_confidence <= 1.0
        ):
            raise ValueError("RTMW hands min_confidence must be finite and in [0, 1]")
        if (
            not math.isfinite(self.transport_latency_budget_ms)
            or self.transport_latency_budget_ms < 0.0
        ):
            raise ValueError(
                "RTMW hands transport_latency_budget_ms must be finite and nonnegative"
            )
        if isinstance(self.max_reference_age_ms, bool) or not isinstance(
            self.max_reference_age_ms, int
        ):
            raise TypeError("RTMW hands max_reference_age_ms must be an integer")
        if self.max_reference_age_ms <= 0:
            raise ValueError("RTMW hands max_reference_age_ms must be positive")
        if not self.left_enabled and not self.right_enabled:
            raise ValueError("at least one RTMW hand must be enabled")
        if not isinstance(self.arm, bool):
            raise TypeError("RTMW hands arm must be a boolean")
        if not isinstance(self.calibration_confirmed, bool):
            raise TypeError("RTMW hands calibration_confirmed must be a boolean")
        if self.arm and not self.calibration_confirmed:
            raise ValueError(
                "armed hand control requires calibration_confirmed=true after "
                "checking all six open/closed endpoints"
            )
        if self.tls_server_name_override is not None and (
            self.tls_root_certificate_path is None
        ):
            raise ValueError("RTMW TLS server-name override requires a CA file")

    def calibration(self, side: HandSide) -> HandRawCalibration:
        return self.left_calibration if side == "left" else self.right_calibration

    def enabled(self, side: HandSide) -> bool:
        return self.left_enabled if side == "left" else self.right_enabled


@dataclass(frozen=True, slots=True)
class RtmwHandTarget:
    side: HandSide
    normalized_positions: NormalizedHandPositions
    minimum_confidence: float


@dataclass(frozen=True, slots=True)
class RtmwHandsFrame:
    sequence: int
    source_timestamp_ns: int
    source_age_ms: int
    received_at: float
    track_id: str
    left: RtmwHandTarget | None
    right: RtmwHandTarget | None
    left_unavailable_reason: str | None = None
    right_unavailable_reason: str | None = None

    def target(self, side: HandSide) -> RtmwHandTarget | None:
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
    [RtmwHandsConfig], ContextManager[HandPositionTransport]
]


class RtmwHandLandmarkMapper:
    """Convert global SMPL-X hand landmarks into palm-local actuator targets."""

    def __init__(self, config: RtmwHandsConfig) -> None:
        self.config = config
        self._joint_index = {
            name: index for index, name in enumerate(_rtmw.RTMW_SMPLX52_NAMES)
        }
        self._tip_index = {
            name: index for index, name in enumerate(RTMW_FINGER_TIP_NAMES)
        }

    def map_side(
        self,
        side: HandSide,
        joint_positions: np.ndarray,
        joint_confidence: np.ndarray,
        joint_valid: np.ndarray,
        tip_positions: np.ndarray,
        tip_confidence: np.ndarray,
    ) -> RtmwHandTarget:
        joint_names = (f"{side}_wrist",) + tuple(
            f"{side}_{finger}{joint}" for finger in _FINGERS for joint in (1, 2, 3)
        )
        tip_names = tuple(f"{side}_{finger}_tip" for finger in _FINGERS)
        joint_indices = np.asarray([self._joint_index[name] for name in joint_names])
        tip_indices = np.asarray([self._tip_index[name] for name in tip_names])
        side_joint_positions = joint_positions[joint_indices]
        side_tip_positions = tip_positions[tip_indices]
        if not np.all(joint_valid[joint_indices]):
            bad = int(np.flatnonzero(~joint_valid[joint_indices])[0])
            raise _HandGeometryUnavailable(f"joint {joint_names[bad]!r} is invalid")
        if not np.all(np.isfinite(side_joint_positions)):
            raise _HandGeometryUnavailable("joint positions contain non-finite values")
        if not np.all(np.isfinite(side_tip_positions)):
            raise _HandGeometryUnavailable(
                "finger-tip positions contain non-finite values"
            )
        confidences = np.concatenate(
            (joint_confidence[joint_indices], tip_confidence[tip_indices])
        )
        minimum_confidence = float(np.min(confidences))
        if minimum_confidence < self.config.min_confidence:
            raise _HandGeometryUnavailable(
                f"landmark confidence is too low: {minimum_confidence:.3f}"
            )

        positions = {
            name: joint_positions[index]
            for name, index in zip(joint_names, joint_indices, strict=True)
        }
        tips = {
            name: tip_positions[index]
            for name, index in zip(tip_names, tip_indices, strict=True)
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
        return RtmwHandTarget(
            side=side,
            normalized_positions=normalized,
            minimum_confidence=minimum_confidence,
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


class RtmwHandCommandFilter:
    """Binary flexion hysteresis plus filtered thumb-abduction control."""

    def __init__(self, config: RtmwHandsConfig) -> None:
        self.config = config
        self._state: dict[HandSide, np.ndarray] = {}
        self._updated_at: dict[HandSide, float] = {}
        self._flexion_closed: dict[HandSide, np.ndarray] = {}

    def reset(self, side: HandSide | None = None) -> None:
        if side is None:
            self._state.clear()
            self._updated_at.clear()
            self._flexion_closed.clear()
            return
        self._state.pop(side, None)
        self._updated_at.pop(side, None)
        self._flexion_closed.pop(side, None)

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
        calibration = self.config.calibration(side)
        opened = np.asarray(calibration.open_positions, dtype=np.float64)
        closed = np.asarray(calibration.closed_positions, dtype=np.float64)
        span = closed - opened
        normalized = np.zeros(6, dtype=np.float64)
        usable = np.abs(span) > np.finfo(np.float64).eps
        normalized[usable] = (measured[usable] - opened[usable]) / span[usable]
        self._flexion_closed[side] = normalized[list(_FLEXION_AXES)] >= 0.5

    def resume(self, side: HandSide, *, now: float) -> None:
        """Resume from the last command with one control period of authority."""

        if not math.isfinite(now):
            raise ValueError("hand filter timestamp must be finite")
        if side not in self._state:
            raise RuntimeError(f"hand filter side {side!r} has not been seeded")
        self._updated_at[side] = now - 1.0 / self.config.control_rate_hz

    def update(
        self,
        target: RtmwHandTarget,
        *,
        now: float,
    ) -> RawHandPositions:
        if not math.isfinite(now):
            raise ValueError("hand filter timestamp must be finite")
        side = target.side
        normalized = np.asarray(target.normalized_positions, dtype=np.float64)
        if normalized.shape != (6,) or not np.all(np.isfinite(normalized)):
            raise ValueError("normalized hand target must contain six finite values")
        if np.any((normalized < 0.0) | (normalized > 1.0)):
            raise ValueError("normalized hand target values must be in [0, 1]")
        closed = np.array(
            self._flexion_closed.get(side, np.zeros(len(_FLEXION_AXES), dtype=bool)),
            copy=True,
        )
        flexion = normalized[list(_FLEXION_AXES)]
        closed[(~closed) & (flexion >= self.config.flexion_close_threshold)] = True
        closed[closed & (flexion <= self.config.flexion_open_threshold)] = False
        self._flexion_closed[side] = closed
        binary = np.array(normalized, copy=True)
        binary[list(_FLEXION_AXES)] = closed.astype(np.float64)
        return self._filter_raw_target(side, binary, now=now)

    def _filter_raw_target(
        self,
        side: HandSide,
        normalized: np.ndarray,
        *,
        now: float,
    ) -> RawHandPositions:
        desired = self.config.calibration(side).raw_target(
            tuple(float(value) for value in normalized)  # type: ignore[arg-type]
        )
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
        flexion_axes = list(_FLEXION_AXES)
        filtered[flexion_axes] = desired[flexion_axes]
        filtered = np.clip(filtered, 0.0, 255.0)
        self._state[side] = filtered
        self._updated_at[side] = now
        rounded = np.floor(filtered + 0.5).astype(np.uint8)
        return tuple(int(value) for value in rounded)  # type: ignore[return-value]


class RtmwHandsStream:
    """Latest-only, validated, reconnecting RTMW hand-target stream."""

    def __init__(self, config: RtmwHandsConfig) -> None:
        self.config = config
        self._mapper = RtmwHandLandmarkMapper(config)
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._generation = 0
        self._consumed_generation = 0
        self._payload: RtmwHandsFrame | BaseException | None = None
        self._terminal_error: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._transport_lock = threading.Lock()
        self._channel: grpc.Channel | None = None
        self._ready_future: Any | None = None
        self._call: Any | None = None
        self._stub: Any | None = None
        self._stub_factory: Callable[[grpc.Channel], Any] | None = None
        self._proto: Any | None = None
        self.stream_instance_id: str | None = None
        self.stream_config_sha256: str | None = None
        self._last_frame_id: int | None = None
        self._last_source_timestamp_ns: int | None = None
        self._active_track_id: str | None = None

    def __enter__(self) -> RtmwHandsStream:
        if self._thread is not None:
            raise RuntimeError("RTMW hand stream is already running")
        skeleton_pb2, skeleton_pb2_grpc = _rtmw.import_pinned_rtmw_proto()
        self._proto = skeleton_pb2
        self._stub_factory = skeleton_pb2_grpc.SkeletonServiceStub
        self._stop.clear()
        try:
            if not self.config.arm:
                self._connect_transport()
            self._thread = threading.Thread(
                target=self._consume,
                name="bello-rtmw-hands-grpc",
                daemon=True,
            )
            self._thread.start()
        except BaseException:
            self._close_transport()
            raise
        return self

    def __exit__(self, *_exception: object) -> bool:
        with self._transport_lock:
            self._stop.set()
        with self._condition:
            self._condition.notify_all()
        self._close_transport()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=3.0)
            if thread.is_alive():
                raise RuntimeError("RTMW hand gRPC worker did not stop")
        self._thread = None
        return False

    def wait_next(
        self,
        *,
        after_sequence: int | None,
        timeout_seconds: float,
    ) -> RtmwHandsFrame | None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise ValueError("RTMW hand poll timeout must be positive and finite")
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while self._generation <= self._consumed_generation:
                if self._terminal_error is not None:
                    raise RuntimeError("RTMW hand gRPC stream failed") from (
                        self._terminal_error
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(remaining)
            self._consumed_generation = self._generation
            payload = self._payload
            self._condition.notify_all()
        if isinstance(payload, (RtmwHandsDiscontinuity, RtmwHandsUnavailable)):
            raise payload
        if isinstance(payload, BaseException):
            raise RuntimeError("RTMW hand gRPC stream failed") from payload
        if payload is None:
            raise RuntimeError("RTMW hand stream published an empty frame")
        if after_sequence is not None and payload.sequence <= after_sequence:
            self._reset_tracking_epoch(self._active_track_id)
            raise RtmwHandsDiscontinuity(
                "RTMW hand frame sequence did not advance: "
                f"{payload.sequence} <= {after_sequence}"
            )
        return payload

    def _create_channel(self) -> grpc.Channel:
        options: list[tuple[str, int | str]] = [
            ("grpc.max_receive_message_length", 4 * 1024 * 1024),
            ("grpc.max_send_message_length", 1024 * 1024),
        ]
        certificate_path = self.config.tls_root_certificate_path
        if certificate_path is None:
            return grpc.insecure_channel(self.config.endpoint, options=options)
        certificate = certificate_path.expanduser().resolve()
        if not certificate.is_file():
            raise FileNotFoundError(f"RTMW TLS CA file does not exist: {certificate}")
        override = self.config.tls_server_name_override
        if override is not None:
            options.append(("grpc.ssl_target_name_override", override))
        credentials = grpc.ssl_channel_credentials(
            root_certificates=certificate.read_bytes()
        )
        return grpc.secure_channel(self.config.endpoint, credentials, options=options)

    def _connect_transport(self) -> None:
        if self._proto is None or self._stub_factory is None:
            raise RuntimeError("RTMW hand protobuf transport is not initialized")
        channel = self._create_channel()
        with self._transport_lock:
            if self._stop.is_set():
                channel.close()
                raise _RtmwStreamStopped("RTMW hand stream stopped during connection")
            self._channel = channel
        ready_future = grpc.channel_ready_future(channel)
        with self._transport_lock:
            if self._stop.is_set():
                ready_future.cancel()
                raise _RtmwStreamStopped("RTMW hand stream stopped during connection")
            self._ready_future = ready_future
        try:
            ready_future.result(timeout=self.config.connect_timeout_seconds)
        finally:
            with self._transport_lock:
                if self._ready_future is ready_future:
                    self._ready_future = None
        stub = self._stub_factory(channel)
        info = stub.GetStreamInfo(
            empty_pb2.Empty(), timeout=self.config.rpc_timeout_seconds
        )
        _rtmw.validate_stream_info(info, self.config)
        status = stub.GetStatus(
            empty_pb2.Empty(), timeout=self.config.rpc_timeout_seconds
        )
        self._validate_service_status(status)
        call = stub.Subscribe(self._proto.SubscribeRequest())
        with self._transport_lock:
            if self._stop.is_set():
                call.cancel()
                raise _RtmwStreamStopped("RTMW hand stream stopped during connection")
            self._stub = stub
            self._call = call
            self.stream_instance_id = info.stream_instance_id
            self.stream_config_sha256 = info.config_sha256

    def _consume(self) -> None:
        outage_active = False
        reconnect_immediately = True
        reconnect_backoff_seconds = _RECONNECT_INITIAL_BACKOFF_SECONDS
        while not self._stop.is_set():
            with self._transport_lock:
                call = self._call
            if call is None:
                failure = "RTMW hand stream is not connected"
            else:
                try:
                    for message in call:
                        if self._stop.is_set():
                            return
                        try:
                            frame = self._parse_frame(message)
                        except RtmwHandsDiscontinuity as error:
                            self._publish_barrier(error)
                        except RtmwHandsUnavailable as error:
                            self._publish(error)
                        else:
                            self._publish(frame)
                            outage_active = False
                            reconnect_immediately = True
                            reconnect_backoff_seconds = (
                                _RECONNECT_INITIAL_BACKOFF_SECONDS
                            )
                    if self._stop.is_set():
                        return
                    failure = "RTMW hand gRPC stream ended"
                except grpc.RpcError as error:
                    if self._stop.is_set():
                        return
                    if error.code() not in _RETRYABLE_GRPC_CODES:
                        self._set_terminal(self._grpc_error(error))
                        return
                    failure = str(self._grpc_error(error))
                except BaseException as error:
                    if self._stop.is_set():
                        return
                    self._set_terminal(error)
                    return

            self._close_transport()
            if not outage_active:
                self._reset_tracking_epoch(None)
                self._publish_barrier(
                    RtmwHandsDiscontinuity(
                        f"{failure}; reconnecting to RTMW hand stream"
                    )
                )
                outage_active = True
            reconnected, reconnect_backoff_seconds = self._reconnect(
                reconnect_backoff_seconds,
                immediately=reconnect_immediately,
            )
            if not reconnected:
                return
            reconnect_immediately = False

    def _reconnect(
        self,
        backoff_seconds: float,
        *,
        immediately: bool,
    ) -> tuple[bool, float]:
        while not self._stop.is_set():
            if not immediately:
                if self._stop.wait(backoff_seconds):
                    return False, backoff_seconds
                backoff_seconds = min(
                    _RECONNECT_MAX_BACKOFF_SECONDS, 2.0 * backoff_seconds
                )
            try:
                self._connect_transport()
            except _RtmwStreamStopped:
                return False, backoff_seconds
            except grpc.FutureTimeoutError:
                if self._stop.is_set():
                    return False, backoff_seconds
            except _RtmwServiceUnavailable:
                if self._stop.is_set():
                    return False, backoff_seconds
            except grpc.RpcError as error:
                if self._stop.is_set():
                    return False, backoff_seconds
                if error.code() not in _RETRYABLE_GRPC_CODES:
                    self._set_terminal(self._grpc_error(error))
                    return False, backoff_seconds
            except BaseException as error:
                if self._stop.is_set():
                    return False, backoff_seconds
                self._set_terminal(error)
                return False, backoff_seconds
            else:
                return True, backoff_seconds
            self._close_transport()
            immediately = False
        return False, backoff_seconds

    def _parse_frame(self, message: Any) -> RtmwHandsFrame:
        received_at = time.monotonic()
        frame_id = int(message.frame_id)
        source_timestamp_ns = int(message.source_timestamp_ns)
        captured_ns = int(message.captured_monotonic_ns)
        output_ns = int(message.output_monotonic_ns)
        if source_timestamp_ns <= 0 or captured_ns <= 0 or output_ns < captured_ns:
            self._reset_tracking_epoch(None)
            raise RtmwHandsDiscontinuity("RTMW hand frame timestamps are invalid")
        if source_timestamp_ns != captured_ns:
            raise ValueError(
                "RTMW hand input must be live camera data with identical source "
                "and capture timestamps"
            )
        if message.stream_instance_id != self.stream_instance_id:
            self._reset_tracking_epoch(None)
            raise RtmwHandsDiscontinuity(
                "RTMW hand frame belongs to another stream instance"
            )
        if message.tracking_state != "tracked":
            self._reset_tracking_epoch(None)
            raise RtmwHandsUnavailable("RTMW hand tracking state is not 'tracked'")
        track_id = str(message.track_id)
        if not track_id:
            self._reset_tracking_epoch(None)
            raise RtmwHandsUnavailable("RTMW tracked hand frame has no track id")
        if self._active_track_id is not None and track_id != self._active_track_id:
            previous = self._active_track_id
            self._reset_tracking_epoch(track_id)
            raise RtmwHandsDiscontinuity(
                f"RTMW hand track changed from {previous!r} to {track_id!r}"
            )
        if self._last_frame_id is not None and frame_id <= self._last_frame_id:
            self._reset_tracking_epoch(track_id)
            raise RtmwHandsDiscontinuity("RTMW hand frame id did not advance")
        if self._last_source_timestamp_ns is not None:
            if source_timestamp_ns <= self._last_source_timestamp_ns:
                self._reset_tracking_epoch(track_id)
                raise RtmwHandsDiscontinuity(
                    "RTMW hand source timestamp did not advance"
                )
            interval = (source_timestamp_ns - self._last_source_timestamp_ns) / 1.0e9
            if interval > self.config.max_source_interval_seconds:
                self._reset_tracking_epoch(track_id)
                raise RtmwHandsDiscontinuity(
                    f"RTMW hand source interval is invalid: {interval:.6f} seconds"
                )
        self._active_track_id = track_id

        joint_count = len(_rtmw.RTMW_SMPLX52_NAMES)
        joint_positions = self._numeric_array(
            message.joint_positions, (joint_count, 3), "joint positions"
        )
        joint_confidence = self._confidence_array(
            message.joint_confidence, joint_count, "joint confidence"
        )
        joint_valid = self._boolean_array(
            message.joint_valid, joint_count, "joint validity"
        )
        tip_positions = self._numeric_array(
            message.finger_tip_positions,
            (len(RTMW_FINGER_TIP_NAMES), 3),
            "finger-tip positions",
        )
        tip_confidence = self._confidence_array(
            message.finger_tip_confidence,
            len(RTMW_FINGER_TIP_NAMES),
            "finger-tip confidence",
        )

        pipeline_latency_ms = float(message.pipeline_latency_ms)
        if not math.isfinite(pipeline_latency_ms) or pipeline_latency_ms < 0.0:
            raise RtmwHandsUnavailable("RTMW hand pipeline latency is invalid")
        if pipeline_latency_ms > self.config.max_pipeline_latency_ms:
            raise RtmwHandsUnavailable(
                f"RTMW hand pipeline latency is too high: {pipeline_latency_ms:.1f} ms"
            )
        timestamp_latency_ms = (output_ns - captured_ns) / 1.0e6
        if abs(timestamp_latency_ms - pipeline_latency_ms) > 1.0:
            raise RtmwHandsUnavailable(
                "RTMW hand latency scalar disagrees with its monotonic timestamps"
            )
        source_age_ms = math.ceil(
            pipeline_latency_ms + self.config.transport_latency_budget_ms
        )
        if source_age_ms > self.config.max_reference_age_ms:
            raise RtmwHandsUnavailable(
                f"RTMW hand source age budget is too high: {source_age_ms} ms"
            )

        targets: dict[HandSide, RtmwHandTarget | None] = {}
        reasons: dict[HandSide, str | None] = {}
        for side in _SIDES:
            if not self.config.enabled(side):
                targets[side] = None
                reasons[side] = "disabled"
                continue
            try:
                targets[side] = self._mapper.map_side(
                    side,
                    joint_positions,
                    joint_confidence,
                    joint_valid,
                    tip_positions,
                    tip_confidence,
                )
            except _HandGeometryUnavailable as error:
                targets[side] = None
                reasons[side] = str(error)
            else:
                reasons[side] = None
        if not any(targets.values()):
            detail = "; ".join(
                f"{side}: {reasons[side]}"
                for side in _SIDES
                if self.config.enabled(side)
            )
            raise RtmwHandsUnavailable(
                f"RTMW hand landmarks are unavailable ({detail})"
            )

        self._last_frame_id = frame_id
        self._last_source_timestamp_ns = source_timestamp_ns
        return RtmwHandsFrame(
            sequence=frame_id,
            source_timestamp_ns=source_timestamp_ns,
            source_age_ms=source_age_ms,
            received_at=received_at,
            track_id=track_id,
            left=targets["left"],
            right=targets["right"],
            left_unavailable_reason=reasons["left"],
            right_unavailable_reason=reasons["right"],
        )

    def _reset_tracking_epoch(self, track_id: str | None) -> None:
        self._active_track_id = track_id
        self._last_frame_id = None
        self._last_source_timestamp_ns = None

    def _publish(self, payload: RtmwHandsFrame | BaseException) -> None:
        with self._condition:
            self._payload = payload
            self._generation += 1
            self._condition.notify_all()

    def _publish_barrier(self, error: RtmwHandsDiscontinuity) -> None:
        with self._condition:
            self._payload = error
            self._generation += 1
            generation = self._generation
            self._condition.notify_all()
            while self._consumed_generation < generation and not self._stop.is_set():
                self._condition.wait(timeout=0.1)

    def _set_terminal(self, error: BaseException) -> None:
        with self._condition:
            self._terminal_error = error
            self._payload = error
            self._generation += 1
            self._condition.notify_all()

    def _close_transport(self) -> None:
        with self._transport_lock:
            call = self._call
            ready_future = self._ready_future
            channel = self._channel
            self._call = None
            self._ready_future = None
            self._stub = None
            self._channel = None
        if call is not None:
            call.cancel()
        if ready_future is not None:
            ready_future.cancel()
        if channel is not None:
            channel.close()

    @staticmethod
    def _validate_service_status(message: Any) -> None:
        if message.terminal_failure:
            raise RuntimeError(f"RTMW service failed: {message.terminal_failure}")
        if not message.ready:
            raise _RtmwServiceUnavailable("RTMW service is not ready")
        if not message.source_running or not message.inference_running:
            raise _RtmwServiceUnavailable(
                "RTMW source or inference worker is not running"
            )

    @staticmethod
    def _grpc_error(error: grpc.RpcError) -> RuntimeError:
        detail = error.details() or "RPC failed"
        return RuntimeError(f"RTMW hand gRPC {error.code().name}: {detail}")

    @staticmethod
    def _numeric_array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"RTMW hand {label} must contain numbers") from error
        expected = math.prod(shape)
        if array.size != expected:
            raise ValueError(
                f"RTMW hand {label} has {array.size} values; expected {expected}"
            )
        return np.array(array.reshape(shape), copy=True, order="C")

    @classmethod
    def _confidence_array(cls, value: Any, count: int, label: str) -> np.ndarray:
        result = cls._numeric_array(value, (count,), label)
        if not np.all(np.isfinite(result)) or np.any((result < 0.0) | (result > 1.0)):
            raise ValueError(f"RTMW hand {label} must be finite and in [0, 1]")
        return result

    @staticmethod
    def _boolean_array(value: Any, count: int, label: str) -> np.ndarray:
        array = np.asarray(value)
        if array.shape != (count,) or array.dtype.kind != "b":
            raise ValueError(f"RTMW hand {label} must contain {count} booleans")
        return np.array(array, dtype=bool, copy=True)


def run_rtmw_hands(
    config: RtmwHandsConfig,
    transport_factory: HandTransportFactory,
    *,
    stop_event: threading.Event | None = None,
    stream_factory: Callable[[RtmwHandsConfig], ContextManager[RtmwHandsStream]] = (
        RtmwHandsStream
    ),
) -> None:
    """Follow the latest fresh RTMW hand pose until stopped.

    Missing, stale, or disconnected RTMW input suspends position writes for the
    affected side. Flexion axes are binary; only thumb abduction uses the
    configured low-pass and slew limits.
    """

    stop = stop_event if stop_event is not None else threading.Event()
    position_filter = RtmwHandCommandFilter(config)
    transport_context: ContextManager[HandPositionTransport | None]
    if config.arm:
        transport_context = transport_factory(config)
    else:
        transport_context = nullcontext(None)
    period = 1.0 / config.control_rate_hz
    latest: RtmwHandsFrame | None = None
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
                        "RTMW hand preflight duration expired before an acceptable "
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
            except (RtmwHandsDiscontinuity, RtmwHandsUnavailable) as error:
                if not stale:
                    _LOGGER.warning("RTMW hand target unavailable: %s", error)
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
                        "RTMW hand target became stale: %.1f ms > %d ms",
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
        raise RuntimeError("RTMW hand preflight stopped before an acceptable frame")


__all__ = (
    "HAND_POSITION_ORDER",
    "RTMW_FINGER_TIP_NAMES",
    "HandGeometryConfig",
    "HandPositionTransport",
    "HandRawCalibration",
    "HandSide",
    "RawHandPositions",
    "RtmwHandCommandFilter",
    "RtmwHandLandmarkMapper",
    "RtmwHandTarget",
    "RtmwHandsConfig",
    "RtmwHandsDiscontinuity",
    "RtmwHandsFrame",
    "RtmwHandsStream",
    "RtmwHandsUnavailable",
    "run_rtmw_hands",
)
