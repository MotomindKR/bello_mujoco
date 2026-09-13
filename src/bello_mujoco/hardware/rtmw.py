"""Pinned RTMW gRPC source for locomotion upper-body control."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
import threading
import time
from typing import Any, Callable

import grpc
from google.protobuf import empty_pb2
import numpy as np

from bello_mujoco.fastsam_conditioning import (
    CausalFootConditionedRoot,
    FootConditionedRootConfig,
)
from bello_mujoco.hardware import rtmw_contract as _rtmw_contract
from bello_mujoco.hardware.rtmw_contract import (
    RTMW_COORDINATE_SPACE as RTMW_COORDINATE_SPACE,
    RTMW_ROTATION_SEMANTICS as RTMW_ROTATION_SEMANTICS,
    RTMW_SCHEMA_ID as RTMW_SCHEMA_ID,
    RTMW_SMPLX52_NAMES,
    RTMW_SMPLX52_PARENTS,
    validate_stream_info as _validate_stream_info,
)
from bello_mujoco.hardware.smplx_retarget import (
    SMPLX_MODEL_TO_WORLD_WXYZ,
    SmplxTargetFrame,
    load_smplx_standing_sample,
)
from bello_mujoco.hardware.reference import ReferenceDiscontinuity, ReferenceUnavailable

_REQUIRED_RANK_TWO_ROTATION_NAMES = ("pelvis",)
_RETRYABLE_GRPC_CODES = frozenset(
    (
        grpc.StatusCode.CANCELLED,
        grpc.StatusCode.DEADLINE_EXCEEDED,
        grpc.StatusCode.UNAVAILABLE,
    )
)
_RECONNECT_INITIAL_BACKOFF_SECONDS = 0.1
_RECONNECT_MAX_BACKOFF_SECONDS = 1.0
_LIVENESS_PROBE_INTERVAL_SECONDS = 0.5
_LIVENESS_FAILURE_LIMIT = 2
_LIVENESS_MISSED_DELIVERY_LIMIT = 2


class _RtmwServiceUnavailable(RuntimeError):
    """The producer is reachable but not ready to stream yet."""


class _RtmwStreamStopped(RuntimeError):
    """Internal signal used to interrupt transport setup during cleanup."""


@dataclass(frozen=True)
class RtmwGrpcConfig:
    """Transport, semantics, and freshness checks for the RTMW stream."""

    endpoint: str
    neutral_model_path: Path
    tls_root_certificate_path: Path | None
    tls_server_name_override: str | None
    connect_timeout_seconds: float
    rpc_timeout_seconds: float
    human_height_m: float
    min_joint_confidence: float
    min_rotation_confidence: float
    max_required_rotation_residual_degrees: float
    max_pipeline_latency_ms: float
    transport_latency_budget_ms: float
    max_source_interval_seconds: float
    max_source_age_ms: int
    camera_to_world_quaternion_xyzw: tuple[float, float, float, float]
    align_initial_heading: bool
    align_initial_floor: bool
    root_conditioning: FootConditionedRootConfig = field(
        default_factory=FootConditionedRootConfig
    )

    def __post_init__(self) -> None:
        if not self.endpoint or "://" in self.endpoint:
            raise ValueError("RTMW endpoint must be a gRPC host:port target")
        if not isinstance(self.neutral_model_path, Path):
            raise TypeError("RTMW neutral_model_path must be a pathlib.Path")
        for name, value in (
            ("connect_timeout_seconds", self.connect_timeout_seconds),
            ("rpc_timeout_seconds", self.rpc_timeout_seconds),
            ("human_height_m", self.human_height_m),
            ("max_source_interval_seconds", self.max_source_interval_seconds),
            (
                "max_required_rotation_residual_degrees",
                self.max_required_rotation_residual_degrees,
            ),
            ("max_pipeline_latency_ms", self.max_pipeline_latency_ms),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"RTMW {name} must be positive and finite")
        if self.max_required_rotation_residual_degrees > 180.0:
            raise ValueError(
                "RTMW max_required_rotation_residual_degrees must not exceed 180"
            )
        for name, value in (
            ("min_joint_confidence", self.min_joint_confidence),
            ("min_rotation_confidence", self.min_rotation_confidence),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"RTMW {name} must be finite and in [0, 1]")
        if self.max_source_age_ms <= 0:
            raise ValueError("RTMW max_source_age_ms must be positive")
        if (
            not math.isfinite(self.transport_latency_budget_ms)
            or self.transport_latency_budget_ms < 0.0
        ):
            raise ValueError(
                "RTMW transport_latency_budget_ms must be finite and nonnegative"
            )
        quaternion = np.asarray(self.camera_to_world_quaternion_xyzw)
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            raise ValueError("RTMW camera rotation must be a finite quaternion")
        if not 0.999 <= float(np.linalg.norm(quaternion)) <= 1.001:
            raise ValueError("RTMW camera rotation quaternion must be normalized")
        if self.tls_server_name_override is not None and (
            self.tls_root_certificate_path is None
        ):
            raise ValueError("RTMW TLS server-name override requires a CA file")


class RtmwCoordinateAdapter:
    """Map RTMW OpenCV camera poses into Bello's z-up world frame."""

    def __init__(self, config: RtmwGrpcConfig) -> None:
        xyzw = np.asarray(config.camera_to_world_quaternion_xyzw, dtype=np.float64)
        self._camera_to_world = xyzw[[3, 0, 1, 2]]
        self._align_heading = config.align_initial_heading
        self._align_floor = config.align_initial_floor
        self.reset()

    def reset(self) -> None:
        self._track_id: int | str | None = None
        self._origin: np.ndarray | None = None
        self._heading: np.ndarray | None = None
        self._previous_quaternions: np.ndarray | None = None

    def transform(
        self,
        positions_camera: np.ndarray,
        quaternions_camera_xyzw: np.ndarray,
        *,
        track_id: int | str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._track_id is not None and track_id != self._track_id:
            self.reset()
        self._track_id = track_id
        positions = _quat_rotate(self._camera_to_world, positions_camera)
        camera_quaternions = quaternions_camera_xyzw[:, (3, 0, 1, 2)]
        quaternions = _quat_multiply(self._camera_to_world, camera_quaternions)
        pelvis = RTMW_SMPLX52_NAMES.index("pelvis")
        if self._origin is None:
            origin = np.zeros(3, dtype=np.float64)
            origin[:2] = positions[pelvis, :2]
            if self._align_floor:
                feet = [
                    RTMW_SMPLX52_NAMES.index(name)
                    for name in ("left_foot", "right_foot")
                ]
                origin[2] = float(np.min(positions[feet, 2]))
            self._origin = origin
            if self._align_heading:
                canonical = np.asarray(SMPLX_MODEL_TO_WORLD_WXYZ, dtype=np.float64)
                relative = _quat_multiply(
                    quaternions[pelvis], _quat_conjugate(canonical)
                )
                yaw = _yaw_wxyz(relative)
            else:
                yaw = 0.0
            self._heading = np.asarray(
                (math.cos(-0.5 * yaw), 0.0, 0.0, math.sin(-0.5 * yaw)),
                dtype=np.float64,
            )
        assert self._origin is not None and self._heading is not None
        positions = _quat_rotate(self._heading, positions - self._origin)
        quaternions = _quat_multiply(self._heading, quaternions)
        if self._previous_quaternions is not None:
            signs = np.sum(quaternions * self._previous_quaternions, axis=1) < 0.0
            quaternions[signs] *= -1.0
        self._previous_quaternions = quaternions.copy()
        return positions, quaternions


class RtmwSmplxStream:
    """Latest-only, validated RTMW SMPL-X stream backed by gRPC."""

    def __init__(
        self,
        config: RtmwGrpcConfig,
        source_body_names: tuple[str, ...],
        *,
        standing_sample_loader: Callable[
            [Path, tuple[str, ...]], SmplxTargetFrame
        ] = load_smplx_standing_sample,
    ) -> None:
        self.config = config
        if not source_body_names or len(set(source_body_names)) != len(
            source_body_names
        ):
            raise ValueError("RTMW source body names must be nonempty and unique")
        unknown = sorted(set(source_body_names) - set(RTMW_SMPLX52_NAMES))
        if unknown:
            raise ValueError(
                "RTMW source body names are absent from SMPL-X: " + ", ".join(unknown)
            )
        self.source_body_names = source_body_names
        self._standing_sample = standing_sample_loader(
            config.neutral_model_path, source_body_names
        )
        self.neutral_model_sha256 = _file_sha256(config.neutral_model_path)
        self._coordinates = RtmwCoordinateAdapter(config)
        self._root_conditioner = CausalFootConditionedRoot(
            RTMW_SMPLX52_NAMES, config.root_conditioning
        )
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._generation = 0
        self._consumed_generation = 0
        self._payload: SmplxTargetFrame | BaseException | None = None
        self._terminal_error: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._liveness_thread: threading.Thread | None = None
        self._liveness_stop = threading.Event()
        self._transport_lock = threading.Lock()
        self._channel: grpc.Channel | None = None
        self._ready_future: Any | None = None
        self._call: Any | None = None
        self._stub: Any | None = None
        self._stub_factory: Callable[[grpc.Channel], Any] | None = None
        self._proto: Any | None = None
        self.stream_instance_id: str | None = None
        self.stream_config_sha256: str | None = None
        self.depth_provenance: str | None = None
        self._last_frame_id: int | None = None
        self._last_source_timestamp_ns: int | None = None
        self._active_track_id: str | None = None
        self._wire_generation = 0
        self._last_wire_received_at = 0.0

    def __enter__(self) -> RtmwSmplxStream:
        if self._thread is not None:
            raise RuntimeError("RTMW stream is already running")
        skeleton_pb2, skeleton_pb2_grpc = import_pinned_rtmw_proto()
        self._proto = skeleton_pb2
        self._stub_factory = skeleton_pb2_grpc.SkeletonServiceStub
        self._stop.clear()
        self._liveness_stop.clear()
        try:
            self._connect_transport()
            self._thread = threading.Thread(
                target=self._consume,
                name="bello-rtmw-grpc",
                daemon=True,
            )
            self._thread.start()
            self._liveness_thread = threading.Thread(
                target=self._monitor_liveness,
                name="bello-rtmw-liveness",
                daemon=True,
            )
            self._liveness_thread.start()
        except BaseException:
            self._stop.set()
            self._liveness_stop.set()
            self._close_transport()
            for thread in (self._thread, self._liveness_thread):
                if thread is not None and thread.is_alive():
                    thread.join(timeout=self.config.rpc_timeout_seconds + 1.0)
            self._thread = None
            self._liveness_thread = None
            raise
        return self

    def __exit__(self, *_exception: object) -> bool:
        with self._transport_lock:
            self._stop.set()
            self._liveness_stop.set()
        with self._condition:
            self._condition.notify_all()
        self._close_transport()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=3.0)
            if thread.is_alive():
                raise RuntimeError("RTMW gRPC worker did not stop")
        self._thread = None
        liveness_thread = self._liveness_thread
        if liveness_thread is not None:
            liveness_thread.join(timeout=self.config.rpc_timeout_seconds + 1.0)
            if liveness_thread.is_alive():
                raise RuntimeError("RTMW gRPC liveness worker did not stop")
        self._liveness_thread = None
        return False

    def standing_sample(self) -> SmplxTargetFrame:
        return self._standing_sample

    def wait_next(
        self,
        *,
        after_sequence: int | None,
        timeout_seconds: float,
    ) -> SmplxTargetFrame | None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise ValueError("RTMW poll timeout must be positive and finite")
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while self._generation <= self._consumed_generation:
                if self._terminal_error is not None:
                    raise RuntimeError("RTMW gRPC stream failed") from (
                        self._terminal_error
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(remaining)
            self._consumed_generation = self._generation
            payload = self._payload
            self._condition.notify_all()
        if isinstance(payload, (ReferenceDiscontinuity, ReferenceUnavailable)):
            raise payload
        if isinstance(payload, BaseException):
            raise RuntimeError("RTMW gRPC stream failed") from payload
        if payload is None:
            raise RuntimeError("RTMW stream published an empty frame")
        if after_sequence is not None and payload.sequence <= after_sequence:
            self._reset_tracking_epoch(self._active_track_id)
            raise ReferenceDiscontinuity(
                "RTMW frame sequence did not advance: "
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
            raise RuntimeError("RTMW protobuf transport is not initialized")
        channel = self._create_channel()
        with self._transport_lock:
            if self._stop.is_set():
                channel.close()
                raise _RtmwStreamStopped("RTMW stream stopped during connection")
            self._channel = channel
        ready_future = grpc.channel_ready_future(channel)
        with self._transport_lock:
            if self._stop.is_set():
                ready_future.cancel()
                raise _RtmwStreamStopped("RTMW stream stopped during connection")
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
        _validate_stream_info(info, self.config)
        status = stub.GetStatus(
            empty_pb2.Empty(), timeout=self.config.rpc_timeout_seconds
        )
        _validate_service_status(status)
        call = stub.Subscribe(self._proto.SubscribeRequest())
        with self._transport_lock:
            if self._stop.is_set():
                call.cancel()
                raise _RtmwStreamStopped("RTMW stream stopped during connection")
            self._stub = stub
            self._call = call
            self._wire_generation = 0
            self._last_wire_received_at = time.monotonic()
            self.stream_instance_id = info.stream_instance_id
            self.stream_config_sha256 = info.config_sha256
            self.depth_provenance = info.depth_provenance

    def _consume(self) -> None:
        outage_active = False
        reconnect_immediately = True
        reconnect_backoff_seconds = _RECONNECT_INITIAL_BACKOFF_SECONDS
        while not self._stop.is_set():
            try:
                with self._transport_lock:
                    call = self._call
                if call is None:
                    raise RuntimeError("RTMW gRPC stream has no active subscription")
                for message in call:
                    if self._stop.is_set():
                        return
                    with self._transport_lock:
                        if self._call is call:
                            self._wire_generation += 1
                            self._last_wire_received_at = time.monotonic()
                    try:
                        frame = self._parse_frame(message)
                    except (ReferenceDiscontinuity, ReferenceUnavailable) as error:
                        if isinstance(error, ReferenceDiscontinuity):
                            self._publish_discontinuity(error)
                        else:
                            self._publish(error)
                    else:
                        self._publish(frame)
                        outage_active = False
                        reconnect_immediately = True
                        reconnect_backoff_seconds = _RECONNECT_INITIAL_BACKOFF_SECONDS
                if self._stop.is_set():
                    return
                failure = "RTMW gRPC stream ended"
            except grpc.RpcError as error:
                if self._stop.is_set():
                    return
                if error.code() not in _RETRYABLE_GRPC_CODES:
                    self._set_terminal(_grpc_error(error))
                    return
                failure = str(_grpc_error(error))
            except BaseException as error:
                if self._stop.is_set():
                    return
                self._set_terminal(error)
                return

            self._close_transport()
            if not outage_active:
                self._reset_tracking_epoch(None)
                self._publish_discontinuity(
                    ReferenceDiscontinuity(f"{failure}; reconnecting to RTMW")
                )
                outage_active = True
            reconnected, reconnect_backoff_seconds = self._reconnect(
                reconnect_backoff_seconds,
                immediately=reconnect_immediately,
            )
            if not reconnected:
                return
            reconnect_immediately = False

    def _monitor_liveness(self) -> None:
        watched_call: Any | None = None
        cancelled_call: Any | None = None
        previous_status_frame_id: int | None = None
        previous_wire_generation = 0
        consecutive_failures = 0
        missed_deliveries = 0
        while not self._liveness_stop.wait(_LIVENESS_PROBE_INTERVAL_SECONDS):
            with self._transport_lock:
                stub = self._stub
                call = self._call
                expected_instance_id = self.stream_instance_id
                wire_generation = self._wire_generation
                last_wire_received_at = self._last_wire_received_at
            if stub is None or call is None or call is cancelled_call:
                continue
            if call is not watched_call:
                watched_call = call
                previous_status_frame_id = None
                previous_wire_generation = wire_generation
                consecutive_failures = 0
                missed_deliveries = 0
            if (
                time.monotonic() - last_wire_received_at
                < _LIVENESS_PROBE_INTERVAL_SECONDS
            ):
                previous_status_frame_id = None
                previous_wire_generation = wire_generation
                consecutive_failures = 0
                missed_deliveries = 0
                continue
            timeout_seconds = min(
                self.config.rpc_timeout_seconds,
                _LIVENESS_PROBE_INTERVAL_SECONDS,
            )
            try:
                info = stub.GetStreamInfo(empty_pb2.Empty(), timeout=timeout_seconds)
                status = stub.GetStatus(empty_pb2.Empty(), timeout=timeout_seconds)
            except grpc.RpcError as error:
                if self._stop.is_set():
                    return
                with self._transport_lock:
                    if self._stub is not stub or self._call is not call:
                        continue
                    current_wire_generation = self._wire_generation
                if current_wire_generation != previous_wire_generation:
                    previous_status_frame_id = None
                    previous_wire_generation = current_wire_generation
                    consecutive_failures = 0
                    missed_deliveries = 0
                    continue
                if error.code() not in _RETRYABLE_GRPC_CODES:
                    self._stop.set()
                    self._liveness_stop.set()
                    self._set_terminal(_grpc_error(error))
                    self._cancel_subscription(stub, call)
                    return
                consecutive_failures += 1
                if consecutive_failures < _LIVENESS_FAILURE_LIMIT:
                    continue
                cancelled_call = call
                self._cancel_subscription(stub, call)
                continue
            except BaseException as error:
                if self._stop.is_set():
                    return
                self._stop.set()
                self._liveness_stop.set()
                self._set_terminal(error)
                self._cancel_subscription(stub, call)
                return

            with self._transport_lock:
                if self._stub is not stub or self._call is not call:
                    continue
                wire_generation = self._wire_generation
            consecutive_failures = 0
            try:
                _validate_stream_info(info, self.config)
                _validate_service_status(status)
            except _RtmwServiceUnavailable:
                cancelled_call = call
                self._cancel_subscription(stub, call)
                continue
            except BaseException as error:
                self._stop.set()
                self._liveness_stop.set()
                self._set_terminal(error)
                self._cancel_subscription(stub, call)
                return
            if info.stream_instance_id != expected_instance_id:
                cancelled_call = call
                self._cancel_subscription(stub, call)
                continue
            status_frame_id = _optional_latest_frame_id(status)
            subscriber_advanced = wire_generation != previous_wire_generation
            if subscriber_advanced:
                missed_deliveries = 0
            elif missed_deliveries > 0:
                missed_deliveries += 1
            if previous_status_frame_id is not None:
                producer_advanced = (
                    status_frame_id is not None
                    and status_frame_id > previous_status_frame_id
                )
                producer_regressed = (
                    status_frame_id is None
                    or status_frame_id < previous_status_frame_id
                )
                if producer_regressed:
                    cancelled_call = call
                    self._cancel_subscription(stub, call)
                    continue
                if producer_advanced and not subscriber_advanced:
                    missed_deliveries = max(1, missed_deliveries)
                if missed_deliveries >= _LIVENESS_MISSED_DELIVERY_LIMIT:
                    cancelled_call = call
                    self._cancel_subscription(stub, call)
                    continue
            previous_status_frame_id = status_frame_id
            previous_wire_generation = wire_generation

    def _cancel_subscription(self, stub: Any, call: Any) -> None:
        with self._transport_lock:
            current = self._stub is stub and self._call is call
        if current:
            call.cancel()

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
                    _RECONNECT_MAX_BACKOFF_SECONDS,
                    2.0 * backoff_seconds,
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
                    self._set_terminal(_grpc_error(error))
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

    def _parse_frame(self, message: Any) -> SmplxTargetFrame:
        received_at = time.monotonic()
        frame_id = int(message.frame_id)
        source_timestamp_ns = int(message.source_timestamp_ns)
        captured_ns = int(message.captured_monotonic_ns)
        output_ns = int(message.output_monotonic_ns)
        if source_timestamp_ns <= 0 or captured_ns <= 0 or output_ns < captured_ns:
            raise ReferenceDiscontinuity("RTMW frame timestamps are invalid")
        if source_timestamp_ns != captured_ns:
            raise ValueError(
                "RTMW hardware input must come from a live camera whose source and "
                "capture timestamps are identical"
            )
        if message.stream_instance_id != self.stream_instance_id:
            raise ReferenceDiscontinuity("RTMW frame belongs to another stream")
        if message.tracking_state != "tracked":
            raise ReferenceUnavailable("RTMW tracking state is not 'tracked'")
        track_id = message.track_id
        if not track_id:
            raise ReferenceUnavailable("RTMW tracked frame has no track id")
        if self._active_track_id is not None and track_id != self._active_track_id:
            previous = self._active_track_id
            self._reset_tracking_epoch(track_id)
            raise ReferenceDiscontinuity(
                f"RTMW track changed from {previous!r} to {track_id!r}"
            )
        if self._last_frame_id is not None and frame_id <= self._last_frame_id:
            self._reset_tracking_epoch(track_id)
            raise ReferenceDiscontinuity("RTMW frame id did not advance")
        if (
            self._last_source_timestamp_ns is not None
            and source_timestamp_ns <= self._last_source_timestamp_ns
        ):
            self._reset_tracking_epoch(track_id)
            raise ReferenceDiscontinuity("RTMW source timestamp did not advance")
        self._active_track_id = track_id

        positions = _finite_array(
            message.joint_positions,
            (len(RTMW_SMPLX52_NAMES), 3),
            "joint positions",
        )
        root = _finite_array(message.root_translation, (3,), "root translation")
        if not np.allclose(root, positions[0], atol=1.0e-5, rtol=0.0):
            raise ValueError("RTMW root translation differs from its pelvis position")
        confidence = _confidence_array(
            message.joint_confidence,
            len(RTMW_SMPLX52_NAMES),
            "joint confidence",
        )
        valid_mask = _boolean_array(
            message.joint_valid, len(RTMW_SMPLX52_NAMES), "joint validity"
        )
        required = np.asarray(
            [RTMW_SMPLX52_NAMES.index(name) for name in self.source_body_names]
        )
        if not np.all(valid_mask[required]):
            name = self.source_body_names[int(np.flatnonzero(~valid_mask[required])[0])]
            raise ReferenceUnavailable(f"RTMW joint {name!r} has no valid position")
        minimum = float(np.min(confidence[required]))
        if minimum < self.config.min_joint_confidence:
            name = self.source_body_names[int(np.argmin(confidence[required]))]
            raise ReferenceUnavailable(
                f"RTMW joint {name!r} confidence is too low: {minimum:.3f}"
            )

        local_quaternions = _normalized_quaternions(
            message.local_joint_quaternions, "local joint quaternions"
        )
        global_quaternions = _normalized_quaternions(
            message.global_joint_quaternions, "global joint quaternions"
        )
        _validate_tree_consistency(local_quaternions, global_quaternions)
        rotation_valid = _boolean_array(
            message.rotation_valid,
            len(RTMW_SMPLX52_NAMES),
            "rotation validity",
        )
        rotation_confidence = _confidence_array(
            message.rotation_confidence,
            len(RTMW_SMPLX52_NAMES),
            "rotation confidence",
        )
        rotation_residual = _rotation_residual_array(
            message.rotation_residual_degrees,
            len(RTMW_SMPLX52_NAMES),
        )
        rotation_rank = _rotation_rank_array(
            message.rotation_constraint_rank,
            len(RTMW_SMPLX52_NAMES),
        )
        for name in _REQUIRED_RANK_TWO_ROTATION_NAMES:
            index = RTMW_SMPLX52_NAMES.index(name)
            if not rotation_valid[index]:
                raise ReferenceUnavailable(
                    f"RTMW joint {name!r} has no valid derived rotation"
                )
            if rotation_confidence[index] < self.config.min_rotation_confidence:
                raise ReferenceUnavailable(
                    f"RTMW joint {name!r} rotation confidence is too low: "
                    f"{rotation_confidence[index]:.3f}"
                )
            if rotation_rank[index] != 2:
                raise ReferenceUnavailable(
                    f"RTMW joint {name!r} rotation has constraint rank "
                    f"{rotation_rank[index]}; expected 2"
                )
            residual = float(rotation_residual[index])
            if not math.isfinite(residual):
                raise ReferenceUnavailable(
                    f"RTMW joint {name!r} rotation has no finite residual"
                )
            if residual > self.config.max_required_rotation_residual_degrees:
                raise ReferenceUnavailable(
                    f"RTMW joint {name!r} rotation residual is too high: "
                    f"{residual:.1f} degrees"
                )
        usable_rotation = (
            rotation_valid
            & (rotation_confidence >= self.config.min_rotation_confidence)
            & np.isfinite(rotation_residual)
            & (rotation_residual <= self.config.max_required_rotation_residual_degrees)
        )
        source_quaternions = global_quaternions.copy()
        for index in required:
            if usable_rotation[index]:
                continue
            ancestor = RTMW_SMPLX52_PARENTS[index]
            while ancestor >= 0 and not usable_rotation[ancestor]:
                ancestor = RTMW_SMPLX52_PARENTS[ancestor]
            if ancestor < 0:
                name = RTMW_SMPLX52_NAMES[index]
                raise ReferenceUnavailable(
                    f"RTMW joint {name!r} has no valid ancestor rotation"
                )
            source_quaternions[index] = global_quaternions[ancestor]

        pipeline_latency_ms = float(message.pipeline_latency_ms)
        if not math.isfinite(pipeline_latency_ms) or pipeline_latency_ms < 0.0:
            raise ReferenceUnavailable("RTMW pipeline latency is invalid")
        if pipeline_latency_ms > self.config.max_pipeline_latency_ms:
            raise ReferenceUnavailable(
                f"RTMW pipeline latency is too high: {pipeline_latency_ms:.1f} ms"
            )
        timestamp_latency_ms = (output_ns - captured_ns) / 1.0e6
        if abs(timestamp_latency_ms - pipeline_latency_ms) > 1.0:
            raise ReferenceUnavailable(
                "RTMW latency scalar disagrees with its monotonic timestamps"
            )
        source_age_ms = math.ceil(
            pipeline_latency_ms + self.config.transport_latency_budget_ms
        )
        if source_age_ms > self.config.max_source_age_ms:
            raise ReferenceUnavailable(
                f"RTMW source age budget is too high: {source_age_ms} ms"
            )
        positions, quaternions = self._coordinates.transform(
            positions, source_quaternions, track_id=track_id
        )
        try:
            positions, foot_contacts = self._root_conditioner.condition(
                positions,
                confidence,
                timestamp_ns=source_timestamp_ns,
            )
        except ValueError as error:
            self._reset_tracking_epoch(track_id)
            raise ReferenceDiscontinuity(str(error)) from error
        lookup = dict(zip(RTMW_SMPLX52_NAMES, positions, strict=True))
        quaternion_lookup = dict(zip(RTMW_SMPLX52_NAMES, quaternions, strict=True))
        self._last_frame_id = frame_id
        self._last_source_timestamp_ns = source_timestamp_ns
        return SmplxTargetFrame(
            transforms={
                name: (lookup[name].copy(), quaternion_lookup[name].copy())
                for name in self.source_body_names
            },
            sequence=frame_id,
            source_timestamp_ns=source_timestamp_ns,
            source_age_ms=source_age_ms,
            received_at=received_at,
            foot_contacts=foot_contacts,
        )

    def _reset_tracking_epoch(self, track_id: str | None) -> None:
        self._coordinates.reset()
        self._root_conditioner.reset()
        self._active_track_id = track_id
        self._last_frame_id = None
        self._last_source_timestamp_ns = None

    def _publish(self, payload: SmplxTargetFrame | BaseException) -> None:
        with self._condition:
            self._payload = payload
            self._generation += 1
            self._condition.notify_all()

    def _publish_discontinuity(self, error: ReferenceDiscontinuity) -> None:
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


def import_pinned_rtmw_proto() -> tuple[Any, Any]:
    return _rtmw_contract.import_pinned_rtmw_proto()


def _validate_service_status(message: Any) -> None:
    if message.terminal_failure:
        raise RuntimeError(f"RTMW service failed: {message.terminal_failure}")
    if not message.ready:
        raise _RtmwServiceUnavailable("RTMW service is not ready")
    if not message.source_running or not message.inference_running:
        raise _RtmwServiceUnavailable("RTMW source or inference worker is not running")


def _optional_latest_frame_id(message: Any) -> int | None:
    has_field = getattr(message, "HasField", None)
    if has_field is not None and not has_field("latest_frame_id"):
        return None
    value = getattr(message, "latest_frame_id", None)
    return None if value is None else int(value)


def _grpc_error(error: grpc.RpcError) -> RuntimeError:
    detail = error.details() or "RPC failed"
    return RuntimeError(f"RTMW gRPC {error.code().name}: {detail}")


def _finite_array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"RTMW {label} must contain finite numbers") from error
    if array.size != math.prod(shape):
        raise ValueError(
            f"RTMW {label} has {array.size} values; expected {math.prod(shape)}"
        )
    result = np.array(array.reshape(shape), copy=True, order="C")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"RTMW {label} must contain finite numbers")
    return result


def _confidence_array(value: Any, count: int, label: str) -> np.ndarray:
    result = _finite_array(value, (count,), label)
    if np.any((result < 0.0) | (result > 1.0)):
        raise ValueError(f"RTMW {label} must be in [0, 1]")
    return result


def _rotation_residual_array(value: Any, count: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("RTMW rotation residuals must contain numbers") from error
    if (
        array.shape != (count,)
        or np.any(np.isinf(array))
        or np.any(array < 0.0)
        or np.any(array > 180.0)
    ):
        raise ValueError(
            f"RTMW rotation residuals must contain {count} values in [0, 180] or NaN"
        )
    return np.array(array, copy=True)


def _rotation_rank_array(value: Any, count: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (count,) or array.dtype.kind not in "iu":
        raise ValueError(f"RTMW rotation ranks must contain {count} integers")
    if np.any((array < 0) | (array > 2)):
        raise ValueError("RTMW rotation ranks must be 0, 1, or 2")
    return np.asarray(array, dtype=np.uint8).copy()


def _boolean_array(value: Any, count: int, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (count,) or array.dtype.kind != "b":
        raise ValueError(f"RTMW {label} must contain {count} booleans")
    return np.array(array, dtype=bool, copy=True)


def _normalized_quaternions(value: Any, label: str) -> np.ndarray:
    result = _finite_array(value, (len(RTMW_SMPLX52_NAMES), 4), label)
    norms = np.linalg.norm(result, axis=1)
    if np.any((norms < 0.999) | (norms > 1.001)):
        raise ValueError(f"RTMW {label} contains a non-normalized quaternion")
    return result / norms[:, None]


def _validate_tree_consistency(local: np.ndarray, global_: np.ndarray) -> None:
    for index, parent in enumerate(RTMW_SMPLX52_PARENTS):
        expected = (
            local[index]
            if parent < 0
            else _quat_multiply_xyzw(global_[parent], local[index])
        )
        if abs(float(np.dot(expected, global_[index]))) < 0.999:
            name = RTMW_SMPLX52_NAMES[index]
            raise ValueError(
                f"RTMW local/global quaternions are inconsistent at {name!r}"
            )


def _quat_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = np.asarray(left, dtype=np.float64)
    rx, ry, rz, rw = np.asarray(right, dtype=np.float64)
    result = np.asarray(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        )
    )
    return result / np.linalg.norm(result)


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = np.broadcast_to(left, right.shape)
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _quat_rotate(quaternion: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    vectors = np.asarray(vectors, dtype=np.float64)
    vector = quaternion[1:]
    cross = np.cross(vector, vectors)
    return vectors + 2.0 * (quaternion[0] * cross + np.cross(vector, cross))


def _quat_conjugate(quaternion: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternion, dtype=np.float64).copy()
    result[1:] *= -1.0
    return result


def _yaw_wxyz(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _file_sha256(path: Path) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"neutral SMPL-X model does not exist: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
