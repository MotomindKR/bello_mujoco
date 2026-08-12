"""Pinned Fast-SAM gRPC source for live TWIST2 hardware references."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib
import math
from pathlib import Path
import string
import subprocess
import sys
import threading
import time
from typing import Any, Iterator

import grpc
from google.protobuf import empty_pb2
import numpy as np

from bello_mujoco.hardware.smplx_retarget import (
    GmrBelloRetargeter,
    GmrLiveRetargetConfig,
    LiveGmrReferenceSource,
    SMPLX_BODY_JOINT_NAMES,
    SMPLX_BODY_JOINT_PARENTS,
    SMPLX_GMR_BODY_NAMES,
    SmplxTargetFrame,
    load_default_smplx_standing,
)
from bello_mujoco.hardware.twist2 import (
    HardwareTwist2Config,
    HardwareTwist2PolicyAdapter,
    ReferenceUnavailable,
)
from bello_mujoco.repository import repository_root


FASTSAM_REVISION = "bc5e5ffb7dbab305bf24373b06535d285692a00e"
_ACTIVE_TRACKING_STATES = frozenset(("ACQUIRED", "TRACKING", "REACQUIRED"))
_ALL_TRACKING_STATES = _ACTIVE_TRACKING_STATES | frozenset(
    ("UNLOCKED", "OCCLUDED", "AMBIGUOUS", "LOST")
)
_EXPECTED_CONVERTER = "learned_mhr2smpl_multiview_single_view"
_EXPECTED_GLOBAL_QUATERNIONS = "smplx_joint_frame_to_camera"
_EXPECTED_LOCAL_QUATERNIONS = "joint_to_smplx_parent"


@dataclass(frozen=True)
class FastSamGrpcConfig:
    """Transport, semantics, and freshness checks for the SMPL-X stream."""

    endpoint: str
    neutral_model_path: Path
    tls_root_certificate_path: Path | None
    tls_server_name_override: str | None
    connect_timeout_seconds: float
    rpc_timeout_seconds: float
    min_tracking_confidence: float
    min_joint_confidence: float
    max_pipeline_latency_ms: float
    transport_latency_budget_ms: float
    camera_to_world_quaternion_xyzw: tuple[float, float, float, float]
    align_initial_heading: bool
    align_initial_floor: bool

    @classmethod
    def from_hardware(cls, config: HardwareTwist2Config) -> FastSamGrpcConfig:
        if config.smplx_neutral_model_path is None:
            raise ValueError("smplx_neutral_model_path is required for TWIST2 hardware")
        return cls(
            endpoint=config.fastsam_endpoint,
            neutral_model_path=config.smplx_neutral_model_path,
            tls_root_certificate_path=config.fastsam_tls_root_certificate_path,
            tls_server_name_override=config.fastsam_tls_server_name_override,
            connect_timeout_seconds=config.fastsam_connect_timeout_seconds,
            rpc_timeout_seconds=config.fastsam_rpc_timeout_seconds,
            min_tracking_confidence=config.fastsam_min_tracking_confidence,
            min_joint_confidence=config.fastsam_min_joint_confidence,
            max_pipeline_latency_ms=config.fastsam_max_pipeline_latency_ms,
            transport_latency_budget_ms=config.fastsam_transport_latency_budget_ms,
            camera_to_world_quaternion_xyzw=(
                config.fastsam_camera_to_world_quaternion_xyzw
            ),
            align_initial_heading=config.fastsam_align_initial_heading,
            align_initial_floor=config.fastsam_align_initial_floor,
        )

    def __post_init__(self) -> None:
        if not self.endpoint or "://" in self.endpoint:
            raise ValueError("Fast-SAM endpoint must be a gRPC host:port target")
        for name, value in (
            ("connect_timeout_seconds", self.connect_timeout_seconds),
            ("rpc_timeout_seconds", self.rpc_timeout_seconds),
            ("max_pipeline_latency_ms", self.max_pipeline_latency_ms),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"Fast-SAM {name} must be positive and finite")
        if (
            not math.isfinite(self.transport_latency_budget_ms)
            or self.transport_latency_budget_ms < 0.0
        ):
            raise ValueError(
                "Fast-SAM transport_latency_budget_ms must be finite and nonnegative"
            )
        for name, value in (
            ("min_tracking_confidence", self.min_tracking_confidence),
            ("min_joint_confidence", self.min_joint_confidence),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Fast-SAM {name} must be finite and in [0, 1]")
        quaternion = np.asarray(self.camera_to_world_quaternion_xyzw, dtype=np.float64)
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            raise ValueError("Fast-SAM camera rotation must be a finite quaternion")
        if not 0.999 <= float(np.linalg.norm(quaternion)) <= 1.001:
            raise ValueError("Fast-SAM camera rotation quaternion must be normalized")
        if self.tls_server_name_override is not None and (
            self.tls_root_certificate_path is None
        ):
            raise ValueError("Fast-SAM TLS server-name override requires a CA file")


class FastSamCoordinateAdapter:
    """Map OpenCV camera poses into the offline retargeter's world frame."""

    def __init__(self, config: FastSamGrpcConfig) -> None:
        xyzw = np.asarray(config.camera_to_world_quaternion_xyzw, dtype=np.float64)
        self._camera_to_world = xyzw[[3, 0, 1, 2]]
        self._align_heading = config.align_initial_heading
        self._align_floor = config.align_initial_floor
        self.reset()

    def reset(self) -> None:
        self._track_id: int | None = None
        self._origin: np.ndarray | None = None
        self._heading: np.ndarray | None = None
        self._previous_quaternions: np.ndarray | None = None

    def transform(
        self,
        positions_camera: np.ndarray,
        quaternions_camera_xyzw: np.ndarray,
        *,
        track_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._track_id is not None and track_id != self._track_id:
            self.reset()
        self._track_id = track_id
        positions = _quat_rotate(self._camera_to_world, positions_camera)
        camera_quaternions = quaternions_camera_xyzw[:, (3, 0, 1, 2)]
        quaternions = _quat_multiply(self._camera_to_world, camera_quaternions)
        pelvis = SMPLX_BODY_JOINT_NAMES.index("pelvis")
        if self._origin is None:
            origin = np.zeros(3, dtype=np.float64)
            origin[:2] = positions[pelvis, :2]
            if self._align_floor:
                feet = [
                    SMPLX_BODY_JOINT_NAMES.index(name)
                    for name in ("left_foot", "right_foot")
                ]
                origin[2] = float(np.min(positions[feet, 2]))
            self._origin = origin
            yaw = _yaw_wxyz(quaternions[pelvis]) if self._align_heading else 0.0
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


class FastSamSmplxStream:
    """Latest-only, validated SMPL-X stream backed by SubscribeEvents."""

    def __init__(self, config: FastSamGrpcConfig) -> None:
        self.config = config
        self._neutral = load_default_smplx_standing(config.neutral_model_path)
        self.neutral_model_sha256 = _file_sha256(config.neutral_model_path)
        self._coordinates = FastSamCoordinateAdapter(config)
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._generation = 0
        self._consumed_generation = 0
        self._payload: SmplxTargetFrame | BaseException | None = None
        self._terminal_error: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._channel: grpc.Channel | None = None
        self._call: Any | None = None
        self._stub: Any | None = None
        self._proto: Any | None = None
        self.stream_instance_id: str | None = None
        self.stream_config_sha256: str | None = None
        self.inference_profile: str | None = None
        self.inference_mode: str | None = None
        self._last_frame_id: int | None = None
        self._last_source_timestamp_ns: int | None = None
        self._active_track_id: int | None = None

    def __enter__(self) -> FastSamSmplxStream:
        if self._thread is not None:
            raise RuntimeError("Fast-SAM stream is already running")
        skeleton_pb2, skeleton_pb2_grpc = import_pinned_fastsam_proto()
        self._proto = skeleton_pb2
        self._stop.clear()
        self._channel = self._create_channel()
        try:
            grpc.channel_ready_future(self._channel).result(
                timeout=self.config.connect_timeout_seconds
            )
            self._stub = skeleton_pb2_grpc.SkeletonServiceStub(self._channel)
            info = self._stub.GetStreamInfo(
                empty_pb2.Empty(), timeout=self.config.rpc_timeout_seconds
            )
            _validate_stream_info(info)
            self.stream_instance_id = info.stream_instance_id
            self.stream_config_sha256 = info.config_sha256
            self.inference_profile = info.inference_profile
            self.inference_mode = info.inference_mode
            status = self._stub.GetStatus(
                empty_pb2.Empty(), timeout=self.config.rpc_timeout_seconds
            )
            _validate_service_status(status)
            self._call = self._stub.SubscribeEvents(skeleton_pb2.SubscribeRequest())
            self._thread = threading.Thread(
                target=self._consume,
                name="bello-fastsam-grpc",
                daemon=True,
            )
            self._thread.start()
        except BaseException:
            self._close_transport()
            raise
        return self

    def __exit__(self, *_exception: object) -> bool:
        self._stop.set()
        call = self._call
        if call is not None:
            call.cancel()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=3.0)
            if thread.is_alive():
                self._close_transport()
                raise RuntimeError("Fast-SAM gRPC worker did not stop")
        self._thread = None
        self._close_transport()
        return False

    def neutral(self) -> SmplxTargetFrame:
        return self._neutral

    def wait_next(
        self,
        *,
        after_sequence: int | None,
        timeout_seconds: float,
    ) -> SmplxTargetFrame | None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise ValueError("Fast-SAM poll timeout must be positive and finite")
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while self._generation <= self._consumed_generation:
                if self._terminal_error is not None:
                    raise RuntimeError("Fast-SAM gRPC stream failed") from (
                        self._terminal_error
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(remaining)
            self._consumed_generation = self._generation
            payload = self._payload
        if isinstance(payload, ReferenceUnavailable):
            raise payload
        if isinstance(payload, BaseException):
            raise RuntimeError("Fast-SAM gRPC stream failed") from payload
        if payload is None:
            raise RuntimeError("Fast-SAM stream published an empty event")
        if after_sequence is not None and payload.sequence <= after_sequence:
            raise ReferenceUnavailable(
                "Fast-SAM frame sequence did not advance: "
                f"{payload.sequence} <= {after_sequence}"
            )
        return payload

    def _create_channel(self) -> grpc.Channel:
        options: list[tuple[str, int | str]] = [
            ("grpc.max_receive_message_length", 8 * 1024 * 1024),
            ("grpc.max_send_message_length", 1024 * 1024),
            ("grpc.keepalive_time_ms", 10_000),
            ("grpc.keepalive_timeout_ms", 3_000),
            ("grpc.keepalive_permit_without_calls", 1),
        ]
        certificate_path = self.config.tls_root_certificate_path
        if certificate_path is None:
            return grpc.insecure_channel(self.config.endpoint, options=options)
        certificate = certificate_path.expanduser().resolve()
        if not certificate.is_file():
            raise FileNotFoundError(
                f"Fast-SAM TLS CA file does not exist: {certificate}"
            )
        override = self.config.tls_server_name_override
        if override is not None:
            options.append(("grpc.ssl_target_name_override", override))
        credentials = grpc.ssl_channel_credentials(
            root_certificates=certificate.read_bytes()
        )
        return grpc.secure_channel(self.config.endpoint, credentials, options=options)

    def _consume(self) -> None:
        try:
            assert self._call is not None
            for event in self._call:
                if self._stop.is_set():
                    return
                event_type = event.WhichOneof("event")
                if event_type == "tracking":
                    self._consume_tracking(event.tracking)
                elif event_type == "skeleton":
                    try:
                        frame = self._parse_frame(event.skeleton)
                    except ReferenceUnavailable as error:
                        self._publish(error)
                    else:
                        self._publish(frame)
                else:
                    raise ValueError("Fast-SAM StreamEvent has no payload")
            if not self._stop.is_set():
                raise RuntimeError("Fast-SAM gRPC stream ended")
        except grpc.RpcError as error:
            if not self._stop.is_set():
                detail = error.details() or "RPC failed"
                self._set_terminal(
                    RuntimeError(f"Fast-SAM gRPC {error.code().name}: {detail}")
                )
        except BaseException as error:
            if not self._stop.is_set():
                self._set_terminal(error)

    def _consume_tracking(self, message: Any) -> None:
        if message.stream_instance_id != self.stream_instance_id:
            raise ValueError("Fast-SAM tracking event belongs to another stream")
        state = message.state
        if state not in _ALL_TRACKING_STATES:
            raise ValueError(f"Fast-SAM emitted unknown tracking state {state!r}")
        if state in ("ACQUIRED", "REACQUIRED"):
            self._coordinates.reset()
            self._active_track_id = (
                int(message.track_id) if message.HasField("track_id") else None
            )
            self._publish(
                ReferenceUnavailable(
                    f"Fast-SAM tracking state is {state}; awaiting its first pose"
                )
            )
        elif state not in _ACTIVE_TRACKING_STATES:
            self._coordinates.reset()
            self._active_track_id = None
            reason = message.reason or "no reason supplied"
            self._publish(
                ReferenceUnavailable(f"Fast-SAM tracking state is {state}: {reason}")
            )

    def _parse_frame(self, message: Any) -> SmplxTargetFrame:
        received_at = time.monotonic()
        frame_id = int(message.frame_id)
        source_timestamp_ns = int(message.source_timestamp_ns)
        submit_ns = int(message.submit_monotonic_ns)
        output_ns = int(message.output_monotonic_ns)
        if source_timestamp_ns <= 0 or submit_ns <= 0 or output_ns < submit_ns:
            raise ReferenceUnavailable("Fast-SAM frame timestamps are invalid")
        if self._last_frame_id is not None and frame_id <= self._last_frame_id:
            raise ReferenceUnavailable("Fast-SAM frame id did not advance")
        if (
            self._last_source_timestamp_ns is not None
            and source_timestamp_ns <= self._last_source_timestamp_ns
        ):
            raise ReferenceUnavailable("Fast-SAM source timestamp did not advance")
        state = message.tracking_state
        if state not in _ACTIVE_TRACKING_STATES:
            raise ReferenceUnavailable(
                f"Fast-SAM skeleton tracking state is not active: {state!r}"
            )
        if not message.HasField("track_id"):
            raise ReferenceUnavailable("Fast-SAM active skeleton has no track id")
        track_id = int(message.track_id)
        if self._active_track_id is not None and track_id != self._active_track_id:
            self._coordinates.reset()
        self._active_track_id = track_id
        positions = _finite_array(
            message.joint_positions, (len(SMPLX_BODY_JOINT_NAMES), 3), "positions"
        )
        root = _finite_array(message.root_translation, (3,), "root translation")
        if not np.allclose(root, positions[0], atol=1.0e-4, rtol=0.0):
            raise ReferenceUnavailable(
                "Fast-SAM root translation differs from the pelvis joint"
            )
        local_quaternions = _finite_array(
            message.local_joint_quaternions,
            (len(SMPLX_BODY_JOINT_NAMES), 4),
            "local quaternions",
        )
        global_quaternions = _finite_array(
            message.global_joint_quaternions,
            (len(SMPLX_BODY_JOINT_NAMES), 4),
            "global quaternions",
        )
        for label, quaternions in (
            ("local", local_quaternions),
            ("global", global_quaternions),
        ):
            norms = np.linalg.norm(quaternions, axis=1)
            if np.any((norms < 0.999) | (norms > 1.001)):
                raise ReferenceUnavailable(
                    f"Fast-SAM {label} joint quaternion is not normalized"
                )
            quaternions /= norms[:, None]
        confidence = _finite_array(
            message.joint_confidence,
            (len(SMPLX_BODY_JOINT_NAMES),),
            "joint confidence",
        )
        tracking_confidence = float(message.tracking_confidence)
        if not math.isfinite(tracking_confidence) or not (
            0.0 <= tracking_confidence <= 1.0
        ):
            raise ReferenceUnavailable("Fast-SAM tracking confidence is invalid")
        if tracking_confidence < self.config.min_tracking_confidence:
            raise ReferenceUnavailable(
                "Fast-SAM tracking confidence is too low: "
                f"{tracking_confidence:.3f} < {self.config.min_tracking_confidence:.3f}"
            )
        if np.any((confidence < 0.0) | (confidence > 1.0)):
            raise ReferenceUnavailable("Fast-SAM joint confidence is outside [0, 1]")
        required = np.asarray(
            [SMPLX_BODY_JOINT_NAMES.index(name) for name in SMPLX_GMR_BODY_NAMES]
        )
        minimum = float(np.min(confidence[required]))
        if minimum < self.config.min_joint_confidence:
            name = SMPLX_GMR_BODY_NAMES[int(np.argmin(confidence[required]))]
            raise ReferenceUnavailable(
                f"Fast-SAM joint {name!r} confidence is too low: {minimum:.3f}"
            )
        pipeline_latency_ms = float(message.pipeline_latency_ms)
        rectification_ms = float(message.rectification_ms)
        conversion_ms = (
            float(message.conversion_ms) if message.HasField("conversion_ms") else None
        )
        timings = [pipeline_latency_ms, rectification_ms]
        if conversion_ms is not None:
            timings.append(conversion_ms)
        if not all(math.isfinite(value) and value >= 0.0 for value in timings):
            raise ReferenceUnavailable("Fast-SAM frame timings are invalid")
        if pipeline_latency_ms > self.config.max_pipeline_latency_ms:
            raise ReferenceUnavailable(
                f"Fast-SAM pipeline latency is too high: {pipeline_latency_ms:.1f} ms"
            )
        timestamp_latency_ms = (output_ns - submit_ns) / 1.0e6
        if abs(timestamp_latency_ms - pipeline_latency_ms) > 1.0:
            raise ReferenceUnavailable(
                "Fast-SAM latency scalar disagrees with its monotonic timestamps"
            )
        positions, quaternions = self._coordinates.transform(
            positions, global_quaternions, track_id=track_id
        )
        lookup = dict(zip(SMPLX_BODY_JOINT_NAMES, positions, strict=True))
        quaternion_lookup = dict(zip(SMPLX_BODY_JOINT_NAMES, quaternions, strict=True))
        self._last_frame_id = frame_id
        self._last_source_timestamp_ns = source_timestamp_ns
        source_age_ms = math.ceil(
            pipeline_latency_ms + self.config.transport_latency_budget_ms
        )
        return SmplxTargetFrame(
            transforms={
                name: (lookup[name].copy(), quaternion_lookup[name].copy())
                for name in SMPLX_GMR_BODY_NAMES
            },
            sequence=frame_id,
            source_timestamp_ns=source_timestamp_ns,
            source_age_ms=source_age_ms,
            received_at=received_at,
        )

    def _publish(self, payload: SmplxTargetFrame | BaseException) -> None:
        with self._condition:
            self._payload = payload
            self._generation += 1
            self._condition.notify_all()

    def _set_terminal(self, error: BaseException) -> None:
        with self._condition:
            self._terminal_error = error
            self._payload = error
            self._generation += 1
            self._condition.notify_all()

    def _close_transport(self) -> None:
        channel = self._channel
        self._call = None
        self._stub = None
        self._channel = None
        if channel is not None:
            channel.close()


@contextmanager
def fastsam_reference_source(
    adapter: HardwareTwist2PolicyAdapter,
    config: HardwareTwist2Config,
) -> Iterator[LiveGmrReferenceSource]:
    """Construct the pinned gRPC -> SMPL-X -> GMR hardware source."""

    grpc_config = FastSamGrpcConfig.from_hardware(config)
    stream = FastSamSmplxStream(grpc_config)
    with stream:
        retargeter = GmrBelloRetargeter(
            adapter.env,
            adapter.metadata,
            GmrLiveRetargetConfig(human_height_m=config.smplx_human_height_m),
        )
        with LiveGmrReferenceSource(stream, retargeter) as source:
            print(
                {
                    "fastsam_connection_preflight": "passed",
                    "endpoint": grpc_config.endpoint,
                    "stream_instance_id": stream.stream_instance_id,
                    "stream_config_sha256": stream.stream_config_sha256,
                    "inference_profile": stream.inference_profile,
                    "inference_mode": stream.inference_mode,
                    "body_representation": "smplx_body_22",
                    "neutral_model_sha256": stream.neutral_model_sha256,
                    "live_target_check": "deferred_until_MOTION",
                }
            )
            yield source


def require_pinned_fastsam_revision() -> Path:
    root = repository_root()
    submodule = root / "deps" / "Fast-SAM-3D-Body"
    gitlink = _git_output(
        root, "ls-tree", "HEAD", "--", "deps/Fast-SAM-3D-Body"
    ).split()
    if len(gitlink) < 3 or gitlink[1] != "commit":
        raise RuntimeError("deps/Fast-SAM-3D-Body is not a pinned Git submodule")
    current = _git_output(submodule, "rev-parse", "HEAD")
    if gitlink[2] != FASTSAM_REVISION or current != FASTSAM_REVISION:
        raise ValueError(
            "Fast-SAM gitlink and checkout must both equal the reviewed SMPL-X "
            f"revision {FASTSAM_REVISION}"
        )
    if _git_output(submodule, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("hardware Fast-SAM submodule must be clean")
    return submodule


def import_pinned_fastsam_proto() -> tuple[Any, Any]:
    submodule = require_pinned_fastsam_revision()
    root_string = str(submodule)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    skeleton_pb2 = importlib.import_module("fastsam_service.proto.skeleton_pb2")
    skeleton_pb2_grpc = importlib.import_module(
        "fastsam_service.proto.skeleton_pb2_grpc"
    )
    for module in (skeleton_pb2, skeleton_pb2_grpc):
        path = Path(module.__file__).resolve()
        try:
            path.relative_to(submodule.resolve())
        except ValueError as error:
            raise RuntimeError(
                "Fast-SAM protobuf module was not imported from the pinned submodule"
            ) from error
    return skeleton_pb2, skeleton_pb2_grpc


def _validate_stream_info(message: Any) -> None:
    if int(message.schema_version) != 1:
        raise ValueError("Fast-SAM schema_version must be 1")
    if not message.stream_instance_id:
        raise ValueError("Fast-SAM stream has no instance id")
    if len(message.config_sha256) != 64 or any(
        character not in string.hexdigits.lower() for character in message.config_sha256
    ):
        raise ValueError("Fast-SAM stream has an invalid config SHA-256")
    if tuple(message.joint_names) != SMPLX_BODY_JOINT_NAMES:
        raise ValueError("Fast-SAM stream has an unexpected SMPL-X joint order")
    parents = np.asarray(message.joint_parents, dtype=np.int32)
    if not np.array_equal(parents, SMPLX_BODY_JOINT_PARENTS):
        raise ValueError("Fast-SAM stream has an unexpected SMPL-X hierarchy")
    exact = {
        "root_index": (int(message.root_index), 0),
        "coordinate_frame": (message.coordinate_frame, "camera_opencv"),
        "quaternion_order": (message.quaternion_order, "xyzw"),
        "global_quaternion_semantics": (
            message.global_quaternion_semantics,
            _EXPECTED_GLOBAL_QUATERNIONS,
        ),
        "local_quaternion_semantics": (
            message.local_quaternion_semantics,
            _EXPECTED_LOCAL_QUATERNIONS,
        ),
        "position_units": (message.position_units, "meters"),
        "joint_set": (message.joint_set, "body"),
        "body_representation": (message.body_representation, "smplx_body_22"),
        "converter": (message.converter, _EXPECTED_CONVERTER),
    }
    for name, (actual, expected) in exact.items():
        if actual != expected:
            raise ValueError(f"Fast-SAM {name} is {actual!r}; expected {expected!r}")
    if message.inference_mode not in ("body", "full"):
        raise ValueError("Fast-SAM inference mode must be 'body' or 'full'")
    if not message.inference_profile:
        raise ValueError("Fast-SAM stream has no inference profile")


def _validate_service_status(message: Any) -> None:
    if message.terminal_failure:
        raise RuntimeError(f"Fast-SAM server failed: {message.terminal_failure}")
    unavailable = [
        name
        for name, available in (
            ("ready", message.ready),
            ("source", message.source_running),
            ("inference", message.inference_running),
            ("tracking", message.tracking_enabled),
        )
        if not available
    ]
    if unavailable:
        raise RuntimeError("Fast-SAM server is not ready: " + ", ".join(unavailable))


def _finite_array(values: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size != math.prod(shape):
        raise ReferenceUnavailable(
            f"Fast-SAM {label} has {array.size} values; expected {math.prod(shape)}"
        )
    result = np.array(array.reshape(shape), copy=True, order="C")
    if not np.all(np.isfinite(result)):
        raise ReferenceUnavailable(f"Fast-SAM {label} contains non-finite values")
    return result


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
    vector_part = quaternion[1:]
    first = np.cross(vector_part, vectors)
    return vectors + 2.0 * (quaternion[0] * first + np.cross(vector_part, first))


def _yaw_wxyz(quaternion: np.ndarray) -> float:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _git_output(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"git {' '.join(arguments)} failed" + (f": {detail}" if detail else "")
        )
    return result.stdout.strip()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
