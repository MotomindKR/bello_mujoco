"""Pinned RTMW gRPC source for locomotion upper-body control."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import importlib
import importlib.util
import math
from pathlib import Path
import string
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterator

import grpc
from google.protobuf import empty_pb2
import numpy as np

from bello_mujoco.fastsam_conditioning import (
    FASTSAM_ROOT_CONDITIONING_VERSION,
    CausalFootConditionedRoot,
    FootConditionedRootConfig,
)
from bello_mujoco.hardware.smplx_retarget import (
    GmrBelloRetargeter,
    GmrLiveRetargetConfig,
    LiveGmrReferenceSource,
    SMPLX_BODY_JOINT_NAMES,
    SMPLX_GMR_BODY_NAMES,
    SmplxTargetFrame,
    load_smplx_standing_sample,
    repository_gmr_source_identity,
)
from bello_mujoco.hardware.twist2 import (
    ReferenceDiscontinuity,
    ReferenceUnavailable,
)
from bello_mujoco.repository import repository_root


RTMW_SCHEMA_VERSION = 1
RTMW_REVISION = "026536040645c24ddc676d80d1d0da188f4021e8"
RTMW_REPRESENTATION = "smplx52_landmark_fit"
RTMW_SCHEMA_ID = "smplx_55_without_jaw_and_eyes_v1"
RTMW_ROTATION_SEMANTICS = (
    "derived_tree_consistent_landmark_ik;bone_length_shape_and_axial_twist_approximate"
)
RTMW_COORDINATE_SPACE = "camera_opencv_x_right_y_down_z_forward"
RTMW_SMPLX52_NAMES = SMPLX_BODY_JOINT_NAMES + tuple(
    f"{side}_{finger}{joint}"
    for side in ("left", "right")
    for finger in ("index", "middle", "pinky", "ring", "thumb")
    for joint in (1, 2, 3)
)
RTMW_SMPLX52_PARENTS = (
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    22,
    23,
    20,
    25,
    26,
    20,
    28,
    29,
    20,
    31,
    32,
    20,
    34,
    35,
    21,
    37,
    38,
    21,
    40,
    41,
    21,
    43,
    44,
    21,
    46,
    47,
    21,
    49,
    50,
)
_REQUIRED_ROTATION_NAMES = (
    "pelvis",
    "spine3",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
)


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
    max_pipeline_latency_ms: float
    transport_latency_budget_ms: float
    max_source_age_ms: int
    camera_to_world_quaternion_xyzw: tuple[float, float, float, float]
    align_initial_heading: bool
    align_initial_floor: bool
    root_conditioning: FootConditionedRootConfig = field(
        default_factory=FootConditionedRootConfig
    )

    @classmethod
    def from_hardware(cls, config: Any) -> RtmwGrpcConfig:
        if config.smplx_neutral_model_path is None:
            raise ValueError(
                "smplx_neutral_model_path is required for live RTMW hardware pose"
            )
        return cls(
            endpoint=config.rtmw_endpoint,
            neutral_model_path=config.smplx_neutral_model_path,
            tls_root_certificate_path=config.rtmw_tls_root_certificate_path,
            tls_server_name_override=config.rtmw_tls_server_name_override,
            connect_timeout_seconds=config.rtmw_connect_timeout_seconds,
            rpc_timeout_seconds=config.rtmw_rpc_timeout_seconds,
            human_height_m=config.rtmw_human_height_m,
            min_joint_confidence=config.rtmw_min_joint_confidence,
            min_rotation_confidence=config.rtmw_min_rotation_confidence,
            max_pipeline_latency_ms=config.rtmw_max_pipeline_latency_ms,
            transport_latency_budget_ms=config.rtmw_transport_latency_budget_ms,
            max_source_age_ms=config.upper_body_follow.max_reference_age_ms,
            camera_to_world_quaternion_xyzw=(
                config.rtmw_camera_to_world_quaternion_xyzw
            ),
            align_initial_heading=config.rtmw_align_initial_heading,
            align_initial_floor=config.rtmw_align_initial_floor,
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
            ("max_pipeline_latency_ms", self.max_pipeline_latency_ms),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"RTMW {name} must be positive and finite")
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


class RtmwSmplxStream:
    """Latest-only, validated RTMW SMPL-X stream backed by gRPC."""

    def __init__(
        self,
        config: RtmwGrpcConfig,
        *,
        standing_sample_loader: Callable[[Path], SmplxTargetFrame] = (
            load_smplx_standing_sample
        ),
    ) -> None:
        self.config = config
        self._standing_sample = standing_sample_loader(config.neutral_model_path)
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
        self._channel: grpc.Channel | None = None
        self._call: Any | None = None
        self._stub: Any | None = None
        self._proto: Any | None = None
        self.stream_instance_id: str | None = None
        self.stream_config_sha256: str | None = None
        self.depth_provenance: str | None = None
        self._last_frame_id: int | None = None
        self._last_source_timestamp_ns: int | None = None
        self._active_track_id: str | None = None

    def __enter__(self) -> RtmwSmplxStream:
        if self._thread is not None:
            raise RuntimeError("RTMW stream is already running")
        skeleton_pb2, skeleton_pb2_grpc = import_pinned_rtmw_proto()
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
            _validate_stream_info(info, self.config)
            self.stream_instance_id = info.stream_instance_id
            self.stream_config_sha256 = info.config_sha256
            self.depth_provenance = info.depth_provenance
            status = self._stub.GetStatus(
                empty_pb2.Empty(), timeout=self.config.rpc_timeout_seconds
            )
            _validate_service_status(status)
            self._call = self._stub.Subscribe(skeleton_pb2.SubscribeRequest())
            self._thread = threading.Thread(
                target=self._consume,
                name="bello-rtmw-grpc",
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
                raise RuntimeError("RTMW gRPC worker did not stop")
        self._thread = None
        self._close_transport()
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
            ("grpc.keepalive_time_ms", 10_000),
            ("grpc.keepalive_timeout_ms", 3_000),
            ("grpc.keepalive_permit_without_calls", 1),
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

    def _consume(self) -> None:
        try:
            assert self._call is not None
            for message in self._call:
                if self._stop.is_set():
                    return
                try:
                    frame = self._parse_frame(message)
                except (ReferenceDiscontinuity, ReferenceUnavailable) as error:
                    self._publish(error)
                else:
                    self._publish(frame)
            if not self._stop.is_set():
                raise RuntimeError("RTMW gRPC stream ended")
        except grpc.RpcError as error:
            if not self._stop.is_set():
                detail = error.details() or "RPC failed"
                self._set_terminal(
                    RuntimeError(f"RTMW gRPC {error.code().name}: {detail}")
                )
        except BaseException as error:
            if not self._stop.is_set():
                self._set_terminal(error)

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
            [RTMW_SMPLX52_NAMES.index(name) for name in SMPLX_GMR_BODY_NAMES]
        )
        if not np.all(valid_mask[required]):
            name = SMPLX_GMR_BODY_NAMES[int(np.flatnonzero(~valid_mask[required])[0])]
            raise ReferenceUnavailable(f"RTMW joint {name!r} has no valid position")
        minimum = float(np.min(confidence[required]))
        if minimum < self.config.min_joint_confidence:
            name = SMPLX_GMR_BODY_NAMES[int(np.argmin(confidence[required]))]
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
        for name in _REQUIRED_ROTATION_NAMES:
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
        source_quaternions = global_quaternions.copy()
        for index in required:
            if rotation_valid[index]:
                continue
            ancestor = RTMW_SMPLX52_PARENTS[index]
            while ancestor >= 0 and not rotation_valid[ancestor]:
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
                for name in SMPLX_GMR_BODY_NAMES
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
def rtmw_locomotion_reference_source(
    adapter: Any,
    config: Any,
) -> Iterator[LiveGmrReferenceSource]:
    """Construct pinned RTMW gRPC -> GMR locomotion references."""

    rtmw_root = require_pinned_rtmw_revision()
    validate_pinned_rtmw_contract(rtmw_root)
    source_config = RtmwGrpcConfig.from_hardware(config)
    stream = RtmwSmplxStream(source_config)
    with stream:
        retargeter = GmrBelloRetargeter(
            adapter.env,
            repository_gmr_source_identity(),
            GmrLiveRetargetConfig(human_height_m=source_config.human_height_m),
            validate_reference=adapter.validate_upper_body_reference,
        )
        with LiveGmrReferenceSource(stream, retargeter) as source:
            print(
                {
                    "rtmw_connection_preflight": "passed",
                    "endpoint": source_config.endpoint,
                    "stream_instance_id": stream.stream_instance_id,
                    "stream_config_sha256": stream.stream_config_sha256,
                    "depth_provenance": stream.depth_provenance,
                    "rtmw_revision": RTMW_REVISION,
                    "schema_version": RTMW_SCHEMA_VERSION,
                    "representation": RTMW_REPRESENTATION,
                    "neutral_model_sha256": stream.neutral_model_sha256,
                    "root_conditioning": FASTSAM_ROOT_CONDITIONING_VERSION,
                    "live_target_check": "required_before_hardware_connection",
                }
            )
            yield source


def require_pinned_rtmw_revision() -> Path:
    """Require the reviewed RTMW source checkout pinned by this repository."""

    root = repository_root()
    relative = "deps/RTMW-SMPLX-Teleop"
    submodule = root / relative
    gitlink = _git_output(root, "ls-tree", "HEAD", "--", relative).split()
    if len(gitlink) < 3 or gitlink[1] != "commit":
        raise RuntimeError(f"{relative} is not a pinned Git submodule")
    current = _git_output(submodule, "rev-parse", "HEAD")
    if gitlink[2] != RTMW_REVISION or current != RTMW_REVISION:
        raise ValueError(
            "RTMW gitlink and checkout must both equal the reviewed protocol "
            f"revision {RTMW_REVISION}"
        )
    if _git_output(submodule, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("hardware RTMW submodule must be clean")
    return submodule


def validate_pinned_rtmw_contract(submodule: Path) -> None:
    """Match the consumer's structural constants to the pinned producer source."""

    module_name = "_bello_pinned_rtmw_contracts"
    contract_path = submodule / "src" / "rtmw_teleop" / "contracts.py"
    spec = importlib.util.spec_from_file_location(module_name, contract_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the pinned RTMW contract module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    if tuple(module.SMPLX52_NAMES) != RTMW_SMPLX52_NAMES:
        raise ValueError("pinned RTMW SMPL-X joint order is incompatible")
    if tuple(int(value) for value in module.SMPLX52_PARENTS) != RTMW_SMPLX52_PARENTS:
        raise ValueError("pinned RTMW SMPL-X hierarchy is incompatible")


def import_pinned_rtmw_proto() -> tuple[Any, Any]:
    """Import generated protobuf modules only from the reviewed submodule."""

    submodule = require_pinned_rtmw_revision()
    source_root = submodule / "src"
    source_string = str(source_root)
    if source_string not in sys.path:
        sys.path.insert(0, source_string)
    skeleton_pb2 = importlib.import_module("rtmw_teleop.proto.skeleton_pb2")
    skeleton_pb2_grpc = importlib.import_module("rtmw_teleop.proto.skeleton_pb2_grpc")
    for module in (skeleton_pb2, skeleton_pb2_grpc):
        path = Path(module.__file__).resolve()
        try:
            path.relative_to(submodule.resolve())
        except ValueError as error:
            raise RuntimeError(
                "RTMW protobuf module was not imported from the pinned submodule"
            ) from error
    return skeleton_pb2, skeleton_pb2_grpc


def _validate_stream_info(message: Any, config: RtmwGrpcConfig) -> None:
    if int(message.schema_version) != RTMW_SCHEMA_VERSION:
        raise ValueError(f"RTMW schema_version must be {RTMW_SCHEMA_VERSION}")
    if not message.stream_instance_id:
        raise ValueError("RTMW stream has no instance id")
    if len(message.config_sha256) != 64 or any(
        character not in string.hexdigits.lower() for character in message.config_sha256
    ):
        raise ValueError("RTMW stream has an invalid config SHA-256")
    if tuple(message.joint_names) != RTMW_SMPLX52_NAMES:
        raise ValueError("RTMW stream has an unexpected SMPL-X joint order")
    if tuple(message.joint_parents) != RTMW_SMPLX52_PARENTS:
        raise ValueError("RTMW stream has an unexpected SMPL-X hierarchy")
    exact = {
        "root_index": (int(message.root_index), 0),
        "coordinate_space": (message.coordinate_space, RTMW_COORDINATE_SPACE),
        "quaternion_order": (message.quaternion_order, "xyzw"),
        "rotation_semantics": (
            message.rotation_semantics,
            RTMW_ROTATION_SEMANTICS,
        ),
        "position_units": (message.position_units, "meters"),
        "representation": (message.representation, RTMW_REPRESENTATION),
        "schema_id": (message.schema_id, RTMW_SCHEMA_ID),
    }
    for name, (actual, expected) in exact.items():
        if actual != expected:
            raise ValueError(f"RTMW {name} is {actual!r}; expected {expected!r}")
    if not message.metric_scale:
        raise ValueError("RTMW hardware pose must have metric_scale=true")
    if not message.global_translation:
        raise ValueError("RTMW hardware pose must have global_translation=true")
    if message.global_depth_is_measured:
        raise ValueError(
            "RTMW monocular hardware pose must declare global_depth_is_measured=false"
        )
    if message.depth_provenance not in {
        "calibrated_camera+assumed_subject_height",
        "fixed_root_depth_assumption",
    }:
        raise ValueError(
            "RTMW hardware pose must use calibrated height-prior or fixed-depth "
            "reconstruction"
        )
    subject_height_m = float(message.subject_height_m)
    if not math.isfinite(subject_height_m) or not math.isclose(
        subject_height_m, config.human_height_m, abs_tol=1.0e-6
    ):
        raise ValueError(
            "RTMW producer subject height does not match rtmw_human_height_m: "
            f"{subject_height_m!r} != {config.human_height_m!r}"
        )


def _validate_service_status(message: Any) -> None:
    if message.terminal_failure:
        raise RuntimeError(f"RTMW service failed: {message.terminal_failure}")
    if not message.ready:
        raise RuntimeError("RTMW service is not ready")
    if not message.source_running or not message.inference_running:
        raise RuntimeError("RTMW source or inference worker is not running")


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
