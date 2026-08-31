"""Pinned GEM-X SOMA-77 source for live SONIC hardware references."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import importlib
import math
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Iterator

import grpc
from google.protobuf import empty_pb2
import numpy as np

from bello_mujoco.fastsam_conditioning import (
    CausalFootConditionedRoot,
    FootConditionedRootConfig,
)
from bello_mujoco.hardware.smplx_retarget import (
    GmrBelloRetargeter,
    GmrLiveRetargetConfig,
    GmrSourceIdentity,
    LiveGmrReferenceSource,
    SMPLX_GMR_BODY_NAMES,
    SmplxTargetFrame,
    load_smplx_standing_sample,
)
from bello_mujoco.hardware.twist2 import (
    ReferenceDiscontinuity,
    ReferenceUnavailable,
)
from bello_mujoco.repository import repository_root


GEMX_REVISION = "ba65a308a1879c27f8184c4391b53f20d1ea86dd"
GEMX_ROOT_CONDITIONING_VERSION = "causal-support-foot-xy-v2"

_ACTIVE_TRACKING_STATES = frozenset(("ACQUIRED", "TRACKING", "REACQUIRED"))
_SOMA_TO_SMPLX_GMR = {
    "pelvis": "Hips",
    "spine3": "Chest",
    "left_hip": "LeftLeg",
    "left_knee": "LeftShin",
    "left_foot": "LeftToeBase",
    "right_hip": "RightLeg",
    "right_knee": "RightShin",
    "right_foot": "RightToeBase",
    "left_shoulder": "LeftArm",
    "left_elbow": "LeftForeArm",
    "left_wrist": "LeftHand",
    "right_shoulder": "RightArm",
    "right_elbow": "RightForeArm",
    "right_wrist": "RightHand",
}

# Reviewed from SOMA_neutral.npz at the pinned SOMA/GEM-X revision. GEM-X
# publishes global skinning-frame rotations with these joint orientations baked
# in. Right-multiplying their inverse produces the common-rest-frame global pose
# rotations expected by the existing GMR target contract.
_SOMA_NEUTRAL_GLOBAL_ROTATION_SHA256 = (
    "67eafbcd74e4dc755c4373a985db7670c983c323ca599b89403fee2fe7a0b5ba"
)
_SOMA_NEUTRAL_GLOBAL_WXYZ = {
    "Hips": (
        0.5008302925024383,
        0.4991683265916306,
        0.5008302923418055,
        0.4991683264315305,
    ),
    "Chest": (
        0.5008835152972019,
        0.4991148342199178,
        0.5008835787008873,
        0.4991149436281523,
    ),
    "LeftLeg": (
        0.4999971824478131,
        -0.5000028173759449,
        0.4999971826076469,
        -0.5000028175368426,
    ),
    "LeftShin": (
        0.5000028755688204,
        -0.4999971242542751,
        0.5000028757286523,
        -0.4999971244151749,
    ),
    "LeftToeBase": (
        0.09968099987681818,
        -0.6995394739894782,
        0.09953408209411663,
        -0.7005720441865802,
    ),
    "RightLeg": (
        0.5000028183008052,
        0.4999971818436756,
        0.5000028181399074,
        0.4999971816838418,
    ),
    "RightShin": (
        0.4999971236286938,
        0.5000028765151242,
        0.4999971234677941,
        0.5000028763552923,
    ),
    "RightToeBase": (
        0.6995390695156261,
        0.09968217376719246,
        0.7005721233082582,
        0.09953519225690008,
    ),
    "LeftArm": (
        0.7071067921989118,
        0.707106768739274,
        3.1838365846704265e-05,
        -3.186826255322174e-05,
    ),
    "LeftForeArm": (
        0.7071067920964915,
        0.7071067686340665,
        -3.409498050484701e-05,
        3.4065085985158724e-05,
    ),
    "LeftHand": (
        0.6447113599582166,
        0.7644260978609734,
        -2.1885839857092693e-05,
        5.263917140275796e-05,
    ),
    "RightArm": (
        0.7071067689973574,
        -0.7071067919361792,
        -3.192015531298841e-05,
        -3.188960218598443e-05,
    ),
    "RightForeArm": (
        0.7071067688917708,
        -0.7071067918334465,
        3.412011922322919e-05,
        3.415067020736645e-05,
    ),
    "RightHand": (
        0.7644260978609734,
        -0.6447113599582166,
        5.263917140275796e-05,
        2.1885839857092693e-05,
    ),
}

if tuple(_SOMA_TO_SMPLX_GMR) != SMPLX_GMR_BODY_NAMES:
    raise RuntimeError("GEM-X SOMA body adapter does not cover the GMR target order")
if set(_SOMA_NEUTRAL_GLOBAL_WXYZ) != set(_SOMA_TO_SMPLX_GMR.values()):
    raise RuntimeError("GEM-X SOMA rest orientations do not cover the GMR targets")


@dataclass(frozen=True)
class GemxGrpcConfig:
    """Transport, semantics, and freshness checks for GEM-X SOMA-77."""

    endpoint: str
    neutral_model_path: Path
    connect_timeout_seconds: float
    rpc_timeout_seconds: float
    min_tracking_confidence: float
    min_joint_confidence: float
    max_pipeline_latency_ms: float
    transport_latency_budget_ms: float
    camera_to_world_quaternion_xyzw: tuple[float, float, float, float]
    align_initial_heading: bool
    align_initial_floor: bool
    root_conditioning: FootConditionedRootConfig = field(
        default_factory=FootConditionedRootConfig
    )

    @classmethod
    def from_hardware(cls, config: Any) -> GemxGrpcConfig:
        if config.smplx_neutral_model_path is None:
            raise ValueError(
                "smplx_neutral_model_path is required to initialize the pinned "
                "GMR standing reference for live GEM-X SONIC"
            )
        return cls(
            endpoint=config.gemx_endpoint,
            neutral_model_path=config.smplx_neutral_model_path,
            connect_timeout_seconds=config.gemx_connect_timeout_seconds,
            rpc_timeout_seconds=config.gemx_rpc_timeout_seconds,
            min_tracking_confidence=config.gemx_min_tracking_confidence,
            min_joint_confidence=config.gemx_min_joint_confidence,
            max_pipeline_latency_ms=config.gemx_max_pipeline_latency_ms,
            transport_latency_budget_ms=config.gemx_transport_latency_budget_ms,
            camera_to_world_quaternion_xyzw=(
                config.gemx_camera_to_world_quaternion_xyzw
            ),
            align_initial_heading=config.gemx_align_initial_heading,
            align_initial_floor=config.gemx_align_initial_floor,
        )

    def __post_init__(self) -> None:
        if not self.endpoint or "://" in self.endpoint:
            raise ValueError("GEM-X endpoint must be a gRPC host:port target")
        for name, value in (
            ("connect_timeout_seconds", self.connect_timeout_seconds),
            ("rpc_timeout_seconds", self.rpc_timeout_seconds),
            ("max_pipeline_latency_ms", self.max_pipeline_latency_ms),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"GEM-X {name} must be positive and finite")
        if (
            not math.isfinite(self.transport_latency_budget_ms)
            or self.transport_latency_budget_ms < 0.0
        ):
            raise ValueError(
                "GEM-X transport_latency_budget_ms must be finite and nonnegative"
            )
        for name, value in (
            ("min_tracking_confidence", self.min_tracking_confidence),
            ("min_joint_confidence", self.min_joint_confidence),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"GEM-X {name} must be finite and in [0, 1]")
        quaternion = np.asarray(self.camera_to_world_quaternion_xyzw)
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            raise ValueError("GEM-X camera rotation must be a finite quaternion")
        if not 0.999 <= float(np.linalg.norm(quaternion)) <= 1.001:
            raise ValueError("GEM-X camera rotation quaternion must be normalized")


class GemxCoordinateAdapter:
    """Map GEM-X OpenCV camera poses into Bello's Z-up world frame."""

    def __init__(self, config: GemxGrpcConfig, soma_joint_names: tuple[str, ...]):
        xyzw = np.asarray(config.camera_to_world_quaternion_xyzw, dtype=np.float64)
        self._camera_to_world = xyzw[[3, 0, 1, 2]]
        self._joint_index = {name: index for index, name in enumerate(soma_joint_names)}
        if len(self._joint_index) != len(soma_joint_names):
            raise ValueError("GEM-X SOMA joint names must be unique")
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
        pelvis = self._joint_index["Hips"]
        if self._origin is None:
            origin = np.zeros(3, dtype=np.float64)
            origin[:2] = positions[pelvis, :2]
            if self._align_floor:
                feet = [
                    self._joint_index[name] for name in ("LeftToeBase", "RightToeBase")
                ]
                origin[2] = float(np.min(positions[feet, 2]))
            self._origin = origin
            yaw = _yaw_wxyz(quaternions[pelvis]) if self._align_heading else 0.0
            self._heading = np.asarray(
                (math.cos(-0.5 * yaw), 0.0, 0.0, math.sin(-0.5 * yaw))
            )
        assert self._origin is not None and self._heading is not None
        positions = _quat_rotate(self._heading, positions - self._origin)
        quaternions = _quat_multiply(self._heading, quaternions)
        if self._previous_quaternions is not None:
            signs = np.sum(quaternions * self._previous_quaternions, axis=1) < 0.0
            quaternions[signs] *= -1.0
        self._previous_quaternions = quaternions.copy()
        return positions, quaternions


class GemxSomaStream:
    """Latest-only validated GEM-X stream adapted to GMR body semantics."""

    def __init__(self, config: GemxGrpcConfig) -> None:
        self.config = config
        self._standing_sample = load_smplx_standing_sample(config.neutral_model_path)
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
        self._soma_joint_names: tuple[str, ...] = ()
        self._soma_joint_parents: tuple[int, ...] = ()
        self._coordinates: GemxCoordinateAdapter | None = None
        self._root_conditioner = CausalFootConditionedRoot(
            SMPLX_GMR_BODY_NAMES, config.root_conditioning
        )
        self.stream_instance_id: str | None = None
        self.nominal_fps: float | None = None
        self.frame_window: int | None = None
        self._last_frame_id: int | None = None
        self._last_source_timestamp_ns: int | None = None
        self._active_track_id: int | None = None

    def __enter__(self) -> GemxSomaStream:
        if self._thread is not None:
            raise RuntimeError("GEM-X stream is already running")
        skeleton_pb2, skeleton_pb2_grpc, constants = import_pinned_gemx_streaming()
        self._proto = skeleton_pb2
        self._soma_joint_names = tuple(constants.SOMA77_JOINT_NAMES)
        self._soma_joint_parents = tuple(constants.SOMA77_JOINT_PARENTS)
        self._coordinates = GemxCoordinateAdapter(self.config, self._soma_joint_names)
        self._stop.clear()
        self._channel = grpc.insecure_channel(
            self.config.endpoint,
            options=(("grpc.max_receive_message_length", 2 * 1024 * 1024),),
        )
        try:
            grpc.channel_ready_future(self._channel).result(
                timeout=self.config.connect_timeout_seconds
            )
            self._stub = skeleton_pb2_grpc.SkeletonServiceStub(self._channel)
            info = self._stub.GetStreamInfo(
                empty_pb2.Empty(), timeout=self.config.rpc_timeout_seconds
            )
            self._validate_stream_info(info)
            status = self._stub.GetStatus(
                empty_pb2.Empty(), timeout=self.config.rpc_timeout_seconds
            )
            self._validate_service_status(status)
            self._call = self._stub.Subscribe(skeleton_pb2.SubscribeRequest())
            self._thread = threading.Thread(
                target=self._consume, name="bello-gemx-grpc", daemon=True
            )
            self._thread.start()
        except BaseException:
            self._close_transport()
            raise
        return self

    def __exit__(self, *_exception: object) -> bool:
        self._stop.set()
        if self._call is not None:
            self._call.cancel()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                self._close_transport()
                raise RuntimeError("GEM-X gRPC worker did not stop")
        self._thread = None
        self._close_transport()
        return False

    def standing_sample(self) -> SmplxTargetFrame:
        return self._standing_sample

    def wait_next(
        self, *, after_sequence: int | None, timeout_seconds: float
    ) -> SmplxTargetFrame | None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise ValueError("GEM-X poll timeout must be positive and finite")
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while self._generation <= self._consumed_generation:
                if self._terminal_error is not None:
                    raise RuntimeError("GEM-X gRPC stream failed") from (
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
            raise RuntimeError("GEM-X gRPC stream failed") from payload
        if payload is None:
            raise RuntimeError("GEM-X stream published an empty event")
        if after_sequence is not None and payload.sequence <= after_sequence:
            raise ReferenceDiscontinuity(
                "GEM-X frame sequence did not advance: "
                f"{payload.sequence} <= {after_sequence}"
            )
        return payload

    def _validate_stream_info(self, message: Any) -> None:
        exact = {
            "schema_version": (int(message.schema_version), 1),
            "joint_names": (tuple(message.joint_names), self._soma_joint_names),
            "joint_parents": (
                tuple(message.joint_parents),
                self._soma_joint_parents,
            ),
            "root_index": (int(message.root_index), 0),
            "coordinate_frame": (message.coordinate_frame, "camera_opencv"),
            "quaternion_order": (message.quaternion_order, "xyzw"),
            "local_quaternion_semantics": (
                message.local_quaternion_semantics,
                "parent_relative",
            ),
            "global_quaternion_semantics": (
                message.global_quaternion_semantics,
                "coordinate_frame_absolute",
            ),
            "position_units": (message.position_units, "meters"),
            "body_representation": (message.body_representation, "soma_77"),
        }
        mismatches = [
            name for name, (actual, expected) in exact.items() if actual != expected
        ]
        if mismatches:
            raise ValueError(
                "GEM-X stream contract differs for: " + ", ".join(mismatches)
            )
        if not message.stream_instance_id:
            raise ValueError("GEM-X stream has no instance id")
        if not math.isfinite(message.nominal_fps) or message.nominal_fps <= 0.0:
            raise ValueError("GEM-X stream nominal_fps must be positive")
        if int(message.frame_window) != 30:
            raise ValueError("GEM-X live stream must use its 30-frame model context")
        self.stream_instance_id = message.stream_instance_id
        self.nominal_fps = float(message.nominal_fps)
        self.frame_window = int(message.frame_window)

    @staticmethod
    def _validate_service_status(message: Any) -> None:
        if message.terminal_failure:
            raise RuntimeError(f"GEM-X server failed: {message.terminal_failure}")
        unavailable = [
            name
            for name, available in (
                ("ready", message.ready),
                ("source", message.source_running),
                ("tracking", message.tracking_enabled),
            )
            if not available
        ]
        if unavailable:
            raise RuntimeError("GEM-X server is not ready: " + ", ".join(unavailable))

    def _consume(self) -> None:
        try:
            assert self._call is not None
            for message in self._call:
                if self._stop.is_set():
                    return
                try:
                    self._publish(self._parse_frame(message))
                except (ReferenceUnavailable, ReferenceDiscontinuity) as error:
                    self._publish(error)
        except BaseException as error:
            if not self._stop.is_set():
                self._set_terminal(error)

    def _parse_frame(self, message: Any) -> SmplxTargetFrame:
        received_at = time.monotonic()
        frame_id = int(message.frame_id)
        source_timestamp_ns = int(message.source_timestamp_ns)
        if frame_id < 0 or source_timestamp_ns <= 0:
            raise ReferenceDiscontinuity("GEM-X frame timestamps are invalid")
        if self._last_frame_id is not None and frame_id <= self._last_frame_id:
            raise ReferenceDiscontinuity("GEM-X frame id did not advance")
        if (
            self._last_source_timestamp_ns is not None
            and source_timestamp_ns <= self._last_source_timestamp_ns
        ):
            raise ReferenceDiscontinuity("GEM-X source timestamp did not advance")
        state = message.tracking_state
        if state not in _ACTIVE_TRACKING_STATES:
            if state in {"UNLOCKED", "LOST"}:
                self._reset_tracking_epoch(None)
                raise ReferenceDiscontinuity(
                    f"GEM-X skeleton tracking state is {state!r}"
                )
            raise ReferenceUnavailable(
                f"GEM-X skeleton tracking state is not active: {state!r}"
            )
        track_id = int(message.track_id)
        if track_id <= 0:
            raise ReferenceUnavailable("GEM-X active skeleton has no track id")
        if self._active_track_id is not None and track_id != self._active_track_id:
            previous = self._active_track_id
            self._reset_tracking_epoch(track_id)
            raise ReferenceDiscontinuity(
                f"GEM-X track changed from {previous} to {track_id}"
            )
        self._active_track_id = track_id
        joint_count = len(self._soma_joint_names)
        positions = _finite_array(
            message.joint_positions, (joint_count, 3), "positions"
        )
        root = _finite_array(message.root_translation, (3,), "root translation")
        if not np.allclose(root, positions[0], atol=1.0e-4, rtol=0.0):
            raise ReferenceUnavailable(
                "GEM-X root translation differs from the Hips joint"
            )
        quaternions = _finite_array(
            message.global_joint_quaternions,
            (joint_count, 4),
            "global quaternions",
        )
        norms = np.linalg.norm(quaternions, axis=1)
        if np.any((norms < 0.995) | (norms > 1.005)):
            raise ReferenceUnavailable(
                "GEM-X global joint quaternion is not normalized"
            )
        quaternions /= norms[:, None]
        confidence = _finite_array(
            message.joint_confidence, (joint_count,), "joint confidence"
        )
        if np.any((confidence < 0.0) | (confidence > 1.0)):
            raise ReferenceUnavailable("GEM-X joint confidence is outside [0, 1]")
        tracking_confidence = float(message.association_confidence)
        if (
            not math.isfinite(tracking_confidence)
            or not 0.0 <= tracking_confidence <= 1.0
        ):
            raise ReferenceUnavailable("GEM-X association confidence is invalid")
        if tracking_confidence < self.config.min_tracking_confidence:
            raise ReferenceUnavailable(
                "GEM-X association confidence is too low: "
                f"{tracking_confidence:.3f} < {self.config.min_tracking_confidence:.3f}"
            )
        required_indices = np.asarray(
            [self._soma_joint_names.index(name) for name in _SOMA_TO_SMPLX_GMR.values()]
        )
        required_confidence = confidence[required_indices]
        if float(np.min(required_confidence)) < self.config.min_joint_confidence:
            index = int(np.argmin(required_confidence))
            name = tuple(_SOMA_TO_SMPLX_GMR.values())[index]
            raise ReferenceUnavailable(
                f"GEM-X joint {name!r} confidence is too low: "
                f"{required_confidence[index]:.3f}"
            )
        pipeline_latency_ms = float(message.pipeline_latency_ms)
        if not math.isfinite(pipeline_latency_ms) or pipeline_latency_ms < 0.0:
            raise ReferenceUnavailable("GEM-X pipeline latency is invalid")
        if pipeline_latency_ms > self.config.max_pipeline_latency_ms:
            raise ReferenceUnavailable(
                f"GEM-X pipeline latency is too high: {pipeline_latency_ms:.1f} ms"
            )
        assert self._coordinates is not None
        positions, quaternions = self._coordinates.transform(
            positions, quaternions, track_id=track_id
        )
        lookup = {name: index for index, name in enumerate(self._soma_joint_names)}
        mapped_positions = np.stack(
            [positions[lookup[name]] for name in _SOMA_TO_SMPLX_GMR.values()]
        )
        mapped_confidence = np.asarray(
            [confidence[lookup[name]] for name in _SOMA_TO_SMPLX_GMR.values()]
        )
        mapped_positions, contacts = self._root_conditioner.condition(
            mapped_positions,
            mapped_confidence,
            timestamp_ns=source_timestamp_ns,
        )
        mapped_quaternions = np.stack(
            [
                _remove_soma_joint_orient(quaternions[lookup[name]], name)
                for name in _SOMA_TO_SMPLX_GMR.values()
            ]
        )
        self._last_frame_id = frame_id
        self._last_source_timestamp_ns = source_timestamp_ns
        return SmplxTargetFrame(
            transforms={
                name: (mapped_positions[index].copy(), mapped_quaternions[index].copy())
                for index, name in enumerate(SMPLX_GMR_BODY_NAMES)
            },
            sequence=frame_id,
            source_timestamp_ns=source_timestamp_ns,
            source_age_ms=math.ceil(
                pipeline_latency_ms + self.config.transport_latency_budget_ms
            ),
            received_at=received_at,
            foot_contacts=contacts,
        )

    def _reset_tracking_epoch(self, track_id: int | None) -> None:
        assert self._coordinates is not None
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
def gemx_sonic_reference_source(
    adapter: Any,
    config: Any,
) -> Iterator[LiveGmrReferenceSource]:
    """Construct the pinned GEM-X SOMA-77 -> GMR -> Bello live source."""

    grpc_config = GemxGrpcConfig.from_hardware(config)
    stream = GemxSomaStream(grpc_config)
    identity = GmrSourceIdentity(
        revision=adapter.metadata.dataset.bello_gmr_revision,
        robot_fingerprint=adapter.metadata.dataset.robot_fingerprint,
    )
    with stream:
        retargeter = GmrBelloRetargeter(
            adapter.env,
            identity,
            GmrLiveRetargetConfig(human_height_m=config.smplx_human_height_m),
            validate_reference=adapter.validate_reference_frame,
        )
        with LiveGmrReferenceSource(stream, retargeter) as source:
            print(
                {
                    "gemx_connection_preflight": "passed",
                    "endpoint": grpc_config.endpoint,
                    "stream_instance_id": stream.stream_instance_id,
                    "body_representation": "soma_77",
                    "semantic_adapter": "soma_77_to_smplx_gmr_body_v1",
                    "soma_rest_orientation_sha256": (
                        _SOMA_NEUTRAL_GLOBAL_ROTATION_SHA256
                    ),
                    "root_conditioning": GEMX_ROOT_CONDITIONING_VERSION,
                    "frame_window": stream.frame_window,
                    "nominal_fps": stream.nominal_fps,
                }
            )
            yield source


def require_pinned_gemx_revision() -> Path:
    root = repository_root()
    submodule = root / "deps" / "GEM-X"
    gitlink = _git_output(root, "ls-tree", "HEAD", "--", "deps/GEM-X").split()
    if len(gitlink) < 3 or gitlink[1] != "commit":
        raise RuntimeError("deps/GEM-X is not a pinned Git submodule")
    current = _git_output(submodule, "rev-parse", "HEAD")
    if gitlink[2] != GEMX_REVISION or current != GEMX_REVISION:
        raise ValueError(
            "GEM-X gitlink and checkout must both equal the reviewed revision "
            f"{GEMX_REVISION}"
        )
    if _git_output(submodule, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("hardware GEM-X submodule must be clean")
    return submodule


def import_pinned_gemx_streaming() -> tuple[Any, Any, Any]:
    submodule = require_pinned_gemx_revision()
    root_string = str(submodule)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    skeleton_pb2 = importlib.import_module("gem.streaming.proto.skeleton_pb2")
    skeleton_pb2_grpc = importlib.import_module("gem.streaming.proto.skeleton_pb2_grpc")
    constants = importlib.import_module("gem.streaming.constants")
    for module in (skeleton_pb2, skeleton_pb2_grpc, constants):
        path = Path(module.__file__).resolve()
        try:
            path.relative_to(submodule.resolve())
        except ValueError as error:
            raise RuntimeError(
                "GEM-X streaming module was not imported from the pinned submodule"
            ) from error
    return skeleton_pb2, skeleton_pb2_grpc, constants


def _finite_array(values: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size != math.prod(shape):
        raise ReferenceUnavailable(
            f"GEM-X {label} has {array.size} values; expected {math.prod(shape)}"
        )
    result = np.array(array.reshape(shape), copy=True, order="C")
    if not np.all(np.isfinite(result)):
        raise ReferenceUnavailable(f"GEM-X {label} contains non-finite values")
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


def _remove_soma_joint_orient(
    quaternion_wxyz: np.ndarray, soma_joint_name: str
) -> np.ndarray:
    rest = np.asarray(_SOMA_NEUTRAL_GLOBAL_WXYZ[soma_joint_name])
    inverse_rest = rest * np.asarray((1.0, -1.0, -1.0, -1.0))
    result = _quat_multiply(np.asarray(quaternion_wxyz), inverse_rest)
    norm = float(np.linalg.norm(result))
    if not math.isfinite(norm) or norm < 1.0e-8:
        raise ReferenceUnavailable(
            f"GEM-X SOMA joint {soma_joint_name!r} pose rotation is invalid"
        )
    return result / norm


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
