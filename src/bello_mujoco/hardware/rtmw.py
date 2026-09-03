"""RTMW-SMPLX-Teleop latest-JSON source for locomotion upper-body control."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterator

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


RTMW_SCHEMA_VERSION = 1
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
_MAX_RECORD_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class RtmwLatestJsonConfig:
    """Strict RTMW latest-file contract and live-pose qualification limits."""

    pose_path: Path
    neutral_model_path: Path
    human_height_m: float
    min_joint_confidence: float
    min_rotation_confidence: float
    max_pipeline_latency_ms: float
    max_source_age_ms: int
    camera_to_world_quaternion_xyzw: tuple[float, float, float, float]
    align_initial_heading: bool
    align_initial_floor: bool
    poll_interval_seconds: float = 0.01
    root_conditioning: FootConditionedRootConfig = field(
        default_factory=FootConditionedRootConfig
    )

    @classmethod
    def from_hardware(cls, config: Any) -> RtmwLatestJsonConfig:
        if config.smplx_neutral_model_path is None:
            raise ValueError(
                "smplx_neutral_model_path is required for live RTMW hardware pose"
            )
        return cls(
            pose_path=config.rtmw_pose_path,
            neutral_model_path=config.smplx_neutral_model_path,
            human_height_m=config.rtmw_human_height_m,
            min_joint_confidence=config.rtmw_min_joint_confidence,
            min_rotation_confidence=config.rtmw_min_rotation_confidence,
            max_pipeline_latency_ms=config.rtmw_max_pipeline_latency_ms,
            max_source_age_ms=config.upper_body_follow.max_reference_age_ms,
            camera_to_world_quaternion_xyzw=(
                config.rtmw_camera_to_world_quaternion_xyzw
            ),
            align_initial_heading=config.rtmw_align_initial_heading,
            align_initial_floor=config.rtmw_align_initial_floor,
        )

    def __post_init__(self) -> None:
        if not isinstance(self.pose_path, Path):
            raise TypeError("RTMW pose_path must be a pathlib.Path")
        if not isinstance(self.neutral_model_path, Path):
            raise TypeError("RTMW neutral_model_path must be a pathlib.Path")
        for name, value in (
            ("human_height_m", self.human_height_m),
            ("max_pipeline_latency_ms", self.max_pipeline_latency_ms),
            ("poll_interval_seconds", self.poll_interval_seconds),
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
        quaternion = np.asarray(self.camera_to_world_quaternion_xyzw)
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            raise ValueError("RTMW camera rotation must be a finite quaternion")
        if not 0.999 <= float(np.linalg.norm(quaternion)) <= 1.001:
            raise ValueError("RTMW camera rotation quaternion must be normalized")


class RtmwCoordinateAdapter:
    """Map RTMW OpenCV camera poses into Bello's z-up world frame."""

    def __init__(self, config: RtmwLatestJsonConfig) -> None:
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


class RtmwLatestJsonStream:
    """Poll RTMW's atomic latest-JSON output with latest-only semantics."""

    def __init__(
        self,
        config: RtmwLatestJsonConfig,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        standing_sample_loader: Callable[[Path], SmplxTargetFrame] = (
            load_smplx_standing_sample
        ),
    ) -> None:
        self.config = config
        self.pose_path = config.pose_path.expanduser().absolute()
        self._clock_ns = clock_ns
        self._standing_sample = standing_sample_loader(config.neutral_model_path)
        self.neutral_model_sha256 = _file_sha256(config.neutral_model_path)
        self._coordinates = RtmwCoordinateAdapter(config)
        self._root_conditioner = CausalFootConditionedRoot(
            RTMW_SMPLX52_NAMES, config.root_conditioning
        )
        self._last_signature: tuple[int, int, int] | None = None
        self._last_frame_id: int | None = None
        self._last_source_timestamp_ns: int | None = None
        self._active_track_id: int | str | None = None

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
        while True:
            document = self._read_new_document()
            if document is not None:
                frame = self._parse_record(document)
                if after_sequence is not None and frame.sequence <= after_sequence:
                    self._reset_tracking_epoch(self._active_track_id)
                    raise ReferenceDiscontinuity(
                        "RTMW frame sequence did not advance: "
                        f"{frame.sequence} <= {after_sequence}"
                    )
                return frame
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return None
            time.sleep(min(self.config.poll_interval_seconds, remaining))

    def _read_new_document(self) -> Mapping[str, Any] | None:
        try:
            with self.pose_path.open("rb") as stream:
                stat = os.fstat(stream.fileno())
                signature = (int(stat.st_ino), int(stat.st_mtime_ns), int(stat.st_size))
                if signature == self._last_signature:
                    return None
                if stat.st_size <= 0 or stat.st_size > _MAX_RECORD_BYTES:
                    raise ValueError(
                        "RTMW latest-JSON record size must be between 1 byte and "
                        f"{_MAX_RECORD_BYTES} bytes"
                    )
                payload = stream.read(_MAX_RECORD_BYTES + 1)
        except FileNotFoundError:
            return None
        self._last_signature = signature
        if len(payload) > _MAX_RECORD_BYTES:
            raise ValueError("RTMW latest-JSON record exceeds the size limit")
        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                "RTMW latest-JSON record is not valid UTF-8 JSON"
            ) from error
        if not isinstance(document, Mapping):
            raise ValueError("RTMW latest-JSON record must be an object")
        return document

    def _parse_record(self, document: Mapping[str, Any]) -> SmplxTargetFrame:
        if (
            _integer(document.get("schema_version"), "schema_version")
            != RTMW_SCHEMA_VERSION
        ):
            raise ValueError(f"RTMW schema_version must be {RTMW_SCHEMA_VERSION}")
        frame_id = _nonnegative_integer(document.get("frame_id"), "frame_id")
        captured_at_ns = _positive_integer(
            document.get("captured_at_ns"), "captured_at_ns"
        )
        source_timestamp_ns = _positive_integer(
            document.get("source_timestamp_ns"), "source_timestamp_ns"
        )
        inferred_at_ns = _positive_integer(
            document.get("inferred_at_ns"), "inferred_at_ns"
        )
        if source_timestamp_ns != captured_at_ns:
            raise ValueError(
                "RTMW hardware input must come from a live camera whose source and "
                "capture timestamps are identical"
            )
        if inferred_at_ns < captured_at_ns:
            raise ValueError("RTMW inferred_at_ns precedes captured_at_ns")
        pipeline_latency_ms = _finite_scalar(document.get("latency_ms"), "latency_ms")
        if pipeline_latency_ms < 0.0:
            raise ValueError("RTMW latency_ms must be nonnegative")
        timestamp_latency_ms = (inferred_at_ns - captured_at_ns) / 1.0e6
        if abs(timestamp_latency_ms - pipeline_latency_ms) > 1.0:
            raise ValueError("RTMW latency_ms disagrees with its timestamps")
        if pipeline_latency_ms > self.config.max_pipeline_latency_ms:
            raise ReferenceUnavailable(
                f"RTMW pipeline latency is too high: {pipeline_latency_ms:.1f} ms"
            )
        age_ms = (self._clock_ns() - captured_at_ns) / 1.0e6
        if age_ms < -1.0:
            raise ValueError(
                "RTMW capture timestamp is from a different monotonic clock; run "
                "RTMW and bello-run-hardware on the same host"
            )
        age_ms = max(0.0, age_ms)
        if age_ms + 1.0 < pipeline_latency_ms:
            raise ValueError("RTMW source age is smaller than its pipeline latency")
        if age_ms > self.config.max_source_age_ms:
            raise ReferenceUnavailable(
                f"RTMW source frame is stale: {age_ms:.1f} ms exceeds "
                f"{self.config.max_source_age_ms} ms"
            )

        tracking = _mapping(document.get("tracking"), "tracking")
        if tracking.get("state") != "tracked":
            raise ReferenceUnavailable("RTMW tracking state is not 'tracked'")
        track_id = _track_id(tracking.get("track_id"), "tracking.track_id")
        body = _mapping(document.get("smplx52_landmark_fit"), "smplx52_landmark_fit")
        body_track_id = _track_id(body.get("track_id"), "smplx52.track_id")
        if body_track_id != track_id:
            raise ValueError("RTMW tracking and SMPL-X track ids differ")
        if _nonnegative_integer(body.get("frame_id"), "smplx52.frame_id") != frame_id:
            raise ValueError("RTMW record and SMPL-X frame ids differ")
        if (
            _positive_integer(body.get("captured_at_ns"), "smplx52.captured_at_ns")
            != captured_at_ns
        ):
            raise ValueError("RTMW record and SMPL-X capture timestamps differ")
        body_inferred_at_ns = _positive_integer(
            body.get("inferred_at_ns"), "smplx52.inferred_at_ns"
        )
        if not captured_at_ns <= body_inferred_at_ns <= inferred_at_ns:
            raise ValueError(
                "RTMW SMPL-X inference timestamp is outside the record's processing "
                "interval"
            )
        self._validate_body_contract(body)

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
            body.get("positions_m"), (len(RTMW_SMPLX52_NAMES), 3), "positions_m"
        )
        joint_positions = _finite_array(
            body.get("joint_positions"),
            (len(RTMW_SMPLX52_NAMES), 3),
            "joint_positions",
        )
        if not np.allclose(positions, joint_positions, atol=1.0e-6, rtol=0.0):
            raise ValueError("RTMW position aliases differ")
        confidence = _confidence_array(
            body.get("confidence"), len(RTMW_SMPLX52_NAMES), "confidence"
        )
        joint_confidence = _confidence_array(
            body.get("joint_confidence"),
            len(RTMW_SMPLX52_NAMES),
            "joint_confidence",
        )
        if not np.allclose(confidence, joint_confidence, atol=1.0e-6, rtol=0.0):
            raise ValueError("RTMW confidence aliases differ")
        valid_mask = _boolean_array(
            body.get("valid_mask"), len(RTMW_SMPLX52_NAMES), "valid_mask"
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
            body.get("local_joint_quaternions"), "local_joint_quaternions"
        )
        global_quaternions = _normalized_quaternions(
            body.get("global_joint_quaternions"), "global_joint_quaternions"
        )
        _validate_tree_consistency(local_quaternions, global_quaternions)
        rotation_valid = _boolean_array(
            body.get("rotation_valid_mask"),
            len(RTMW_SMPLX52_NAMES),
            "rotation_valid_mask",
        )
        rotation_confidence = _confidence_array(
            body.get("rotation_confidence"),
            len(RTMW_SMPLX52_NAMES),
            "rotation_confidence",
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

        root = _finite_array(body.get("root_translation_m"), (3,), "root_translation")
        if not np.allclose(root, positions[0], atol=1.0e-6, rtol=0.0):
            raise ValueError("RTMW root translation differs from its pelvis position")
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
            source_age_ms=math.ceil(age_ms),
            received_at=time.monotonic(),
            foot_contacts=foot_contacts,
        )

    def _validate_body_contract(self, body: Mapping[str, Any]) -> None:
        exact = {
            "representation": (body.get("representation"), RTMW_REPRESENTATION),
            "schema_id": (body.get("schema_id"), RTMW_SCHEMA_ID),
            "coordinate_space": (
                body.get("coordinate_space"),
                RTMW_COORDINATE_SPACE,
            ),
            "quaternion_order": (body.get("quaternion_order"), "xyzw"),
            "position_units": (body.get("position_units"), "meters"),
            "rotation_semantics": (
                body.get("rotation_semantics"),
                RTMW_ROTATION_SEMANTICS,
            ),
        }
        for name, (actual, expected) in exact.items():
            if actual != expected:
                raise ValueError(f"RTMW {name} is {actual!r}; expected {expected!r}")
        if body.get("metric_scale") is not True:
            raise ValueError("RTMW hardware pose must have metric_scale=true")
        if body.get("global_translation") is not True:
            raise ValueError("RTMW hardware pose must have global_translation=true")
        if body.get("depth_provenance") not in {
            "calibrated_camera+assumed_subject_height",
            "fixed_root_depth_assumption",
        }:
            raise ValueError(
                "RTMW hardware pose must use calibrated height-prior or fixed-depth "
                "reconstruction"
            )
        if body.get("global_depth_is_measured") is not False:
            raise ValueError(
                "RTMW monocular hardware pose must declare "
                "global_depth_is_measured=false"
            )
        if _integer(body.get("root_index"), "root_index") != 0:
            raise ValueError("RTMW root_index must be 0")
        names = body.get("joint_names")
        if not isinstance(names, list) or tuple(names) != RTMW_SMPLX52_NAMES:
            raise ValueError("RTMW stream has an unexpected SMPL-X joint order")
        parents = body.get("joint_parents")
        if not isinstance(parents, list) or tuple(parents) != RTMW_SMPLX52_PARENTS:
            raise ValueError("RTMW stream has an unexpected SMPL-X hierarchy")

    def _reset_tracking_epoch(self, track_id: int | str | None) -> None:
        self._coordinates.reset()
        self._root_conditioner.reset()
        self._active_track_id = track_id
        self._last_frame_id = None
        self._last_source_timestamp_ns = None


@contextmanager
def rtmw_locomotion_reference_source(
    adapter: Any,
    config: Any,
) -> Iterator[LiveGmrReferenceSource]:
    """Construct RTMW latest-JSON -> pinned GMR locomotion references."""

    source_config = RtmwLatestJsonConfig.from_hardware(config)
    stream = RtmwLatestJsonStream(source_config)
    retargeter = GmrBelloRetargeter(
        adapter.env,
        repository_gmr_source_identity(),
        GmrLiveRetargetConfig(human_height_m=source_config.human_height_m),
        validate_reference=adapter.validate_upper_body_reference,
    )
    with LiveGmrReferenceSource(stream, retargeter) as source:
        print(
            {
                "rtmw_source_preflight": "passed",
                "pose_path": str(stream.pose_path),
                "schema_version": RTMW_SCHEMA_VERSION,
                "representation": RTMW_REPRESENTATION,
                "neutral_model_sha256": stream.neutral_model_sha256,
                "root_conditioning": FASTSAM_ROOT_CONDITIONING_VERSION,
                "live_target_check": "required_before_hardware_connection",
            }
        )
        yield source


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"RTMW {label} must be an object")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"RTMW {label} must be an integer")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    result = _integer(value, label)
    if result < 0:
        raise ValueError(f"RTMW {label} must be nonnegative")
    return result


def _positive_integer(value: Any, label: str) -> int:
    result = _integer(value, label)
    if result <= 0:
        raise ValueError(f"RTMW {label} must be positive")
    return result


def _finite_scalar(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"RTMW {label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"RTMW {label} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"RTMW {label} must be finite")
    return result


def _track_id(value: Any, label: str) -> int | str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"RTMW {label} must be an integer or string")
    if isinstance(value, int) and value < 0:
        raise ValueError(f"RTMW {label} must be nonnegative")
    if isinstance(value, str) and not value:
        raise ValueError(f"RTMW {label} must not be empty")
    return value


def _finite_array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"RTMW {label} must contain finite numbers") from error
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"RTMW {label} must have finite shape {shape}")
    return np.array(array, copy=True, order="C")


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
