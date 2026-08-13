"""Import recorded Fast-SAM SMPL-X streams as TWIST in-house sources."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from bello_mujoco.fastsam_conditioning import (
    FASTSAM_ROOT_CONDITIONING_VERSION,
    CausalFootConditionedRoot,
    FootConditionedRootConfig,
)
from bello_mujoco.hardware.fastsam import FastSamCoordinateAdapter, FastSamGrpcConfig
from bello_mujoco.hardware.smplx_retarget import (
    SMPLX_BODY_JOINT_NAMES,
    SMPLX_BODY_JOINT_PARENTS,
)
from bello_mujoco.motion.twist_dataset import (
    SourceManifest,
    SourceSequence,
    TWIST_SOURCE_MANIFEST_SCHEMA_VERSION,
    load_source_manifest,
    validate_source_manifest,
)


FASTSAM_RECORDING_SOURCE_FORMAT = "fastsam-smplx-body-22-v1"
FASTSAM_RECORDING_FORMAT_VERSION = 1
_ACTIVE_TRACKING_STATES = frozenset(("ACQUIRED", "TRACKING", "REACQUIRED"))


@dataclass(frozen=True)
class FastSamRecordingImportConfig:
    target_fps: float = 50.0
    maximum_clip_seconds: float = 10.0
    minimum_clip_seconds: float = 2.1
    maximum_source_gap_seconds: float = 0.2
    sampling_weight: float = 1.0
    human_height_m: float = 1.66
    root_conditioning: FootConditionedRootConfig = field(
        default_factory=FootConditionedRootConfig
    )
    camera_to_world_quaternion_xyzw: tuple[float, float, float, float] = (
        0.5,
        -0.5,
        0.5,
        -0.5,
    )

    def __post_init__(self) -> None:
        positive = {
            "target_fps": self.target_fps,
            "maximum_clip_seconds": self.maximum_clip_seconds,
            "minimum_clip_seconds": self.minimum_clip_seconds,
            "maximum_source_gap_seconds": self.maximum_source_gap_seconds,
            "sampling_weight": self.sampling_weight,
            "human_height_m": self.human_height_m,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.minimum_clip_seconds > self.maximum_clip_seconds:
            raise ValueError("minimum clip duration exceeds maximum clip duration")


@dataclass(frozen=True)
class FastSamRecordingImportReport:
    recording_root: str
    source_root: str
    source_manifest: str
    combined_manifest: str | None
    input_frames: int
    source_epochs: int
    emitted_clips: int
    emitted_frames: int
    emitted_seconds: float
    discarded_short_frames: int


@dataclass(frozen=True)
class _RecordedFrame:
    stream_instance_id: str
    track_id: int
    frame_id: int
    timestamp_ns: int
    positions: np.ndarray
    quaternions_wxyz: np.ndarray
    confidence: np.ndarray
    foot_contacts: np.ndarray | None = None


def import_fastsam_recording(
    recording_root: Path,
    source_root: Path,
    source_manifest_path: Path,
    *,
    combined_manifest_path: Path | None = None,
    base_manifest_path: Path | None = None,
    source_id: str = "inhouse_fastsam",
    subject_id: str = "operator",
    owner: str = "operator",
    ownership_basis: str = "operator-provided recording",
    license_id: str = "in-house-owner-authorized",
    config: FastSamRecordingImportConfig = FastSamRecordingImportConfig(),
    overwrite: bool = False,
) -> FastSamRecordingImportReport:
    """Convert one complete recording into portable, train-only source clips."""

    recording_root = Path(recording_root).expanduser().resolve()
    source_root = Path(source_root).expanduser().resolve()
    source_manifest_path = Path(source_manifest_path).expanduser().resolve()
    combined_manifest_path = (
        None
        if combined_manifest_path is None
        else Path(combined_manifest_path).expanduser().resolve()
    )
    base_manifest_path = (
        None
        if base_manifest_path is None
        else Path(base_manifest_path).expanduser().resolve()
    )
    if not recording_root.is_dir():
        raise NotADirectoryError(f"Fast-SAM recording not found: {recording_root}")
    for path in (source_manifest_path, combined_manifest_path):
        if path is not None and path.exists() and not overwrite:
            raise FileExistsError(f"output already exists: {path}")
    session = _load_json(recording_root / "session.json")
    if session.get("schema_version") != 1 or session.get("status") != "complete":
        raise ValueError("Fast-SAM recording session is incomplete or unsupported")
    if int(session.get("counts", {}).get("queue_drops", -1)) != 0:
        raise ValueError("Fast-SAM recording contains writer queue drops")
    streams = [
        json.loads(line)
        for line in (recording_root / "streams.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if len(streams) != 1:
        raise ValueError("Fast-SAM recording must contain exactly one stream contract")
    stream = streams[0]
    _validate_stream_contract(stream)
    session_id = recording_root.name
    dataset_release = (
        f"Fast-SAM-3D-Body-{session['software']['version']}"
        f"+{stream['config_sha256'][:12]}"
    )
    source_root.mkdir(parents=True, exist_ok=True)
    coordinate_config = FastSamGrpcConfig(
        endpoint="recording.invalid:1",
        neutral_model_path=Path("unused.npz"),
        tls_root_certificate_path=None,
        tls_server_name_override=None,
        connect_timeout_seconds=1.0,
        rpc_timeout_seconds=1.0,
        min_tracking_confidence=0.0,
        min_joint_confidence=0.0,
        max_pipeline_latency_ms=1.0,
        transport_latency_budget_ms=0.0,
        camera_to_world_quaternion_xyzw=(config.camera_to_world_quaternion_xyzw),
        align_initial_heading=True,
        align_initial_floor=True,
    )
    adapter = FastSamCoordinateAdapter(coordinate_config)
    root_conditioner = CausalFootConditionedRoot(
        SMPLX_BODY_JOINT_NAMES, config.root_conditioning
    )
    sequences: list[SourceSequence] = []
    input_frames = 0
    emitted_frames = 0
    discarded_short_frames = 0
    epoch_index = -1
    chunk_index = 0
    chunk: list[_RecordedFrame] = []
    last: _RecordedFrame | None = None

    def flush() -> None:
        nonlocal chunk, chunk_index, emitted_frames, discarded_short_frames
        if not chunk:
            return
        duration = (chunk[-1].timestamp_ns - chunk[0].timestamp_ns) / 1.0e9
        if duration < config.minimum_clip_seconds:
            discarded_short_frames += len(chunk)
            chunk = []
            return
        sequence_id = f"{session_id}__epoch{epoch_index:02d}__clip{chunk_index:03d}"
        path = source_root / f"{source_id}__{sequence_id}.npz"
        arrays = _resample_conditioned(chunk, config)
        _write_source_clip(path, arrays, config, stream, overwrite=overwrite)
        digest = _file_sha256(path)
        frames = int(arrays["joint_positions"].shape[0])
        sequences.append(
            SourceSequence(
                source_kind="in_house",
                source_id=source_id,
                sequence_id=sequence_id,
                dataset_release=dataset_release,
                license_id=license_id,
                local_path=path,
                subject_id=subject_id,
                session_id=session_id,
                original_sequence_id=f"epoch{epoch_index:02d}_clip{chunk_index:03d}",
                pose_representation="smpl-x",
                frame_rate=config.target_fps,
                requested_split="train",
                sampling_weight=config.sampling_weight,
                source_sha256=digest,
                subdataset=None,
                near_duplicate_fingerprint=None,
                metadata={
                    "owner": owner,
                    "ownership_basis": ownership_basis,
                    "source_format": FASTSAM_RECORDING_SOURCE_FORMAT,
                    "recording_session": session_id,
                    "stream_instance_id": chunk[0].stream_instance_id,
                    "track_id": str(chunk[0].track_id),
                    "source_frame_first": str(chunk[0].frame_id),
                    "source_frame_last": str(chunk[-1].frame_id),
                    "frame_count": str(frames),
                    "root_conditioning": FASTSAM_ROOT_CONDITIONING_VERSION,
                    "root_conditioning_config": json.dumps(
                        asdict(config.root_conditioning),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                },
            )
        )
        emitted_frames += frames
        chunk_index += 1
        chunk = []

    skeleton_path = recording_root / "skeletons.jsonl"
    with skeleton_path.open(encoding="utf-8") as lines:
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            frame = _parse_frame(json.loads(line), stream, line_number)
            input_frames += 1
            discontinuity = _is_discontinuity(last, frame, config)
            if discontinuity:
                flush()
                epoch_index += 1
                chunk_index = 0
                adapter.reset()
                root_conditioner.reset()
            positions, quaternions = adapter.transform(
                frame.positions,
                frame.quaternions_wxyz[:, (1, 2, 3, 0)],
                track_id=frame.track_id,
            )
            positions, foot_contacts = root_conditioner.condition(
                positions,
                frame.confidence,
                timestamp_ns=frame.timestamp_ns,
            )
            transformed = _RecordedFrame(
                stream_instance_id=frame.stream_instance_id,
                track_id=frame.track_id,
                frame_id=frame.frame_id,
                timestamp_ns=frame.timestamp_ns,
                positions=positions,
                quaternions_wxyz=quaternions,
                confidence=frame.confidence,
                foot_contacts=foot_contacts,
            )
            if (
                chunk
                and (transformed.timestamp_ns - chunk[0].timestamp_ns) / 1.0e9
                >= config.maximum_clip_seconds
            ):
                flush()
            chunk.append(transformed)
            last = frame
    flush()
    if not sequences:
        raise ValueError("Fast-SAM recording produced no clips long enough for TWIST")
    source_manifest = SourceManifest(
        schema_version=TWIST_SOURCE_MANIFEST_SCHEMA_VERSION,
        split_seed=20251009,
        sequences=tuple(sequences),
    )
    # These sources already contain the complete SMPL-X joint transforms and do
    # not require a licensed parametric body model until other manifest sources
    # are validated for a combined dataset build.
    validate_source_manifest(source_manifest, verify_files=False)
    for sequence in source_manifest.sequences:
        if _file_sha256(sequence.local_path) != sequence.source_sha256:
            raise RuntimeError(
                f"generated source checksum drifted: {sequence.local_path}"
            )
    _write_manifest(source_manifest_path, source_manifest)
    combined = None
    if base_manifest_path is not None or combined_manifest_path is not None:
        if base_manifest_path is None or combined_manifest_path is None:
            raise ValueError(
                "base_manifest_path and combined_manifest_path must be provided together"
            )
        base = load_source_manifest(base_manifest_path, verify_files=False)
        combined = SourceManifest(
            schema_version=base.schema_version,
            split_seed=base.split_seed,
            sequences=base.sequences + source_manifest.sequences,
        )
        validate_source_manifest(combined, verify_files=False)
        _write_manifest(combined_manifest_path, combined)
    report = FastSamRecordingImportReport(
        recording_root=str(recording_root),
        source_root=str(source_root),
        source_manifest=str(source_manifest_path),
        combined_manifest=(
            None if combined_manifest_path is None else str(combined_manifest_path)
        ),
        input_frames=input_frames,
        source_epochs=epoch_index + 1,
        emitted_clips=len(sequences),
        emitted_frames=emitted_frames,
        emitted_seconds=emitted_frames / config.target_fps,
        discarded_short_frames=discarded_short_frames,
    )
    report_path = source_manifest_path.with_suffix(".import_report.json")
    report_path.write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _parse_frame(
    document: dict[str, Any], stream: dict[str, Any], line: int
) -> _RecordedFrame:
    state = str(document.get("tracking_state", ""))
    if state not in _ACTIVE_TRACKING_STATES:
        raise ValueError(f"skeletons.jsonl:{line}: inactive tracking state {state!r}")
    stream_id = str(document.get("stream_instance_id", ""))
    if stream_id != stream["stream_instance_id"]:
        raise ValueError(f"skeletons.jsonl:{line}: stream identity mismatch")
    positions = np.asarray(document.get("joint_positions"), dtype=np.float64)
    quaternions_xyzw = np.asarray(
        document.get("global_joint_quaternions"), dtype=np.float64
    )
    confidence = np.asarray(document.get("joint_confidence"), dtype=np.float64)
    expected = len(SMPLX_BODY_JOINT_NAMES)
    if positions.shape != (expected, 3):
        raise ValueError(f"skeletons.jsonl:{line}: invalid joint positions")
    if quaternions_xyzw.shape != (expected, 4):
        raise ValueError(f"skeletons.jsonl:{line}: invalid global quaternions")
    if confidence.shape != (expected,):
        raise ValueError(f"skeletons.jsonl:{line}: invalid joint confidence")
    if not all(
        np.all(np.isfinite(value))
        for value in (positions, quaternions_xyzw, confidence)
    ):
        raise ValueError(f"skeletons.jsonl:{line}: non-finite skeleton values")
    norms = np.linalg.norm(quaternions_xyzw, axis=1)
    if np.any((norms < 0.999) | (norms > 1.001)):
        raise ValueError(f"skeletons.jsonl:{line}: non-unit global quaternion")
    quaternions_xyzw /= norms[:, None]
    return _RecordedFrame(
        stream_instance_id=stream_id,
        track_id=int(document["track_id"]),
        frame_id=int(document["frame_id"]),
        timestamp_ns=int(document["source_timestamp_ns"]),
        positions=positions,
        quaternions_wxyz=quaternions_xyzw[:, (3, 0, 1, 2)],
        confidence=confidence,
    )


def _is_discontinuity(
    previous: _RecordedFrame | None,
    current: _RecordedFrame,
    config: FastSamRecordingImportConfig,
) -> bool:
    if previous is None:
        return True
    return (
        current.stream_instance_id != previous.stream_instance_id
        or current.track_id != previous.track_id
        or current.frame_id <= previous.frame_id
        or current.timestamp_ns <= previous.timestamp_ns
        or (current.timestamp_ns - previous.timestamp_ns) / 1.0e9
        > config.maximum_source_gap_seconds
    )


def _resample_conditioned(
    frames: list[_RecordedFrame], config: FastSamRecordingImportConfig
) -> dict[str, np.ndarray]:
    source_time = np.asarray(
        [(frame.timestamp_ns - frames[0].timestamp_ns) / 1.0e9 for frame in frames],
        dtype=np.float64,
    )
    duration = float(source_time[-1])
    count = math.floor(duration * config.target_fps + 1.0e-8) + 1
    target_time = np.arange(count, dtype=np.float64) / config.target_fps
    positions = _interpolate_linear(
        source_time, np.stack([frame.positions for frame in frames]), target_time
    )
    source_quaternions = np.stack([frame.quaternions_wxyz for frame in frames])
    quaternions = np.stack(
        [
            _interpolate_quaternion_rows(
                source_time, source_quaternions[:, joint], target_time
            )
            for joint in range(source_quaternions.shape[1])
        ],
        axis=1,
    )
    confidence = _interpolate_linear(
        source_time, np.stack([frame.confidence for frame in frames]), target_time
    )
    if any(frame.foot_contacts is None for frame in frames):
        raise RuntimeError("Fast-SAM source frame was not root-conditioned")
    source_contacts = np.stack([frame.foot_contacts for frame in frames])
    contact_indices = np.searchsorted(source_time, target_time, side="right") - 1
    contact_indices = np.clip(contact_indices, 0, len(source_time) - 1)
    contacts = source_contacts[contact_indices]
    absolute_source_time = frames[0].timestamp_ns / 1.0e9 + target_time
    return {
        "timestamp_seconds": target_time,
        "source_timestamp_seconds": absolute_source_time,
        "joint_positions": positions.astype(np.float32),
        "global_joint_quaternions_wxyz": quaternions.astype(np.float32),
        "joint_confidence": confidence.astype(np.float32),
        "foot_contacts": contacts.astype(np.float32),
    }


def _interpolate_linear(
    source_time: np.ndarray, values: np.ndarray, target_time: np.ndarray
) -> np.ndarray:
    flat = values.reshape((values.shape[0], -1))
    result = np.stack(
        [
            np.interp(target_time, source_time, flat[:, index])
            for index in range(flat.shape[1])
        ],
        axis=1,
    )
    return result.reshape((len(target_time),) + values.shape[1:])


def _interpolate_quaternion_rows(
    source_time: np.ndarray, quaternions: np.ndarray, target_time: np.ndarray
) -> np.ndarray:
    source = np.asarray(quaternions, dtype=np.float64).copy()
    for index in range(1, len(source)):
        if float(source[index - 1] @ source[index]) < 0.0:
            source[index] *= -1.0
    upper = np.searchsorted(source_time, target_time, side="right")
    upper = np.clip(upper, 1, len(source_time) - 1)
    lower = upper - 1
    denominator = source_time[upper] - source_time[lower]
    fraction = np.divide(
        target_time - source_time[lower],
        denominator,
        out=np.zeros_like(target_time),
        where=denominator > 0.0,
    )
    left = source[lower]
    right = source[upper]
    dots = np.clip(np.sum(left * right, axis=1), -1.0, 1.0)
    angles = np.arccos(dots)
    sin_angles = np.sin(angles)
    linear = sin_angles < 1.0e-8
    left_weight = np.empty_like(fraction)
    right_weight = np.empty_like(fraction)
    left_weight[linear] = 1.0 - fraction[linear]
    right_weight[linear] = fraction[linear]
    left_weight[~linear] = (
        np.sin((1.0 - fraction[~linear]) * angles[~linear]) / sin_angles[~linear]
    )
    right_weight[~linear] = (
        np.sin(fraction[~linear] * angles[~linear]) / sin_angles[~linear]
    )
    result = left_weight[:, None] * left + right_weight[:, None] * right
    return result / np.linalg.norm(result, axis=1, keepdims=True)


def _write_source_clip(
    path: Path,
    arrays: dict[str, np.ndarray],
    config: FastSamRecordingImportConfig,
    stream: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"source clip already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        format_version=np.asarray(FASTSAM_RECORDING_FORMAT_VERSION, dtype=np.int32),
        source_format=np.asarray(FASTSAM_RECORDING_SOURCE_FORMAT),
        joint_names=np.asarray(SMPLX_BODY_JOINT_NAMES),
        joint_parents=SMPLX_BODY_JOINT_PARENTS,
        human_height_m=np.asarray(config.human_height_m, dtype=np.float64),
        stream_config_sha256=np.asarray(stream["config_sha256"]),
        root_conditioning_version=np.asarray(FASTSAM_ROOT_CONDITIONING_VERSION),
        root_conditioning_config=np.asarray(
            json.dumps(
                asdict(config.root_conditioning),
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        **arrays,
    )


def _validate_stream_contract(stream: dict[str, Any]) -> None:
    expected = {
        "body_representation": "smplx_body_22",
        "coordinate_frame": "camera_opencv",
        "quaternion_order": "xyzw",
        "global_quaternion_semantics": "smplx_joint_frame_to_camera",
        "position_units": "meters",
    }
    mismatches = [name for name, value in expected.items() if stream.get(name) != value]
    if mismatches:
        raise ValueError("Fast-SAM stream contract mismatch: " + ", ".join(mismatches))
    if tuple(stream.get("joint_names", ())) != SMPLX_BODY_JOINT_NAMES:
        raise ValueError("Fast-SAM stream has an unexpected joint order")
    if not np.array_equal(stream.get("joint_parents"), SMPLX_BODY_JOINT_PARENTS):
        raise ValueError("Fast-SAM stream has an unexpected joint hierarchy")
    if int(stream.get("root_index", -1)) != 0:
        raise ValueError("Fast-SAM stream has an unexpected root index")


def _write_manifest(path: Path, manifest: SourceManifest) -> None:
    payload = asdict(manifest)
    for sequence in payload["sequences"]:
        sequence["local_path"] = str(sequence["local_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(payload), path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"invalid Fast-SAM recording metadata {path}: {error}"
        ) from error
    if not isinstance(document, dict):
        raise ValueError(f"Fast-SAM recording metadata must be an object: {path}")
    return document


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
