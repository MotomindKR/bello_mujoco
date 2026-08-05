"""Licensed-source manifests and portable packed datasets for TWIST.

Raw human motion and body-model files deliberately stay outside the repository.
This module handles host-side validation, deterministic splitting, retargeted
clip quality control, and safe (pickle-free) NPZ packing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Literal

import numpy as np
from omegaconf import OmegaConf

from bello_mujoco.configuration import dataclass_from_dict

TWIST_DATASET_SCHEMA_VERSION = 1
TWIST_POLICY_FPS = 50.0
TWIST_FUTURE_OFFSETS = (
    1,
    5,
    10,
    15,
    20,
    25,
    30,
    35,
    40,
    45,
    50,
    55,
    60,
    65,
    70,
    75,
    80,
    85,
    90,
    95,
)
_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SHA256_PATTERN = re.compile(r"(?:sha256:)?([0-9a-f]{64})\Z")

SourceKind = Literal["amass", "omomo", "in_house"]
RequestedSplit = Literal["auto", "train", "evaluation"]
AssignedSplit = Literal["train", "evaluation"]


@dataclass(frozen=True)
class SourceSequence:
    source_kind: SourceKind
    source_id: str
    sequence_id: str
    dataset_release: str
    license_id: str
    local_path: Path
    subject_id: str
    session_id: str
    original_sequence_id: str
    pose_representation: str
    frame_rate: float
    requested_split: RequestedSplit = "auto"
    sampling_weight: float = 1.0
    source_sha256: str = ""
    subdataset: str | None = None
    near_duplicate_fingerprint: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceManifest:
    schema_version: int
    split_seed: int
    sequences: tuple[SourceSequence, ...]


@dataclass(frozen=True)
class AssignedSequence:
    sequence: SourceSequence
    split: AssignedSplit


@dataclass(frozen=True)
class QualityConfig:
    policy_fps: float = TWIST_POLICY_FPS
    future_horizon_steps: int = max(TWIST_FUTURE_OFFSETS)
    quaternion_norm_tolerance: float = 0.05
    max_quaternion_step_radians: float = 2.8
    max_timestamp_gap_factor: float = 2.0
    max_joint_limit_violation_radians: float = 0.02
    max_joint_limit_violation_fraction: float = 0.02
    max_root_linear_velocity: float = 12.0
    max_root_angular_velocity: float = 20.0
    max_body_linear_velocity: float = 20.0
    max_body_angular_velocity: float = 30.0
    min_root_height: float = 0.25
    max_root_height: float = 2.0
    max_floor_penetration: float = 0.08
    max_frames_per_shard: int = 250_000


@dataclass(frozen=True)
class ClipDecision:
    source_id: str
    sequence_id: str
    status: Literal["accepted", "quarantined", "rejected"]
    reasons: tuple[str, ...]
    frames: int = 0
    duration_seconds: float = 0.0
    retargeted_sha256: str | None = None


def load_quality_config(path: Path) -> QualityConfig:
    path = Path(path).expanduser().resolve()
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    return dataclass_from_dict(
        QualityConfig,
        raw,
        strict=True,
        require_all_fields=True,
        path="quality",
    )


def load_source_manifest(
    path: Path,
    *,
    body_model_root: Path | None = None,
    verify_files: bool = True,
) -> SourceManifest:
    """Load a strict human-authored YAML/JSON manifest.

    Relative source paths are resolved against the manifest directory. They are
    used for local import only and are removed from portable fingerprints.
    """

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"TWIST source manifest not found: {path}")
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("TWIST source manifest root must be a mapping")
    manifest = dataclass_from_dict(
        SourceManifest,
        raw,
        strict=True,
        require_all_fields=True,
        path="manifest",
    )
    sequences = tuple(
        _resolve_sequence(sequence, path.parent) for sequence in manifest.sequences
    )
    manifest = SourceManifest(
        schema_version=manifest.schema_version,
        split_seed=manifest.split_seed,
        sequences=sequences,
    )
    validate_source_manifest(
        manifest,
        body_model_root=body_model_root,
        verify_files=verify_files,
    )
    return manifest


def scan_amass_manifest(
    source_root: Path,
    output_path: Path,
    *,
    dataset_release: str,
    license_id: str,
    split_seed: int,
    sampling_weight: float = 1.0,
    overwrite: bool = False,
    progress: Callable[[int, int, Path], None] | None = None,
) -> SourceManifest:
    """Scan operator-supplied AMASS Stage II archives into a strict manifest."""

    source_root = Path(source_root).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not source_root.is_dir():
        raise NotADirectoryError(f"AMASS source root not found: {source_root}")
    if not dataset_release.strip():
        raise ValueError("dataset_release is required for an AMASS scan")
    if not license_id.strip():
        raise ValueError("license_id is required for an AMASS scan")
    if not math.isfinite(sampling_weight) or sampling_weight <= 0.0:
        raise ValueError("sampling_weight must be positive and finite")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"AMASS manifest already exists: {output_path}; set overwrite=true "
            "to replace it"
        )

    paths = sorted(source_root.rglob("*_stageii.npz"))
    if not paths:
        raise ValueError(f"no AMASS Stage II NPZ archives found below {source_root}")
    sequences = []
    rejected: list[dict[str, str]] = []
    quarantined: list[dict[str, str]] = []
    source_checksums: dict[str, str] = {}
    motion_fingerprints: dict[str, str] = {}
    for index, path in enumerate(paths, start=1):
        relative = path.relative_to(source_root)
        if len(relative.parts) < 3:
            rejected.append(
                {
                    "relative_path": relative.as_posix(),
                    "reason": "archive must be nested as subdataset/subject/file",
                }
            )
            if progress is not None:
                progress(index, len(paths), path)
            continue
        try:
            archive = _inspect_amass_archive(path)
        except ValueError as error:
            rejected.append(
                {"relative_path": relative.as_posix(), "reason": str(error)}
            )
            if progress is not None:
                progress(index, len(paths), path)
            continue
        source_sha256 = _file_sha256(path)
        if source_sha256 in source_checksums:
            quarantined.append(
                {
                    "relative_path": relative.as_posix(),
                    "reason": ("exact duplicate of " + source_checksums[source_sha256]),
                }
            )
            if progress is not None:
                progress(index, len(paths), path)
            continue
        motion_fingerprint = str(archive["near_duplicate_fingerprint"])
        if motion_fingerprint in motion_fingerprints:
            quarantined.append(
                {
                    "relative_path": relative.as_posix(),
                    "reason": (
                        "near duplicate of " + motion_fingerprints[motion_fingerprint]
                    ),
                }
            )
            if progress is not None:
                progress(index, len(paths), path)
            continue
        source_checksums[source_sha256] = relative.as_posix()
        motion_fingerprints[motion_fingerprint] = relative.as_posix()
        subdataset = relative.parts[0]
        subject_id = relative.parts[1]
        original_sequence_id = path.stem.removesuffix("_stageii")
        session_id = "__".join(relative.parts[1:-1])
        sequence_id = _portable_identifier(relative.with_suffix("").parts)
        sequences.append(
            SourceSequence(
                source_kind="amass",
                source_id="amass",
                sequence_id=sequence_id,
                dataset_release=dataset_release.strip(),
                license_id=license_id.strip(),
                local_path=path,
                subject_id=subject_id,
                session_id=session_id,
                original_sequence_id=original_sequence_id,
                pose_representation=archive["pose_representation"],
                frame_rate=float(archive["frame_rate"]),
                requested_split="auto",
                sampling_weight=sampling_weight,
                source_sha256=source_sha256,
                subdataset=subdataset,
                near_duplicate_fingerprint=archive["near_duplicate_fingerprint"],
                metadata={
                    "relative_path": relative.as_posix(),
                    "surface_model_type": archive["surface_model_type"],
                    "gender": archive["gender"],
                    "frame_count": str(archive["frame_count"]),
                },
            )
        )
        if progress is not None:
            progress(index, len(paths), path)

    manifest = SourceManifest(
        schema_version=TWIST_DATASET_SCHEMA_VERSION,
        split_seed=split_seed,
        sequences=tuple(sequences),
    )
    validate_source_manifest(manifest, verify_files=False)
    payload = asdict(manifest)
    for sequence in payload["sequences"]:
        sequence["local_path"] = str(sequence["local_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(payload), output_path)
    scan_report_path = output_path.with_suffix(".scan_report.json")
    scan_report_path.write_text(
        json.dumps(
            {
                "source_root": str(source_root),
                "discovered": len(paths),
                "accepted": len(sequences),
                "rejected": rejected,
                "quarantined": quarantined,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _inspect_amass_archive(path: Path) -> dict[str, str | float | int]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            fields = set(archive.files)
            frame_rate_name = next(
                (
                    name
                    for name in ("mocap_frame_rate", "mocap_framerate")
                    if name in fields
                ),
                None,
            )
            if frame_rate_name is None:
                raise ValueError("missing mocap_frame_rate")
            if "trans" not in fields:
                raise ValueError("missing trans")
            has_standardized_pose = {"root_orient", "pose_body"} <= fields
            if not has_standardized_pose and "poses" not in fields:
                raise ValueError("missing root_orient/pose_body or poses")

            frame_rate = float(np.asarray(archive[frame_rate_name]).item())
            if not math.isfinite(frame_rate) or frame_rate <= 0.0:
                raise ValueError(f"invalid frame rate {frame_rate!r}")
            trans = np.asarray(archive["trans"], dtype=np.float32)
            if has_standardized_pose:
                root_orient = np.asarray(archive["root_orient"], dtype=np.float32)
                pose_body = np.asarray(archive["pose_body"], dtype=np.float32)
            else:
                poses = np.asarray(archive["poses"], dtype=np.float32)
                if poses.ndim != 2 or poses.shape[1] < 66:
                    raise ValueError("poses must have shape [frames, >=66]")
                root_orient = poses[:, :3]
                pose_body = poses[:, 3:66]
            frame_count = int(trans.shape[0])
            expected = (frame_count, 3)
            if trans.shape != expected or root_orient.shape != expected:
                raise ValueError(
                    "trans and root_orient must have matching [frames, 3] shapes"
                )
            if pose_body.ndim != 2 or pose_body.shape[0] != frame_count:
                raise ValueError("pose_body must have the same frame count as trans")
            if frame_count < 2:
                raise ValueError("motion contains fewer than two frames")
            if not all(
                np.all(np.isfinite(value)) for value in (trans, root_orient, pose_body)
            ):
                raise ValueError("motion contains non-finite pose values")

            surface = (
                str(np.asarray(archive["surface_model_type"]).item()).lower()
                if "surface_model_type" in fields
                else "smplx"
                if pose_body.shape[1] == 63
                else "smplh"
            )
            representation = {
                "smplx": "smpl-x",
                "smpl-x": "smpl-x",
                "smplh": "smpl-h",
                "smpl-h": "smpl-h",
            }.get(surface)
            if representation is None:
                raise ValueError(f"unsupported surface_model_type {surface!r}")
            gender = (
                str(np.asarray(archive["gender"]).item())
                if "gender" in fields
                else "unknown"
            )
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"invalid AMASS archive {path}: {error}") from error

    sample_indices = np.linspace(
        0, frame_count - 1, num=min(frame_count, 128), dtype=np.int64
    )
    relative_translation = trans[sample_indices] - trans[0]
    signature = np.concatenate(
        (root_orient[sample_indices], pose_body[sample_indices], relative_translation),
        axis=1,
    )
    quantized = np.rint(signature * 10_000.0).astype(np.int32)
    digest = hashlib.sha256()
    digest.update(str(frame_count).encode())
    digest.update(np.asarray(frame_rate, dtype=np.float64).tobytes())
    digest.update(quantized.tobytes())
    return {
        "pose_representation": representation,
        "surface_model_type": surface,
        "gender": gender,
        "frame_rate": frame_rate,
        "frame_count": frame_count,
        "near_duplicate_fingerprint": "amass-motion-v1:" + digest.hexdigest(),
    }


def _portable_identifier(parts: tuple[str, ...]) -> str:
    value = "__".join(parts)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-")
    if not value:
        raise ValueError(f"cannot derive a stable identifier from {parts!r}")
    return value


def validate_source_manifest(
    manifest: SourceManifest,
    *,
    body_model_root: Path | None = None,
    verify_files: bool = True,
) -> None:
    if manifest.schema_version != TWIST_DATASET_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported TWIST manifest schema {manifest.schema_version}; "
            f"expected {TWIST_DATASET_SCHEMA_VERSION}"
        )
    if not manifest.sequences:
        raise ValueError("TWIST source manifest contains no sequences")
    stable_ids: set[tuple[str, str]] = set()
    source_identities: set[tuple[str, str, str, str]] = set()
    checksums: dict[str, tuple[str, str]] = {}
    near_duplicates: dict[str, tuple[str, str]] = {}
    needs_smpl_models = False
    for sequence in manifest.sequences:
        _validate_identifier("source_id", sequence.source_id)
        _validate_identifier("sequence_id", sequence.sequence_id)
        key = (sequence.source_id, sequence.sequence_id)
        if key in stable_ids:
            raise ValueError(f"duplicate stable sequence identifier: {key}")
        stable_ids.add(key)
        if not sequence.dataset_release.strip():
            raise ValueError(f"{key}: dataset_release is required")
        if not sequence.license_id.strip():
            raise ValueError(f"{key}: license_id is required")
        for name in ("subject_id", "session_id", "original_sequence_id"):
            if not getattr(sequence, name).strip():
                raise ValueError(f"{key}: {name} is required")
        if not math.isfinite(sequence.frame_rate) or sequence.frame_rate <= 0.0:
            raise ValueError(f"{key}: frame_rate must be positive and finite")
        if (
            not math.isfinite(sequence.sampling_weight)
            or sequence.sampling_weight <= 0.0
        ):
            raise ValueError(f"{key}: sampling_weight must be positive and finite")
        pose = sequence.pose_representation.lower()
        if sequence.source_kind == "amass":
            if pose not in {"smpl-h", "smplh", "smpl-x", "smplx"}:
                raise ValueError(f"{key}: AMASS requires SMPL-H or SMPL-X poses")
            if not sequence.subdataset:
                raise ValueError(f"{key}: AMASS subdataset is required")
        elif sequence.source_kind == "omomo":
            if pose not in {"smpl-x", "smplx"}:
                raise ValueError(f"{key}: OMOMO requires a SMPL-X trajectory")
        elif sequence.source_kind == "in_house":
            if pose not in {"smpl-x", "smplx", "bvh"}:
                raise ValueError(f"{key}: in-house data requires SMPL-X or BVH")
            for field_name in ("owner", "ownership_basis"):
                if not sequence.metadata.get(field_name):
                    raise ValueError(
                        f"{key}: in-house metadata.{field_name} is required"
                    )
            if pose == "bvh" and not sequence.metadata.get("bvh_skeleton"):
                raise ValueError(f"{key}: in-house BVH requires metadata.bvh_skeleton")
        else:
            raise ValueError(f"{key}: unsupported source_kind {sequence.source_kind!r}")
        needs_smpl_models |= pose in {"smpl-h", "smplh", "smpl-x", "smplx"}
        checksum = _normalize_sha256(sequence.source_sha256, f"{key} source_sha256")
        if checksum in checksums:
            other = checksums[checksum]
            raise ValueError(
                f"duplicate source content for {other} and {key}: sha256:{checksum}"
            )
        checksums[checksum] = key
        if sequence.near_duplicate_fingerprint:
            fingerprint = sequence.near_duplicate_fingerprint.strip()
            if fingerprint in near_duplicates:
                other = near_duplicates[fingerprint]
                raise ValueError(
                    f"near-duplicate source sequences detected: {other} and {key}"
                )
            near_duplicates[fingerprint] = key
        identity = (
            sequence.source_id,
            sequence.subject_id,
            sequence.session_id,
            sequence.original_sequence_id,
        )
        if identity in source_identities:
            raise ValueError(f"duplicate source identity: {identity}")
        source_identities.add(identity)
        if verify_files:
            if not sequence.local_path.is_file():
                raise FileNotFoundError(
                    f"{key}: source file not found: {sequence.local_path}"
                )
            actual = _file_sha256(sequence.local_path)
            if actual != checksum:
                raise ValueError(
                    f"{key}: source checksum mismatch; expected sha256:{checksum}, "
                    f"got sha256:{actual}"
                )
    if verify_files and needs_smpl_models:
        _validate_body_model_root(body_model_root)


def assign_splits(manifest: SourceManifest) -> tuple[AssignedSequence, ...]:
    """Assign immutable train/evaluation splits before conversion.

    Explicit (normally official) splits win. Automatic entries are grouped by
    source and subject so a subject cannot leak between train and evaluation.
    """

    groups: dict[tuple[str, str], list[SourceSequence]] = {}
    for sequence in manifest.sequences:
        groups.setdefault((sequence.source_id, sequence.subject_id), []).append(
            sequence
        )
    explicit: dict[tuple[str, str], AssignedSplit] = {}
    automatic_groups: list[tuple[str, str]] = []
    for group, sequences in groups.items():
        requested = {
            item.requested_split for item in sequences if item.requested_split != "auto"
        }
        if len(requested) > 1:
            raise ValueError(
                f"subject group {group} requests both train and evaluation splits"
            )
        if requested:
            explicit[group] = requested.pop()  # type: ignore[assignment]
        else:
            automatic_groups.append(group)
    ordered = sorted(
        automatic_groups,
        key=lambda group: _split_score(manifest.split_seed, group),
    )
    evaluation_count = math.ceil(len(ordered) * 0.10) if len(ordered) > 1 else 0
    auto_eval = set(ordered[:evaluation_count])
    assigned = []
    for sequence in manifest.sequences:
        group = (sequence.source_id, sequence.subject_id)
        split: AssignedSplit = explicit.get(
            group, "evaluation" if group in auto_eval else "train"
        )
        assigned.append(AssignedSequence(sequence, split))
    validate_split_isolation(assigned)
    return tuple(assigned)


def validate_split_isolation(assignments: tuple[AssignedSequence, ...]) -> None:
    identities: dict[tuple[str, str, str, str], AssignedSplit] = {}
    contents: dict[str, AssignedSplit] = {}
    near_duplicates: dict[str, AssignedSplit] = {}
    for assigned in assignments:
        sequence = assigned.sequence
        identity = (
            sequence.source_id,
            sequence.subject_id,
            sequence.session_id,
            sequence.original_sequence_id,
        )
        checksum = _normalize_sha256(sequence.source_sha256, "source_sha256")
        _record_split(identities, identity, assigned.split, "source identity")
        _record_split(contents, checksum, assigned.split, "source content")
        if sequence.near_duplicate_fingerprint:
            _record_split(
                near_duplicates,
                sequence.near_duplicate_fingerprint,
                assigned.split,
                "near-duplicate content",
            )


def portable_manifest(assignments: tuple[AssignedSequence, ...]) -> dict[str, Any]:
    return {
        "schema_version": TWIST_DATASET_SCHEMA_VERSION,
        "sequences": [
            {
                **{
                    key: value
                    for key, value in asdict(item.sequence).items()
                    if key != "local_path"
                },
                "source_sha256": (
                    "sha256:"
                    + _normalize_sha256(item.sequence.source_sha256, "source_sha256")
                ),
                "assigned_split": item.split,
            }
            for item in sorted(
                assignments,
                key=lambda value: (
                    value.sequence.source_id,
                    value.sequence.sequence_id,
                ),
            )
        ],
    }


def portable_manifest_fingerprint(
    assignments: tuple[AssignedSequence, ...],
) -> str:
    return _json_fingerprint(portable_manifest(assignments))


def build_packed_dataset(
    manifest_path: Path,
    retargeted_root: Path,
    output_root: Path,
    *,
    body_model_root: Path | None,
    quality: QualityConfig = QualityConfig(),
    robot_model_path: Path,
    gmr_root: Path,
) -> dict[str, Any]:
    """Validate, normalize, and pack retargeted clips into NPZ shards."""

    manifest = load_source_manifest(
        manifest_path, body_model_root=body_model_root, verify_files=True
    )
    assignments = assign_splits(manifest)
    retargeted_root = Path(retargeted_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"TWIST dataset output is not empty: {output_root}; builds never overwrite"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    robot_model_path = Path(robot_model_path).resolve()
    if not robot_model_path.is_file():
        raise FileNotFoundError(f"canonical Bello model not found: {robot_model_path}")
    gmr_revision = _git_revision(gmr_root)
    robot_fingerprint = f"sha256:{_file_sha256(robot_model_path)}"
    gmr_config_path = (
        Path(gmr_root).resolve()
        / "general_motion_retargeting"
        / "ik_configs"
        / "smplx_to_bello.json"
    )
    gmr_config_fingerprint = f"sha256:{_file_sha256(gmr_config_path)}"
    decisions: list[ClipDecision] = []
    accepted: list[tuple[AssignedSequence, dict[str, np.ndarray], str]] = []
    for assigned in assignments:
        sequence = assigned.sequence
        path = retargeted_root / f"{sequence.source_id}__{sequence.sequence_id}.npz"
        if not path.is_file():
            decisions.append(
                ClipDecision(
                    sequence.source_id,
                    sequence.sequence_id,
                    "rejected",
                    ("retargeted clip is missing",),
                )
            )
            continue
        try:
            arrays = load_and_normalize_retargeted_clip(path, quality)
            _validate_retargeted_provenance(
                arrays,
                sequence,
                gmr_revision=gmr_revision,
                gmr_config_fingerprint=gmr_config_fingerprint,
                robot_fingerprint=robot_fingerprint,
            )
            reasons = clip_quality_reasons(arrays, quality)
        except (OSError, ValueError, KeyError) as error:
            decisions.append(
                ClipDecision(
                    sequence.source_id,
                    sequence.sequence_id,
                    "rejected",
                    (str(error),),
                )
            )
            continue
        digest = _arrays_fingerprint(arrays)
        frames = int(arrays["qpos"].shape[0])
        duration = float(arrays["timestamp_seconds"][-1])
        if reasons:
            decisions.append(
                ClipDecision(
                    sequence.source_id,
                    sequence.sequence_id,
                    "quarantined",
                    tuple(reasons),
                    frames,
                    duration,
                    digest,
                )
            )
            continue
        decisions.append(
            ClipDecision(
                sequence.source_id,
                sequence.sequence_id,
                "accepted",
                (),
                frames,
                duration,
                digest,
            )
        )
        accepted.append((assigned, arrays, digest))
    if not accepted:
        _write_report(output_root / "conversion_report.json", decisions)
        raise ValueError("TWIST dataset build accepted no clips; see conversion report")
    _validate_retargeted_split_hashes(accepted)
    heldout = portable_manifest(
        tuple(item for item in assignments if item.split == "evaluation")
    )
    _write_immutable_json(output_root / "heldout_manifest.json", heldout)
    shard_entries, joint_names, body_names = _write_shards(
        output_root, accepted, quality.max_frames_per_shard
    )
    manifest_portable = portable_manifest(assignments)
    fingerprint_payload = {
        "schema_version": TWIST_DATASET_SCHEMA_VERSION,
        "portable_manifest": manifest_portable,
        "accepted_clip_hashes": sorted(
            {
                f"{item.sequence.source_id}/{item.sequence.sequence_id}": digest
                for item, _, digest in accepted
            }.items()
        ),
        "quality": asdict(quality),
        "robot_fingerprint": robot_fingerprint,
        "gmr_revision": gmr_revision,
        "gmr_config_fingerprint": gmr_config_fingerprint,
    }
    dataset_fingerprint = _json_fingerprint(fingerprint_payload)
    index = {
        "schema_version": TWIST_DATASET_SCHEMA_VERSION,
        "dataset_fingerprint": dataset_fingerprint,
        "portable_manifest_fingerprint": _json_fingerprint(manifest_portable),
        "robot_fingerprint": robot_fingerprint,
        "gmr_revision": gmr_revision,
        "gmr_configuration": (
            "general_motion_retargeting/ik_configs/smplx_to_bello.json"
        ),
        "gmr_config_fingerprint": gmr_config_fingerprint,
        "policy_fps": quality.policy_fps,
        "future_offsets": list(TWIST_FUTURE_OFFSETS),
        "joint_names": list(joint_names),
        "body_names": list(body_names),
        "clips": shard_entries,
    }
    _write_json(output_root / "index.json", index)
    _write_report(output_root / "conversion_report.json", decisions)
    return index


def materialize_packed_clip(
    index_path: Path,
    output_path: Path,
    *,
    split: AssignedSplit,
    sequence_id: str | None = None,
    source_id: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Materialize one canonical motion clip for native/MJX environment setup.

    The packed index remains the source of truth. This helper is intentionally
    deterministic and is primarily used for evaluation and bounded CPU smoke
    runs; large training jobs consume the packed sampler directly.
    """

    index_path = Path(index_path).expanduser().resolve()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schema_version") != TWIST_DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported TWIST packed dataset schema")
    candidates = [
        clip
        for clip in index["clips"]
        if clip["split"] == split
        and (sequence_id is None or clip["sequence_id"] == sequence_id)
        and (source_id is None or clip["source_id"] == source_id)
    ]
    if not candidates:
        raise ValueError(
            f"TWIST dataset contains no {split} clip"
            + (f" named {sequence_id!r}" if sequence_id else "")
        )
    clip = sorted(
        candidates, key=lambda value: (value["source_id"], value["sequence_id"])
    )[0]
    shard_path = index_path.parent / clip["shard"]
    start = int(clip["offset"])
    stop = start + int(clip["frames"])
    with np.load(shard_path, allow_pickle=False) as shard:
        arrays = {name: np.asarray(shard[name][start:stop]) for name in shard.files}
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        format_version=np.asarray(1, dtype=np.int32),
        fps=np.asarray(index["policy_fps"], dtype=np.float32),
        joint_names=np.asarray(index["joint_names"]),
        body_names=np.asarray(index["body_names"]),
        source_sha256=np.asarray(
            _normalize_sha256(clip["source_sha256"], "source_sha256")
        ),
        qpos=arrays["qpos"],
        qvel=arrays["qvel"],
        body_pos=arrays["body_pos"],
        body_quat_wxyz=arrays["body_quat_wxyz"],
        body_lin_vel=arrays["body_lin_vel"],
        body_ang_vel=arrays["body_ang_vel"],
        foot_contacts=arrays["foot_contacts"],
    )
    return output_path, clip


@dataclass(frozen=True)
class PackedReferenceBank:
    """Concatenated JAX-friendly arrays and immutable per-clip metadata."""

    qpos: np.ndarray
    qvel: np.ndarray
    body_pos: np.ndarray
    body_quat_wxyz: np.ndarray
    body_lin_vel: np.ndarray
    body_ang_vel: np.ndarray
    foot_contacts: np.ndarray
    offsets: np.ndarray
    lengths: np.ndarray
    sampling_probabilities: np.ndarray
    source_ids: tuple[str, ...]
    sequence_ids: tuple[str, ...]


def load_packed_reference_bank(
    index_path: Path, *, split: AssignedSplit
) -> PackedReferenceBank:
    """Load a split into concatenated arrays suitable for one device transfer."""

    index_path = Path(index_path).expanduser().resolve()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    clips = [clip for clip in index["clips"] if clip["split"] == split]
    if not clips:
        raise ValueError(f"TWIST packed dataset has no {split} clips")
    array_names = (
        "qpos",
        "qvel",
        "body_pos",
        "body_quat_wxyz",
        "body_lin_vel",
        "body_ang_vel",
        "foot_contacts",
    )
    lengths = np.asarray([int(clip["frames"]) for clip in clips], dtype=np.int32)
    total_frames = int(lengths.sum(dtype=np.int64))
    if total_frames > np.iinfo(np.int32).max:
        raise ValueError("TWIST reference bank exceeds the JAX int32 index range")
    offsets = np.zeros(len(clips), dtype=np.int32)
    if len(clips) > 1:
        offsets[1:] = np.cumsum(lengths[:-1], dtype=np.int64).astype(np.int32)

    first = clips[0]
    with np.load(index_path.parent / first["shard"], allow_pickle=False) as shard:
        packed = {
            name: np.empty(
                (total_frames,) + shard[name].shape[1:], dtype=shard[name].dtype
            )
            for name in array_names
        }

    clips_by_shard: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for clip_index, clip in enumerate(clips):
        clips_by_shard.setdefault(clip["shard"], []).append((clip_index, clip))
    for shard_name, shard_clips in clips_by_shard.items():
        with np.load(index_path.parent / shard_name, allow_pickle=False) as shard:
            for clip_index, clip in shard_clips:
                source_start = int(clip["offset"])
                frames = int(lengths[clip_index])
                source_stop = source_start + frames
                target_start = int(offsets[clip_index])
                target_stop = target_start + frames
                for name in array_names:
                    value = np.asarray(shard[name][source_start:source_stop])
                    expected_shape = (frames,) + packed[name].shape[1:]
                    if value.shape != expected_shape:
                        raise ValueError(
                            f"TWIST clip {clip['source_id']}/{clip['sequence_id']} "
                            f"has invalid {name} shape {value.shape}; expected "
                            f"{expected_shape}"
                        )
                    packed[name][target_start:target_stop] = value
    weights = np.asarray(
        [clip["frames"] * clip["sampling_weight"] for clip in clips], dtype=np.float64
    )
    if split == "evaluation":
        weights = np.ones_like(weights)
    probabilities = weights / weights.sum()
    return PackedReferenceBank(
        **packed,
        offsets=offsets,
        lengths=lengths,
        sampling_probabilities=probabilities.astype(np.float32),
        source_ids=tuple(clip["source_id"] for clip in clips),
        sequence_ids=tuple(clip["sequence_id"] for clip in clips),
    )


def sample_packed_start_frames(
    rng,
    lengths,
    probabilities,
    *,
    batch_size: int,
    future_horizon_steps: int = max(TWIST_FUTURE_OFFSETS),
):
    """JAX-compatible weighted clip/start sampler without cross-clip windows."""

    import jax
    import jax.numpy as jnp

    clip_key, frame_key = jax.random.split(rng)
    logits = jnp.log(jnp.asarray(probabilities))
    clip_indices = jax.random.categorical(clip_key, logits, shape=(batch_size,)).astype(
        jnp.int32
    )
    maximum_start = jnp.maximum(
        jnp.asarray(lengths)[clip_indices] - future_horizon_steps - 1, 0
    )
    uniform = jax.random.uniform(frame_key, (batch_size,))
    start_frames = jnp.floor(uniform * (maximum_start + 1)).astype(jnp.int32)
    return clip_indices, start_frames


def load_and_normalize_retargeted_clip(
    path: Path, quality: QualityConfig
) -> dict[str, np.ndarray]:
    required = {
        "timestamp_seconds",
        "qpos",
        "qvel",
        "body_pos",
        "body_quat_wxyz",
        "body_lin_vel",
        "body_ang_vel",
        "foot_contacts",
        "joint_names",
        "body_names",
        "source_timestamp_seconds",
        "source_sha256",
        "retarget_format_version",
        "gmr_revision",
        "gmr_config_sha256",
        "robot_fingerprint",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"{path.name} is missing fields: {missing}")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if int(np.asarray(arrays["retarget_format_version"]).item()) != 4:
        raise ValueError(f"{path.name}: unsupported retarget format version")
    frames = arrays["qpos"].shape[0]
    for name in (
        "timestamp_seconds",
        "source_timestamp_seconds",
        "qvel",
        "body_pos",
        "body_quat_wxyz",
        "body_lin_vel",
        "body_ang_vel",
        "foot_contacts",
    ):
        if arrays[name].shape[0] != frames:
            raise ValueError(
                f"{path.name}: {name} has {arrays[name].shape[0]} frames, expected {frames}"
            )
    if frames < 2:
        raise ValueError(f"{path.name}: retargeted clip has fewer than two frames")
    timestamps = np.asarray(arrays["timestamp_seconds"], dtype=np.float64)
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError(f"{path.name}: timestamps must be finite and increasing")
    normalized = _resample_clip(arrays, quality.policy_fps)
    return _align_floor_and_heading(normalized)


def clip_quality_reasons(
    arrays: dict[str, np.ndarray], quality: QualityConfig
) -> list[str]:
    reasons = []
    numeric_names = (
        "timestamp_seconds",
        "source_timestamp_seconds",
        "qpos",
        "qvel",
        "body_pos",
        "body_quat_wxyz",
        "body_lin_vel",
        "body_ang_vel",
        "foot_contacts",
    )
    for name in numeric_names:
        if not np.all(np.isfinite(arrays[name])):
            return [f"{name} contains non-finite values"]
    frames = arrays["qpos"].shape[0]
    if frames <= quality.future_horizon_steps:
        reasons.append(
            f"clip has {frames} frames but future horizon requires "
            f"{quality.future_horizon_steps + 1}"
        )
    dt = np.diff(arrays["timestamp_seconds"])
    expected_dt = 1.0 / quality.policy_fps
    if np.max(dt) > quality.max_timestamp_gap_factor * expected_dt:
        reasons.append("timestamp gap exceeds configured limit")
    quaternions = np.concatenate(
        (
            arrays["qpos"][:, 3:7, None],
            np.swapaxes(arrays["body_quat_wxyz"], 1, 2),
        ),
        axis=2,
    )
    norms = np.linalg.norm(quaternions, axis=1)
    if np.max(np.abs(norms - 1.0)) > quality.quaternion_norm_tolerance:
        reasons.append("quaternion norm exceeds configured tolerance")
    normalized = quaternions / np.maximum(norms[:, None, :], 1.0e-12)
    dots = np.abs(np.sum(normalized[1:] * normalized[:-1], axis=1))
    max_step = 2.0 * np.arccos(np.clip(np.min(dots), -1.0, 1.0))
    if max_step > quality.max_quaternion_step_radians:
        reasons.append("quaternion trajectory is discontinuous")
    if np.max(np.linalg.norm(arrays["qvel"][:, :3], axis=-1)) > (
        quality.max_root_linear_velocity
    ):
        reasons.append("root linear velocity exceeds configured limit")
    if np.max(np.linalg.norm(arrays["qvel"][:, 3:6], axis=-1)) > (
        quality.max_root_angular_velocity
    ):
        reasons.append("root angular velocity exceeds configured limit")
    if np.max(np.linalg.norm(arrays["body_lin_vel"], axis=-1)) > (
        quality.max_body_linear_velocity
    ):
        reasons.append("body linear velocity exceeds configured limit")
    if np.max(np.linalg.norm(arrays["body_ang_vel"], axis=-1)) > (
        quality.max_body_angular_velocity
    ):
        reasons.append("body angular velocity exceeds configured limit")
    root_height = arrays["qpos"][:, 2]
    if root_height.min() < quality.min_root_height:
        reasons.append("root height falls below configured limit")
    if root_height.max() > quality.max_root_height:
        reasons.append("root height exceeds configured limit")
    if arrays["body_pos"][..., 2].min() < -quality.max_floor_penetration:
        reasons.append("body floor penetration exceeds configured limit")
    if {"joint_lower", "joint_upper"} <= arrays.keys():
        joint = arrays["qpos"][:, 7:]
        lower = np.asarray(arrays["joint_lower"])
        upper = np.asarray(arrays["joint_upper"])
        violations = (joint < lower - quality.max_joint_limit_violation_radians) | (
            joint > upper + quality.max_joint_limit_violation_radians
        )
        if np.mean(violations) > quality.max_joint_limit_violation_fraction:
            reasons.append("persistent joint-limit violations")
    return reasons


def _validate_retargeted_provenance(
    arrays: dict[str, np.ndarray],
    sequence: SourceSequence,
    *,
    gmr_revision: str,
    gmr_config_fingerprint: str,
    robot_fingerprint: str,
) -> None:
    expected = {
        "source_sha256": _normalize_sha256(sequence.source_sha256, "source_sha256"),
        "gmr_revision": gmr_revision,
        "gmr_config_sha256": gmr_config_fingerprint,
        "robot_fingerprint": robot_fingerprint,
    }
    actual = {name: str(np.asarray(arrays[name]).item()) for name in expected}
    actual["source_sha256"] = _normalize_sha256(
        actual["source_sha256"], "retargeted source_sha256"
    )
    if not actual["gmr_config_sha256"].startswith("sha256:"):
        actual["gmr_config_sha256"] = "sha256:" + actual["gmr_config_sha256"]
    if not actual["robot_fingerprint"].startswith("sha256:"):
        actual["robot_fingerprint"] = "sha256:" + actual["robot_fingerprint"]
    mismatches = [name for name in expected if actual[name] != expected[name]]
    if mismatches:
        raise ValueError("retargeted provenance mismatch for " + ", ".join(mismatches))


def _write_shards(
    output_root: Path,
    clips: list[tuple[AssignedSequence, dict[str, np.ndarray], str]],
    max_frames: int,
) -> tuple[list[dict[str, Any]], tuple[str, ...], tuple[str, ...]]:
    if max_frames <= 0:
        raise ValueError("max_frames_per_shard must be positive")
    entries: list[dict[str, Any]] = []
    shard_clips: list[tuple[AssignedSequence, dict[str, np.ndarray], str]] = []
    shard_frames = 0
    shard_index = 0
    joint_names: tuple[str, ...] | None = None
    body_names: tuple[str, ...] | None = None
    for clip in clips:
        names = tuple(str(value) for value in clip[1]["joint_names"].tolist())
        bodies = tuple(str(value) for value in clip[1]["body_names"].tolist())
        if joint_names is None:
            joint_names, body_names = names, bodies
        elif names != joint_names or bodies != body_names:
            raise ValueError("accepted clips do not share one joint/body ordering")
        frames = int(clip[1]["qpos"].shape[0])
        if shard_clips and shard_frames + frames > max_frames:
            entries.extend(_flush_shard(output_root, shard_index, shard_clips))
            shard_index += 1
            shard_clips = []
            shard_frames = 0
        shard_clips.append(clip)
        shard_frames += frames
    if shard_clips:
        entries.extend(_flush_shard(output_root, shard_index, shard_clips))
    assert joint_names is not None and body_names is not None
    total_train_frames = sum(
        entry["frames"] for entry in entries if entry["split"] == "train"
    )
    for entry in entries:
        if entry["split"] == "train":
            weighted = entry["frames"] * entry["sampling_weight"]
            entry["sampling_probability_numerator"] = weighted
            entry["sampling_probability_denominator"] = sum(
                item["frames"] * item["sampling_weight"]
                for item in entries
                if item["split"] == "train"
            )
        else:
            entry["evaluation_order"] = sum(
                item["split"] == "evaluation"
                for item in entries[: entries.index(entry)]
            )
    if total_train_frames <= 0:
        raise ValueError("packed dataset contains no training frames")
    return entries, joint_names, body_names


def _flush_shard(
    output_root: Path,
    shard_index: int,
    clips: list[tuple[AssignedSequence, dict[str, np.ndarray], str]],
) -> list[dict[str, Any]]:
    shard_name = f"shard_{shard_index:05d}.npz"
    pack_names = (
        "timestamp_seconds",
        "source_timestamp_seconds",
        "qpos",
        "qvel",
        "body_pos",
        "body_quat_wxyz",
        "body_lin_vel",
        "body_ang_vel",
        "foot_contacts",
    )
    packed = {
        name: np.concatenate([clip[1][name] for clip in clips], axis=0)
        for name in pack_names
    }
    np.savez_compressed(output_root / shard_name, **packed)
    offset = 0
    entries = []
    for assigned, arrays, digest in clips:
        frames = int(arrays["qpos"].shape[0])
        sequence = assigned.sequence
        entries.append(
            {
                "source_kind": sequence.source_kind,
                "source_id": sequence.source_id,
                "sequence_id": sequence.sequence_id,
                "subject_id": sequence.subject_id,
                "split": assigned.split,
                "sampling_weight": sequence.sampling_weight,
                "shard": shard_name,
                "offset": offset,
                "frames": frames,
                "duration_seconds": float(arrays["timestamp_seconds"][-1]),
                "source_sha256": "sha256:"
                + _normalize_sha256(sequence.source_sha256, "source_sha256"),
                "retargeted_sha256": digest,
                "object_metadata": (
                    sequence.metadata if sequence.source_kind == "omomo" else {}
                ),
            }
        )
        offset += frames
    return entries


def _resample_clip(
    arrays: dict[str, np.ndarray], target_fps: float
) -> dict[str, np.ndarray]:
    source_time = np.asarray(arrays["timestamp_seconds"], dtype=np.float64)
    duration = float(source_time[-1] - source_time[0])
    target_count = math.floor(duration * target_fps + 1.0e-8) + 1
    target_time = np.arange(target_count, dtype=np.float64) / target_fps
    absolute_target = target_time + source_time[0]
    result: dict[str, np.ndarray] = {}
    quaternion_names = {"body_quat_wxyz"}
    for name, values in arrays.items():
        values = np.asarray(values)
        if name in {
            "timestamp_seconds",
            "source_timestamp_seconds",
            "source_sha256",
            "retarget_format_version",
            "joint_names",
            "body_names",
            "gmr_revision",
            "gmr_config_sha256",
            "robot_fingerprint",
            "joint_lower",
            "joint_upper",
        }:
            continue
        if values.shape[0] != source_time.shape[0]:
            continue
        if name == "qpos":
            interpolated = _interpolate_linear(source_time, values, absolute_target)
            interpolated[:, 3:7] = _interpolate_quaternions(
                source_time, values[:, 3:7], absolute_target
            )
        elif name in quaternion_names:
            flat = values.reshape((values.shape[0], -1, 4))
            interpolated = np.stack(
                [
                    _interpolate_quaternions(
                        source_time, flat[:, index], absolute_target
                    )
                    for index in range(flat.shape[1])
                ],
                axis=1,
            ).reshape((target_count,) + values.shape[1:])
        elif name == "foot_contacts":
            indices = np.searchsorted(source_time, absolute_target, side="left")
            indices = np.clip(indices, 0, len(source_time) - 1)
            interpolated = values[indices]
        else:
            interpolated = _interpolate_linear(source_time, values, absolute_target)
        result[name] = interpolated.astype(np.float32)
    result["timestamp_seconds"] = target_time.astype(np.float64)
    result["source_timestamp_seconds"] = absolute_target.astype(np.float64)
    for name in (
        "joint_names",
        "body_names",
        "source_sha256",
        "retarget_format_version",
        "gmr_revision",
        "gmr_config_sha256",
        "robot_fingerprint",
        "joint_lower",
        "joint_upper",
    ):
        if name in arrays:
            result[name] = arrays[name]
    return result


def _align_floor_and_heading(
    arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    result = {name: np.array(value, copy=True) for name, value in arrays.items()}
    root_quat = result["qpos"][0, 3:7]
    yaw = math.atan2(
        2.0 * (root_quat[0] * root_quat[3] + root_quat[1] * root_quat[2]),
        1.0 - 2.0 * (root_quat[2] ** 2 + root_quat[3] ** 2),
    )
    rotation = np.asarray(
        [
            [math.cos(-yaw), -math.sin(-yaw), 0.0],
            [math.sin(-yaw), math.cos(-yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    initial_xy = result["qpos"][0, :2].copy()
    result["qpos"][:, :2] -= initial_xy
    result["qpos"][:, :3] = result["qpos"][:, :3] @ rotation.T
    result["body_pos"][..., :2] -= initial_xy
    result["body_pos"] = result["body_pos"] @ rotation.T
    result["qvel"][:, :3] = result["qvel"][:, :3] @ rotation.T
    result["body_lin_vel"] = result["body_lin_vel"] @ rotation.T
    yaw_quat = np.asarray([math.cos(-yaw / 2.0), 0.0, 0.0, math.sin(-yaw / 2.0)])
    result["qpos"][:, 3:7] = _quat_multiply(
        np.broadcast_to(yaw_quat, result["qpos"][:, 3:7].shape),
        result["qpos"][:, 3:7],
    )
    result["body_quat_wxyz"] = _quat_multiply(
        np.broadcast_to(yaw_quat, result["body_quat_wxyz"].shape),
        result["body_quat_wxyz"],
    )
    body_names = tuple(str(value) for value in result["body_names"].tolist())
    foot_indices = [
        index
        for index, name in enumerate(body_names)
        if "ankle" in name or "foot" in name
    ]
    floor_source = (
        result["body_pos"][:, foot_indices, 2]
        if foot_indices
        else result["body_pos"][..., 2]
    )
    floor_height = float(np.quantile(floor_source, 0.01))
    result["qpos"][:, 2] -= floor_height
    result["body_pos"][..., 2] -= floor_height
    return result


def _interpolate_linear(
    source_time: np.ndarray, values: np.ndarray, target_time: np.ndarray
) -> np.ndarray:
    flat = values.reshape((values.shape[0], -1))
    columns = [
        np.interp(target_time, source_time, flat[:, index])
        for index in range(flat.shape[1])
    ]
    return np.stack(columns, axis=-1).reshape((len(target_time),) + values.shape[1:])


def _interpolate_quaternions(
    source_time: np.ndarray, values: np.ndarray, target_time: np.ndarray
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).copy()
    values /= np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1.0e-12)
    for index in range(1, len(values)):
        if np.dot(values[index - 1], values[index]) < 0.0:
            values[index] *= -1.0
    upper = np.searchsorted(source_time, target_time, side="right")
    upper = np.clip(upper, 1, len(source_time) - 1)
    lower = upper - 1
    denominator = source_time[upper] - source_time[lower]
    alpha = (target_time - source_time[lower]) / denominator
    q0, q1 = values[lower], values[upper]
    dot = np.sum(q0 * q1, axis=-1)
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    near = sin_theta < 1.0e-6
    denominator = np.where(near, 1.0, sin_theta)
    weight0 = np.where(
        near,
        1.0 - alpha,
        np.sin((1.0 - alpha) * theta) / denominator,
    )
    weight1 = np.where(
        near,
        alpha,
        np.sin(alpha * theta) / denominator,
    )
    out = weight0[:, None] * q0 + weight1[:, None] * q1
    return out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1.0e-12)


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
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


def _resolve_sequence(sequence: SourceSequence, root: Path) -> SourceSequence:
    path = sequence.local_path.expanduser()
    if not path.is_absolute():
        path = root / path
    return SourceSequence(**{**asdict(sequence), "local_path": path.resolve()})


def _validate_body_model_root(root: Path | None) -> None:
    if root is None:
        raise ValueError(
            "body_model_root is required for SMPL-H/SMPL-X sources; "
            "datasets are not downloaded automatically"
        )
    root = Path(root).expanduser().resolve()
    expected = (
        root / "smplx" / "SMPLX_NEUTRAL.npz",
        root / "smplx" / "SMPLX_FEMALE.npz",
        root / "smplx" / "SMPLX_MALE.npz",
    )
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "required SMPL-X body models are missing: " + ", ".join(missing)
        )


def _validate_identifier(name: str, value: str) -> None:
    if _ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must match {_ID_PATTERN.pattern!r}, got {value!r}")


def _normalize_sha256(value: str, name: str) -> str:
    match = _SHA256_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return match.group(1)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _arrays_fingerprint(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.asarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes(order="C"))
    return "sha256:" + digest.hexdigest()


def _json_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _split_score(seed: int, group: tuple[str, str]) -> str:
    return hashlib.sha256(f"{seed}\0{group[0]}\0{group[1]}".encode()).hexdigest()


def _record_split(
    values: dict[Any, AssignedSplit],
    key: Any,
    split: AssignedSplit,
    label: str,
) -> None:
    previous = values.setdefault(key, split)
    if previous != split:
        raise ValueError(f"split leakage: {label} {key!r} occurs in both splits")


def _git_revision(root: Path) -> str:
    root = Path(root).resolve()
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError(f"invalid GMR revision from {root}: {revision!r}")
    return revision


def _validate_retargeted_split_hashes(
    accepted: list[tuple[AssignedSequence, dict[str, np.ndarray], str]],
) -> None:
    seen: dict[str, AssignedSplit] = {}
    for assigned, _, digest in accepted:
        _record_split(seen, digest, assigned.split, "retargeted content")


def _write_report(path: Path, decisions: list[ClipDecision]) -> None:
    counts = {
        status: sum(decision.status == status for decision in decisions)
        for status in ("accepted", "quarantined", "rejected")
    }
    _write_json(
        path,
        {
            "schema_version": TWIST_DATASET_SCHEMA_VERSION,
            "counts": counts,
            "sequences": [asdict(decision) for decision in decisions],
        },
    )


def _write_immutable_json(path: Path, value: Any) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"held-out manifest is immutable and differs from {path}")
        return
    path.write_text(encoded, encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
