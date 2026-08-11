"""Resumable generation of synchronized G1/SMPL SONIC sidecars."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable

import numpy as np

from bello_mujoco.motion.sonic_dataset import (
    SONIC_G1_RETARGET_FORMAT_VERSION,
    SONIC_SMPL_COORDINATE_CONVENTION,
    load_sonic_g1_retarget,
)
from bello_mujoco.motion.twist_dataset import SourceSequence, load_source_manifest

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class SonicRetargetDecision:
    source_id: str
    sequence_id: str
    status: str
    output_path: str
    message: str = ""


@dataclass(frozen=True)
class SonicBatchRetargetReport:
    manifest_path: str
    output_root: str
    normalized_root: str
    gmr_revision: str
    selected: int
    completed: int
    skipped: int
    failed: int
    decisions: tuple[SonicRetargetDecision, ...]


def batch_retarget_sonic_manifest(
    manifest_path: Path,
    *,
    body_model_root: Path,
    output_root: Path,
    normalized_root: Path,
    gmr_python: Path,
    jobs: int = 1,
    overwrite: bool = False,
    keep_normalized: bool = False,
    limit: int | None = None,
    command_runner: CommandRunner = subprocess.run,
    gmr_environment: dict[str, str] | None = None,
    progress: Callable[[int, int, SonicRetargetDecision], None] | None = None,
) -> SonicBatchRetargetReport:
    """Normalize each manifest source and atomically create its G1/SMPL view."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    body_model_root = Path(body_model_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    normalized_root = Path(normalized_root).expanduser().resolve()
    gmr_python = Path(os.path.abspath(Path(gmr_python).expanduser()))
    if output_root == normalized_root:
        raise ValueError("SONIC output and normalized roots must be different")
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise ValueError("jobs must be a positive integer")
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
    ):
        raise ValueError("limit must be a positive integer when provided")
    if not gmr_python.is_file():
        raise FileNotFoundError(f"GMR Python interpreter not found: {gmr_python}")
    _validate_body_models(body_model_root)
    manifest = load_source_manifest(manifest_path, verify_files=False)
    sequences = tuple(
        sorted(
            manifest.sequences,
            key=lambda item: (item.source_id, item.sequence_id),
        )[:limit]
    )
    unsupported = [
        sequence
        for sequence in sequences
        if sequence.pose_representation.lower()
        not in {"smpl-x", "smplx", "smpl-h", "smplh"}
    ]
    if unsupported:
        raise ValueError(
            "SONIC G1 retargeting requires SMPL-H/SMPL-X sources: "
            + ", ".join(item.sequence_id for item in unsupported[:5])
        )
    missing = [item.local_path for item in sequences if not item.local_path.is_file()]
    if missing:
        raise FileNotFoundError(f"manifest source file not found: {missing[0]}")
    changed = [
        item.local_path
        for item in sequences
        if _sha256(item.local_path) != item.source_sha256.removeprefix("sha256:")
    ]
    if changed:
        raise ValueError(f"manifest source digest has changed: {changed[0]}")

    repository = Path(__file__).resolve().parents[3]
    gmr_root = repository / "deps" / "GMR"
    if gmr_environment is None:
        gmr_environment = _nix_develop_environment(gmr_root)
    normalize_tool = repository / "tools" / "twist_normalize_smpl.py"
    retarget_tool = repository / "tools" / "sonic_gmr_retarget.py"
    config_path = (
        gmr_root / "general_motion_retargeting" / "ik_configs" / "smplx_to_g1.json"
    )
    robot_path = gmr_root / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
    gmr_revision = _git_revision(gmr_root)
    expected = {
        "retarget_format_version": str(SONIC_G1_RETARGET_FORMAT_VERSION),
        "gmr_revision": gmr_revision,
        "gmr_config_sha256": _sha256(config_path),
        "g1_robot_sha256": _sha256(robot_path),
        "coordinate_convention": SONIC_SMPL_COORDINATE_CONVENTION,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    normalized_root.mkdir(parents=True, exist_ok=True)

    def convert(sequence: SourceSequence) -> SonicRetargetDecision:
        stem = f"{sequence.source_id}__{sequence.sequence_id}"
        normalized_path = normalized_root / f"{stem}.npz"
        output_path = output_root / f"{stem}.npz"
        partial_path = output_root / f".{stem}.partial.npz"
        source_sha256 = sequence.source_sha256.removeprefix("sha256:")
        if not overwrite and _output_matches(
            output_path, source_sha256=source_sha256, expected=expected
        ):
            return SonicRetargetDecision(
                sequence.source_id, sequence.sequence_id, "skipped", str(output_path)
            )
        partial_path.unlink(missing_ok=True)
        try:
            command_runner(
                [
                    sys.executable,
                    str(normalize_tool),
                    "--source",
                    str(sequence.local_path),
                    "--output",
                    str(normalized_path),
                ],
                check=True,
                capture_output=True,
                text=True,
                cwd=repository,
            )
            command_runner(
                [
                    str(gmr_python),
                    str(retarget_tool),
                    "--source",
                    str(normalized_path),
                    "--provenance-source",
                    str(sequence.local_path),
                    "--output",
                    str(partial_path),
                    "--body-model-root",
                    str(body_model_root),
                ],
                check=True,
                capture_output=True,
                text=True,
                cwd=gmr_root,
                env=gmr_environment,
            )
            if not _output_matches(
                partial_path, source_sha256=source_sha256, expected=expected
            ):
                raise ValueError("G1/SMPL output provenance does not match inputs")
            os.replace(partial_path, output_path)
            if not keep_normalized:
                normalized_path.unlink(missing_ok=True)
            return SonicRetargetDecision(
                sequence.source_id,
                sequence.sequence_id,
                "completed",
                str(output_path),
            )
        except Exception as error:
            partial_path.unlink(missing_ok=True)
            return SonicRetargetDecision(
                sequence.source_id,
                sequence.sequence_id,
                "failed",
                str(output_path),
                _command_error_message(error),
            )

    decisions = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        pending = iter(sequences)
        futures = {
            executor.submit(convert, item)
            for item in tuple(next(pending, None) for _ in range(jobs))
            if item is not None
        }
        completed = 0
        while futures:
            finished, futures = wait(futures, return_when=FIRST_COMPLETED)
            for future in finished:
                decision = future.result()
                decisions.append(decision)
                completed += 1
                if progress is not None:
                    progress(completed, len(sequences), decision)
                item = next(pending, None)
                if item is not None:
                    futures.add(executor.submit(convert, item))
    ordered = tuple(
        sorted(decisions, key=lambda item: (item.source_id, item.sequence_id))
    )
    report = SonicBatchRetargetReport(
        manifest_path=str(manifest_path),
        output_root=str(output_root),
        normalized_root=str(normalized_root),
        gmr_revision=gmr_revision,
        selected=len(sequences),
        completed=sum(item.status == "completed" for item in ordered),
        skipped=sum(item.status == "skipped" for item in ordered),
        failed=sum(item.status == "failed" for item in ordered),
        decisions=ordered,
    )
    _write_report(output_root / "batch_retarget_report.json", report)
    return report


def _output_matches(
    path: Path, *, source_sha256: str, expected: dict[str, str]
) -> bool:
    if not path.is_file():
        return False
    try:
        arrays = load_sonic_g1_retarget(path)
        actual = {
            name: str(np.asarray(arrays[name]).item())
            for name in ("source_sha256", *expected)
        }
    except (OSError, TypeError, ValueError, KeyError):
        return False
    return actual == {"source_sha256": source_sha256, **expected}


def _validate_body_models(root: Path) -> None:
    expected = tuple(
        root / "smplx" / f"SMPLX_{gender}.npz"
        for gender in ("NEUTRAL", "FEMALE", "MALE")
    )
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "required licensed SMPL-X body models are missing: " + ", ".join(missing)
        )


def _nix_develop_environment(root: Path) -> dict[str, str]:
    clean = os.environ.copy()
    clean.pop("VIRTUAL_ENV", None)
    result = subprocess.run(
        ["nix", "develop", "-c", "env", "-0"],
        cwd=root,
        check=True,
        capture_output=True,
        env=clean,
    )
    environment = {}
    for entry in result.stdout.split(b"\0"):
        if entry:
            key, separator, value = entry.partition(b"=")
            if separator:
                environment[os.fsdecode(key)] = os.fsdecode(value)
    return environment


def _git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _command_error_message(error: Exception) -> str:
    if isinstance(error, subprocess.CalledProcessError):
        return (error.stderr or error.stdout or str(error)).strip()[-4000:]
    return str(error)


def _write_report(path: Path, report: SonicBatchRetargetReport) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
