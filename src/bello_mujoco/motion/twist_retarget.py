"""Resumable manifest-driven batch retargeting for TWIST source motions."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable

import numpy as np

from bello_mujoco.motion.twist_dataset import SourceSequence, load_source_manifest


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
RETARGET_FORMAT_VERSION = 6


@dataclass(frozen=True)
class RetargetDecision:
    source_id: str
    sequence_id: str
    status: str
    output_path: str
    message: str = ""


@dataclass(frozen=True)
class BatchRetargetReport:
    manifest_path: str
    output_root: str
    normalized_root: str
    robot_xml: str
    gmr_revision: str
    selected: int
    completed: int
    skipped: int
    failed: int
    decisions: tuple[RetargetDecision, ...]


def batch_retarget_manifest(
    manifest_path: Path,
    *,
    body_model_root: Path,
    robot_xml: Path,
    output_root: Path,
    normalized_root: Path,
    gmr_python: Path,
    jobs: int = 1,
    overwrite: bool = False,
    keep_normalized: bool = False,
    limit: int | None = None,
    command_runner: CommandRunner = subprocess.run,
    gmr_environment: dict[str, str] | None = None,
    progress: Callable[[int, int, RetargetDecision], None] | None = None,
) -> BatchRetargetReport:
    """Normalize and retarget every selected manifest sequence.

    Each output is first written to a partial file and atomically promoted.
    Existing outputs are reused only when their source, GMR, robot, and
    retargeting-configuration fingerprints match the current inputs.
    """

    manifest_path = Path(manifest_path).expanduser().resolve()
    body_model_root = Path(body_model_root).expanduser().resolve()
    robot_xml = Path(robot_xml).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    normalized_root = Path(normalized_root).expanduser().resolve()
    gmr_python = Path(os.path.abspath(Path(gmr_python).expanduser()))
    if jobs < 1:
        raise ValueError("jobs must be at least one")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when provided")
    if not robot_xml.is_file():
        raise FileNotFoundError(f"canonical Bello MJCF not found: {robot_xml}")
    if not gmr_python.is_file():
        raise FileNotFoundError(
            f"GMR Python interpreter not found: {gmr_python}; initialize it with "
            "`cd deps/GMR && nix develop -c true`"
        )
    _validate_body_models(body_model_root)
    manifest = load_source_manifest(manifest_path, verify_files=False)
    sequences = tuple(
        sorted(
            manifest.sequences,
            key=lambda item: (item.source_id, item.sequence_id),
        )[:limit]
    )
    unsupported = [
        item
        for item in sequences
        if item.pose_representation.lower()
        not in {"smpl-x", "smplx", "smpl-h", "smplh"}
    ]
    if unsupported:
        names = ", ".join(item.sequence_id for item in unsupported[:5])
        raise ValueError(f"batch retargeting requires SMPL-H/SMPL-X sources: {names}")
    missing = [item.local_path for item in sequences if not item.local_path.is_file()]
    if missing:
        raise FileNotFoundError(f"manifest source file not found: {missing[0]}")

    repository = Path(__file__).resolve().parents[3]
    gmr_root = repository / "deps" / "GMR"
    if gmr_environment is None:
        gmr_environment = _nix_develop_environment(gmr_root)
    normalize_tool = repository / "tools" / "twist_normalize_smpl.py"
    retarget_tool = repository / "tools" / "twist_gmr_retarget.py"
    config_path = (
        gmr_root / "general_motion_retargeting" / "ik_configs" / "smplx_to_bello.json"
    )
    robot_source = (
        repository
        / "deps"
        / "bello-stack"
        / "descriptions"
        / "bello"
        / "full_body"
        / "urdf"
        / "bello_full_body.urdf"
    )
    gmr_revision = _git_revision(gmr_root)
    expected = {
        "retarget_format_version": str(RETARGET_FORMAT_VERSION),
        "gmr_revision": gmr_revision,
        "gmr_config_sha256": _sha256(config_path),
        "robot_fingerprint": "sha256:" + _sha256(robot_source),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    normalized_root.mkdir(parents=True, exist_ok=True)

    def convert(sequence: SourceSequence) -> RetargetDecision:
        stem = f"{sequence.source_id}__{sequence.sequence_id}"
        normalized_path = normalized_root / f"{stem}.npz"
        output_path = output_root / f"{stem}.npz"
        partial_path = output_root / f".{stem}.partial.npz"
        source_sha256 = sequence.source_sha256.removeprefix("sha256:")
        if not overwrite and _output_matches(
            output_path,
            source_sha256=source_sha256,
            expected=expected,
        ):
            return RetargetDecision(
                sequence.source_id,
                sequence.sequence_id,
                "skipped",
                str(output_path),
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
                    "--robot-xml",
                    str(robot_xml),
                ],
                check=True,
                capture_output=True,
                text=True,
                cwd=gmr_root,
                env=gmr_environment,
            )
            if not _output_matches(
                partial_path,
                source_sha256=source_sha256,
                expected=expected,
            ):
                raise ValueError("retargeted output provenance does not match inputs")
            os.replace(partial_path, output_path)
            if not keep_normalized:
                normalized_path.unlink(missing_ok=True)
            return RetargetDecision(
                sequence.source_id,
                sequence.sequence_id,
                "completed",
                str(output_path),
            )
        except Exception as error:
            partial_path.unlink(missing_ok=True)
            message = _command_error_message(error)
            return RetargetDecision(
                sequence.source_id,
                sequence.sequence_id,
                "failed",
                str(output_path),
                message,
            )

    decisions: list[RetargetDecision] = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(convert, item): item for item in sequences}
        for completed, future in enumerate(as_completed(futures), start=1):
            decision = future.result()
            decisions.append(decision)
            if progress is not None:
                progress(completed, len(sequences), decision)
    ordered = tuple(
        sorted(decisions, key=lambda item: (item.source_id, item.sequence_id))
    )
    report = BatchRetargetReport(
        manifest_path=str(manifest_path),
        output_root=str(output_root),
        normalized_root=str(normalized_root),
        robot_xml=str(robot_xml),
        gmr_revision=gmr_revision,
        selected=len(sequences),
        completed=sum(item.status == "completed" for item in ordered),
        skipped=sum(item.status == "skipped" for item in ordered),
        failed=sum(item.status == "failed" for item in ordered),
        decisions=ordered,
    )
    _write_report(output_root / "batch_retarget_report.json", report)
    return report


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


def _output_matches(
    path: Path,
    *,
    source_sha256: str,
    expected: dict[str, str],
) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            actual = {
                name: str(np.asarray(archive[name]).item())
                for name in ("source_sha256", *expected)
            }
    except (OSError, ValueError, KeyError):
        return False
    return actual == {"source_sha256": source_sha256, **expected}


def _git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _nix_develop_environment(root: Path) -> dict[str, str]:
    clean_environment = os.environ.copy()
    clean_environment.pop("VIRTUAL_ENV", None)
    result = subprocess.run(
        ["nix", "develop", "-c", "env", "-0"],
        cwd=root,
        check=True,
        capture_output=True,
        env=clean_environment,
    )
    environment = {}
    for entry in result.stdout.split(b"\0"):
        if not entry:
            continue
        key, separator, value = entry.partition(b"=")
        if separator:
            environment[os.fsdecode(key)] = os.fsdecode(value)
    return environment


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _command_error_message(error: Exception) -> str:
    if isinstance(error, subprocess.CalledProcessError):
        detail = (error.stderr or error.stdout or str(error)).strip()
        return detail[-4000:]
    return str(error)


def _write_report(path: Path, report: BatchRetargetReport) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
