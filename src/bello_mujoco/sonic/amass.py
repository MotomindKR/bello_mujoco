"""Offline AMASS SMPL-X to canonical Mini clips using the unmodified pinned GMR."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np
import yaml

from bello_mujoco.motion.retarget import retarget_gmr
from bello_mujoco.motion.smplx import SmplxMotionReader
from bello_mujoco.hardware.smplx_contract import (
    SMPLX_BODY_JOINT_NAMES,
)
from bello_mujoco.motion.clip import (
    MotionClip,
    load_motion_clip,
    motion_tracking_asset_config,
    save_motion_clip,
)
from bello_mujoco.sim import load_model
from bello_mujoco.sonic.corpus import (
    MANIFEST,
    file_sha256,
    prepare_corpus,
    verify_corpus,
)
from bello_mujoco.terrain import FLAT_TERRAIN
from bello_mujoco.repository import repository_root


@dataclass(frozen=True)
class AmassPreparationConfig:
    fps: int = 50
    train_hours: float = 8.0
    evaluation_hours: float = 0.8
    seed: int = 20260928
    min_seconds: float = 2.0
    max_seconds: float = 30.0
    max_joint_speed: float = 20.0
    max_penetration_m: float = 0.03
    contact_height_m: float = 0.03

    def __post_init__(self):
        if any(
            not math.isfinite(value) or value <= 0 for value in asdict(self).values()
        ):
            raise ValueError("AMASS preparation settings must be positive and finite")
        if self.min_seconds > self.max_seconds:
            raise ValueError("AMASS minimum duration exceeds maximum duration")


def select_candidates(manifest: Path, config: AmassPreparationConfig) -> list[dict]:
    """Select whole sequences; hold out subjects and deduplicate source fingerprints."""
    document = yaml.safe_load(manifest.read_text())
    excluded = re.compile(
        r"stairs?|ladder|slope|chair|table|bench|treadmill|craw[l]?|lie|lying|sitting|kneel|jump|hopping|kick|cartwheel|backflip|handstand",
        re.I,
    )
    groups = defaultdict(list)
    seen = set()
    for item in document["sequences"]:
        if item["source_kind"] != "amass" or item["pose_representation"] != "smpl-x":
            continue
        # These collections concentrate on object interaction or extreme poses.
        if item["subdataset"] in {"GRAB", "PosePrior"}:
            continue
        relative = Path(item["metadata"]["relative_path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or excluded.search(str(relative))
        ):
            continue
        seconds = (int(item["metadata"]["frame_count"]) - 1) / float(item["frame_rate"])
        if not config.min_seconds <= seconds <= config.max_seconds:
            continue
        identity = item.get("near_duplicate_fingerprint") or item["source_sha256"]
        if identity in seen:
            continue
        seen.add(identity)
        subject = source_subject(item["subdataset"], item["subject_id"])
        held_out = (
            int(hashlib.sha256(f"{config.seed}/{subject}".encode()).hexdigest()[:8], 16)
            % 10
            == 0
        )
        groups[item["subdataset"]].append(
            {
                "path": relative.as_posix(),
                "source_sha256": item["source_sha256"],
                "source_sequence": identity,
                "subject": subject,
                "split": "evaluation" if held_out else "train",
                "seconds": seconds,
            }
        )
    random = np.random.default_rng(config.seed)
    for entries in groups.values():
        random.shuffle(entries)
    # Round-robin subdatasets avoids allowing the largest archive to dominate.
    selected = []
    while any(groups.values()):
        for name in sorted(groups):
            if groups[name]:
                selected.append(groups[name].pop())
    return selected


def source_subject(subdataset: str, subject: str) -> str:
    if subdataset == "ACCAD":
        match = re.match(r"(?:Female|Male)\d+", subject, re.I)
        if match:
            subject = match.group().lower()
    elif subdataset == "DanceDB":
        subject = re.sub(r"^\d{8}_", "", subject).lower()
    elif subdataset == "HUMAN4D":
        subject = subject.rsplit("_", 1)[-1].lower()
    elif subdataset in {"DFaust", "MoSh", "SSM"}:
        # These MPI collections can reuse the same registered subject IDs.
        return f"MPI/{subject.rsplit('_', 1)[-1]}"
    return f"{subdataset}/{subject}"


class AmassRetargeter:
    def __init__(self, model_directory: Path, config: AmassPreparationConfig):
        self.config = config
        self.motion_reader = SmplxMotionReader(model_directory)
        self.model = load_model(
            motion_tracking_asset_config(asset="bello_mini"),
            terrain=FLAT_TERRAIN,
            load_mjx=False,
        )

    def retarget(self, source: Path, output: Path) -> dict:
        if output.exists():
            raise FileExistsError(output)
        positions, orientations, height = self.motion_reader.read(
            source, self.config.fps
        )
        targets = {
            name: (positions[:, index], orientations[:, index])
            for index, name in enumerate(SMPLX_BODY_JOINT_NAMES)
        }
        clip, report = retarget_gmr(
            self.model,
            targets,
            fps=self.config.fps,
            human_height_m=height,
            contact_height_m=self.config.contact_height_m,
        )
        peak_speed = report["peak_joint_speed_rad_s"]
        if peak_speed > self.config.max_joint_speed:
            raise ValueError(
                f"retarget joint speed exceeds threshold: {peak_speed:.3f} rad/s"
            )
        worst_penetration = min(
            report["worst_floor_penetration_m"], report["worst_self_penetration_m"]
        )
        if worst_penetration < -self.config.max_penetration_m:
            raise ValueError(
                f"retarget penetration exceeds threshold: {worst_penetration:.4f} m"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        publish_motion_clip(output, clip, self.model)
        return report


def publish_motion_clip(output: Path, clip: MotionClip, model):
    """Validate before publication; support SMB without requiring hard links."""
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=".pending-"
    ) as temporary:
        pending = Path(temporary) / "clip.npz"
        save_motion_clip(pending, clip)
        load_motion_clip(pending, model)
        # Reserve this exact destination exclusively before replacing it. A crash
        # leaves an unjournaled placeholder, which resume preserves and recomputes.
        with output.open("xb"):
            pass
        os.replace(pending, output)
        remove_generated_macos_sidecar(output)


def remove_generated_macos_sidecar(output: Path):
    """Discard only AppleDouble metadata for a file we just generated on SMB."""
    sidecar = output.with_name("._" + output.name)
    if sidecar.exists():
        with sidecar.open("rb") as stream:
            if stream.read(4) != b"\x00\x05\x16\x07":
                raise RuntimeError(f"unexpected non-AppleDouble sidecar: {sidecar}")
        sidecar.unlink()


_worker_retargeter: AmassRetargeter | None = None


def _initialize_worker(model_directory: Path, config: AmassPreparationConfig):
    global _worker_retargeter
    _worker_retargeter = AmassRetargeter(model_directory, config)


def _process_candidate(arguments: tuple[Path, Path, dict]) -> dict:
    root, output, item = arguments
    path = root / item["path"]
    if file_sha256(path) != item["source_sha256"]:
        raise ValueError(f"AMASS source checksum differs from source manifest: {path}")
    name = "amass__" + "__".join(Path(item["path"]).with_suffix("").parts)
    assert _worker_retargeter is not None
    try:
        quality = _worker_retargeter.retarget(path, output / "clips" / f"{name}.npz")
    except ValueError as error:
        return {**item, "reason": str(error)}
    return {
        **item,
        "clip": name,
        "quality": quality,
        "sha256": file_sha256(output / "clips" / f"{name}.npz"),
    }


def preparation_identity(root: Path, manifest: Path, config: AmassPreparationConfig):
    repository = repository_root()
    return {
        "schema": 1,
        "config": asdict(config),
        "source_manifest_sha256": file_sha256(manifest),
        "body_models": {
            gender: file_sha256(root / f"body_models/smplx/SMPLX_{gender}.npz")
            for gender in ("MALE", "FEMALE", "NEUTRAL")
        },
        "dependencies": {
            name: subprocess.check_output(
                ["git", "-C", str(repository / "deps" / name), "rev-parse", "HEAD"],
                text=True,
            ).strip()
            for name in ("GMR", "bello-stack")
        },
        "preparation_sources": {
            str(path.relative_to(repository)): file_sha256(path)
            for path in sorted((repository / "src/bello_mujoco").rglob("*.py"))
        },
        "lock_sha256": file_sha256(repository / "uv.lock"),
    }


def summarize_preparation(records: list[dict]) -> dict:
    accepted = [item for item in records if "clip" in item]
    return {
        "accepted": accepted,
        "rejected": [item for item in records if "reason" in item],
        "seconds": {
            split: sum(
                item["quality"]["seconds"]
                for item in accepted
                if item["split"] == split
            )
            for split in ("train", "evaluation")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-clips",
        type=int,
        default=0,
        help="0 processes candidates until both duration budgets are met",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.max_clips < 0:
        parser.error("workers must be positive and max-clips nonnegative")
    if args.output.resolve().is_relative_to(args.root.resolve()):
        parser.error("derived output must be outside the read-only AMASS source root")
    config = AmassPreparationConfig()
    candidates = select_candidates(args.manifest, config)
    identity = preparation_identity(args.root, args.manifest, config)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    identity_path = args.output / "identity.json"
    journal_path = args.output / "progress.jsonl"
    if args.resume:
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError(
                "preparation inputs/code/config changed; start a new corpus"
            )
        records = [json.loads(line) for line in journal_path.read_text().splitlines()]
        for item in records:
            if (
                "clip" in item
                and file_sha256(args.output / "clips" / f"{item['clip']}.npz")
                != item["sha256"]
            ):
                raise ValueError(f"prepared clip changed: {item['clip']}")
        recorded_clips = {f"{item['clip']}.npz" for item in records if "clip" in item}
        for path in (args.output / "clips").glob("*.npz"):
            if path.name not in recorded_clips:
                recovery = Path(
                    tempfile.mkdtemp(prefix="interrupted-", dir=args.output)
                )
                path.rename(recovery / path.name)
                print(f"Preserved unjournaled clip in {recovery}", flush=True)
    else:
        identity_path.write_text(json.dumps(identity, indent=2) + "\n")
        records = []
    summary = summarize_preparation(records)
    totals = summary["seconds"]
    budgets = {
        "train": config.train_hours * 3600,
        "evaluation": config.evaluation_hours * 3600,
    }
    if (args.output / "clips" / MANIFEST).exists():
        if not all(totals[key] >= budgets[key] for key in budgets):
            raise ValueError(
                "completed pilot corpus is immutable; use a new output directory"
            )
        verify_corpus(
            args.output / "clips",
            tuple(
                item["clip"]
                for item in summary["accepted"]
                if item["split"] == "evaluation"
            ),
        )
        print(json.dumps({"complete": True, "seconds": totals}), flush=True)
        return
    processed = {item["path"] for item in records}
    remaining = iter(item for item in candidates if item["path"] not in processed)
    # Bound outstanding work and consume results in selection order: worker count
    # must not change which clips make it into the immutable corpus.
    with (
        ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(args.root / "body_models/smplx", config),
        ) as pool,
        journal_path.open("a") as journal,
    ):
        while not all(totals[key] >= budgets[key] for key in budgets):
            count = 4
            if any(
                0 < budgets[key] - totals[key] < 4 * config.max_seconds
                for key in budgets
            ):
                count = 1
            if args.max_clips:
                count = min(count, args.max_clips - len(summary["accepted"]))
            if count <= 0:
                break
            batch = []
            for item in remaining:
                if totals[item["split"]] < budgets[item["split"]]:
                    batch.append((args.root, args.output, item))
                if len(batch) == count:
                    break
            if not batch:
                break
            for record in pool.map(_process_candidate, batch):
                journal.write(json.dumps(record) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                records.append(record)
                summary = summarize_preparation(records)
                totals = summary["seconds"]
                print(
                    json.dumps(
                        {
                            "processed": len(records),
                            "accepted": len(summary["accepted"]),
                            "seconds": totals,
                            "result": record,
                        }
                    ),
                    flush=True,
                )
            temporary = args.output / "preparation.json.tmp"
            temporary.write_text(json.dumps(summary, indent=2) + "\n")
            temporary.replace(args.output / "preparation.json")
    if not args.max_clips and any(totals[key] < budgets[key] for key in budgets):
        raise RuntimeError(f"insufficient accepted AMASS motion: {totals}")
    splits = {
        item["clip"]: {
            "source_sequence": item["source_sequence"],
            "split": item["split"],
        }
        for item in summary["accepted"]
    }
    prepare_corpus(
        args.output / "clips",
        splits,
        provenance={
            "identity": identity,
            "accepted": summary["accepted"],
            "seconds": totals,
            "rejected": len(summary["rejected"]),
            "rejection_categories": dict(
                Counter(item["reason"].split(":")[0] for item in summary["rejected"])
            ),
            "use": "Non-commercial research; AMASS and SMPL-X source terms apply",
            "storage_migration": (
                json.loads((args.output / "storage-migration.json").read_text())
                if (args.output / "storage-migration.json").exists()
                else None
            ),
        },
    )
    remove_generated_macos_sidecar(args.output / "clips" / MANIFEST)


if __name__ == "__main__":
    main()
