"""Resumable, full BONES-SEED SOMA-X / native SMPL-X / stock Mini GMR cache."""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
from time import perf_counter

import hydra
import mink

from bello_mujoco.configuration import command_config, config_path
from bello_mujoco.hardware.smplx_contract import SMPLX_BODY_JOINT_NAMES
from bello_mujoco.motion.bvh import parse_bvh, read_bvh
from bello_mujoco.motion.clip import motion_tracking_asset_config, save_motion_clip
from bello_mujoco.motion.retarget import RetargetValidationError, retarget_gmr
from bello_mujoco.motion.soma import (
    SomaBonesConverter,
    SomaConversionConfig,
    soma_asset_paths,
)
from bello_mujoco.repository import capture_repository_commit
from bello_mujoco.sim import load_model
from bello_mujoco.sonic.corpus import file_sha256
from bello_mujoco.terrain import FLAT_TERRAIN


@dataclass(frozen=True)
class BonesPreparationConfig:
    source_directory: Path
    output_directory: Path
    metadata_file: str
    bind_file: str
    excluded_name_terms: tuple[str, ...]
    fps: int
    workers: int
    contact_height_m: float
    smplx_model_directory: Path
    soma: SomaConversionConfig

    def __post_init__(self):
        for name in ("fps", "workers"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.contact_height_m) or self.contact_height_m <= 0:
            raise ValueError("contact_height_m must be positive and finite")
        if not self.excluded_name_terms or any(
            not term.strip() for term in self.excluded_name_terms
        ):
            raise ValueError("name exclusions must be explicit nonempty terms")
        safe_member(self.metadata_file)
        safe_member(self.bind_file)


def safe_member(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("archive and metadata paths must be relative and confined")
    return path


def family(row):
    name = row["content_name"].removesuffix("_M")
    if not name:
        raise ValueError("BONES content_name is required for source identity")
    return name


def read_selection(path, excluded_name_terms):
    payload = path.read_bytes()
    previous_limit = csv.field_size_limit()
    try:
        csv.field_size_limit(max(previous_limit, len(payload)))
        selection = select_bones(
            csv.DictReader(io.StringIO(payload.decode("utf-8"))), excluded_name_terms
        )
    finally:
        csv.field_size_limit(previous_limit)
    return selection, hashlib.sha256(payload).hexdigest()


def select_bones(rows, excluded_name_terms):
    """Keep whole recordings without duration caps or skill-dependent quotas."""
    clips, excluded, names = [], [], set()
    terms = tuple(
        (term, "".join(c for c in term.casefold() if c.isalnum()))
        for term in excluded_name_terms
    )
    if any(not normalized for _, normalized in terms):
        raise ValueError("name exclusions must contain letters or digits")
    for row in rows:
        name = row["filename"]
        if not name or Path(name).name != name or name in names:
            raise ValueError("BONES filenames must be unique flat stems")
        names.add(name)
        path = str(safe_member(row["move_soma_uniform_path"]))
        if PurePosixPath(path).stem != name or not path.startswith("soma_uniform/bvh/"):
            raise ValueError("BONES SOMA path does not match filename")
        frames = int(row["move_duration_frames"])
        if frames < 3:
            raise ValueError("BONES recording requires at least three source frames")
        normalized = "".join(c for c in name.casefold() if c.isalnum())
        term = next((term for term, token in terms if token in normalized), None)
        if term is not None:
            excluded.append(
                {"sequence": name, "reason": "unsupported_support", "term": term}
            )
            continue
        clips.append(
            {
                "sequence": name,
                "source_path": path,
                "source_sequence": f"bones-seed/{family(row)}",
                "source_frames": frames,
                "estimated_seconds": (frames - 1) / 120.0,
                "is_mirror": row["is_mirror"] == "True",
            }
        )
    return {
        "clips": clips,
        "excluded": excluded,
        "estimated_hours": sum(e["estimated_seconds"] for e in clips) / 3600,
    }


def archive_sources(path, entries):
    """Stream selected regular BVHs directly into a bounded processing queue."""
    requested = {str(safe_member(e["source_path"])): e for e in entries}
    if len(requested) != len(entries):
        raise ValueError("duplicate selected source paths")
    if not requested:
        return
    with (
        path.open("rb", buffering=8 * 1024**2) as raw,
        tarfile.open(fileobj=raw, mode="r|gz", bufsize=1024**2) as archive,
    ):
        for member in archive:
            name = member.name.removeprefix("./")
            if name not in requested:
                continue
            if not member.isfile():
                raise ValueError(
                    f"selected archive member is not a regular file: {name}"
                )
            with archive.extractfile(member) as source:
                yield requested.pop(name), source.read()
            if not requested:
                break
    if requested:
        raise ValueError(
            f"archive omits {len(requested)} selected sources: {sorted(requested)[:3]}"
        )


def clip_directory(output, sequence):
    shard = hashlib.sha256(sequence.encode()).hexdigest()[:2]
    return output / "clips" / shard / sequence


_worker = None


def _initialize_worker(config, bind):
    global _worker
    converter = SomaBonesConverter(config.soma, config.smplx_model_directory, bind)
    model = load_model(
        motion_tracking_asset_config(asset="bello_mini"),
        terrain=FLAT_TERRAIN,
        load_mjx=False,
    )
    _worker = config, converter, model


def _prepare_job(entry, payload):
    config, converter, model = _worker
    started = perf_counter()
    motion = parse_bvh(payload.decode("utf-8"))
    if len(motion.position) != entry["source_frames"]:
        raise ValueError("source BVH frame count differs from pinned metadata")
    motion = motion.resample(config.fps)
    destination = clip_directory(config.output_directory, entry["sequence"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{entry['sequence']}-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary) / "complete"
        staging.mkdir()
        native_path = staging / "human.npz"
        provenance = converter.convert(motion, native_path)
        converted = perf_counter()
        positions, orientations, height = converter.reader.read(native_path, config.fps)
        targets = {
            name: (positions[:, i], orientations[:, i])
            for i, name in enumerate(SMPLX_BODY_JOINT_NAMES)
        }
        report = {
            **entry,
            **provenance,
            "source_sha256": hashlib.sha256(payload).hexdigest(),
            "human_height_m": height,
            "output_frames": len(positions),
        }
        try:
            clip, diagnostics = retarget_gmr(
                model,
                targets,
                fps=config.fps,
                human_height_m=height,
                contact_height_m=config.contact_height_m,
            )
        except (mink.NoSolutionFound, RetargetValidationError) as error:
            report.update(status="rejected", error=f"{type(error).__name__}: {error}")
        else:
            with (staging / "mini.npz").open("xb") as stream:
                save_motion_clip(stream, clip)
            report.update(
                diagnostics,
                status="prepared",
                mini_sha256=file_sha256(staging / "mini.npz"),
            )
        report.update(
            human_sha256=file_sha256(native_path),
            conversion_seconds=converted - started,
            retarget_seconds=perf_counter() - converted,
        )
        (staging / "preparation.json").write_text(
            json.dumps(report, allow_nan=False) + "\n"
        )
        staging.rename(destination)
    return report


def completed_sequences(output):
    root = output / "clips"
    if not root.exists():
        return set()
    return {
        clip.name
        for shard in os.scandir(root)
        if shard.is_dir()
        for clip in os.scandir(shard.path)
        if clip.is_dir() and not clip.name.startswith(".")
    }


def prepare_bones(config):
    """Cache all name-filtered recordings; keep explicit physical rejection receipts."""
    commit = capture_repository_commit()
    output = config.output_directory
    source = config.source_directory.resolve()
    if output.resolve().is_relative_to(source):
        raise ValueError(
            "output_directory must be outside the read-only source dataset"
        )
    assets = {
        path.name: file_sha256(path)
        for path in soma_asset_paths(config.soma, config.smplx_model_directory)
    }
    output.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {"phase": "reading_metadata", "source": str(config.source_directory)}
        ),
        flush=True,
    )
    metadata = config.source_directory / config.metadata_file
    selection, metadata_hash = read_selection(metadata, config.excluded_name_terms)
    bind_path = config.source_directory / config.bind_file
    bind = read_bvh(bind_path)
    settings = json.loads(
        json.dumps(
            {
                "config": asdict(config),
                "git_commit": commit,
                "metadata_sha256": metadata_hash,
                "bind_sha256": file_sha256(bind_path),
                "human_asset_sha256": assets,
            },
            default=str,
        )
    )
    settings_path = output / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("existing BONES preparation settings or source commit differ")
    for path, document in (
        (settings_path, settings),
        (output / "selection.json", selection),
    ):
        temporary = path.with_suffix(".partial")
        temporary.write_text(json.dumps(document) + "\n")
        temporary.replace(path)
    done = completed_sequences(output)
    selected = {e["sequence"] for e in selection["clips"]}
    if done - selected:
        raise ValueError("cache contains sequences outside the pinned selection")
    pending = [e for e in selection["clips"] if e["sequence"] not in done]
    print(
        json.dumps(
            {
                "selected_clips": len(selected),
                "completed_clips": len(done),
                "excluded_clips": len(selection["excluded"]),
                "estimated_hours": selection["estimated_hours"],
            }
        ),
        flush=True,
    )
    if not pending:
        return
    with (
        ProcessPoolExecutor(
            max_workers=config.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(config, bind),
        ) as pool,
        (output / "preparation.jsonl").open("a") as journal,
    ):
        jobs = set()

        def record_completed():
            nonlocal jobs
            finished, jobs = wait(jobs, return_when=FIRST_COMPLETED)
            for future in finished:
                report = future.result()
                journal.write(json.dumps(report, allow_nan=False) + "\n")
                journal.flush()
                done.add(report["sequence"])
                print(
                    json.dumps(
                        {
                            "sequence": report["sequence"],
                            "status": report["status"],
                            "completed": len(done),
                            "total": len(selected),
                            "conversion_seconds": report["conversion_seconds"],
                            "retarget_seconds": report["retarget_seconds"],
                        }
                    ),
                    flush=True,
                )

        for entry, source in archive_sources(
            config.source_directory / "soma_uniform.tar.gz", pending
        ):
            jobs.add(pool.submit(_prepare_job, entry, source))
            if len(jobs) >= 2 * config.workers:
                record_completed()
        while jobs:
            record_completed()
    print(json.dumps({"phase": "complete", "completed": len(done)}), flush=True)


@hydra.main(
    version_base=None, config_path=config_path("commands"), config_name="prepare_bones"
)
def main(config):
    prepare_bones(command_config(BonesPreparationConfig, config))


if __name__ == "__main__":
    main()
