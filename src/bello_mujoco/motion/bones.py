"""Whole-recording BONES-SEED selection and calibrated SOMA-to-Mini preparation."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import multiprocessing
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile

import hydra
import mink
import numpy as np

from bello_mujoco.configuration import command_config, config_path
from bello_mujoco.motion.bvh import read_bvh, soma_targets
from bello_mujoco.motion.calibration import KinematicRetargetConfig, planted_feet
from bello_mujoco.motion.clip import save_motion_clip
from bello_mujoco.motion.lafan import calibrated_clip
from bello_mujoco.sonic.corpus import (
    MANIFEST,
    file_sha256,
    prepare_corpus,
    verify_corpus,
)


BUCKETS = ("ground", "jump", "dance", "sport", "locomotion", "other")
SUPPORT = re.compile(
    r"stair|ladder|\bbox\b|\bchair\b|scaffold|\bwall\b|crutch|fence|"
    r"\bbench\b|\bsofa\b|\bcouch\b|\bstool\b|\btable\b|shel(?:f|ves)|"
    r"platform|railing|jump off 50cm|horse riding|\bbicycle\b|\bmotorcycle\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class BonesPreparationConfig:
    source_directory: Path
    output_directory: Path
    lafan_corpus: Path
    metadata_file: str = "metadata/seed_metadata_v004.csv"
    bind_file: str = "soma_shapes/soma_base_rig/soma_base_skel_minimal.bvh"
    training_hours: float = 7.0
    evaluation_hours: float = 0.5
    seed: int = 0
    fps: int = 50
    human_height_m: float = 1.8
    workers: int = 8
    penetration_tolerance_m: float = 0.001
    contact_height_m: float = 0.03
    max_retarget_failure_fraction: float = 0.0
    retargeting: KinematicRetargetConfig = field(
        default_factory=lambda: KinematicRetargetConfig(
            initialization_mode="constrained_neutral"
        )
    )

    def __post_init__(self):
        for name in ("fps", "workers"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in (
            "training_hours",
            "evaluation_hours",
            "human_height_m",
            "penetration_tolerance_m",
            "contact_height_m",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        safe_member(self.metadata_file)
        safe_member(self.bind_file)
        if (
            not math.isfinite(self.max_retarget_failure_fraction)
            or not 0 <= self.max_retarget_failure_fraction < 0.1
        ):
            raise ValueError("retarget failure fraction must lie in [0, 0.1)")


def safe_member(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("archive and metadata paths must be relative and confined")
    return path


def family(row):
    name = row["content_name"].removesuffix("_M")
    if not name:
        raise ValueError("BONES content_name is required for disjoint splits")
    return name


def bucket(row):
    text = (
        " ".join(
            row.get(k, "")
            for k in (
                "content_body_position",
                "content_type_of_movement",
                "content_name",
            )
        )
        .lower()
        .replace("_", " ")
    )
    if re.search(
        r"\b(?:floor|crawl\w*|kneel\w*|lying|all fours|roll(?:ing)?|stand(?:ing)? up|get(?:ting)? up|sit(?:ting)?)\b",
        text,
    ):
        return "ground"
    if re.search(r"jump|flip|somersault|stunt|handstand", text):
        return "jump"
    if row["package"] == "Dances" or "danc" in text:
        return "dance"
    if row["package"] == "Sport" or re.search(
        r"\b(?:kick\w*|punch\w*|combat|fight\w*|boxing|martial)\b", text
    ):
        return "sport"
    if row["package"] == "Locomotion" or re.search(
        r"\b(?:walk\w*|jog\w*|run\w*|turn\w*)\b", text
    ):
        return "locomotion"
    return "other"


def _rank(seed, name):
    return hashlib.sha256(f"{seed}/{name}".encode()).digest()


def select_bones(rows, training_hours, evaluation_hours, seed):
    """Balance duration across skills and split entire named motion families."""
    candidates = {
        (split, kind): [] for split in ("train", "evaluation") for kind in BUCKETS
    }
    excluded = []
    names = set()
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
        text = " ".join(
            row.get(k, "")
            for k in (
                "content_name",
                "content_props",
                "content_body_position",
            )
        ).replace("_", " ")
        if match := SUPPORT.search(text):
            excluded.append(
                {"sequence": name, "reason": "unsupported_support", "term": match[0]}
            )
            continue
        group = family(row)
        split = (
            "evaluation"
            if int.from_bytes(_rank(seed, group)[:4], "big") % 10 == 0
            else "train"
        )
        candidates[split, bucket(row)].append(
            {
                "sequence": name,
                "source_path": path,
                "source_sequence": f"bones-seed/{group}",
                "split": split,
                "bucket": bucket(row),
                "source_frames": frames,
                "estimated_seconds": (frames - 1) / 120.0,
                "is_mirror": row["is_mirror"] == "True",
                "package": row["package"],
                "category": row["category"],
            }
        )
    selected = []
    for (split, kind), available in candidates.items():
        target = (
            (training_hours if split == "train" else evaluation_hours)
            * 3600
            / len(BUCKETS)
        )
        # Prefer breadth of named families before adding more takes of one action.
        groups = {}
        for entry in sorted(available, key=lambda e: _rank(seed, e["sequence"])):
            groups.setdefault(entry["source_sequence"], []).append(entry)
        ordered_groups = sorted(groups, key=lambda g: _rank(seed, g))
        seconds = 0.0
        for index in range(max((len(v) for v in groups.values()), default=0)):
            for group in ordered_groups:
                if index >= len(groups[group]):
                    continue
                entry = groups[group][index]
                selected.append(entry)
                seconds += entry["estimated_seconds"]
                if seconds >= target:
                    break
            if seconds >= target:
                break
        if seconds < target:
            raise ValueError(
                f"insufficient {split}/{kind} source duration: {seconds} < {target}"
            )
    return {
        "clips": selected,
        "excluded": sorted(excluded, key=lambda e: e["sequence"]),
        "seed": seed,
        "estimated_hours": {
            s: sum(e["estimated_seconds"] for e in selected if e["split"] == s) / 3600
            for s in ("train", "evaluation")
        },
    }


def extract_sources(archive_path, destination, entries):
    """Stream the read-only archive and copy only explicitly selected regular files."""
    requested = {e["source_path"]: e["sequence"] for e in entries}
    if len(requested) != len(entries):
        raise ValueError("duplicate selected source paths")
    if any(not name or Path(name).name != name for name in requested.values()):
        raise ValueError("selected source names must be flat stems")
    for path in requested:
        safe_member(path)
    destination.mkdir(parents=True, exist_ok=True)
    missing = {
        path
        for path, name in requested.items()
        if not (destination / f"{name}.bvh").is_file()
    }
    if not missing:
        return
    with (
        archive_path.open("rb", buffering=8 * 1024**2) as raw,
        tarfile.open(fileobj=raw, mode="r|gz", bufsize=1024**2) as archive,
    ):
        for member in archive:
            name = member.name.removeprefix("./")
            if name not in missing:
                continue
            safe_member(name)
            if not member.isfile():
                raise ValueError(
                    f"selected archive member is not a regular file: {name}"
                )
            output = destination / f"{requested[name]}.bvh"
            temporary = output.with_suffix(".partial")
            with archive.extractfile(member) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024**2)
            temporary.replace(output)
            missing.remove(name)
            if len(missing) % 100 == 0:
                print(json.dumps({"source_files_remaining": len(missing)}), flush=True)
            if not missing:
                break
    if missing:
        raise ValueError(
            f"archive omits {len(missing)} selected sources: {sorted(missing)[:3]}"
        )


def retarget_bones(config, entry):
    path = config.output_directory / "sources" / f"{entry['sequence']}.bvh"
    original = read_bvh(path)
    if len(original.position) != entry["source_frames"]:
        raise ValueError("source BVH frame count differs from pinned metadata")
    motion = original.resample(config.fps)
    targets = soma_targets(
        motion, read_bvh(config.output_directory / "source-bind.bvh")
    )
    feet = ("left_foot", "right_foot")
    labels = planted_feet(
        np.stack([targets[n][0] for n in feet], axis=1),
        np.stack([targets[n][1] for n in feet], axis=1),
        config.fps,
        config.retargeting.source_contacts,
    )
    clip, report = calibrated_clip(
        targets,
        labels,
        config.fps,
        config.human_height_m,
        config.retargeting,
        config.penetration_tolerance_m,
        config.contact_height_m,
    )
    destination = config.output_directory / "corpus" / f"bones__{entry['sequence']}.npz"
    temporary = destination.with_suffix(".partial")
    with temporary.open("wb") as stream:
        save_motion_clip(stream, clip)
    temporary.replace(destination)
    return {
        **entry,
        "source_sha256": file_sha256(path),
        "source_fps": 1 / original.timestep,
        **report,
    }


def _retarget_job(config, entry):
    try:
        return retarget_bones(config, entry)
    except Exception as error:
        physical = isinstance(error, mink.NoSolutionFound) or (
            isinstance(error, ValueError)
            and str(error).startswith(
                (
                    "contact projection did not converge",
                    "reference violates ",
                )
            )
        )
        source = config.output_directory / "sources" / f"{entry['sequence']}.bvh"
        return {
            **entry,
            "error": f"{type(error).__name__}: {error}",
            "failure_kind": "retargeting" if physical else "unexpected",
            "source_sha256": file_sha256(source) if source.is_file() else None,
        }


def validate_quarantine(entries, failures, maximum):
    """Bound explicit physical retarget failures by count and duration per split."""
    if any(e.get("failure_kind") != "retargeting" for e in failures):
        raise ValueError("unexpected preparation failure; no corpus published")
    summary = {}
    for split in ("train", "evaluation"):
        selected = [e for e in entries if e["split"] == split]
        if not selected:
            raise ValueError("quarantine validation requires both source splits")
        rejected = [e for e in failures if e["split"] == split]
        count = len(rejected) / len(selected)
        duration = sum(e["estimated_seconds"] for e in rejected) / sum(
            e["estimated_seconds"] for e in selected
        )
        if max(count, duration) > maximum:
            raise ValueError(
                f"{split} quarantine exceeds configured fraction {maximum}: clips={count}, duration={duration}"
            )
        summary[split] = {
            "selected": len(selected),
            "quarantined": len(rejected),
            "clip_fraction": count,
            "duration_fraction": duration,
        }
    return summary


def prepare_bones(config):
    output = config.output_directory
    output.mkdir(parents=True, exist_ok=True)
    settings = json.loads(json.dumps(asdict(config), default=str))
    settings_path = output / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("existing BONES preparation settings differ")
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    metadata = config.source_directory / config.metadata_file
    bind_source = config.source_directory / config.bind_file
    bind_hash = file_sha256(bind_source)
    bind_destination = output / "source-bind.bvh"
    if bind_destination.exists() and file_sha256(bind_destination) != bind_hash:
        raise ValueError("existing SOMA bind rig differs from source")
    if not bind_destination.exists():
        shutil.copyfile(bind_source, bind_destination)
    selection_path = output / "selection.json"
    with metadata.open(newline="") as stream:
        selection = select_bones(
            list(csv.DictReader(stream)),
            config.training_hours,
            config.evaluation_hours,
            config.seed,
        )
    selection["metadata_sha256"] = file_sha256(metadata)
    selection["bind_sha256"] = bind_hash
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError("existing BONES selection differs from metadata")
    selection_path.write_text(json.dumps(selection, indent=2) + "\n")
    entries = selection["clips"]
    print(
        json.dumps(
            {
                "selected_clips": len(entries),
                "estimated_hours": selection["estimated_hours"],
            }
        ),
        flush=True,
    )
    extract_sources(
        config.source_directory / "soma_uniform.tar.gz", output / "sources", entries
    )
    (output / "corpus").mkdir(exist_ok=True)
    journal_path = output / "preparation.jsonl"
    previous = (
        [json.loads(line) for line in journal_path.read_text().splitlines()]
        if journal_path.exists()
        else []
    )
    done = {e["sequence"]: e for e in previous if "error" not in e}
    if any(not (output / "corpus" / f"bones__{name}.npz").is_file() for name in done):
        raise ValueError("successful journal entry has no prepared clip")
    pending = [e for e in entries if e["sequence"] not in done]
    failures = []
    with (
        ProcessPoolExecutor(
            max_workers=config.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool,
        journal_path.open("a") as journal,
    ):
        jobs = [pool.submit(_retarget_job, config, entry) for entry in pending]
        for future in as_completed(jobs):
            report = future.result()
            journal.write(json.dumps(report, allow_nan=False) + "\n")
            journal.flush()
            if "error" in report:
                failures.append(report)
            else:
                done[report["sequence"]] = report
            print(
                json.dumps(
                    {
                        "sequence": report["sequence"],
                        "prepared": len(done),
                        "total": len(entries),
                        "error": report.get("error"),
                    }
                ),
                flush=True,
            )
    quarantine = validate_quarantine(
        entries, failures, config.max_retarget_failure_fraction
    )
    (output / "quarantine.json").write_text(
        json.dumps({"summary": quarantine, "clips": failures}, indent=2) + "\n"
    )
    entries = [e for e in entries if e["sequence"] in done]
    lafan = json.loads((config.lafan_corpus / MANIFEST).read_text())
    lafan_eval = tuple(
        Path(e["file"]).stem for e in lafan["clips"] if e["split"] == "evaluation"
    )
    lafan_hash = verify_corpus(config.lafan_corpus, lafan_eval)
    splits = {
        f"bones__{e['sequence']}": {
            "split": e["split"],
            "source_sequence": e["source_sequence"],
        }
        for e in entries
    }
    for entry in lafan["clips"]:
        name = Path(entry["file"]).stem
        if name in splits:
            raise ValueError("LAFAN and BONES corpus names overlap")
        destination = output / "corpus" / entry["file"]
        if not destination.exists():
            shutil.copyfile(config.lafan_corpus / entry["file"], destination)
        if file_sha256(destination) != entry["sha256"]:
            raise ValueError("retained LAFAN reference changed")
        splits[name] = {k: entry[k] for k in ("split", "source_sequence")}
    evaluation = tuple(
        name for name, entry in splits.items() if entry["split"] == "evaluation"
    )
    if not (output / "corpus" / MANIFEST).exists():
        prepare_corpus(
            output / "corpus",
            splits,
            provenance={
                "dataset": "BONES-SEED SOMA Uniform plus fixed LAFAN1",
                "settings": settings,
                "metadata_sha256": selection["metadata_sha256"],
                "bind_sha256": bind_hash,
                "lafan_manifest_sha256": lafan_hash,
                "sequences": [done[e["sequence"]] for e in entries],
                "quarantine": {"summary": quarantine, "clips": failures},
            },
        )
    digest = verify_corpus(output / "corpus", evaluation)
    print(
        json.dumps({"corpus_manifest_sha256": digest, "evaluation_clips": evaluation}),
        flush=True,
    )
    return digest


@hydra.main(
    version_base=None, config_path=config_path("commands"), config_name="prepare_bones"
)
def main(config):
    print(prepare_bones(command_config(BonesPreparationConfig, config)))


if __name__ == "__main__":
    main()
