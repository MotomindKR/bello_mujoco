"""Add synchronized original human references to an immutable Mini corpus."""

import argparse
import json
from pathlib import Path
import tempfile
import os
from functools import lru_cache
from io import BytesIO
import hashlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from bello_mujoco.sonic.amass import resample_rotations
from bello_mujoco.sonic.corpus import (
    MANIFEST,
    _validate_document,
    file_sha256,
    prepare_corpus,
)
from bello_mujoco.sonic.smpl import SmplSkeleton


def pair_corpus(
    corpus: Path,
    source_root: Path,
    sources: Path,
    models: Path,
    output: Path,
    *,
    workers: int = 4,
):
    if workers < 1:
        raise ValueError("workers must be positive")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-", dir=output.parent
    ) as temporary:
        staged = Path(temporary) / "clips"
        result = _pair_corpus(corpus, source_root, sources, models, staged, workers)
        os.rename(staged, output)
    return result


def _pair_corpus(corpus, source_root, sources, models, output, workers):
    document = json.loads((corpus / MANIFEST).read_text())
    _validate_document(document)
    if {p.name for p in corpus.iterdir()} != {e["file"] for e in document["clips"]} | {
        MANIFEST
    }:
        raise ValueError("corpus files differ from its manifest")
    digest = file_sha256(corpus / MANIFEST)
    mapping = json.loads(sources.read_text())
    if set(mapping) != {Path(e["file"]).stem for e in document["clips"]}:
        raise ValueError("source mapping must describe every corpus clip exactly once")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    skeleton = SmplSkeleton.load(models / "SMPL_NEUTRAL.npz")
    source_hashes, splits = {}, {}

    @lru_cache(maxsize=16)
    def read_human(path, expected_hash):
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_hash:
            raise ValueError(f"AMASS source checksum mismatch: {path}")
        with np.load(BytesIO(payload), allow_pickle=False) as raw:
            if str(raw["surface_model_type"]) != "smplx":
                raise ValueError("expected AMASS SMPL-X source motion")
            fps = float(raw["mocap_frame_rate"])
            root = np.asarray(raw["root_orient"])
            pose = np.asarray(raw["pose_body"]).reshape(len(root), 21, 3)
        return fps, root, pose

    def pair_clip(entry):
        name = Path(entry["file"]).stem
        source = mapping[name]
        if set(source) != {"path", "start_seconds", "sha256"}:
            raise ValueError("source records require path, start_seconds and sha256")
        relative = Path(source["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("AMASS source paths must be relative to source_root")
        start = float(source["start_seconds"])
        if not np.isfinite(start) or start < 0:
            raise ValueError("invalid source start time")
        path = source_root / relative
        robot_path = corpus / entry["file"]
        raw_robot = robot_path.read_bytes()
        if (
            robot_path.is_symlink()
            or hashlib.sha256(raw_robot).hexdigest() != entry["sha256"]
        ):
            raise ValueError(f"robot corpus checksum mismatch: {robot_path}")
        with np.load(BytesIO(raw_robot), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        fps, root, pose = read_human(path, source["sha256"])
        frame = (start + np.arange(len(arrays["qpos"])) / float(arrays["fps"])) * fps
        if not np.isfinite(frame).all() or fps <= 0 or frame[-1] > len(root) - 1 + 1e-4:
            raise ValueError(f"human source does not cover the Mini clip: {name}")
        rotations = np.concatenate(
            (root[:, None], pose, np.zeros((len(root), 2, 3))), axis=1
        )
        local = resample_rotations(rotations, frame)
        # Use the same neutral human geometry as the released live PICO path.
        joints, roots = skeleton.forward(local, np.zeros(0))
        arrays.update(
            smpl_joints=joints,
            smpl_root_quat=roots,
            smpl_frame="sonic_root_relative_z_up",
        )
        np.savez_compressed(output / entry["file"], **arrays)
        return name, source, {k: entry[k] for k in ("split", "source_sequence")}

    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        for index, (name, source, split) in enumerate(
            pool.map(pair_clip, document["clips"])
        ):
            source_hashes[name] = source
            splits[name] = split
            if (index + 1) % 100 == 0:
                print(
                    json.dumps({"paired": index + 1, "total": len(document["clips"])}),
                    flush=True,
                )
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return prepare_corpus(
        output,
        splits,
        provenance={
            "robot_corpus_manifest_sha256": digest,
            "source_mapping_sha256": file_sha256(sources),
            "smpl_model_sha256": file_sha256(models / "SMPL_NEUTRAL.npz"),
            "human_sources": source_hashes,
            "human_geometry": "SONIC 24-point FK from neutral SMPL-X body rotations; thumb endpoints, neutral fingers",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "source-root", "sources", "models", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    pair_corpus(
        args.corpus,
        args.source_root,
        args.sources,
        args.models,
        args.output,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
