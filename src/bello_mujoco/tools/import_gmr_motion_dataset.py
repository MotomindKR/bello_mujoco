"""Import a directory of trusted GMR Bello motions into canonical archives."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from bello_mujoco.motion import BEYONDMIMIC_TRACKED_BODY_NAMES
from bello_mujoco.tools.import_gmr_motion import import_gmr_motion


def _import_one(source: str, output: str, overwrite: bool) -> dict:
    source_path = Path(source)
    output_path = Path(output)
    if overwrite or not output_path.exists():
        import_gmr_motion(
            source_path,
            output_path,
            collision_mode="box",
            tracked_body_names=BEYONDMIMIC_TRACKED_BODY_NAMES,
            reject_self_collisions=False,
        )
    with np.load(output_path) as motion:
        frames = int(motion["qpos"].shape[0])
        fps = float(motion["fps"])
        source_sha256 = str(motion["source_sha256"])
        body_names = [str(name) for name in motion["body_names"]]
        collision_frames = int(motion["self_collision_frames"])
        max_penetration = float(motion["max_self_collision_penetration_m"])
        deepest_pair = [str(name) for name in motion["deepest_self_collision_pair"]]
    return {
        "sequence": source_path.stem,
        "source_file": source_path.name,
        "source_sha256": source_sha256,
        "motion_file": output_path.name,
        "frames": frames,
        "fps": fps,
        "duration_seconds": frames / fps,
        "tracked_bodies": body_names,
        "collision_clean": collision_frames == 0,
        "self_collision_frames": collision_frames,
        "max_self_collision_penetration_m": max_penetration,
        "deepest_self_collision_pair": deepest_pair,
    }


def import_dataset(
    source_dir: Path,
    output_dir: Path,
    *,
    jobs: int,
    overwrite: bool,
) -> tuple[list[dict], list[dict]]:
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    sources = sorted(source_dir.resolve().glob("*.pkl"))
    if not sources:
        raise ValueError(f"no GMR .pkl motions found in {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    motions = []
    failures = []
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                _import_one,
                str(source),
                str((output_dir / f"{source.stem}.npz").resolve()),
                overwrite,
            ): source
            for source in sources
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            source = futures[future]
            try:
                motions.append(future.result())
                print(f"[{completed}/{len(futures)}] {source.stem}", flush=True)
            except Exception as error:  # noqa: BLE001 - report every bad sequence
                failures.append({"sequence": source.stem, "error": repr(error)})
                print(
                    f"[{completed}/{len(futures)}] FAILED {source.stem}: {error}",
                    flush=True,
                )
    return (
        sorted(motions, key=lambda item: item["sequence"]),
        sorted(failures, key=lambda item: item["sequence"]),
    )
