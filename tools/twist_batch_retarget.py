#!/usr/bin/env python3
"""Retarget a TWIST source manifest to Bello with resumable workers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from bello_mujoco.assets_pipeline.prepare import prepare_asset
from bello_mujoco.motion.clip import motion_tracking_asset_config
from bello_mujoco.motion.twist_retarget import batch_retarget_manifest


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--body-model-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--normalized-root", type=Path, required=True)
    parser.add_argument("--robot-xml", type=Path, default=None)
    parser.add_argument(
        "--gmr-python",
        type=Path,
        default=ROOT / "deps/GMR/.venv/bin/python",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-normalized", action="store_true")
    args = parser.parse_args()
    robot_xml = args.robot_xml
    if robot_xml is None:
        robot_xml = prepare_asset(
            motion_tracking_asset_config(output_root=Path("assets/generated/twist_gmr"))
        ).mjcf_path

    def progress(current, total, decision) -> None:
        print(
            f"[{current}/{total}] {decision.status}: "
            f"{decision.source_id}/{decision.sequence_id}",
            file=sys.stderr,
            flush=True,
        )

    report = batch_retarget_manifest(
        args.manifest,
        body_model_root=args.body_model_root,
        robot_xml=robot_xml,
        output_root=args.output_root,
        normalized_root=args.normalized_root,
        gmr_python=args.gmr_python,
        jobs=args.jobs,
        overwrite=args.overwrite,
        keep_normalized=args.keep_normalized,
        limit=args.limit,
        progress=progress,
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in report.__dict__.items()
                if key != "decisions"
            },
            indent=2,
            sort_keys=True,
        )
    )
    if report.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
