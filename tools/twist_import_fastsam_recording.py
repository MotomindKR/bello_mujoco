#!/usr/bin/env python3
"""Import a complete Fast-SAM recording into TWIST source manifests."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from bello_mujoco.motion.fastsam_recording import (
    FastSamRecordingImportConfig,
    import_fastsam_recording,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path)
    parser.add_argument("--combined-manifest", type=Path)
    parser.add_argument("--source-id", default="inhouse_fastsam")
    parser.add_argument("--subject-id", default="operator")
    parser.add_argument("--owner", default="operator")
    parser.add_argument("--ownership-basis", default="operator-provided recording")
    parser.add_argument("--license-id", default="in-house-owner-authorized")
    parser.add_argument("--sampling-weight", type=float, default=1.0)
    parser.add_argument("--maximum-clip-seconds", type=float, default=10.0)
    parser.add_argument("--minimum-clip-seconds", type=float, default=2.1)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    report = import_fastsam_recording(
        arguments.recording_root,
        arguments.source_root,
        arguments.source_manifest,
        combined_manifest_path=arguments.combined_manifest,
        base_manifest_path=arguments.base_manifest,
        source_id=arguments.source_id,
        subject_id=arguments.subject_id,
        owner=arguments.owner,
        ownership_basis=arguments.ownership_basis,
        license_id=arguments.license_id,
        config=FastSamRecordingImportConfig(
            sampling_weight=arguments.sampling_weight,
            maximum_clip_seconds=arguments.maximum_clip_seconds,
            minimum_clip_seconds=arguments.minimum_clip_seconds,
        ),
        overwrite=arguments.overwrite,
    )
    print(json.dumps(asdict(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
