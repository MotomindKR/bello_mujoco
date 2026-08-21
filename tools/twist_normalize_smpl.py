#!/usr/bin/env python3
"""Normalize supported AMASS SMPL-H/SMPL-X archives for pinned GMR."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.source.expanduser().resolve(), allow_pickle=False) as source:
        if {"pose_body", "root_orient", "trans"} <= set(source.files):
            pose_body = np.asarray(source["pose_body"])
            root_orient = np.asarray(source["root_orient"])
            translation = np.asarray(source["trans"])
        elif {"poses", "trans"} <= set(source.files):
            poses = np.asarray(source["poses"])
            if poses.ndim != 2 or poses.shape[1] < 66:
                raise ValueError("AMASS SMPL-H poses must contain at least 66 values")
            root_orient = poses[:, :3]
            pose_body = poses[:, 3:66]
            translation = np.asarray(source["trans"])
        else:
            raise ValueError(
                "source must contain standardized SMPL-X fields or AMASS poses/trans"
            )
        if pose_body.shape != (len(translation), 63):
            raise ValueError(
                f"pose_body shape must be {(len(translation), 63)}, got {pose_body.shape}"
            )
        betas = np.asarray(source["betas"]).reshape(-1)
        betas = np.pad(betas[:16], (0, max(0, 16 - len(betas))))[:16]
        gender = str(np.asarray(source["gender"]).item())
        fps_key = (
            "mocap_frame_rate"
            if "mocap_frame_rate" in source.files
            else "mocap_framerate"
        )
        if fps_key not in source.files:
            raise ValueError("source is missing mocap frame rate metadata")
        fps = float(np.asarray(source[fps_key]).item())
    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.expanduser().resolve(),
        pose_body=pose_body.astype(np.float32),
        root_orient=root_orient.astype(np.float32),
        trans=translation.astype(np.float32),
        betas=betas.astype(np.float32),
        gender=np.asarray(gender),
        mocap_frame_rate=np.asarray(fps, dtype=np.float32),
    )


if __name__ == "__main__":
    main()
