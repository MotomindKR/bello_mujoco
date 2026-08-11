#!/usr/bin/env python3
"""Prepare the exact fixed-joint Bello model supplied to pinned GMR."""

from pathlib import Path

from bello_mujoco.assets_pipeline.prepare import prepare_asset
from bello_mujoco.motion.clip import motion_tracking_asset_config


def main() -> None:
    prepared = prepare_asset(
        motion_tracking_asset_config(output_root=Path("assets/generated/twist_gmr"))
    )
    print(prepared.mjcf_path.resolve())


if __name__ == "__main__":
    main()
