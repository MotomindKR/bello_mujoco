#!/usr/bin/env python3
"""Prepare a robot's fixed-joint model for pinned GMR."""

import argparse
from pathlib import Path

from bello_mujoco.assets_pipeline.prepare import prepare_asset
from bello_mujoco.motion.clip import motion_tracking_asset_config
from bello_mujoco.robot import robot_definition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True)
    args = parser.parse_args()
    robot_definition(args.asset)
    prepared = prepare_asset(
        motion_tracking_asset_config(
            asset=args.asset,
            output_root=Path("assets/generated/twist_gmr"),
        )
    )
    print(prepared.mjcf_path.resolve())


if __name__ == "__main__":
    main()
