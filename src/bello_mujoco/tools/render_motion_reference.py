"""Render a canonical fixed-waist, fixed-head Bello reference clip."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from bello_mujoco.motion import (
    load_motion_clip,
    motion_tracking_asset_config,
    sample_motion,
)
from bello_mujoco.render import CameraConfig, VideoWriter, make_camera, update_camera
from bello_mujoco.sim import load_bello_model
from bello_mujoco.terrain import FLAT_TERRAIN


def render_reference(
    motion_path: Path,
    output_path: Path,
    *,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
) -> Path:
    model = load_bello_model(
        motion_tracking_asset_config(),
        terrain=FLAT_TERRAIN,
        use_viewer_model=True,
        load_mjx=False,
    )
    motion = load_motion_clip(motion_path, model)
    mj_model = model.mj_model
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, width=width, height=height)
    camera_config = CameraConfig(
        mode="follow", distance=3.0, azimuth=135.0, elevation=-16.0
    )
    camera = make_camera(mj_model, camera_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = VideoWriter(output_path, width=width, height=height, fps=fps)
    frame_count = int(np.floor(motion.duration * fps)) + 1
    try:
        for frame in range(frame_count):
            reference = sample_motion(motion, min(frame / fps, motion.duration))
            mj_data.qpos[:] = np.asarray(reference.qpos)
            mj_data.qvel[:] = np.asarray(reference.qvel)
            mujoco.mj_forward(mj_model, mj_data)
            update_camera(camera, mj_model, mj_data, camera_config)
            renderer.update_scene(mj_data, camera=camera)
            writer.append_data(renderer.render())
    finally:
        writer.close()
        renderer.close()
    return output_path


def entrypoint() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    print(
        render_reference(
            args.motion,
            args.output,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
    )


if __name__ == "__main__":
    entrypoint()
