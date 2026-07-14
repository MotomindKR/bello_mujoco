"""Render a canonical fixed-waist Bello reference clip."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.motion import load_motion_clip, sample_motion
from bello_mujoco.render import CameraConfig, make_camera, update_camera
from bello_mujoco.sim import load_bello_model
from bello_mujoco.terrain import FLAT_TERRAIN


def render_reference(
    motion_path: Path,
    output_path: Path,
    *,
    width: int = 960,
    height: int = 720,
    fps: int = 30,
) -> Path:
    model = load_bello_model(
        AssetPrepConfig(
            asset="full_body",
            collision_mode="primitive",
            end_effector_sphere_mode="both",
            fixed_joint_names=("waist_roll_joint", "waist_pitch_joint"),
        ),
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
    writer = imageio.get_writer(output_path, fps=fps)
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
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
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
