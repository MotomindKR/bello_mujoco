"""Software-rendered video resources shared by policy and reference rollouts."""

from __future__ import annotations
from contextlib import ExitStack
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal
import imageio_ffmpeg
import mujoco
import numpy as np
from bello_mujoco.evaluation.software import validate_software_renderer


@dataclass(frozen=True)
class CameraConfig:
    mode: Literal["follow", "fixed"] = "follow"
    distance: float = 3.0
    azimuth: float = 135.0
    elevation: float = -18.0
    lookat_offset: tuple[float, float, float] = (0.0, 0.0, 0.35)


class VideoWriter:
    """Raw-RGB FFmpeg writer started with ``posix_spawn``.

    Python's usual subprocess path may call ``fork``. Forking after JAX has
    created worker threads can deadlock, so rollout video uses the OS's atomic
    spawn-and-exec primitive directly.
    """

    def __init__(self, path: Path, *, width: int, height: int, fps: int) -> None:
        if not hasattr(os, "posix_spawn"):
            raise RuntimeError("rollout video requires os.posix_spawn support")
        path.parent.mkdir(parents=True, exist_ok=True)
        executable = imageio_ffmpeg.get_ffmpeg_exe()
        read_fd, write_fd = os.pipe()
        argv = [
            executable,
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
        file_actions = [
            (os.POSIX_SPAWN_DUP2, read_fd, 0),
            (os.POSIX_SPAWN_CLOSE, read_fd),
            (os.POSIX_SPAWN_CLOSE, write_fd),
        ]
        try:
            self._pid = os.posix_spawn(
                executable,
                argv,
                os.environ.copy(),
                file_actions=file_actions,
            )
        except BaseException:
            os.close(read_fd)
            os.close(write_fd)
            raise
        os.close(read_fd)
        try:
            self._stream = os.fdopen(write_fd, "wb")
        except BaseException:
            try:
                os.close(write_fd)
            except OSError:
                pass
            os.waitpid(self._pid, 0)
            raise
        self._shape = (height, width, 3)
        self._closed = False

    def append_data(self, frame: np.ndarray) -> None:
        image = np.asarray(frame, dtype=np.uint8)
        if image.shape != self._shape:
            raise ValueError(
                f"video frame has shape {image.shape}, expected {self._shape}"
            )
        self._stream.write(image.tobytes(order="C"))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        stream_error: BaseException | None = None
        try:
            self._stream.close()
        except BaseException as error:
            stream_error = error
        _, status = os.waitpid(self._pid, 0)
        if stream_error is not None:
            raise stream_error
        if status != 0:
            exit_code = os.waitstatus_to_exitcode(status)
            raise RuntimeError(f"FFmpeg video encoder exited with status {exit_code}")


def open_rollout_video_resources(
    mj_model: mujoco.MjModel,
    output_path: Path,
    *,
    width: int,
    height: int,
    fps: int,
    camera_config: CameraConfig,
):
    """Acquire rollout rendering resources with failure-safe cleanup."""

    validate_software_renderer()
    resources = ExitStack()
    try:
        renderer = mujoco.Renderer(mj_model, height=height, width=width)
        resources.callback(renderer.close)
        camera = make_camera(mj_model, camera_config)
        writer = VideoWriter(output_path, width=width, height=height, fps=fps)
        resources.callback(writer.close)
    except BaseException:
        resources.close()
        raise
    return resources, renderer, camera, writer


def make_camera(mj_model: mujoco.MjModel, config: CameraConfig) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(mj_model, camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = config.distance
    camera.azimuth = config.azimuth
    camera.elevation = config.elevation
    return camera


def update_camera(
    camera: mujoco.MjvCamera,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    config: CameraConfig,
    *,
    root_body_name: str,
) -> None:
    if config.mode != "follow":
        return
    root_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)
    if root_body_id < 0:
        raise ValueError(f"follow camera requires body {root_body_name!r}")
    camera.lookat[:] = mj_data.xpos[root_body_id] + np.asarray(config.lookat_offset)


def write_video_frame(
    renderer, camera, writer, mj_model, mj_data, camera_config, *, root_body_name
):
    update_camera(
        camera, mj_model, mj_data, camera_config, root_body_name=root_body_name
    )
    renderer.update_scene(mj_data, camera=camera)
    writer.append_data(renderer.render())
