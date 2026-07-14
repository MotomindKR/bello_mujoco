"""In-process MP4 encoding that is safe after JAX initializes threads."""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np


class Mp4Writer:
    def __init__(self, path: Path, *, width: int, height: int, fps: int) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._container = av.open(str(self.path), mode="w")
        self._stream = self._container.add_stream("libx264", rate=fps)
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "yuv420p"
        self._stream.options = {"crf": "23", "preset": "medium"}
        self._closed = False

    def append_data(self, image: np.ndarray) -> None:
        image = np.asarray(image)
        expected = (self._stream.height, self._stream.width, 3)
        if image.shape != expected:
            raise ValueError(f"video frame has shape {image.shape}; expected {expected}")
        if image.dtype != np.uint8:
            raise ValueError(f"video frame must use uint8 pixels, got {image.dtype}")
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def close(self) -> None:
        if self._closed:
            return
        try:
            for packet in self._stream.encode():
                self._container.mux(packet)
        finally:
            self._container.close()
            self._closed = True

    def __enter__(self) -> "Mp4Writer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
