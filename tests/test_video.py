from pathlib import Path

import av
import numpy as np
import pytest

from bello_mujoco.video import Mp4Writer


def test_mp4_writer_encodes_frames_without_subprocess(tmp_path: Path) -> None:
    path = tmp_path / "video.mp4"
    with Mp4Writer(path, width=32, height=24, fps=20) as writer:
        writer.append_data(np.zeros((24, 32, 3), dtype=np.uint8))
        writer.append_data(np.full((24, 32, 3), 127, dtype=np.uint8))

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = list(container.decode(stream))
    assert (stream.width, stream.height) == (32, 24)
    assert len(frames) == 2


def test_mp4_writer_rejects_wrong_frame_shape(tmp_path: Path) -> None:
    with Mp4Writer(tmp_path / "video.mp4", width=32, height=24, fps=20) as writer:
        with pytest.raises(ValueError, match="video frame has shape"):
            writer.append_data(np.zeros((12, 32, 3), dtype=np.uint8))
