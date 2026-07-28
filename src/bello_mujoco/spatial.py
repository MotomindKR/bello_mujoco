"""Small NumPy spatial transforms shared by host-side runtimes."""

from __future__ import annotations

import numpy as np


def quat_rotate_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a vector by a unit quaternion in wxyz order."""

    vector = np.asarray(vector)
    w, x, y, z = np.asarray(quaternion, dtype=vector.dtype)
    quaternion_vector = np.asarray((x, y, z), dtype=vector.dtype)
    uv = np.cross(quaternion_vector, vector)
    uuv = np.cross(quaternion_vector, uv)
    return np.asarray(vector + 2.0 * (w * uv + uuv), dtype=vector.dtype)


def quat_rotate_inverse_wxyz(
    quaternion: np.ndarray,
    vector: np.ndarray,
) -> np.ndarray:
    """Inverse-rotate a vector by a unit quaternion in wxyz order."""

    vector = np.asarray(vector)
    w, x, y, z = np.asarray(quaternion, dtype=vector.dtype)
    quaternion_vector = np.asarray((x, y, z), dtype=vector.dtype)
    uv = np.cross(quaternion_vector, vector)
    uuv = np.cross(quaternion_vector, uv)
    return np.asarray(vector + 2.0 * (-w * uv + uuv), dtype=vector.dtype)
