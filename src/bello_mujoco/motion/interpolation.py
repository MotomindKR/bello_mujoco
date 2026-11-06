"""Offline interpolation shared by BVH and native SMPL-X sources."""

import numpy as np


def resample_quaternions(quaternions, frame):
    """Shortest-arc SLERP; quaternion component order is preserved."""
    lower = np.floor(frame).astype(int)
    upper = np.minimum(lower + 1, len(quaternions) - 1)
    alpha = (frame - lower)[:, None, None]
    a, b = quaternions[lower], quaternions[upper]
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    angle = np.arccos(np.clip(np.abs(dot), 0, 1))
    sine = np.sin(angle)
    result = np.where(
        sine > 1e-7,
        (np.sin((1 - alpha) * angle) * a + np.sin(alpha * angle) * b)
        / np.maximum(sine, 1e-7),
        (1 - alpha) * a + alpha * b,
    )
    return result / np.linalg.norm(result, axis=-1, keepdims=True)
