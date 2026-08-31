"""Shared physical target-envelope helpers for hardware policy adapters."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


ANKLE_HARD_LIMIT_INSET_RADIANS = 1.0e-9
ANKLE_PROJECTION_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class AnkleTargetEnvelope:
    side: str
    motor_indices: tuple[int, int]
    logical_indices: tuple[int, int]
    motor_to_joint: np.ndarray
    joint_low: np.ndarray
    joint_high: np.ndarray


def _clip_polygon_to_halfspace(
    polygon: np.ndarray,
    normal: np.ndarray,
    upper: float,
) -> np.ndarray:
    if polygon.size == 0:
        return polygon.reshape(0, 2)
    clipped: list[np.ndarray] = []
    previous = polygon[-1]
    previous_value = float(normal @ previous - upper)
    previous_inside = previous_value <= ANKLE_PROJECTION_TOLERANCE
    for current in polygon:
        current_value = float(normal @ current - upper)
        current_inside = current_value <= ANKLE_PROJECTION_TOLERANCE
        if current_inside != previous_inside:
            denominator = previous_value - current_value
            if abs(denominator) <= ANKLE_PROJECTION_TOLERANCE:
                raise RuntimeError("ankle envelope clipping encountered a flat edge")
            fraction = float(np.clip(previous_value / denominator, 0.0, 1.0))
            clipped.append(previous + fraction * (current - previous))
        if current_inside:
            clipped.append(current)
        previous = current
        previous_value = current_value
        previous_inside = current_inside
    if not clipped:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(clipped, dtype=np.float64)


def ankle_feasible_polygon(
    envelope: AnkleTargetEnvelope,
    control_limits: np.ndarray,
) -> np.ndarray:
    motor_indices = np.asarray(envelope.motor_indices, dtype=np.int32)
    limits = np.asarray(control_limits, dtype=np.float64)[motor_indices]
    if limits.shape != (2, 2) or not np.all(np.isfinite(limits)):
        raise ValueError(f"{envelope.side} ankle motor limits are invalid")
    low = limits[:, 0]
    high = limits[:, 1]
    if np.any(low >= high):
        raise ValueError(f"{envelope.side} ankle motor limits do not overlap")
    polygon = np.asarray(
        (
            (low[0], low[1]),
            (high[0], low[1]),
            (high[0], high[1]),
            (low[0], high[1]),
        ),
        dtype=np.float64,
    )
    safe_low = envelope.joint_low + ANKLE_HARD_LIMIT_INSET_RADIANS
    safe_high = envelope.joint_high - ANKLE_HARD_LIMIT_INSET_RADIANS
    if np.any(safe_low >= safe_high):
        raise ValueError(f"{envelope.side} ankle hard joint limits are invalid")
    for row, joint_low, joint_high in zip(
        envelope.motor_to_joint, safe_low, safe_high, strict=True
    ):
        polygon = _clip_polygon_to_halfspace(polygon, row, float(joint_high))
        polygon = _clip_polygon_to_halfspace(polygon, -row, float(-joint_low))
    if polygon.size == 0:
        raise ValueError(
            f"{envelope.side} ankle motor and hard joint envelopes do not overlap"
        )
    return polygon


def _project_point_to_polygon(point: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    if polygon.shape == (1, 2):
        return polygon[0].copy()
    best: np.ndarray | None = None
    best_distance = math.inf
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        edge = end - start
        denominator = float(edge @ edge)
        if denominator <= ANKLE_PROJECTION_TOLERANCE:
            candidate = start
        else:
            fraction = float(np.clip((point - start) @ edge / denominator, 0.0, 1.0))
            candidate = start + fraction * edge
        distance = float((candidate - point) @ (candidate - point))
        if distance < best_distance:
            best = candidate
            best_distance = distance
    if best is None:
        raise RuntimeError("ankle envelope projection has no boundary")
    return best.copy()


def project_ankle_motor_target(
    point: np.ndarray,
    envelope: AnkleTargetEnvelope,
    control_limits: np.ndarray,
) -> np.ndarray:
    """Return the closest feasible pair of differential-ankle motor targets."""

    point = np.asarray(point, dtype=np.float64)
    polygon = ankle_feasible_polygon(envelope, control_limits)
    safe_low = envelope.joint_low + ANKLE_HARD_LIMIT_INSET_RADIANS
    safe_high = envelope.joint_high - ANKLE_HARD_LIMIT_INSET_RADIANS
    resolved = envelope.motor_to_joint @ point
    motor_limits = np.asarray(control_limits, dtype=np.float64)[
        np.asarray(envelope.motor_indices, dtype=np.int32)
    ]
    if (
        np.all(point >= motor_limits[:, 0])
        and np.all(point <= motor_limits[:, 1])
        and np.all(resolved >= safe_low)
        and np.all(resolved <= safe_high)
    ):
        return point.copy()
    projected = _project_point_to_polygon(point, polygon)
    projected_resolved = envelope.motor_to_joint @ projected
    violation = np.maximum(
        safe_low - projected_resolved,
        projected_resolved - safe_high,
    )
    if float(np.max(violation)) > ANKLE_PROJECTION_TOLERANCE:
        raise RuntimeError(
            f"{envelope.side} ankle projection remains outside its hard envelope"
        )
    return projected
