"""Causal Fast-SAM root conditioning shared by import and deployment."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


FASTSAM_ROOT_CONDITIONING_VERSION = "causal-support-foot-xy-v2"


@dataclass(frozen=True)
class FootConditionedRootConfig:
    """Parameters for a causal, confidence-aware support-foot root estimator."""

    contact_enter_speed_mps: float = 0.20
    contact_exit_speed_mps: float = 0.45
    contact_enter_height_m: float = 0.06
    contact_exit_height_m: float = 0.10
    minimum_foot_confidence: float = 0.50
    contact_enter_seconds: float = 0.08
    contact_exit_seconds: float = 0.06
    detection_filter_time_constant_seconds: float = 0.05
    support_blend_time_constant_seconds: float = 0.06
    maximum_interval_seconds: float = 0.20

    def __post_init__(self) -> None:
        positive = {
            "contact_enter_speed_mps": self.contact_enter_speed_mps,
            "contact_exit_speed_mps": self.contact_exit_speed_mps,
            "contact_enter_height_m": self.contact_enter_height_m,
            "contact_exit_height_m": self.contact_exit_height_m,
            "contact_enter_seconds": self.contact_enter_seconds,
            "contact_exit_seconds": self.contact_exit_seconds,
            "detection_filter_time_constant_seconds": (
                self.detection_filter_time_constant_seconds
            ),
            "support_blend_time_constant_seconds": (
                self.support_blend_time_constant_seconds
            ),
            "maximum_interval_seconds": self.maximum_interval_seconds,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.contact_enter_speed_mps >= self.contact_exit_speed_mps:
            raise ValueError("foot-contact speed hysteresis is not ordered")
        if self.contact_enter_height_m >= self.contact_exit_height_m:
            raise ValueError("foot-contact height hysteresis is not ordered")
        if (
            not math.isfinite(self.minimum_foot_confidence)
            or not 0.0 <= self.minimum_foot_confidence <= 1.0
        ):
            raise ValueError("minimum_foot_confidence must be finite and in [0, 1]")


class CausalFootConditionedRoot:
    """Remove common-mode root XY drift while preserving motion over planted feet."""

    def __init__(
        self,
        joint_names: Sequence[str],
        config: FootConditionedRootConfig = FootConditionedRootConfig(),
    ) -> None:
        names = tuple(joint_names)
        if len(set(names)) != len(names):
            raise ValueError("root conditioner joint names must be unique")
        try:
            self._pelvis = names.index("pelvis")
            self._feet = np.asarray(
                [names.index(name) for name in ("left_foot", "right_foot")],
                dtype=np.int32,
            )
        except ValueError as error:
            raise ValueError(
                "root conditioner requires pelvis, left_foot, and right_foot"
            ) from error
        self._joint_count = len(names)
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._timestamp_ns: int | None = None
        self._filtered_positions: np.ndarray | None = None
        self._previous_filtered_root_xy: np.ndarray | None = None
        self._previous_filtered_feet: np.ndarray | None = None
        self._corrected_root_xy: np.ndarray | None = None
        self._contacts = np.zeros(2, dtype=bool)
        self._enter_elapsed = np.zeros(2, dtype=np.float64)
        self._exit_elapsed = np.zeros(2, dtype=np.float64)
        self._anchors = np.zeros((2, 2), dtype=np.float64)

    @property
    def contacts(self) -> np.ndarray:
        return self._contacts.copy()

    def condition(
        self,
        positions: np.ndarray,
        confidence: np.ndarray,
        *,
        timestamp_ns: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        positions = np.asarray(positions, dtype=np.float64)
        confidence = np.asarray(confidence, dtype=np.float64)
        if positions.shape != (self._joint_count, 3):
            raise ValueError(
                f"root conditioner positions have shape {positions.shape}; "
                f"expected {(self._joint_count, 3)}"
            )
        if confidence.shape != (self._joint_count,):
            raise ValueError(
                f"root conditioner confidence has shape {confidence.shape}; "
                f"expected {(self._joint_count,)}"
            )
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(confidence)):
            raise ValueError("root conditioner inputs must be finite")
        if np.any((confidence < 0.0) | (confidence > 1.0)):
            raise ValueError("root conditioner confidence must be in [0, 1]")
        timestamp_ns = int(timestamp_ns)
        if timestamp_ns <= 0:
            raise ValueError("root conditioner timestamp must be positive")
        if self._timestamp_ns is None:
            return self._initialize(positions, confidence, timestamp_ns)
        assert self._filtered_positions is not None
        assert self._previous_filtered_root_xy is not None
        assert self._previous_filtered_feet is not None
        assert self._corrected_root_xy is not None
        dt = (timestamp_ns - self._timestamp_ns) / 1.0e9
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("root conditioner timestamp did not advance")
        if dt > self.config.maximum_interval_seconds:
            self.reset()
            return self._initialize(positions, confidence, timestamp_ns)

        detection_alpha = _time_constant_alpha(
            dt, self.config.detection_filter_time_constant_seconds
        )
        self._filtered_positions += detection_alpha * (
            positions - self._filtered_positions
        )
        filtered_root_xy = self._filtered_positions[self._pelvis, :2]
        filtered_feet = self._filtered_positions[self._feet]
        foot_speed = (
            np.linalg.norm(filtered_feet - self._previous_filtered_feet, axis=1) / dt
        )
        relative_feet_xy = filtered_feet[:, :2] - filtered_root_xy
        lowest_foot_z = float(np.min(filtered_feet[:, 2]))
        foot_confidence = confidence[self._feet]
        raw_prediction = self._corrected_root_xy + (
            filtered_root_xy - self._previous_filtered_root_xy
        )
        entered = self._update_contacts(
            foot_speed,
            filtered_feet[:, 2] - lowest_foot_z,
            foot_confidence,
            dt,
        )
        for foot in np.flatnonzero(entered):
            self._anchors[foot] = raw_prediction + relative_feet_xy[foot]
        supporting = np.flatnonzero(self._contacts)
        if len(supporting):
            support_roots = self._anchors[supporting] - relative_feet_xy[supporting]
            weights = np.maximum(foot_confidence[supporting], 1.0e-6)
            support_target = np.average(support_roots, axis=0, weights=weights)
            blend = _time_constant_alpha(
                dt, self.config.support_blend_time_constant_seconds
            )
            corrected_root_xy = raw_prediction + blend * (
                support_target - raw_prediction
            )
        else:
            corrected_root_xy = raw_prediction

        result = positions.copy()
        result[:, :2] += corrected_root_xy - positions[self._pelvis, :2]
        self._timestamp_ns = timestamp_ns
        self._previous_filtered_root_xy = filtered_root_xy.copy()
        self._previous_filtered_feet = filtered_feet.copy()
        self._corrected_root_xy = corrected_root_xy
        return result, self.contacts

    def _initialize(
        self,
        positions: np.ndarray,
        confidence: np.ndarray,
        timestamp_ns: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._timestamp_ns = timestamp_ns
        self._filtered_positions = positions.copy()
        self._previous_filtered_root_xy = positions[self._pelvis, :2].copy()
        self._previous_filtered_feet = positions[self._feet].copy()
        self._corrected_root_xy = positions[self._pelvis, :2].copy()
        feet = positions[self._feet]
        relative_height = feet[:, 2] - float(np.min(feet[:, 2]))
        foot_confidence = confidence[self._feet]
        self._contacts = (foot_confidence >= self.config.minimum_foot_confidence) & (
            relative_height <= self.config.contact_enter_height_m
        )
        relative_feet_xy = feet[:, :2] - positions[self._pelvis, :2]
        self._anchors[self._contacts] = (
            self._corrected_root_xy + relative_feet_xy[self._contacts]
        )
        return positions.copy(), self.contacts

    def _update_contacts(
        self,
        speed: np.ndarray,
        relative_height: np.ndarray,
        confidence: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        entered = np.zeros(2, dtype=bool)
        for foot in range(2):
            confident = confidence[foot] >= self.config.minimum_foot_confidence
            if self._contacts[foot]:
                should_exit = (
                    not confident
                    or speed[foot] >= self.config.contact_exit_speed_mps
                    or relative_height[foot] >= self.config.contact_exit_height_m
                )
                self._exit_elapsed[foot] = (
                    self._exit_elapsed[foot] + dt if should_exit else 0.0
                )
                if self._exit_elapsed[foot] >= self.config.contact_exit_seconds:
                    self._contacts[foot] = False
                    self._exit_elapsed[foot] = 0.0
                self._enter_elapsed[foot] = 0.0
                continue
            should_enter = (
                confident
                and speed[foot] <= self.config.contact_enter_speed_mps
                and relative_height[foot] <= self.config.contact_enter_height_m
            )
            self._enter_elapsed[foot] = (
                self._enter_elapsed[foot] + dt if should_enter else 0.0
            )
            if self._enter_elapsed[foot] >= self.config.contact_enter_seconds:
                self._contacts[foot] = True
                entered[foot] = True
                self._enter_elapsed[foot] = 0.0
            self._exit_elapsed[foot] = 0.0
        return entered


def _time_constant_alpha(dt: float, time_constant: float) -> float:
    return 1.0 - math.exp(-dt / time_constant)
