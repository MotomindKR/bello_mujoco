"""Task-independent retargeted references and stream availability errors."""

from __future__ import annotations
from dataclasses import dataclass
import time
from typing import Protocol
import numpy as np


@dataclass(frozen=True)
class RobotReferenceFrame:
    """One GMR-retargeted Bello reference in MuJoCo qpos/qvel ordering."""

    qpos: np.ndarray
    qvel: np.ndarray
    sequence: int
    source_age_ms: int | None
    received_at: float
    torso_command: np.ndarray  # Anatomical roll/pitch lean and pelvis-heading twist.

    def age_ms(self, now: float | None = None) -> float | None:
        if self.source_age_ms is None:
            return None
        current = time.monotonic() if now is None else now
        return self.source_age_ms + 1_000.0 * max(0.0, current - self.received_at)


class ReferenceUnavailable(RuntimeError):
    """The live reference is temporarily unsafe or unavailable."""


class ReferenceDiscontinuity(ReferenceUnavailable):
    """The tracked subject or source timeline changed and must be re-anchored."""


class RobotReferenceSource(Protocol):
    """Reference boundary implemented by the vision/GMR integration."""

    def standing(self) -> RobotReferenceFrame: ...

    def latest(self) -> RobotReferenceFrame: ...
