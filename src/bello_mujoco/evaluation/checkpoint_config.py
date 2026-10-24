"""Operational CPU checkpoint evaluation settings, separate from learning math."""

from dataclasses import dataclass
import math

from bello_mujoco.evaluation.world_model_audit import MotionWindow


@dataclass(frozen=True)
class CheckpointEvaluationConfig:
    enabled: bool = False
    cpu_cores: int = 1
    finish_timeout_seconds: float = 900.0
    windows: tuple[MotionWindow, ...] = ()

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("checkpoint evaluation enabled must be a boolean")
        if type(self.cpu_cores) is not int or not 1 <= self.cpu_cores <= 4:
            raise ValueError("cpu_cores must be between one and four")
        if (
            not math.isfinite(self.finish_timeout_seconds)
            or self.finish_timeout_seconds <= 0
        ):
            raise ValueError("finish_timeout_seconds must be positive and finite")
        identities = [(w.clip, w.start_frame) for w in self.windows]
        if self.enabled and not identities:
            raise ValueError("enabled checkpoint evaluation requires explicit windows")
        if len(set(identities)) != len(identities):
            raise ValueError("windows must have unique clip/start pairs")
