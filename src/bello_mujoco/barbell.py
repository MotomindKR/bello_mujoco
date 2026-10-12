"""Barbell mode sequencing, independent of policy inference and device I/O."""

from dataclasses import dataclass, fields
from enum import StrEnum
import math


class BarbellPhase(StrEnum):
    STAND = "normal_standing"
    REACH = "hands_forward"
    RECEIVE = "await_bar"
    CLOSE = "close_hands"
    LOAD = "transfer_load"
    ENTER_MOTION = "to_motion"
    MOTION = "motion"
    RETURN = "to_standing"
    SUPPORT = "await_support"
    UNLOAD = "transfer_to_holder"
    OPEN = "open_hands"
    CLEAR = "await_bar_clear"
    RETRACT = "hands_normal"
    FINISH = "normal_standing_final"
    COMPLETE = "complete"
    INTERRUPTED = "interrupted"
    FAULT = "fault"


class HandoffConfirmation(StrEnum):
    BAR_READY = "bar_ready"
    GRASP_SECURED = "grasp_secured"
    BAR_SUPPORTED = "bar_supported"
    HANDS_OPEN = "hands_open"
    BAR_CLEAR = "bar_clear"


@dataclass(frozen=True)
class BarbellModeConfig:
    initial_stand_seconds: float = 4.0
    reach_seconds: float = 4.0
    load_transfer_seconds: float = 2.0
    policy_blend_seconds: float = 4.0
    motion_seconds: float = 20.2
    retract_seconds: float = 4.0
    final_stand_seconds: float = 6.0

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field.name} must be positive and finite")


def _smooth(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value**3 * (10 + value * (-15 + 6 * value))


@dataclass(frozen=True)
class BarbellCommand:
    phase: BarbellPhase
    phase_id: int
    elapsed_seconds: float
    arm_fraction: float
    load_fraction: float
    motion_weight: float
    motion_time: float
    finger_closure: float
    awaiting: HandoffConfirmation | None
    interrupted: bool
    reason: str


class BarbellMode:
    """Confirmation-gated cycle; a stop never authorizes releasing a load.

    The body adapter continuously runs standing, including confirmation waits.
    During both policy blends it evaluates both policies on the live state.
    Confirmations carry a phase generation so stale button presses cannot advance
    another phase. Position agreement is never a grasp confirmation.
    """

    def __init__(self, config: BarbellModeConfig) -> None:
        self.config = config
        self.phase = BarbellPhase.STAND
        self.phase_id = 0
        self.elapsed = 0.0
        self._arm = 0.0
        self._load = 0.0
        self._weight = 0.0
        self._motion_time = 0.0
        self._closure = 0.0
        self._entry_arm = 0.0
        self._entry_load = 0.0
        self._entry_weight = 0.0
        self._possibly_grasped = False
        self.interrupted = False
        self.reason = ""

    def _enter(self, phase: BarbellPhase) -> None:
        self.phase = phase
        self.phase_id += 1
        self.elapsed = 0.0
        self._entry_arm = self._arm
        self._entry_load = self._load
        self._entry_weight = self._weight
        if phase == BarbellPhase.CLOSE:
            self._closure = 1.0
            self._possibly_grasped = True
        elif phase == BarbellPhase.OPEN:
            self._closure = 0.0

    def snapshot(self) -> BarbellCommand:
        awaiting = {
            BarbellPhase.RECEIVE: HandoffConfirmation.BAR_READY,
            BarbellPhase.CLOSE: HandoffConfirmation.GRASP_SECURED,
            BarbellPhase.SUPPORT: HandoffConfirmation.BAR_SUPPORTED,
            BarbellPhase.OPEN: HandoffConfirmation.HANDS_OPEN,
            BarbellPhase.CLEAR: HandoffConfirmation.BAR_CLEAR,
        }.get(self.phase)
        if self.phase == BarbellPhase.INTERRUPTED and self._possibly_grasped:
            awaiting = HandoffConfirmation.BAR_SUPPORTED
        return BarbellCommand(
            self.phase,
            self.phase_id,
            self.elapsed,
            self._arm,
            self._load,
            self._weight,
            self._motion_time,
            self._closure,
            awaiting,
            self.interrupted,
            self.reason,
        )

    def confirm(self, event: HandoffConfirmation, *, phase_id: int) -> None:
        if phase_id != self.phase_id or event != self.snapshot().awaiting:
            raise ValueError(
                "confirmation does not match the current phase and generation"
            )
        next_phase = {
            HandoffConfirmation.BAR_READY: BarbellPhase.CLOSE,
            HandoffConfirmation.GRASP_SECURED: BarbellPhase.LOAD,
            HandoffConfirmation.BAR_SUPPORTED: BarbellPhase.UNLOAD,
            HandoffConfirmation.HANDS_OPEN: BarbellPhase.CLEAR,
            HandoffConfirmation.BAR_CLEAR: BarbellPhase.RETRACT,
        }[event]
        if event == HandoffConfirmation.HANDS_OPEN:
            self._possibly_grasped = False
        self._enter(next_phase)

    def stop(self, reason: str) -> None:
        """Cancel motion and retain fingers/load until a fresh support confirmation."""
        if self.phase in (BarbellPhase.COMPLETE, BarbellPhase.FAULT):
            return
        if self.interrupted:
            return
        self.interrupted = True
        self.reason = reason
        if self._weight > 0:
            self._enter(BarbellPhase.RETURN)
        elif self._possibly_grasped:
            self._enter(BarbellPhase.INTERRUPTED)
        elif self.phase in (
            BarbellPhase.RECEIVE,
            BarbellPhase.OPEN,
            BarbellPhase.CLEAR,
        ):
            self._enter(BarbellPhase.CLEAR)
        else:
            self._enter(BarbellPhase.RETRACT)

    def fault(self, reason: str) -> None:
        """Latch invalid body state; preserve the last finger and preload intent."""
        self.reason = reason
        self.interrupted = True
        self._enter(BarbellPhase.FAULT)

    def advance(self, dt: float) -> BarbellCommand:
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("control timestep must be positive and finite")
        if self.phase in (BarbellPhase.COMPLETE, BarbellPhase.FAULT):
            return self.snapshot()
        self.elapsed += dt
        phase = self.phase
        durations = {
            BarbellPhase.STAND: self.config.initial_stand_seconds,
            BarbellPhase.REACH: self.config.reach_seconds,
            BarbellPhase.LOAD: self.config.load_transfer_seconds,
            BarbellPhase.ENTER_MOTION: self.config.policy_blend_seconds,
            BarbellPhase.MOTION: self.config.motion_seconds,
            BarbellPhase.RETURN: self.config.policy_blend_seconds,
            BarbellPhase.UNLOAD: self.config.load_transfer_seconds,
            BarbellPhase.RETRACT: self.config.retract_seconds,
            BarbellPhase.FINISH: self.config.final_stand_seconds,
        }
        duration = durations.get(phase)
        if duration is None:
            return self.snapshot()
        fraction = min(1.0, self.elapsed / duration)
        smooth = _smooth(fraction)
        if phase == BarbellPhase.REACH:
            self._arm = smooth
        elif phase == BarbellPhase.LOAD:
            self._load = smooth
        elif phase == BarbellPhase.ENTER_MOTION:
            self._weight = fraction
        elif phase == BarbellPhase.MOTION:
            self._motion_time = min(self.elapsed, duration)
        elif phase == BarbellPhase.RETURN:
            self._weight = self._entry_weight * (1 - fraction)
        elif phase == BarbellPhase.UNLOAD:
            self._load = self._entry_load * (1 - smooth)
        elif phase == BarbellPhase.RETRACT:
            self._arm = self._entry_arm * (1 - smooth)
        # Return the endpoint command before advancing to the next phase. This
        # retains one full control interval for every requested trajectory sample.
        result = self.snapshot()
        if self.elapsed + 1e-9 >= duration:
            next_phase = {
                BarbellPhase.STAND: BarbellPhase.REACH,
                BarbellPhase.REACH: BarbellPhase.RECEIVE,
                BarbellPhase.LOAD: BarbellPhase.ENTER_MOTION,
                BarbellPhase.ENTER_MOTION: BarbellPhase.MOTION,
                BarbellPhase.MOTION: BarbellPhase.RETURN,
                BarbellPhase.RETURN: BarbellPhase.SUPPORT,
                BarbellPhase.UNLOAD: BarbellPhase.OPEN,
                BarbellPhase.RETRACT: BarbellPhase.FINISH,
                BarbellPhase.FINISH: BarbellPhase.COMPLETE,
            }[phase]
            self._enter(next_phase)
        return result
