"""Standalone Linker Hand transport for application and tracking control."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
import time
from typing import Literal

from bello_mujoco.hardware.linker_hand import (
    LinkerHandAddress,
    LinkerHandCanEndpoint,
    LinkerHandPositionResponse,
)


HandSide = Literal["left", "right"]


@dataclass(frozen=True, slots=True)
class LinkerHandsCanConfig:
    """Per-hand bus settings for two Linker O6/L6-compatible hands."""

    left_interface: str = "can2"
    right_interface: str = "can3"
    model: Literal["O6", "L6"] = "O6"
    left_enabled: bool = True
    right_enabled: bool = True
    probe_timeout_seconds: float = 0.5
    feedback_poll_interval_seconds: float = 0.2
    feedback_timeout_seconds: float = 0.5
    tx_spacing_seconds: float = 0.003

    def __post_init__(self) -> None:
        for name in ("left_interface", "right_interface"):
            interface = getattr(self, name)
            if not interface or "\0" in interface:
                raise ValueError(f"{name} must be a nonempty SocketCAN interface")
        if self.model not in {"O6", "L6"}:
            raise ValueError("hand model must be O6 or L6")
        if not self.left_enabled and not self.right_enabled:
            raise ValueError("at least one hand must be enabled")
        for name in (
            "probe_timeout_seconds",
            "feedback_poll_interval_seconds",
            "feedback_timeout_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.feedback_timeout_seconds < self.feedback_poll_interval_seconds:
            raise ValueError(
                "feedback_timeout_seconds must not be shorter than the poll interval"
            )
        if not math.isfinite(self.tx_spacing_seconds) or self.tx_spacing_seconds < 0.0:
            raise ValueError("tx_spacing_seconds must be finite and nonnegative")


class LinkerHandsCanTransport:
    """Own filtered sockets while leaving the shared CAN link untouched.

    The transport probes each configured hand with a read-only position request.
    It never brings the interface up or down and never invents an emergency-stop
    command that is absent from the reviewed O6/L6 protocol.
    """

    def __init__(
        self,
        config: LinkerHandsCanConfig,
        *,
        endpoint_factory: Callable[
            [str, LinkerHandAddress], LinkerHandCanEndpoint
        ] = LinkerHandCanEndpoint,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._endpoint_factory = endpoint_factory
        self._clock = clock
        self._sleep = sleep
        self._endpoints: dict[HandSide, LinkerHandCanEndpoint] = {}
        self._positions: dict[HandSide, tuple[int, int, int, int, int, int]] = {}
        self._last_feedback: dict[HandSide, float] = {}
        self._last_query: dict[HandSide, float] = {}
        self._last_tx: float | None = None
        self._entered = False

    @property
    def initial_positions(
        self,
    ) -> dict[HandSide, tuple[int, int, int, int, int, int]]:
        if not self._entered:
            raise RuntimeError("hand CAN transport is not running")
        return dict(self._positions)

    def __enter__(self) -> LinkerHandsCanTransport:
        if self._entered:
            raise RuntimeError("hand CAN transport is already running")
        try:
            for side in self._configured_sides():
                endpoint = self._endpoint_factory(
                    getattr(self.config, f"{side}_interface"),
                    LinkerHandAddress(self.config.model, side),
                )
                self._endpoints[side] = endpoint
            for side, endpoint in self._endpoints.items():
                self._stagger()
                response = endpoint.probe_positions(
                    timeout_s=self.config.probe_timeout_seconds
                )
                now = self._clock()
                self._record_response(side, response, now)
                self._last_query[side] = now
                self._last_tx = now
        except BaseException:
            self.close()
            raise
        self._entered = True
        return self

    def __exit__(self, *_exception: object) -> bool:
        self.close()
        return False

    def close(self) -> None:
        first_error: BaseException | None = None
        for endpoint in reversed(tuple(self._endpoints.values())):
            try:
                endpoint.close()
            except BaseException as error:  # pragma: no cover - defensive cleanup
                first_error = first_error or error
        self._endpoints.clear()
        self._entered = False
        if first_error is not None:
            raise first_error

    def set_positions(self, side: HandSide, positions: Sequence[int]) -> None:
        """Write the latest six-byte target after checking live feedback."""

        if not self._entered:
            raise RuntimeError("hand CAN transport is not running")
        endpoint = self._endpoints.get(side)
        if endpoint is None:
            raise ValueError(f"hand side {side!r} is not enabled")
        now = self._clock()
        self._drain_positions(side, endpoint, now)
        if now - self._last_feedback[side] > self.config.feedback_timeout_seconds:
            # A caller must explicitly refresh and reseed its command filter;
            # sending a target computed from the old state could jump a hand
            # that was reset or back-driven while feedback was absent.
            raise TimeoutError(
                f"{side} Linker Hand feedback is stale; refresh and reseed before "
                "writing"
            )
        if now - self._last_query[side] >= (self.config.feedback_poll_interval_seconds):
            self._stagger()
            endpoint.query_positions()
            now = self._clock()
            self._last_query[side] = now
            self._last_tx = now
        self._stagger()
        endpoint.write_positions(positions)
        self._last_tx = self._clock()

    def refresh_positions(self, side: HandSide) -> tuple[int, int, int, int, int, int]:
        """Force a read-after-drain position sample for recovery reseeding."""

        if not self._entered:
            raise RuntimeError("hand CAN transport is not running")
        endpoint = self._endpoints.get(side)
        if endpoint is None:
            raise ValueError(f"hand side {side!r} is not enabled")
        self._stagger()
        response = endpoint.probe_positions(timeout_s=self.config.probe_timeout_seconds)
        now = self._clock()
        self._record_response(side, response, now)
        self._last_query[side] = now
        self._last_tx = now
        return response.positions

    def poll_feedback(self, side: HandSide) -> None:
        """Poll even suspended hands without writing an actuator target."""
        if not self._entered:
            raise RuntimeError("hand CAN transport is not running")
        endpoint = self._endpoints[side]
        now = self._clock()
        self._drain_positions(side, endpoint, now)
        if now - self._last_query[side] >= self.config.feedback_poll_interval_seconds:
            self._stagger()
            endpoint.query_positions()
            self._last_query[side] = self._clock()
            self._last_tx = self._last_query[side]

    def feedback(self, side: HandSide) -> tuple[tuple[int, ...], float]:
        """Return cached measured positions and their local receipt time."""
        if not self._entered:
            raise RuntimeError("hand CAN transport is not running")
        return self._positions[side], self._last_feedback[side]

    def _configured_sides(self) -> tuple[HandSide, ...]:
        sides: list[HandSide] = []
        if self.config.left_enabled:
            sides.append("left")
        if self.config.right_enabled:
            sides.append("right")
        return tuple(sides)

    def _record_response(
        self,
        side: HandSide,
        response: LinkerHandPositionResponse,
        now: float,
    ) -> None:
        self._positions[side] = response.positions
        self._last_feedback[side] = now

    def _drain_positions(
        self,
        side: HandSide,
        endpoint: LinkerHandCanEndpoint,
        now: float,
    ) -> None:
        while (response := endpoint.receive_positions()) is not None:
            self._record_response(side, response, now)

    def _stagger(self) -> None:
        if self._last_tx is None:
            return
        remaining = self.config.tx_spacing_seconds - (self._clock() - self._last_tx)
        if remaining > 0.0:
            self._sleep(remaining)


__all__ = [
    "HandSide",
    "LinkerHandsCanConfig",
    "LinkerHandsCanTransport",
]
