"""Exclusive hand leases for the barbell coordinator."""

import math
import time
from collections.abc import Callable

from bello_mujoco.hardware import hand_pb2 as pb
from bello_mujoco.hardware.hand_server import HandControlClient


class BarbellHandSession:
    """Never reacquire a lost lease or substitute tracking during a handoff."""

    def __init__(
        self,
        client: HandControlClient,
        *,
        renew_seconds: float = 0.2,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(renew_seconds) or renew_seconds <= 0:
            raise ValueError("renew_seconds must be positive and finite")
        self.client = client
        self.renew_seconds = renew_seconds
        self.clock = clock
        self._leases = []
        self._last_renew = 0.0
        self._closure = None

    def __enter__(self):
        status = self.client.status()
        if self.renew_seconds >= status.lease_timeout_seconds / 2:
            raise ValueError(
                "hand renewal interval must be less than half the lease timeout"
            )
        if {hand.side for hand in status.hands} != {pb.LEFT, pb.RIGHT}:
            raise ValueError("barbell mode requires both hands")
        try:
            for side in (pb.LEFT, pb.RIGHT):
                self._leases.append(self.client.acquire(side))
        except BaseException as error:
            try:
                self.close()
            except RuntimeError as cleanup_error:
                error.add_note(str(cleanup_error))
            raise
        self._last_renew = self.clock()
        return self

    def update(self, closure: float) -> pb.Status:
        if len(self._leases) != 2:
            raise RuntimeError("barbell hand session is not acquired")
        if closure not in (0.0, 1.0):
            raise ValueError("barbell finger closure must be zero or one")
        if closure != self._closure:
            for lease in self._leases:
                self.client.command(
                    pb.Command(
                        side=lease.side,
                        token=lease.token,
                        source=pb.APPLICATION,
                        positions=pb.Positions(values=[closure] * 6),
                    )
                )
            self._closure = closure
            self._last_renew = self.clock()
        elif self.clock() - self._last_renew >= self.renew_seconds:
            for lease in self._leases:
                self.client.keep_alive(lease)
            self._last_renew = self.clock()
        status = self.client.status()
        if len(status.hands) != 2 or any(
            not hand.owned
            or hand.source != pb.APPLICATION
            or not hand.feedback_fresh
            or hand.transport_fault
            or list(hand.target) != [closure] * 6
            for hand in status.hands
        ):
            raise RuntimeError(
                "barbell hand ownership, source, target, or feedback failed"
            )
        return status

    def close(self) -> None:
        errors = []
        for lease in reversed(self._leases):
            try:
                self.client.release(lease)
            except Exception as error:
                errors.append(error)
        self._leases.clear()
        self._closure = None
        if errors:
            raise RuntimeError(
                "failed to release barbell hand leases; await server expiry"
            ) from errors[0]

    def __exit__(self, _type, error, _traceback):
        try:
            self.close()
        except RuntimeError as cleanup_error:
            if error is None:
                raise
            error.add_note(str(cleanup_error))
        return False
