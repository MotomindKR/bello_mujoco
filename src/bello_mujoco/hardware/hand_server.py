"""Leased hand control with one CAN writer and an optional tracking input."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import logging
import math
import secrets
import threading
import time
from typing import Callable

import grpc

from bello_mujoco.hardware import hand_pb2 as pb
from bello_mujoco.hardware.hand_control import LinkerHandsCanTransport
from bello_mujoco.hardware.xrobot_hands import (
    HAND_POSITION_ORDER,
    XRobotHandCommandFilter,
    XRobotHandTarget,
    XRobotHandsConfig,
    XRobotHandsFrame,
    XRobotHandsStream,
    XRobotHandsUnavailable,
)

_SIDES = {pb.LEFT: "left", pb.RIGHT: "right"}
_SERVICE = "bello.hand.v1.HandControl"
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HandServerConfig:
    listen: str = "127.0.0.1:50057"
    lease_timeout_seconds: float = 1.0

    def __post_init__(self) -> None:
        if not self.listen or "\0" in self.listen:
            raise ValueError("hand server listen address must be nonempty")
        if (
            not math.isfinite(self.lease_timeout_seconds)
            or self.lease_timeout_seconds <= 0
        ):
            raise ValueError("lease_timeout_seconds must be positive and finite")


@dataclass
class _HandState:
    token: str = ""
    expires_at: float = 0.0
    source: int = pb.HOLD
    target: tuple[float, ...] = ()
    rate: float = 0.0
    active: bool = False
    primed_at: float | None = None
    commanded: tuple[int, ...] = ()
    measured: tuple[int, ...] = ()
    feedback_at: float = 0.0
    suspension: str = "No control source selected"
    fault: str = ""


class HandController:
    """Serialize authority changes and CAN writes; RPCs never access CAN.

    A lease grants exclusive control of one hand. All sources, including tracking,
    need a live lease. Expiry never opens a hand or restores tracking implicitly.
    """

    def __init__(
        self,
        config: XRobotHandsConfig,
        server_config: HandServerConfig,
        transport: LinkerHandsCanTransport,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not config.arm:
            raise ValueError("hand server requires armed, calibrated hand control")
        self.config = config
        self.server_config = server_config
        self.transport = transport
        self.clock = clock
        self._lock = threading.RLock()
        self._states = {}
        self._filters = {}
        self._frame: XRobotHandsFrame | None = None
        for side_id, side in _SIDES.items():
            if config.enabled(side):
                positions, stamp = transport.feedback(side)
                self._states[side_id] = _HandState(
                    rate=config.max_raw_rate_per_second,
                    measured=positions,
                    feedback_at=stamp,
                )
                self._filters[side_id] = XRobotHandCommandFilter(config)

    def _state(self, side: int) -> _HandState:
        if side not in self._states:
            raise ValueError("side must identify an enabled hand")
        return self._states[side]

    @staticmethod
    def _suspend(state: _HandState, reason: str) -> None:
        state.active = False
        state.primed_at = None
        state.suspension = reason

    def _expire(self, state: _HandState) -> None:
        if state.token and self.clock() >= state.expires_at:
            state.token = ""
            state.source = pb.HOLD
            state.target = ()
            self._suspend(state, "Application lease expired; writes suspended")

    def _owned(self, lease: pb.Lease | pb.Command) -> _HandState:
        state = self._state(lease.side)
        self._expire(state)
        if not lease.token or not secrets.compare_digest(
            lease.token.encode(), state.token.encode()
        ):
            raise PermissionError(
                "hand lease is absent, expired, or owned by another client"
            )
        return state

    def acquire(self, request: pb.Hand) -> pb.Lease:
        with self._lock:
            state = self._state(request.side)
            self._expire(state)
            if state.token:
                raise RuntimeError("hand already has an owner")
            state.token = secrets.token_urlsafe(24)
            state.expires_at = self.clock() + self.server_config.lease_timeout_seconds
            state.source = pb.HOLD
            state.target = ()
            self._suspend(state, "Owner acquired; waiting for command")
            return pb.Lease(side=request.side, token=state.token)

    def command(self, request: pb.Command) -> pb.Status:
        with self._lock:
            state = self._owned(request)
            if request.source not in (pb.HOLD, pb.TRACKING, pb.APPLICATION):
                raise ValueError("source must be HOLD, TRACKING, or APPLICATION")
            kind = request.WhichOneof("target")
            target = ()
            if request.source == pb.APPLICATION:
                if kind == "positions":
                    target = tuple(request.positions.values)
                elif kind in ("open", "close") and getattr(request, kind):
                    target = (0.0 if kind == "open" else 1.0,) * 6
                else:
                    raise ValueError(
                        "APPLICATION requires positions, open=true, or close=true"
                    )
                self.config.calibration(_SIDES[request.side]).raw_target(target)
            elif kind is not None:
                raise ValueError("only APPLICATION accepts a target")
            rate = (
                request.max_raw_rate_per_second
                if request.HasField("max_raw_rate_per_second")
                else self.config.max_raw_rate_per_second
            )
            if (
                not math.isfinite(rate)
                or not 0 < rate <= self.config.max_raw_rate_per_second
            ):
                raise ValueError(
                    "rate must be positive, finite, and no greater than the server limit"
                )
            if state.source != request.source or state.rate != rate:
                self._suspend(
                    state, "Control changed; awaiting measured-position reseed"
                )
                self._filters[request.side] = XRobotHandCommandFilter(
                    replace(self.config, max_raw_rate_per_second=rate)
                )
            state.source = request.source
            state.target = target
            state.rate = rate
            state.expires_at = self.clock() + self.server_config.lease_timeout_seconds
            if state.source == pb.HOLD:
                self._suspend(state, "HOLD selected; writes suspended")
            return self.status()

    def keep_alive(self, lease: pb.Lease) -> pb.Status:
        with self._lock:
            state = self._owned(lease)
            state.expires_at = self.clock() + self.server_config.lease_timeout_seconds
            return self.status()

    def release(self, lease: pb.Lease) -> pb.Status:
        with self._lock:
            state = self._owned(lease)
            state.token = ""
            state.source = pb.HOLD
            state.target = ()
            self._suspend(state, "Owner released; writes suspended")
            return self.status()

    def publish_tracking(self, frame: XRobotHandsFrame | None) -> None:
        with self._lock:
            changed_epoch = (
                self._frame is not None
                and frame is not None
                and self._frame.track_id != frame.track_id
            )
            self._frame = frame
            for side_id, state in self._states.items():
                if state.source == pb.TRACKING and (
                    changed_epoch
                    or frame is None
                    or frame.target(_SIDES[side_id]) is None
                ):
                    self._suspend(state, "Tracking unavailable or epoch changed")

    def _tracking_target(self, side_id: int) -> XRobotHandTarget | None:
        frame = self._frame
        if frame is None:
            return None
        age = frame.source_age_ms + (self.clock() - frame.received_at) * 1000
        if age < 0 or age > self.config.max_reference_age_ms:
            return None
        return frame.target(_SIDES[side_id])

    def tick(self) -> None:
        for side_id, state in self._states.items():
            with self._lock:
                self._expire(state)
                side = _SIDES[side_id]
                try:
                    self.transport.poll_feedback(side)
                    state.measured, state.feedback_at = self.transport.feedback(side)
                    if not state.measured or (
                        self.clock() - state.feedback_at
                        > self.transport.config.feedback_timeout_seconds
                    ):
                        raise TimeoutError(f"{side} CAN feedback unavailable or stale")
                    if state.fault:
                        state.fault = ""
                        self._suspend(state, "CAN recovered; awaiting position reseed")
                    if state.source == pb.HOLD:
                        continue
                    target = (
                        self._tracking_target(side_id)
                        if state.source == pb.TRACKING
                        else XRobotHandTarget(side, state.target)
                    )
                    if target is None:
                        self._suspend(state, "Tracking unavailable or stale")
                        continue
                    position_filter = self._filters[side_id]
                    if not state.active:
                        if state.primed_at is None:
                            measured = self.transport.refresh_positions(side)
                            state.measured, state.feedback_at = self.transport.feedback(
                                side
                            )
                            position_filter.seed(side, measured, now=self.clock())
                            state.primed_at = self.clock()
                            # Require a tracking observation received after the read.
                            # Application commands do not depend on a tracking frame.
                        if state.source == pb.TRACKING and (
                            self._frame is None
                            or self._frame.received_at <= state.primed_at
                        ):
                            state.suspension = (
                                "Waiting for tracking after position probe"
                            )
                            continue
                        state.active = True
                    self._expire(state)
                    if state.source == pb.HOLD:
                        continue
                    if state.source == pb.TRACKING:
                        target = self._tracking_target(side_id)
                        if target is None:
                            self._suspend(
                                state, "Tracking expired during position probe"
                            )
                            continue
                    # Never accumulate more than one control period after a stall.
                    now = self.clock()
                    position_filter.resume(side, now=now)
                    positions = position_filter.update(target, now=now)
                    self.transport.set_positions(side, positions)
                    state.commanded = positions
                    state.suspension = ""
                    state.fault = ""
                except OSError as error:
                    state.fault = str(error)
                    self._suspend(state, "CAN unavailable; waiting for fresh feedback")
                except ValueError as error:
                    state.fault = str(error)
                    state.source = pb.HOLD
                    state.target = ()
                    self._suspend(
                        state, "Transport fault; select a source again to retry"
                    )

    def status(self) -> pb.Status:
        with self._lock:
            result = pb.Status(
                position_order=HAND_POSITION_ORDER,
                lease_timeout_seconds=self.server_config.lease_timeout_seconds,
                grasp_feedback_supported=False,
                force_control_supported=False,
            )
            for side_id, state in self._states.items():
                self._expire(state)
                age = max(0.0, self.clock() - state.feedback_at)
                result.hands.add(
                    side=side_id,
                    source=state.source,
                    owned=bool(state.token),
                    lease_remaining_seconds=max(0.0, state.expires_at - self.clock())
                    if state.token
                    else 0.0,
                    target=state.target,
                    commanded_positions=state.commanded,
                    measured_positions=state.measured,
                    feedback_age_seconds=age,
                    feedback_fresh=bool(state.measured)
                    and not state.fault
                    and age <= self.transport.config.feedback_timeout_seconds,
                    tracking_fresh=self._tracking_target(side_id) is not None,
                    suspension_reason=state.suspension,
                    transport_fault=state.fault,
                    max_raw_rate_per_second=state.rate,
                )
            return result


_METHODS = {
    "Acquire": (pb.Hand, pb.Lease, "acquire"),
    "SetCommand": (pb.Command, pb.Status, "command"),
    "KeepAlive": (pb.Lease, pb.Status, "keep_alive"),
    "Release": (pb.Lease, pb.Status, "release"),
    "GetStatus": (pb.Empty, pb.Status, "status"),
}


def hand_service_handler(controller: HandController):
    def callback(method):
        def invoke(request, context):
            try:
                return (
                    controller.status()
                    if method == "status"
                    else getattr(controller, method)(request)
                )
            except ValueError as error:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
            except PermissionError as error:
                context.abort(grpc.StatusCode.PERMISSION_DENIED, str(error))
            except RuntimeError as error:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))

        return invoke

    return grpc.method_handlers_generic_handler(
        _SERVICE,
        {
            name: grpc.unary_unary_rpc_method_handler(
                callback(method),
                request_deserializer=request_type.FromString,
                response_serializer=response_type.SerializeToString,
            )
            for name, (request_type, response_type, method) in _METHODS.items()
        },
    )


class HandControlClient:
    """Typed client; callers renew leases while their selected source is active."""

    def __init__(self, channel: grpc.Channel, *, timeout_seconds: float = 2.0) -> None:
        self.timeout_seconds = timeout_seconds
        self._calls = {
            name: channel.unary_unary(
                f"/{_SERVICE}/{name}",
                request_serializer=request_type.SerializeToString,
                response_deserializer=response_type.FromString,
            )
            for name, (request_type, response_type, _) in _METHODS.items()
        }

    def acquire(self, side: int) -> pb.Lease:
        return self._calls["Acquire"](pb.Hand(side=side), timeout=self.timeout_seconds)

    def command(self, command: pb.Command) -> pb.Status:
        return self._calls["SetCommand"](command, timeout=self.timeout_seconds)

    def keep_alive(self, lease: pb.Lease) -> pb.Status:
        return self._calls["KeepAlive"](lease, timeout=self.timeout_seconds)

    def release(self, lease: pb.Lease) -> pb.Status:
        return self._calls["Release"](lease, timeout=self.timeout_seconds)

    def status(self) -> pb.Status:
        return self._calls["GetStatus"](pb.Empty(), timeout=self.timeout_seconds)


def _receive_tracking(config, controller, stop, stream_factory):
    while not stop.is_set():
        try:
            with stream_factory(config) as stream:
                sequence = None
                while not stop.is_set():
                    try:
                        frame = stream.wait_next(
                            after_sequence=sequence, timeout_seconds=0.1
                        )
                    except XRobotHandsUnavailable:
                        controller.publish_tracking(None)
                        sequence = None
                        stop.wait(0.01)
                    else:
                        if frame is not None:
                            sequence = frame.sequence
                            controller.publish_tracking(frame)
        except Exception as error:
            controller.publish_tracking(None)
            _LOGGER.warning("Hand tracking unavailable: %s", error)
            stop.wait(1.0)


def run_hand_server(
    config: XRobotHandsConfig,
    server_config: HandServerConfig,
    transport_factory,
    *,
    stop_event: threading.Event | None = None,
    stream_factory=XRobotHandsStream,
) -> None:
    stop = stop_event if stop_event is not None else threading.Event()
    if not config.arm:
        raise ValueError("hand server requires arm=true")
    with (
        transport_factory(config) as transport,
        ThreadPoolExecutor(max_workers=4) as executor,
    ):
        controller = HandController(config, server_config, transport)
        server = grpc.server(
            executor,
            options=(
                ("grpc.so_reuseport", 0),
                ("grpc.max_receive_message_length", 16384),
            ),
        )
        server.add_generic_rpc_handlers((hand_service_handler(controller),))
        if not server.add_insecure_port(server_config.listen):
            raise RuntimeError("cannot bind hand control server")
        tracking = threading.Thread(
            target=_receive_tracking,
            args=(config, controller, stop, stream_factory),
            name="hand-tracking",
            daemon=True,
        )
        server.start()
        tracking.start()
        _LOGGER.info(
            "Hand control listening at %s; both sources start suspended",
            server_config.listen,
        )
        deadline = (
            None
            if config.max_duration_seconds is None
            else time.monotonic() + config.max_duration_seconds
        )
        try:
            while not stop.is_set() and (
                deadline is None or time.monotonic() < deadline
            ):
                started = time.monotonic()
                controller.tick()
                stop.wait(
                    max(0, 1.0 / config.control_rate_hz - (time.monotonic() - started))
                )
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
            server.stop(0).wait(timeout=2.0)
            tracking.join(
                timeout=config.xrobot.connect_timeout_seconds
                + config.xrobot.rpc_timeout_seconds
                + 1.0
            )
