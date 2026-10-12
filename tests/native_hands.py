"""Real leased hand API over synthetic positions and tracking; never CAN."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading
import time

import grpc
import pytest

from bello_mujoco.hardware.hand_server import (
    HandController,
    HandServerConfig,
    hand_service_handler,
)
from bello_mujoco.hardware.pico_grpc import PicoGrpcConfig
from bello_mujoco.hardware.xrobot_hands import (
    XRobotHandsConfig,
    XRobotHandsFrame,
    XRobotHandTarget,
)


class Transport:
    config = SimpleNamespace(feedback_timeout_seconds=0.5)

    def __init__(self):
        self.positions = {side: (200,) * 6 for side in ("left", "right")}
        self.stamps = {side: time.monotonic() for side in self.positions}
        self.writes = []

    def feedback(self, side):
        return self.positions[side], self.stamps[side]

    def poll_feedback(self, side):
        self.stamps[side] = time.monotonic()

    def refresh_positions(self, side):
        self.poll_feedback(side)
        return self.positions[side]

    def set_positions(self, side, positions):
        self.positions[side] = tuple(positions)
        self.writes.append((side, tuple(positions)))


@pytest.fixture
def hand_service():
    transport = Transport()
    controller = HandController(
        XRobotHandsConfig(
            xrobot=PicoGrpcConfig(endpoint="127.0.0.1:1"),
            arm=True,
            calibration_confirmed=True,
            smoothing_time_constant_ms=0,
        ),
        HandServerConfig(),
        transport,
    )
    pool = ThreadPoolExecutor(max_workers=4)
    server = grpc.server(pool)
    server.add_generic_rpc_handlers((hand_service_handler(controller),))
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    stopped = threading.Event()
    service = SimpleNamespace(
        controller=controller,
        transport=transport,
        server=server,
        tracking=True,
        config=dict(
            endpoint=f"http://127.0.0.1:{port}",
            rpc_timeout_seconds=0.05,
            renew_period_seconds=0.1,
        ),
    )

    def tick():
        sequence = 0
        while not stopped.wait(0.01):
            sequence += 1
            controller.publish_tracking(
                XRobotHandsFrame(
                    sequence=sequence,
                    source_timestamp_ns=sequence,
                    source_age_ms=0,
                    received_at=time.monotonic(),
                    track_id="synthetic",
                    left=XRobotHandTarget("left", (0.5,) * 6),
                    right=XRobotHandTarget("right", (0.5,) * 6),
                )
                if service.tracking
                else None
            )
            controller.tick()

    thread = threading.Thread(target=tick, daemon=True)
    thread.start()
    try:
        yield service
    finally:
        stopped.set()
        thread.join(timeout=2)
        server.stop(0).wait()
        pool.shutdown(wait=True)


def wait_hands(service, predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while True:
        status = service.controller.status()
        if predicate(status.hands):
            return status
        assert time.monotonic() < deadline, status
        time.sleep(0.01)
