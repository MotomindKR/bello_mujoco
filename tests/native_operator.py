"""Operator client for loopback runner integration tests."""

import socket
import threading
import time

import grpc

from bello_mujoco.hardware import _operator_pb2 as pb


def operator_endpoint():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return f"127.0.0.1:{listener.getsockname()[1]}"


class OperatorClient:
    def __init__(self, endpoint, fake=None, *, buttons=None, wait_neutral=True):
        self.fake = fake
        self.buttons = buttons or {}
        self.connected = True
        self.status = None
        self.error = None
        self.closed = threading.Event()
        self.paused = threading.Event()
        self.previous = {}
        self.channel = grpc.insecure_channel(endpoint)
        grpc.channel_ready_future(self.channel).result(timeout=8)
        method = self.channel.stream_stream(
            "/bello.operator.v1.OperatorService/Exchange",
            request_serializer=pb.OperatorInput.SerializeToString,
            response_deserializer=pb.OperatorStatus.FromString,
        )
        self.call = method(self.samples())
        self.thread = threading.Thread(target=self.receive, daemon=True)
        self.thread.start()
        if wait_neutral:
            self.wait(lambda status: not status.neutral_required)

    def samples(self):
        sequence = 0
        while not self.closed.is_set():
            if self.paused.is_set():
                self.closed.wait(0.02)
                continue
            sequence += 1
            fields = dict(self.buttons)
            forward = fields.pop("forward", 0.0)
            lateral = fields.pop("lateral", 0.0)
            yaw = fields.pop("yaw", 0.0)
            selection = fields.pop("selection", None)
            buttons = {
                name: bool(fields.pop(name, False))
                for name in (
                    "back",
                    "start",
                    "a",
                    "y",
                    "l1",
                    "r1",
                    "l2",
                    "r2",
                    "dpad_up",
                    "dpad_down",
                    "dpad_left",
                    "dpad_right",
                )
            }
            if fields:
                raise ValueError(f"unknown controller controls: {sorted(fields)}")
            if not self.connected:
                forward = lateral = yaw = 0.0
                buttons = dict.fromkeys(buttons, False)
            neutral_axes = all(abs(value) <= 0.05 for value in (forward, lateral, yaw))
            other_controls = any(
                value for name, value in buttons.items() if name not in ("l2", "r2")
            )
            abort = bool(self.fake and self.fake.abort) or (
                buttons["l2"] and buttons["r2"]
            )
            confirm = (
                buttons["a"]
                and not abort
                and not any(
                    value
                    for name, value in buttons.items()
                    if name
                    not in ("a", "dpad_up", "dpad_down", "dpad_left", "dpad_right")
                )
            )
            mode = self.status.mode if self.status else pb.MODE_DISARMED
            selected = self.status.selected if self.status else ""
            mode_step = int(
                buttons["dpad_right"]
                and not buttons["dpad_left"]
                and not self.previous.get("dpad_right")
            ) - int(
                buttons["dpad_left"]
                and not buttons["dpad_right"]
                and not self.previous.get("dpad_left")
            )
            motion_step = int(
                buttons["dpad_down"]
                and not buttons["dpad_up"]
                and not self.previous.get("dpad_down")
            ) - int(
                buttons["dpad_up"]
                and not buttons["dpad_down"]
                and not self.previous.get("dpad_up")
            )
            toggle_tracking = buttons["y"] and not self.previous.get("y")
            self.previous = buttons
            yield pb.OperatorInput(
                sequence=sequence,
                sample_monotonic_ns=time.monotonic_ns(),
                connected=self.connected,
                name="synthetic controller",
                forward=forward,
                lateral=lateral,
                yaw=yaw,
                controls_neutral=neutral_axes and not any(buttons.values()),
                abort_other_controls_neutral=neutral_axes and not other_controls,
                request_hold=buttons["back"]
                and buttons["start"]
                and not abort
                and not any(
                    value
                    for name, value in buttons.items()
                    if name not in ("back", "start")
                ),
                confirm_startup=confirm and mode == pb.MODE_HOLD,
                confirm_sit=confirm and mode == pb.MODE_SIT,
                confirm_fault_recovery=confirm and mode == pb.MODE_FAULT,
                request_mode=(
                    selected
                    if mode == pb.MODE_STAND
                    else "Stand"
                    if mode in (pb.MODE_LOCOMOTION, pb.MODE_TRACKING)
                    else ""
                )
                if confirm
                else "",
                abort=abort,
                selection=selection,
                mode_step=mode_step,
                motion_step=motion_step,
                navigating=any(
                    buttons[name]
                    for name in ("dpad_up", "dpad_down", "dpad_left", "dpad_right")
                ),
                toggle_tracking=toggle_tracking,
                input_protocol_version=3,
            )
            self.closed.wait(0.02)

    def receive(self):
        try:
            for status in self.call:
                self.status = status
        except grpc.RpcError as error:
            self.error = error

    def wait(self, predicate, timeout=8):
        deadline = time.monotonic() + timeout
        while self.status is None or not predicate(self.status):
            if time.monotonic() >= deadline:
                raise AssertionError(f"operator timed out: {self.status}; {self.error}")
            time.sleep(0.01)
        return self.status

    def press(self, seconds=0.08, **buttons):
        self.buttons = buttons
        time.sleep(seconds)
        self.buttons = {}
        time.sleep(0.08)

    def stand(self):
        self.press(2.1, back=True, start=True)
        self.wait(
            lambda status: status.mode == pb.MODE_HOLD and status.phase == "holding"
        )
        # Phase changes require a released A sample before a new confirmation.
        self.press()
        self.press(1.1, a=True)
        self.wait(lambda status: status.mode == pb.MODE_STAND)

    def select(self, name):
        mode = "Motion" if name in self.status.motions else name
        for _ in range(len(self.status.modes)):
            if self.status.selected == mode:
                break
            self.press(dpad_right=True)
        else:
            raise AssertionError(f"mode {name} is unavailable: {self.status}")
        if mode == "Motion":
            for _ in range(len(self.status.motions)):
                if self.status.selected_motion == name:
                    return
                self.press(dpad_down=True)
            raise AssertionError(f"motion {name} is unavailable: {self.status}")

    def close(self):
        self.closed.set()
        self.call.cancel()
        self.thread.join(timeout=2)
        self.channel.close()
