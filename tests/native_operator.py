"""Controller client for loopback runner integration tests."""

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
        self.channel = grpc.insecure_channel(endpoint)
        grpc.channel_ready_future(self.channel).result(timeout=8)
        method = self.channel.stream_stream(
            "/bello.operator.v1.OperatorService/Exchange",
            request_serializer=pb.ControllerSample.SerializeToString,
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
            if self.fake and self.fake.abort:
                fields.update(l1=True, r1=True)
            yield pb.ControllerSample(
                sequence=sequence,
                sample_monotonic_ns=time.monotonic_ns(),
                connected=self.connected,
                name="synthetic controller",
                **fields,
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
