"""Producer-to-consumer native hand contract over a real local gRPC connection."""

import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from bello_mujoco.hardware.pico_grpc import PicoGrpcConfig
from bello_mujoco.hardware.xrobot_hands import XRobotHandsConfig, XRobotHandsStream
from test_xrobot_hands import native_positions, neutral_landmarks


SERVER = r"""
import asyncio
import json
from pathlib import Path
import sys
import time

import grpc
import pose_stream_pb2_grpc as rpc
from server import Bridge
from source import Source
from tracking import Tracking

async def main():
    source = Source()
    source.connected = True
    bridge = Bridge(source, Tracking())
    server = grpc.aio.server()
    rpc.add_PoseStreamServicer_to_server(bridge, server)
    port = server.add_insecure_port('127.0.0.1:0')
    await server.start()
    Path(sys.argv[1]).write_text(str(port))
    hands = json.loads(Path(sys.argv[2]).read_text())
    sequence = 0
    while True:
        sequence += 1
        packet = {'timeStampNs': time.time_ns(), 'Hand': {}}
        for side, positions in hands.items():
            joints = []
            for x, y, z in positions:
                joints.append({'p': f'{x + sequence * 1e-6},{y},{z},0,0,0,1', 's': 15})
            packet['Hand'][side + 'Hand'] = {'isActive': 1, 'HandJointLocations': joints}
        source.ingest('synthetic-headset', packet)
        bridge.refresh()
        await asyncio.sleep(.02)

asyncio.run(main())
"""


def test_raw_sdk_packets_reach_hand_mapper_without_body_model_or_calibration(tmp_path):
    bridge = Path(__file__).parents[1] / "deps/pico-smplx-bridge"
    ready = tmp_path / "port"
    payload = tmp_path / "hands.json"
    joints, tips = neutral_landmarks()
    payload.write_text(
        json.dumps(
            {
                side: native_positions(side, joints, tips).tolist()
                for side in ("left", "right")
            }
        )
    )
    with (tmp_path / "server.log").open("w+") as log:
        process = subprocess.Popen(
            [sys.executable, "-c", SERVER, str(ready), str(payload)],
            cwd=bridge,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < deadline:
                if process.poll() is not None:
                    log.seek(0)
                    pytest.fail(log.read())
                time.sleep(0.01)
            assert ready.exists(), "synthetic bridge failed to start"
            config = XRobotHandsConfig(PicoGrpcConfig(f"127.0.0.1:{ready.read_text()}"))
            with XRobotHandsStream(config) as stream:
                frame = stream.wait_next(after_sequence=None, timeout_seconds=3)
            assert frame is not None and frame.track_id == "synthetic-headset"
            for side in ("left", "right"):
                target = frame.target(side)
                assert target is not None
                np.testing.assert_allclose(target.normalized_positions, 0, atol=1e-12)
        finally:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
