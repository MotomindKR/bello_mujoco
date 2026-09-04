"""Bounded, non-blocking policy-step logging for hardware runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import struct
import sys
import time
from types import TracebackType
from typing import Any
import warnings

import numpy as np


_HEADER = b"H"
_READY = b"R"
_ERROR = b"E"
_STEP = b"S"
_STOP = b"X"
_MAX_PACKET_BYTES = 1 << 20
_START_TIMEOUT_SECONDS = 5.0
_STOP_TIMEOUT_SECONDS = 10.0
_WRITER_SOCKET_FD = 3
_STEP_HEADER = struct.Struct("<HQQQdBd")
_ARRAY_HEADER = struct.Struct("<BBI")
_DROPPED_COUNT = struct.Struct("<Q")
_DTYPES = {
    1: np.dtype("<f4"),
    2: np.dtype("<f8"),
}


def policy_phase_schema(adapter: Any) -> dict[str, Any]:
    """Describe the ordered observation, action, and target arrays in a log."""

    env = adapter.env
    return {
        "observation_layout": [
            {"name": name, "size": size}
            for name, size in env.observation_layout.components
        ],
        "observation_stack_size": adapter.metadata.ppo.policy.obs_stack_size,
        "action_names": [
            env.model.actuator_joint_names[index]
            for index in env.model.policy_ctrl_indices
        ],
        "physical_target_names": list(adapter.physical_coordinate_names),
    }


class PolicyStepLogger:
    """Send fixed-size snapshots to an isolated, bounded log writer process."""

    def __init__(
        self,
        path: Path,
        *,
        task: str,
        checkpoints: dict[str, Path],
        phases: dict[str, dict[str, Any]],
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.open("x").close()
        self._phase_ids = {name: index for index, name in enumerate(phases)}
        if len(self._phase_ids) > 0xFFFF:
            self.path.unlink(missing_ok=True)
            raise ValueError("policy log has too many phases")
        self._dropped_records = 0
        self._closed = False
        self._unavailable = False
        header = {
            "type": "header",
            "format_version": 1,
            "task": task,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoints": {
                name: str(Path(checkpoint).expanduser().resolve())
                for name, checkpoint in checkpoints.items()
            },
            "phases": phases,
        }
        self._socket, child_socket = socket.socketpair(
            socket.AF_UNIX,
            socket.SOCK_DGRAM,
        )
        try:
            child_fd = child_socket.fileno()
            file_actions = [(os.POSIX_SPAWN_DUP2, child_fd, _WRITER_SOCKET_FD)]
            if child_fd != _WRITER_SOCKET_FD:
                file_actions.append((os.POSIX_SPAWN_CLOSE, child_fd))
            self._writer_pid = os.posix_spawn(
                sys.executable,
                (
                    sys.executable,
                    "-m",
                    "bello_mujoco.hardware.policy_log",
                    "--writer",
                    str(_WRITER_SOCKET_FD),
                    str(self.path),
                ),
                os.environ,
                file_actions=file_actions,
            )
        except BaseException:
            self._socket.close()
            child_socket.close()
            self.path.unlink(missing_ok=True)
            raise
        child_socket.close()
        try:
            self._start_writer(header)
        except BaseException:
            self._socket.close()
            _terminate_writer(self._writer_pid)
            self.path.unlink(missing_ok=True)
            raise
        print(f"Hardware policy-step log: {self.path}")

    @property
    def dropped_records(self) -> int:
        return self._dropped_records

    def write_step(
        self,
        *,
        phase: str,
        step: int,
        elapsed_seconds: float,
        observation: Any,
        policy_observation: Any,
        action: Any,
        physical_target: Any,
        raw_action: Any | None = None,
        command: Any | None = None,
        reference_time: float | None = None,
        policy_target: Any | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("policy-step logger is closed")
        if self._unavailable:
            self._dropped_records += 1
            return
        try:
            packet = _encode_step(
                phase_id=self._phase_ids[phase],
                step=step,
                wall_time_ns=time.time_ns(),
                monotonic_time_ns=time.monotonic_ns(),
                elapsed_seconds=elapsed_seconds,
                observation=observation,
                policy_observation=policy_observation,
                raw_action=raw_action,
                action=action,
                physical_target=physical_target,
                command=command,
                reference_time=reference_time,
                policy_target=policy_target,
            )
            self._send_packet(packet)
        except (BlockingIOError, InterruptedError):
            self._dropped_records += 1
        except (BrokenPipeError, ConnectionError, OSError):
            self._dropped_records += 1
            self._unavailable = True
        except Exception:
            self._dropped_records += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close_error: BaseException | None = None
        try:
            if not self._unavailable:
                self._socket.setblocking(True)
                self._socket.settimeout(_START_TIMEOUT_SECONDS)
                self._socket.sendall(_STOP + _DROPPED_COUNT.pack(self._dropped_records))
                self._socket.shutdown(socket.SHUT_WR)
        except (BrokenPipeError, ConnectionError, OSError) as error:
            close_error = error
        finally:
            self._socket.close()
        return_code = _wait_writer(self._writer_pid, _STOP_TIMEOUT_SECONDS)
        if return_code is None:
            _terminate_writer(self._writer_pid)
            raise RuntimeError("policy-step log writer did not stop")
        if return_code != 0:
            raise RuntimeError(
                f"policy-step log writer exited with status {return_code}"
            ) from close_error
        if close_error is not None:
            raise RuntimeError(
                "policy-step log writer closed unexpectedly"
            ) from close_error

    def __enter__(self) -> PolicyStepLogger:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> bool:
        try:
            self.close()
        except BaseException as error:
            if exception_type is None:
                raise
            warnings.warn(
                f"policy-step log cleanup failed: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
        return False

    def _start_writer(self, header: dict[str, Any]) -> None:
        self._socket.settimeout(_START_TIMEOUT_SECONDS)
        payload = (
            _HEADER
            + json.dumps(
                header,
                allow_nan=False,
                separators=(",", ":"),
            ).encode()
        )
        if self._socket.send(payload) != len(payload):
            raise RuntimeError("failed to initialize policy-step log writer")
        response = self._socket.recv(_MAX_PACKET_BYTES)
        if response != _READY:
            detail = response.removeprefix(_ERROR).decode(errors="replace")
            raise RuntimeError(f"failed to initialize policy-step log writer: {detail}")
        self._socket.setblocking(False)

    def _send_packet(self, packet: bytes) -> None:
        if self._socket.send(packet) != len(packet):
            raise BlockingIOError("partial policy log packet submission")


def _encode_step(
    *,
    phase_id: int,
    step: int,
    wall_time_ns: int,
    monotonic_time_ns: int,
    elapsed_seconds: float,
    observation: Any,
    policy_observation: Any,
    raw_action: Any | None,
    action: Any,
    physical_target: Any,
    command: Any | None,
    reference_time: float | None,
    policy_target: Any | None,
) -> bytes:
    flags = 0
    flags |= int(raw_action is not None) << 0
    flags |= int(command is not None) << 1
    flags |= int(reference_time is not None) << 2
    flags |= int(policy_target is not None) << 3
    parts = [
        _STEP,
        _STEP_HEADER.pack(
            phase_id,
            step,
            wall_time_ns,
            monotonic_time_ns,
            elapsed_seconds,
            flags,
            0.0 if reference_time is None else reference_time,
        ),
    ]
    for value in (
        observation,
        policy_observation,
        raw_action,
        action,
        physical_target,
        command,
        policy_target,
    ):
        parts.append(_encode_array(value))
    packet = b"".join(parts)
    if len(packet) > _MAX_PACKET_BYTES:
        raise ValueError("policy-step log record exceeds maximum packet size")
    return packet


def _encode_array(value: Any | None) -> bytes:
    if value is None:
        return _ARRAY_HEADER.pack(0, 0, 0)
    array = np.asarray(value)
    if not np.all(np.isfinite(array)):
        raise ValueError("policy-step log array contains a non-finite value")
    if array.ndim > 0xFF or any(size > 0xFFFF_FFFF for size in array.shape):
        raise ValueError("policy-step log array shape is too large")
    if array.dtype == np.float32:
        dtype_code = 1
        contiguous = np.ascontiguousarray(array, dtype="<f4")
    else:
        dtype_code = 2
        contiguous = np.ascontiguousarray(array, dtype="<f8")
    data = contiguous.tobytes()
    shape = struct.pack(f"<{array.ndim}I", *array.shape)
    return _ARRAY_HEADER.pack(dtype_code, array.ndim, len(data)) + shape + data


def _decode_step(packet: bytes, phase_names: tuple[str, ...]) -> dict[str, Any]:
    view = memoryview(packet)
    offset = 1
    (
        phase_id,
        step,
        wall_time_ns,
        monotonic_time_ns,
        elapsed_seconds,
        flags,
        reference_time,
    ) = _STEP_HEADER.unpack_from(view, offset)
    offset += _STEP_HEADER.size
    arrays = []
    for _ in range(7):
        value, offset = _decode_array(view, offset)
        arrays.append(value)
    if offset != len(view):
        raise ValueError("policy-step log packet has trailing bytes")
    record = {
        "type": "policy_step",
        "phase": phase_names[phase_id],
        "step": step,
        "wall_time_ns": wall_time_ns,
        "monotonic_time_ns": monotonic_time_ns,
        "elapsed_seconds": elapsed_seconds,
        "observation": arrays[0],
        "policy_observation": arrays[1],
        "action": arrays[3],
        "physical_target": arrays[4],
    }
    if flags & (1 << 0):
        record["raw_action"] = arrays[2]
    if flags & (1 << 1):
        record["command"] = arrays[5]
    if flags & (1 << 2):
        record["reference_time"] = reference_time
    if flags & (1 << 3):
        record["policy_target"] = arrays[6]
    return record


def _decode_array(view: memoryview, offset: int) -> tuple[Any | None, int]:
    dtype_code, dimensions, byte_count = _ARRAY_HEADER.unpack_from(view, offset)
    offset += _ARRAY_HEADER.size
    if dtype_code == 0:
        if dimensions != 0 or byte_count != 0:
            raise ValueError("invalid absent policy-step log array")
        return None, offset
    dtype = _DTYPES.get(dtype_code)
    if dtype is None:
        raise ValueError("unknown policy-step log array dtype")
    shape_size = dimensions * 4
    shape = struct.unpack_from(f"<{dimensions}I", view, offset)
    offset += shape_size
    end = offset + byte_count
    if end > len(view):
        raise ValueError("truncated policy-step log array")
    expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if expected != byte_count:
        raise ValueError("invalid policy-step log array byte count")
    array = np.frombuffer(view[offset:end], dtype=dtype).reshape(shape)
    return array.tolist(), end


def _write_json_line(output: Any, value: dict[str, Any]) -> None:
    json.dump(value, output, allow_nan=False, separators=(",", ":"))
    output.write("\n")


def _writer(socket_fd: int, path: Path) -> int:
    try:
        os.nice(10)
    except OSError:
        pass
    channel = socket.socket(fileno=socket_fd)
    try:
        packet = channel.recv(_MAX_PACKET_BYTES)
        if not packet.startswith(_HEADER):
            raise ValueError("policy-step log header packet is missing")
        header = json.loads(packet[1:])
        phase_names = tuple(header["phases"])
        with path.open("w", encoding="utf-8", buffering=1 << 20) as output:
            _write_json_line(output, header)
            output.flush()
            channel.sendall(_READY)
            records_since_flush = 0
            last_flush = time.monotonic()
            while True:
                packet = channel.recv(_MAX_PACKET_BYTES)
                if not packet:
                    break
                if packet.startswith(_STEP):
                    _write_json_line(output, _decode_step(packet, phase_names))
                    records_since_flush += 1
                    now = time.monotonic()
                    if records_since_flush >= 256 or now - last_flush >= 1.0:
                        output.flush()
                        records_since_flush = 0
                        last_flush = now
                    continue
                if packet.startswith(_STOP):
                    (dropped_records,) = _DROPPED_COUNT.unpack(packet[1:])
                    if dropped_records:
                        _write_json_line(
                            output,
                            {
                                "type": "logger_summary",
                                "dropped_records": dropped_records,
                            },
                        )
                    break
                raise ValueError("unknown policy-step log packet")
            output.flush()
    except BaseException as error:
        try:
            channel.sendall(_ERROR + str(error).encode())
        except OSError:
            pass
        print(f"policy-step log writer failed: {error}", file=sys.stderr)
        return 1
    finally:
        channel.close()
    return 0


def _wait_writer(pid: int, timeout_seconds: float) -> int | None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        waited_pid, status = os.waitpid(pid, os.WNOHANG)
        if waited_pid == pid:
            return os.waitstatus_to_exitcode(status)
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.01)


def _terminate_writer(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass


def _main(arguments: list[str]) -> int:
    if len(arguments) != 3 or arguments[0] != "--writer":
        print("policy_log is an internal writer process", file=sys.stderr)
        return 2
    return _writer(int(arguments[1]), Path(arguments[2]))


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
