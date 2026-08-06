"""Asynchronous, lossless policy-step logging for hardware runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import threading
import time
from types import TracebackType
from typing import Any
import warnings

import numpy as np


_STOP = object()


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
    """Write complete per-step policy records without file I/O in the control loop."""

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
        self._file = self.path.open("x", encoding="utf-8")
        self._queue: queue.SimpleQueue[dict[str, Any] | object] = queue.SimpleQueue()
        self._error: BaseException | None = None
        self._closed = False
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
        self._write_json(header)
        self._file.flush()
        self._thread = threading.Thread(
            target=self._run,
            name="bello-policy-step-log",
            daemon=True,
        )
        self._thread.start()
        print(f"Hardware policy-step log: {self.path}")

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
        self._raise_if_failed()
        if self._closed:
            raise RuntimeError("policy-step logger is closed")
        record: dict[str, Any] = {
            "type": "policy_step",
            "phase": phase,
            "step": step,
            "wall_time_ns": time.time_ns(),
            "monotonic_time_ns": time.monotonic_ns(),
            "elapsed_seconds": elapsed_seconds,
            "observation": _array_values(observation),
            "policy_observation": _array_values(policy_observation),
            "action": _array_values(action),
            "physical_target": _array_values(physical_target),
        }
        if raw_action is not None:
            record["raw_action"] = _array_values(raw_action)
        if command is not None:
            record["command"] = _array_values(command)
        if reference_time is not None:
            record["reference_time"] = reference_time
        if policy_target is not None:
            record["policy_target"] = _array_values(policy_target)
        self._queue.put(record)

    def close(self) -> None:
        if self._closed:
            self._raise_if_failed()
            return
        self._closed = True
        self._queue.put(_STOP)
        self._thread.join()
        self._raise_if_failed()

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

    def _run(self) -> None:
        try:
            while True:
                record = self._queue.get()
                if record is _STOP:
                    break
                self._write_json(record)
                self._file.flush()
        except BaseException as error:
            self._error = error
        finally:
            self._file.close()

    def _write_json(self, value: dict[str, Any]) -> None:
        json.dump(value, self._file, allow_nan=False, separators=(",", ":"))
        self._file.write("\n")

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("policy-step log writer failed") from self._error


def _array_values(value: Any) -> list[Any]:
    array = np.asarray(value)
    if not np.all(np.isfinite(array)):
        raise ValueError("policy-step log arrays must contain only finite values")
    return array.tolist()
