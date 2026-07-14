"""Crash-isolated checkpoint evaluation for motion training."""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import time
import traceback

from bello_mujoco.tools.eval_motion_policy import (
    MotionRolloutStats,
    render_checkpoint_with_stats,
)


@dataclass(frozen=True)
class MotionEvaluationRequest:
    checkpoint: Path
    output_path: Path
    steps: int
    width: int
    height: int
    fps: int


@dataclass(frozen=True)
class MotionEvaluationResponse:
    stats: MotionRolloutStats | None = None
    error: str | None = None


def _worker_main(requests, responses) -> None:
    while True:
        request = requests.get()
        if request is None:
            return
        try:
            stats = render_checkpoint_with_stats(
                request.checkpoint,
                request.output_path,
                steps=request.steps,
                width=request.width,
                height=request.height,
                fps=request.fps,
            )
            responses.put(MotionEvaluationResponse(stats=stats))
        except BaseException:  # noqa: BLE001 - report worker failure to PPO
            responses.put(MotionEvaluationResponse(error=traceback.format_exc()))


class MotionEvaluationWorker:
    def __init__(self) -> None:
        context = mp.get_context("spawn")
        self._requests = context.Queue()
        self._responses = context.Queue()
        self._process = context.Process(
            target=_worker_main,
            args=(self._requests, self._responses),
            name="bello-motion-evaluator",
        )
        child_environment = {
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.05",
        }
        previous = {name: os.environ.get(name) for name in child_environment}
        try:
            os.environ.update(child_environment)
            self._process.start()
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def evaluate(
        self,
        request: MotionEvaluationRequest,
        *,
        timeout_seconds: float = 300.0,
    ) -> MotionRolloutStats:
        if not self._process.is_alive():
            raise RuntimeError(
                f"motion evaluator exited with code {self._process.exitcode}"
            )
        self._requests.put(request)
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(
                    f"motion evaluation exceeded {timeout_seconds:.0f} seconds"
                )
            try:
                response = self._responses.get(timeout=min(1.0, remaining))
                break
            except Empty as error:
                if not self._process.is_alive():
                    raise RuntimeError(
                        "motion evaluator exited while rendering with code "
                        f"{self._process.exitcode}"
                    ) from error
        if response.error is not None:
            raise RuntimeError(f"motion evaluator failed:\n{response.error}")
        if response.stats is None:
            raise RuntimeError("motion evaluator returned neither stats nor error")
        return response.stats

    def close(self) -> None:
        if self._process.is_alive():
            self._requests.put(None)
            self._process.join(timeout=10.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5.0)
        self._requests.close()
        self._responses.close()

    def __enter__(self) -> "MotionEvaluationWorker":
        return self

    def __exit__(self, exc_type, exc_value, traceback_value) -> None:
        self.close()
