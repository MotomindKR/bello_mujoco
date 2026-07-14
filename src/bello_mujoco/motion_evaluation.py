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


def _worker_main(request: MotionEvaluationRequest, responses) -> None:
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
        self._context = mp.get_context("spawn")
        self._process = None
        self._responses = None

    def _start(self, request: MotionEvaluationRequest) -> None:
        if self._process is not None:
            raise RuntimeError("motion evaluator is already rendering")
        responses = self._context.Queue(maxsize=1)
        process = self._context.Process(
            target=_worker_main,
            args=(request, responses),
            name="bello-motion-evaluator",
        )
        child_environment = {
            "JAX_PLATFORMS": "cpu",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
        previous = {name: os.environ.get(name) for name in child_environment}
        try:
            os.environ.update(child_environment)
            process.start()
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        self._process = process
        self._responses = responses

    def _stop(self) -> None:
        process = self._process
        responses = self._responses
        if process is not None:
            process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
            process.close()
        if responses is not None:
            responses.close()
            responses.join_thread()
        self._process = None
        self._responses = None

    def evaluate(
        self,
        request: MotionEvaluationRequest,
        *,
        timeout_seconds: float = 300.0,
    ) -> MotionRolloutStats:
        self._start(request)
        try:
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
        finally:
            # JAX and Warp retain compiled executables and allocator pools. A fresh
            # process for every video guarantees those resources are reclaimed.
            self._stop()

    def close(self) -> None:
        self._stop()

    def __enter__(self) -> "MotionEvaluationWorker":
        return self

    def __exit__(self, exc_type, exc_value, traceback_value) -> None:
        self.close()
