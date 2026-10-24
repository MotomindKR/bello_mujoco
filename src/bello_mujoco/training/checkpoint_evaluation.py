"""Submit immutable checkpoints to an isolated CPU process without blocking rollout."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from bello_mujoco.training.tracking import to_builtin
from bello_mujoco.sonic.corpus import file_sha256


def write_json_atomic(path, document):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class CheckpointEvaluator:
    def __init__(self, root, run_config, *, parent_run_url=None):
        self.root = Path(root) / "checkpoint-evaluation"
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "jobs").mkdir()
        self.timeout = run_config.checkpoint_evaluation.finish_timeout_seconds
        self.submitted = {}
        self.closed = False
        self.announced = False
        write_json_atomic(
            self.root / "request.json",
            {
                "dataset": str(run_config.dataset.directory.resolve()),
                "evaluation": to_builtin(run_config.checkpoint_evaluation),
                "wandb": to_builtin(run_config.wandb_config),
                "parent_run_url": parent_run_url,
                "run_name": self.root.parent.name,
            },
        )
        environment = {
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        with (self.root / "worker.log").open("w") as log:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "bello_mujoco.evaluation.checkpoint_worker",
                    str(self.root),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )

    def check(self):
        code = self.process.poll()
        if code is not None and not self.closed:
            error = self.root / "error.json"
            detail = error.read_text() if error.exists() else f"exit code {code}"
            raise RuntimeError(
                f"checkpoint evaluation worker stopped: {detail}; see {self.root / 'worker.log'}"
            )
        worker_info = self.root / "worker.json"
        if not self.announced and worker_info.exists():
            print(
                json.dumps(
                    {"checkpoint_evaluation": json.loads(worker_info.read_text())}
                ),
                flush=True,
            )
            self.announced = True

    def submit(self, checkpoint):
        if self.closed:
            raise RuntimeError("checkpoint evaluation is already closed")
        self.check()
        checkpoint = Path(checkpoint).resolve()
        metadata = json.loads((checkpoint / "metadata.json").read_text())
        update = metadata["update"]
        identity = {
            key: metadata[key]
            for key in (
                "git_commit",
                "dataset_manifest_sha256",
                "actor_sha256",
                "dynamics_sha256",
            )
        }
        identity["metadata_sha256"] = file_sha256(checkpoint / "metadata.json")
        if update in self.submitted:
            if self.submitted[update] != identity:
                raise RuntimeError(
                    "different checkpoint contents share the same update"
                )
            return
        if self.submitted and update < max(self.submitted):
            raise ValueError(
                "checkpoint evaluation submissions must increase in update order"
            )
        write_json_atomic(
            self.root / "jobs" / f"{update:09d}.json",
            {
                "checkpoint": str(checkpoint),
                "update": update,
                "identity": identity,
            },
        )
        self.submitted[update] = identity

    def _terminate(self):
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)

    def close(self, *, wait):
        if self.closed:
            return
        if not wait:
            self._terminate()
            self.closed = True
            return
        self.check()
        (self.root / "finish").touch()
        try:
            code = self.process.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired as error:
            self._terminate()
            self.closed = True
            raise RuntimeError(
                "checkpoint evaluation exceeded its finish timeout; worker terminated"
            ) from error
        self.closed = True
        if code != 0:
            raise RuntimeError(
                f"checkpoint evaluation failed with exit code {code}; see {self.root / 'worker.log'}"
            )
        status_path = self.root / "status.json"
        if not status_path.exists():
            raise RuntimeError(
                "checkpoint evaluation did not publish completion status"
            )
        status = json.loads(status_path.read_text())
        expected = {f"{update:09d}.json" for update in self.submitted}
        if not status["finished"] or set(status["completed"]) != expected:
            raise RuntimeError(
                "checkpoint evaluation did not complete every submitted update"
            )
