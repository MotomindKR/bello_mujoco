from dataclasses import replace
import json
from pathlib import Path
import subprocess

import pytest

from bello_mujoco.configuration import supermimic_training_profile
from bello_mujoco.evaluation import checkpoint_worker
from bello_mujoco.evaluation.checkpoint_config import CheckpointEvaluationConfig
from bello_mujoco.sonic.corpus import file_sha256
from bello_mujoco.supermimic.checkpoint import resume_contract
from bello_mujoco.training import checkpoint_evaluation as host


@pytest.fixture
def evaluator(tmp_path, monkeypatch):
    class Process:
        pid = 123456
        returncode = None
        timeout = False

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            if self.timeout and self.returncode is None:
                raise subprocess.TimeoutExpired("worker", timeout)
            if self.returncode is None:
                self.returncode = 0
                host.write_json_atomic(
                    self.root / "status.json",
                    {
                        "finished": True,
                        "completed": [
                            p.name for p in (self.root / "jobs").glob("*.json")
                        ],
                    },
                )
            return self.returncode

    process = Process()

    def launch(command, **kwargs):
        assert kwargs["env"]["JAX_PLATFORMS"] == "cpu"
        assert kwargs["start_new_session"]
        process.root = Path(command[-1])
        return process

    monkeypatch.setattr(host.subprocess, "Popen", launch)
    monkeypatch.setattr(
        host.os, "killpg", lambda pid, sig: setattr(process, "returncode", -int(sig))
    )
    run = supermimic_training_profile(tmp_path / "corpus")
    result = host.CheckpointEvaluator(tmp_path / "run", run)
    yield result, process
    result.close(wait=False)


def checkpoint(tmp_path, name, update, actor="actor"):
    folder = tmp_path / name
    folder.mkdir()
    metadata = {
        "update": update,
        "git_commit": "source",
        "dataset_manifest_sha256": "dataset",
        "actor_sha256": actor,
        "dynamics_sha256": "dynamics",
    }
    (folder / "metadata.json").write_text(json.dumps(metadata))
    return folder, metadata


def test_queue_deduplicates_identical_final_and_checks_all_completed(
    evaluator, tmp_path
):
    worker, _ = evaluator
    first, _ = checkpoint(tmp_path, "update", 2)
    final, _ = checkpoint(tmp_path, "final", 2)
    worker.submit(first)
    worker.submit(final)
    assert len(list((worker.root / "jobs").glob("*.json"))) == 1
    changed, _ = checkpoint(tmp_path, "different", 2, actor="changed")
    with pytest.raises(RuntimeError, match="same update"):
        worker.submit(changed)
    old, _ = checkpoint(tmp_path, "old", 1)
    with pytest.raises(ValueError, match="increase"):
        worker.submit(old)
    worker.close(wait=True)
    assert json.loads((worker.root / "status.json").read_text())["finished"]
    with pytest.raises(RuntimeError, match="already closed"):
        worker.submit(final)


def test_timeout_terminates_worker_and_failed_worker_is_not_silent(evaluator):
    worker, process = evaluator
    process.timeout = True
    with pytest.raises(RuntimeError, match="finish timeout"):
        worker.close(wait=True)
    assert process.returncode is not None


def test_worker_failure_propagates_at_training_poll(evaluator):
    worker, process = evaluator
    process.returncode = 1
    (worker.root / "error.json").write_text('{"traceback":"identity changed"}')
    with pytest.raises(RuntimeError, match="identity changed"):
        worker.check()


def test_finish_drains_every_queued_checkpoint_in_order(tmp_path, monkeypatch):
    (tmp_path / "jobs").mkdir()
    for update in (20, 10):
        host.write_json_atomic(
            tmp_path / "jobs" / f"{update:09d}.json", {"update": update}
        )
    (tmp_path / "finish").touch()
    evaluated = []
    monkeypatch.setattr(
        checkpoint_worker,
        "evaluate_job",
        lambda job, *a: evaluated.append(job["update"]),
    )
    checkpoint_worker.run_queue(tmp_path, {}, None)
    assert evaluated == [10, 20]
    assert json.loads((tmp_path / "status.json").read_text())["finished"]


def test_job_selects_fixed_training_windows_and_rejects_metadata_mutation(
    tmp_path, monkeypatch
):
    from bello_mujoco.supermimic import checkpoint as snapshots
    from bello_mujoco.evaluation import world_model_audit

    path, metadata = checkpoint(tmp_path, "checkpoint", 10)
    dataset = tmp_path / "corpus"
    dataset.mkdir()
    (dataset / "sonic-corpus.json").write_text(
        json.dumps(
            {
                "clips": [
                    {"file": "a.npz", "split": "train"},
                    {"file": "b.npz", "split": "train"},
                    {"file": "held.npz", "split": "evaluation"},
                ]
            }
        )
    )
    identity = {k: v for k, v in metadata.items() if k != "update"}
    identity["metadata_sha256"] = file_sha256(path / "metadata.json")
    job = {"checkpoint": str(path), "update": 10, "identity": identity}
    request = {
        "dataset": str(dataset),
        "evaluation": {"enabled": True, "windows": [{"clip": "b", "start_frame": 4}]},
        "run_name": "test",
    }
    monkeypatch.setattr(snapshots, "checkpoint_metadata", lambda p: metadata)
    calls = []

    def evaluate(config):
        calls.append(config.windows)
        config.output_path.parent.mkdir(parents=True)
        return {
            "windows": {
                "b:4": {
                    "horizons": {},
                    "tracking": {
                        "terminal": False,
                        "normalized_return": 0.5,
                    },
                }
            }
        }

    monkeypatch.setattr(world_model_audit, "audit_world_model", evaluate)
    checkpoint_worker.evaluate_job(job, request, tmp_path / "worker", None)
    assert [[(w.clip, w.start_frame) for w in windows] for windows in calls] == [
        [("b", 4)]
    ]
    summary = json.loads(
        (tmp_path / "worker/results/000000010/summary.json").read_text()
    )
    assert summary["tracking"]["mean_normalized_return"] == 0.5
    request["evaluation"]["windows"] = [{"clip": "held"}]
    with pytest.raises(ValueError, match="training clips"):
        checkpoint_worker.evaluate_job(job, request, tmp_path / "worker", None)
    (path / "metadata.json").write_text("changed")
    with pytest.raises(ValueError, match="identity changed"):
        checkpoint_worker.evaluate_job(job, request, tmp_path / "worker", None)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enabled": 1},
        {"enabled": True},
        {"cpu_cores": 0},
        {"cpu_cores": 5},
        {"finish_timeout_seconds": float("nan")},
    ],
)
def test_checkpoint_evaluation_settings_validate_resource_bounds(kwargs):
    with pytest.raises(ValueError):
        CheckpointEvaluationConfig(**kwargs)


def test_evaluation_is_operational_and_requires_saved_checkpoints():
    run = supermimic_training_profile(
        Path("corpus"), name="bello_mini_supermimic_lafan"
    )
    assert run.checkpoint_evaluation.enabled
    assert resume_contract(run) == resume_contract(
        replace(
            run, checkpoint_evaluation=replace(run.checkpoint_evaluation, cpu_cores=2)
        )
    )
    with pytest.raises(ValueError, match="enabled checkpoint saving"):
        replace(run, checkpoint_config=replace(run.checkpoint_config, enabled=False))


def test_real_cpu_worker_starts_and_finishes_without_initializing_a_gpu(tmp_path):
    run = supermimic_training_profile(tmp_path / "corpus")
    run = replace(
        run,
        checkpoint_evaluation=replace(
            run.checkpoint_evaluation, finish_timeout_seconds=20
        ),
    )
    worker = host.CheckpointEvaluator(tmp_path / "real-run", run)
    try:
        worker.close(wait=True)
        assert worker.process.returncode == 0
        assert json.loads((worker.root / "status.json").read_text()) == {
            "completed": [],
            "finished": True,
        }
        assert (worker.root / "worker.json").is_file()
    finally:
        worker.close(wait=False)
