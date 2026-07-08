import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp

from bello_mujoco.tracking import (
    log_config,
    metric_update_count,
    to_builtin,
    track_checkpoint_artifact,
)
from bello_mujoco.tracking import track_metric_history
from bello_mujoco.tracking import track_metric_snapshot


class FakeRun:
    def __init__(self) -> None:
        self.config = FakeConfig()
        self.logged = []
        self.artifacts = []
        self.finished = False

    def log(self, payload, step=None) -> None:
        self.logged.append({"payload": payload, "step": step})

    def log_artifact(self, artifact) -> None:
        self.artifacts.append(artifact)

    def finish(self) -> None:
        self.finished = True


class FakeConfig:
    def __init__(self) -> None:
        self.updates = []

    def update(self, value, allow_val_change=False) -> None:
        self.updates.append((value, allow_val_change))


@dataclass(frozen=True)
class NestedConfig:
    path: Path
    bounds: tuple[float, float]


def test_to_builtin_handles_dataclass_paths_and_tuples() -> None:
    value = to_builtin(NestedConfig(path=Path("/tmp/x"), bounds=(-1.0, 1.0)))

    assert value == {"path": "/tmp/x", "bounds": [-1.0, 1.0]}


def test_log_config_updates_wandb_config() -> None:
    run = FakeRun()

    log_config(run, {"nested": NestedConfig(path=Path("/tmp/x"), bounds=(0.0, 1.0))})

    assert run.config.updates == [
        ({"nested": {"path": "/tmp/x", "bounds": [0.0, 1.0]}}, True)
    ]


def test_track_metric_history_logs_one_payload_per_update() -> None:
    run = FakeRun()
    metrics = {
        "mean_reward": jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        "loss": jnp.asarray([0.5, 0.25], dtype=jnp.float32),
    }

    track_metric_history(
        run,
        metrics,
        samples_per_update=8,
    )

    assert metric_update_count(metrics) == 2
    assert run.logged == [
        {"payload": {"update": 1, "loss": 0.5, "mean_reward": 1.0}, "step": 8},
        {"payload": {"update": 2, "loss": 0.25, "mean_reward": 2.0}, "step": 16},
    ]


def test_track_metric_history_can_prefix_context() -> None:
    run = FakeRun()
    metrics = {
        "mean_reward": jnp.asarray([1.0], dtype=jnp.float32),
    }

    track_metric_history(
        run,
        metrics,
        samples_per_update=8,
        context={"phase": "train"},
    )

    assert run.logged == [
        {"payload": {"update": 1, "train/mean_reward": 1.0}, "step": 8}
    ]


def test_track_metric_snapshot_logs_specific_update_and_step() -> None:
    run = FakeRun()

    track_metric_snapshot(
        run,
        {
            "mean_reward": jnp.asarray(3.0, dtype=jnp.float32),
            "loss": jnp.asarray(0.125, dtype=jnp.float32),
        },
        env_steps=32,
        update=4,
        context={"phase": "train"},
    )

    assert run.logged == [
        {
            "payload": {
                "update": 4,
                "train/loss": 0.125,
                "train/mean_reward": 3.0,
            },
            "step": 32,
        }
    ]


def test_track_checkpoint_artifact_logs_directory(monkeypatch, tmp_path) -> None:
    artifacts = []

    class FakeArtifact:
        def __init__(self, *, name, type):
            self.name = name
            self.type = type
            self.dirs = []
            artifacts.append(self)

        def add_dir(self, path) -> None:
            self.dirs.append(path)

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(Artifact=FakeArtifact),
    )
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    run = FakeRun()

    track_checkpoint_artifact(
        run,
        checkpoint_dir,
        name="checkpoint-name",
        artifact_type="model",
    )

    assert run.artifacts == artifacts
    assert artifacts[0].name == "checkpoint-name"
    assert artifacts[0].type == "model"
    assert artifacts[0].dirs == [str(checkpoint_dir)]
