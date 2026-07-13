import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp

from bello_mujoco.tracking import (
    WandbConfig,
    create_wandb_run,
    log_config,
    to_builtin,
    track_checkpoint_artifact,
    track_video,
)
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


def test_create_wandb_run_uses_thread_start_method(monkeypatch, tmp_path) -> None:
    calls = {}

    class FakeWandb:
        @staticmethod
        def Settings(**kwargs):
            calls["settings"] = kwargs
            return SimpleNamespace(**kwargs)

        @staticmethod
        def init(**kwargs):
            calls["init"] = kwargs
            return "run"

    monkeypatch.setitem(sys.modules, "wandb", FakeWandb)

    run = create_wandb_run(
        WandbConfig(enabled=True, project="project", dir=str(tmp_path / "wandb"))
    )

    assert run == "run"
    assert calls["settings"] == {"start_method": "thread"}
    assert calls["init"]["settings"].start_method == "thread"


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


def test_track_video_logs_encoded_file_without_fps(monkeypatch, tmp_path) -> None:
    videos = []

    class FakeVideo:
        def __init__(self, path, **kwargs):
            self.path = path
            self.kwargs = kwargs
            videos.append(self)

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(Video=FakeVideo),
    )
    video_path = tmp_path / "rollout.mp4"
    video_path.write_bytes(b"not a real mp4")
    run = FakeRun()

    track_video(
        run,
        video_path,
        name="train/rollout",
        step=128,
        fps=50,
        caption="rollout",
    )

    assert run.logged == [
        {"payload": {"train/rollout": videos[0]}, "step": 128}
    ]
    assert videos[0].path == str(video_path)
    assert videos[0].kwargs == {"format": "mp4", "caption": "rollout"}
