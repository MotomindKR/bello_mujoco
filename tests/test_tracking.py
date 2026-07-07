from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp

from bello_mujoco.tracking import log_config, metric_update_count, to_builtin
from bello_mujoco.tracking import track_metric_history


class FakeRun:
    def __init__(self) -> None:
        self.params = {}
        self.tracked = []

    def set(self, key, value, strict=True) -> None:
        self.params[key] = (value, strict)

    def track(self, value, name=None, step=None, context=None) -> None:
        self.tracked.append(
            {"value": value, "name": name, "step": step, "context": context}
        )


@dataclass(frozen=True)
class NestedConfig:
    path: Path
    bounds: tuple[float, float]


def test_to_builtin_handles_dataclass_paths_and_tuples() -> None:
    value = to_builtin(NestedConfig(path=Path("/tmp/x"), bounds=(-1.0, 1.0)))

    assert value == {"path": "/tmp/x", "bounds": [-1.0, 1.0]}


def test_log_config_sets_top_level_blocks() -> None:
    run = FakeRun()

    log_config(run, {"nested": NestedConfig(path=Path("/tmp/x"), bounds=(0.0, 1.0))})

    assert run.params["nested"] == (
        {"path": "/tmp/x", "bounds": [0.0, 1.0]},
        False,
    )


def test_track_metric_history_uses_env_steps() -> None:
    run = FakeRun()
    metrics = {
        "mean_reward": jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        "loss": jnp.asarray([0.5, 0.25], dtype=jnp.float32),
    }

    track_metric_history(
        run,
        metrics,
        samples_per_update=8,
        context={"phase": "train"},
    )

    assert metric_update_count(metrics) == 2
    assert [entry["step"] for entry in run.tracked] == [8, 8, 8, 16, 16, 16]
    assert run.tracked[0] == {
        "value": 1,
        "name": "update",
        "step": 8,
        "context": {"phase": "train"},
    }
    first_step_metrics = {
        entry["name"]: entry["value"]
        for entry in run.tracked
        if entry["step"] == 8 and entry["name"] != "update"
    }
    assert first_step_metrics == {"mean_reward": 1.0, "loss": 0.5}
