from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from bello_mujoco.evaluation import motion as evaluation
from bello_mujoco.training import motion as training
from bello_mujoco.training.ppo_config import PPOConfig


@pytest.mark.parametrize("steps", [1, 3000])
def test_motion_evaluation_dispatches_checkpoint_to_software_worker(monkeypatch, steps):
    calls = []
    monkeypatch.setattr(
        evaluation,
        "run_software_evaluation",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {"rendered_steps": 1},
    )
    request = evaluation.MotionEvaluationRequest(
        checkpoint=Path("checkpoint"),
        output_path=Path("video.mp4"),
        steps=steps,
        width=32,
        height=24,
        fps=20,
    )
    assert evaluation.evaluate_motion_video(request) == {"rendered_steps": 1}
    args, kwargs = calls[0]
    assert args == (
        evaluation.render_checkpoint_with_stats,
        request.checkpoint,
        request.output_path,
    )
    assert kwargs == {
        "steps": steps,
        "width": 32,
        "height": 24,
        "fps": 20,
        "timeout_seconds": 300.0 + steps,
    }


def test_motion_video_failure_stops_training(monkeypatch, tmp_path):
    monkeypatch.setattr(training, "save_motion_checkpoint", lambda *args: None)

    def fail(*args, **kwargs):
        raise RuntimeError("software renderer unavailable")

    monkeypatch.setattr(training, "evaluate_motion_video", fail)
    with pytest.raises(RuntimeError, match="software renderer unavailable"):
        training.motion_train_update_loop(
            SimpleNamespace(motion=object()),
            PPOConfig(),
            object(),
            training.MotionEvalVideoConfig(enabled=True, render_initial=True),
            None,
            checkpoint_root=tmp_path,
            checkpoint_context=(object(), "commit", object()),
            initial_runner=(SimpleNamespace(update=jnp.asarray(0)), object(), object()),
        )
