from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

import bello_mujoco.tools.train_motion_ppo as train_motion
from bello_mujoco.tools.train_motion_ppo import (
    MotionEvalVideoConfig,
    render_motion_training_video,
    validate_motion_video_config,
)


def test_motion_video_config_is_noop_when_disabled() -> None:
    validate_motion_video_config(
        MotionEvalVideoConfig(enabled=False, every_updates=0, steps=0)
    )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (MotionEvalVideoConfig(every_updates=0), "every_updates"),
        (MotionEvalVideoConfig(steps=0), "steps"),
        (MotionEvalVideoConfig(width=0), "width and height"),
        (MotionEvalVideoConfig(fps=0), "fps"),
        (MotionEvalVideoConfig(filename_prefix=""), "filename_prefix"),
        (MotionEvalVideoConfig(wandb_name=""), "wandb_name"),
    ],
)
def test_motion_video_config_rejects_invalid_values(config, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_motion_video_config(config)


def test_motion_training_video_renders_and_uploads_to_wandb(
    monkeypatch, tmp_path: Path
) -> None:
    calls = {}

    monkeypatch.setattr(
        train_motion,
        "runner_train_state",
        lambda runner_state, static_policy: SimpleNamespace(policy="policy"),
    )

    def fake_render(env, policy, output_path, **kwargs):
        calls["render"] = (env, policy, output_path, kwargs)
        return {
            "rendered_steps": 17,
            "episode_return": 4.25,
            "terminal": False,
            "timeout": True,
        }

    def fake_track_video(run, path, **kwargs):
        calls["track"] = (run, path, kwargs)

    monkeypatch.setattr(train_motion, "render_motion_policy", fake_render)
    monkeypatch.setattr(train_motion, "track_video", fake_track_video)
    run = SimpleNamespace()
    config = MotionEvalVideoConfig(
        output_dir=tmp_path,
        filename_prefix="wholebody",
        steps=20,
        width=320,
        height=240,
        fps=25,
        wandb_name="eval/bello",
    )

    output_path = render_motion_training_video(
        "env",
        SimpleNamespace(update=jnp.asarray(7)),
        "static",
        config,
        run,
        env_steps=2048,
        obs_stack_size=4,
    )

    assert output_path == tmp_path / "wholebody_000007.mp4"
    assert calls["render"] == (
        "env",
        "policy",
        output_path,
        {
            "steps": 20,
            "width": 320,
            "height": 240,
            "fps": 25,
            "obs_stack_size": 4,
        },
    )
    assert calls["track"][0:2] == (run, output_path)
    assert calls["track"][2]["name"] == "eval/bello"
    assert calls["track"][2]["step"] == 2048
    assert "rendered_steps=17" in calls["track"][2]["caption"]
    assert "timeout=1" in calls["track"][2]["caption"]
