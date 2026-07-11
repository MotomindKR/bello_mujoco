import pytest
import jax.numpy as jnp
from types import SimpleNamespace

import bello_mujoco.tools.train_ppo as train_ppo
from bello_mujoco.tools.train_ppo import (
    TrainingVideoConfig,
    render_training_video,
    should_render_training_video,
    training_update_range,
    validate_video_config,
)


def test_training_video_schedule_skips_zero_update() -> None:
    config = TrainingVideoConfig(enabled=True, every_updates=2, render_final=True)

    assert not should_render_training_video(0, 10, config)


def test_training_video_schedule_skips_disabled_config() -> None:
    config = TrainingVideoConfig(enabled=False, every_updates=1, render_final=True)

    assert not should_render_training_video(1, 1, config)


def test_training_video_schedule_uses_interval_and_final() -> None:
    config = TrainingVideoConfig(enabled=True, every_updates=3, render_final=True)

    assert not should_render_training_video(1, 10, config)
    assert should_render_training_video(3, 10, config)
    assert should_render_training_video(10, 10, config)


def test_training_video_schedule_can_disable_final() -> None:
    config = TrainingVideoConfig(enabled=True, every_updates=4, render_final=False)

    assert not should_render_training_video(10, 10, config)


def test_training_update_range_wraps_tqdm(monkeypatch) -> None:
    calls = {}

    def fake_tqdm(iterable, **kwargs):
        calls["iterable"] = iterable
        calls["kwargs"] = kwargs
        return iterable

    monkeypatch.setattr(train_ppo, "tqdm", fake_tqdm)

    assert list(training_update_range(3)) == [0, 1, 2]
    assert calls["kwargs"]["desc"] == "PPO updates"
    assert calls["kwargs"]["unit"] == "update"
    assert calls["kwargs"]["dynamic_ncols"] is True


def test_run_training_initializes_wandb_before_env_load(monkeypatch) -> None:
    events = []

    def fake_create_wandb_run(config):
        events.append("wandb")
        return "run"

    def fake_load(*args, **kwargs):
        events.append("env")
        raise RuntimeError("stop after env load")

    def fake_close_run(run):
        events.append(("close", run))

    monkeypatch.setattr(train_ppo, "create_wandb_run", fake_create_wandb_run)
    monkeypatch.setattr(train_ppo.BelloLocomotionEnv, "load", fake_load)
    monkeypatch.setattr(train_ppo, "close_run", fake_close_run)

    with pytest.raises(RuntimeError, match="stop after env load"):
        train_ppo.run_training(
            train_ppo.TrainingRunConfig(
                dry_run=False,
                wandb_config=train_ppo.WandbConfig(enabled=True),
            )
        )

    assert events == ["wandb", "env", ("close", "run")]


def test_validate_video_config_is_noop_when_disabled() -> None:
    validate_video_config(TrainingVideoConfig(enabled=False, every_updates=0))


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            TrainingVideoConfig(enabled=True, every_updates=0),
            "every_updates must be positive",
        ),
        (TrainingVideoConfig(enabled=True, steps=0), "steps must be positive"),
        (TrainingVideoConfig(enabled=True, width=0), "width and .*height"),
        (TrainingVideoConfig(enabled=True, fps=0), "fps must be positive"),
        (
            TrainingVideoConfig(enabled=True, action_seed=-1),
            "action_seed must be nonnegative",
        ),
        (
            TrainingVideoConfig(enabled=True, simulation_backend="bad"),
            "simulation_backend must be",
        ),
    ],
)
def test_validate_video_config_rejects_invalid_enabled_config(
    config: TrainingVideoConfig, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_video_config(config)


def test_render_training_video_writes_expected_path_and_tracks(monkeypatch, tmp_path):
    calls = {}

    def fake_runner_train_state(runner_state, static_policy):
        return SimpleNamespace(policy="policy")

    def fake_render_policy_rollout(*args, **kwargs):
        calls["render"] = (args, kwargs)
        return {
            "steps": 10,
            "completed_episodes": 1,
            "first_done_step": 4,
            "first_done_seconds": 0.08,
            "done/terminal": 1,
            "done/timeout": 0,
        }

    def fake_track_video(*args, **kwargs):
        calls["track"] = (args, kwargs)

    run = SimpleNamespace()
    monkeypatch.setattr(train_ppo, "runner_train_state", fake_runner_train_state)
    monkeypatch.setattr(train_ppo, "render_policy_rollout", fake_render_policy_rollout)
    monkeypatch.setattr(train_ppo, "track_video", fake_track_video)

    output_path = render_training_video(
        env="env",
        runner_state=SimpleNamespace(update=jnp.asarray(7)),
        static_policy="static",
        config=TrainingVideoConfig(
            enabled=True,
            output_dir=tmp_path,
            filename_prefix="eval",
            fps=12,
            wandb_name="video",
            stochastic_actions=True,
            action_seed=123,
        ),
        run=run,
        env_steps=128,
    )

    assert output_path == tmp_path / "eval_000007.mp4"
    assert calls["render"][1]["output_path"] == output_path
    assert calls["render"][1]["simulation_backend"] == "mujoco"
    assert calls["render"][1]["stochastic_actions"]
    assert calls["render"][1]["action_seed"] == 123
    assert calls["track"][0] == (run, output_path)
    assert calls["track"][1]["step"] == 128
    assert "first_done_step=4" in calls["track"][1]["caption"]
    assert "metric_log" not in calls
