from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from bello_mujoco.training import runtime
from bello_mujoco.training.ppo_config import PPOConfig


@pytest.fixture
def updates(monkeypatch):
    monkeypatch.delenv("BELLO_JOB_DIR", raising=False)
    logged = []
    monkeypatch.setattr(runtime.jax, "jit", lambda function: function)
    monkeypatch.setattr(runtime, "tqdm", lambda iterable, **kwargs: iterable)
    monkeypatch.setattr(
        runtime,
        "track_metric_snapshot",
        lambda run, metrics, **kwargs: logged.append((metrics, kwargs)),
    )
    return logged


def test_training_updates_resume_and_log_before_yield(monkeypatch, updates):
    def step(state, *args):
        return SimpleNamespace(update=state.update + 1), {"loss": jnp.asarray(0.5)}

    monkeypatch.setattr(runtime, "update_runner_state", step)
    config = PPOConfig(num_envs=2, num_steps=2, total_timesteps=12)
    initial = (SimpleNamespace(update=jnp.asarray(1)), object(), object())
    result = []
    for state, metrics, env_steps in runtime.training_updates(
        object(), config, initial, None
    ):
        assert updates[-1][1]["update"] == int(state.update)
        assert updates[-1][0] is metrics
        result.append(env_steps)
    assert result == [8, 12]


def test_invalid_update_is_neither_logged_nor_yielded(monkeypatch, updates):
    monkeypatch.setattr(
        runtime,
        "update_runner_state",
        lambda state, *args: (
            SimpleNamespace(update=state.update + 1),
            {"invalid/gradient_norm": jnp.asarray(1.0)},
        ),
    )
    initial = (SimpleNamespace(update=jnp.asarray(0)), object(), object())
    with pytest.raises(RuntimeError, match="invalid gradient_norm"):
        next(runtime.training_updates(object(), PPOConfig(), initial, None))
    assert updates == []


@pytest.mark.parametrize("update", [-1, 2])
def test_completed_or_invalid_training_range_is_rejected(updates, update):
    config = PPOConfig(num_envs=2, num_steps=2, total_timesteps=8)
    initial = (SimpleNamespace(update=jnp.asarray(update)), object(), object())
    with pytest.raises(ValueError, match="unfinished training range"):
        next(runtime.training_updates(object(), config, initial, None))


def test_stop_request_returns_latest_valid_update_for_final_save(
    monkeypatch, tmp_path, updates
):
    monkeypatch.setenv("BELLO_JOB_DIR", str(tmp_path))

    def step(state, *args):
        (tmp_path / "stop_requested").touch()
        return SimpleNamespace(update=state.update + 1), {"loss": jnp.asarray(0.5)}

    monkeypatch.setattr(runtime, "update_runner_state", step)
    config = PPOConfig(num_envs=2, num_steps=2, total_timesteps=20)
    initial = (SimpleNamespace(update=jnp.asarray(2)), object(), object())
    results = list(runtime.training_updates(object(), config, initial, None))
    assert len(results) == 1
    assert int(results[0][0].update) == 3
    assert results[0][2] == 12
    assert updates[-1][1]["update"] == 3
    schedule = SimpleNamespace(enabled=True, every_updates=1, render_final=True)
    assert not runtime.should_render_training_video(3, 5, schedule)
    assert not runtime.should_render_training_video(5, 5, schedule)
