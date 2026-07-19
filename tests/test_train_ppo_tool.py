import jax.numpy as jnp
import numpy as np
import pytest

import bello_mujoco.render as render
import bello_mujoco.tools.train_ppo as train_ppo
from bello_mujoco.checkpoint import CheckpointConfig
from bello_mujoco.policy import ActorCriticConfig
from bello_mujoco.render import VideoWriter
from bello_mujoco.train.ppo import RunnerState, TrainState
from bello_mujoco.tools.train_ppo import (
    TrainingVideoConfig,
    raise_on_invalid_training_metrics,
    render_training_video,
    should_save_periodic_checkpoint,
    should_render_training_video,
    training_update_range,
    validate_checkpoint_config,
    validate_video_config,
)


def runner_state(update: int = 0) -> RunnerState:
    return RunnerState(
        params={},
        opt_state=(),
        update=jnp.asarray(update, dtype=jnp.int32),
        env_state=None,
        obs_stack=jnp.zeros((1,), dtype=jnp.float32),
        critic_obs_stack=jnp.zeros((1,), dtype=jnp.float32),
        rng=jnp.asarray([0, 0], dtype=jnp.uint32),
        learning_rate=jnp.asarray(1.0e-3, dtype=jnp.float32),
    )


def test_training_video_schedule_skips_zero_update() -> None:
    config = TrainingVideoConfig(enabled=True, every_updates=2, render_final=True)

    assert not should_render_training_video(0, 10, config)


def test_video_writer_encodes_without_python_subprocess_fork(tmp_path) -> None:
    jnp.ones((4, 4)).block_until_ready()
    output_path = tmp_path / "rollout.mp4"
    writer = VideoWriter(output_path, width=4, height=4, fps=10)
    writer.append_data(np.zeros((4, 4, 3), dtype=np.uint8))
    writer.append_data(np.full((4, 4, 3), 255, dtype=np.uint8))
    writer.close()

    assert output_path.stat().st_size > 0


def test_rollout_video_acquisition_closes_renderer_when_camera_setup_fails(
    tmp_path, monkeypatch
) -> None:
    closed = []

    class Renderer:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def close(self) -> None:
            closed.append("renderer")

    def fail_camera(*args, **kwargs):
        raise RuntimeError("camera failed")

    monkeypatch.setattr(render.mujoco, "Renderer", Renderer)
    monkeypatch.setattr(render, "make_camera", fail_camera)

    with pytest.raises(RuntimeError, match="camera failed"):
        render.open_rollout_video_resources(
            object(),
            tmp_path / "unused.mp4",
            width=4,
            height=4,
            fps=10,
            camera_config=render.CameraConfig(),
        )

    assert closed == ["renderer"]


def test_rollout_video_cleanup_closes_renderer_when_writer_close_fails(
    tmp_path, monkeypatch
) -> None:
    closed = []

    class Renderer:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def close(self) -> None:
            closed.append("renderer")

    class Writer:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def close(self) -> None:
            raise RuntimeError("writer failed")

    monkeypatch.setattr(render.mujoco, "Renderer", Renderer)
    monkeypatch.setattr(render, "make_camera", lambda *args, **kwargs: object())
    monkeypatch.setattr(render, "VideoWriter", Writer)
    resources, _, _, _ = render.open_rollout_video_resources(
        object(),
        tmp_path / "unused.mp4",
        width=4,
        height=4,
        fps=10,
        camera_config=render.CameraConfig(),
    )

    with pytest.raises(RuntimeError, match="writer failed"):
        resources.close()

    assert closed == ["renderer"]


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


def test_training_update_range_covers_each_update() -> None:
    assert list(training_update_range(1, 3)) == [1, 2]


def test_train_update_loop_logs_updates_and_renders_on_schedule(monkeypatch) -> None:
    snapshots = []
    renders = []
    checkpoints = []
    config = train_ppo.PPOConfig(total_timesteps=18, num_envs=2, num_steps=3)

    def fake_init_runner_state(rng, env, config):
        return runner_state(), "static", "optimizer"

    def fake_update_runner_state(state, static_policy, optimizer, env, config):
        update = int(state.update) + 1
        metrics = {"loss": jnp.asarray(float(update), dtype=jnp.float32)}
        return state._replace(update=jnp.asarray(update, dtype=jnp.int32)), metrics

    def fake_track_metric_snapshot(run, metrics, *, env_steps, update, context):
        snapshots.append(
            {
                "run": run,
                "loss": float(metrics["loss"]),
                "env_steps": env_steps,
                "update": update,
                "context": context,
            }
        )

    def fake_render_training_video(
        env,
        state,
        static_policy,
        video_config,
        run,
        *,
        env_steps,
        obs_stack_size,
    ):
        renders.append((int(state.update), env_steps, obs_stack_size))

    monkeypatch.setattr(train_ppo, "init_runner_state", fake_init_runner_state)
    monkeypatch.setattr(train_ppo, "update_runner_state", fake_update_runner_state)
    monkeypatch.setattr(train_ppo, "track_metric_snapshot", fake_track_metric_snapshot)
    monkeypatch.setattr(train_ppo, "render_training_video", fake_render_training_video)

    out = train_ppo.train_update_loop(
        env="env",
        render_env="render_env",
        config=config,
        rng=jnp.asarray(0),
        video_config=TrainingVideoConfig(
            enabled=True,
            every_updates=2,
            render_initial=True,
            render_final=True,
        ),
        run="run",
        jit=False,
        checkpoint_every_updates=2,
        checkpoint_callback=lambda state: checkpoints.append(int(state.update)),
    )

    assert int(out["runner_state"].update) == 3
    assert out["metrics"]["loss"].shape == ()
    assert float(out["metrics"]["loss"]) == 3.0
    assert snapshots == [
        {
            "run": "run",
            "loss": 1.0,
            "env_steps": 6,
            "update": 1,
            "context": {"phase": "train"},
        },
        {
            "run": "run",
            "loss": 2.0,
            "env_steps": 12,
            "update": 2,
            "context": {"phase": "train"},
        },
        {
            "run": "run",
            "loss": 3.0,
            "env_steps": 18,
            "update": 3,
            "context": {"phase": "train"},
        },
    ]
    assert renders == [
        (0, 0, config.policy.obs_stack_size),
        (2, 12, config.policy.obs_stack_size),
        (3, 18, config.policy.obs_stack_size),
    ]
    assert checkpoints == [2]


def test_resumed_loop_runs_only_remaining_updates_and_skips_initial_video(
    monkeypatch,
) -> None:
    renders = []
    config = train_ppo.PPOConfig(total_timesteps=18, num_envs=2, num_steps=3)

    def unexpected_init(*args, **kwargs):
        raise AssertionError("resume must use the restored runner")

    def fake_update(state, static_policy, optimizer, env, config):
        update = int(state.update) + 1
        return state._replace(update=jnp.asarray(update)), {"loss": jnp.asarray(1.0)}

    monkeypatch.setattr(train_ppo, "init_runner_state", unexpected_init)
    monkeypatch.setattr(train_ppo, "update_runner_state", fake_update)
    monkeypatch.setattr(train_ppo, "track_metric_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(
        train_ppo,
        "render_training_video",
        lambda *a, **k: renders.append(int(a[1].update)),
    )

    out = train_ppo.train_update_loop(
        env="env",
        render_env="render_env",
        config=config,
        rng=jnp.asarray(0),
        video_config=TrainingVideoConfig(
            enabled=True,
            every_updates=10,
            render_initial=True,
            render_final=True,
        ),
        run=None,
        jit=False,
        initial_runner=(runner_state(1), "static", "optimizer"),
    )

    assert int(out["runner_state"].update) == 3
    assert renders == [3]


def test_periodic_checkpoint_schedule_excludes_final_update() -> None:
    assert should_save_periodic_checkpoint(4, 10, 4)
    assert not should_save_periodic_checkpoint(10, 10, 2)


def test_completed_runner_cannot_be_resumed() -> None:
    config = train_ppo.PPOConfig(total_timesteps=6, num_envs=2, num_steps=3)
    with pytest.raises(ValueError, match="unfinished training range"):
        train_ppo.train_update_loop(
            env="env",
            render_env="env",
            config=config,
            rng=jnp.asarray(0),
            video_config=TrainingVideoConfig(),
            run=None,
            jit=False,
            initial_runner=(runner_state(1), "static", "optimizer"),
        )


def test_checkpoint_config_rejects_invalid_periodic_settings() -> None:
    with pytest.raises(ValueError, match="null or positive"):
        validate_checkpoint_config(CheckpointConfig(enabled=True, every_updates=0))
    with pytest.raises(ValueError, match="enabled=true"):
        validate_checkpoint_config(CheckpointConfig(every_updates=10))


def test_run_training_closes_tracking_run_on_failure(monkeypatch) -> None:
    run = object()
    closed = []

    def fake_create_wandb_run(config):
        return run

    def fake_load(*args, **kwargs):
        raise RuntimeError("stop after env load")

    def fake_close_run(tracking_run, *, exit_code):
        closed.append((tracking_run, exit_code))

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

    assert closed == [(run, 1)]


def test_run_training_rejects_policy_environment_action_limit_mismatch(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        train_ppo.BelloLocomotionEnv,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("invalid action contract must fail before model loading")
        ),
    )

    with pytest.raises(ValueError, match="action_limit must equal.*action_clip"):
        train_ppo.run_training(
            train_ppo.TrainingRunConfig(
                config=train_ppo.PPOConfig(policy=ActorCriticConfig(action_limit=1.0)),
                env_config=train_ppo.BelloEnvConfig(action_clip=0.5),
            )
        )


@pytest.mark.parametrize(
    ("backend", "expected_load_mjx"),
    [("mujoco", False), ("mjx", True)],
)
def test_run_training_loads_only_the_video_backend_it_needs(
    monkeypatch, backend: str, expected_load_mjx: bool
) -> None:
    load_calls: list[dict] = []

    def fake_load(*args, **kwargs):
        load_calls.append(kwargs)
        if len(load_calls) == 2:
            raise RuntimeError("stop after render environment load")
        return object()

    monkeypatch.setattr(train_ppo.BelloLocomotionEnv, "load", fake_load)

    with pytest.raises(RuntimeError, match="stop after render environment load"):
        train_ppo.run_training(
            train_ppo.TrainingRunConfig(
                video_config=TrainingVideoConfig(
                    enabled=True,
                    simulation_backend=backend,
                )
            )
        )

    assert "load_mjx" not in load_calls[0]
    assert load_calls[1]["load_mjx"] is expected_load_mjx


@pytest.mark.parametrize(
    "name",
    [
        "loss",
        "metrics",
        "gradients",
        "optimizer_update",
        "parameters",
        "optimizer_state",
        "rollout_observation",
        "rollout_critic_observation",
        "rollout_reward",
        "rollout_action",
    ],
)
def test_named_nonfinite_contamination_identifies_source(name: str) -> None:
    with pytest.raises(RuntimeError, match=rf"nonfinite {name} at update 17"):
        raise_on_invalid_training_metrics(
            {f"nonfinite/{name}": jnp.asarray(1.0)},
            update=17,
        )


def test_zero_nonfinite_counters_do_not_stop_training() -> None:
    raise_on_invalid_training_metrics(
        {
            "nonfinite/gradients": jnp.asarray(0.0),
            "nonfinite/rollout_reward": jnp.asarray(0.0),
            "loss": jnp.asarray(0.25),
        },
        update=17,
    )


@pytest.mark.parametrize(
    "name",
    [
        "rollout_observation",
        "rollout_reward",
        "rollout_value",
        "target",
        "policy_sample",
        "gradient_norm",
    ],
)
def test_implausible_training_input_identifies_source(name: str) -> None:
    with pytest.raises(RuntimeError, match=rf"invalid {name} at update 19"):
        raise_on_invalid_training_metrics(
            {f"invalid/{name}": jnp.asarray(1.0)},
            update=19,
        )


def test_invalid_update_stops_training() -> None:
    with pytest.raises(RuntimeError, match="implausible update at update 23"):
        raise_on_invalid_training_metrics(
            {"invalid_update": jnp.asarray(1.0)},
            update=23,
        )


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

    def fake_runner_train_state(state, static_policy):
        return TrainState(policy=static_policy, opt_state=(), update=state.update)

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

    run = object()
    monkeypatch.setattr(train_ppo, "runner_train_state", fake_runner_train_state)
    monkeypatch.setattr(train_ppo, "render_policy_rollout", fake_render_policy_rollout)
    monkeypatch.setattr(train_ppo, "track_video", fake_track_video)

    output_path = render_training_video(
        env="env",
        runner_state=runner_state(update=7),
        static_policy="policy",
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
