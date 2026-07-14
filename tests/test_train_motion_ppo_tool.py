from pathlib import Path
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import bello_mujoco.tools.train_motion_ppo as train_motion
from bello_mujoco.checkpoint import (
    CheckpointConfig,
    CheckpointMetadata,
    ObservationNormMetadata,
    ValueNormMetadata,
)
from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.envs.locomotion import BelloEnvConfig, DoneConfig
from bello_mujoco.envs.rewards import CommandConfig, RewardConfig
from bello_mujoco.sim import WarpConfig
from bello_mujoco.tools.train_motion_ppo import (
    MotionEvalVideoConfig,
    render_motion_training_video,
    resume_runner_state,
    should_save_motion_checkpoint,
    validate_motion_checkpoint_config,
    validate_motion_video_config,
    try_render_motion_training_video,
)
from bello_mujoco.train.ppo import (
    PPOConfig,
    RunnerState,
    ValueNormState,
    init_observation_norm_state,
    init_train_state,
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
            "completion_fraction": 1.0,
            "root_position_rms_m": 0.04,
            "root_orientation_rms_deg": 3.0,
            "joint_position_rms_rad": 0.12,
        }

    def fake_track_video(run, path, **kwargs):
        calls["track"] = (run, path, kwargs)

    def fake_track_metrics(run, metrics, **kwargs):
        calls["metrics"] = (run, metrics, kwargs)

    monkeypatch.setattr(train_motion, "render_motion_policy", fake_render)
    monkeypatch.setattr(train_motion, "track_video", fake_track_video)
    monkeypatch.setattr(train_motion, "track_metric_snapshot", fake_track_metrics)
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
    assert calls["metrics"][1]["completion_fraction"] == 1.0
    assert calls["metrics"][1]["joint_position_rms_rad"] == pytest.approx(0.12)
    assert "rendered_steps=17" in calls["track"][2]["caption"]
    assert "root_orientation_rms=3.0deg" in calls["track"][2]["caption"]
    assert "timeout=1" in calls["track"][2]["caption"]


def test_motion_checkpoint_schedule_is_periodic() -> None:
    config = CheckpointConfig(enabled=True, every_updates=100)

    assert not should_save_motion_checkpoint(0, config)
    assert not should_save_motion_checkpoint(99, config)
    assert should_save_motion_checkpoint(100, config)
    assert not should_save_motion_checkpoint(101, config)


def test_motion_checkpoint_schedule_can_be_disabled() -> None:
    assert not should_save_motion_checkpoint(
        100, CheckpointConfig(enabled=False, every_updates=100)
    )
    assert not should_save_motion_checkpoint(
        100, CheckpointConfig(enabled=True, every_updates=None)
    )


def test_motion_checkpoint_config_rejects_invalid_interval() -> None:
    with pytest.raises(ValueError, match="every_updates"):
        validate_motion_checkpoint_config(
            CheckpointConfig(enabled=True, every_updates=0)
        )


def test_motion_video_failure_is_nonfatal(monkeypatch) -> None:
    monkeypatch.setattr(
        train_motion,
        "render_motion_training_video",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("encoder")),
    )
    runner_state = SimpleNamespace(update=jnp.asarray(12))

    with pytest.warns(RuntimeWarning, match="update 12: encoder"):
        output = try_render_motion_training_video("env", runner_state, "policy")

    assert output is None


def test_resume_runner_state_restores_optimizer_and_normalization() -> None:
    config = PPOConfig(
        total_timesteps=8,
        num_envs=2,
        num_steps=2,
        num_minibatches=1,
    )
    initial, _ = init_train_state(jax.random.PRNGKey(0), 4, 2, config)
    restored, _ = init_train_state(jax.random.PRNGKey(1), 4, 2, config)
    initial_params, _ = eqx.partition(initial.policy, eqx.is_array)
    runner = RunnerState(
        params=initial_params,
        opt_state=initial.opt_state,
        update=jnp.asarray(0),
        env_state=None,
        obs_stack=jnp.zeros(1),
        critic_obs_stack=jnp.zeros(1),
        command_ranges=jnp.zeros(1),
        rng=jnp.asarray([1, 2], dtype=jnp.uint32),
        learning_rate=jnp.asarray(0.001),
        value_norm_state=ValueNormState(
            jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray(0.0)
        ),
        observation_norm_state=init_observation_norm_state(4),
        critic_observation_norm_state=init_observation_norm_state(4),
    )
    metadata = CheckpointMetadata(
        ppo=config,
        asset=AssetPrepConfig(),
        reward=RewardConfig(),
        command=CommandConfig(),
        done=DoneConfig(),
        env=BelloEnvConfig(),
        warp=WarpConfig(),
        obs_size=4,
        action_size=2,
        value_norm=ValueNormMetadata(mean=3.0, var=4.0, count=5.0),
        observation_norm=ObservationNormMetadata(
            mean=(1.0, 2.0, 3.0, 4.0),
            var=(2.0, 3.0, 4.0, 5.0),
            count=6.0,
        ),
        critic_observation_norm=ObservationNormMetadata(
            mean=(4.0, 3.0, 2.0, 1.0),
            var=(5.0, 4.0, 3.0, 2.0),
            count=7.0,
        ),
        rng=(7, 8),
        learning_rate=0.0002,
    )
    restored = restored.replace(update=jnp.asarray(9))

    resumed = resume_runner_state(runner, restored, metadata)

    assert int(resumed.update) == 9
    assert jnp.array_equal(resumed.rng, jnp.asarray([7, 8], dtype=jnp.uint32))
    assert float(resumed.learning_rate) == pytest.approx(0.0002)
    assert float(resumed.value_norm_state.mean) == pytest.approx(3.0)
    assert float(resumed.value_norm_state.var) == pytest.approx(4.0)
    assert resumed.observation_norm_state.mean.tolist() == pytest.approx(
        [1.0, 2.0, 3.0, 4.0]
    )
    assert float(resumed.observation_norm_state.count) == pytest.approx(6.0)
    assert resumed.critic_observation_norm_state.var.tolist() == pytest.approx(
        [5.0, 4.0, 3.0, 2.0]
    )
    assert float(resumed.value_norm_state.count) == pytest.approx(5.0)
