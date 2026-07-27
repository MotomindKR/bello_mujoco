from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import bello_mujoco.evaluation.motion as eval_motion
from bello_mujoco.envs.motion_tracking import MotionTrackingEnvConfig
from bello_mujoco.policy import ActorCriticConfig
from bello_mujoco.training.ppo import ObservationNormState, PPOConfig


def rollout_stats() -> eval_motion.MotionRolloutStats:
    return {
        "rendered_steps": 2,
        "episode_return": 3.0,
        "terminal": False,
        "timeout": True,
        "completion_fraction": 0.5,
        "root_position_rms_m": 0.0,
        "root_orientation_rms_deg": 0.0,
        "joint_position_rms_rad": 0.0,
    }


def test_render_checkpoint_rejects_non_motion_task(monkeypatch, tmp_path: Path) -> None:
    metadata = SimpleNamespace(
        task="locomotion",
        motion_env=None,
        motion=None,
    )
    monkeypatch.setattr(
        eval_motion,
        "restore_evaluation_checkpoint",
        lambda path: (SimpleNamespace(policy=object()), metadata, object()),
    )
    monkeypatch.setattr(
        eval_motion.BelloMotionTrackingEnv,
        "load_native",
        staticmethod(lambda *args, **kwargs: pytest.fail("must not load motion env")),
    )

    with pytest.raises(ValueError, match="not motion_tracking"):
        eval_motion.render_checkpoint_with_stats(
            tmp_path / "checkpoint", tmp_path / "rollout.mp4"
        )


def test_relocated_motion_is_loaded_validated_and_rendered(
    monkeypatch, tmp_path: Path
) -> None:
    original = tmp_path / "original.npz"
    relocated = tmp_path / "relocated.npz"
    motion_contract = SimpleNamespace(path=original)
    ppo = PPOConfig(policy=ActorCriticConfig(obs_stack_size=3))
    normalization = SimpleNamespace(observation=object())
    metadata = SimpleNamespace(
        task="motion_tracking",
        motion_env=MotionTrackingEnvConfig(motion_path=original),
        motion=motion_contract,
        asset=object(),
        ppo=ppo,
    )
    policy = object()
    clip = object()
    env = SimpleNamespace(motion=clip, max_episode_steps=17)
    calls = {}

    monkeypatch.setattr(
        eval_motion,
        "restore_evaluation_checkpoint",
        lambda path: (SimpleNamespace(policy=policy), metadata, normalization),
    )

    def load_native(asset, env_config, *, use_viewer_model):
        calls["load"] = asset, env_config, use_viewer_model
        return env

    monkeypatch.setattr(
        eval_motion.BelloMotionTrackingEnv,
        "load_native",
        staticmethod(load_native),
    )

    def validate(path, loaded_clip, expected):
        calls["validate"] = path, loaded_clip, expected

    monkeypatch.setattr(eval_motion, "validate_motion_clip_contract", validate)

    def render(loaded_env, loaded_policy, output_path, **kwargs):
        calls["render"] = loaded_env, loaded_policy, output_path, kwargs
        return rollout_stats()

    monkeypatch.setattr(eval_motion, "render_motion_policy", render)

    output = tmp_path / "rollout.mp4"
    result = eval_motion.render_checkpoint_with_stats(
        tmp_path / "checkpoint",
        output,
        motion_path=relocated,
        width=80,
        height=60,
        fps=25,
    )

    _, loaded_config, use_viewer_model = calls["load"]
    assert loaded_config.motion_path == relocated
    assert use_viewer_model is True
    assert calls["validate"] == (relocated, clip, motion_contract)
    loaded_env, loaded_policy, loaded_output, render_kwargs = calls["render"]
    assert loaded_env is env
    assert loaded_policy is policy
    assert loaded_output == output
    assert render_kwargs["steps"] == env.max_episode_steps
    assert render_kwargs["obs_stack_size"] == 3
    assert render_kwargs["observation_norm_state"] is normalization.observation
    assert result == rollout_stats()


def test_relocated_motion_validation_failure_prevents_rollout(
    monkeypatch, tmp_path: Path
) -> None:
    original = tmp_path / "original.npz"
    relocated = tmp_path / "tampered.npz"
    metadata = SimpleNamespace(
        task="motion_tracking",
        motion_env=MotionTrackingEnvConfig(motion_path=original),
        motion=SimpleNamespace(path=original),
        asset=object(),
        ppo=PPOConfig(),
    )
    env = SimpleNamespace(motion=object(), max_episode_steps=1)
    monkeypatch.setattr(
        eval_motion,
        "restore_evaluation_checkpoint",
        lambda path: (
            SimpleNamespace(policy=object()),
            metadata,
            SimpleNamespace(observation=object()),
        ),
    )
    monkeypatch.setattr(
        eval_motion.BelloMotionTrackingEnv,
        "load_native",
        staticmethod(lambda *args, **kwargs: env),
    )
    monkeypatch.setattr(
        eval_motion,
        "validate_motion_clip_contract",
        lambda *args: (_ for _ in ()).throw(
            ValueError("motion clip does not match checkpoint content/schema")
        ),
    )
    monkeypatch.setattr(
        eval_motion,
        "render_motion_policy",
        lambda *args, **kwargs: pytest.fail("invalid motion must not be rendered"),
    )

    with pytest.raises(ValueError, match="content/schema"):
        eval_motion.render_checkpoint_with_stats(
            tmp_path / "checkpoint",
            tmp_path / "rollout.mp4",
            motion_path=relocated,
        )


class _Resources:
    def __init__(self) -> None:
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


class _Renderer:
    def __init__(self) -> None:
        self.update_count = 0

    def update_scene(self, data, *, camera) -> None:
        self.update_count += 1

    def render(self) -> np.ndarray:
        return np.zeros((3, 4, 3), dtype=np.uint8)


class _Writer:
    def __init__(self) -> None:
        self.frames = []

    def append_data(self, frame: np.ndarray) -> None:
        self.frames.append(frame)


def install_native_rollout_fakes(monkeypatch):
    qpos = jnp.asarray(
        [0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0, 0.1, -0.2],
        dtype=jnp.float32,
    )
    state = SimpleNamespace(
        data=SimpleNamespace(qpos=qpos),
        reference_time=jnp.asarray(0.0),
        alignment_xy=jnp.zeros(2),
        alignment_yaw=jnp.asarray(0.0),
    )
    reference = SimpleNamespace(qpos=qpos)
    env = SimpleNamespace(
        model=SimpleNamespace(
            mj_model=object(),
            root_qpos_adr=0,
            policy_qpos_indices=np.asarray([7, 8]),
        ),
        config=SimpleNamespace(formulation="reference_residual"),
        max_episode_steps=4,
        _aligned_reference=lambda *args: reference,
    )
    resources = _Resources()
    renderer = _Renderer()
    writer = _Writer()
    monkeypatch.setattr(eval_motion.mujoco, "MjData", lambda model: object())
    monkeypatch.setattr(
        eval_motion,
        "reset_native_motion_rollout",
        lambda loaded_env, data: (state, jnp.asarray([3.0, 5.0])),
    )
    monkeypatch.setattr(
        eval_motion,
        "open_rollout_video_resources",
        lambda *args, **kwargs: (resources, renderer, object(), writer),
    )
    monkeypatch.setattr(eval_motion, "update_camera", lambda *args, **kwargs: None)
    return env, state, resources, renderer, writer


def test_native_rollout_uses_normalized_deterministic_actor_and_reports_stats(
    monkeypatch, tmp_path: Path
) -> None:
    env, state, resources, renderer, writer = install_native_rollout_fakes(monkeypatch)
    actor_inputs = []

    def apply_actor(policy, observation):
        actor_inputs.append(observation)
        return jnp.asarray([0.25], dtype=jnp.float32)

    monkeypatch.setattr(eval_motion, "apply_actor", apply_actor)
    step_count = 0

    def native_step(loaded_env, data, _previous_state, action):
        nonlocal step_count
        step_count += 1
        done = step_count == 2
        return (
            state,
            jnp.asarray([3.0, 5.0]),
            jnp.asarray(float(step_count)),
            jnp.asarray(done),
            {
                "done/terminal": jnp.asarray(False),
                "done/timeout": jnp.asarray(done),
                "done/nonfinite": jnp.asarray(False),
            },
        )

    monkeypatch.setattr(eval_motion, "native_motion_step", native_step)
    config = PPOConfig(
        normalize_observations=True,
        policy=ActorCriticConfig(obs_stack_size=2),
    )
    normalization = ObservationNormState(
        mean=jnp.ones(4, dtype=jnp.float32),
        var=jnp.full(4, 4.0, dtype=jnp.float32),
        count=jnp.asarray(100.0, dtype=jnp.float32),
    )

    stats = eval_motion.render_motion_policy(
        env,
        policy=object(),
        output_path=tmp_path / "rollout.mp4",
        steps=4,
        width=4,
        height=3,
        fps=20,
        obs_stack_size=2,
        ppo_config=config,
        observation_norm_state=normalization,
    )

    expected_actor_input = jnp.asarray([1.0, 2.0, 1.0, 2.0])
    assert len(actor_inputs) == 2
    assert all(
        np.allclose(np.asarray(observation), np.asarray(expected_actor_input))
        for observation in actor_inputs
    )
    assert stats == rollout_stats()
    assert renderer.update_count == 2
    assert len(writer.frames) == 2
    assert resources.close_count == 1


def test_native_rollout_closes_video_resources_when_step_fails(
    monkeypatch, tmp_path: Path
) -> None:
    env, _, resources, _, writer = install_native_rollout_fakes(monkeypatch)
    monkeypatch.setattr(
        eval_motion,
        "apply_actor",
        lambda policy, observation: jnp.zeros(1, dtype=jnp.float32),
    )
    monkeypatch.setattr(
        eval_motion,
        "native_motion_step",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("step failed")),
    )

    with pytest.raises(RuntimeError, match="step failed"):
        eval_motion.render_motion_policy(
            env,
            policy=object(),
            output_path=tmp_path / "rollout.mp4",
            steps=1,
            width=4,
            height=3,
            fps=20,
            obs_stack_size=1,
        )

    assert len(writer.frames) == 1
    assert resources.close_count == 1
