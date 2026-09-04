from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from bello_mujoco.assets_pipeline.prepare import PreparedAsset
from bello_mujoco.envs.locomotion import BelloLocomotionEnv, GaitConfig
from bello_mujoco.evaluation import locomotion as evaluation
from bello_mujoco.evaluation import render
from bello_mujoco.gamepad import GamepadInput
from bello_mujoco.policy import PolicyInferenceStep
from bello_mujoco.training import locomotion as training
from bello_mujoco.training.locomotion import TrainingVideoConfig
from bello_mujoco.training.ppo import (
    PPOConfig,
    init_observation_norm_state,
)


def component_slice(env: BelloLocomotionEnv, name: str) -> slice:
    offset = 0
    for component_name, size in env.observation_layout.components:
        if component_name == name:
            return slice(offset, offset + size)
        offset += size
    raise AssertionError(f"missing observation component {name!r}")


def test_gamepad_command_is_used_by_same_step_policy_and_log(
    monkeypatch,
    bello_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(bello_asset, load_mjx=False)
    commands = [
        np.zeros(3, dtype=np.float32),
        np.asarray((0.4, 0.0, 0.0), dtype=np.float32),
    ]
    inferred_commands = []
    logged_steps = []
    command_indices = component_slice(env, "command")

    class Gamepad:
        def __init__(self, *_args):
            self.poll_count = 0

        def poll(self):
            command = commands[self.poll_count]
            self.poll_count += 1
            return GamepadInput(command, connected=True)

        def close(self):
            pass

    class Logger:
        def __init__(self, *_args):
            pass

        def start(self):
            pass

        def close(self):
            pass

        def log_event(self, _message):
            pass

        def log_step(self, **values):
            logged_steps.append(values)

    def infer_policy_action(_policy, observation, *_args, **_kwargs):
        command = observation[command_indices]
        inferred_commands.append(np.asarray(command))
        action = jnp.zeros(env.action_size, dtype=jnp.float32).at[0].set(command[0])
        return PolicyInferenceStep(action, action, action)

    monkeypatch.setattr(evaluation, "GamepadCommandSource", Gamepad)
    monkeypatch.setattr(evaluation, "RerunEvaluationLogger", Logger)
    monkeypatch.setattr(evaluation, "infer_policy_action", infer_policy_action)
    monkeypatch.setattr(evaluation.eqx, "filter_jit", lambda function: function)
    monkeypatch.setattr(evaluation, "mujoco_eval_info", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(evaluation.time, "sleep", lambda _seconds: None)

    evaluation.run_rerun_evaluation(
        env,
        object(),
        normalization=init_observation_norm_state(env.observation_size),
        ppo_config=PPOConfig(),
        urdf_path=Path("unused.urdf"),
        command=(0.0, 0.0, 0.0),
        gamepad_config=evaluation.GamepadConfig(enabled=True),
        steps=2,
        stop_on_done=False,
        obs_stack_size=1,
        stochastic_actions=False,
        action_seed=0,
    )

    assert len(logged_steps) == 2
    assert inferred_commands[1] == pytest.approx(
        commands[1] * env.config.observation.scale.command
    )
    assert logged_steps[1]["command"] == pytest.approx(commands[1])
    assert logged_steps[1]["action"][0] == pytest.approx(inferred_commands[1][0])


def test_gamepad_standing_waits_for_next_gait_boundary_and_logs_effective_command(
    monkeypatch,
    bello_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(bello_asset, load_mjx=False)
    env = replace(
        env,
        config=replace(env.config, gait=GaitConfig(frequency_hz=12.5)),
        command_config=replace(
            env.command_config,
            standing_transitions_enabled=True,
        ),
    )
    moving = np.asarray((0.4, 0.0, 0.0), dtype=np.float32)
    commands = [moving, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)]
    inferred_commands = []
    inferred_gait_phases = []
    logged_commands = []
    command_indices = component_slice(env, "command")
    gait_phase_indices = component_slice(env, "gait_phase")

    class Gamepad:
        def __init__(self, *_args):
            self.poll_count = 0

        def poll(self):
            command = commands[self.poll_count]
            self.poll_count += 1
            return GamepadInput(command, connected=True)

        def close(self):
            pass

    class Logger:
        def __init__(self, *_args):
            pass

        def start(self):
            pass

        def close(self):
            pass

        def log_event(self, _message):
            pass

        def log_step(self, **values):
            logged_commands.append(values["command"].copy())

    def infer_policy_action(_policy, observation, *_args, **_kwargs):
        inferred_commands.append(np.asarray(observation[command_indices]))
        inferred_gait_phases.append(np.asarray(observation[gait_phase_indices]))
        action = jnp.zeros(env.action_size, dtype=jnp.float32)
        return PolicyInferenceStep(action, action, action)

    monkeypatch.setattr(evaluation, "GamepadCommandSource", Gamepad)
    monkeypatch.setattr(evaluation, "RerunEvaluationLogger", Logger)
    monkeypatch.setattr(evaluation, "infer_policy_action", infer_policy_action)
    monkeypatch.setattr(evaluation.eqx, "filter_jit", lambda function: function)
    monkeypatch.setattr(evaluation, "mujoco_eval_info", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(evaluation.time, "sleep", lambda _seconds: None)

    evaluation.run_rerun_evaluation(
        env,
        object(),
        normalization=init_observation_norm_state(env.observation_size),
        ppo_config=PPOConfig(),
        urdf_path=Path("unused.urdf"),
        command=(0.0, 0.0, 0.0),
        gamepad_config=evaluation.GamepadConfig(enabled=True),
        steps=3,
        stop_on_done=False,
        obs_stack_size=1,
        stochastic_actions=False,
        action_seed=0,
    )

    expected = np.asarray((moving, moving, np.zeros(3)), dtype=np.float32)
    assert np.asarray(inferred_commands) == pytest.approx(expected)
    assert np.asarray(logged_commands) == pytest.approx(expected)
    assert np.asarray(inferred_gait_phases) == pytest.approx(
        np.asarray(
            ((0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
            dtype=np.float32,
        ),
        abs=1.0e-6,
    )


def test_rollout_action_applies_observation_normalization(monkeypatch) -> None:
    observation_stack = jnp.asarray(((3.0, 5.0), (7.0, 9.0)))
    normalization = init_observation_norm_state(4)._replace(
        mean=jnp.ones(4),
        var=jnp.full(4, 4.0),
    )
    actor_inputs = []

    def apply_actor(_policy, observation):
        actor_inputs.append(np.asarray(observation))
        return observation[:2]

    monkeypatch.setattr(render, "apply_actor", apply_actor)

    action = render.rollout_policy_action(
        object(),
        observation_stack,
        action_clip=10.0,
        ppo_config=PPOConfig(normalize_observations=True),
        observation_norm_state=normalization,
    )

    assert actor_inputs == [pytest.approx((1.0, 2.0, 3.0, 4.0))]
    assert action == pytest.approx((1.0, 2.0))


def test_training_video_passes_runner_normalization_to_rollout(monkeypatch) -> None:
    normalization = init_observation_norm_state(4)
    runner_state = SimpleNamespace(
        update=jnp.asarray(3),
        observation_norm_state=normalization,
    )
    ppo_config = PPOConfig(normalize_observations=True)
    rollout_calls = []
    monkeypatch.setattr(
        training,
        "runner_train_state",
        lambda _runner_state, _static_policy: SimpleNamespace(policy=object()),
    )
    monkeypatch.setattr(
        training,
        "render_policy_rollout",
        lambda *_args, **kwargs: (
            rollout_calls.append(kwargs)
            or {"first_done_step": -1, "completed_episodes": 0}
        ),
    )

    training.render_training_video(
        SimpleNamespace(),
        runner_state,
        object(),
        TrainingVideoConfig(log_wandb=False),
        None,
        env_steps=10,
        ppo_config=ppo_config,
        obs_stack_size=2,
    )

    assert rollout_calls[0]["ppo_config"] is ppo_config
    assert rollout_calls[0]["observation_norm_state"] is normalization


def test_training_video_failure_is_nonfatal(monkeypatch) -> None:
    runner_state = SimpleNamespace(update=jnp.asarray(512))
    monkeypatch.setattr(
        training,
        "render_training_video",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("framebuffer unavailable")
        ),
    )

    with pytest.warns(
        RuntimeWarning,
        match="evaluation video failed at update 512.*disabling later videos",
    ):
        available = training.try_render_training_video(
            object(),
            runner_state,
            object(),
            TrainingVideoConfig(),
            None,
            env_steps=0,
            ppo_config=PPOConfig(),
        )

    assert available is False
