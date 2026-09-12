from dataclasses import fields, replace
import pickle
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from bello_mujoco.models.compiler import CompiledRobot
from bello_mujoco.envs.locomotion import BelloLocomotionEnv
from bello_mujoco.envs.gait import GaitConfig
from bello_mujoco.envs.locomotion_config import (
    ObservationLatencyConfig,
    ObservationLowPassFilterConfig,
)
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


def test_native_feedback_retains_filter_delay_and_resets(bello_asset) -> None:
    env = BelloLocomotionEnv.from_compiled(bello_asset, load_mjx=False)
    env = replace(
        env,
        config=replace(
            env.config,
            observation=replace(
                env.config.observation,
                noise=replace(env.config.observation.noise, enabled=False),
            ),
            observation_low_pass_filter=ObservationLowPassFilterConfig(enabled=True),
            observation_latency=ObservationLatencyConfig(
                enabled=True, delay_steps=(2, 2)
            ),
        ),
    )
    data = mujoco.MjData(env.model.mj_model)
    render.reset_mujoco_data(env, data)
    feedback = render.NativeObservationPipeline(env, seed=7)
    kwargs = dict(command=np.zeros(3), last_action=np.zeros(env.action_size))
    initial = feedback.observe(data, step_count=0, **kwargs)
    data.qvel[env.model.root_qvel_adr + 3] = 1.0
    mujoco.mj_forward(env.model.mj_model, data)
    raw = render.mujoco_policy_observation(env, data, step_count=1, **kwargs)
    gyro = component_slice(env, "imu_angular_velocity")
    assert not np.allclose(raw[gyro], initial[gyro])
    for step in (1, 2):
        delayed = feedback.observe(data, step_count=step, **kwargs)
        np.testing.assert_allclose(delayed[gyro], initial[gyro], atol=1e-7)
    filtered = feedback.observe(data, step_count=3, **kwargs)
    alpha = 1.0 - np.exp(-2 * np.pi * 10.0 * env.control_timestep)
    np.testing.assert_allclose(
        filtered[gyro], initial[gyro] + alpha * (raw[gyro] - initial[gyro]), atol=1e-7
    )
    reread = feedback.observe(
        data, step_count=3, **{**kwargs, "command": np.array([0.4, 0.0, 0.0])}
    )
    np.testing.assert_array_equal(reread[gyro], filtered[gyro])
    assert reread[component_slice(env, "command")][0] == pytest.approx(0.4)
    feedback.reset()
    reset = feedback.observe(data, step_count=0, **kwargs)
    np.testing.assert_allclose(reset[gyro], raw[gyro], atol=1e-7)
    with pytest.raises(ValueError, match="expected step 1"):
        feedback.observe(data, step_count=2, **kwargs)


def test_native_feedback_noise_is_seeded_and_not_redrawn_on_reread(bello_asset) -> None:
    env = BelloLocomotionEnv.from_compiled(bello_asset, load_mjx=False)
    env = replace(
        env,
        config=replace(
            env.config,
            observation_low_pass_filter=ObservationLowPassFilterConfig(
                enabled=True, noise_before_filter=True, residual_noise_scale=0.1
            ),
            observation=replace(
                env.config.observation,
                noise=replace(env.config.observation.noise, enabled=True),
            ),
        ),
    )
    data = mujoco.MjData(env.model.mj_model)
    render.reset_mujoco_data(env, data)
    kwargs = dict(command=np.zeros(3), last_action=np.zeros(env.action_size))
    first = render.NativeObservationPipeline(env, seed=7)
    same = render.NativeObservationPipeline(env, seed=7)
    different = render.NativeObservationPipeline(env, seed=8)
    initial = first.observe(data, step_count=0, **kwargs)
    np.testing.assert_array_equal(initial, first.observe(data, step_count=0, **kwargs))
    np.testing.assert_array_equal(initial, same.observe(data, step_count=0, **kwargs))
    assert not np.allclose(initial, different.observe(data, step_count=0, **kwargs))
    next_observation = first.observe(data, step_count=1, **kwargs)
    np.testing.assert_array_equal(
        next_observation, same.observe(data, step_count=1, **kwargs)
    )
    assert not np.allclose(initial, next_observation)


@pytest.mark.parametrize("command_delay", [0.0, 0.01])
def test_training_video_preserves_feedback_delay_across_physics_steps(
    monkeypatch, bello_asset, tmp_path, command_delay
) -> None:
    env = BelloLocomotionEnv.from_compiled(bello_asset, load_mjx=False)
    env = replace(
        env,
        config=replace(
            env.config,
            command_delay_seconds=command_delay,
            observation=replace(
                env.config.observation,
                noise=replace(env.config.observation.noise, enabled=False),
            ),
            observation_latency=ObservationLatencyConfig(
                enabled=True, delay_steps=(2, 2)
            ),
        ),
    )
    captured = []
    applied_controls = []
    mj_step = render.mujoco.mj_step

    def record_step(model, data):
        applied_controls.append(data.ctrl.copy())
        mj_step(model, data)

    monkeypatch.setattr(render.mujoco, "mj_step", record_step)

    def action(_policy, stack, **_kwargs):
        captured.append(np.asarray(stack[-1]))
        return jnp.full(env.action_size, 0.2)

    monkeypatch.setattr(render, "rollout_policy_action", action)
    monkeypatch.setattr(render, "mujoco_done_flags", lambda *_args: {"done/any": False})
    monkeypatch.setattr(render, "update_camera", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        render,
        "open_rollout_video_resources",
        lambda *_args, **_kwargs: (
            SimpleNamespace(close=lambda: None),
            SimpleNamespace(
                update_scene=lambda *_args, **_kwargs: None, render=lambda: None
            ),
            object(),
            SimpleNamespace(append_data=lambda _frame: None),
        ),
    )
    render.render_policy_rollout_mujoco(
        env,
        object(),
        output_path=tmp_path / "unused.mp4",
        command=(0.0, 0.0, 0.0),
        steps=4,
        width=2,
        height=2,
        fps=50,
        stop_on_done=False,
        camera_config=render.CameraConfig(),
        ppo_config=PPOConfig(),
        observation_norm_state=init_observation_norm_state(env.observation_size),
    )
    previous = render.mujoco_action_to_ctrl(env, np.zeros(env.action_size))
    requested = render.mujoco_action_to_ctrl(env, np.full(env.action_size, 0.2))
    for index in range(env.decimation):
        expected = (
            previous
            if index * float(env.model.mj_model.opt.timestep) < command_delay
            else requested
        )
        np.testing.assert_allclose(applied_controls[index], expected, atol=1e-6)
    velocity = component_slice(env, "joint_velocity")
    for observation in captured[:3]:
        np.testing.assert_array_equal(observation[velocity], captured[0][velocity])
    assert not np.allclose(captured[3][velocity], captured[0][velocity])


def test_gamepad_command_is_used_by_same_step_policy_and_log(
    monkeypatch,
    bello_asset: CompiledRobot,
) -> None:
    env = BelloLocomotionEnv.from_compiled(bello_asset, load_mjx=False)
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
    bello_asset: CompiledRobot,
) -> None:
    env = BelloLocomotionEnv.from_compiled(bello_asset, load_mjx=False)
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
        "render_policy_rollout_isolated",
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


def test_training_video_failure_stops_training(monkeypatch) -> None:
    runner_state = SimpleNamespace(update=jnp.asarray(0))
    monkeypatch.setattr(
        training,
        "render_training_video",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("framebuffer unavailable")
        ),
    )

    with pytest.raises(RuntimeError, match="framebuffer unavailable"):
        training.train_update_loop(
            object(),
            object(),
            PPOConfig(),
            object(),
            TrainingVideoConfig(enabled=True, render_initial=True),
            None,
            initial_runner=(runner_state, object(), object()),
        )


def test_isolated_video_sends_host_policy_without_environment_caches(monkeypatch):
    env = object.__new__(BelloLocomotionEnv)
    model = SimpleNamespace(
        mjx_model=None,
        mj_model=mujoco.MjModel.from_xml_string("<mujoco/>"),
    )
    for field in fields(env):
        object.__setattr__(
            env, field.name, model if field.name == "model" else field.default
        )
    object.__setattr__(env, "cached_device_value", jnp.ones(1))
    captured = []

    def capture(function, env_fields, policy, kwargs):
        # Native MuJoCo models must also survive the spawn serialization boundary.
        payload = pickle.loads(pickle.dumps((env_fields, policy, kwargs)))
        captured.append(payload)
        return {"rendered_steps": 1}

    monkeypatch.setattr(render, "run_software_evaluation", capture)
    stats = render.render_policy_rollout_isolated(
        env,
        jnp.ones(2),
        observation_norm_state=init_observation_norm_state(2),
    )
    assert stats == {"rendered_steps": 1}
    env_fields, policy, kwargs = captured[0]
    assert "cached_device_value" not in env_fields
    assert env_fields["model"].mj_model.nq == 0
    assert isinstance(policy, np.ndarray)
    assert isinstance(kwargs["observation_norm_state"].mean, np.ndarray)


def test_isolated_video_rejects_training_model():
    with pytest.raises(ValueError, match="native-only"):
        render.render_policy_rollout_isolated(
            SimpleNamespace(model=SimpleNamespace(mjx_model=object())),
            object(),
        )
