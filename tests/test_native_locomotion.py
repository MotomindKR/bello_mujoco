"""CPU cross-language behavior checks with synthetic motors and observations."""

import json
import subprocess

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from bello_mujoco.contracts import ObservationLayout
from bello_mujoco.deployment import actor_onnx, deployment_bundle
from bello_mujoco.envs.gait import (
    GaitConfig,
    advance_gait_phase,
    gait_is_active,
    gait_phase_features,
    resolve_gait_command_transition,
)
from bello_mujoco.envs.locomotion_config import (
    BelloEnvConfig,
    ObservationConfig,
    ObservationScaleConfig,
)
from bello_mujoco.envs.rewards import CommandConfig
from bello_mujoco.envs.observations import build_observation
from bello_mujoco.policy import ActorCritic, ActorCriticConfig, infer_policy_action
from bello_mujoco.training.ppo import PPOConfig, ObservationNormState
from bello_mujoco.training.tracking import to_builtin

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "slew,transitions", [(False, False), (False, True), (True, False), (True, True)]
)
def test_native_telemetry_matches_python_through_command_transitions(
    rust_policy, slew, transitions
):
    run, _, source = rust_policy
    gait = GaitConfig(
        command_slew_enabled=slew,
        directional_start=True,
        stop_extra_steps=1,
        speed_dependent=True,
    )
    command_config = CommandConfig(standing_transitions_enabled=transitions)
    scales = ObservationScaleConfig(
        base_lin_acc=0.2,
        base_ang_vel=0.3,
        joint_pos=1.2,
        joint_vel=0.4,
        command=0.8,
        last_action=0.9,
    )
    env_config = BelloEnvConfig(gait=gait, observation=ObservationConfig(scale=scales))
    layout = ObservationLayout(
        (
            ("imu_linear_acceleration", 3),
            ("imu_angular_velocity", 3),
            ("projected_gravity", 3),
            ("command", 3),
            ("gait_phase", 3),
            ("joint_position", 2),
            ("joint_velocity", 2),
            ("last_action", 2),
        )
    )
    config = PPOConfig(policy=ActorCriticConfig(hidden_dims=(8,), obs_stack_size=3))
    size = layout.size * config.policy.obs_stack_size
    actor = ActorCritic(size, 2, config.policy, key=jax.random.PRNGKey(3))
    norm = ObservationNormState(
        jnp.linspace(-0.1, 0.1, size), jnp.linspace(0.8, 1.2, size), jnp.array(3.0)
    )
    inverse = np.array([[0.5, 0.5, 0], [-0.5, 0.5, 0], [0, 0, 1.0]])
    default = np.array([0.1, -0.2, 0.3])
    rotation = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    indices = np.array([0, 2])
    limits = np.array([[-1, 1], [-1, 1], [-1, 1]])
    nominal = np.linalg.inv(inverse) @ default
    metadata = dict(
        git_commit=subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        task="locomotion",
        robot="synthetic",
        observation_size=size,
        observation_frame=layout.components,
        observation_stack_size=3,
        observation_max_abs=100.0,
        coordinates=["sum", "difference", "held"],
        policy_indices=indices.tolist(),
        default_position=nominal.tolist(),
        kp=[2.0] * 3,
        kd=[0.1] * 3,
        control_timestep=0.02,
        environment=to_builtin(env_config),
        command=to_builtin(command_config),
        joint_from_motor=inverse.tolist(),
        default_joint_position=default.tolist(),
        policy_from_hardware=rotation.tolist(),
        left_axis=[0.0, 1.0],
        action_clip=1.0,
        hardware_imu_frame="imu",
    )
    graph = actor_onnx(
        actor,
        norm,
        config,
        default_ctrl=nominal,
        ctrl_limits=limits,
        policy_indices=indices,
        action_clip=1.0,
        metadata=metadata,
    )
    bundle = deployment_bundle(graph.SerializeToString(), physics=b"<mujoco/>")
    last_action = np.zeros(2, dtype=np.float32)
    requests = (
        [[0, 0, 0]] * 3
        + [[0, 0.4, 0]] * 90
        + [[0, -0.4, 0.2]] * 90
        + [[0, 0, 0]] * 130
        + [[0.4, 0, 0]] * 20
        + [[0, 0, 0]] * 130
    )
    rng = np.random.default_rng(11)
    samples, expected = [], []
    current, phase, pending, preference = (
        jnp.zeros(3),
        jnp.array(0.0),
        jnp.array(0),
        True,
    )
    history = None
    infer = jax.jit(lambda obs: infer_policy_action(actor, obs, norm, config, 1.0))

    @jax.jit
    def transition(current, request, phase, pending, preference, started):
        next_phase = jnp.where(
            started,
            advance_gait_phase(phase, current, command_config, gait, 0.02, pending),
            phase,
        )
        return resolve_gait_command_transition(
            current,
            request,
            phase,
            next_phase,
            pending,
            command_config,
            gait,
            preference,
            (0.0, 1.0),
            jnp.where(started, 0.02, 0.0),
        )

    for index, request in enumerate(requests):
        sample = dict(
            motor_position=(nominal + rng.normal(0, 0.01, 3)).tolist(),
            motor_velocity=rng.normal(0, 0.1, 3).tolist(),
            orientation_xyzw=[
                0.05,
                0.1,
                0.2,
                float(np.sqrt(1 - 0.05**2 - 0.1**2 - 0.2**2)),
            ],
            angular_velocity=rng.normal(0, 0.1, 3).tolist(),
            linear_acceleration=rng.normal(0, 0.1, 3).tolist(),
            command=request,
        )
        samples.append(sample)
        active = bool(gait_is_active(current, command_config, pending, gait))
        current, phase, pending = transition(
            current, jnp.array(request), phase, pending, preference, index > 0
        )
        next_active = bool(gait_is_active(current, command_config, pending, gait))
        if not active and next_active:
            preference = not preference
        world_from_hardware = np.empty(9)
        quaternion = np.asarray(sample["orientation_xyzw"])[[3, 0, 1, 2]]
        mujoco.mju_quat2Mat(world_from_hardware, quaternion)
        world_from_policy = world_from_hardware.reshape(3, 3) @ rotation.T
        gravity = world_from_policy.T @ np.array([0.0, 0.0, -1.0])
        # Restore specific force from hardwared's gravity-removed acceleration,
        # then exercise the actual training observation assembler.
        terms = tuple(
            jnp.asarray(value, dtype=jnp.float32)
            for value in (
                np.zeros(3),
                rotation @ sample["linear_acceleration"] - 9.80665 * gravity,
                rotation @ sample["angular_velocity"],
                gravity,
                np.zeros(1),
                current,
                (inverse @ sample["motor_position"] - default)[indices],
                (inverse @ sample["motor_velocity"])[indices],
                np.zeros(2),
                last_action,
            )
        )
        observation = np.asarray(
            build_observation(
                terms,
                gait_phase_features(phase, current, command_config, pending, gait),
                env_config.observation,
                None,
            )
        )
        history = (
            np.repeat(observation[None], 3, axis=0)
            if history is None
            else np.concatenate((history[1:], observation[None]))
        )
        result = infer(jnp.array(history.reshape(-1)))
        last_action = np.asarray(result.applied_action)
        span = np.where(
            last_action >= 0,
            limits[indices, 1] - nominal[indices],
            nominal[indices] - limits[indices, 0],
        )
        target = nominal.copy()
        target[indices] += np.clip(last_action, -1, 1) * span
        expected.append(
            (
                observation,
                np.asarray(result.requested_action),
                np.clip(target, -1, 1),
                float(phase),
                int(pending),
            )
        )
    result = run(samples, content=bundle, telemetry=True)
    assert result.returncode == 0, result.stderr
    outputs = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(outputs) == len(expected)
    for index, (output, reference) in enumerate(zip(outputs, expected, strict=True)):
        observation, action, target, phase, pending = reference
        np.testing.assert_allclose(
            output["observation"], observation, atol=2e-5, err_msg=f"step {index}"
        )
        np.testing.assert_allclose(
            output["raw_action"], action, atol=2e-5, err_msg=f"step {index}"
        )
        np.testing.assert_allclose(
            output["motor_target"], target, atol=2e-5, err_msg=f"step {index}"
        )
        assert output["gait"]["pending"] == pending
        assert output["gait"]["phase"] == pytest.approx(phase, abs=2e-6)
    invalid = dict(samples[0], orientation_xyzw=[0, 0, 0, 0])
    result = run([invalid], content=bundle, telemetry=True)
    assert result.returncode != 0 and "invalid IMU quaternion" in result.stderr
