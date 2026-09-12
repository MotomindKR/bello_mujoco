"""CPU cross-language behavior checks with synthetic motors and observations."""

from types import SimpleNamespace
import json
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bello_mujoco.contracts import ObservationLayout
from bello_mujoco.deployment import actor_onnx, deployment_bundle
from bello_mujoco.envs.gait import (
    GaitConfig,
    advance_gait_phase,
    resolve_gait_command_transition,
)
from bello_mujoco.envs.locomotion_config import (
    BelloEnvConfig,
    ObservationConfig,
    ObservationScaleConfig,
)
from bello_mujoco.envs.rewards import CommandConfig
from bello_mujoco.hardware.locomotion import (
    HardwarePolicyAdapter,
    _transform_hardware_imu_frame,
)
from bello_mujoco.hardware.runtime import HardwareFrame
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
    adapter = HardwarePolicyAdapter.__new__(HardwarePolicyAdapter)
    adapter.env = SimpleNamespace(
        config=env_config, command_config=command_config, observation_layout=layout
    )
    adapter.inverse_transform = inverse
    adapter.default_joint_target = default
    adapter.policy_indices = indices
    adapter._last_action = np.zeros(2, dtype=np.float32)
    adapter._gait_phase = 0.0
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
        active = (
            adapter._gait_command_is_moving(np.asarray(current)) or int(pending) != 0
        )
        current, phase, pending = transition(
            current, jnp.array(request), phase, pending, preference, index > 0
        )
        next_active = (
            adapter._gait_command_is_moving(np.asarray(current)) or int(pending) != 0
        )
        if not active and next_active:
            preference = not preference
        adapter._gait_phase = float(phase)
        frame = HardwareFrame(
            positions=np.array(sample["motor_position"]),
            velocities=np.array(sample["motor_velocity"]),
            orientation_xyzw=np.array(sample["orientation_xyzw"]),
            angular_velocity=np.array(sample["angular_velocity"]),
            linear_acceleration=np.array(sample["linear_acceleration"]),
            max_motor_age_ms=0,
            imu_age_ms=0,
            group_states=(),
            group_causes=(),
            group_details=(),
            max_group_age_ms=0,
            sample_span_ms=0.0,
            state_frame_round_trip_ms=0.0,
        )
        frame = _transform_hardware_imu_frame(frame, rotation)
        observation = adapter._policy_frame_observation(
            frame, np.asarray(current), gait_active=next_active
        )
        history = (
            np.repeat(observation[None], 3, axis=0)
            if history is None
            else np.concatenate((history[1:], observation[None]))
        )
        result = infer(jnp.array(history.reshape(-1)))
        adapter._last_action = np.asarray(result.applied_action)
        span = np.where(
            adapter._last_action >= 0,
            limits[indices, 1] - nominal[indices],
            nominal[indices] - limits[indices, 0],
        )
        target = nominal.copy()
        target[indices] += np.clip(adapter._last_action, -1, 1) * span
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
