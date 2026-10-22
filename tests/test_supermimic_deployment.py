from dataclasses import replace
import json
import subprocess

import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from onnx.reference import ReferenceEvaluator
import pytest

from bello_mujoco.deployment import multi_clip_deployment_bundle
from bello_mujoco.models.target_limits import TargetLimits
from bello_mujoco.motion.clip import MotionClip, save_motion_clip
from bello_mujoco.sonic.reference import ReferenceBank
from bello_mujoco.supermimic import rotations as rot
from bello_mujoco.supermimic.config import ObservationConfig
from bello_mujoco.supermimic.control import ActionMapping
from bello_mujoco.supermimic.deployment import actor_onnx
from bello_mujoco.supermimic.kinematics import Kinematics
from bello_mujoco.supermimic.model import Actor, Moments, NetworkConfig
from bello_mujoco.supermimic.observations import ObservationCodec
from bello_mujoco.supermimic.state import PhysicalState


@pytest.fixture
def exported(tmp_path):
    from types import SimpleNamespace

    physics = b"""<mujoco><worldbody><body name="free" pos="0 0 1"><freejoint/><geom size=".1"/>
      <body name="root" pos=".02 .01 .03" quat=".9 .1 .2 .3">
        <geom size=".1"/><body name="imu" quat=".99 0 0 .1"/>
        <body name="a"><joint name="joint_a" axis="0 1 0"/><geom size=".1"/></body>
        <body name="b"><joint name="joint_b" axis="1 0 0"/><geom size=".1"/></body>
      </body></body></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(physics.decode())
    names = ("joint_b", "joint_a")
    kin = Kinematics.from_model(model, names, ("root", "a", "b"), root_body_name="root")
    frames = 10
    qpos = jnp.tile(jnp.asarray(model.qpos0), (frames, 1))
    qpos = qpos.at[:, 0].add(jnp.arange(frames) * 0.002)
    qpos = qpos.at[:, 3:7].set(
        jax.vmap(rot.exp)(
            jnp.stack(
                (jnp.zeros(frames), jnp.zeros(frames), jnp.arange(frames) * 0.01),
                axis=-1,
            )
        )
    )
    qpos = qpos.at[:, 7:].set(
        jnp.stack((jnp.arange(frames) * 0.01, jnp.arange(frames) * -0.02), axis=-1)
    )
    qvel = jnp.tile(
        jnp.array([0.1, 0.02, -0.03, 0.2, 0.1, 0.3, 0.5, -0.1]), (frames, 1)
    )
    physical = jax.vmap(kin.from_qpos)(qpos, qvel)
    bodies = jax.vmap(kin.forward)(physical)
    clip = MotionClip(
        50.0,
        names,
        kin.body_names,
        qpos,
        qvel,
        bodies.position,
        bodies.quaternion,
        bodies.linear_velocity,
        bodies.angular_velocity,
        jnp.zeros((frames, 2)),
    )
    clip_path = tmp_path / "motion.npz"
    save_motion_clip(clip_path, clip)
    bank = ReferenceBank.from_clips((("motion", clip, "train"),))
    codec = ObservationCodec(
        ObservationConfig(history_frames=3, future_frames=2), jnp.array([0.05, -0.1])
    )
    limits = TargetLimits(
        np.array([[-2.0, 2.0]] * 2),
        np.array([[-1.0, 1.0]] * 2),
        np.eye(2),
        np.array([0.15, 0.15]),
    )
    mapping = ActionMapping(
        names,
        2.0,
        codec.neutral_joint,
        codec.neutral_joint,
        jnp.eye(2),
        jnp.array([0.2, 0.3]),
        jnp.asarray(limits.matrix),
        jnp.asarray(limits.bounds),
        jnp.asarray(limits.membership),
        jnp.asarray(limits.groups),
        jnp.asarray(limits.position_step),
    )
    env = SimpleNamespace(codec=codec, mapping=mapping, action_clip=2.0)
    networks = NetworkConfig(
        actor_widths=(8,), dynamics_widths=(8,), critic_widths=(8,), latent_size=4
    )
    actor = Actor(
        codec.sample_size, 3, codec.goal_size, 2, config=networks, key=jax.random.key(9)
    )
    actor = eqx.tree_at(
        lambda net: net.decoder.layers[-1].bias, actor, jnp.array([2.0, -3.0])
    )
    obs_norm = replace(
        Moments.empty(codec.sample_size),
        mean=jnp.linspace(-0.1, 0.1, codec.sample_size),
        variance=jnp.linspace(0.2, 2.0, codec.sample_size),
    )
    goal_norm = replace(
        Moments.empty(codec.goal_size), mean=jnp.linspace(-0.1, 0.2, codec.goal_size)
    )
    imu = rot.normalize(jnp.array([0.99, 0.0, 0.0, 0.1]))
    metadata = {
        "git_commit": "0" * 40,
        "task": "supermimic",
        "robot": "synthetic",
        "observation_size": 3 * codec.sample_size + codec.goal_size,
        "observation_stack_size": 1,
        "observation_max_abs": 10000.0,
        "observation_frame": [
            ["proprioceptive_history", 3 * codec.sample_size],
            ["future_reference", codec.goal_size],
        ],
        "coordinates": ["b", "a"],
        "policy_indices": [0, 1],
        "default_position": codec.neutral_joint.tolist(),
        "kp": [1.0, 1.0],
        "kd": [0.1, 0.1],
        "control_timestep": 0.02,
        "joint_names": list(names),
        "joint_from_motor": np.eye(2).tolist(),
        "motor_from_joint": np.eye(2).tolist(),
        "default_joint_position": codec.neutral_joint.tolist(),
        "motor_limits": limits.motor.tolist(),
        "joint_limits": limits.joint.tolist(),
        "position_step_limits": limits.position_step.tolist(),
        "action_clip": 2.0,
        "reference_clips": ["motion"],
        "reference_root_body": "root",
        "hardware_imu_frame": "imu",
        "root_from_hardware_imu_xyzw": np.roll(np.asarray(imu), -1).tolist(),
        "history_frames": 3,
        "future_frames": 2,
        "reference_orientation": True,
        "goal_joint_velocity_scale": 0.1,
        "rotation_6d_layout": "columns",
        "sample_frame": [
            ["joint_position", 2],
            ["joint_velocity", 2],
            ["base_angular_velocity", 3],
            ["gravity_direction", 3],
            ["last_action", 2],
            ["reference_orientation_6d", 6],
        ],
    }
    return env, actor, obs_norm, goal_norm, metadata, physics, bank, kin, clip_path, imu


def trace(exported, *, vertical=False):
    env, actor, obs_norm, goal_norm, _, _, bank, kin, _, imu = exported
    history, alignment = None, None
    previous_action = jnp.zeros(2)
    previous_motor = env.mapping.neutral_motor
    rows, expected = [], []
    for index in range(3):
        orientation = rot.exp(jnp.array([0.1, 0.02, 0.2 + index * 0.01]))
        if vertical:
            orientation = rot.multiply(
                rot.exp(jnp.array([0.0, 0.0, 0.2 + index * 0.01])),
                rot.exp(jnp.array([0.0, jnp.pi / 2, 0.0])),
            )
        current_sample = bank.sample_local_frame(
            jnp.asarray(0), jnp.asarray(float(index))
        )
        current = kin.from_qpos(current_sample.qpos, current_sample.qvel)
        if alignment is None:
            alignment = rot.multiply(
                rot.heading(orientation), rot.conjugate(rot.heading(current.quaternion))
            )
        position = jnp.array([0.05 + 0.01 * index, -0.1 + 0.03 * index])
        velocity = jnp.array([0.2, -0.15])
        angular = jnp.array([0.1, 0.2, 0.3])
        feedback = PhysicalState(
            jnp.zeros(3),
            orientation,
            jnp.zeros(3),
            rot.rotate(orientation, angular),
            position,
            velocity,
        )
        sample = env.codec.sample(
            feedback,
            previous_action,
            rot.multiply(alignment, current.quaternion),
            jnp.zeros(2),
        )
        history = (
            env.codec.initial_history(sample)
            if history is None
            else env.codec.advance_history(history, sample)
        )
        future_samples = bank.window(
            jnp.asarray(0), jnp.asarray(float(index)), jnp.arange(1, 3)
        )
        future = jax.vmap(kin.from_qpos)(future_samples.qpos, future_samples.qvel)
        goal = env.codec.goal(current, future)
        action = actor(history, goal, obs_norm, goal_norm)
        command = env.mapping.command(action, previous_motor, jnp.zeros(2))
        rows.append(
            {
                "motor_position": position.tolist(),
                "motor_velocity": velocity.tolist(),
                "previous_motor_target": previous_motor.tolist(),
                "orientation_xyzw": np.roll(
                    np.asarray(rot.multiply(orientation, imu)), -1
                ).tolist(),
                "angular_velocity_local": np.asarray(
                    rot.rotate(rot.conjugate(imu), angular)
                ).tolist(),
                "reference_time": index * 0.02,
                "initialize": index == 0,
            }
        )
        expected.append((jnp.concatenate((history.reshape(-1), goal)), action, command))
        previous_action, previous_motor = command.action, command.accepted
    return rows, expected


def test_onnx_actor_matches_normalized_jax_history_goal_and_motor_proposals(exported):
    env, actor, obs, goal, metadata, *_ = exported
    graph = actor_onnx(actor, obs, goal, env=env, metadata=metadata)
    evaluator = ReferenceEvaluator(graph)
    _, expected = trace(exported)
    for observation, action, command in expected:
        outputs = evaluator.run(None, {"observation": np.asarray(observation)[None]})
        np.testing.assert_allclose(outputs[0][0], action, atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(outputs[1][0], command.requested, atol=2e-6)


@pytest.mark.integration
@pytest.mark.parametrize("vertical", [False, True])
def test_rust_codec_matches_jax_including_named_joints_fixed_root_and_step_limits(
    exported, rust_policy, vertical
):
    run, _, source = rust_policy
    env, actor, obs, goal, metadata, physics, _, _, clip_path, _ = exported
    metadata["git_commit"] = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    graph = actor_onnx(actor, obs, goal, env=env, metadata=metadata)
    bundle = multi_clip_deployment_bundle(
        graph.SerializeToString(), physics=physics, clips=(("motion", clip_path),)
    )
    assert run(content=bundle, check=True, clip="motion").returncode == 0
    rows, expected = trace(exported, vertical=vertical)
    result = run(rows, content=bundle, telemetry=True, clip="motion")
    assert result.returncode == 0, result.stderr
    actual = [json.loads(line) for line in result.stdout.splitlines()]
    for row, (observation, action, command) in zip(actual, expected, strict=True):
        np.testing.assert_allclose(
            row["observation"], observation, atol=1e-5, rtol=1e-5
        )
        np.testing.assert_allclose(row["raw_action"], action, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(row["motor_target"], command.accepted, atol=2e-6)
    bad = dict(rows[0], initialize=False)
    assert run((bad,), content=bundle, telemetry=True, clip="motion").returncode != 0
    assert (
        run(
            (rows[0], rows[0] | {"initialize": False}),
            content=bundle,
            telemetry=True,
            clip="motion",
        ).returncode
        != 0
    )
