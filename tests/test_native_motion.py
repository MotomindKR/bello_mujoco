"""Native motion math/ONNX parity on a small articulated model, without robot assets."""

from dataclasses import replace
import json
import subprocess
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from bello_mujoco.deployment import actor_onnx, deployment_bundle
from bello_mujoco.envs.actions import bounded_motor_target
from bello_mujoco.envs.motion_tracking import (
    BelloMotionTrackingEnv,
    MotionTrackingEnvConfig,
)
from bello_mujoco.motion.clip import MotionClip, sample_motion
from bello_mujoco.policy import ActorCritic, ActorCriticConfig
from bello_mujoco.training.ppo_types import ObservationNormState
from bello_mujoco.training.ppo_config import PPOConfig
from bello_mujoco.training.normalization import normalize_observation
from bello_mujoco.training.tracking import to_builtin

pytestmark = pytest.mark.integration

XML = """<mujoco><worldbody><body name="root"><freejoint/>
  <geom type="sphere" size=".1" mass="1"/>
  <body name="anchor" pos=".2 .1 .1"><joint name="a" axis="0 1 0"/>
    <geom type="capsule" size=".03 .15" mass=".2"/>
    <body name="tip" pos="0 0 -.3"><joint name="b" axis="1 0 0"/>
      <geom type="sphere" size=".03" mass=".1"/>
    </body>
  </body></body></worldbody></mujoco>"""


class ObservationEnvironment(SimpleNamespace):
    # Exercise the production JAX math with synthetic model dimensions.
    observation = BelloMotionTrackingEnv.observation
    observation_layout = BelloMotionTrackingEnv.observation_layout
    _full_observation_layout = BelloMotionTrackingEnv._full_observation_layout
    _actor_observation_indices = BelloMotionTrackingEnv._actor_observation_indices
    _observation_terms = BelloMotionTrackingEnv._observation_terms
    aligned_reference = BelloMotionTrackingEnv.aligned_reference
    unaligned_reference = BelloMotionTrackingEnv.unaligned_reference
    bounded_reference_time = BelloMotionTrackingEnv.bounded_reference_time
    advance_reference_time = BelloMotionTrackingEnv.advance_reference_time
    _unpack_actor_feedback = BelloMotionTrackingEnv._unpack_actor_feedback
    _corrupt_observation_terms = BelloMotionTrackingEnv._corrupt_observation_terms


@pytest.fixture(params=["full_state", "proprioceptive"])
def observation_mode(request):
    return request.param


@pytest.fixture(params=[("a", "b"), ("b", "a")], ids=["model-order", "policy-order"])
def motion_fixture(rust_policy, request, observation_mode):
    run, _, source = rust_policy
    model = mujoco.MjModel.from_xml_string(XML)
    joint_names = request.param
    qpos_indices = np.array([model.joint(name).qposadr[0] for name in joint_names])
    qvel_indices = np.array([model.joint(name).dofadr[0] for name in joint_names])
    data = mujoco.MjData(model)
    ids = [model.body(name).id for name in ("root", "anchor", "tip")]
    qpos, qvel, positions, quaternions = [], [], [], []
    for i in range(4):
        quat = Rotation.from_euler("ZYX", [0.3 + i * 0.16, -0.12, 0.08]).as_quat()
        data.qpos[:] = [
            i * 0.04,
            -0.2,
            0.9,
            *quat[[3, 0, 1, 2]],
            0.1 + 0.08 * i,
            -0.2 + 0.06 * i,
        ]
        mujoco.mj_kinematics(model, data)
        qpos.append(data.qpos.copy())
        qvel.append([0.4, 0.1, -0.02, 0.2, 0.3, 0.4, 0.8, 0.6])
        positions.append(data.xpos[ids].copy())
        quaternions.append(data.xquat[ids].copy() * (-1 if i % 2 else 1))
    clip = MotionClip(
        fps=10.0,
        joint_names=joint_names,
        body_names=("root", "anchor", "tip"),
        qpos=jnp.array(qpos),
        qvel=jnp.array(qvel),
        body_pos=jnp.array(positions),
        body_quat=jnp.array(quaternions),
        body_lin_vel=jnp.zeros((4, 3, 3)),
        body_ang_vel=jnp.zeros((4, 3, 3)),
        foot_contacts=jnp.zeros((4, 2)),
    )
    default = np.array([0.1, -0.2])[qpos_indices - 7]
    inverse = np.array([[0.5, 0.5], [-0.5, 0.5]])
    transform = np.linalg.inv(inverse)
    config = MotionTrackingEnvConfig(
        anchor_body_name="anchor", observation_mode=observation_mode
    )
    env = ObservationEnvironment(
        motion=clip,
        action_size=2,
        config=config,
        control_timestep=0.02,
        anchor_body_index=1,
        model=SimpleNamespace(
            mj_model=model,
            policy_qpos_indices=qpos_indices,
            policy_qvel_indices=qvel_indices,
            policy_ctrl_indices=np.arange(2),
            default_joint_target=default,
        ),
    )
    ppo = PPOConfig(policy=ActorCriticConfig(hidden_dims=(8,), obs_stack_size=3))
    size = env.observation_layout.size * 3
    actor = ActorCritic(size, 2, ppo.policy, key=jax.random.PRNGKey(41))
    norm = ObservationNormState(
        jnp.linspace(-0.1, 0.1, size), jnp.linspace(0.8, 1.2, size), jnp.array(3.0)
    )
    limits = np.array([[-1.0, 1.0], [-1.5, 1.5]])
    metadata = dict(
        git_commit=subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        task="motion_tracking",
        robot="synthetic",
        observation_size=size,
        observation_frame=env.observation_layout.components,
        observation_stack_size=3,
        observation_max_abs=100.0,
        coordinates=["sum", "difference"],
        policy_indices=[0, 1],
        default_position=(transform @ default).tolist(),
        kp=[2.0, 2.0],
        kd=[0.1, 0.1],
        control_timestep=0.02,
        joint_names=list(joint_names),
        joint_from_motor=inverse.tolist(),
        motor_limits=limits.tolist(),
        joint_limits=limits.tolist(),
        position_step_limits=[2.0] * 2,
        default_joint_position=default.tolist(),
        end_effector_names=["tip"],
        environment=to_builtin(config),
        action_clip=1.0,
    )

    def bundle(*, metadata_updates=None, motion=None, effort_scale=None, physics=XML):
        graph = actor_onnx(
            actor,
            norm,
            ppo,
            default_ctrl=transform @ default,
            ctrl_limits=limits,
            policy_indices=np.arange(2),
            action_clip=1.0,
            metadata=metadata | (metadata_updates or {}),
            effort_scale=effort_scale,
        )
        return deployment_bundle(
            graph.SerializeToString(), physics=physics.encode(), motion=motion or clip
        )

    sample = dict(
        motor_position=(transform @ default).tolist(),
        motor_velocity=[0.2, -0.1],
        position=[1.0, -0.5, 0.9],
        orientation_xyzw=Rotation.from_euler("ZYX", [0.9, -0.1, 0.05])
        .as_quat()
        .tolist(),
        linear_velocity_world=[0.3, -0.2, 0.04],
        angular_velocity_local=[0.2, -0.1, 0.3],
        reference_time=0.03,
        initialize=True,
    )
    return SimpleNamespace(**locals())


def test_motion_bundle_check_initializes_named_joint_kinematics(motion_fixture):
    f = motion_fixture
    result = f.run(content=f.bundle(), check=True)
    assert result.returncode == 0, result.stderr
    result = f.run(
        content=f.bundle(physics=XML.replace('name="a"', 'name="missing"')),
        check=True,
    )
    assert result.returncode != 0
    assert "missing motion joint a" in result.stderr


@pytest.mark.parametrize("effort_scale", [None, np.array([0.7, 1.2])])
def test_motion_telemetry_matches_jax_with_alignment_history_and_coupling(
    motion_fixture, effort_scale
):
    f = motion_fixture
    history = None
    action = np.zeros(2)
    samples, expected = [], []
    for i in range(12):
        initialize = i in (0, 7)
        time = 0.295 if i == 11 else 0.03 + 0.017 * (i if i < 7 else i - 7)
        pose = Rotation.from_euler("ZYX", [0.9 + 0.025 * i, -0.1 + 0.002 * i, 0.05])
        position = np.array([1.0 + 0.005 * i, -0.5 + 0.001 * i, 0.9])
        sample = f.sample | dict(
            reference_time=time,
            initialize=initialize,
            orientation_xyzw=pose.as_quat().tolist(),
            position=position.tolist(),
            motor_position=(
                f.transform @ (f.default + np.array([0.002 * i, -0.001 * i]))
            ).tolist(),
        )
        if initialize:
            source = sample_motion(f.clip, jnp.array(time))
            source_rotation = Rotation.from_quat(
                np.asarray(source.qpos[3:7])[[1, 2, 3, 0]]
            )
            yaw = pose.as_euler("ZYX")[0] - source_rotation.as_euler("ZYX")[0]
            xy = (
                position
                - Rotation.from_euler("Z", yaw).apply(np.asarray(source.qpos[:3]))
            )[:2]
            action = np.zeros(2)
            history = None
        f.data.qpos[:7] = [
            *position,
            *pose.as_quat()[[3, 0, 1, 2]],
        ]
        f.data.qpos[f.qpos_indices] = f.inverse @ sample["motor_position"]
        f.data.qvel[:6] = [
            *sample["linear_velocity_world"],
            *sample["angular_velocity_local"],
        ]
        f.data.qvel[f.qvel_indices] = f.inverse @ sample["motor_velocity"]
        mujoco.mj_kinematics(f.model, f.data)
        state = SimpleNamespace(
            reference_time=jnp.array(time),
            alignment_xy=jnp.array(xy),
            alignment_yaw=jnp.array(yaw),
            last_action=jnp.array(action),
            joint_default_offset=jnp.zeros(2),
        )
        feedback = jnp.array(
            np.concatenate(
                (
                    f.data.qpos,
                    f.data.qvel,
                    f.data.xpos[f.ids[1]],
                    f.data.xquat[f.ids[1]],
                )
            )
        )
        observation = np.asarray(f.env.observation(state, _feedback=feedback))
        history = (
            np.tile(observation, (3, 1))
            if history is None
            else np.vstack((history[1:], observation))
        )
        raw = np.asarray(
            f.actor.action_mean(
                normalize_observation(jnp.array(history.reshape(-1)), f.norm, f.ppo)
            )
        )
        action = np.clip(raw, -1.0, 1.0)
        if effort_scale is None:
            target = bounded_motor_target(
                jnp.array(action),
                jnp.array(f.transform @ f.default),
                jnp.array(f.limits),
                jnp.arange(2),
                1.0,
            )
        else:
            target = np.clip(
                f.transform @ f.default + effort_scale * action,
                f.limits[:, 0],
                f.limits[:, 1],
            )
        expected.append((history.copy().reshape(-1), raw, action, target))
        samples.append(sample)
    result = f.run(samples, content=f.bundle(effort_scale=effort_scale), telemetry=True)
    assert result.returncode == 0, result.stderr
    actual = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(actual) == len(expected)
    for row, values in zip(actual, expected, strict=True):
        for key, value in zip(
            ("observation", "raw_action", "action", "motor_target"), values, strict=True
        ):
            np.testing.assert_allclose(
                row[key], value, atol=2e-5, rtol=2e-5, err_msg=key
            )


@pytest.mark.parametrize(
    "update,message",
    [
        ({"initialize": False}, "initialize motion alignment"),
        ({"orientation_xyzw": [0.0, 0.0, 0.0, 0.0]}, "invalid motion quaternion"),
        ({"motor_position": [0.0]}, "feedback shape mismatch"),
        ({"reference_time": 0.3}, "motion complete"),
        ({"reference_time": -0.1}, "invalid reference time"),
        ({"orientation_xyzw": [1.0, 0.0, 0.0, 0.0]}, "anchor orientation"),
    ],
)
def test_motion_feedback_fails_closed(motion_fixture, update, message):
    f = motion_fixture
    result = f.run([f.sample | update], content=f.bundle(), telemetry=True)
    assert result.returncode != 0
    assert message in result.stderr
    assert not result.stdout


def test_motion_rejects_mismatched_clip_and_invalid_quaternion(motion_fixture):
    f = motion_fixture
    for clip, message in [
        (
            replace(f.clip, joint_names=tuple(reversed(f.joint_names))),
            "joint orders differ",
        ),
        (
            replace(f.clip, body_quat=f.clip.body_quat.at[0, 1].set(0.0)),
            "invalid motion quaternion",
        ),
    ]:
        result = f.run([f.sample], content=f.bundle(motion=clip), telemetry=True)
        assert result.returncode != 0
        assert message in result.stderr


def test_motion_checks_end_effector_height_before_inference(motion_fixture):
    f = motion_fixture
    environment = to_builtin(
        replace(
            f.config, done=replace(f.config.done, end_effector_position_error=0.001)
        )
    )
    result = f.run(
        [f.sample],
        content=f.bundle(metadata_updates={"environment": environment}),
        telemetry=True,
    )
    if f.observation_mode == "full_state":
        assert result.returncode != 0
        assert "end effector height" in result.stderr
        assert not result.stdout
    else:
        assert result.returncode == 0, result.stderr


def test_motion_global_feedback_is_required_only_for_full_state(motion_fixture):
    f = motion_fixture
    bundle = f.bundle()
    baseline = f.run([f.sample], content=bundle, telemetry=True)
    assert baseline.returncode == 0, baseline.stderr
    for update, error in [
        ({"position": [1.0, -0.5, 2.0]}, "anchor height"),
        ({"position": None}, "requires position"),
        ({"linear_velocity_world": None}, "requires linear_velocity_world"),
    ]:
        result = f.run([f.sample | update], content=bundle, telemetry=True)
        if f.observation_mode == "full_state":
            assert result.returncode != 0
            assert error in result.stderr
        else:
            assert result.returncode == 0, result.stderr
            assert json.loads(result.stdout) == json.loads(baseline.stdout)


def test_motion_mode_must_match_exported_actor_layout(motion_fixture):
    f = motion_fixture
    other = "proprioceptive" if f.observation_mode == "full_state" else "full_state"
    for mode, error in [(other, "observation layout"), ("unknown", "unknown variant")]:
        environment = f.metadata["environment"] | {"observation_mode": mode}
        result = f.run(
            content=f.bundle(metadata_updates={"environment": environment}), check=True
        )
        assert result.returncode != 0
        assert error in result.stderr
