from dataclasses import dataclass, replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from bello_mujoco.envs.motion_tracking import MotionTrackingEnvConfig
from bello_mujoco.models.target_limits import TargetLimits
from bello_mujoco.motion.clip import MotionClip
from bello_mujoco.sonic.reference import ReferenceBank
from bello_mujoco.supermimic.collection import initialize_collection
from bello_mujoco.supermimic.config import LearningConfig, ObservationConfig
from bello_mujoco.supermimic.control import ActionMapping
from bello_mujoco.supermimic.kinematics import Kinematics
from bello_mujoco.supermimic.mdp import TrackingConfig, TrackingMDP
from bello_mujoco.supermimic.model import NetworkConfig
from bello_mujoco.supermimic.observations import ObservationCodec
from bello_mujoco.supermimic.optimization import initialize_learner
from bello_mujoco.supermimic.sampling import ReturnSampler
from bello_mujoco.training.supermimic import compile_updates, require_valid_metrics


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ToyData:
    qpos: jax.Array
    qvel: jax.Array
    ctrl: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ToyState:
    data: ToyData
    last_action: jax.Array
    actor_feedback_history: jax.Array
    actor_feedback_delay_steps: jax.Array
    reference_time: jax.Array
    alignment_xy: jax.Array
    alignment_yaw: jax.Array
    joint_default_offset: jax.Array
    episode_length: jax.Array


@dataclass(frozen=True)
class ToyEnv:
    reference_bank: ReferenceBank
    kinematics: Kinematics
    codec: ObservationCodec
    mapping: ActionMapping
    mdp: TrackingMDP
    sampler: ReturnSampler
    reset_probabilities: jax.Array
    config: MotionTrackingEnvConfig = MotionTrackingEnvConfig()
    action_size: int = 1
    action_clip: float = 20.0
    control_timestep: float = 0.02
    maximum_observation_delay_steps: int = 0

    def physical_state(self, state):
        return self.kinematics.from_qpos(state.data.qpos, state.data.qvel)

    def reference_state(self, frame, xy, yaw):
        sample = self.reference_bank.sample_global_frame(frame)
        return self.kinematics.from_qpos(sample.qpos, sample.qvel)

    def clip_id(self, frame):
        return jnp.zeros_like(frame, dtype=jnp.int32)

    def advance_reference_time(self, frame):
        return frame + 1

    def goal(self, frame):
        reference = self.reference_state(frame, jnp.zeros(2), 0.0)
        future = jax.tree.map(lambda x: jnp.stack((x, x)), reference)
        return self.codec.goal(reference, future)

    def context(self, state, world_id):
        return jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    def observe(self, state):
        physical = self.physical_state(state)
        return self.codec.sample(
            physical,
            state.last_action,
            jnp.array([1.0, 0.0, 0.0, 0.0]),
            state.joint_default_offset,
        )

    def reset_batch(self, count, rng):
        qpos = jnp.broadcast_to(self.reference_bank.qpos[0], (count, 8))
        data = ToyData(qpos, jnp.zeros((count, 7)), jnp.zeros((count, 1)))
        physical = jax.vmap(self.kinematics.from_qpos)(data.qpos, data.qvel)
        state = ToyState(
            data,
            jnp.zeros((count, 1)),
            physical.flatten()[:, None],
            jnp.zeros(count, jnp.int32),
            jnp.zeros(count),
            jnp.zeros((count, 2)),
            jnp.zeros(count),
            jnp.zeros((count, 1)),
            jnp.zeros(count, jnp.int32),
        )
        return state, jax.vmap(self.observe)(state)

    def step_batch(self, state, action, rng, *, reset):
        command = jax.vmap(self.mapping.command)(
            action, state.data.ctrl, state.joint_default_offset
        )
        qpos = state.data.qpos.at[:, 7:].add(self.control_timestep * command.accepted)
        qvel = state.data.qvel.at[:, 6:].set(command.accepted)
        data = ToyData(qpos, qvel, command.accepted)
        physical = jax.vmap(self.kinematics.from_qpos)(qpos, qvel)
        stepped = replace(
            state,
            data=data,
            last_action=command.action,
            actor_feedback_history=physical.flatten()[:, None],
            reference_time=state.reference_time + 1,
            episode_length=state.episode_length + 1,
        )
        terminal = stepped.episode_length == 2
        reward, _ = jax.vmap(
            lambda s, a, p: self.mdp.reward(
                s, self.reference_state(0.0, jnp.zeros(2), 0.0), a, p
            )
        )(physical, action, state.last_action)
        return (
            stepped,
            jax.vmap(self.observe)(stepped),
            reward,
            terminal,
            {
                "done/terminal": terminal,
                "done/timeout": jnp.zeros_like(terminal),
                "done/nonfinite": jnp.zeros_like(terminal),
            },
        )


@pytest.fixture
def toy_runner():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="root" pos="0 0 1"><freejoint/><geom size=".1"/><body name="arm" pos=".1 0 0"><joint name="joint" axis="0 1 0"/><geom size=".1"/></body></body></worldbody></mujoco>'
    )
    kin = Kinematics.from_model(
        model, ("joint",), ("root", "arm"), root_body_name="root"
    )
    state = kin.from_qpos(jnp.asarray(model.qpos0), jnp.zeros(model.nv))
    poses = kin.forward(state)

    def repeat(value):
        return jnp.repeat(jnp.asarray(value)[None], 20, axis=0)

    clip = MotionClip(
        50.0,
        ("joint",),
        ("root", "arm"),
        repeat(model.qpos0),
        repeat(np.zeros(model.nv)),
        repeat(poses.position),
        repeat(poses.quaternion),
        repeat(poses.linear_velocity),
        repeat(poses.angular_velocity),
        jnp.zeros((20, 2)),
    )
    bank = ReferenceBank.from_clips((("toy", clip, "train"),))
    limits = TargetLimits(
        np.array([[-2.0, 2.0]]), np.array([[-1.0, 1.0]]), np.eye(1), np.array([0.2])
    )
    description = SimpleNamespace(
        actuator_joint_names=("joint",),
        actuator_coordinate_transform=np.eye(1),
        policy_ctrl_indices=np.arange(1),
        target_limits=limits,
        default_joint_target=np.zeros(1),
        default_ctrl=np.zeros(1),
    )
    actuators = SimpleNamespace(
        by_name={"joint": SimpleNamespace(effort_limit=4.0, gains=(1.0, 0.1))}
    )
    mapping = ActionMapping.from_model(description, actuators)
    codec = ObservationCodec(
        ObservationConfig(history_frames=2, future_frames=2), jnp.zeros(1)
    )
    sampler = ReturnSampler.from_bank(bank)
    env = ToyEnv(
        bank,
        kin,
        codec,
        mapping,
        TrackingMDP(kin, jnp.array([-1.0]), jnp.array([1.0]), 0.02, TrackingConfig()),
        sampler,
        sampler.probabilities(sampler.initial_state(), 10.0),
    )
    config = LearningConfig(
        num_envs=2,
        total_timesteps=64,
        replay_steps=8,
        warmup_steps=2,
        window_length=2,
        model_warmup_updates=0,
        model_batch_size=4,
        model_batches=1,
        imagined_batch_size=2,
        imagination_horizon=2,
        critic_epochs=1,
        critic_minibatches=1,
    )
    networks = NetworkConfig(
        dynamics_widths=(8,), actor_widths=(8,), critic_widths=(8,), latent_size=4
    )
    learner = initialize_learner(env, config, networks, 6, jax.random.PRNGKey(0))
    runner = initialize_collection(env, config, learner, jax.random.PRNGKey(1))
    return env, config, runner


def test_compiled_collect_fit_imagine_loop_preserves_terminal_successors(toy_runner):
    env, config, runner = toy_runner
    collect, fit, improve = compile_updates(env, config)
    initial_actor = runner.learner.actor
    for _ in range(2):
        runner, metrics = collect(runner, env.reference_bank)
        require_valid_metrics(metrics)
    assert np.all(runner.collector.environment.episode_length == 0)
    assert np.all(runner.replay.data.terminal[1])
    assert np.any(np.asarray(runner.replay.data.successor.joint_position[1]) != 0)
    np.testing.assert_array_equal(runner.collector.environment.data.qpos[:, 7:], 0.0)
    runner, metrics = fit(runner)
    require_valid_metrics(metrics)
    runner, metrics = improve(runner, env.reference_bank)
    require_valid_metrics(metrics)
    assert runner.actor_updates == 1
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(
            jax.tree.leaves(initial_actor),
            jax.tree.leaves(runner.learner.actor),
            strict=True,
        )
    )


def test_nonfinite_metrics_stop_training():
    with pytest.raises(RuntimeError, match="model/loss"):
        require_valid_metrics({"model/loss": jnp.nan})
    with pytest.raises(RuntimeError, match="invalid/collection"):
        require_valid_metrics({"invalid/collection": 1.0})
