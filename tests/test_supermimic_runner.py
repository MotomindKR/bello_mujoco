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
from bello_mujoco.supermimic.collection import initialize_collection, maximum_return
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
    episode_return: jax.Array


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

    @property
    def max_episode_steps(self):
        return self.reference_bank.max_clip_frames - 1

    def physical_state(self, state):
        return self.kinematics.from_qpos(state.data.qpos, state.data.qvel)

    def physics_metrics(self, data):
        return {}

    def reference_state(self, frame, xy, yaw):
        sample = self.reference_bank.sample_global_frame(frame)
        return self.kinematics.from_qpos(sample.qpos, sample.qvel)

    def reference_bodies(self, frame, xy, yaw):
        return self.kinematics.forward(self.reference_state(frame, xy, yaw))

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

    def observation(self, state):
        return self.observe(state)

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
            jnp.zeros(count),
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
        stepped = replace(stepped, episode_return=state.episode_return + reward)
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


def test_collection_episode_cap_preserves_timeout_successor_and_resets(toy_runner):
    env, config, runner = toy_runner
    config = replace(config, max_episode_steps=1, gamma=1.0)
    collect, _, _ = compile_updates(env, config)
    result, metrics = collect(runner, env.reference_bank)
    require_valid_metrics(metrics)
    stored = result.replay.data
    assert np.asarray(stored.timeout[0]).all()
    assert not np.asarray(stored.terminal[0]).any()
    np.testing.assert_array_equal(result.collector.environment.episode_length, 0)
    np.testing.assert_array_equal(result.collector.episode_ids, 1)
    np.testing.assert_array_equal(stored.episode_step[0], 0)
    assert maximum_return(env, config) == pytest.approx(5 * env.control_timestep)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_episode_cap_rejects_invalid_steps(value):
    with pytest.raises(ValueError, match="max_episode_steps"):
        LearningConfig(max_episode_steps=value)


def test_collection_rejects_overflowed_physics_capacity():
    from bello_mujoco.envs.supermimic import BelloSuperMimicEnv

    data = SimpleNamespace(
        _impl=SimpleNamespace(
            nefc=jnp.array([3, 9]), njmax=8, nacon=jnp.array([12]), naconmax=10
        )
    )
    metrics = BelloSuperMimicEnv.physics_metrics(None, data)
    assert float(metrics["physics/max_constraints"]) == 9
    assert float(metrics["invalid/contact_capacity"]) == 1
    with pytest.raises(RuntimeError, match="constraint_capacity"):
        require_valid_metrics(metrics)


def test_collection_rejects_nonfinite_feedback_before_replay_learning(
    toy_runner, monkeypatch
):
    env, cfg, runner = toy_runner
    original = ToyEnv.step_batch

    def bad_feedback(self, *args, **kwargs):
        state, observation, reward, done, info = original(self, *args, **kwargs)
        state = replace(
            state,
            actor_feedback_history=state.actor_feedback_history.at[0, 0, 0].set(
                jnp.nan
            ),
        )
        return state, observation, reward, done, info

    monkeypatch.setattr(ToyEnv, "step_batch", bad_feedback)
    collect, _, _ = compile_updates(env, cfg)
    _, metrics = collect(runner, env.reference_bank)
    with pytest.raises(RuntimeError, match="invalid/collection"):
        require_valid_metrics(metrics)


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
    assert int(runner.learner.input_norm.count) == config.num_envs * 2
    assert int(runner.learner.output_norm.count) == config.num_envs * 2
    runner, metrics = fit(runner)
    require_valid_metrics(metrics)
    assert int(runner.learner.input_norm.count) == config.num_envs * 2
    assert int(runner.learner.output_norm.count) == config.num_envs * 2
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


def test_critic_has_reanchored_body_errors_without_actor_history(toy_runner):
    from bello_mujoco.supermimic.mdp import critic_features

    env, _, runner = toy_runner
    state = env.reference_state(jnp.asarray(0.0), jnp.zeros(2), 0.0)
    reference = replace(state, position=state.position + jnp.array([1.0, 2.0, 0.2]))
    context = jnp.array([0.0, 0.75, 0.0, 0.0, 0.0, 0.0])
    goal = env.goal(jnp.asarray(0.0))
    features = critic_features(state, reference, goal, context, env.kinematics)
    np.testing.assert_allclose(features[16:22], [0, 0, 0.2, 0, 0, 0.2], atol=1e-7)
    np.testing.assert_allclose(features[22:25], [1, 2, 0.2], atol=1e-7)
    np.testing.assert_allclose(features[31:37], 0)
    assert features.size == runner.learner.critic_norm.mean.size
    derivative = jax.grad(
        lambda z: critic_features(
            replace(state, position=state.position.at[2].set(z)),
            reference,
            goal,
            context,
            env.kinematics,
        )[16:22].sum()
    )(state.position[2])
    assert float(derivative) == pytest.approx(-2.0)


def test_collector_bootstraps_timeout_curriculum_but_preserves_replay_flags(
    toy_runner, monkeypatch
):
    env, config, runner = toy_runner
    original_step = ToyEnv.step_batch

    def timeout_step(self, *args, **kwargs):
        state, observation, reward, done, info = original_step(self, *args, **kwargs)
        info = {**info, "done/terminal": jnp.zeros_like(done), "done/timeout": done}
        return state, observation, reward, done, info

    monkeypatch.setattr(ToyEnv, "step_batch", timeout_step)
    collect, _, _ = compile_updates(env, config)
    for _ in range(2):
        runner, _ = collect(runner, env.reference_bank)
    rewards = np.asarray(runner.replay.data.reward[:2])
    expected = (rewards[0] + config.gamma * rewards[1]) + (
        config.gamma**2 * rewards.mean(axis=0) / (1 - config.gamma)
    )
    assert float(runner.sampling.returns[0]) == pytest.approx(
        env.sampler.config.return_alpha * expected.mean(), rel=1e-5
    )
    assert np.all(runner.replay.data.timeout[1])
    assert not np.any(runner.replay.data.terminal[1])


def test_reward_reanchors_tilted_reference_with_relative_rotation_yaw(toy_runner):
    from bello_mujoco.envs.geometry import yaw_quat
    from bello_mujoco.supermimic import rotations as rot

    env, _, _ = toy_runner
    state = env.reference_state(0.0, jnp.zeros(2), 0.0)
    reference = replace(state, quaternion=rot.exp(jnp.array([-0.2, 0.5, 0.4])))
    state = replace(state, quaternion=rot.exp(jnp.array([0.6, -0.3, 0.2])))
    alignment = yaw_quat(
        rot.multiply(state.quaternion, rot.conjugate(reference.quaternion))
    )
    targets = env.kinematics.forward(reference)
    bodies = env.kinematics.forward(state)
    expected = state.position + rot.rotate(
        alignment, targets.position - reference.position
    )
    _, terms = env.mdp.reward(state, reference, jnp.zeros(1), jnp.zeros(1))
    np.testing.assert_allclose(
        terms["body_position_squared"],
        jnp.mean(jnp.sum((bodies.position - expected) ** 2, axis=-1)),
        rtol=1e-5,
    )


def test_wall_budget_publishes_partial_final_checkpoint_after_actor_update(
    toy_runner, tmp_path, monkeypatch
):
    import json
    from bello_mujoco.configuration import supermimic_training_profile
    from bello_mujoco.supermimic import checkpoint
    from bello_mujoco.training import supermimic as training

    env, cfg, runner = toy_runner
    (tmp_path / "sonic-corpus.json").write_text(
        json.dumps({"clips": [{"array_bytes": 1}]})
    )
    run = replace(
        supermimic_training_profile(tmp_path),
        config=cfg,
        observations=env.codec.config,
        max_training_seconds=1,
    )
    clock = iter((0.0, 0.0, 2.0))
    monkeypatch.setattr(
        training, "time", SimpleNamespace(monotonic=lambda: next(clock))
    )
    monkeypatch.setattr(training, "capture_repository_commit", lambda: "0" * 40)
    monkeypatch.setattr(
        training.jax, "devices", lambda: [SimpleNamespace(platform="gpu")]
    )
    monkeypatch.setattr(training, "verify_corpus", lambda *a: "digest")
    monkeypatch.setattr(training, "load_environment", lambda *a, **k: env)
    monkeypatch.setattr(training, "initialize_learner", lambda *a, **k: runner.learner)
    monkeypatch.setattr(training, "initialize_collection", lambda *a, **k: runner)

    def collect(state, bank):
        return replace(
            state,
            collector=replace(
                state.collector, vector_steps=state.collector.vector_steps + 1
            ),
        ), {"invalid/collection": 0.0}

    def improve(state, bank):
        return replace(state, actor_updates=state.actor_updates + 1), {
            "invalid/actor_critic_update": 0.0
        }

    monkeypatch.setattr(
        training,
        "compile_updates",
        lambda *a: (collect, lambda s: (s, {"invalid/model_update": 0.0}), improve),
    )
    monkeypatch.setattr(training, "checkpoint_path", lambda *a: tmp_path / "saved")
    saved = []
    monkeypatch.setattr(
        checkpoint, "save_checkpoint", lambda p, s, *a, **k: saved.append((p, s))
    )
    training.run_training(run)
    [(path, restored)] = saved
    assert path == tmp_path / "saved/final"
    assert int(restored.actor_updates) == 1
    assert int(restored.collector.vector_steps) * cfg.num_envs < cfg.total_timesteps
