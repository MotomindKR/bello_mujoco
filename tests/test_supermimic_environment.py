from dataclasses import replace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from bello_mujoco.envs.motion_tracking import MotionTrackingEnvConfig, MotionResetConfig
from bello_mujoco.envs.supermimic import BelloSuperMimicEnv
from bello_mujoco.evaluation.motion import (
    make_native_motion_step,
    reset_native_motion_rollout,
)
from bello_mujoco.models.compiler import ModelConfig
from bello_mujoco.motion.clip import MotionClip, save_motion_clip
from bello_mujoco.robot_tasks import motion_task_definition
from bello_mujoco.sim import load_model
from bello_mujoco.supermimic.kinematics import Kinematics
from bello_mujoco.supermimic.mdp import imagined_return, lambda_returns
from bello_mujoco.supermimic.sampling import ReturnSampler, completed_return


@pytest.mark.parametrize("gamma", [0.99, 1.0])
def test_timeout_curriculum_return_is_independent_of_successful_tail_length(gamma):
    steps = jnp.array([1, 50, 500])
    reward = 0.1
    observed = (
        reward * (1 - gamma**steps) / (1 - gamma) if gamma < 1 else reward * steps
    )
    score = jax.jit(
        lambda terminal, timeout: completed_return(
            observed,
            reward * steps,
            steps,
            terminal,
            timeout,
            gamma=gamma,
            max_episode_steps=500,
        )
    )
    np.testing.assert_allclose(score(False, True), 10 if gamma < 1 else 50, rtol=1e-6)
    np.testing.assert_allclose(score(True, True), observed)
    np.testing.assert_allclose(score(True, False), observed)
    np.testing.assert_allclose(score(False, False), observed)


def test_imagined_return_and_lambda_targets_handle_failure_and_bootstrap():
    rewards = jnp.array([1.0, 2.0, 3.0])
    dones = jnp.array([0.0, 1.0, 0.0])
    values = jnp.array([10.0, 20.0, 30.0, 40.0])
    assert imagined_return(rewards, dones, 40.0, 0.5) == 2.0
    np.testing.assert_allclose(
        lambda_returns(rewards, dones, values, 0.5, 1.0), [2.0, 2.0, 23.0]
    )
    np.testing.assert_allclose(
        lambda_returns(rewards, dones, values, 0.5, 0.0), [11.0, 2.0, 23.0]
    )
    soft = jnp.array([0.1, 0.2, 0.3])
    expected = (
        1.0 + 0.5 * 0.9 * 2.0 + 0.25 * 0.9 * 0.8 * 3.0 + 0.125 * 0.9 * 0.8 * 0.7 * 40.0
    )
    assert float(imagined_return(rewards, soft, 40.0, 0.5)) == pytest.approx(expected)
    derivative = jax.grad(lambda final: imagined_return(rewards, soft, final, 0.5))(
        40.0
    )
    assert derivative == pytest.approx(0.125 * 0.9 * 0.8 * 0.7)


def test_return_sampler_excludes_held_out_bins_and_preserves_final_bin_width():
    from types import SimpleNamespace

    bank = SimpleNamespace(
        fps=50.0,
        time_offsets=np.array([0, 64]),
        lengths=np.array([63, 70]),
        split_ids=np.array([0, 1]),
    )
    sampler = ReturnSampler.from_bank(bank)
    np.testing.assert_array_equal(sampler.starts, [0, 50, 64, 114])
    np.testing.assert_array_equal(sampler.lengths, [50, 12, 50, 19])
    state = sampler.initial_state()
    state = sampler.update(
        state,
        jnp.array([0, 1, 1]),
        jnp.array([10.0, 0.0, 2.0]),
        jnp.array([True, True, True]),
        10.0,
    )
    probabilities = sampler.probabilities(state, 10.0)
    np.testing.assert_allclose(state.returns, [2.0, 0.2, 0.0, 0.0])
    expected_past = (
        0.999 * sampler.initial_state().past_weights
        + 0.001 * jax.nn.softmax(jnp.array([-5.0, -0.5, -jnp.inf, -jnp.inf]))
    )
    np.testing.assert_allclose(state.past_weights, expected_past)
    assert probabilities[1] > probabilities[0]
    np.testing.assert_array_equal(probabilities[2:], [0.0, 0.0])
    draws = jax.jit(jax.vmap(lambda k: sampler.sample(k, probabilities)))(
        jax.random.split(jax.random.key(0), 500)
    )
    assert draws.max() <= 61 and draws.min() >= 0
    assert np.unique(np.asarray(draws)[np.asarray(draws) >= 50]).size == 12
    unchanged = sampler.update(
        state, jnp.array([0]), jnp.array([0.0]), jnp.array([False]), 10.0
    )
    np.testing.assert_array_equal(unchanged.past_weights, state.past_weights)


@pytest.fixture
def mini_corpus(tmp_path):
    asset = ModelConfig(asset="bello_mini", actuator_mode="policy")
    model = load_model(asset, load_mjx=False)
    bodies = motion_task_definition("bello_mini").beyondmimic_tracked_body_names
    kin = Kinematics.from_model(
        model.mj_model, model.actuator_joint_names, bodies, root_body_name="pelvis_link"
    )
    physical = kin.from_qpos(
        jnp.asarray(model.default_qpos), jnp.asarray(model.default_qvel)
    )
    poses = kin.forward(physical)
    frames = 12

    def repeat(value):
        return np.repeat(np.asarray(value)[None], frames, axis=0)

    clip = MotionClip(
        50.0,
        model.actuator_joint_names,
        bodies,
        repeat(model.default_qpos),
        repeat(model.default_qvel),
        repeat(poses.position),
        repeat(poses.quaternion),
        repeat(poses.linear_velocity),
        repeat(poses.angular_velocity),
        np.zeros((frames, 2)),
    )
    path = tmp_path / "standing.npz"
    save_motion_clip(path, clip)
    return asset, path


def test_filtered_gpu_training_is_rejected_before_loading_assets():
    from bello_mujoco.envs.feedback import ObservationLowPassFilterConfig

    config = replace(
        MotionTrackingEnvConfig(),
        observation_low_pass_filter=ObservationLowPassFilterConfig(enabled=True),
    )
    with pytest.raises(ValueError, match="filtered SuperMimic MJX-Warp"):
        BelloSuperMimicEnv.load_corpus(
            ModelConfig(asset="bello_mini", actuator_mode="policy"),
            config,
            (),
            load_mjx=True,
        )


@pytest.mark.integration
def test_mini_reset_motor_target_tracks_the_constrained_reference_pose(mini_corpus):
    asset, path = mini_corpus
    env = BelloSuperMimicEnv.load_corpus(
        asset, MotionTrackingEnvConfig(), (path,), load_mjx=False
    )
    command = env.mapping.command(
        jnp.full(env.action_size, 0.02),
        jnp.zeros(env.action_size),
    )
    qpos = (
        jnp.asarray(env.model.default_qpos)
        .at[env.model.actuator_qpos_indices]
        .set(jnp.asarray(env.model.target_limits.joint_from_motor) @ command.accepted)
    )
    target = jax.jit(env._initial_ctrl)(jnp.zeros(env.action_size), qpos)
    np.testing.assert_allclose(target, command.accepted, atol=1e-6)
    assert np.max(np.abs(np.asarray(target - env.mapping.neutral_motor))) > 1e-6


@pytest.mark.integration
def test_native_raw_pd_targets_keep_physical_force_limits(mini_corpus):
    asset, path = mini_corpus
    env = BelloSuperMimicEnv.load_corpus(
        asset, MotionTrackingEnvConfig(), (path,), load_mjx=False
    )
    data = mujoco.MjData(env.model.mj_model)
    capacities = {"contact_capacity": 4096, "force_capacity": 65536}
    state, _ = reset_native_motion_rollout(env, data, **capacities)
    action = jnp.full(env.action_size, 2 * env.action_clip)
    expected = env.mapping.command(action, state.joint_default_offset).accepted
    state, _, _, _, info = make_native_motion_step(env, **capacities)(
        data, state, action
    )
    np.testing.assert_array_equal(state.last_action, action)
    np.testing.assert_allclose(data.ctrl, expected, atol=1e-6)
    assert not bool(info["done/nonfinite"])
    assert np.all(
        np.abs(data.actuator_force) <= env.model.actuator_peak_force * (1 + 1e-6)
    )


@pytest.mark.integration
def test_native_mini_collection_uses_new_codec_reward_and_terminal_successors(
    mini_corpus,
):
    asset, path = mini_corpus
    config = MotionTrackingEnvConfig(reset=MotionResetConfig(random_start=False))
    env = BelloSuperMimicEnv.load_corpus(asset, config, (path,), load_mjx=False)
    data = mujoco.MjData(env.model.mj_model)
    state, sample = reset_native_motion_rollout(env, data)
    assert sample.shape == (env.codec.sample_size,)
    critic = env.critic_observation(state)
    assert critic.shape == (env.critic_observation_size,)
    assert np.isfinite(critic).all()
    assert env.goal(state.reference_time).shape == (env.codec.goal_size,)
    step = make_native_motion_step(env)
    first = state
    for _ in range(3):
        state, sample, reward, done, info = step(
            data, state, jnp.zeros(env.action_size)
        )
        assert np.isfinite(reward) and np.isfinite(sample).all()
    assert state.reference_time == 3.0
    expected_reward, _ = env.mdp.reward(
        env.physical_state(state),
        env.reference_state(
            state.reference_time, state.alignment_xy, state.alignment_yaw
        ),
        state.last_action,
        first.last_action,
    )
    np.testing.assert_allclose(expected_reward, reward, atol=1e-6)
    terminal = replace(
        env.physical_state(state),
        position=env.physical_state(state).position + jnp.array([0.0, 0.0, 1.0]),
    )
    assert env.mdp.terminal(
        terminal,
        env.reference_state(
            state.reference_time, state.alignment_xy, state.alignment_yaw
        ),
    )


@pytest.mark.integration
@pytest.mark.parametrize("precision", ["float32", "bfloat16"])
def test_cuda_mini_compiled_collect_fit_and_improve(mini_corpus, precision):
    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("requires CUDA/MJX-Warp")
    from bello_mujoco.configuration import supermimic_training_profile
    from bello_mujoco.supermimic.collection import initialize_collection
    from bello_mujoco.supermimic.optimization import initialize_learner
    from bello_mujoco.training.supermimic import (
        compile_updates,
        load_environment,
        require_valid_metrics,
    )

    _, path = mini_corpus
    run = supermimic_training_profile(path.parent, name="bello_mini_supermimic_smoke")
    cfg = replace(
        run.config,
        num_envs=2,
        warmup_steps=4,
        window_length=2,
        model_batch_size=4,
        imagined_batch_size=2,
        imagination_horizon=2,
    )
    run = replace(
        run, config=cfg, networks=replace(run.networks, matmul_precision=precision)
    )
    env = load_environment(run, load_mjx=True)
    learner = initialize_learner(
        env, cfg, run.networks, env.action_size + 5, jax.random.PRNGKey(0)
    )
    runner = initialize_collection(env, cfg, learner, jax.random.PRNGKey(1))
    collect, fit, improve = compile_updates(env, cfg)
    for _ in range(8):
        runner, metrics = collect(runner, env.reference_bank)
        require_valid_metrics(metrics)
    runner, metrics = fit(runner)
    require_valid_metrics(metrics)
    runner, metrics = improve(runner, env.reference_bank)
    require_valid_metrics(metrics)
    assert int(runner.actor_updates) == 1


@pytest.mark.integration
@pytest.mark.parametrize("precision", ["float32", "bfloat16"])
def test_mini_checkpoint_actor_restores_exports_and_evaluates_held_out_clip(
    mini_corpus, monkeypatch, tmp_path, request, precision
):
    from types import SimpleNamespace
    from zipfile import ZipFile
    import json
    import os
    import shutil
    import subprocess
    import onnx
    from onnx.reference import ReferenceEvaluator

    from bello_mujoco.configuration import supermimic_training_profile
    from bello_mujoco.deployment import ExportConfig, export_checkpoint
    from bello_mujoco.evaluation import supermimic as evaluation
    from bello_mujoco.sonic.corpus import prepare_corpus, verify_corpus
    from bello_mujoco.supermimic import checkpoint, deployment
    from bello_mujoco.supermimic.collection import CollectorState, RunnerState
    from bello_mujoco.supermimic.optimization import initialize_learner
    from bello_mujoco.training.supermimic import load_environment

    _, path = mini_corpus
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    shutil.copyfile(path, corpus / path.name)
    path = corpus / path.name
    held_out = path.with_stem("held_out")
    shutil.copyfile(path, held_out)
    prepare_corpus(
        path.parent,
        {
            path.stem: {"split": "train", "source_sequence": "training"},
            held_out.stem: {"split": "evaluation", "source_sequence": "evaluation"},
        },
    )
    run = supermimic_training_profile(path.parent, name="bello_mini_supermimic_smoke")
    run = replace(
        run,
        dataset=replace(run.dataset, evaluation_clips=(held_out.stem,)),
        networks=replace(run.networks, matmul_precision=precision),
    )
    env = load_environment(run, load_mjx=False)
    learner = initialize_learner(
        env, run.config, run.networks, env.action_size + 5, jax.random.PRNGKey(1)
    )
    # Actor-only consumers do not need a simulator/replay allocation.
    collector = CollectorState(*([jnp.zeros(1)] * 6), vector_steps=jnp.asarray(4))
    runner = RunnerState(
        learner,
        collector,
        None,
        env.sampler.initial_state(),
        jax.random.PRNGKey(2),
        jnp.asarray(1),
        jnp.asarray(1),
    )
    source_checks = []
    commit = "0" * 40
    native = None
    if os.environ.get("BELLO_POLICY_BIN"):
        native, _, source = request.getfixturevalue("rust_policy")
        commit = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    monkeypatch.setattr(checkpoint, "require_repository_commit", source_checks.append)
    monkeypatch.setattr(deployment, "require_repository_commit", source_checks.append)
    monkeypatch.setattr(deployment, "capture_repository_commit", lambda: commit)
    saved = tmp_path / "checkpoint"
    checkpoint.save_checkpoint(
        saved,
        runner,
        run,
        git_commit=commit,
        dataset_digest=verify_corpus(path.parent, (held_out.stem,)),
    )
    bundle = export_checkpoint(
        ExportConfig(
            checkpoint_path=saved,
            output_path=tmp_path / "policy.zip",
            motion_dataset=path.parent,
            reference_clips=(held_out.stem,),
        )
    )
    if native is not None:
        result = native(content=bundle.read_bytes(), check=True, clip=held_out.stem)
        assert result.returncode == 0, result.stderr
    restored, actor, obs_norm, goal_norm, _ = checkpoint.load_actor(
        saved, dataset_directory=path.parent, reference_clips=(held_out.stem,)
    )
    data = mujoco.MjData(restored.model.mj_model)
    state, observation = reset_native_motion_rollout(restored, data)
    fixed_data = mujoco.MjData(restored.model.mj_model)
    fixed, _ = reset_native_motion_rollout(
        restored, fixed_data, contact_capacity=4096, force_capacity=65536
    )
    assert fixed.data._impl.contact__geom.shape == (4096, 2)
    assert fixed.data._impl.efc__force.shape == (65536,)
    step_default = make_native_motion_step(restored)
    step_fixed = make_native_motion_step(
        restored, contact_capacity=4096, force_capacity=65536
    )
    action = jnp.zeros(restored.action_size)
    advanced, _, reward, _, _ = step_default(data, state, action)
    fixed_advanced, _, fixed_reward, _, _ = step_fixed(fixed_data, fixed, action)
    np.testing.assert_array_equal(data.qpos, fixed_data.qpos)
    np.testing.assert_array_equal(data.qvel, fixed_data.qvel)
    np.testing.assert_allclose(fixed_reward, reward, atol=1e-6)
    np.testing.assert_array_equal(
        fixed_advanced.data._impl.nacon, advanced.data._impl.nacon
    )
    np.testing.assert_allclose(
        fixed_advanced.data._impl.efc__force[: data.nefc], data.efc_force
    )
    with pytest.raises(ValueError, match="contact_capacity"):
        reset_native_motion_rollout(restored, fixed_data, contact_capacity=0)
    with pytest.raises(ValueError, match="force_capacity"):
        reset_native_motion_rollout(restored, fixed_data, force_capacity=0)
    state, observation = reset_native_motion_rollout(restored, data)
    visible, *_ = checkpoint.load_actor(
        saved,
        dataset_directory=path.parent,
        reference_clips=(held_out.stem,),
        use_viewer_model=True,
    )
    visual_data = mujoco.MjData(visible.model.mj_model)
    visual_data.qpos[:], visual_data.qvel[:], visual_data.ctrl[:] = (
        data.qpos,
        data.qvel,
        data.ctrl,
    )
    mujoco.mj_forward(restored.model.mj_model, data)
    mujoco.mj_forward(visible.model.mj_model, visual_data)
    np.testing.assert_allclose(visual_data.qacc, data.qacc, atol=1e-7, rtol=1e-7)
    history = restored.codec.initial_history(observation)
    goal = restored.goal(state.reference_time)
    with ZipFile(bundle) as archive:
        assert set(archive.namelist()) == {
            "policy.onnx",
            "physics.xml",
            "clips.json",
            "clips/held_out.npz",
        }
        graph = onnx.load_from_string(archive.read("policy.onnx"))
        mujoco.MjModel.from_xml_string(archive.read("physics.xml").decode())
    output = ReferenceEvaluator(graph).run(
        None,
        {
            "observation": np.asarray(jnp.concatenate((history.reshape(-1), goal)))[
                None
            ],
        },
    )
    np.testing.assert_allclose(
        output[0][0], actor(history, goal, obs_norm, goal_norm), atol=2e-6
    )
    closed = []

    def visible_video_resources(model, *args, **kwargs):
        assert np.any(model.geom_rgba[model.geom_bodyid > 0, 3] > 0)
        return SimpleNamespace(close=lambda: closed.append(True)), None, None, None

    monkeypatch.setattr(
        evaluation,
        "open_rollout_video_resources",
        visible_video_resources,
    )
    monkeypatch.setattr(evaluation, "write_video_frame", lambda *a, **k: None)
    report = evaluation.render_supermimic_checkpoint(
        saved,
        path.parent,
        (held_out.stem,),
        tmp_path / "evaluation",
        steps=3,
    )
    row = report["clips"][held_out.stem]
    assert closed == [True]
    assert row["rendered_steps"] == 3
    assert row["normalized_return"] == pytest.approx(
        row["episode_return"] / (5 * restored.control_timestep * row["planned_steps"])
    )
    assert json.loads((tmp_path / "evaluation/summary.json").read_text()) == report
    monkeypatch.setattr(
        evaluation,
        "open_rollout_video_resources",
        lambda *a, **k: pytest.fail("headless evaluation opened a renderer"),
    )
    headless = evaluation.render_supermimic_checkpoint(
        saved,
        path.parent,
        (held_out.stem,),
        tmp_path / "headless",
        steps=3,
        video_clips=(),
    )
    headless_row = headless["clips"][held_out.stem]
    assert headless_row["rendered_steps"] == 0
    for key in row:
        if key in ("motor_diagnostics", "failure_event"):
            assert headless_row[key] == row[key]
        else:
            assert headless_row[key] == pytest.approx(
                0 if key == "rendered_steps" else row[key]
            )
    assert set(headless_row["motor_diagnostics"]) == set(
        restored.model.actuator_coordinate_names
    )

    from bello_mujoco.evaluation.world_model_audit import (
        MotionWindow,
        WorldModelAuditConfig,
        audit_world_model,
    )

    model_report = audit_world_model(
        WorldModelAuditConfig(
            saved,
            path.parent,
            tmp_path / "model-audit.json",
            (MotionWindow(held_out.stem, start_frame=1, max_steps=3),),
            horizons=(1, 8),
            anchor_stride=1,
        )
    )
    window = model_report["windows"][f"{held_out.stem}:1"]
    assert window["recorded_split"] == "evaluation"
    assert window["observed_steps"] == window["tracking"]["planned_steps"] == 3
    assert window["horizons"]["1"]["all"]["finite_windows"] == 3
    assert window["horizons"]["8"]["all"]["windows"] == 0
    assert window["tracking"]["normalized_return"] == pytest.approx(
        window["tracking"]["episode_return"] / (5 * restored.control_timestep * 3)
    )
    assert source_checks and all(value == commit for value in source_checks)
    with pytest.raises(ValueError, match="recorded held_out split"):
        evaluation.render_supermimic_checkpoint(
            saved,
            path.parent,
            (path.stem,),
            tmp_path / "invalid",
        )


@pytest.mark.integration
def test_environment_rejects_unrecorded_dynamics_randomization(mini_corpus):
    asset, path = mini_corpus
    config = MotionTrackingEnvConfig()
    config = replace(
        config,
        model_randomization=replace(
            config.model_randomization, enabled=True, joint_origin_range=(-0.01, 0.01)
        ),
    )
    with pytest.raises(ValueError, match="context currently supports"):
        BelloSuperMimicEnv.load_corpus(asset, config, (path,), load_mjx=False)


@pytest.mark.integration
def test_mini_imagination_updates_actor_and_critic_without_updating_dynamics(
    mini_corpus,
):
    import equinox as eqx
    from bello_mujoco.supermimic.config import LearningConfig, ObservationConfig
    from bello_mujoco.supermimic.model import NetworkConfig
    from bello_mujoco.supermimic.optimization import (
        initialize_learner,
        update_actor_critic,
    )
    from bello_mujoco.supermimic.replay import ReplayStart, Transition
    from bello_mujoco.supermimic.state import PhysicalState

    asset, path = mini_corpus
    env = BelloSuperMimicEnv.load_corpus(
        asset,
        MotionTrackingEnvConfig(),
        (path,),
        observation_config=ObservationConfig(history_frames=2, future_frames=2),
        load_mjx=False,
    )
    data = mujoco.MjData(env.model.mj_model)
    state, observation = reset_native_motion_rollout(env, data)
    physical = env.physical_state(state)
    context = env.context(state, 0)
    transition = Transition(
        state=physical,
        successor=physical,
        filtered_feedback=PhysicalState.unflatten(
            state.actor_feedback_history[0], env.action_size
        ),
        previous_motor=state.data.ctrl,
        motor=state.data.ctrl,
        action=state.last_action,
        previous_action=state.last_action,
        observation=observation,
        context=context,
        alignment_xy=state.alignment_xy,
        alignment_yaw=state.alignment_yaw,
        reference_frame=state.reference_time,
        clip_id=jnp.asarray(0),
        episode_id=jnp.asarray(0),
        episode_step=jnp.asarray(0),
        remaining_steps=jnp.asarray(11),
        reward=jnp.asarray(0.0),
        terminal=jnp.asarray(False),
        timeout=jnp.asarray(False),
    )
    starts = ReplayStart(
        jax.tree.map(lambda x: x[None], transition),
        env.codec.initial_history(observation)[None],
        PhysicalState.unflatten(state.actor_feedback_history[None], env.action_size),
        jnp.asarray(True),
    )
    config = LearningConfig(
        num_envs=1,
        total_timesteps=100,
        replay_steps=8,
        warmup_steps=2,
        window_length=2,
        model_batches=1,
        model_batch_size=2,
        imagined_batch_size=1,
        imagination_horizon=2,
        critic_epochs=1,
        critic_minibatches=1,
    )
    networks = NetworkConfig(
        dynamics_widths=(8,), actor_widths=(8,), critic_widths=(8,), latent_size=4
    )
    learner = initialize_learner(
        env, config, networks, context.size, jax.random.key(12)
    )
    update = eqx.filter_jit(
        lambda value, start, key: update_actor_critic(value, start, env, config, 0, key)
    )
    updated, metrics = update(learner, starts, jax.random.key(13))
    assert metrics["invalid/actor_critic_update"] == 0
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(metrics))
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(
            jax.tree.leaves(updated.actor), jax.tree.leaves(learner.actor), strict=True
        )
    )
    for a, b in zip(
        jax.tree.leaves(updated.model), jax.tree.leaves(learner.model), strict=True
    ):
        np.testing.assert_array_equal(a, b)
