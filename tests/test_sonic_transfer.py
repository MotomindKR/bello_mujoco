from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import equinox as eqx
from onnx.reference import ReferenceEvaluator
from types import SimpleNamespace
from typing import NamedTuple

from bello_mujoco.deployment import sonic_actor_onnx
from bello_mujoco.configuration import sonic_transfer_training_profile
from bello_mujoco.models.compiler import ModelConfig
from bello_mujoco.motion.clip import MotionClip
from bello_mujoco.policy import evaluate_actions
from bello_mujoco.sonic.embodiment import (
    G1_JOINT_NAMES,
    MiniEmbodimentAdapter,
    g1_neutral_positions,
)
from bello_mujoco.sonic.model import (
    LoRALinear,
    SonicActorCritic,
    SonicMLP,
    finite_scalar_quantize,
)
from bello_mujoco.sonic.observations import SonicObservationAssembler
from bello_mujoco.sonic.reference import ReferenceBank
from bello_mujoco.sonic.sampling import (
    SonicSampler,
    SonicSamplingConfig,
    SonicTaskState,
)
from bello_mujoco.envs.sonic_transfer import BelloSonicTransferEnv
from bello_mujoco.envs.motion_tracking import BelloMotionTrackingEnv
from bello_mujoco.training.ppo_config import PPOConfig, parameter_learning_rate
from bello_mujoco.training.normalization import (
    init_observation_norm_state,
    init_value_norm_state,
    normalize_critic_observation,
    update_critic_observation_norm_state,
)


from bello_mujoco.training.ppo import make_optimizer, update_ppo
from bello_mujoco.training.ppo_types import RunnerState
from bello_mujoco.training.ppo_types import TrainState, Transition
from bello_mujoco.training.sonic_transfer import (
    SonicDatasetConfig,
    SonicPretrainedConfig,
    SonicTransferTrainingRunConfig,
    _restore_checkpoint,
    _save_checkpoint,
    compile_sonic_update,
)


def test_any2any_critic_normalizer_updates_after_forward_only_during_optimization():
    config = PPOConfig(
        critic_normalization_mode="any2any", normalize_observations=False
    )
    state = init_observation_norm_state(2)._replace(count=jnp.asarray(1.0))
    values = jnp.asarray([[2.0, 4.0], [4.0, 8.0]])
    np.testing.assert_allclose(
        normalize_critic_observation(values, state, config),
        np.clip(np.asarray(values) / np.sqrt(1.0 + 1e-5), -5, 5),
    )
    assert update_critic_observation_norm_state(state, values, config) is state
    updated = update_critic_observation_norm_state(
        state, values, config, during_optimization=True
    )
    np.testing.assert_allclose(updated.mean, [2.0, 4.0])
    np.testing.assert_allclose(
        updated.var, [(1 + 2 * 2 + 9 * 2 / 3) / 3, (1 + 8 * 2 + 36 * 2 / 3) / 3]
    )
    assert float(updated.count) == 3.0


def synthetic_adapter(transform: np.ndarray | None = None) -> MiniEmbodimentAdapter:
    source_names = tuple(name for name in G1_JOINT_NAMES if name != "waist_pitch_joint")
    target_names = tuple(
        name.replace("_elbow_joint", "_elbow_pitch_joint") for name in source_names
    )
    source_indices = np.asarray(
        [G1_JOINT_NAMES.index(name) for name in source_names], dtype=np.int32
    )
    source_to_target = np.full(len(G1_JOINT_NAMES), -1, dtype=np.int32)
    source_to_target[source_indices] = np.arange(len(target_names))
    if transform is None:
        transform = np.eye(len(target_names), dtype=np.float32)
    return MiniEmbodimentAdapter(
        target_joint_names=target_names,
        target_neutral=np.linspace(-0.2, 0.2, len(target_names), dtype=np.float32),
        target_to_source_indices=source_indices,
        source_to_target_indices=source_to_target,
        signs=np.ones(len(target_names), dtype=np.float32),
        source_neutral=g1_neutral_positions(),
        motor_from_joint=transform,
    )


def synthetic_clip(name_seed: float, frames: int = 4) -> MotionClip:
    adapter = synthetic_adapter()
    qpos = np.zeros((frames, 7 + adapter.target_size), dtype=np.float32)
    qpos[:, 0] = name_seed + np.arange(frames)
    qpos[:, 3] = 1.0
    qvel = np.zeros((frames, 6 + adapter.target_size), dtype=np.float32)
    body_pos = np.zeros((frames, 1, 3), dtype=np.float32)
    body_pos[:, 0, 0] = name_seed + np.arange(frames)
    body_quat = np.zeros((frames, 1, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    return MotionClip(
        fps=50.0,
        joint_names=adapter.target_joint_names,
        body_names=("pelvis",),
        qpos=jnp.asarray(qpos),
        qvel=jnp.asarray(qvel),
        body_pos=jnp.asarray(body_pos),
        body_quat=jnp.asarray(body_quat),
        body_lin_vel=jnp.zeros((frames, 1, 3)),
        body_ang_vel=jnp.zeros((frames, 1, 3)),
        foot_contacts=jnp.zeros((frames, 2)),
    )


def test_reference_windows_clamp_inside_each_clip() -> None:
    bank = ReferenceBank.from_clips(
        (
            ("train", synthetic_clip(0.0), "train"),
            ("held_out", synthetic_clip(100.0), "evaluation"),
        )
    )

    window = bank.window(jnp.asarray(0), jnp.asarray(3), jnp.arange(4))

    np.testing.assert_allclose(window.qpos[:, 0], np.full(4, 3.0))
    assert bank.offsets.tolist() == [0, 4]
    assert bank.time_offsets.tolist() == [0, 5]
    assert bank.lengths.tolist() == [4, 4]
    np.testing.assert_allclose(
        bank.clip_sampling_probabilities(jnp.asarray([1.0, 20.0]), split="train"),
        [1.0, 0.0],
    )
    selected = jax.jit(
        lambda key, failure: bank.sample_clip(key, failure, split="train")
    )(jax.random.PRNGKey(0), jnp.asarray([1.0, 20.0]))
    assert int(selected) == 0


def test_compiled_sonic_update_passes_bank_as_dynamic_buffers(monkeypatch):
    bank = ReferenceBank.from_clips((("train", synthetic_clip(0.0), "train"),))

    @dataclass(frozen=True)
    class Environment:
        reference_bank: ReferenceBank
        sampler: SonicSampler

    class State(NamedTuple):
        value: jax.Array
        task_state: SonicTaskState

    def fake_update(state, static, optimizer, env, config):
        return state._replace(value=state.value + env.reference_bank.qpos[0, 0]), {}

    monkeypatch.setattr(
        "bello_mujoco.training.sonic_transfer.update_runner_state", fake_update
    )
    sampler = SonicSampler.from_bank(bank, SonicSamplingConfig())
    state = State(
        jnp.asarray(1.0), SonicTaskState(jnp.zeros((3, 100)), sampler.initial_state())
    )
    step = compile_sonic_update(Environment(bank, sampler), None, None, None)
    assert float(step(state)[0].value) == 1.0
    shifted = replace(bank, qpos=bank.qpos + 10.0)
    assert float(step.func(state, shifted)[0].value) == 11.0
    traced = jax.make_jaxpr(step.func)(state, bank)
    assert all(np.asarray(value).size < bank.qpos.size for value in traced.consts)
    assert len(jax.tree_util.tree_leaves(bank)) == 11
    assert bank.max_clip_frames == 4
    assert bank.clock_end == 4


def test_reference_encoder_preserves_release_blockwise_joint_layout() -> None:
    adapter = synthetic_adapter()
    clip = synthetic_clip(0.0, frames=20)
    qpos = np.asarray(clip.qpos).copy()
    qvel = np.asarray(clip.qvel).copy()
    for frame in range(20):
        qpos[frame, 7:] = frame * 100.0 + np.arange(adapter.target_size)
        qvel[frame, 6:] = 10_000.0 + frame * 100.0 + np.arange(adapter.target_size)
    clip = replace(clip, qpos=jnp.asarray(qpos), qvel=jnp.asarray(qvel))
    bank = ReferenceBank.from_clips((("walk", clip, "train"),))

    reference = (
        SonicObservationAssembler(adapter)
        .reference(
            bank,
            jnp.asarray(0),
            jnp.asarray(0.0),
            jnp.asarray([1.0, 0.0, 0.0, 0.0]),
        )
        .reshape(10, 64)
    )
    sample_frames = np.asarray([0, 5, 10, 15, 19, 19, 19, 19, 19, 19])
    positions = np.stack(
        [
            np.asarray(adapter.target_position_to_source(qpos[frame, 7:]))
            for frame in sample_frames
        ]
    )
    velocities = np.stack(
        [
            np.asarray(adapter.target_vector_to_source(qvel[frame, 6:]))
            for frame in sample_frames
        ]
    )
    expected_command = np.concatenate((positions.reshape(-1), velocities.reshape(-1)))

    np.testing.assert_allclose(reference[:, :58].reshape(-1), expected_command)
    np.testing.assert_allclose(
        reference[:, 58:64], np.tile([1, 0, 0, 1, 0, 0], (10, 1))
    )


def test_embodiment_codec_fills_only_waist_pitch_and_round_trips_transmission() -> None:
    transform = np.eye(28, dtype=np.float32)
    transform[4:6, 4:6] = np.asarray([[1.0, 1.0], [1.0, -1.0]])
    adapter = synthetic_adapter(transform)
    target = jnp.linspace(-0.5, 0.5, adapter.target_size)

    source = adapter.target_position_to_source(target)
    recovered_action = adapter.source_action_to_target(
        adapter.target_vector_to_source(target)
    )
    motor = adapter.logical_target_to_motor(target)

    assert source.shape == (29,)
    assert source[G1_JOINT_NAMES.index("waist_pitch_joint")] == pytest.approx(0.0)
    np.testing.assert_allclose(recovered_action, target)
    np.testing.assert_allclose(adapter.motor_feedback_to_logical(motor), target)

    signed = replace(adapter, signs=-np.ones(adapter.target_size, dtype=np.float32))
    source_scale = jnp.linspace(0.1, 2.9, signed.source_size)
    np.testing.assert_allclose(
        signed.source_scale_to_target(source_scale),
        np.asarray(source_scale)[signed.target_to_source_indices],
    )


def test_source_joint_order_matches_pinned_any2any_bridge() -> None:
    assert G1_JOINT_NAMES[:11] == (
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "waist_yaw_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "waist_roll_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "waist_pitch_joint",
        "left_knee_joint",
        "right_knee_joint",
    )
    assert G1_JOINT_NAMES[-6:] == (
        "left_wrist_roll_joint",
        "right_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "right_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_yaw_joint",
    )


def test_zero_lora_preserves_base_and_only_adapters_receive_gradients() -> None:
    weight = np.arange(12, dtype=np.float32).reshape(3, 4) / 10.0
    bias = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
    layer = LoRALinear(
        weight,
        bias,
        rank=2,
        alpha=4.0,
        key=jax.random.PRNGKey(3),
    )
    value = jnp.asarray([0.5, -1.0, 0.25, 2.0])

    np.testing.assert_allclose(layer(value), value @ weight.T + bias, rtol=1e-6)
    gradients = jax.grad(lambda module: jnp.sum(module(value)))(layer)
    np.testing.assert_allclose(gradients.weight, 0.0)
    np.testing.assert_allclose(gradients.bias, 0.0)
    assert np.linalg.norm(np.asarray(gradients.lora_b)) > 0.0


def test_lora_uses_float32_matmul_precision_independent_of_process_defaults() -> None:
    layer = LoRALinear(
        np.ones((3, 4), dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        rank=2,
        alpha=4.0,
        key=jax.random.PRNGKey(3),
    )
    with jax.default_matmul_precision("bfloat16"):
        graphs = (
            jax.make_jaxpr(layer)(jnp.ones((2, 4))),
            jax.make_jaxpr(layer.merged_weight)(),
        )
    products = [
        equation
        for graph in graphs
        for equation in graph.jaxpr.eqns
        if equation.primitive.name == "dot_general"
    ]
    assert len(products) == 4
    assert all(
        equation.params["precision"]
        == (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)
        for equation in products
    )


def test_fsq_uses_release_even_level_offset() -> None:
    inputs = jnp.asarray([-100.0, 0.0, 100.0])
    values = finite_scalar_quantize(inputs, levels=32)
    np.testing.assert_allclose(values, [-1.0, 0.0, 0.9375], atol=1.0e-6)
    half_level = 31.0 * (1.0 - 1.0e-3) / 2.0
    shift = np.arctanh(0.5 / half_level)
    expected_gradient = (
        half_level * (1.0 - np.tanh(np.asarray(inputs) + shift) ** 2) / 16.0
    )
    gradient = jax.grad(lambda value: finite_scalar_quantize(value, 32).sum())(inputs)
    np.testing.assert_allclose(gradient, expected_gradient, atol=1.0e-6)


def test_reference_time_separator_never_interpolates_into_next_clip() -> None:
    bank = ReferenceBank.from_clips(
        (
            ("train", synthetic_clip(0.0), "train"),
            ("held_out", synthetic_clip(100.0), "evaluation"),
        )
    )

    in_separator = bank.sample_time(jnp.asarray(4.25 / bank.fps))
    next_start = bank.sample_time(jnp.asarray(5.0 / bank.fps))

    assert in_separator.qpos[0] == pytest.approx(3.0)
    assert next_start.qpos[0] == pytest.approx(100.0)


@pytest.mark.parametrize("fps", [25.0, 50.0, 100.0])
def test_sonic_frame_clock_does_not_accumulate_hours_offset_error(fps):
    bank = ReferenceBank.from_clips((("train", synthetic_clip(0), "train"),))
    bank = replace(bank, fps=fps, lengths=jnp.asarray([2_000_000]))
    env = SimpleNamespace(reference_bank=bank, control_timestep=0.02)
    start = jnp.asarray(1_500_000.125, dtype=jnp.float32)
    end = jax.jit(
        lambda value: jax.lax.fori_loop(
            0,
            500,
            lambda _, clock: BelloSonicTransferEnv.advance_reference_time(env, clock),
            value,
        )
    )(start)
    assert float(end - start) == 10 * fps
    assert float(
        BelloMotionTrackingEnv.advance_reference_time(env, jnp.asarray(0.0))
    ) == pytest.approx(0.02)


@pytest.mark.parametrize("fps", [25.0, 50.0, 100.0])
def test_reference_clock_step_cannot_cross_into_next_clip(fps):
    bank = ReferenceBank.from_clips(
        (("a", synthetic_clip(0), "train"), ("b", synthetic_clip(100), "evaluation"))
    )
    advanced = jax.jit(lambda t: bank.advance_global_frame(t, fps * 0.02))(
        jnp.asarray(2.75)
    )
    assert float(advanced) == 3.0
    assert float(bank.sample_global_frame(advanced).qpos[0]) == 3.0


def test_large_frame_offset_keeps_clip_boundaries_and_fractional_phase():
    bank = ReferenceBank.from_clips(
        (("a", synthetic_clip(0), "train"), ("b", synthetic_clip(100), "evaluation"))
    )
    bank = replace(bank, time_offsets=jnp.asarray([0, 1_500_000], dtype=jnp.int32))
    for fraction in (0.0, 0.125, 0.5, 1.0):
        sample = jax.jit(bank.sample_global_frame)(
            jnp.asarray(1_500_000 + fraction, dtype=jnp.float32)
        )
        assert float(sample.qpos[0]) == pytest.approx(100 + fraction)


def test_sonic_trainable_filter_excludes_pretrained_backbones() -> None:
    adapter = synthetic_adapter()
    key = jax.random.PRNGKey(7)

    def mlp(dims, rank, alpha, subkey):
        tensors = tuple(
            (
                np.zeros((output_size, input_size), dtype=np.float32),
                np.zeros(output_size, dtype=np.float32),
            )
            for input_size, output_size in zip(dims[:-1], dims[1:], strict=True)
        )
        return SonicMLP(tensors, rank=rank, alpha=alpha, key=subkey)

    from bello_mujoco.sonic.model import SonicActor, SonicCritic, SonicNetworkConfig

    config = SonicNetworkConfig()
    encoder_key, decoder_key, critic_key = jax.random.split(key, 3)
    actor = SonicActor(
        mlp((640, 64), 0, 0.0, encoder_key),
        mlp((994, 29), 2, 4.0, decoder_key),
        adapter,
        config,
    )
    policy = SonicActorCritic(
        actor,
        SonicCritic(
            mlp((1645, 1), 2, 2.0, critic_key),
            adapter,
            body_count=14,
        ),
        jnp.zeros(29),
    )
    trainable, frozen = eqx.partition(policy, policy.trainable_filter())

    assert trainable.actor.decoder.layers[0].weight is None
    assert frozen.actor.decoder.layers[0].weight is not None
    assert trainable.actor.decoder.layers[0].lora_a is not None
    assert trainable.actor.encoder.layers[0].lora_a is None
    assert trainable.std is not None


def test_shared_ppo_update_preserves_sonic_frozen_backbone(tmp_path) -> None:
    from bello_mujoco.sonic.model import SonicActor, SonicCritic, SonicNetworkConfig

    adapter = synthetic_adapter()
    generator = np.random.default_rng(31)

    def mlp(input_size, output_size, rank, key):
        tensors = (
            (
                generator.normal(size=(output_size, input_size)).astype(np.float32)
                * 0.01,
                np.zeros(output_size, dtype=np.float32),
            ),
        )
        return SonicMLP(tensors, rank=rank, alpha=float(rank), key=key)

    keys = jax.random.split(jax.random.PRNGKey(32), 3)
    network = SonicNetworkConfig()
    policy = SonicActorCritic(
        SonicActor(
            mlp(640, 64, 0, keys[0]),
            mlp(994, 29, 2, keys[1]),
            adapter,
            network,
        ),
        SonicCritic(mlp(1645, 1, 2, keys[2]), adapter, body_count=14),
        jnp.full(29, np.log(0.05), dtype=jnp.float32),
    )
    profile = sonic_transfer_training_profile(
        tmp_path / "motions", tmp_path / "last.pt", "a" * 40
    )
    config = replace(
        profile.config,
        num_envs=1,
        num_steps=2,
        num_minibatches=1,
        update_epochs=1,
        total_timesteps=2,
        schedule="constant",
    )
    observations = jnp.asarray(
        generator.normal(size=(2, 1, policy.actor.observation_size)),
        dtype=jnp.float32,
    )
    critic_observations = jnp.asarray(
        generator.normal(size=(2, 1, policy.critic.observation_size)),
        dtype=jnp.float32,
    )
    means, _, values = jax.vmap(policy)(observations[:, 0], critic_observations[:, 0])
    actions = means + jnp.asarray([[0.1] * 28, [-0.1] * 28])
    log_probability, _, _ = evaluate_actions(
        policy,
        observations[:, 0],
        actions,
        critic_observations[:, 0],
        latent_action=actions,
    )
    transitions = Transition(
        done=jnp.zeros((2, 1), dtype=bool),
        terminal=jnp.zeros((2, 1), dtype=bool),
        bootstrap_value=values[:, None],
        action=actions[:, None],
        latent_action=actions[:, None],
        env_action=actions[:, None],
        action_mean=means[:, None],
        value=values[:, None],
        reward=jnp.zeros((2, 1)),
        log_prob=log_probability[:, None],
        obs=observations,
        critic_obs=critic_observations,
        info={},
    )
    params, _ = eqx.partition(policy, policy.trainable_filter())
    optimizer = make_optimizer(config)
    state = TrainState(
        policy=policy,
        opt_state=optimizer.init(params),
        update=jnp.asarray(0, dtype=jnp.int32),
    )

    updated, metrics = update_ppo(
        state,
        optimizer,
        transitions,
        jnp.asarray([[1.0], [-1.0]]),
        values[:, None] + jnp.asarray([[0.5], [-0.5]]),
        jax.random.PRNGKey(33),
        config,
    )

    assert float(updated.critic_norm.count) == 3.0
    assert "analytic_kl" in metrics
    assert updated.policy.std.shape == (adapter.target_size,)

    np.testing.assert_array_equal(
        updated.policy.actor.encoder.layers[0].weight,
        policy.actor.encoder.layers[0].weight,
    )
    np.testing.assert_array_equal(
        updated.policy.actor.decoder.layers[0].weight,
        policy.actor.decoder.layers[0].weight,
    )
    np.testing.assert_array_equal(
        updated.policy.critic.network.layers[0].weight,
        policy.critic.network.layers[0].weight,
    )
    assert not np.array_equal(
        updated.policy.actor.decoder.layers[0].lora_b,
        policy.actor.decoder.layers[0].lora_b,
    )


def test_sonic_onnx_matches_jax_actor_and_logical_motor_mapping() -> None:
    from bello_mujoco.sonic.model import SonicActor, SonicCritic, SonicNetworkConfig

    adapter = synthetic_adapter()
    generator = np.random.default_rng(11)

    def mlp(input_size, output_size, rank, key):
        tensors = (
            (
                generator.normal(size=(output_size, input_size)).astype(np.float32)
                * 0.01,
                generator.normal(size=output_size).astype(np.float32) * 0.01,
            ),
        )
        return SonicMLP(tensors, rank=rank, alpha=float(rank), key=key)

    encoder_key, decoder_key, critic_key = jax.random.split(jax.random.PRNGKey(9), 3)
    network_config = SonicNetworkConfig()
    actor = SonicActor(
        mlp(640, 64, 0, encoder_key),
        mlp(994, 29, 2, decoder_key),
        adapter,
        network_config,
    )
    policy = SonicActorCritic(
        actor,
        SonicCritic(mlp(1645, 1, 2, critic_key), adapter, body_count=14),
        jnp.zeros(29),
    )
    joint = SimpleNamespace(effort_limit=4.0, gains=(2.0, 0.1))
    model = SimpleNamespace(
        actuator_joint_names=adapter.target_joint_names,
        default_joint_target=np.linspace(-0.1, 0.1, 28, dtype=np.float32),
        actuator_coordinate_transform=adapter.motor_from_joint,
        mj_model=SimpleNamespace(actuator_ctrlrange=np.tile([-2.0, 2.0], (28, 1))),
    )
    env = SimpleNamespace(
        action_clip=100.0,
        actuator_config=SimpleNamespace(
            by_name={name: joint for name in adapter.target_joint_names}
        ),
    )
    graph = sonic_actor_onnx(policy, model=model, env=env, metadata={})
    observation = generator.normal(size=(1, actor.observation_size)).astype(np.float32)

    action, motor = ReferenceEvaluator(graph).run(None, {"observation": observation})
    expected_action = np.asarray(eqx.filter_jit(actor)(jnp.asarray(observation[0])))
    expected_motor = np.clip(
        model.default_joint_target + 0.5 * expected_action,
        -2.0,
        2.0,
    )

    np.testing.assert_allclose(action[0], expected_action, atol=2.0e-5)
    np.testing.assert_allclose(motor[0], expected_motor, atol=2.0e-5)


def test_sonic_profile_keeps_any2any_lora_and_ppo_recipe(tmp_path) -> None:
    profile = sonic_transfer_training_profile(
        tmp_path / "motions",
        tmp_path / "last.pt",
        "a" * 40,
    )

    assert profile.asset_config.asset == "bello_mini"
    assert profile.config.num_steps == 24
    assert profile.config.update_epochs == 5
    assert profile.config.num_minibatches == 4
    assert profile.config.learning_rate == pytest.approx(5.0e-5)
    assert profile.config.critic_learning_rate is None
    assert profile.config.adaptive_lr_interval == "minibatch"
    assert profile.config.value_loss_scale == 1.0
    assert profile.config.max_grad_norm == pytest.approx(0.1)
    assert profile.config.gradient_norm_max is None
    assert profile.config.policy.obs_stack_size == 10
    assert profile.config.policy.action_distribution == "gaussian"
    assert profile.env_config.simulation_physics.timestep == pytest.approx(0.005)
    assert profile.env_config.decimation == 4
    assert profile.env_config.anchor_body_name == "pelvis_link"
    assert profile.env_config.reward.local_five_point_weight == pytest.approx(2.0)
    assert profile.env_config.reward.anti_shake_weight == pytest.approx(-0.005)
    assert profile.env_config.reward.foot_acceleration_weight == pytest.approx(-2.5e-6)
    assert profile.config.timeout_bootstrap == "current_state"
    assert profile.config.policy.action_limit == profile.env_config.action_clip == 20
    assert profile.env_config.push.interval_seconds == (4.0, 6.0)
    assert profile.env_config.done.root_height_error == 0.25
    assert profile.env_config.done.root_orientation_error == 1.0
    assert profile.env_config.done.end_effector_position_error == 0.3


def test_sonic_integrates_all_reward_terms_and_keeps_beyondmimic_unchanged(monkeypatch):
    monkeypatch.setattr(
        BelloMotionTrackingEnv,
        "_reward",
        lambda self, previous, state, world_id: (
            state - 0.5,
            {"reward/tracking": state, "reward/penalty": jnp.array(-0.5)},
        ),
    )
    env = object.__new__(BelloSonicTransferEnv)
    object.__setattr__(env, "config", SimpleNamespace(decimation=4))
    object.__setattr__(
        env,
        "model",
        SimpleNamespace(mj_model=SimpleNamespace(opt=SimpleNamespace(timestep=0.005))),
    )
    rewards, terms = jax.jit(jax.vmap(lambda x: env._reward(None, x)))(
        jnp.array([1.0, 2.0])
    )
    np.testing.assert_allclose(rewards, [0.01, 0.03])
    np.testing.assert_allclose(terms["reward/tracking"], [0.02, 0.04])
    np.testing.assert_allclose(terms["reward/penalty"], [-0.01, -0.01])
    sanitized, _ = env._sanitize_nonfinite_reward(
        rewards, terms, {"done/nonfinite": jnp.array([False, False])}
    )
    np.testing.assert_allclose(sanitized, rewards)
    assert BelloMotionTrackingEnv._reward(env, None, jnp.array(2.0), 0)[0] == 1.5


def test_sonic_orientation_termination_uses_full_squared_angle_including_yaw():
    env = object.__new__(BelloSonicTransferEnv)
    object.__setattr__(
        env, "config", SimpleNamespace(done=SimpleNamespace(root_orientation_error=1.0))
    )
    identity = jnp.array([1.0, 0.0, 0.0, 0.0])
    angles = np.array([0.99, 1.01, 2.0])
    quats = jnp.asarray(
        np.stack(
            [np.cos(angles / 2), np.zeros(3), np.zeros(3), np.sin(angles / 2)], axis=1
        )
    )
    result = jax.jit(
        jax.vmap(lambda q: env._anchor_orientation_terminated(identity, q))
    )
    np.testing.assert_array_equal(result(quats), [False, True, True])
    np.testing.assert_array_equal(result(-quats), [False, True, True])
    assert not BelloMotionTrackingEnv._anchor_orientation_terminated(
        env, identity, quats[2]
    )


def test_sonic_ankle_cost_uses_physics_acceleration_in_logical_joint_order():
    env = object.__new__(BelloSonicTransferEnv)
    object.__setattr__(
        env,
        "model",
        SimpleNamespace(
            actuator_joint_names=("right_ankle_roll", "elbow", "left_ankle_pitch"),
            policy_qvel_indices=(3, 1, 2),
        ),
    )
    object.__setattr__(env, "config", SimpleNamespace(decimation=4))
    env.model.mj_model = SimpleNamespace(opt=SimpleNamespace(timestep=0.005))
    state = SimpleNamespace(
        data=SimpleNamespace(qvel=jnp.array([100.0, 200.0, 0.015, 0.02])),
        last_substep_qvel=jnp.zeros(4),
    )
    assert env._foot_acceleration_cost(None, state) == pytest.approx(25)


def test_physics_history_captures_final_substep_and_preserves_delayed_controls(
    monkeypatch,
):
    class Data(NamedTuple):
        qvel: jax.Array
        ctrl: jax.Array

        def replace(self, **kwargs):
            return self._replace(**kwargs)

    env = object.__new__(BelloMotionTrackingEnv)
    object.__setattr__(env, "config", SimpleNamespace(decimation=4))
    object.__setattr__(env, "model", SimpleNamespace(mjx_model=None))
    object.__setattr__(
        env, "delayed_ctrl", lambda old, new, index: jnp.where(index < 2, old, new)
    )
    monkeypatch.setattr(
        "bello_mujoco.envs.motion_tracking.mjx.step",
        lambda model, data: data.replace(qvel=data.qvel + data.ctrl),
    )
    data = Data(jnp.zeros(2), jnp.zeros(2))
    stepped, previous = jax.jit(env._step_physics)(
        data, jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
    )
    np.testing.assert_array_equal(stepped.qvel, [8.0, 12.0])
    np.testing.assert_array_equal(previous, [5.0, 8.0])


def test_sonic_episode_metrics_keep_last_100_completions_across_rollouts():
    env = object.__new__(BelloSonicTransferEnv)
    bank = ReferenceBank.from_clips((("train", synthetic_clip(0.0), "train"),))
    object.__setattr__(
        env, "sampler", SonicSampler.from_bank(bank, SonicSamplingConfig())
    )
    state = env.initial_task_state()
    update = jax.jit(env.update_task_state)
    for start, count in ((0, 70), (70, 0), (70, 60), (130, 120)):
        values = jnp.arange(start, start + 128, dtype=jnp.float32)
        state = update(
            state,
            {
                "done/any": jnp.arange(128) < count,
                "episode_return": values,
                "episode_length": values * 2,
                "sampling/visited_bin": jnp.zeros(128, dtype=jnp.int32),
                "done/terminal": jnp.zeros(128, dtype=jnp.bool_),
            },
        )
        total = start + count
        metrics = env.task_state_metrics(state)
        assert metrics["objective/rewards"] == pytest.approx(
            np.mean(np.arange(max(total - 100, 0), total))
        )
        assert metrics["objective/length"] == 2 * metrics["objective/rewards"]
        assert metrics["objective/completed_episodes"] == min(total, 100)


def test_separate_critic_learning_rate_is_selected_by_parameter_path() -> None:
    config = PPOConfig(learning_rate=5.0e-5, critic_learning_rate=1.0e-3)

    assert parameter_learning_rate(
        config, config.learning_rate, (jax.tree_util.GetAttrKey("actor"),)
    ) == pytest.approx(5.0e-5)
    assert parameter_learning_rate(
        config, config.learning_rate, (jax.tree_util.GetAttrKey("critic"),)
    ) == pytest.approx(1.0e-3)


def test_sonic_checkpoint_restores_policy_optimizer_and_sampling_state(
    tmp_path, monkeypatch
) -> None:
    from bello_mujoco.sonic.model import SonicActor, SonicCritic, SonicNetworkConfig

    adapter = synthetic_adapter()

    def mlp(input_size, output_size, rank, key):
        return SonicMLP(
            (
                (
                    np.arange(output_size * input_size, dtype=np.float32).reshape(
                        output_size, input_size
                    )
                    / max(output_size * input_size, 1),
                    np.zeros(output_size, dtype=np.float32),
                ),
            ),
            rank=rank,
            alpha=float(rank),
            key=key,
        )

    keys = jax.random.split(jax.random.PRNGKey(14), 3)
    network = SonicNetworkConfig()
    policy = SonicActorCritic(
        SonicActor(
            mlp(640, 64, 0, keys[0]),
            mlp(994, 29, 2, keys[1]),
            adapter,
            network,
        ),
        SonicCritic(mlp(1645, 1, 2, keys[2]), adapter, body_count=14),
        jnp.zeros(29),
    )
    ppo = PPOConfig(
        num_envs=1,
        num_steps=1,
        num_minibatches=1,
        total_timesteps=2,
    )
    params, static = eqx.partition(policy, policy.trainable_filter())
    optimizer = make_optimizer(ppo)
    runner = RunnerState(
        params=params,
        opt_state=optimizer.init(params),
        update=jnp.asarray(1, dtype=jnp.int32),
        env_state={"sampling_failure_ema": jnp.asarray([0.25, 0.75])},
        obs_stack=jnp.zeros((1, 10, 730)),
        critic_obs_stack=jnp.zeros((1, 10, 805)),
        rng=jax.random.PRNGKey(22),
        learning_rate=jnp.asarray(5.0e-5),
        value_norm_state=init_value_norm_state(),
        observation_norm_state=init_observation_norm_state(7300),
        critic_observation_norm_state=init_observation_norm_state(8050),
        task_state=jnp.zeros((0,), dtype=jnp.float32),
    )
    run_config = SonicTransferTrainingRunConfig(
        asset_config=ModelConfig(asset="bello_mini", actuator_mode="policy"),
        dataset=SonicDatasetConfig(
            directory=tmp_path / "dataset",
            evaluation_clips=("held_out",),
            artifact="entity/project/dataset:v0",
        ),
        pretrained=SonicPretrainedConfig(tmp_path / "last.pt", "a" * 40),
        config=ppo,
    )
    monkeypatch.setattr(
        "bello_mujoco.training.sonic_transfer.require_repository_commit",
        lambda _commit: None,
    )
    checkpoint = tmp_path / "checkpoint"
    _save_checkpoint(
        checkpoint,
        runner,
        static,
        run_config,
        git_commit="b" * 40,
        pretrained_sha256="c" * 64,
        clip_names=("train", "held_out"),
    )

    restored, restored_static, _ = _restore_checkpoint(
        checkpoint,
        (runner, static, optimizer),
        run_config,
        git_commit="b" * 40,
        pretrained_sha256="c" * 64,
        clip_names=("train", "held_out"),
    )

    np.testing.assert_array_equal(restored.rng, runner.rng)
    np.testing.assert_allclose(restored.env_state["sampling_failure_ema"], [0.25, 0.75])
    observation = jnp.zeros(policy.actor.observation_size)
    np.testing.assert_allclose(
        eqx.combine(restored.params, restored_static).action_mean(observation),
        policy.action_mean(observation),
    )
    assert (checkpoint / "policy.eqx").is_file()
