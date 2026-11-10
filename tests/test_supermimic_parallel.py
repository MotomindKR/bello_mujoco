"""Exercise distributed numerics in a fresh process with two CPU devices."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_two_device_training_numerics():
    environment = os.environ.copy()
    environment.update(
        JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=2"
    )
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.integration
@pytest.mark.parametrize("randomization", [True, False])
def test_two_cuda_devices_collect_fit_imagine(tmp_path, randomization):
    from dataclasses import replace
    import jax
    import numpy as np
    from bello_mujoco.configuration import supermimic_training_profile
    from bello_mujoco.supermimic.optimization import initialize_learner
    from bello_mujoco.supermimic.parallel import DataParallel
    from bello_mujoco.training.supermimic import (
        load_environment,
        compile_updates,
        require_valid_metrics,
    )
    from test_supermimic_environment import mini_corpus

    if len(jax.local_devices()) < 2 or jax.devices()[0].platform != "gpu":
        pytest.skip("requires two CUDA devices")
    _, path = mini_corpus.__wrapped__(tmp_path)
    run = replace(
        supermimic_training_profile(path.parent, name="bello_mini_supermimic_smoke"),
        num_devices=2,
    )
    run = replace(
        run,
        environment=replace(
            run.environment,
            randomization=replace(run.environment.randomization, enabled=randomization),
        ),
    )
    cfg = replace(run.config, num_envs=4)
    run = replace(run, config=cfg)
    parallel = DataParallel.create(2)
    env = parallel.environment(
        load_environment(
            run, load_mjx=True, paths=(path,), reference_device=parallel.replicated
        )
    )
    initial_key, collect_key = jax.random.split(jax.random.PRNGKey(0))
    learner = jax.device_put(
        initialize_learner(env, cfg, run.networks, env.action_size + 5, initial_key),
        parallel.replicated,
    )
    physics, specs = parallel.physics(env)
    initialize, collect = parallel.collection(env, cfg, physics, specs)
    runner = initialize(
        learner,
        jax.device_put(collect_key, parallel.replicated),
        env.reference_bank,
        physics,
    )
    for _ in range(cfg.warmup_steps):
        runner, metrics = collect(runner, env.reference_bank)
        require_valid_metrics(metrics)
    _, fit, improve = compile_updates(
        env, cfg, parallel=parallel, parallel_collect=collect
    )
    for _ in range(2):
        runner, metrics = fit(runner)
        require_valid_metrics(metrics)
        runner, metrics = collect(runner, env.reference_bank)
        require_valid_metrics(metrics)
        runner, metrics = improve(runner, env.reference_bank)
        require_valid_metrics(metrics)
        runner, metrics = collect(runner, env.reference_bank)
        require_valid_metrics(metrics)
    assert int(runner.actor_updates) == 2
    assert runner.collector.environment.reference_time.dtype == np.int32
    qpos = runner.collector.environment.data.qpos
    assert qpos.shape[:2] == (parallel.count, cfg.num_envs // parallel.count)
    assert len(qpos.addressable_shards) == 2
    assert all(
        shard.data.shape[:2] == (1, cfg.num_envs // parallel.count)
        for shard in qpos.addressable_shards
    )
    for leaf in jax.tree.leaves(runner.learner):
        assert len(leaf.addressable_shards) == 2
        np.testing.assert_array_equal(
            leaf.addressable_shards[0].data, leaf.addressable_shards[1].data
        )


def exercise_parallel_training():
    from dataclasses import replace, dataclass, fields
    import json
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import PartitionSpec as P

    from bello_mujoco.supermimic import checkpoint
    from bello_mujoco.supermimic.model import Moments
    from bello_mujoco.supermimic.optimization import (
        update_world_model,
        update_actor_critic,
    )
    from bello_mujoco.supermimic.collection import collect_step
    from bello_mujoco.supermimic.parallel import (
        EnvironmentArrays,
        collection_layout,
        pmap_axes,
        reduce_collection_metrics,
    )
    from bello_mujoco.supermimic.parallel import AXIS, DataParallel, runner_specs
    from bello_mujoco.supermimic.replay import ReplayBuffer
    from bello_mujoco.supermimic.sampling import ReturnSampler
    from bello_mujoco.training.supermimic import (
        DatasetConfig,
        SuperMimicTrainingRunConfig,
        compile_updates,
    )
    from bello_mujoco.models.compiler import ModelConfig
    from bello_mujoco.sonic.corpus import MANIFEST, file_sha256
    from test_supermimic_runner import toy_runner, ToyEnv

    parallel = DataParallel.create(2)
    cpu = jax.devices()[0]

    def equal(a, b):
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
            np.testing.assert_allclose(x, y, rtol=3e-5, atol=3e-6)

    values = jnp.arange(24, dtype=jnp.float32).reshape(8, 3)
    mask = jnp.array([True, False, True, True, False, False, False, False])
    moments = Moments.empty(3)
    update = jax.jit(
        jax.shard_map(
            lambda m, x, w: m.update(x, w, axis_name=AXIS),
            mesh=parallel.mesh,
            in_specs=(P(), P(AXIS), P(AXIS)),
            out_specs=P(),
        )
    )
    actual = update(
        parallel.put(moments, P()),
        parallel.put(values, P(AXIS)),
        parallel.put(mask, P(AXIS)),
    )
    equal(actual, moments.update(values, mask))

    sampler = ReturnSampler.from_bank(
        SimpleNamespace(
            fps=50,
            time_offsets=np.array([0, 64]),
            lengths=np.array([63, 70]),
            split_ids=np.array([0, 0]),
        )
    )
    sampler = jax.device_put(sampler, parallel.replicated)
    state = sampler.initial_state()
    bins = jnp.array([0, 1, 1, 2])
    returns = jnp.array([10.0, 1.0, 3.0, 4.0])
    completed = jnp.array([True, True, True, False])
    update_sampler = jax.jit(
        jax.shard_map(
            lambda sampler, s, b, r, c: sampler.update(
                s, b, r, c, 10.0, axis_name=AXIS
            ),
            mesh=parallel.mesh,
            in_specs=(P(), P(), P(AXIS), P(AXIS), P(AXIS)),
            out_specs=P(),
        )
    )
    expected = jax.device_put(sampler, cpu).update(
        jax.device_put(state, cpu), bins, returns, completed, 10.0
    )
    equal(
        update_sampler(
            sampler,
            state,
            parallel.put(bins, P(AXIS)),
            parallel.put(returns, P(AXIS)),
            parallel.put(completed, P(AXIS)),
        ),
        expected,
    )

    env, config, runner = toy_runner.__wrapped__()

    @jax.tree_util.register_dataclass
    @dataclass(frozen=True)
    class ToyModelArrays:
        jnt_pos: object
        body_inertia: object
        body_mass: object
        body_ipos: object
        geom_friction: object
        actuator_gainprm: object
        actuator_biasprm: object

    @dataclass(frozen=True)
    class ToyModel:
        mjx_model: object
        joint_angle_offset: object
        torso_com_offset: object
        physical_randomization: object
        gravity_fraction: object
        actuator_peak_force: object
        actuator_coordinate_names: tuple
        warp_naconmax: int = 128
        warp_naccdmax: object = None

    @dataclass(frozen=True)
    class ParallelToyEnv(ToyEnv):
        model: object = None
        motion: object = None

        def reset_batch(self, count, rng):
            state, observation = super().reset_batch(count, rng)
            return replace(
                state, data=replace(state.data, qacc=state.data.qacc[:, :0])
            ), observation

        def step_batch(self, state, action, rng, *, reset):
            stepped, *outputs = super().step_batch(state, action, rng, reset=reset)
            stepped = replace(
                stepped, data=replace(stepped.data, qacc=stepped.data.qacc[:, :0])
            )
            return stepped, *outputs

        def context(self, state, world_id):
            return jnp.concatenate(
                (
                    state.joint_default_offset,
                    self.model.physical_randomization[world_id, 4:5],
                    self.model.torso_com_offset[world_id],
                    jnp.zeros(1),
                )
            )

    physical = np.zeros((2, 7), np.float32)
    physical[:, 4] = [0.6, 1.1]
    toy_model = ToyModel(
        ToyModelArrays(*(np.zeros((2, 1), np.float32) for _ in range(7))),
        np.zeros((2, 1), np.float32),
        np.zeros((2, 3), np.float32),
        physical,
        np.zeros(2, np.float32),
        np.full(1, 4.0, np.float32),
        ("joint",),
    )
    values = {field.name: getattr(env, field.name) for field in fields(env)}
    values["config"] = replace(
        env.config,
        model_randomization=replace(env.config.model_randomization, enabled=True),
    )
    collection_env = parallel.environment(ParallelToyEnv(**values, model=toy_model))
    physics, physics_specs = parallel.physics(collection_env)
    initialize, distributed_collect = parallel.collection(
        collection_env, config, physics, physics_specs
    )
    collection_runner = initialize(
        jax.device_put(runner.learner, parallel.replicated),
        jax.device_put(jax.random.PRNGKey(11), parallel.replicated),
        collection_env.reference_bank,
        physics,
    )

    def fused_collect(state, bank, physical, arrays):
        current = physical.apply(
            replace(arrays.apply(collection_env), reference_bank=bank)
        )
        result, metrics = collect_step(
            state, current, parallel.local_config(config), axis_name=AXIS
        )
        return result, reduce_collection_metrics(metrics, config.num_envs // 2)

    fused_collect = jax.jit(
        jax.pmap(
            fused_collect,
            axis_name=AXIS,
            in_axes=(pmap_axes(runner_specs()), None, pmap_axes(physics_specs), None),
            out_axes=(pmap_axes(runner_specs()), None),
        )
    )
    expected_runner = parallel.pack(collection_runner, collection_layout())
    for _ in range(4):
        expected_runner, expected_metrics = fused_collect(
            expected_runner,
            collection_env.reference_bank,
            parallel.pack(physics, physics_specs),
            EnvironmentArrays.from_environment(collection_env),
        )
        collection_runner, collection_metrics = distributed_collect(
            collection_runner, collection_env.reference_bank
        )
        equal(collection_runner, parallel.unpack(expected_runner, collection_layout()))
        equal(collection_metrics, expected_metrics)
    np.testing.assert_allclose(
        collection_runner.replay.data.context[0, :, 1], [0.6, 1.1]
    )
    assert int(collection_runner.collector.vector_steps) == 4
    assert float(collection_metrics["invalid/collection"]) == 0
    assert collection_runner.collector.environment.data.qpos.shape[:2] == (2, 1)
    assert collection_runner.collector.environment.data.qacc.shape == (2, 1, 0)
    large_bank = replace(
        env.reference_bank,
        time_offsets=jnp.array([50_000_000]),
        clock_end=50_000_020,
        qpos=env.reference_bank.qpos.at[:, 0].set(jnp.arange(20)),
    )
    frames = jnp.arange(50_000_000, 50_000_020, dtype=jnp.int32)
    samples = jax.jit(jax.vmap(large_bank.sample_global_frame))(frames)
    np.testing.assert_array_equal(samples.qpos[:, 0], np.arange(20))
    advanced = jax.jit(
        jax.vmap(lambda f: large_bank.advance_global_frame(f, jnp.int32(1)))
    )(frames)
    np.testing.assert_array_equal(
        advanced, np.minimum(np.asarray(frames) + 1, 50_000_019)
    )
    collect, _, _ = compile_updates(env, config)
    for _ in range(4):
        runner, _ = collect(runner, env.reference_bank)
    replay = replace(
        runner.replay,
        data=replace(
            runner.replay.data,
            clip_id=jnp.tile(jnp.array([0, 1]), (config.replay_steps, 1)),
        ),
    )
    specs = ReplayBuffer(P(None, AXIS), P(None, AXIS), P(), P())
    replay = parallel.put(replay, specs)
    key = jax.device_put(jax.random.PRNGKey(10), parallel.replicated)
    sample_windows = jax.jit(lambda r, k: parallel.sample_windows(r, k, 8, 2))
    window = sample_windows(replay, key)
    np.testing.assert_array_equal(window.transitions.clip_id[:4], 0)
    np.testing.assert_array_equal(window.transitions.clip_id[4:], 1)
    successor = replace(
        window.transitions.successor,
        joint_position=window.transitions.successor.joint_position
        + jnp.arange(8)[:, None, None] / 3,
    )
    window = replace(
        window, transitions=replace(window.transitions, successor=successor)
    )
    learner = jax.device_put(runner.learner, parallel.replicated)
    step = jax.jit(
        lambda learner, window: update_world_model(learner, window, config, 0)
    )
    expected = step(jax.device_put(learner, cpu), jax.device_put(window, cpu))
    actual = step(learner, window)
    equal(actual, expected)
    assert len(actual[0].model.network.layers[0].weight.addressable_shards) == 2

    replay = replace(
        replay, data=replace(replay.data, clip_id=jnp.zeros_like(replay.data.clip_id))
    )
    starts = jax.jit(
        lambda r, k: parallel.sample_starts(
            r, k, 2, history_frames=2, feedback_frames=1, horizon=2
        )
    )(replay, key)
    assert bool(starts.valid)
    fields = (
        "reference_bank",
        "kinematics",
        "codec",
        "mapping",
        "mdp",
        "sampler",
        "reset_probabilities",
    )
    env_parallel = replace(
        env,
        **{
            name: jax.device_put(getattr(env, name), parallel.replicated)
            for name in fields
        },
    )
    env_single = replace(
        env, **{name: jax.device_put(getattr(env, name), cpu) for name in fields}
    )
    step_parallel = jax.jit(
        lambda learner, starts: update_actor_critic(
            learner, starts, env_parallel, config, 0, key
        )
    )
    step_single = jax.jit(
        lambda learner, starts: update_actor_critic(
            learner, starts, env_single, config, 0, jax.device_put(key, cpu)
        )
    )
    equal(
        step_parallel(learner, starts),
        step_single(jax.device_put(learner, cpu), jax.device_put(starts, cpu)),
    )

    distributed = parallel.put(runner, runner_specs())
    _, fit, improve = compile_updates(env_parallel, config, parallel=parallel)
    distributed, model_metrics = fit(distributed)
    distributed, actor_metrics = improve(distributed, env_parallel.reference_bank)
    assert float(model_metrics["invalid/model_update"]) == 0
    assert float(actor_metrics["invalid/actor_critic_update"]) == 0
    assert int(distributed.actor_updates) == 1
    checkpoint.require_repository_commit = lambda _: None
    with TemporaryDirectory() as directory:
        directory = Path(directory)
        corpus = directory / "corpus"
        corpus.mkdir()
        (corpus / MANIFEST).write_text('{"test":true}\n')
        run = SuperMimicTrainingRunConfig(
            ModelConfig(asset="bello_mini"),
            DatasetConfig(corpus),
            config=config,
            observations=env.codec.config,
            num_devices=2,
        )
        digest = file_sha256(corpus / MANIFEST)
        checkpoint.save_checkpoint(
            directory / "checkpoint",
            distributed,
            run,
            git_commit="0" * 40,
            dataset_digest=digest,
        )
        restored = checkpoint.restore_checkpoint(
            directory / "checkpoint",
            distributed,
            run,
            git_commit="0" * 40,
            dataset_digest=digest,
        )
        equal(distributed, restored)
        for a, b in zip(
            jax.tree.leaves(distributed), jax.tree.leaves(restored), strict=True
        ):
            assert a.sharding == b.sharding
    print(
        json.dumps(
            {
                "two_device_moments": True,
                "partitioned_collection": True,
                "curriculum": True,
                "local_replay": True,
                "global_model_and_actor_gradients": True,
                "sharded_checkpoint": True,
            }
        )
    )


if __name__ == "__main__":
    exercise_parallel_training()
