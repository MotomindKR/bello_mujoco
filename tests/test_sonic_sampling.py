from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import ast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bello_mujoco.sonic.sampling import SonicSampler, SonicSamplingConfig


def sampler(config=SonicSamplingConfig()):
    bank = SimpleNamespace(
        lengths=jnp.array([125, 60, 75]),
        num_clips=3,
        split_ids=jnp.array([0, 0, 1]),
        time_offsets=jnp.array([0, 126, 187]),
    )
    return SonicSampler.from_bank(bank, config)


def test_bins_split_mask_and_source_length_weights():
    s = sampler()
    np.testing.assert_array_equal(s.starts, [0, 50, 100, 0, 50, 0, 50])
    np.testing.assert_array_equal(s.lengths, [50, 50, 25, 50, 10, 50, 25])
    np.testing.assert_allclose(s.weights, [50 / 3, 50 / 3, 25 / 3, 25, 5, 25, 12.5])
    expected = np.array([50 / 3, 50 / 3, 25 / 3, 25, 5, 0, 0])
    np.testing.assert_allclose(
        s.initial_state().probabilities, expected / expected.sum()
    )
    np.testing.assert_array_equal(
        s.bins_for_frames(jnp.array([0, 49, 50, 124, 125, 126, 185, 186, 187, 261])),
        [0, 0, 1, 2, 2, 3, 4, 4, 5, 6],
    )


def test_exposure_counts_nonterminal_steps_and_never_heldout():
    s = sampler()
    initial = s.initial_state()
    state = jax.jit(s.accumulate)(
        initial, jnp.array([0, 0, 4, 5]), jnp.array([0, 1, 1, 1])
    )
    np.testing.assert_allclose(state.exposure, [1.04, 1, 1, 1, 1.1, 1, 1])
    np.testing.assert_array_equal(state.failures, [2, 1, 1, 1, 2, 1, 1])
    # Distribution stays fixed until the PPO-iteration boundary.
    np.testing.assert_array_equal(state.probabilities, initial.probabilities)
    assert not np.array_equal(s.refresh(state).probabilities, initial.probabilities)


def test_equal_failure_rates_do_not_reward_exposure_volume():
    s = sampler()
    state = s.initial_state()
    state = state._replace(
        failures=jnp.array([100.0, 1, 1, 1, 1, 1, 1]),
        exposure=jnp.array([100.0, 1, 1, 1, 1, 1, 1]),
    )
    np.testing.assert_allclose(
        s.refresh(state).probabilities, s.initial_state().probabilities
    )


def test_sampling_starts_before_selected_bin_without_crossing_clip_boundary():
    s = sampler()
    state = s.initial_state()._replace(probabilities=jnp.array([0.0, 0, 0, 0, 1, 0, 0]))
    keys = jax.random.split(jax.random.PRNGKey(21), 4096)
    frames = np.asarray(jax.jit(jax.vmap(lambda key: s.sample(state, key)))(keys))
    assert np.all(frames >= 126) and np.all(frames < 186)
    assert np.all(frames == np.floor(frames))
    assert np.mean(frames == 126) > 0.65
    assert np.any(frames > 126)
    no_offset = replace(s, config=replace(s.config, pre_failure_sample_window=0))
    frames = np.asarray(jax.vmap(lambda key: no_offset.sample(state, key))(keys))
    assert np.all(frames >= 176) and np.all(frames < 186)


def test_checkpoint_restores_counters_probabilities_and_next_samples(tmp_path):
    s = sampler()
    state = s.refresh(
        s.accumulate(s.initial_state(), jnp.array([0, 4]), jnp.array([1, 0]))
    )
    path = tmp_path / "sampler.eqx"
    eqx.tree_serialise_leaves(path, state)
    restored = eqx.tree_deserialise_leaves(path, s.initial_state())
    for a, b in zip(state, restored, strict=True):
        np.testing.assert_array_equal(a, b)
    keys = jax.random.split(jax.random.PRNGKey(7), 128)
    np.testing.assert_array_equal(
        jax.vmap(lambda k: s.sample(state, k))(keys),
        jax.vmap(lambda k: s.sample(restored, k))(keys),
    )


def test_pinned_pytorch_sampler_count_and_probability_parity():
    torch = pytest.importorskip("torch")
    source = (
        Path(__file__).resolve().parents[1]
        / "deps/gear_sonic/utils/motion_lib/motion_lib_base.py"
    )
    tree = ast.parse(source.read_text())
    names = {"update_adaptive_sampling", "update_adaptive_sampling_probabilities"}
    methods = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(methods) == 2
    scope = {"torch": torch}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), scope)
    s = sampler()

    def to_torch(value):
        return torch.tensor(np.asarray(value).copy())

    active = torch.arange(5)
    ref = SimpleNamespace(
        get_motion_ids_in_dataset=lambda ids: ids,
        adp_samp_length_starts=torch.tensor([0, 125, 185]),
        adp_samp_frame_to_bin=torch.repeat_interleave(
            torch.arange(7), to_torch(s.lengths).long()
        ),
        adp_samp_num_bins=7,
        adp_samp_bin_motion_length=to_torch(s.lengths),
        adp_samp_num_episodes=torch.ones(7),
        adp_samp_num_failures=torch.ones(7),
        adaptive_sampling_cfg={},
        adp_samp_active_motion_bins=active,
        adp_samp_failure_rate_max_over_mean=200.0,
        uniform_sampling_rate=0.1,
        adp_samp_bin_weights=to_torch(s.weights),
        max_prob_per_bin_cfg=None,
        max_prob_per_motion_cfg=None,
    )
    state = s.initial_state()
    for _ in range(5):
        ids = torch.tensor([0, 0, 1, 1])
        frames = torch.tensor([5, 55, 7, 59])
        failures = torch.tensor([True, False, False, True])
        scope["update_adaptive_sampling"](ref, failures, ids, frames)
        bins = s.bins_for_frames(jnp.array([5, 55, 133, 185]))
        state = s.accumulate(state, bins, jnp.array([True, False, False, True]))
    ref.adp_samp_failure_rate = ref.adp_samp_num_failures / ref.adp_samp_num_episodes
    scope["update_adaptive_sampling_probabilities"](ref)
    state = s.refresh(state)
    np.testing.assert_allclose(
        state.failures, ref.adp_samp_num_failures.numpy(), rtol=1e-6
    )
    np.testing.assert_allclose(
        state.exposure, ref.adp_samp_num_episodes.numpy(), rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(state.probabilities)[:5],
        ref.adp_sampling_active_prob.numpy(),
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bin_size": 0},
        {"pre_failure_sample_window": -1},
        {"uniform_sampling_rate": 2},
        {"failure_rate_max_over_mean": float("nan")},
        {"init_num_failures": 0},
    ],
)
def test_invalid_sampler_config_fails_explicitly(kwargs):
    with pytest.raises(ValueError):
        SonicSamplingConfig(**kwargs)
