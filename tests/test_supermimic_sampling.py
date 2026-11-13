from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bello_mujoco.supermimic.sampling import (
    ReturnSampler,
    SamplingConfig,
    SamplingWindow,
)


def sampler(lengths=(63, 70), splits=(0, 1), **kwargs):
    return ReturnSampler.from_bank(
        SimpleNamespace(
            fps=50.0,
            time_offsets=np.r_[0, np.cumsum(np.array(lengths[:-1]) + 1)],
            lengths=np.array(lengths),
            split_ids=np.array(splits),
        ),
        SamplingConfig(**kwargs),
    )


def window(
    bins,
    rewards,
    *,
    next_bins=None,
    terminal=None,
    clip_end=None,
    timeout=None,
    valid=None,
):
    bins = jnp.asarray(bins, jnp.int32)

    def flags(value):
        return (
            jnp.zeros(bins.shape, bool) if value is None else jnp.asarray(value, bool)
        )

    return SamplingWindow(
        bins,
        bins if next_bins is None else jnp.asarray(next_bins),
        jnp.asarray(rewards),
        flags(terminal),
        flags(clip_end),
        flags(timeout),
        jnp.ones(bins.shape, bool) if valid is None else jnp.asarray(valid, bool),
    )


def test_clip_end_does_not_extrapolate_reset_action_penalty_into_catastrophic_return():
    s = sampler((10001,), (0,))
    state = s.initial_state(10.0)
    tail = window([[999]], [[-0.65]], clip_end=[[True]], timeout=[[True]])
    result = jax.jit(lambda state, w: s.update(state, w, 10.0, 0.99))(state, tail)
    assert float(result.returns[999]) == pytest.approx(-0.65 + 0.99 * 10.0)
    assert float(result.probabilities.max()) < 0.05
    failed = s.update(state, replace(tail, terminal=jnp.ones((1, 1), bool)), 10.0, 0.99)
    assert float(failed.returns[999]) == pytest.approx(-0.65)


def test_buffered_updates_match_independent_per_episode_bin_return_calculation():
    s = sampler((31,), (0,), memory=2)
    data = window(
        [[0, 0, 1, 1], [0, 1, 1, 2]],
        [[0.1, 0.2, -0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        next_bins=[[0, 1, 1, 2], [1, 1, 2, 2]],
        terminal=[[False, False, True, False], [False, False, False, False]],
        clip_end=[[False] * 4, [False, False, False, True]],
        timeout=[[False] * 4, [False, True, False, True]],
    )
    state = s.initial_state(10.0)
    for _ in range(4):
        values = np.asarray(state.returns)
        sums = np.zeros(len(values))
        counts = np.zeros(len(values), int)
        for env in range(2):
            returns = np.zeros(4)
            carry = values[int(data.next_bins[env, -1])]
            for t in reversed(range(4)):
                bootstrap = (
                    0.0
                    if data.terminal[env, t]
                    else values.mean()
                    if data.clip_end[env, t]
                    else values[int(data.next_bins[env, t])]
                )
                if (
                    data.terminal[env, t]
                    or data.clip_end[env, t]
                    or data.timeout[env, t]
                ):
                    carry = bootstrap
                returns[t] = float(data.rewards[env, t]) + 0.9 * carry
                carry = returns[t]
            episode = 0
            groups = {}
            for t in range(4):
                groups.setdefault((episode, int(data.bins[env, t])), []).append(
                    returns[t]
                )
                episode += bool(
                    data.terminal[env, t]
                    or data.clip_end[env, t]
                    or data.timeout[env, t]
                )
            for (_, b), samples in groups.items():
                sums[b] += np.mean(samples)
                counts[b] += 1
        expected_counts = np.minimum(np.asarray(state.visits), 2) + counts
        expected_values = values + (sums - counts * values) / np.maximum(
            expected_counts, 1
        )
        result = s.update(state, data, 10.0, 0.9)
        np.testing.assert_allclose(result.returns, expected_values, atol=1e-6)
        np.testing.assert_array_equal(result.visits, expected_counts)
        focus = jax.nn.softmax(-25 * result.returns / 10.0)
        expected_effort = state.effort + focus**2
        np.testing.assert_allclose(result.effort, expected_effort, atol=1e-7)
        expected_probability = (
            0.5 / len(values) + 0.4 * focus + 0.1 * result.effort / result.effort.sum()
        )
        np.testing.assert_allclose(
            result.probabilities, expected_probability, rtol=1e-6
        )
        state = result


def test_short_final_bins_are_sampled_without_leaking_into_held_out_clips():
    s = sampler()
    np.testing.assert_array_equal(s.lengths, [10] * 6 + [2] + [10] * 6 + [9])
    state = s.initial_state(10.0)
    result = s.update(
        state,
        window(
            [[6, 6]], [[-20.0, jnp.nan]], terminal=[[True, True]], valid=[[True, False]]
        ),
        10.0,
        0.99,
    )
    assert result.visits[6] == 1
    assert result.returns[6] == -20.0
    np.testing.assert_array_equal(result.probabilities[7:], 0.0)
    draws = jax.jit(jax.vmap(lambda k: s.sample(k, result.probabilities)))(
        jax.random.split(jax.random.key(0), 1000)
    )
    assert np.max(draws) <= 61 and np.min(draws) >= 0
    assert set(np.asarray(draws)[np.asarray(draws) >= 60]) == {60, 61}


def test_cdf_roundoff_cannot_select_trailing_held_out_bins(monkeypatch):
    s = sampler((21, 11), (0, 1))
    half = jnp.nextafter(jnp.float32(0.5), jnp.float32(0.0))
    probabilities = jnp.array([half, half, 0.0])
    monkeypatch.setattr(
        jax.random,
        "uniform",
        lambda key: jnp.nextafter(jnp.float32(1.0), jnp.float32(0.0)),
    )
    frame = jax.jit(lambda key: s.sample(key, probabilities))(jax.random.key(2))
    assert 10 <= frame < 20
