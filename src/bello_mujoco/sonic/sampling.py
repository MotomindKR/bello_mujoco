"""Any2Any's exposure-normalized, temporal-bin motion curriculum in JAX."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from bello_mujoco.sonic.reference import ReferenceBank


@dataclass(frozen=True)
class SonicSamplingConfig:
    bin_size: int = 50
    pre_failure_sample_window: int = 200
    uniform_sampling_rate: float = 0.1
    failure_rate_max_over_mean: float = 200.0
    init_num_failures: float = 1.0

    def __post_init__(self):
        if self.bin_size <= 0 or self.pre_failure_sample_window < 0:
            raise ValueError(
                "SONIC sampler requires positive bins and a nonnegative window"
            )
        if not 0 <= self.uniform_sampling_rate <= 1:
            raise ValueError("SONIC uniform_sampling_rate must be in [0, 1]")
        if (
            not np.isfinite(self.failure_rate_max_over_mean)
            or self.failure_rate_max_over_mean <= 0
        ):
            raise ValueError("SONIC failure-rate bound must be positive and finite")
        if not np.isfinite(self.init_num_failures) or self.init_num_failures <= 0:
            raise ValueError("SONIC sampler prior must be positive and finite")


class SonicSamplingState(NamedTuple):
    failures: jax.Array
    exposure: jax.Array
    probabilities: jax.Array


class SonicTaskState(NamedTuple):
    episodes: jax.Array
    sampling: SonicSamplingState


@dataclass(frozen=True)
class SonicSampler:
    config: SonicSamplingConfig
    clip_ids: jax.Array
    starts: jax.Array
    lengths: jax.Array
    weights: jax.Array
    eligible: jax.Array
    clip_bin_offsets: jax.Array
    clip_time_offsets: jax.Array
    clip_lengths: jax.Array

    @classmethod
    def from_bank(cls, bank: ReferenceBank, config: SonicSamplingConfig):
        lengths = np.asarray(bank.lengths)
        counts = (lengths + config.bin_size - 1) // config.bin_size
        clips = np.repeat(np.arange(bank.num_clips), counts)
        starts = np.concatenate([np.arange(n) * config.bin_size for n in counts])
        bin_lengths = np.minimum(config.bin_size, lengths[clips] - starts)
        # The source's division by mean bin length cancels on normalization.
        weights = bin_lengths / counts[clips]
        return cls(
            config,
            jnp.asarray(clips),
            jnp.asarray(starts),
            jnp.asarray(bin_lengths, dtype=jnp.float32),
            jnp.asarray(weights, dtype=jnp.float32),
            jnp.asarray(np.asarray(bank.split_ids)[clips] == 0),
            jnp.asarray(np.concatenate(([0], np.cumsum(counts[:-1])))),
            bank.time_offsets,
            bank.lengths,
        )

    @property
    def num_bins(self):
        return self.starts.size

    def initial_state(self):
        prior = jnp.full(self.num_bins, self.config.init_num_failures)
        return self.refresh(SonicSamplingState(prior, prior, jnp.zeros_like(prior)))

    def refresh(self, state):
        rates = state.failures / state.exposure
        mean = jnp.sum(jnp.where(self.eligible, rates, 0)) / jnp.sum(self.eligible)
        rates = jnp.where(
            self.eligible,
            jnp.minimum(rates, mean * self.config.failure_rate_max_over_mean),
            0,
        )
        uniform = self.eligible / jnp.sum(self.eligible)
        probabilities = (
            (1 - self.config.uniform_sampling_rate) * rates / jnp.sum(rates)
            + self.config.uniform_sampling_rate * uniform
        ) * self.weights
        return state._replace(probabilities=probabilities / jnp.sum(probabilities))

    def bins_for_frames(self, frames):
        clips = jnp.clip(
            jnp.searchsorted(self.clip_time_offsets, frames, side="right") - 1,
            0,
            self.clip_lengths.size - 1,
        )
        local = jnp.clip(
            frames - self.clip_time_offsets[clips], 0, self.clip_lengths[clips] - 1
        )
        return (
            self.clip_bin_offsets[clips]
            + local.astype(jnp.int32) // self.config.bin_size
        )

    def accumulate(self, state, bins, failed):
        # Source num_episodes is occupied-bin steps / bin length, not reset count.
        eligible = self.eligible[bins].astype(jnp.float32)
        exposure = jnp.zeros(self.num_bins).at[bins].add(eligible / self.lengths[bins])
        failures = jnp.zeros(self.num_bins).at[bins].add(eligible * failed)
        return state._replace(
            failures=state.failures + failures,
            exposure=state.exposure + exposure,
        )

    def sample(self, state, key):
        bin_key, frame_key, offset_key = jax.random.split(key, 3)
        index = jax.random.choice(bin_key, self.num_bins, p=state.probabilities)
        frame = self.starts[index] + jnp.floor(
            jax.random.uniform(frame_key) * self.lengths[index]
        )
        if self.config.pre_failure_sample_window:
            frame -= jax.random.randint(
                offset_key, (), 0, self.config.pre_failure_sample_window
            )
        return (
            self.clip_time_offsets[self.clip_ids[index]] + jnp.maximum(frame, 0)
        ).astype(jnp.float32)

    def metrics(self, state):
        probabilities = state.probabilities
        clip_probabilities = (
            jnp.zeros(self.clip_lengths.size).at[self.clip_ids].add(probabilities)
        )
        rates = state.failures / state.exposure
        return {
            "sampling/entropy": -jnp.sum(probabilities * jnp.log(probabilities + 1e-12))
            / jnp.maximum(jnp.log(jnp.sum(self.eligible)), 1),
            "sampling/top_bin_probability": jnp.max(probabilities),
            "sampling/top_clip_probability": jnp.max(clip_probabilities),
            "sampling/failure_rate_mean": jnp.sum(jnp.where(self.eligible, rates, 0))
            / jnp.sum(self.eligible),
            "sampling/failure_rate_max": jnp.max(jnp.where(self.eligible, rates, 0)),
        }
