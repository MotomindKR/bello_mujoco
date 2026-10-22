"""Return-based reset curriculum over one-second motion bins."""

from dataclasses import dataclass, field
import math

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class SamplingConfig:
    uniform_weight: float = 0.5
    difficult_weight: float = 0.4
    past_weight: float = 0.1
    temperature: float = 25.0
    return_alpha: float = 0.1

    def __post_init__(self):
        values = (self.uniform_weight, self.difficult_weight, self.past_weight)
        if any(not math.isfinite(v) or v < 0 for v in values) or not math.isclose(
            sum(values), 1.0
        ):
            raise ValueError(
                "sampling mixture weights must be nonnegative and sum to one"
            )
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("sampling temperature must be positive and finite")
        if not 0 < self.return_alpha <= 1:
            raise ValueError("sampling return_alpha must be in (0, 1]")


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SamplingState:
    returns: jax.Array
    visits: jax.Array
    past_weights: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ReturnSampler:
    starts: jax.Array
    lengths: jax.Array
    eligible: jax.Array
    config: SamplingConfig = field(metadata={"static": True})

    @classmethod
    def from_bank(cls, bank, config=SamplingConfig()):
        starts, lengths, eligible = [], [], []
        bin_size = max(1, round(bank.fps))
        for offset, length, split in zip(
            np.asarray(bank.time_offsets),
            np.asarray(bank.lengths),
            np.asarray(bank.split_ids),
            strict=True,
        ):
            for start in range(0, int(length) - 1, bin_size):
                starts.append(int(offset) + start)
                lengths.append(min(bin_size, int(length) - 1 - start))
                eligible.append(split == 0)
        if not any(eligible):
            raise ValueError("reset sampler requires at least one training bin")
        return cls(
            jnp.asarray(starts, jnp.int32),
            jnp.asarray(lengths, jnp.int32),
            jnp.asarray(eligible),
            config,
        )

    def initial_state(self):
        return SamplingState(
            jnp.zeros_like(self.starts, dtype=jnp.float32),
            jnp.zeros_like(self.starts, dtype=jnp.int32),
            self.eligible.astype(jnp.float32) / self.eligible.sum(),
        )

    def probabilities(self, state, maximum_return):
        uniform = self.eligible.astype(jnp.float32) / self.eligible.sum()
        logits = jnp.where(
            self.eligible,
            -self.config.temperature * state.returns / maximum_return,
            -jnp.inf,
        )
        difficult = jax.nn.softmax(logits)
        past = jnp.where(self.eligible, state.past_weights, 0.0)
        past = past / jnp.maximum(past.sum(), 1e-12)
        cfg = self.config
        return (
            cfg.uniform_weight * uniform
            + cfg.difficult_weight * difficult
            + cfg.past_weight * past
        )

    def sample(self, key, probabilities):
        bin_key, frame_key = jax.random.split(key)
        draw = jax.random.uniform(bin_key)
        index = jnp.minimum(
            jnp.searchsorted(jnp.cumsum(probabilities), draw, side="right"),
            len(self.starts) - 1,
        )
        offset = jax.random.randint(frame_key, (), 0, self.lengths[index])
        return (self.starts[index] + offset).astype(jnp.float32)

    def update(self, state, start_bins, returns, completed, maximum_return):
        counts = (
            jnp.zeros_like(state.visits).at[start_bins].add(completed.astype(jnp.int32))
        )
        totals = (
            jnp.zeros_like(state.returns)
            .at[start_bins]
            .add(jnp.where(completed, returns, 0.0))
        )
        means = totals / jnp.maximum(counts, 1)
        updated = jnp.where(
            state.visits == 0,
            means,
            state.returns + self.config.return_alpha * (means - state.returns),
        )
        updated = jnp.where(counts > 0, updated, state.returns)
        current = jax.nn.softmax(
            jnp.where(
                self.eligible,
                -self.config.temperature * updated / maximum_return,
                -jnp.inf,
            )
        )
        return SamplingState(
            updated, state.visits + counts, state.past_weights + current
        )
