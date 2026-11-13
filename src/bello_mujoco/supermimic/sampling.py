"""Buffered per-bin return curriculum with clip-end value bootstrapping."""

from dataclasses import dataclass, field, replace
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
    bin_seconds: float = 0.2
    horizon: int = 20
    memory: int = 256

    def __post_init__(self):
        weights = (self.uniform_weight, self.difficult_weight, self.past_weight)
        if any(not math.isfinite(v) or v < 0 for v in weights) or not math.isclose(
            sum(weights), 1.0
        ):
            raise ValueError("sampling weights must be nonnegative and sum to one")
        for name in ("temperature", "bin_seconds"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"sampling {name} must be positive and finite")
        for name in ("horizon", "memory"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"sampling {name} must be a positive integer")


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SamplingWindow:
    bins: jax.Array
    next_bins: jax.Array
    rewards: jax.Array
    terminal: jax.Array
    clip_end: jax.Array
    timeout: jax.Array
    valid: jax.Array

    @classmethod
    def empty(cls, num_envs, horizon):
        shape = (num_envs, horizon)
        return cls(
            jnp.zeros(shape, jnp.int32),
            jnp.zeros(shape, jnp.int32),
            jnp.zeros(shape),
            *(jnp.zeros(shape, bool) for _ in range(4)),
        )

    def record(
        self, index, bins, next_bins, rewards, terminal, clip_end, timeout, valid
    ):
        values = type(self)(
            bins, next_bins, rewards, terminal, clip_end, timeout, valid
        )
        return jax.tree.map(
            lambda buffer, value: buffer.at[:, index].set(value), self, values
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SamplingState:
    returns: jax.Array
    visits: jax.Array
    effort: jax.Array
    effort_compensation: jax.Array
    probabilities: jax.Array


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
        bin_size = max(1, round(bank.fps * config.bin_seconds))
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

    @property
    def uniform_probabilities(self):
        return self.eligible.astype(jnp.float32) / self.eligible.sum()

    def initial_state(self, maximum_return):
        zeros = jnp.zeros_like(self.starts, dtype=jnp.float32)
        return SamplingState(
            jnp.full_like(zeros, maximum_return),
            jnp.zeros_like(self.starts),
            zeros,
            zeros,
            self.uniform_probabilities,
        )

    def bin_index(self, frames):
        return jnp.clip(
            jnp.searchsorted(self.starts, frames, side="right") - 1,
            0,
            len(self.starts) - 1,
        )

    def sample(self, key, probabilities):
        bin_key, frame_key = jax.random.split(key)
        cumulative = jnp.cumsum(probabilities)
        cumulative = cumulative / cumulative[-1]
        index = jnp.searchsorted(cumulative, jax.random.uniform(bin_key), side="right")
        offset = jax.random.randint(frame_key, (), 0, self.lengths[index])
        return (self.starts[index] + offset).astype(jnp.int32)

    def update(self, state, window, maximum_return, gamma, *, axis_name=None):
        next_value = state.returns[window.next_bins]
        mean_value = (
            jnp.sum(jnp.where(self.eligible, state.returns, 0.0)) / self.eligible.sum()
        )
        next_value = jnp.where(window.clip_end, mean_value, next_value)
        next_value = jnp.where(window.terminal | ~window.valid, 0.0, next_value)
        done = window.terminal | window.clip_end | window.timeout | ~window.valid

        def backward(carry, values):
            reward, ended, bootstrap, valid = values
            carry = jnp.where(ended, bootstrap, carry)
            value = jnp.where(valid, reward + gamma * carry, 0.0)
            return value, value

        _, returns = jax.lax.scan(
            backward,
            next_value[:, -1],
            tuple(
                jnp.swapaxes(value, 0, 1)
                for value in (window.rewards, done, next_value, window.valid)
            ),
            reverse=True,
        )
        returns = jnp.swapaxes(returns, 0, 1)
        episode = jnp.cumsum(done.astype(jnp.int32), axis=1) - done
        same = (window.bins[:, :, None] == window.bins[:, None, :]) & (
            episode[:, :, None] == episode[:, None, :]
        )
        same &= window.valid[:, :, None] & window.valid[:, None, :]
        samples = (same * returns[:, None, :]).sum(axis=-1) / jnp.maximum(
            same.sum(axis=-1), 1
        )
        order = jnp.arange(window.bins.shape[1])
        first = window.valid & ~jnp.any(
            same & (order[None, :] < order[:, None]), axis=-1
        )
        bins = window.bins
        if axis_name is not None:
            bins, samples, first = jax.tree.map(
                lambda value: jax.lax.all_gather(
                    value, axis_name, tiled=True, to="invarying"
                ),
                (bins, samples, first),
            )
        bins, samples, first = (value.reshape(-1) for value in (bins, samples, first))
        counts = jnp.zeros_like(state.visits).at[bins].add(first.astype(jnp.int32))
        totals = (
            jnp.zeros_like(state.returns).at[bins].add(jnp.where(first, samples, 0.0))
        )
        visits = jnp.minimum(state.visits, self.config.memory) + counts
        values = state.returns + (totals - counts * state.returns) / jnp.maximum(
            visits, 1
        )
        focus = jax.nn.softmax(
            jnp.where(
                self.eligible,
                -self.config.temperature * values / maximum_return,
                -jnp.inf,
            )
        )
        # Retain thin focus contributions without enabling JAX float64 globally.
        increment = focus**2 - state.effort_compensation
        effort = state.effort + increment
        compensation = (effort - state.effort) - increment
        keep = effort / effort.sum()
        cfg = self.config
        probabilities = (
            cfg.uniform_weight * self.uniform_probabilities
            + cfg.difficult_weight * focus
            + cfg.past_weight * keep
        )
        return replace(
            state,
            returns=values,
            visits=visits,
            effort=effort,
            effort_compensation=compensation,
            probabilities=probabilities,
        )
