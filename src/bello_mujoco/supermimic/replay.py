"""Fixed-capacity device replay with episode-safe windows and actor histories."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from bello_mujoco.supermimic.state import PhysicalState


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Transition:
    state: PhysicalState
    successor: PhysicalState
    filtered_feedback: PhysicalState
    previous_motor: jax.Array
    motor: jax.Array
    action: jax.Array
    previous_action: jax.Array
    observation: jax.Array
    context: jax.Array
    alignment_xy: jax.Array
    alignment_yaw: jax.Array
    reference_frame: jax.Array
    clip_id: jax.Array
    episode_id: jax.Array
    episode_step: jax.Array
    remaining_steps: jax.Array
    reward: jax.Array
    terminal: jax.Array
    timeout: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ReplayWindow:
    transitions: Transition
    mask: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ReplayStart:
    transition: Transition
    history: jax.Array
    feedback_history: PhysicalState
    valid: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ReplayBuffer:
    data: Transition
    valid: jax.Array
    cursor: jax.Array
    size: jax.Array

    @classmethod
    def empty(cls, example, capacity):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("replay capacity must be a positive integer")
        leaves = jax.tree.leaves(example)
        if any(
            value.ndim == 0 or value.shape[0] != leaves[0].shape[0] for value in leaves
        ):
            raise ValueError("replay example fields require the same environment batch")
        return cls(
            jax.tree.map(lambda x: jnp.zeros((capacity,) + x.shape, x.dtype), example),
            jnp.zeros((capacity, leaves[0].shape[0]), bool),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(0, jnp.int32),
        )

    @property
    def capacity(self):
        return self.data.reward.shape[0]

    @property
    def num_envs(self):
        return self.data.reward.shape[1]

    def insert(self, transition, valid=None):
        if valid is None:
            valid = jnp.ones(self.num_envs, bool)
        transition = jax.tree.map(
            lambda x: jnp.where(
                valid.reshape(valid.shape + (1,) * (x.ndim - 1)), x, jnp.zeros_like(x)
            ),
            transition,
        )
        values = jax.tree.map(
            lambda x, y: x.at[self.cursor].set(y), self.data, transition
        )
        return ReplayBuffer(
            values,
            self.valid.at[self.cursor].set(valid),
            (self.cursor + 1) % self.capacity,
            jnp.minimum(self.size + 1, self.capacity),
        )

    def rows(self, age):
        return (self.cursor - self.size + age) % self.capacity

    def sample_windows(self, key, batch_size, length):
        if not 2 <= length <= self.capacity or batch_size <= 0:
            raise ValueError("window length and batch size exceed replay dimensions")
        time_key, env_key = jax.random.split(key)
        ages = jnp.arange(self.capacity)
        rows = self.rows(ages)
        following = self.rows(ages + 1)
        boundary = (
            self.data.terminal[rows]
            | self.data.timeout[rows]
            | ~self.valid[rows]
            | ~self.valid[following]
            | (self.data.episode_id[rows] != self.data.episode_id[following])
            | (self.data.clip_id[rows] != self.data.clip_id[following])
            | jnp.any(self.data.context[rows] != self.data.context[following], axis=-1)
        )
        prefix = jnp.concatenate(
            (jnp.zeros((1, self.num_envs), jnp.int32), jnp.cumsum(boundary, axis=0)),
            axis=0,
        )
        end = jnp.minimum(ages + length - 1, self.capacity)
        eligible = (
            (ages[:, None] + length <= self.size)
            & self.valid[rows]
            & (prefix[end] == prefix[ages])
        )
        counts = jnp.cumsum(eligible.T, axis=1, dtype=jnp.int32)
        totals = counts[:, -1]
        available = totals > 0
        probability = available / jnp.maximum(available.sum(), 1)
        probability = jnp.where(
            available.any(), probability, jnp.full(self.num_envs, 1 / self.num_envs)
        )
        env = jax.random.choice(env_key, self.num_envs, (batch_size,), p=probability)
        draw = jax.random.randint(
            time_key, (batch_size,), 0, jnp.maximum(totals[env], 1)
        )
        age = jax.vmap(
            lambda count, value: jnp.searchsorted(count, value, side="right")
        )(counts[env], draw)
        age = jnp.minimum(age, self.capacity - length)
        ages = age[:, None] + jnp.arange(length - 1)
        rows = self.rows(ages)
        transitions = jax.tree.map(lambda x: x[rows, env[:, None]], self.data)
        mask = jnp.broadcast_to(available.any(), ages.shape).astype(jnp.float32)
        return ReplayWindow(transitions, mask)

    def sample_starts(self, key, batch_size, *, history_frames, feedback_frames):
        if min(batch_size, history_frames, feedback_frames) <= 0:
            raise ValueError("start sampling dimensions must be positive")
        if max(history_frames, feedback_frames) > self.capacity:
            raise ValueError("replay cannot reconstruct the requested history")
        ages = jnp.arange(self.capacity)[:, None]
        rows = self.rows(ages[:, 0])
        eligible = (
            (ages < self.size)
            & self.valid[rows]
            & (self.data.remaining_steps[rows] > 0)
        )
        counts = jnp.cumsum(eligible.reshape(-1), dtype=jnp.int32)
        count = counts[-1]
        draw = jax.random.randint(key, (batch_size,), 0, jnp.maximum(1, count))
        flat = jnp.searchsorted(counts, draw, side="right")
        flat = jnp.minimum(flat, counts.size - 1)
        age, env = flat // self.num_envs, flat % self.num_envs
        row = self.rows(age)
        transition = jax.tree.map(lambda x: x[row, env], self.data)

        def history_indices(length, newest_first=False):
            offsets = (
                jnp.arange(length) if newest_first else jnp.arange(length - 1, -1, -1)
            )
            back = jnp.minimum(offsets[None], transition.episode_step[:, None])
            back = jnp.minimum(back, age[:, None])
            return self.rows(age[:, None] - back)

        history = self.data.observation[history_indices(history_frames), env[:, None]]
        feedback = jax.tree.map(
            lambda x: x[history_indices(feedback_frames, True), env[:, None]],
            self.data.filtered_feedback,
        )
        return ReplayStart(transition, history, feedback, count > 0)
