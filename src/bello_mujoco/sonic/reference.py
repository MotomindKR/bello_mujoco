"""JAX-compatible multi-clip reference storage and boundary-safe sampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

import jax
import jax.numpy as jnp
import numpy as np

from bello_mujoco.motion.clip import (
    MotionClip,
    MotionSample,
    load_motion_clip,
    quat_slerp,
)

SplitName = Literal["train", "evaluation"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ReferenceBank:
    fps: float = field(metadata={"static": True})
    joint_names: tuple[str, ...] = field(metadata={"static": True})
    body_names: tuple[str, ...] = field(metadata={"static": True})
    clip_names: tuple[str, ...] = field(metadata={"static": True})
    max_clip_frames: int = field(metadata={"static": True})
    clock_end: int = field(metadata={"static": True})
    offsets: jax.Array
    time_offsets: jax.Array
    lengths: jax.Array
    split_ids: jax.Array
    split_counts: tuple[int, int] = field(metadata={"static": True})
    qpos: jax.Array
    qvel: jax.Array
    body_pos: jax.Array
    body_quat: jax.Array
    body_lin_vel: jax.Array
    body_ang_vel: jax.Array
    foot_contacts: jax.Array

    @classmethod
    def from_clips(
        cls,
        clips: Iterable[tuple[str, MotionClip, SplitName]],
    ) -> ReferenceBank:
        entries = tuple(clips)
        if not entries:
            raise ValueError("reference bank requires at least one clip")
        names = tuple(name for name, _, _ in entries)
        if any(not name or Path(name).name != name for name in names):
            raise ValueError("reference clip names must be nonempty path components")
        if len(set(names)) != len(names):
            raise ValueError("reference clip names must be unique")
        first = entries[0][1]
        for name, clip, _ in entries:
            if not np.isclose(clip.fps, first.fps):
                raise ValueError(
                    f"reference clip {name!r} has fps={clip.fps}; expected {first.fps}"
                )
            if clip.joint_names != first.joint_names:
                raise ValueError(f"reference clip {name!r} has a different joint order")
            if clip.body_names != first.body_names:
                raise ValueError(f"reference clip {name!r} has a different body order")
        lengths = np.asarray(
            [clip.num_frames for _, clip, _ in entries], dtype=np.int32
        )
        offsets = np.concatenate((np.zeros(1, dtype=np.int32), np.cumsum(lengths[:-1])))
        # A one-frame separator makes the corpus-global time coordinate safe:
        # an episode that advances slightly beyond its final frame still clamps
        # to that clip instead of observing the first frame of the next clip.
        time_offsets = np.concatenate(
            (
                np.zeros(1, dtype=np.int32),
                np.cumsum((lengths + 1)[:-1]),
            )
        )

        def concatenate(field: str) -> jax.Array:
            return jnp.asarray(
                np.concatenate(
                    tuple(np.asarray(getattr(clip, field)) for _, clip, _ in entries),
                    axis=0,
                )
            )

        split_lookup = {"train": 0, "evaluation": 1}
        return cls(
            fps=first.fps,
            joint_names=first.joint_names,
            body_names=first.body_names,
            clip_names=names,
            max_clip_frames=int(lengths.max()),
            clock_end=int(time_offsets[-1] + lengths[-1]),
            offsets=jnp.asarray(offsets),
            time_offsets=jnp.asarray(time_offsets),
            lengths=jnp.asarray(lengths),
            split_ids=jnp.asarray(
                [split_lookup[split_name] for _, _, split_name in entries]
            ),
            split_counts=(
                sum(split_name == "train" for _, _, split_name in entries),
                sum(split_name == "evaluation" for _, _, split_name in entries),
            ),
            qpos=concatenate("qpos"),
            qvel=concatenate("qvel"),
            body_pos=concatenate("body_pos"),
            body_quat=concatenate("body_quat"),
            body_lin_vel=concatenate("body_lin_vel"),
            body_ang_vel=concatenate("body_ang_vel"),
            foot_contacts=concatenate("foot_contacts"),
        )

    @property
    def num_clips(self) -> int:
        return len(self.clip_names)

    def split_mask(self, split: SplitName) -> jax.Array:
        return self.split_ids == (0 if split == "train" else 1)

    def frame_indices(self, clip_id: jax.Array, local_frames: jax.Array) -> jax.Array:
        clip_id = jnp.asarray(clip_id, dtype=jnp.int32)
        local_frames = jnp.asarray(local_frames)
        bounded = jnp.clip(local_frames, 0, self.lengths[clip_id] - 1)
        return self.offsets[clip_id] + bounded.astype(jnp.int32)

    def sample_frames(
        self, clip_id: jax.Array, local_frames: jax.Array
    ) -> MotionSample:
        index = self.frame_indices(clip_id, local_frames)
        return MotionSample(
            qpos=self.qpos[index],
            qvel=self.qvel[index],
            body_pos=self.body_pos[index],
            body_quat=self.body_quat[index],
            body_lin_vel=self.body_lin_vel[index],
            body_ang_vel=self.body_ang_vel[index],
            foot_contacts=self.foot_contacts[index],
        )

    def sample_time(self, time: jax.Array) -> MotionSample:
        """Interpolate a corpus-global time without crossing a clip boundary."""

        global_frame = jnp.maximum(jnp.asarray(time) * self.fps, 0.0)
        return self.sample_global_frame(global_frame)

    def sample_global_frame(self, global_frame: jax.Array) -> MotionSample:
        """Sample a frame clock without a lossy frame/seconds round trip."""
        global_frame = jnp.maximum(jnp.asarray(global_frame), 0.0)
        clip_id = jnp.clip(
            jnp.searchsorted(self.time_offsets, global_frame, side="right") - 1,
            0,
            self.num_clips - 1,
        )
        local = global_frame - self.time_offsets[clip_id]
        return self.sample_local_frame(clip_id, local)

    def sample_local_frame(
        self, clip_id: jax.Array, local_frame: jax.Array
    ) -> MotionSample:
        """Interpolate one clip-local frame and clamp both endpoints to the clip."""

        clip_id = jnp.asarray(clip_id, dtype=jnp.int32)
        local = jnp.clip(jnp.asarray(local_frame), 0.0, self.lengths[clip_id] - 1)
        lower = jnp.floor(local).astype(jnp.int32)
        upper = jnp.minimum(lower + 1, self.lengths[clip_id] - 1)
        lower = jnp.minimum(lower, self.lengths[clip_id] - 1)
        alpha = jnp.clip(local - lower.astype(local.dtype), 0.0, 1.0)
        a = self.sample_frames(clip_id, lower)
        b = self.sample_frames(clip_id, upper)

        def lerp(x, y):
            return x + alpha * (y - x)

        qpos = (
            lerp(a.qpos, b.qpos)
            .at[3:7]
            .set(quat_slerp(a.qpos[3:7], b.qpos[3:7], alpha))
        )
        body_quat = jax.vmap(quat_slerp, in_axes=(0, 0, None))(
            a.body_quat, b.body_quat, alpha
        )
        return MotionSample(
            qpos=qpos,
            qvel=lerp(a.qvel, b.qvel),
            body_pos=lerp(a.body_pos, b.body_pos),
            body_quat=body_quat,
            body_lin_vel=lerp(a.body_lin_vel, b.body_lin_vel),
            body_ang_vel=lerp(a.body_ang_vel, b.body_ang_vel),
            foot_contacts=jnp.where(alpha < 0.5, a.foot_contacts, b.foot_contacts),
        )

    def window(
        self,
        clip_id: jax.Array,
        local_frame: jax.Array,
        frame_offsets: jax.Array,
    ) -> MotionSample:
        """Gather a window while clamping every index inside its source clip."""

        frames = jnp.asarray(local_frame) + jnp.asarray(frame_offsets)
        return jax.vmap(self.sample_local_frame, in_axes=(None, 0))(clip_id, frames)

    def clip_sampling_probabilities(
        self,
        failure_ema: jax.Array,
        *,
        split: SplitName,
        uniform_ratio: float = 0.1,
    ) -> jax.Array:
        failure_ema = jnp.asarray(failure_ema, dtype=jnp.float32)
        if failure_ema.shape != (self.num_clips,):
            raise ValueError(
                f"failure_ema must have shape ({self.num_clips},), got {failure_ema.shape}"
            )
        eligible = self.split_mask(split)
        if self.split_counts[0 if split == "train" else 1] == 0:
            raise ValueError(f"reference bank has no {split} clips")
        scores = jnp.where(eligible, jnp.maximum(failure_ema, 0.0), 0.0)
        eligible_count = jnp.sum(eligible)
        uniform = eligible.astype(jnp.float32) / eligible_count
        adaptive = jnp.where(jnp.sum(scores) > 0, scores / jnp.sum(scores), uniform)
        return uniform_ratio * uniform + (1.0 - uniform_ratio) * adaptive

    def sample_clip(
        self,
        rng: jax.Array,
        failure_ema: jax.Array,
        *,
        split: SplitName,
        uniform_ratio: float = 0.1,
    ) -> jax.Array:
        probabilities = self.clip_sampling_probabilities(
            failure_ema, split=split, uniform_ratio=uniform_ratio
        )
        return jax.random.categorical(rng, jnp.log(probabilities)).astype(jnp.int32)


def load_reference_bank(
    paths: Iterable[Path],
    model,
    *,
    evaluation_names: Iterable[str] = (),
) -> ReferenceBank:
    paths = tuple(Path(path).expanduser().resolve() for path in paths)
    evaluation = frozenset(evaluation_names)
    clips = []
    # Validate/load on the host, then transfer each concatenated field once.
    for path in paths:
        split: SplitName = "evaluation" if path.stem in evaluation else "train"
        clips.append((path.stem, load_motion_clip(path, model, as_numpy=True), split))
    unknown = sorted(evaluation - {path.stem for path in paths})
    if unknown:
        raise ValueError(f"evaluation clips were not found in the corpus: {unknown}")
    return ReferenceBank.from_clips(clips)
