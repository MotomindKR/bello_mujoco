"""Reference and proprioception assembly shared by training and deployment."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from bello_mujoco.envs.geometry import quat_rotate_inverse
from bello_mujoco.envs.motion_tracking import (
    quat_conjugate,
    quat_multiply,
    rotation_6d,
    yaw_quat_from_angle,
)
from bello_mujoco.sonic.embodiment import MiniEmbodimentAdapter
from bello_mujoco.sonic.reference import ReferenceBank


@dataclass(frozen=True)
class SonicObservationAssembler:
    adapter: MiniEmbodimentAdapter
    future_frames: int = 10
    future_dt: float = 0.1

    @property
    def reference_size(self) -> int:
        return self.future_frames * 64

    @property
    def per_step_size(self) -> int:
        return self.reference_size + 6 + 3 * self.adapter.target_size

    def reference(
        self,
        bank: ReferenceBank,
        clip_id: jax.Array,
        local_frame: jax.Array,
        robot_anchor_quaternion: jax.Array,
        alignment_yaw: jax.Array = jnp.asarray(0.0, dtype=jnp.float32),
    ) -> jax.Array:
        offsets = jnp.rint(
            jnp.arange(self.future_frames, dtype=jnp.float32)
            * self.future_dt
            * bank.fps
        ).astype(jnp.int32)
        samples = bank.window(clip_id, local_frame, offsets)
        target_positions = samples.qpos[..., 7:]
        target_velocities = samples.qvel[..., 6:]
        source_positions = jax.vmap(self.adapter.target_position_to_source)(
            target_positions
        )
        source_velocities = jax.vmap(self.adapter.target_vector_to_source)(
            target_velocities
        )
        alignment = yaw_quat_from_angle(alignment_yaw)
        aligned_root_quaternion = jax.vmap(lambda quat: quat_multiply(alignment, quat))(
            samples.qpos[..., 3:7]
        )
        root_relative = jax.vmap(
            lambda quat: rotation_6d(
                quat_multiply(quat_conjugate(robot_anchor_quaternion), quat)
            )
        )(aligned_root_quaternion)
        # The release observation first concatenates the complete t-major position
        # and velocity blocks, then exposes that flat buffer as (future_frames, 58).
        # This means each row does not represent one time step.  Preserve that
        # historical layout because the frozen encoder was trained with it.
        joint_command = jnp.concatenate(
            (source_positions.reshape(-1), source_velocities.reshape(-1))
        ).reshape(self.future_frames, 2 * self.adapter.source_size)
        return jnp.concatenate((joint_command, root_relative), axis=-1).reshape(-1)

    def step(
        self,
        reference: jax.Array,
        *,
        root_quaternion: jax.Array,
        root_angular_velocity: jax.Array,
        logical_joint_position: jax.Array,
        logical_joint_velocity: jax.Array,
        previous_action: jax.Array,
    ) -> jax.Array:
        reference = jnp.asarray(reference)
        if reference.shape != (self.reference_size,):
            raise ValueError(
                f"SONIC reference must have shape ({self.reference_size},), "
                f"got {reference.shape}"
            )
        gravity = quat_rotate_inverse(
            root_quaternion, jnp.asarray([0.0, 0.0, -1.0], dtype=jnp.float32)
        )
        return jnp.concatenate(
            (
                reference,
                gravity,
                root_angular_velocity,
                logical_joint_position,
                logical_joint_velocity,
                previous_action,
            )
        )
