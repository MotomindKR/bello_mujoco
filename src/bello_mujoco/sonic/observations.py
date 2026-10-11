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
    qpos_indices: tuple[int, ...]
    qvel_indices: tuple[int, ...]
    future_frames: int = 10
    future_dt: float = 0.1
    smpl_enabled: bool = False

    @property
    def input_reference_size(self) -> int:
        return self.reference_size + (841 if self.smpl_enabled else 0)

    def __post_init__(self) -> None:
        count = self.adapter.target_size
        if (
            len(self.qpos_indices) != count
            or set(self.qpos_indices) != set(range(7, 7 + count))
            or len(self.qvel_indices) != count
            or tuple(i - 1 for i in self.qpos_indices) != self.qvel_indices
        ):
            raise ValueError("SONIC reference indices must map each named scalar joint")

    @property
    def reference_size(self) -> int:
        return self.future_frames * 64

    @property
    def per_step_size(self) -> int:
        return self.input_reference_size + 6 + 3 * self.adapter.target_size

    def human_reference(self, bank, clip_id, local_frame, robot_root, alignment_yaw):
        from bello_mujoco.sonic.smpl import smpl_reference

        if bank.smpl_joints is None or bank.smpl_root_quat is None:
            raise ValueError("SMPL input requires paired human motion in every clip")
        offsets = jnp.rint(jnp.arange(10, dtype=jnp.float32) * 0.02 * bank.fps)
        indices = bank.frame_indices(clip_id, local_frame + offsets)
        positions = bank.qpos[indices][:, jnp.asarray(self.qpos_indices)]
        source = jax.vmap(self.adapter.target_position_to_source)(positions)
        return smpl_reference(
            bank.smpl_joints[indices],
            bank.smpl_root_quat[indices],
            source[:, 23:29],
            robot_root,
            yaw_quat_from_angle(alignment_yaw),
        )

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
        target_positions = samples.qpos[..., jnp.asarray(self.qpos_indices)]
        target_velocities = samples.qvel[..., jnp.asarray(self.qvel_indices)]
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
        if reference.shape != (self.input_reference_size,):
            raise ValueError(
                f"SONIC reference must have shape ({self.input_reference_size},), "
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
