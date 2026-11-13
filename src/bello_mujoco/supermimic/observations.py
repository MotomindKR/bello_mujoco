"""Actor samples and strided reference goals without pretrained G1 coordinates."""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from bello_mujoco.supermimic import rotations as rot
from bello_mujoco.supermimic.config import ObservationConfig


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ObservationCodec:
    config: ObservationConfig = field(metadata={"static": True})
    neutral_joint: jax.Array

    @property
    def sample_size(self):
        return 3 * self.neutral_joint.shape[0] + 6

    @property
    def goal_size(self):
        return (9 + 2 * self.neutral_joint.shape[0]) * self.config.future_frames

    @property
    def actor_goal_size(self):
        return self.goal_size - 3 * self.config.future_frames

    def actor_goal(self, goal):
        frames = self.config.future_frames
        return jnp.concatenate(
            (goal[..., : -9 * frames], goal[..., -6 * frames :]), axis=-1
        )

    def sample(self, feedback, previous_action, calibration_offset):
        count = self.neutral_joint.shape[0]
        if any(
            value.shape != (count,)
            for value in (
                feedback.joint_position,
                feedback.joint_velocity,
                previous_action,
                calibration_offset,
            )
        ):
            raise ValueError("actor feedback does not match logical joint layout")
        inverse = rot.conjugate(feedback.quaternion)
        parts = [
            rot.rotate(inverse, jnp.asarray([0.0, 0.0, -1.0])),
            rot.rotate(inverse, feedback.angular_velocity),
            feedback.joint_position - self.neutral_joint - calibration_offset,
            feedback.joint_velocity,
            previous_action,
        ]
        return jnp.concatenate(parts)

    def goal(self, current, future):
        if future.position.shape != (self.config.future_frames, 3):
            raise ValueError("reference window does not match future_frames")
        inverse = rot.conjugate(current.quaternion)
        parts = (
            jnp.concatenate(
                (future.joint_position, future.joint_velocity), axis=-1
            ).reshape(-1),
            rot.rotate(inverse, future.position - current.position).reshape(-1),
            rot.rotation_6d(rot.multiply(inverse, future.quaternion)).reshape(-1),
        )
        return jnp.concatenate(parts)

    def initial_history(self, sample):
        if sample.shape != (self.sample_size,):
            raise ValueError("sample does not match observation layout")
        return jnp.broadcast_to(sample, (self.config.history_frames, self.sample_size))

    def advance_history(self, history, sample):
        if history.shape != (self.config.history_frames, self.sample_size):
            raise ValueError("history does not match observation layout")
        if sample.shape != (self.sample_size,):
            raise ValueError("sample does not match observation layout")
        return jnp.concatenate((history[1:], sample[None]))
