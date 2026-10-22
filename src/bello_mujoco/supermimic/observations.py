"""Actor samples and consecutive-frame goals without pretrained G1 coordinates."""

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
        return (
            3 * self.neutral_joint.shape[0] + 6 + 6 * self.config.reference_orientation
        )

    @property
    def goal_size(self):
        return (15 + 2 * self.neutral_joint.shape[0]) * self.config.future_frames

    def sample(
        self, feedback, previous_action, reference_quaternion, calibration_offset
    ):
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
            feedback.joint_position - self.neutral_joint - calibration_offset,
            feedback.joint_velocity,
            rot.rotate(inverse, feedback.angular_velocity),
            rot.rotate(inverse, jnp.asarray([0.0, 0.0, -1.0])),
            previous_action,
        ]
        if self.config.reference_orientation:
            parts.append(rot.rotation_6d(rot.multiply(inverse, reference_quaternion)))
        return jnp.concatenate(parts)

    def goal(self, current, future):
        if future.position.shape != (self.config.future_frames, 3):
            raise ValueError("reference window does not match future_frames")
        inverse_heading = rot.conjugate(rot.heading(current.quaternion))
        parts = (
            rot.rotate(inverse_heading, future.position - current.position),
            rot.rotation_6d(rot.multiply(inverse_heading, future.quaternion)),
            rot.rotate(rot.conjugate(future.quaternion), future.linear_velocity),
            rot.rotate(rot.conjugate(future.quaternion), future.angular_velocity),
            future.joint_position,
            future.joint_velocity * self.config.goal_joint_velocity_scale,
        )
        return jnp.concatenate(parts, axis=-1).reshape(-1)

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
