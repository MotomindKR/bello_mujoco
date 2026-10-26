"""Direct PD setpoints in canonical motor coordinates for simulation learning."""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from bello_mujoco.envs.actions import logical_action_scale


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MotorCommand:
    action: jax.Array
    requested: jax.Array
    projected: jax.Array
    accepted: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ActionMapping:
    joint_names: tuple[str, ...] = field(metadata={"static": True})
    neutral_joint: jax.Array
    neutral_motor: jax.Array
    motor_from_joint: jax.Array
    scale: jax.Array

    @classmethod
    def from_model(cls, model, actuator_config):
        count = len(model.actuator_joint_names)
        transform = np.asarray(model.actuator_coordinate_transform)
        if (
            transform.shape != (count, count)
            or not np.isfinite(transform).all()
            or np.linalg.matrix_rank(transform) != count
        ):
            raise ValueError(
                "SuperMimic requires an invertible joint/motor transmission"
            )
        if not np.array_equal(model.policy_ctrl_indices, np.arange(count)):
            raise ValueError("SuperMimic requires every motor to be policy controlled")
        limits = model.target_limits
        limits.validate_reference(model.default_ctrl)
        arrays = (
            model.default_joint_target,
            model.default_ctrl,
            transform,
            logical_action_scale(model, actuator_config),
        )
        return cls(
            tuple(model.actuator_joint_names),
            *(jnp.asarray(value) for value in arrays),
        )

    def command(self, action, calibration_offset):
        count = len(self.joint_names)
        if any(value.shape != (count,) for value in (action, calibration_offset)):
            raise ValueError("action and calibration must match joint layout")
        target = self.neutral_joint + calibration_offset + self.scale * action
        requested = jnp.matmul(
            self.motor_from_joint, target, precision=jax.lax.Precision.HIGHEST
        )
        return MotorCommand(action, requested, requested, requested)


def motor_limit_flags(command, actuator_force, peak_force):
    """Separate command position/rate interventions from observed force clipping."""
    return {
        "position_projected": jnp.abs(command.projected - command.requested) > 1e-6,
        "rate_limited": jnp.abs(command.accepted - command.projected) > 1e-6,
        "torque_saturated": jnp.abs(actuator_force) >= 0.99 * peak_force,
    }
