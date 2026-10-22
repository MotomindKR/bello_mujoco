"""Differentiable logical proposals constrained in canonical motor coordinates."""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from bello_mujoco.envs.actions import (
    limit_motor_step,
    logical_action_scale,
    project_motor_target,
)
from bello_mujoco.supermimic.config import ControlConfig


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MotorCommand:
    action: jax.Array
    requested: jax.Array
    accepted: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ActionMapping:
    joint_names: tuple[str, ...] = field(metadata={"static": True})
    action_clip: float = field(metadata={"static": True})
    neutral_joint: jax.Array
    neutral_motor: jax.Array
    motor_from_joint: jax.Array
    scale: jax.Array
    matrix: jax.Array
    bounds: jax.Array
    membership: jax.Array
    groups: jax.Array
    position_step: jax.Array

    @classmethod
    def from_model(cls, model, actuator_config, config=ControlConfig()):
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
            limits.matrix,
            limits.bounds,
            limits.membership,
            limits.groups,
            limits.position_step,
        )
        return cls(
            tuple(model.actuator_joint_names),
            config.action_clip,
            *(jnp.asarray(value) for value in arrays),
        )

    def command(self, action, previous_motor_target, calibration_offset):
        count = len(self.joint_names)
        if any(
            value.shape != (count,)
            for value in (action, previous_motor_target, calibration_offset)
        ):
            raise ValueError(
                "action, previous target, and calibration must match joint layout"
            )
        action = jnp.clip(action, -self.action_clip, self.action_clip)
        target = self.neutral_joint + calibration_offset + self.scale * action
        requested = jnp.matmul(
            self.motor_from_joint, target, precision=jax.lax.Precision.HIGHEST
        )
        projected = project_motor_target(requested, self.neutral_motor, self)
        return MotorCommand(
            action, requested, limit_motor_step(projected, previous_motor_target, self)
        )
