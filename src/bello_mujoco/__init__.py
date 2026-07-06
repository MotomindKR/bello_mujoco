"""MuJoCo/MJX scaffolding for Bello locomotion experiments."""

from bello_mujoco.contracts import POLICY_JOINT_ORDER
from bello_mujoco.envs.locomotion import BelloLocomotionEnv
from bello_mujoco.sim import load_bello_model

__all__ = ["BelloLocomotionEnv", "POLICY_JOINT_ORDER", "load_bello_model"]
