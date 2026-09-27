"""SONIC cross-embodiment transfer primitives."""

from bello_mujoco.sonic.embodiment import (
    G1_JOINT_NAMES,
    MiniEmbodimentAdapter,
)
from bello_mujoco.sonic.reference import ReferenceBank, load_reference_bank

__all__ = [
    "G1_JOINT_NAMES",
    "MiniEmbodimentAdapter",
    "ReferenceBank",
    "load_reference_bank",
]
