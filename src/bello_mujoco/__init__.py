"""MuJoCo/MJX scaffolding for Bello locomotion experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bello_mujoco.envs.locomotion import BelloLocomotionEnv
    from bello_mujoco.sim import load_bello_model

__all__ = ["BelloLocomotionEnv", "load_bello_model"]


def __getattr__(name: str):
    if name == "BelloLocomotionEnv":
        from bello_mujoco.envs.locomotion import BelloLocomotionEnv

        globals()[name] = BelloLocomotionEnv
        return BelloLocomotionEnv
    if name == "load_bello_model":
        from bello_mujoco.sim import load_bello_model

        globals()[name] = load_bello_model
        return load_bello_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
