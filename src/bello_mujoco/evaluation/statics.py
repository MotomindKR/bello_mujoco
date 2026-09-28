"""Gravity-only equilibrium screens, not dynamic or thermal certification."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from bello_mujoco.models.barbell import two_end_weight_inertia as two_end_weight_inertia


@dataclass(frozen=True)
class StaticEquilibrium:
    motor_torque: np.ndarray
    normal_force: np.ndarray
    maximum_limit_fraction: float
    residual: float


def static_equilibrium(
    motor_map: np.ndarray,
    support_map: np.ndarray,
    required: np.ndarray,
    effort_map: np.ndarray,
    effort_bounds: np.ndarray,
) -> StaticEquilibrium | None:
    """Find the least maximum torque-limit fraction with normal reactions.

    All matrices map forces into generalized coordinates, including the free
    root. No horizontal ground reactions are used. This is an optimistic static
    load distribution, not the distribution a controller will necessarily use.
    Limits are normalized rather than enforced: overload must remain visible.
    """
    nv, nu = motor_map.shape
    if (
        support_map.ndim != 2
        or support_map.shape[0] != nv
        or required.shape != (nv,)
        or effort_map.ndim != 2
        or effort_map.shape[1] != nu
        or effort_bounds.shape != (len(effort_map), 2)
        or len(effort_map) == 0
        or nu == 0
        or np.any(effort_bounds[:, 0] >= 0)
        or np.any(effort_bounds[:, 1] <= 0)
        or not all(
            np.isfinite(x).all()
            for x in (motor_map, support_map, required, effort_map, effort_bounds)
        )
    ):
        raise ValueError("invalid static equilibrium dimensions or values")
    ns = support_map.shape[1]
    objective = np.zeros(nu + ns + 1)
    objective[-1] = 1
    nc = len(effort_map)
    upper = np.zeros((2 * nc, len(objective)))
    upper[:nc, :nu] = effort_map
    upper[nc:, :nu] = -effort_map
    upper[:, -1] = np.r_[-effort_bounds[:, 1], effort_bounds[:, 0]]
    equality = np.column_stack((motor_map, support_map, np.zeros(nv)))
    result = linprog(
        objective,
        A_ub=upper,
        b_ub=np.zeros(2 * nc),
        A_eq=equality,
        b_eq=required,
        bounds=[(None, None)] * nu + [(0, None)] * (ns + 1),
        method="highs",
    )
    if result.status == 2:
        return None
    if not result.success:
        raise RuntimeError(f"static equilibrium solver failed: {result.message}")
    residual = float(np.max(np.abs(equality @ result.x - required)))
    if residual > 1e-6:
        raise RuntimeError(f"static equilibrium residual is too large: {residual}")
    return StaticEquilibrium(
        result.x[:nu], result.x[nu:-1], float(result.x[-1]), residual
    )
