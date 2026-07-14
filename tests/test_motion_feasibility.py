import math

import numpy as np
import pytest

from bello_mujoco.tools.check_motion_feasibility import (
    interpolate_qpos,
    quaternion_distance,
    quaternion_slerp,
)


def test_quaternion_slerp_uses_shortest_arc() -> None:
    start = np.asarray([1.0, 0.0, 0.0, 0.0])
    end = -np.asarray([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])

    midpoint = quaternion_slerp(start, end, 0.5)

    assert quaternion_distance(start, midpoint) == pytest.approx(math.pi / 4)


def test_interpolate_qpos_slerps_root_quaternion() -> None:
    qpos = np.zeros((2, 8))
    qpos[0, 3] = 1.0
    qpos[1, 6] = 1.0
    qpos[1, 7] = 2.0

    midpoint = interpolate_qpos(qpos, fps=1.0, time=0.5)

    assert np.linalg.norm(midpoint[3:7]) == pytest.approx(1.0)
    assert quaternion_distance(qpos[0, 3:7], midpoint[3:7]) == pytest.approx(
        math.pi / 2
    )
    assert midpoint[7] == pytest.approx(1.0)
