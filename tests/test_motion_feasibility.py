import math
from types import SimpleNamespace

import numpy as np
import pytest

from bello_mujoco.motion.feasibility import (
    interpolate_qpos,
    physical_ctrl_from_logical_target,
    quaternion_distance,
    quaternion_slerp,
    tracked_end_effector_error,
)
from bello_mujoco.robot_tasks import motion_task_definition


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


def test_feasibility_targets_use_differential_transform_and_motor_limits() -> None:
    model = SimpleNamespace(
        actuator_coordinate_transform=np.asarray([[-1.0, 1.0], [1.0, 1.0]]),
        mj_model=SimpleNamespace(
            nu=2,
            actuator_ctrlrange=np.asarray([[-0.5, 0.5], [-0.5, 0.5]]),
        ),
    )

    ctrl = physical_ctrl_from_logical_target(model, np.asarray([0.8, 0.4]))

    np.testing.assert_allclose(ctrl, np.asarray([-0.4, 0.5]))
    with pytest.raises(ValueError, match="logical target has shape"):
        physical_ctrl_from_logical_target(model, np.zeros(3))


def test_feasibility_uses_selected_robot_end_effectors(monkeypatch) -> None:
    task = motion_task_definition("bello_mini")
    body_names = task.default_tracked_body_names
    body_index = {name: index for index, name in enumerate(body_names)}
    reference = np.zeros((1, len(body_names), 3))
    current = np.zeros((2, 3))
    reference[0, body_index["left_wrist_roll_link"]] = (0.1, 0.2, 0.3)
    reference[0, body_index["right_wrist_roll_link"]] = (-0.1, -0.2, 0.3)
    current[0] = (0.1, 0.2, 0.3)
    current[1] = (-0.1, -0.2, 0.8)
    end_effector_ids = dict(
        zip(task.default_end_effector_body_names, range(2), strict=True)
    )
    requested_names = []

    def body_id(_model, _object_type, name):
        requested_names.append(name)
        return end_effector_ids[name]

    monkeypatch.setattr("mujoco.mj_name2id", body_id)
    error = tracked_end_effector_error(
        SimpleNamespace(
            robot=SimpleNamespace(asset="bello_mini"),
            mj_model=object(),
        ),
        SimpleNamespace(xpos=current),
        SimpleNamespace(
            fps=30.0,
            num_frames=1,
            body_names=body_names,
            body_pos=reference,
        ),
        0.0,
    )

    assert requested_names == list(task.default_end_effector_body_names)
    assert error == pytest.approx(0.5)
