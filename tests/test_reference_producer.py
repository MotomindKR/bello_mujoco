"""Joint-reference wire behavior with a synthetic source; no robot assets/GPU."""

from types import SimpleNamespace
import time

import grpc
import numpy as np
import pytest

from bello_mujoco.hardware.reference import RobotReferenceFrame, ReferenceUnavailable
from bello_mujoco.hardware.reference_producer import ReferenceService
from bello_mujoco.hardware import reference_pb2 as pb


class Context:
    def abort(self, code, message):
        raise ValueError(code, message)


def test_named_reference_keeps_joint_order_units_and_source_age():
    frame = RobotReferenceFrame(
        np.array([4.0, 3.0, 2.0, 1.0]),
        np.array([0.4, 0.3, 0.2]),
        17,
        25,
        time.monotonic() - 0.01,
        np.array([0.1, -0.2, 0.3]),
    )
    service = ReferenceService(
        "test", SimpleNamespace(latest=lambda: frame), ["b", "a"], [3, 1], [2, 0], 4, 3
    )
    result = service.get(pb.GetReferenceRequest(robot="test"), Context())
    assert result.available and result.sequence == 17
    assert list(result.joint_names) == ["b", "a"]
    assert list(result.position) == [1.0, 3.0]
    assert list(result.velocity) == [0.2, 0.4]
    assert list(result.torso_command) == [0.1, -0.2, 0.3]
    assert result.source_age_ms >= 35


@pytest.mark.parametrize(
    "failure",
    [
        "outage",
        "shape",
        "nonfinite",
        "age",
        "robot",
        "terminal",
        "torso_shape",
        "torso_nan",
    ],
)
def test_unavailability_is_distinct_from_malformed_or_failed_producer(failure):
    def latest():
        if failure == "outage":
            raise ReferenceUnavailable("lost subject")
        if failure == "terminal":
            raise RuntimeError("retargeter failed")
        return RobotReferenceFrame(
            np.array([np.nan if failure == "nonfinite" else 0.0]),
            np.zeros(2 if failure == "shape" else 1),
            1,
            None if failure == "age" else 0,
            time.monotonic(),
            np.full(
                2 if failure == "torso_shape" else 3,
                np.nan if failure == "torso_nan" else 0.0,
            ),
        )

    service = ReferenceService(
        "test", SimpleNamespace(latest=latest), ["q"], [0], [0], 1, 1
    )
    request = pb.GetReferenceRequest(robot="wrong" if failure == "robot" else "test")
    if failure == "outage":
        assert not service.get(request, Context()).available
    else:
        with pytest.raises(ValueError) as error:
            service.get(request, Context())
        assert error.value.args[0] == (
            grpc.StatusCode.INVALID_ARGUMENT
            if failure == "robot"
            else grpc.StatusCode.INTERNAL
        )


def test_full_body_and_held_joints_come_from_the_same_frame():
    frame = RobotReferenceFrame(
        np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.3, 0.4]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.6]),
        19,
        0,
        time.monotonic(),
        np.zeros(3),
    )
    service = ReferenceService(
        "test",
        SimpleNamespace(latest=lambda: frame),
        ["b"],
        [8],
        [7],
        9,
        8,
        full_body=(["a", "b"], [7, 8], [6, 7]),
    )
    result = service.get(pb.GetReferenceRequest(robot="test"), Context())
    assert result.sequence == 19
    assert list(result.position) == [0.4]
    assert list(result.full_body.joint_names) == ["a", "b"]
    assert list(result.full_body.position) == [0.3, 0.4]
    assert list(result.full_body.velocity) == [0.5, 0.6]
    assert list(result.full_body.root_orientation_wxyz) == [1.0, 0.0, 0.0, 0.0]
    frame.qpos[3:7] = 0
    with pytest.raises(ValueError, match="invalid root orientation"):
        service.get(pb.GetReferenceRequest(robot="test"), Context())


@pytest.mark.parametrize("indices", [[6, 8], [7, 7], [7, 9], [7.0, 8.0]])
def test_full_body_indices_are_validated_before_serving(indices):
    with pytest.raises(ValueError, match="full-body joint indices"):
        ReferenceService(
            "test",
            None,
            ["b"],
            [8],
            [7],
            9,
            8,
            full_body=(["a", "b"], indices, [6, 7]),
        )
