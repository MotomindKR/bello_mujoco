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
    )
    service = ReferenceService(
        "test", SimpleNamespace(latest=lambda: frame), ["b", "a"], [3, 1], [2, 0], 4, 3
    )
    result = service.get(pb.GetReferenceRequest(robot="test"), Context())
    assert result.available and result.sequence == 17
    assert list(result.joint_names) == ["b", "a"]
    assert list(result.position) == [1.0, 3.0]
    assert list(result.velocity) == [0.2, 0.4]
    assert result.source_age_ms >= 35


@pytest.mark.parametrize(
    "failure", ["outage", "shape", "nonfinite", "age", "robot", "terminal"]
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
