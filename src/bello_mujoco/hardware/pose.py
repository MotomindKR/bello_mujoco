"""Latest-wins implementation of the neutral Bello pose stream."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
from threading import Condition
import time
from types import TracebackType
from typing import Iterator

import grpc

from bello_mujoco.hardware import _pose_pb2


@dataclass(frozen=True)
class PoseServiceConfig:
    """Configuration for the runner's simulation-ground-truth pose service."""

    enabled: bool = True
    listen: str = "0.0.0.0:50052"
    reference_frame: str = "odom"
    tracked_frame: str = "pelvis_link"
    publish_rate_hz: float = 200.0


@dataclass(frozen=True)
class _PoseSample:
    sequence: int
    translation: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    linear_velocity: tuple[float, float, float]
    sample_monotonic_ns: int


class PoseServiceServer:
    """Threaded gRPC server fed by nonblocking simulation-loop publications."""

    def __init__(self, config: PoseServiceConfig) -> None:
        self.config = config
        self._condition = Condition()
        self._latest: _PoseSample | None = None
        self._closed = False
        self._sequence = 0
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._server = grpc.server(self._executor)
        handler = grpc.method_handlers_generic_handler(
            "bello.pose.v1.PoseService",
            {
                "WatchPose": grpc.unary_stream_rpc_method_handler(
                    self._watch_pose,
                    request_deserializer=_pose_pb2.WatchPoseRequest.FromString,
                    response_serializer=_pose_pb2.WatchPoseResponse.SerializeToString,
                )
            },
        )
        self._server.add_generic_rpc_handlers((handler,))
        self.port = self._server.add_insecure_port(config.listen)
        if self.port == 0:
            self._executor.shutdown(wait=False, cancel_futures=True)
            raise RuntimeError(f"cannot bind PoseService to {config.listen!r}")
        self._started = False

    def __enter__(self) -> PoseServiceServer:
        self._server.start()
        self._started = True
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._started:
            self._server.stop(grace=1.0).wait(timeout=2.0)
        self._executor.shutdown(wait=True, cancel_futures=True)

    def publish(
        self,
        translation: tuple[float, float, float],
        orientation_wxyz: tuple[float, float, float, float],
        linear_velocity: tuple[float, float, float],
        *,
        sample_monotonic_ns: int,
    ) -> None:
        """Publish one world pose and velocity without waiting for a consumer."""

        values = (*translation, *orientation_wxyz, *linear_velocity)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                "pose translation, orientation, and velocity must be finite"
            )
        norm = math.sqrt(sum(value * value for value in orientation_wxyz))
        if norm <= 1.0e-12:
            raise ValueError("pose orientation must have nonzero norm")
        w, x, y, z = (value / norm for value in orientation_wxyz)
        if sample_monotonic_ns <= 0:
            raise ValueError("pose source timestamp must be positive")
        with self._condition:
            if self._closed:
                return
            self._sequence += 1
            self._latest = _PoseSample(
                sequence=self._sequence,
                translation=translation,
                orientation_xyzw=(x, y, z, w),
                linear_velocity=linear_velocity,
                sample_monotonic_ns=sample_monotonic_ns,
            )
            self._condition.notify_all()

    def _watch_pose(
        self,
        request: _pose_pb2.WatchPoseRequest,
        context: grpc.ServicerContext,
    ) -> Iterator[_pose_pb2.WatchPoseResponse]:
        if (
            request.reference_frame != self.config.reference_frame
            or request.tracked_frame != self.config.tracked_frame
        ):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "configured pose is "
                f"{self.config.reference_frame}_from_{self.config.tracked_frame}; "
                "requested "
                f"{request.reference_frame}_from_{request.tracked_frame}",
            )

        last_sequence = 0
        while context.is_active():
            with self._condition:
                self._condition.wait_for(
                    lambda: (
                        self._closed
                        or (
                            self._latest is not None
                            and self._latest.sequence != last_sequence
                        )
                    ),
                    timeout=0.5,
                )
                if self._closed:
                    return
                sample = self._latest
            if sample is None or sample.sequence == last_sequence:
                continue
            last_sequence = sample.sequence
            age_ms = max(
                0, (time.monotonic_ns() - sample.sample_monotonic_ns) // 1_000_000
            )
            yield _pose_pb2.WatchPoseResponse(
                reference_frame=self.config.reference_frame,
                tracked_frame=self.config.tracked_frame,
                sequence=sample.sequence,
                translation_metres=_pose_pb2.Translation(
                    x=sample.translation[0],
                    y=sample.translation[1],
                    z=sample.translation[2],
                ),
                orientation_xyzw=_pose_pb2.Orientation(
                    x=sample.orientation_xyzw[0],
                    y=sample.orientation_xyzw[1],
                    z=sample.orientation_xyzw[2],
                    w=sample.orientation_xyzw[3],
                ),
                sample_age_ms=age_ms,
                linear_velocity_metres_per_second=_pose_pb2.LinearVelocity(
                    x=sample.linear_velocity[0],
                    y=sample.linear_velocity[1],
                    z=sample.linear_velocity[2],
                ),
                sample_monotonic_ns=sample.sample_monotonic_ns,
            )
