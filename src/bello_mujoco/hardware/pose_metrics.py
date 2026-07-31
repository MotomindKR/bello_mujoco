"""Compare an estimated PoseService stream with MuJoCo ground truth."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import threading
import time

import grpc
import numpy as np

from bello_mujoco.hardware import _pose_pb2


@dataclass(frozen=True)
class PoseObservation:
    source_time: float
    position: np.ndarray
    orientation_xyzw: np.ndarray


@dataclass(frozen=True)
class PoseErrorMetrics:
    sample_count: int
    duration_seconds: float
    position_rmse_xyz_m: tuple[float, float, float]
    position_rmse_m: float
    maximum_position_error_m: float
    final_position_error_xyz_m: tuple[float, float, float]
    absolute_z_rmse_m: float
    orientation_rmse_radians: float
    maximum_orientation_error_radians: float
    maximum_pair_time_error_ms: float


def calculate_pose_error_metrics(
    estimates: list[PoseObservation],
    ground_truth: list[PoseObservation],
    *,
    maximum_pair_time_error: float,
) -> PoseErrorMetrics:
    if maximum_pair_time_error <= 0.0 or not math.isfinite(maximum_pair_time_error):
        raise ValueError("maximum_pair_time_error must be positive and finite")
    if not estimates or not ground_truth:
        raise ValueError("both pose streams must contain samples")

    truth_times = np.asarray(
        [sample.source_time for sample in ground_truth], dtype=np.float64
    )
    pairs: list[tuple[PoseObservation, PoseObservation, float]] = []
    for estimate in estimates:
        insertion = int(np.searchsorted(truth_times, estimate.source_time))
        candidates = [
            index
            for index in (insertion - 1, insertion)
            if 0 <= index < len(ground_truth)
        ]
        nearest = min(
            candidates,
            key=lambda index: abs(truth_times[index] - estimate.source_time),
        )
        time_error = abs(truth_times[nearest] - estimate.source_time)
        if time_error <= maximum_pair_time_error:
            pairs.append((estimate, ground_truth[nearest], time_error))
    if len(pairs) < 2:
        raise ValueError("fewer than two time-aligned pose pairs were available")

    initial_offset = pairs[0][1].position - pairs[0][0].position
    position_errors = np.stack(
        [
            estimate.position + initial_offset - truth.position
            for estimate, truth, _time_error in pairs
        ]
    )
    absolute_z_errors = np.asarray(
        [
            estimate.position[2] - truth.position[2]
            for estimate, truth, _time_error in pairs
        ],
        dtype=np.float64,
    )
    orientation_errors = np.asarray(
        [
            _quaternion_distance(
                estimate.orientation_xyzw,
                truth.orientation_xyzw,
            )
            for estimate, truth, _time_error in pairs
        ],
        dtype=np.float64,
    )
    norms = np.linalg.norm(position_errors, axis=1)
    rmse_xyz = np.sqrt(np.mean(np.square(position_errors), axis=0))
    duration = pairs[-1][0].source_time - pairs[0][0].source_time
    return PoseErrorMetrics(
        sample_count=len(pairs),
        duration_seconds=max(0.0, duration),
        position_rmse_xyz_m=tuple(float(value) for value in rmse_xyz),
        position_rmse_m=float(np.sqrt(np.mean(np.square(norms)))),
        maximum_position_error_m=float(np.max(norms)),
        final_position_error_xyz_m=tuple(float(value) for value in position_errors[-1]),
        absolute_z_rmse_m=float(np.sqrt(np.mean(np.square(absolute_z_errors)))),
        orientation_rmse_radians=float(np.sqrt(np.mean(np.square(orientation_errors)))),
        maximum_orientation_error_radians=float(np.max(orientation_errors)),
        maximum_pair_time_error_ms=1000.0
        * max(time_error for _estimate, _truth, time_error in pairs),
    )


def collect_pose_stream(
    endpoint: str,
    *,
    duration_seconds: float,
    reference_frame: str,
    tracked_frame: str,
) -> list[PoseObservation]:
    if duration_seconds <= 0.0 or not math.isfinite(duration_seconds):
        raise ValueError("duration_seconds must be positive and finite")
    endpoint = endpoint.removeprefix("http://")
    if endpoint.startswith("https://"):
        raise ValueError("pose metrics currently require local insecure endpoints")
    observations: list[PoseObservation] = []
    with grpc.insecure_channel(endpoint) as channel:
        method = channel.unary_stream(
            "/bello.pose.v1.PoseService/WatchPose",
            request_serializer=lambda value: value.SerializeToString(),
            response_deserializer=_pose_pb2.WatchPoseResponse.FromString,
        )
        request = _pose_pb2.WatchPoseRequest(
            reference_frame=reference_frame,
            tracked_frame=tracked_frame,
        )
        call = method(request, timeout=duration_seconds + 2.0)
        deadline = time.monotonic() + duration_seconds
        previous_sequence: int | None = None
        try:
            for response in call:
                received = time.monotonic()
                if (
                    response.reference_frame != reference_frame
                    or response.tracked_frame != tracked_frame
                ):
                    raise RuntimeError(
                        "PoseService response frame does not match the request"
                    )
                sequence = int(response.sequence)
                if previous_sequence is not None and sequence <= previous_sequence:
                    raise RuntimeError("PoseService sequence did not increase")
                previous_sequence = sequence
                observations.append(
                    PoseObservation(
                        source_time=received - float(response.sample_age_ms) / 1000.0,
                        position=np.asarray(
                            (
                                response.translation_metres.x,
                                response.translation_metres.y,
                                response.translation_metres.z,
                            ),
                            dtype=np.float64,
                        ),
                        orientation_xyzw=_normalized_quaternion(
                            np.asarray(
                                (
                                    response.orientation_xyzw.x,
                                    response.orientation_xyzw.y,
                                    response.orientation_xyzw.z,
                                    response.orientation_xyzw.w,
                                ),
                                dtype=np.float64,
                            )
                        ),
                    )
                )
                if received >= deadline:
                    call.cancel()
                    break
        except grpc.RpcError as error:
            if error.code() not in {
                grpc.StatusCode.CANCELLED,
                grpc.StatusCode.DEADLINE_EXCEEDED,
            }:
                raise
    return observations


def compare_live_pose_streams(
    *,
    estimator_endpoint: str,
    ground_truth_endpoint: str,
    duration_seconds: float,
    maximum_pair_time_error: float,
    reference_frame: str = "odom",
    tracked_frame: str = "pelvis_link",
) -> PoseErrorMetrics:
    results: dict[str, list[PoseObservation]] = {}
    errors: list[BaseException] = []

    def collect(name: str, endpoint: str) -> None:
        try:
            results[name] = collect_pose_stream(
                endpoint,
                duration_seconds=duration_seconds,
                reference_frame=reference_frame,
                tracked_frame=tracked_frame,
            )
        except BaseException as error:
            errors.append(error)

    threads = [
        threading.Thread(
            target=collect,
            args=("estimator", estimator_endpoint),
            name="pose-metrics-estimator",
        ),
        threading.Thread(
            target=collect,
            args=("ground_truth", ground_truth_endpoint),
            name="pose-metrics-ground-truth",
        ),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=duration_seconds + 5.0)
    if any(thread.is_alive() for thread in threads):
        raise TimeoutError("pose metrics collection did not stop on time")
    if errors:
        raise RuntimeError("pose metrics collection failed") from errors[0]
    return calculate_pose_error_metrics(
        results.get("estimator", []),
        results.get("ground_truth", []),
        maximum_pair_time_error=maximum_pair_time_error,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare bello-state-estimatord with MuJoCo PoseService."
    )
    parser.add_argument("--estimator-endpoint", default="127.0.0.1:50053")
    parser.add_argument("--ground-truth-endpoint", default="127.0.0.1:50052")
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--maximum-pair-time-error-ms", type=float, default=20.0)
    arguments = parser.parse_args()
    metrics = compare_live_pose_streams(
        estimator_endpoint=arguments.estimator_endpoint,
        ground_truth_endpoint=arguments.ground_truth_endpoint,
        duration_seconds=arguments.duration_seconds,
        maximum_pair_time_error=arguments.maximum_pair_time_error_ms / 1000.0,
    )
    print(json.dumps(asdict(metrics), indent=2))


def _normalized_quaternion(quaternion: np.ndarray) -> np.ndarray:
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("pose orientation must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise ValueError("pose orientation must have nonzero norm")
    return quaternion / norm


def _quaternion_distance(left: np.ndarray, right: np.ndarray) -> float:
    dot = float(
        np.clip(
            abs(np.dot(_normalized_quaternion(left), _normalized_quaternion(right))),
            0.0,
            1.0,
        )
    )
    return 2.0 * math.acos(dot)


if __name__ == "__main__":
    main()
