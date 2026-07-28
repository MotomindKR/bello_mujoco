import time

import grpc
import pytest

from bello_mujoco.hardware import _pose_pb2
from bello_mujoco.hardware.pose import PoseServiceConfig, PoseServiceServer


def pose_stream(channel: grpc.Channel):
    return channel.unary_stream(
        "/bello.pose.v1.PoseService/WatchPose",
        request_serializer=_pose_pb2.WatchPoseRequest.SerializeToString,
        response_deserializer=_pose_pb2.WatchPoseResponse.FromString,
    )


def test_pose_service_streams_latest_world_pose_in_xyzw_order() -> None:
    config = PoseServiceConfig(listen="127.0.0.1:0")
    with PoseServiceServer(config) as server:
        server.publish((1.0, 2.0, 3.0), (1.0, 0.0, 0.0, 0.0))
        server.publish(
            (4.0, 5.0, 6.0),
            (2.0, 0.0, 0.0, 0.0),
            produced_at=time.monotonic() - 0.02,
        )
        with grpc.insecure_channel(f"127.0.0.1:{server.port}") as channel:
            stream = pose_stream(channel)(
                _pose_pb2.WatchPoseRequest(
                    reference_frame="odom",
                    tracked_frame="pelvis_link",
                ),
                timeout=1.0,
            )
            sample = next(stream)
            stream.cancel()

    assert sample.reference_frame == "odom"
    assert sample.tracked_frame == "pelvis_link"
    assert sample.sequence == 2
    assert sample.translation_metres.x == pytest.approx(4.0)
    assert sample.translation_metres.y == pytest.approx(5.0)
    assert sample.translation_metres.z == pytest.approx(6.0)
    assert sample.orientation_xyzw.x == pytest.approx(0.0)
    assert sample.orientation_xyzw.y == pytest.approx(0.0)
    assert sample.orientation_xyzw.z == pytest.approx(0.0)
    assert sample.orientation_xyzw.w == pytest.approx(1.0)
    assert sample.sample_age_ms >= 15


def test_pose_service_rejects_an_unconfigured_frame_pair() -> None:
    config = PoseServiceConfig(listen="127.0.0.1:0")
    with PoseServiceServer(config) as server:
        with grpc.insecure_channel(f"127.0.0.1:{server.port}") as channel:
            stream = pose_stream(channel)(
                _pose_pb2.WatchPoseRequest(
                    reference_frame="map",
                    tracked_frame="pelvis_link",
                ),
                timeout=1.0,
            )
            with pytest.raises(grpc.RpcError) as caught:
                next(stream)

    assert caught.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_pose_service_rejects_invalid_samples() -> None:
    with PoseServiceServer(PoseServiceConfig(listen="127.0.0.1:0")) as server:
        with pytest.raises(ValueError, match="finite"):
            server.publish((float("nan"), 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        with pytest.raises(ValueError, match="nonzero norm"):
            server.publish((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0))
