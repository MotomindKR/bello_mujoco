import time

import grpc
import pytest

from bello_mujoco.hardware import _pose_pb2
from bello_mujoco.hardware.pose import PoseServiceConfig, PoseServiceServer


def test_emulator_publishes_link_origin_velocity_not_inertial_center_velocity():
    from types import SimpleNamespace

    import mujoco
    import numpy as np

    from bello_mujoco.robstride.emulator import publish_body_pose

    model = mujoco.MjModel.from_xml_string("""<mujoco>
      <compiler alignfree="false"/>
      <worldbody><body name="link" pos="0 0 1"><freejoint/>
        <inertial pos=".4 0 0" quat=".7071067811865476 .7071067811865476 0 0"
          mass="1" diaginertia=".1 .2 .3"/>
        <geom size=".1" contype="0" conaffinity="0"/>
      </body></worldbody>
    </mujoco>""")
    data = mujoco.MjData(model)
    data.qvel[:] = [1, 2, 3, 0, 0, 2]
    mujoco.mj_forward(model, data)
    recorded = []
    server = SimpleNamespace(publish=lambda *args, **kwargs: recorded.append(args))
    publish_body_pose(server, model, data, model.body("link").id)
    position, quaternion, linear, angular = recorded[0]
    np.testing.assert_allclose(position, [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(quaternion, [1, 0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(linear, [1, 2, 3], atol=1e-12)
    np.testing.assert_allclose(angular, [0, 0, 2], atol=1e-12)


def pose_stream(channel: grpc.Channel):
    return channel.unary_stream(
        "/bello.pose.v1.PoseService/WatchPose",
        request_serializer=_pose_pb2.WatchPoseRequest.SerializeToString,
        response_deserializer=_pose_pb2.WatchPoseResponse.FromString,
    )


def test_pose_service_streams_latest_world_state_in_xyzw_order() -> None:
    config = PoseServiceConfig(listen="127.0.0.1:0")
    with PoseServiceServer(config) as server:
        server.publish(
            (1.0, 2.0, 3.0),
            (1.0, 0.0, 0.0, 0.0),
            (0.1, 0.2, 0.3),
            (0.4, 0.5, 0.6),
            sample_monotonic_ns=time.monotonic_ns(),
        )
        source_ns = time.monotonic_ns() - 20_000_000
        server.publish(
            (4.0, 5.0, 6.0),
            (2.0, 0.0, 0.0, 0.0),
            (0.4, 0.5, 0.6),
            (0.7, 0.8, 0.9),
            sample_monotonic_ns=source_ns,
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
    assert sample.linear_velocity_metres_per_second.x == pytest.approx(0.4)
    assert sample.linear_velocity_metres_per_second.y == pytest.approx(0.5)
    assert sample.linear_velocity_metres_per_second.z == pytest.approx(0.6)
    assert sample.angular_velocity_radians_per_second.x == pytest.approx(0.7)
    assert sample.angular_velocity_radians_per_second.y == pytest.approx(0.8)
    assert sample.angular_velocity_radians_per_second.z == pytest.approx(0.9)
    assert sample.sample_monotonic_ns == source_ns
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
            server.publish(
                (float("nan"), 0.0, 0.0),
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                sample_monotonic_ns=time.monotonic_ns(),
            )
        with pytest.raises(ValueError, match="nonzero norm"):
            server.publish(
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                sample_monotonic_ns=time.monotonic_ns(),
            )
        with pytest.raises(ValueError, match="timestamp must be positive"):
            server.publish(
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                sample_monotonic_ns=0,
            )
