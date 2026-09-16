import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from bello_mujoco.hardware import _smplx_pb2 as pb
from bello_mujoco.hardware.pico_grpc import PicoGrpcConfig
from bello_mujoco.hardware.reference import ReferenceUnavailable
from bello_mujoco.hardware.xrobot_stream import BODY_NAMES, parse_frame


def packet():
    return pb.PoseFrame(
        sequence=1,
        source_timestamp_ns=100,
        device_id="synthetic",
        upstream_connected=True,
        packet_age_seconds=0.01,
        xrobot_body=pb.XRobotBody(
            joint_names=BODY_NAMES,
            positions=np.tile([1, 2, 3], 24),
            orientations=np.tile([1, 0, 0, 0], 24),
            frame="pico_native_y_up",
            valid=True,
            age_seconds=0.02,
        ),
    )


def test_raw_body_needs_no_smplx_calibration_or_neutral_model():
    frame = parse_frame(packet(), PicoGrpcConfig("localhost:1"))
    p, q = frame.transforms["Left_Wrist"]
    np.testing.assert_allclose(p, [1, -3, 2])
    np.testing.assert_allclose(
        Rotation.from_quat(q, scalar_first=True).as_matrix(),
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        atol=1e-12,
    )
    assert frame.source_age_ms == 30
    assert not p.flags.writeable and not q.flags.writeable
    with pytest.raises(TypeError):
        frame.transforms["Pelvis"] = (p, q)


def test_global_wrist_rotation_is_not_parent_accumulated_or_reflected():
    message = packet()
    rotations = Rotation.from_euler("xyz", [[0.1 * i, 0.2, -0.3] for i in range(24)])
    message.xrobot_body.orientations[:] = rotations.as_quat(scalar_first=True).ravel()
    result = parse_frame(message, PicoGrpcConfig("localhost:1"))
    basis = Rotation.from_euler("x", 90, degrees=True)
    for i, name in enumerate(BODY_NAMES):
        actual = Rotation.from_quat(result.transforms[name][1], scalar_first=True)
        np.testing.assert_allclose(
            actual.as_matrix(), (basis * rotations[i]).as_matrix(), atol=1e-12
        )


def test_missing_typed_body_is_an_explicit_upgrade_error():
    message = packet()
    message.ClearField("xrobot_body")
    with pytest.raises(ValueError, match="upgrade"):
        parse_frame(message, PicoGrpcConfig("localhost:1"))


@pytest.mark.parametrize(
    "field,values",
    [
        ("positions", [0] * 71),
        ("positions", [float("nan")] * 72),
        ("orientations", [0] * 96),
        ("orientations", [1] * 96),
        ("orientations", [0] * 95),
        ("joint_names", ["Pelvis"] * 24),
    ],
)
def test_bad_body_shapes_names_and_quaternions_fail_closed(field, values):
    message = packet()
    getattr(message.xrobot_body, field)[:] = values
    with pytest.raises(ValueError):
        parse_frame(message, PicoGrpcConfig("localhost:1"))


def test_wrong_frame_and_invalid_body_age_fail_closed():
    message = packet()
    message.xrobot_body.frame = "unity_left_handed"
    with pytest.raises(ValueError, match="coordinates"):
        parse_frame(message, PicoGrpcConfig("localhost:1"))
    message = packet()
    message.xrobot_body.age_seconds = float("nan")
    with pytest.raises(ValueError, match="age"):
        parse_frame(message, PicoGrpcConfig("localhost:1"))


def test_body_age_not_fresh_outer_packet_decides_availability():
    config = PicoGrpcConfig("localhost:1")
    message = packet()
    message.xrobot_body.age_seconds = 1
    with pytest.raises(ReferenceUnavailable, match="stale"):
        parse_frame(message, config)
    message = packet()
    message.xrobot_body.valid = False
    with pytest.raises(ReferenceUnavailable):
        parse_frame(message, config)


@pytest.mark.parametrize("missing", ["source_timestamp_ns", "device_id"])
@pytest.mark.parametrize("outage", ["invalid", "packet_stale", "body_stale"])
def test_outage_with_missing_identity_is_recoverable(missing, outage):
    message = packet()
    message.ClearField(missing)
    if outage == "invalid":
        message.xrobot_body.valid = False
    elif outage == "packet_stale":
        message.packet_age_seconds = 1
    else:
        message.xrobot_body.age_seconds = 1
    with pytest.raises(ReferenceUnavailable, match="invalid or stale"):
        parse_frame(message, PicoGrpcConfig("localhost:1"))


@pytest.mark.parametrize("missing", ["source_timestamp_ns", "device_id"])
def test_fresh_body_still_requires_timestamp_and_identity(missing):
    message = packet()
    message.ClearField(missing)
    with pytest.raises(ValueError, match="timestamp and device identity"):
        parse_frame(message, PicoGrpcConfig("localhost:1"))
