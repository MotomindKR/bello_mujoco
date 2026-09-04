from __future__ import annotations

from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from bello_mujoco.hardware import rtmw
from bello_mujoco.hardware.rtmw import (
    RTMW_COORDINATE_SPACE,
    RTMW_REPRESENTATION,
    RTMW_ROTATION_SEMANTICS,
    RTMW_SCHEMA_ID,
    RTMW_SMPLX52_NAMES,
    RTMW_SMPLX52_PARENTS,
    RtmwCoordinateAdapter,
    RtmwGrpcConfig,
    RtmwSmplxStream,
    _validate_stream_info,
    import_pinned_rtmw_proto,
    validate_pinned_rtmw_contract,
)
from bello_mujoco.hardware.smplx_retarget import (
    SmplxTargetFrame,
)
from bello_mujoco.hardware.twist2 import ReferenceUnavailable
from bello_mujoco.repository import repository_root
from bello_mujoco.robot_tasks import motion_task_definition


GMR_SOURCE_BODY_NAMES = motion_task_definition("bello").gmr.source_body_names


def rtmw_config(tmp_path: Path, **overrides: object) -> RtmwGrpcConfig:
    values = {
        "endpoint": "127.0.0.1:50055",
        "neutral_model_path": tmp_path / "SMPLX_NEUTRAL.npz",
        "tls_root_certificate_path": None,
        "tls_server_name_override": None,
        "connect_timeout_seconds": 5.0,
        "rpc_timeout_seconds": 2.0,
        "human_height_m": 1.75,
        "min_joint_confidence": 0.25,
        "min_rotation_confidence": 0.10,
        "max_pipeline_latency_ms": 250.0,
        "transport_latency_budget_ms": 10.0,
        "max_source_age_ms": 1_000,
        "camera_to_world_quaternion_xyzw": (0.0, 0.0, 0.0, 1.0),
        "align_initial_heading": False,
        "align_initial_floor": False,
    }
    values.update(overrides)
    return RtmwGrpcConfig(**values)


def standing_sample(
    _path: Path, source_body_names: tuple[str, ...]
) -> SmplxTargetFrame:
    identity = np.asarray((1.0, 0.0, 0.0, 0.0))
    return SmplxTargetFrame(
        transforms={name: (np.zeros(3), identity.copy()) for name in source_body_names},
        sequence=0,
        source_timestamp_ns=0,
        source_age_ms=0,
        received_at=time.monotonic(),
    )


def rtmw_frame(
    captured_at_ns: int,
    *,
    frame_id: int = 7,
    track_id: str = "3",
) -> SimpleNamespace:
    count = len(RTMW_SMPLX52_NAMES)
    positions = np.stack(
        (
            np.linspace(0.0, 0.51, count),
            np.linspace(0.2, -0.31, count),
            np.linspace(1.0, 0.49, count),
        ),
        axis=1,
    )
    identity = np.tile(np.asarray((0.0, 0.0, 0.0, 1.0)), (count, 1))
    confidence = np.full(count, 0.9)
    rotation_valid = np.zeros(count, dtype=bool)
    for name in (
        "pelvis",
        "spine3",
        "left_shoulder",
        "left_elbow",
        "left_wrist",
        "right_shoulder",
        "right_elbow",
        "right_wrist",
    ):
        rotation_valid[RTMW_SMPLX52_NAMES.index(name)] = True
    return SimpleNamespace(
        frame_id=frame_id,
        source_timestamp_ns=captured_at_ns,
        captured_monotonic_ns=captured_at_ns,
        output_monotonic_ns=captured_at_ns + 10_000_000,
        stream_instance_id="test-stream",
        track_id=track_id,
        tracking_state="tracked",
        root_translation=positions[0].tolist(),
        joint_positions=positions.reshape(-1).tolist(),
        local_joint_quaternions=identity.reshape(-1).tolist(),
        global_joint_quaternions=identity.reshape(-1).tolist(),
        joint_confidence=confidence.tolist(),
        joint_valid=np.ones(count, dtype=bool).tolist(),
        rotation_valid=rotation_valid.tolist(),
        rotation_confidence=np.where(rotation_valid, 0.8, 0.0).tolist(),
        pipeline_latency_ms=10.0,
    )


def stream_info(**overrides: object) -> SimpleNamespace:
    values = {
        "schema_version": 1,
        "stream_instance_id": "test-stream",
        "config_sha256": "1" * 64,
        "joint_names": RTMW_SMPLX52_NAMES,
        "joint_parents": RTMW_SMPLX52_PARENTS,
        "root_index": 0,
        "coordinate_space": RTMW_COORDINATE_SPACE,
        "quaternion_order": "xyzw",
        "rotation_semantics": RTMW_ROTATION_SEMANTICS,
        "position_units": "meters",
        "representation": RTMW_REPRESENTATION,
        "schema_id": RTMW_SCHEMA_ID,
        "depth_provenance": "calibrated_camera+assumed_subject_height",
        "metric_scale": True,
        "global_translation": True,
        "global_depth_is_measured": False,
        "subject_height_m": 1.75,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_stream(tmp_path: Path, **config_overrides: object) -> RtmwSmplxStream:
    config = rtmw_config(tmp_path, **config_overrides)
    config.neutral_model_path.write_bytes(b"test-model")
    stream = RtmwSmplxStream(
        config,
        GMR_SOURCE_BODY_NAMES,
        standing_sample_loader=standing_sample,
    )
    stream.stream_instance_id = "test-stream"
    return stream


def test_rtmw_grpc_accepts_the_published_schema(tmp_path: Path) -> None:
    captured_at_ns = 5_000_000_000
    stream = make_stream(tmp_path)

    frame = stream._parse_frame(rtmw_frame(captured_at_ns))

    assert frame.sequence == 7
    assert frame.source_timestamp_ns == captured_at_ns
    assert frame.source_age_ms == 20
    assert tuple(frame.transforms) == GMR_SOURCE_BODY_NAMES
    assert frame.transforms["pelvis"][0] == pytest.approx((0.0, 0.0, 1.0))
    assert frame.transforms["pelvis"][1] == pytest.approx((1.0, 0.0, 0.0, 0.0))
    assert frame.foot_contacts is not None


def test_pinned_rtmw_contract_and_protobuf_match_consumer(monkeypatch) -> None:
    submodule = repository_root() / "deps" / "RTMW-SMPLX-Teleop"
    validate_pinned_rtmw_contract(submodule)
    monkeypatch.setattr(rtmw, "require_pinned_rtmw_revision", lambda: submodule)
    for name in tuple(sys.modules):
        if name == "rtmw_teleop" or name.startswith("rtmw_teleop."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    skeleton_pb2, skeleton_pb2_grpc = import_pinned_rtmw_proto()

    assert skeleton_pb2.StreamInfo.DESCRIPTOR.full_name == "rtmw.teleop.v1.StreamInfo"
    assert hasattr(skeleton_pb2_grpc, "SkeletonServiceStub")


def test_rtmw_reference_source_uses_selected_robot_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_config = SimpleNamespace(
        endpoint="127.0.0.1:50055",
        human_height_m=1.75,
    )
    identity = object()
    calls: dict[str, object] = {}

    class FakeStream:
        neutral_model_sha256 = "a" * 64
        stream_instance_id = "test-stream"
        stream_config_sha256 = "b" * 64
        depth_provenance = "test-depth"

        def __enter__(self) -> FakeStream:
            return self

        def __exit__(self, *_exception: object) -> bool:
            return False

    stream = FakeStream()

    class FakeSource:
        def __init__(self, actual_stream: object, retargeter: object) -> None:
            calls["source"] = (actual_stream, retargeter)

        def __enter__(self) -> FakeSource:
            return self

        def __exit__(self, *_exception: object) -> bool:
            return False

    monkeypatch.setattr(
        rtmw.RtmwGrpcConfig,
        "from_hardware",
        classmethod(lambda _cls, _config: source_config),
    )
    monkeypatch.setattr(rtmw, "require_pinned_rtmw_revision", lambda: tmp_path)
    monkeypatch.setattr(rtmw, "validate_pinned_rtmw_contract", lambda _root: None)

    def make_stream(actual_config: object, source_body_names: tuple[str, ...]):
        calls["stream"] = (actual_config, source_body_names)
        return stream

    def source_identity(asset: str) -> object:
        calls["identity_asset"] = asset
        return identity

    def make_retargeter(
        env: object,
        actual_identity: object,
        config: object,
        *,
        validate_reference: object,
    ) -> object:
        retargeter = object()
        calls["retargeter"] = (
            env,
            actual_identity,
            config,
            validate_reference,
            retargeter,
        )
        return retargeter

    monkeypatch.setattr(rtmw, "RtmwSmplxStream", make_stream)
    monkeypatch.setattr(rtmw, "repository_gmr_source_identity", source_identity)
    monkeypatch.setattr(rtmw, "GmrRetargeter", make_retargeter)
    monkeypatch.setattr(rtmw, "LiveGmrReferenceSource", FakeSource)

    validation = object()
    env = SimpleNamespace(model=SimpleNamespace(robot=SimpleNamespace(asset="bello")))
    adapter = SimpleNamespace(
        env=env,
        validate_upper_body_reference=validation,
    )
    with rtmw.rtmw_locomotion_reference_source(adapter, object()) as result:
        assert isinstance(result, FakeSource)

    assert calls["stream"] == (source_config, GMR_SOURCE_BODY_NAMES)
    assert calls["identity_asset"] == "bello"
    retargeter_call = calls["retargeter"]
    assert isinstance(retargeter_call, tuple)
    assert retargeter_call[0] is env
    assert retargeter_call[1] is identity
    assert retargeter_call[2].human_height_m == pytest.approx(1.75)
    assert retargeter_call[3] is validation
    assert calls["source"] == (stream, retargeter_call[4])


def test_rtmw_rejects_an_invalid_required_rotation(tmp_path: Path) -> None:
    stream = make_stream(tmp_path)
    message = rtmw_frame(8_000_000_000)
    index = RTMW_SMPLX52_NAMES.index("left_shoulder")
    message.rotation_valid[index] = False

    with pytest.raises(ReferenceUnavailable, match="left_shoulder"):
        stream._parse_frame(message)


def test_rtmw_rejects_root_relative_or_uncalibrated_stream(tmp_path: Path) -> None:
    config = rtmw_config(tmp_path)
    info = stream_info(
        coordinate_space="root_relative_opencv_x_right_y_down_z_forward",
        depth_provenance="rtmw_root_relative_only",
        metric_scale=False,
        global_translation=False,
    )

    with pytest.raises(ValueError, match="coordinate_space"):
        _validate_stream_info(info, config)


def test_rtmw_rejects_latency_mismatch_or_excess(tmp_path: Path) -> None:
    stream = make_stream(tmp_path)
    mismatch = rtmw_frame(8_000_000_000)
    mismatch.pipeline_latency_ms = 1.0
    with pytest.raises(ReferenceUnavailable, match="disagrees"):
        stream._parse_frame(mismatch)

    stream = make_stream(tmp_path, max_pipeline_latency_ms=5.0)
    with pytest.raises(ReferenceUnavailable, match="too high"):
        stream._parse_frame(rtmw_frame(9_000_000_000))


def test_rtmw_default_camera_rotation_maps_opencv_axes(tmp_path: Path) -> None:
    config = rtmw_config(
        tmp_path,
        camera_to_world_quaternion_xyzw=(0.5, -0.5, 0.5, -0.5),
    )
    adapter = RtmwCoordinateAdapter(config)
    positions = np.zeros((len(RTMW_SMPLX52_NAMES), 3))
    positions[1] = (1.0, 2.0, 3.0)
    quaternions = np.tile(
        np.asarray((0.0, 0.0, 0.0, 1.0)),
        (len(RTMW_SMPLX52_NAMES), 1),
    )

    world, _ = adapter.transform(positions, quaternions, track_id="1")

    assert world[1] == pytest.approx((3.0, -1.0, -2.0))
