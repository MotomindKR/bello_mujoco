from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bello_mujoco.hardware.fastsam import (
    FastSamCoordinateAdapter,
    FastSamGrpcConfig,
    FastSamSmplxStream,
    _validate_service_status,
    _validate_stream_info,
    import_pinned_fastsam_proto,
)
from bello_mujoco.hardware.gmr_retarget import (
    SMPLX_BODY_JOINT_NAMES,
    SMPLX_BODY_JOINT_PARENTS,
    load_smplx_standing_sample,
)
from bello_mujoco.hardware.reference import ReferenceDiscontinuity, ReferenceUnavailable
from bello_mujoco.robot_tasks import motion_task_definition


GMR_SOURCE_BODY_NAMES = motion_task_definition("bello").gmr.source_body_names


def neutral_model(tmp_path: Path) -> Path:
    joint_count = len(SMPLX_BODY_JOINT_NAMES)
    template = np.asarray(
        (
            (0.00, 1.00, 0.00),
            (0.10, 0.92, 0.00),
            (-0.10, 0.92, 0.00),
            (0.00, 1.12, 0.00),
            (0.10, 0.52, 0.00),
            (-0.10, 0.52, 0.00),
            (0.00, 1.25, 0.00),
            (0.10, 0.12, 0.00),
            (-0.10, 0.12, 0.00),
            (0.00, 1.38, 0.00),
            (0.10, 0.04, 0.12),
            (-0.10, 0.04, 0.12),
            (0.00, 1.55, 0.00),
            (0.10, 1.46, 0.00),
            (-0.10, 1.46, 0.00),
            (0.00, 1.70, 0.00),
            (0.22, 1.48, 0.00),
            (-0.22, 1.48, 0.00),
            (0.48, 1.48, 0.00),
            (-0.48, 1.48, 0.00),
            (0.72, 1.48, 0.00),
            (-0.72, 1.48, 0.00),
        ),
        dtype=np.float64,
    )
    assert template.shape == (joint_count, 3)
    regressor = np.eye(joint_count, dtype=np.float64)
    parents = SMPLX_BODY_JOINT_PARENTS.astype(np.int64).copy()
    parents[0] = 0
    path = tmp_path / "SMPLX_NEUTRAL.npz"
    np.savez(
        path,
        v_template=template,
        J_regressor=regressor,
        kintree_table=np.stack((parents, np.arange(joint_count))),
    )
    return path


def grpc_config(tmp_path: Path, **overrides: object) -> FastSamGrpcConfig:
    config = FastSamGrpcConfig(
        endpoint="127.0.0.1:50052",
        neutral_model_path=neutral_model(tmp_path),
        tls_root_certificate_path=None,
        tls_server_name_override=None,
        connect_timeout_seconds=1.0,
        rpc_timeout_seconds=1.0,
        min_tracking_confidence=0.5,
        min_joint_confidence=0.25,
        max_pipeline_latency_ms=20.0,
        transport_latency_budget_ms=5.0,
        camera_to_world_quaternion_xyzw=(0.5, -0.5, 0.5, -0.5),
        align_initial_heading=False,
        align_initial_floor=False,
    )
    return replace(config, **overrides)


def stream_info(**overrides: object) -> SimpleNamespace:
    values = {
        "schema_version": 1,
        "stream_instance_id": "stream-1",
        "config_sha256": "a" * 64,
        "joint_names": SMPLX_BODY_JOINT_NAMES,
        "joint_parents": SMPLX_BODY_JOINT_PARENTS,
        "root_index": 0,
        "coordinate_frame": "camera_opencv",
        "quaternion_order": "xyzw",
        "global_quaternion_semantics": "smplx_joint_frame_to_camera",
        "local_quaternion_semantics": "joint_to_smplx_parent",
        "position_units": "meters",
        "inference_profile": "large",
        "inference_mode": "body",
        "joint_set": "body",
        "body_representation": "smplx_body_22",
        "converter": "learned_mhr2smpl_multiview_single_view",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FrameMessage(SimpleNamespace):
    def HasField(self, name: str) -> bool:
        return name in {"track_id", "conversion_ms"}


def frame_message(**overrides: object) -> FrameMessage:
    joint_count = len(SMPLX_BODY_JOINT_NAMES)
    positions = np.zeros((joint_count, 3), dtype=np.float64)
    positions[:, 2] = 2.0
    positions[:, 1] = np.linspace(-0.5, 0.5, joint_count)
    quaternions = np.zeros((joint_count, 4), dtype=np.float64)
    quaternions[:, 3] = 1.0
    values = {
        "frame_id": 1,
        "source_timestamp_ns": 1_000_000_000,
        "submit_monotonic_ns": 2_000_000_000,
        "output_monotonic_ns": 2_010_000_000,
        "root_translation": positions[0],
        "joint_positions": positions.reshape(-1),
        "local_joint_quaternions": quaternions.reshape(-1),
        "global_joint_quaternions": quaternions.reshape(-1),
        "joint_confidence": np.ones(joint_count),
        "tracking_confidence": 0.9,
        "tracking_state": "TRACKING",
        "track_id": 7,
        "pipeline_latency_ms": 10.0,
        "rectification_ms": 1.0,
        "conversion_ms": 2.0,
    }
    values.update(overrides)
    return FrameMessage(**values)


def test_smplx_standing_sample_is_derived_from_model(tmp_path: Path) -> None:
    frame = load_smplx_standing_sample(neutral_model(tmp_path), GMR_SOURCE_BODY_NAMES)

    assert tuple(frame.transforms) == GMR_SOURCE_BODY_NAMES
    assert frame.source_timestamp_ns == 0
    assert frame.source_age_ms == 0
    feet = [frame.transforms[name][0][2] for name in ("left_foot", "right_foot")]
    assert min(feet) == pytest.approx(0.0)
    for position, quaternion in frame.transforms.values():
        assert position.shape == (3,)
        assert np.linalg.norm(quaternion) == pytest.approx(1.0)
    for side in ("left", "right"):
        shoulder = frame.transforms[f"{side}_shoulder"][0]
        elbow = frame.transforms[f"{side}_elbow"][0]
        wrist = frame.transforms[f"{side}_wrist"][0]
        assert shoulder[2] > elbow[2] > wrist[2]
    assert not np.allclose(
        frame.transforms["left_wrist"][1], frame.transforms["pelvis"][1]
    )


def test_stream_info_requires_corrected_smplx_contract() -> None:
    _validate_stream_info(stream_info())

    with pytest.raises(ValueError, match="body_representation"):
        _validate_stream_info(stream_info(body_representation="mhr_native"))
    with pytest.raises(ValueError, match="joint order"):
        _validate_stream_info(stream_info(joint_names=("pelvis",)))
    with pytest.raises(ValueError, match="global_quaternion_semantics"):
        _validate_stream_info(stream_info(global_quaternion_semantics="unknown"))


@pytest.mark.integration
def test_fastsam_protobuf_is_loaded_from_pinned_submodule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skeleton_pb2, skeleton_pb2_grpc = import_pinned_fastsam_proto()

    assert skeleton_pb2.StreamInfo.DESCRIPTOR.fields_by_name["converter"].number == 17
    assert (
        skeleton_pb2.SkeletonFrame.DESCRIPTOR.fields_by_name["conversion_ms"].number
        == 20
    )
    assert skeleton_pb2_grpc.GRPC_GENERATED_VERSION == "1.83.0"


def test_service_status_must_enable_tracking() -> None:
    _validate_service_status(
        SimpleNamespace(
            ready=True,
            source_running=True,
            inference_running=True,
            tracking_enabled=True,
            terminal_failure="",
        )
    )
    with pytest.raises(RuntimeError, match="tracking"):
        _validate_service_status(
            SimpleNamespace(
                ready=True,
                source_running=True,
                inference_running=True,
                tracking_enabled=False,
                terminal_failure="",
            )
        )


def test_camera_adapter_preserves_live_translation_after_initial_alignment(
    tmp_path: Path,
) -> None:
    adapter = FastSamCoordinateAdapter(grpc_config(tmp_path))
    positions = np.zeros((len(SMPLX_BODY_JOINT_NAMES), 3), dtype=np.float64)
    positions[:, 2] = 2.0
    quaternions = np.zeros((len(SMPLX_BODY_JOINT_NAMES), 4), dtype=np.float64)
    quaternions[:, 3] = 1.0

    first, _ = adapter.transform(positions, quaternions, track_id=1)
    moved = positions.copy()
    moved[:, 2] += 0.1
    second, _ = adapter.transform(moved, quaternions, track_id=1)

    np.testing.assert_allclose(first[0], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(second[0], (0.1, 0.0, 0.0), atol=1.0e-12)


def test_frame_validation_maps_camera_pose_and_accounts_for_latency(
    tmp_path: Path,
) -> None:
    stream = FastSamSmplxStream(grpc_config(tmp_path), GMR_SOURCE_BODY_NAMES)
    frame = stream._parse_frame(frame_message())

    assert frame.sequence == 1
    assert frame.source_age_ms == 15
    assert tuple(frame.transforms) == GMR_SOURCE_BODY_NAMES
    assert np.all(np.isfinite(frame.transforms["pelvis"][0]))

    low_confidence = np.ones(len(SMPLX_BODY_JOINT_NAMES))
    low_confidence[SMPLX_BODY_JOINT_NAMES.index("left_wrist")] = 0.1
    with pytest.raises(ReferenceUnavailable, match="left_wrist") as low_error:
        FastSamSmplxStream(grpc_config(tmp_path), GMR_SOURCE_BODY_NAMES)._parse_frame(
            frame_message(joint_confidence=low_confidence)
        )
    assert not isinstance(low_error.value, ReferenceDiscontinuity)
    with pytest.raises(ReferenceUnavailable, match="latency is too high"):
        FastSamSmplxStream(grpc_config(tmp_path), GMR_SOURCE_BODY_NAMES)._parse_frame(
            frame_message(
                output_monotonic_ns=2_030_000_000,
                pipeline_latency_ms=30.0,
            )
        )


def test_track_identity_change_is_a_retargeting_discontinuity(tmp_path: Path) -> None:
    stream = FastSamSmplxStream(grpc_config(tmp_path), GMR_SOURCE_BODY_NAMES)
    stream._parse_frame(frame_message())

    with pytest.raises(ReferenceDiscontinuity, match="track changed"):
        stream._parse_frame(
            frame_message(
                frame_id=2,
                source_timestamp_ns=1_033_333_333,
                track_id=8,
            )
        )
