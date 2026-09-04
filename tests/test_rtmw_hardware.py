from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pytest

from bello_mujoco.hardware.rtmw import (
    RTMW_COORDINATE_SPACE,
    RTMW_REPRESENTATION,
    RTMW_ROTATION_SEMANTICS,
    RTMW_SCHEMA_ID,
    RTMW_SMPLX52_NAMES,
    RTMW_SMPLX52_PARENTS,
    RtmwCoordinateAdapter,
    RtmwLatestJsonConfig,
    RtmwLatestJsonStream,
    validate_pinned_rtmw_contract,
)
from bello_mujoco.hardware.smplx_retarget import (
    SMPLX_GMR_BODY_NAMES,
    SmplxTargetFrame,
)
from bello_mujoco.hardware.twist2 import ReferenceUnavailable
from bello_mujoco.repository import repository_root


def rtmw_config(tmp_path: Path, **overrides: object) -> RtmwLatestJsonConfig:
    values = {
        "pose_path": tmp_path / "latest.json",
        "neutral_model_path": tmp_path / "SMPLX_NEUTRAL.npz",
        "human_height_m": 1.75,
        "min_joint_confidence": 0.25,
        "min_rotation_confidence": 0.10,
        "max_pipeline_latency_ms": 250.0,
        "max_source_age_ms": 1_000,
        "camera_to_world_quaternion_xyzw": (0.0, 0.0, 0.0, 1.0),
        "align_initial_heading": False,
        "align_initial_floor": False,
    }
    values.update(overrides)
    return RtmwLatestJsonConfig(**values)


def standing_sample(_path: Path) -> SmplxTargetFrame:
    identity = np.asarray((1.0, 0.0, 0.0, 0.0))
    return SmplxTargetFrame(
        transforms={
            name: (np.zeros(3), identity.copy()) for name in SMPLX_GMR_BODY_NAMES
        },
        sequence=0,
        source_timestamp_ns=0,
        source_age_ms=0,
        received_at=time.monotonic(),
    )


def rtmw_record(
    captured_at_ns: int,
    *,
    frame_id: int = 7,
    track_id: int = 3,
) -> dict[str, object]:
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
    inferred_at_ns = captured_at_ns + 10_000_000
    body = {
        "frame_id": frame_id,
        "captured_at_ns": captured_at_ns,
        "inferred_at_ns": captured_at_ns + 8_000_000,
        "track_id": track_id,
        "positions_m": positions.tolist(),
        "confidence": confidence.tolist(),
        "valid_mask": np.ones(count, dtype=bool).tolist(),
        "coordinate_space": RTMW_COORDINATE_SPACE,
        "depth_provenance": "calibrated_camera+assumed_subject_height",
        "root_depth_m": 2.5,
        "metric_scale": True,
        "global_translation": True,
        "global_depth_is_measured": False,
        "representation": RTMW_REPRESENTATION,
        "processing": ["rtmw133_to_smplx52_v1", "tree_consistent_ik:test"],
        "schema_id": RTMW_SCHEMA_ID,
        "root_index": 0,
        "root_translation": positions[0].tolist(),
        "root_translation_m": positions[0].tolist(),
        "joint_positions": positions.tolist(),
        "joint_confidence": confidence.tolist(),
        "local_joint_quaternions": identity.tolist(),
        "global_joint_quaternions": identity.tolist(),
        "rotation_valid_mask": rotation_valid.tolist(),
        "rotation_confidence": np.where(rotation_valid, 0.8, 0.0).tolist(),
        "rotation_residual_degrees": [0.0] * count,
        "rotation_constraint_rank": np.where(rotation_valid, 2, 0).tolist(),
        "rotation_semantics": RTMW_ROTATION_SEMANTICS,
        "quaternion_order": "xyzw",
        "joint_names": list(RTMW_SMPLX52_NAMES),
        "joint_parents": list(RTMW_SMPLX52_PARENTS),
        "position_units": "meters",
    }
    return {
        "schema_version": 1,
        "frame_id": frame_id,
        "captured_at_ns": captured_at_ns,
        "source_timestamp_ns": captured_at_ns,
        "inferred_at_ns": inferred_at_ns,
        "latency_ms": 10.0,
        "wholebody133": {},
        "smplx52_landmark_fit": body,
        "tracking": {"track_id": track_id, "state": "tracked"},
        "timings_ms": {},
    }


def make_stream(
    tmp_path: Path,
    record: dict[str, object],
    *,
    clock_ns: int,
) -> RtmwLatestJsonStream:
    config = rtmw_config(tmp_path)
    config.neutral_model_path.write_bytes(b"test-model")
    config.pose_path.write_text(json.dumps(record), encoding="utf-8")
    return RtmwLatestJsonStream(
        config,
        clock_ns=lambda: clock_ns,
        standing_sample_loader=standing_sample,
    )


def test_rtmw_latest_json_accepts_the_published_schema(tmp_path: Path) -> None:
    captured_at_ns = 5_000_000_000
    stream = make_stream(
        tmp_path,
        rtmw_record(captured_at_ns),
        clock_ns=captured_at_ns + 12_000_000,
    )

    frame = stream.wait_next(after_sequence=None, timeout_seconds=0.01)

    assert frame is not None
    assert frame.sequence == 7
    assert frame.source_timestamp_ns == captured_at_ns
    assert frame.source_age_ms == 12
    assert tuple(frame.transforms) == SMPLX_GMR_BODY_NAMES
    assert frame.transforms["pelvis"][0] == pytest.approx((0.0, 0.0, 1.0))
    assert frame.transforms["pelvis"][1] == pytest.approx((1.0, 0.0, 0.0, 0.0))
    assert frame.foot_contacts is not None
    assert stream.wait_next(after_sequence=7, timeout_seconds=0.001) is None


def test_pinned_rtmw_contract_matches_consumer() -> None:
    validate_pinned_rtmw_contract(repository_root() / "deps" / "RTMW-SMPLX-Teleop")


def test_rtmw_rejects_an_invalid_required_rotation(tmp_path: Path) -> None:
    captured_at_ns = 8_000_000_000
    record = rtmw_record(captured_at_ns)
    body = record["smplx52_landmark_fit"]
    assert isinstance(body, dict)
    index = RTMW_SMPLX52_NAMES.index("left_shoulder")
    body["rotation_valid_mask"][index] = False
    stream = make_stream(
        tmp_path,
        record,
        clock_ns=captured_at_ns + 12_000_000,
    )

    with pytest.raises(ReferenceUnavailable, match="left_shoulder"):
        stream.wait_next(after_sequence=None, timeout_seconds=0.01)


def test_rtmw_rejects_root_relative_or_uncalibrated_output(tmp_path: Path) -> None:
    captured_at_ns = 8_000_000_000
    record = rtmw_record(captured_at_ns)
    body = record["smplx52_landmark_fit"]
    assert isinstance(body, dict)
    body["coordinate_space"] = "root_relative_opencv_x_right_y_down_z_forward"
    body["depth_provenance"] = "relative_reference_depth_assumption"
    body["metric_scale"] = False
    body["global_translation"] = False
    stream = make_stream(
        tmp_path,
        record,
        clock_ns=captured_at_ns + 12_000_000,
    )

    with pytest.raises(ValueError, match="coordinate_space"):
        stream.wait_next(after_sequence=None, timeout_seconds=0.01)


def test_rtmw_rejects_cross_host_or_stale_monotonic_time(tmp_path: Path) -> None:
    captured_at_ns = 8_000_000_000
    record = rtmw_record(captured_at_ns)
    cross_host = make_stream(
        tmp_path,
        record,
        clock_ns=captured_at_ns - 2_000_000,
    )
    with pytest.raises(ValueError, match="same host"):
        cross_host.wait_next(after_sequence=None, timeout_seconds=0.01)

    record = rtmw_record(captured_at_ns, frame_id=8)
    stale = make_stream(
        tmp_path,
        record,
        clock_ns=captured_at_ns + 2_000_000_000,
    )
    with pytest.raises(ReferenceUnavailable, match="stale"):
        stale.wait_next(after_sequence=None, timeout_seconds=0.01)


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

    world, _ = adapter.transform(positions, quaternions, track_id=1)

    assert world[1] == pytest.approx((3.0, -1.0, -2.0))
