from __future__ import annotations

from contextlib import nullcontext
from enum import Enum
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import numpy as np
import pytest

from bello_mujoco.hardware_locomotion import (
    HardwareFrame,
    HardwareLocomotionConfig,
)

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_motion_hardware.py"
SPEC = importlib.util.spec_from_file_location("run_motion_hardware", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
motion_hardware = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = motion_hardware
SPEC.loader.exec_module(motion_hardware)


def pose_frame(*, age_ms: int = 0) -> motion_hardware.PoseFrame:
    return motion_hardware.PoseFrame(
        position=np.asarray((0.0, 0.0, 0.8)),
        orientation_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
        linear_velocity=np.zeros(3),
        sequence=1,
        source_age_ms=age_ms,
        received_at=time.monotonic(),
    )


def test_pose_service_dynamic_messages_round_trip() -> None:
    request_type, response_type = motion_hardware._pose_message_types()
    request = request_type(reference_frame="odom", tracked_frame="pelvis_link")
    response = response_type(
        reference_frame="odom",
        tracked_frame="pelvis_link",
        sequence=9,
        sample_age_ms=4,
    )

    restored = response_type.FromString(response.SerializeToString())

    assert request.reference_frame == "odom"
    assert request.tracked_frame == "pelvis_link"
    assert restored.sequence == 9
    assert restored.sample_age_ms == 4


def test_pose_validation_is_fail_closed() -> None:
    config = motion_hardware.MotionHardwareConfig(max_pose_age_ms=10)

    motion_hardware.validate_pose_frame(pose_frame(age_ms=5), config)
    with pytest.raises(RuntimeError, match="stale"):
        motion_hardware.validate_pose_frame(pose_frame(age_ms=20), config)
    with pytest.raises(RuntimeError, match="near-zero"):
        motion_hardware.validate_pose_frame(
            motion_hardware.PoseFrame(
                position=np.zeros(3),
                orientation_xyzw=np.zeros(4),
                linear_velocity=np.zeros(3),
                sequence=1,
                source_age_ms=0,
                received_at=time.monotonic(),
            ),
            config,
        )


def test_pose_and_imu_orientation_must_agree() -> None:
    frame = HardwareFrame(
        positions=np.zeros(1),
        velocities=np.zeros(1),
        orientation_xyzw=np.asarray((0.0, 0.0, 1.0, 0.0)),
        angular_velocity=np.zeros(3),
        linear_acceleration=np.zeros(3),
        max_motor_age_ms=0,
        imu_age_ms=0,
        group_states=("IDLE",),
    )
    with pytest.raises(RuntimeError, match="orientation mismatch"):
        motion_hardware.validate_pose_imu_agreement(
            frame,
            pose_frame(),
            motion_hardware.MotionHardwareConfig(max_pose_imu_orientation_error=0.1),
        )


def test_motion_config_rejects_unsafe_ranges() -> None:
    with pytest.raises(ValueError, match="start_time_seconds"):
        motion_hardware.validate_motion_config(
            motion_hardware.MotionHardwareConfig(start_time_seconds=3.0), 2.0
        )
    with pytest.raises(ValueError, match="max_action_abs"):
        motion_hardware.validate_motion_config(
            motion_hardware.MotionHardwareConfig(
                max_action_abs=2.0, max_raw_action_abs=1.0
            ),
            3.0,
        )


def test_nominal_physical_target_uses_model_default_pose() -> None:
    adapter = motion_hardware.MotionHardwarePolicyAdapter.__new__(
        motion_hardware.MotionHardwarePolicyAdapter
    )
    adapter.transform = np.asarray(((1.0, 0.5), (0.0, 2.0)))
    adapter.env = SimpleNamespace(
        model=SimpleNamespace(
            default_joint_target=np.asarray((0.2, -0.4)),
            mj_model=SimpleNamespace(
                actuator_ctrlrange=np.asarray(((-1.0, 1.0), (-0.5, 0.5)))
            ),
        )
    )

    assert adapter.nominal_physical_target == pytest.approx((0.0, -0.5))


class FakeGroupState(Enum):
    IDLE = 1


class FakeDeviceState(Enum):
    ONLINE = 1


class FakeUnavailableError(Exception):
    pass


def test_read_only_preflight_never_acquires_control(monkeypatch, capsys) -> None:
    coordinates = ("left", "right")
    control_requested = False

    def forbidden_control(*_args, **_kwargs):
        nonlocal control_requested
        control_requested = True
        raise AssertionError("read-only preflight acquired control")

    groups = {
        "lower_body": SimpleNamespace(
            name="lower_body",
            coordinates=("left",),
            control_rate_hz=100.0,
            initial_state=FakeGroupState.IDLE,
            motors=(
                SimpleNamespace(
                    name="left",
                    status=SimpleNamespace(state=FakeDeviceState.ONLINE),
                ),
            ),
            control=forbidden_control,
        ),
        "upper_body": SimpleNamespace(
            name="upper_body",
            coordinates=("right",),
            control_rate_hz=100.0,
            initial_state=FakeGroupState.IDLE,
            motors=(
                SimpleNamespace(
                    name="right",
                    status=SimpleNamespace(state=FakeDeviceState.ONLINE),
                ),
            ),
            control=forbidden_control,
        ),
    }

    def vector(*values):
        return SimpleNamespace(x=values[0], y=values[1], z=values[2])

    states = SimpleNamespace(
        coordinate=lambda _name: SimpleNamespace(
            position=0.0, velocity=0.0, feedback_age_ms=0
        ),
        imu=lambda _name: SimpleNamespace(
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            angular_velocity=vector(0.0, 0.0, 0.0),
            linear_acceleration=vector(0.0, 0.0, 0.0),
            sample_age_ms=0,
        ),
        group=lambda _name: SimpleNamespace(state=SimpleNamespace(name="IDLE")),
    )
    robot = SimpleNamespace(
        group_names=tuple(groups),
        group=lambda name: groups[name],
        inventory=SimpleNamespace(
            imus=(
                SimpleNamespace(
                    name="imu",
                    status=SimpleNamespace(state=FakeDeviceState.ONLINE),
                ),
            )
        ),
        states=lambda **_kwargs: nullcontext(states),
    )
    binding = SimpleNamespace(
        Robot=SimpleNamespace(connect=lambda *_args, **_kwargs: nullcontext(robot)),
        GroupState=FakeGroupState,
        DeviceState=FakeDeviceState,
        StateUnavailableError=FakeUnavailableError,
    )
    preview = motion_hardware.MotionHardwareStep(
        action=np.zeros(2),
        raw_action=np.zeros(2),
        physical_target=np.zeros(2),
        observation=np.zeros(4),
        reference_time=0.0,
        anchor_position_error=0.0,
        anchor_orientation_error=0.0,
        action_clipped=False,
    )
    adapter = SimpleNamespace(
        physical_coordinate_names=coordinates,
        nominal_physical_target=np.zeros(2),
        env=SimpleNamespace(action_size=25),
        control_timestep=0.02,
        start_time=0.0,
        precompile=lambda: None,
        initialize_alignment=lambda _pose: None,
        step=lambda _frame, _pose, reference_time: preview,
        reset=lambda: None,
    )
    monkeypatch.setattr(
        motion_hardware,
        "load_motion_hardware_policy",
        lambda *_args, **_kwargs: (adapter, SimpleNamespace(task="motion_tracking")),
    )

    class FakePoseStream:
        def __init__(self, _config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def latest(self):
            return pose_frame()

    motion_hardware.run_motion_hardware(
        Path("unused"),
        HardwareLocomotionConfig(),
        motion_hardware.MotionHardwareConfig(),
        hardware_binding=binding,
        pose_stream_factory=FakePoseStream,
    )

    assert not control_requested
    assert "no lease was acquired" in capsys.readouterr().out
