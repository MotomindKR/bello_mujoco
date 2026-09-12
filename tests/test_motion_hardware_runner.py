from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bello_mujoco.models.compiler import ModelConfig
from bello_mujoco.hardware import motion as motion_hardware
from bello_mujoco.hardware.runtime import HardwareFrame


def pose_frame(*, age_ms: int = 0) -> motion_hardware.PoseFrame:
    return motion_hardware.PoseFrame(
        position=np.asarray((0.0, 0.0, 0.8)),
        orientation_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
        sequence=1,
        source_age_ms=age_ms,
        received_at=time.monotonic(),
    )


def test_pose_validation_is_fail_closed() -> None:
    config = motion_hardware.HardwareMotionConfig(max_pose_age_ms=100)

    motion_hardware.validate_pose_frame(pose_frame(age_ms=5), config)
    with pytest.raises(RuntimeError, match="stale"):
        motion_hardware.validate_pose_frame(pose_frame(age_ms=200), config)
    with pytest.raises(RuntimeError, match="near-zero"):
        motion_hardware.validate_pose_frame(
            motion_hardware.PoseFrame(
                position=np.zeros(3),
                orientation_xyzw=np.zeros(4),
                linear_velocity=np.zeros(3),
                angular_velocity=np.zeros(3),
                sequence=1,
                source_age_ms=0,
                received_at=time.monotonic(),
            ),
            config,
        )
    with pytest.raises(RuntimeError, match="position must contain 3"):
        motion_hardware.validate_pose_frame(
            replace(pose_frame(), position=np.zeros(2)),
            config,
        )
    with pytest.raises(RuntimeError, match="angular velocity must contain 3"):
        motion_hardware.validate_pose_frame(
            replace(pose_frame(), angular_velocity=np.zeros(2)),
            config,
        )
    with pytest.raises(RuntimeError, match="estimated base angular speed"):
        motion_hardware.validate_pose_frame(
            replace(pose_frame(), angular_velocity=np.asarray((21.0, 0.0, 0.0))),
            config,
        )


def pose_response(
    *,
    sequence: int,
    sample_monotonic_ns: int,
    position: tuple[float, float, float],
    velocity: tuple[float, float, float] | None,
    angular_velocity: tuple[float, float, float] | None = (0.0, 0.0, 0.0),
):
    response = motion_hardware._pose_pb2.WatchPoseResponse(
        reference_frame="odom",
        tracked_frame="pelvis_link",
        sequence=sequence,
        translation_metres=motion_hardware._pose_pb2.Translation(
            x=position[0],
            y=position[1],
            z=position[2],
        ),
        orientation_xyzw=motion_hardware._pose_pb2.Orientation(w=1.0),
        sample_age_ms=0,
        sample_monotonic_ns=sample_monotonic_ns,
    )
    if velocity is not None:
        response.linear_velocity_metres_per_second.CopyFrom(
            motion_hardware._pose_pb2.LinearVelocity(
                x=velocity[0],
                y=velocity[1],
                z=velocity[2],
            )
        )
    if angular_velocity is not None:
        response.angular_velocity_radians_per_second.CopyFrom(
            motion_hardware._pose_pb2.AngularVelocity(
                x=angular_velocity[0],
                y=angular_velocity[1],
                z=angular_velocity[2],
            )
        )
    return response


def test_pose_stream_uses_estimator_velocities_without_differentiating_pose() -> None:
    stream = motion_hardware.PoseStream(
        motion_hardware.HardwareMotionConfig(max_base_linear_velocity=0.5)
    )
    stream._call = iter(
        (
            pose_response(
                sequence=1,
                sample_monotonic_ns=1_000_000,
                position=(0.0, 0.0, 0.8),
                velocity=(0.1, 0.2, 0.0),
                angular_velocity=(0.3, 0.4, 0.5),
            ),
            pose_response(
                sequence=2,
                sample_monotonic_ns=3_000_000,
                position=(10.0, -10.0, 0.8),
                velocity=(0.1, 0.2, 0.0),
                angular_velocity=(0.3, 0.4, 0.5),
            ),
        )
    )

    stream._consume()

    latest = stream.latest()
    assert latest.sequence == 2
    assert latest.position == pytest.approx((10.0, -10.0, 0.8))
    assert latest.linear_velocity == pytest.approx((0.1, 0.2, 0.0))
    assert latest.angular_velocity == pytest.approx((0.3, 0.4, 0.5))


def test_pose_stream_requires_estimator_velocity() -> None:
    stream = motion_hardware.PoseStream(motion_hardware.HardwareMotionConfig())
    stream._call = iter(
        (
            pose_response(
                sequence=1,
                sample_monotonic_ns=1_000_000,
                position=(0.0, 0.0, 0.8),
                velocity=None,
            ),
        )
    )

    stream._consume()

    with pytest.raises(RuntimeError, match="pose-estimator stream failed") as raised:
        stream.latest()
    assert "no linear velocity" in str(raised.value.__cause__)


def test_pose_stream_requires_estimator_angular_velocity() -> None:
    stream = motion_hardware.PoseStream(motion_hardware.HardwareMotionConfig())
    stream._call = iter(
        (
            pose_response(
                sequence=1,
                sample_monotonic_ns=1_000_000,
                position=(0.0, 0.0, 0.8),
                velocity=(0.0, 0.0, 0.0),
                angular_velocity=None,
            ),
        )
    )

    stream._consume()

    with pytest.raises(RuntimeError, match="pose-estimator stream failed") as raised:
        stream.latest()
    assert "no angular velocity" in str(raised.value.__cause__)


def test_pose_stream_requires_source_timestamp() -> None:
    stream = motion_hardware.PoseStream(motion_hardware.HardwareMotionConfig())
    stream._call = iter(
        (
            pose_response(
                sequence=1,
                sample_monotonic_ns=0,
                position=(0.0, 0.0, 0.8),
                velocity=(0.0, 0.0, 0.0),
            ),
        )
    )

    stream._consume()

    with pytest.raises(RuntimeError, match="pose-estimator stream failed") as raised:
        stream.latest()
    assert "source timestamp must be positive" in str(raised.value.__cause__)


def test_pose_and_imu_orientation_must_agree() -> None:
    frame = HardwareFrame(
        positions=np.zeros(1),
        velocities=np.zeros(1),
        orientation_xyzw=np.asarray((0.0, 0.0, 1.0, 0.0)),
        angular_velocity=np.zeros(3),
        linear_acceleration=np.zeros(3),
        max_motor_age_ms=0,
        imu_age_ms=0,
        max_group_age_ms=0,
        sample_span_ms=0.0,
        state_frame_round_trip_ms=0.0,
        group_states=("IDLE",),
        group_causes=("UNSPECIFIED",),
        group_details=("",),
    )
    with pytest.raises(RuntimeError, match="orientation mismatch"):
        motion_hardware.validate_pose_imu_agreement(
            frame,
            pose_frame(),
            motion_hardware.HardwareMotionConfig(max_pose_imu_orientation_error=0.1),
        )


def test_motion_config_rejects_unsafe_ranges() -> None:
    assert motion_hardware.HardwareMotionConfig().max_duration_seconds is None
    assert motion_hardware.HardwareMotionConfig().max_state_sample_span_ms == 40.0
    with pytest.raises(ValueError, match="end_effector_payload_mass_kg"):
        motion_hardware.validate_hardware_motion_config(
            motion_hardware.HardwareMotionConfig(end_effector_payload_mass_kg=-1.0)
        )
    motion_hardware.validate_hardware_motion_config(
        motion_hardware.HardwareMotionConfig(max_duration_seconds=None)
    )
    with pytest.raises(ValueError, match="max_duration_seconds"):
        motion_hardware.validate_hardware_motion_config(
            motion_hardware.HardwareMotionConfig(max_duration_seconds=0.0)
        )
    with pytest.raises(ValueError, match="start_time_seconds"):
        motion_hardware.validate_hardware_motion_config(
            motion_hardware.HardwareMotionConfig(start_time_seconds=-1.0)
        )
    with pytest.raises(ValueError, match="max_action_abs"):
        motion_hardware.validate_hardware_motion_config(
            motion_hardware.HardwareMotionConfig(
                max_action_abs=2.0, max_raw_action_abs=1.0
            )
        )
    with pytest.raises(ValueError, match="action_lpf_cutoff_hz"):
        motion_hardware.validate_hardware_motion_config(
            motion_hardware.HardwareMotionConfig(action_lpf_cutoff_hz=0.0)
        )
    with pytest.raises(ValueError, match="telemetry age limits"):
        motion_hardware.validate_hardware_motion_config(
            motion_hardware.HardwareMotionConfig(max_group_age_ms=0)
        )
    with pytest.raises(ValueError, match="max_state_sample_span_ms"):
        motion_hardware.validate_hardware_motion_config(
            motion_hardware.HardwareMotionConfig(max_state_sample_span_ms=0.0)
        )


def test_motion_hardware_rejects_payload_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = SimpleNamespace(
        asset=ModelConfig(end_effector_payload_mass_kg=4.5359237)
    )
    runtime = SimpleNamespace(
        env=object(),
        policy=object(),
        metadata=metadata,
        normalization=object(),
    )
    monkeypatch.setattr(
        motion_hardware,
        "load_motion_policy_runtime",
        lambda *args, **kwargs: runtime,
    )
    monkeypatch.setattr(
        motion_hardware,
        "HardwareMotionPolicyAdapter",
        lambda *args, **kwargs: object(),
    )

    with pytest.raises(ValueError, match="does not match the motion checkpoint"):
        motion_hardware.load_hardware_motion_policy(
            Path("checkpoint"),
            motion_hardware.HardwareMotionConfig(end_effector_payload_mass_kg=3.0),
        )

    adapter, restored = motion_hardware.load_hardware_motion_policy(
        Path("checkpoint"),
        motion_hardware.HardwareMotionConfig(end_effector_payload_mass_kg=4.5359237),
    )
    assert adapter is not None
    assert restored is metadata


def test_motion_transition_requires_matching_impedance_gains() -> None:
    standing = SimpleNamespace(
        physical_coordinate_names=("left", "right"),
        nominal_physical_target=np.zeros(2),
        kp=np.asarray((1.0, 2.0)),
        kd=np.asarray((0.1, 0.2)),
    )
    motion = SimpleNamespace(
        physical_coordinate_names=("left", "right"),
        nominal_physical_target=np.zeros(2),
        kp=np.asarray((1.0, 3.0)),
        kd=np.asarray((0.1, 0.2)),
    )

    with pytest.raises(ValueError, match="different impedance gains"):
        motion_hardware.validate_motion_transition_adapters(standing, motion)


def test_motion_alignment_reuses_cached_start_reference() -> None:
    calls = 0

    class FakeEnv:
        action_size = 2

        def aligned_reference(self, *_args):
            nonlocal calls
            calls += 1
            return SimpleNamespace(
                qpos=jnp.asarray((1.0, 2.0, 0.8, 1.0, 0.0, 0.0, 0.0))
            )

    adapter = object.__new__(motion_hardware.HardwareMotionPolicyAdapter)
    adapter.env = FakeEnv()
    adapter.config = motion_hardware.HardwareMotionConfig()
    adapter._alignment_source = None
    adapter._aligned = False

    adapter.initialize_alignment(pose_frame())
    first_alignment = adapter._alignment_xy.copy()
    adapter.initialize_alignment(pose_frame())

    assert calls == 1
    np.testing.assert_array_equal(adapter._alignment_xy, first_alignment)


def test_fused_motion_inference_updates_stack_and_clips_action(monkeypatch) -> None:
    class FakeEnv:
        sampling_bin_count = 1
        observation_size = 2
        action_size = 2
        anchor_body_id = 0
        anchor_body_index = 0
        maximum_observation_delay_steps = 0
        control_timestep = 0.02

        def _actor_feedback(self, data):
            return data.qpos

        def termination_flags(self, _state):
            return {"done/any": jnp.asarray(False)}

        def observation(self, state):
            return state.data.qpos[:2] + state.last_action[:2]

        def action_target(self, state, action):
            return action + state.joint_default_offset[:2]

        def aligned_reference(self, _time, _alignment_xy, _alignment_yaw):
            return SimpleNamespace(
                body_pos=jnp.zeros((1, 3), dtype=jnp.float32),
                body_quat=jnp.asarray(((1.0, 0.0, 0.0, 0.0),)),
            )

    monkeypatch.setattr(
        motion_hardware,
        "normalize_observation",
        lambda observation, *_args: observation,
    )
    monkeypatch.setattr(
        motion_hardware,
        "apply_actor",
        lambda _policy, observation: observation[:2],
    )
    metadata = SimpleNamespace(
        ppo=SimpleNamespace(policy=SimpleNamespace(obs_stack_size=2))
    )
    env = FakeEnv()
    env.model = SimpleNamespace(mj_model=SimpleNamespace(nq=2, nv=2, nu=2, nbody=1))
    inference = motion_hardware._build_motion_inference(
        env,
        None,
        None,
        metadata,
        motion_hardware.HardwareMotionConfig(max_action_abs=0.5),
    )
    layout = motion_hardware._motion_inference_layout(env)
    hardware_state = np.zeros(layout.size, dtype=np.float32)
    hardware_state[layout.qpos] = (0.75, -0.25)
    hardware_state[layout.xpos] = (1.0, 0.0, 0.0)
    hardware_state[layout.xquat] = (1.0, 0.0, 0.0, 0.0)
    hardware_state[layout.last_action] = (0.1, 0.2)
    hardware_state[layout.alignment_xy] = 0.0
    hardware_state[layout.initialize_stack] = 1.0
    result = jax.device_get(
        inference(
            hardware_state,
            jnp.zeros((2, 2)),
        )
    )

    assert result.observation == pytest.approx((0.85, -0.05))
    np.testing.assert_allclose(
        result.observation_stack,
        ((0.85, -0.05), (0.85, -0.05)),
    )
    assert result.raw_action == pytest.approx((0.85, -0.05))
    alpha = 1.0 - math.exp(-2.0 * math.pi * 15.0 * env.control_timestep)
    expected_action = np.asarray((0.1, 0.2)) + alpha * (
        np.asarray((0.5, -0.05)) - np.asarray((0.1, 0.2))
    )
    assert result.action == pytest.approx(expected_action)
    assert result.physical_target == pytest.approx(expected_action)
    assert not bool(result.done_flags["done/any"])
    assert float(result.anchor_position_error) == pytest.approx(1.0)
    assert float(result.anchor_orientation_error) == pytest.approx(0.0)


def test_read_only_preflight_never_acquires_control(
    monkeypatch,
    capsys,
    hardware_preflight_factory,
) -> None:
    coordinates = ("left", "right")
    hardware = hardware_preflight_factory(coordinates, (2.0, 2.0))
    adapter = SimpleNamespace(
        physical_coordinate_names=coordinates,
        hardware_imu_frame_name="pelvis_link",
        coordinate_index={"left": 0, "right": 1},
        nominal_physical_target=np.zeros(2),
        kp=np.ones(2),
        kd=np.ones(2),
        env=SimpleNamespace(action_size=25, motion=SimpleNamespace(duration=1.0)),
        control_timestep=0.02,
        start_time=0.0,
        precompile=lambda: None,
        initialize_alignment=lambda _pose: pytest.fail(
            "motion alignment must wait until after standing"
        ),
        step=lambda *_args, **_kwargs: pytest.fail(
            "motion termination must wait until after standing"
        ),
        reset=lambda: None,
    )
    standing_preview = SimpleNamespace(action=np.zeros(2))
    standing_adapter = SimpleNamespace(
        physical_coordinate_names=coordinates,
        hardware_imu_frame_name="pelvis_link",
        coordinate_index={"left": 0, "right": 1},
        nominal_physical_target=np.zeros(2),
        kp=np.ones(2),
        kd=np.ones(2),
        env=SimpleNamespace(action_size=13),
        control_timestep=0.02,
        precompile=lambda: None,
        step=lambda _frame, _command: standing_preview,
        reset=lambda: None,
    )
    monkeypatch.setattr(
        motion_hardware,
        "load_hardware_motion_policy",
        lambda *_args, **_kwargs: (
            adapter,
            SimpleNamespace(
                task="motion_tracking",
                asset=ModelConfig(),
            ),
        ),
    )
    monkeypatch.setattr(
        motion_hardware,
        "load_hardware_policy",
        lambda *_args, **_kwargs: (
            standing_adapter,
            SimpleNamespace(
                task="locomotion",
                command=SimpleNamespace(
                    enabled=True,
                    standing_probability=1.0,
                ),
            ),
        ),
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

    motion_hardware.run_hardware_motion(
        Path("unused"),
        Path("standing"),
        motion_hardware.HardwareMotionConfig(),
        hardware_binding=hardware.binding,
        pose_stream_factory=FakePoseStream,
    )

    assert not hardware.control_requests
    output = capsys.readouterr().out
    assert "'nominal_ramp_distance_max': 2.0" in output
    assert "no lease was acquired" in output


@pytest.mark.parametrize(
    "motion_exit",
    (StopIteration, KeyboardInterrupt),
)
def test_motion_runner_flows_from_standing_to_motion_and_back(
    monkeypatch,
    capsys,
    motion_exit,
) -> None:
    coordinates = ("left", "right")
    frame = HardwareFrame(
        positions=np.zeros(2),
        velocities=np.zeros(2),
        orientation_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
        angular_velocity=np.zeros(3),
        linear_acceleration=np.zeros(3),
        max_motor_age_ms=0,
        imu_age_ms=0,
        max_group_age_ms=0,
        sample_span_ms=0.0,
        state_frame_round_trip_ms=0.0,
        group_states=("RUNNING",),
        group_causes=("START",),
        group_details=("",),
    )
    transitions = []
    standing_steps = []
    tilt_limits = []

    standing_adapter = SimpleNamespace(
        physical_coordinate_names=coordinates,
        nominal_physical_target=np.zeros(2),
        control_timestep=0.02,
        reset=lambda: None,
        step=lambda _frame, command: (
            standing_steps.append(command.copy())
            or SimpleNamespace(
                physical_target=np.zeros(2),
                tilt_radians=0.0,
                action=np.zeros(2),
            )
        ),
    )

    motion_steps = []

    def motion_step(*_args, **_kwargs):
        motion_steps.append(True)
        if len(motion_steps) > 1:
            raise motion_exit
        return SimpleNamespace(physical_target=np.zeros(2))

    motion_adapter = SimpleNamespace(
        physical_coordinate_names=coordinates,
        nominal_physical_target=np.zeros(2),
        control_timestep=0.02,
        start_time=0.0,
        initialize_alignment=lambda _pose: transitions.append("aligned"),
        step=motion_step,
    )

    class FakeControl:
        deadline = 0.0
        previous_target = np.zeros(2)

        @property
        def adapter(self):
            return self._adapter

        @adapter.setter
        def adapter(self, adapter):
            self._adapter = adapter
            transitions.append("motion" if adapter is motion_adapter else "standing")

        def ramp_to_nominal(self, _positions):
            return frame

        def wait_for_run(
            self,
            _prompt,
            *,
            operator_input,
            validate_hold,
        ):
            del operator_input
            validate_hold(frame)
            return True

        def read_frame(self, *, max_tilt_radians=None):
            tilt_limits.append(max_tilt_radians)
            return frame

        def send_target(self, target):
            self.previous_target = np.asarray(target)

        def pace(self):
            time.sleep(0.002)

        def reset_deadline(self):
            return None

        def drain(self):
            return None

    control = FakeControl()
    monkeypatch.setattr(
        motion_hardware,
        "armed_control_session",
        lambda *_args, **_kwargs: nullcontext(control),
    )

    def confirm_motion(expected, _prompt, hold_step, _operator_input):
        assert expected == "MOTION"
        assert hold_step()
        return True

    monkeypatch.setattr(
        motion_hardware,
        "wait_for_operator_confirmation_while_holding",
        confirm_motion,
    )
    poses = SimpleNamespace(latest=pose_frame)

    motion_hardware.control_physical_motion(
        SimpleNamespace(),
        {},
        SimpleNamespace(),
        poses,
        frame,
        standing_adapter,
        motion_adapter,
        motion_hardware.HardwareMotionConfig(
            max_duration_seconds=0.01,
            policy_transition_seconds=0.001,
        ),
    )

    assert transitions[:3] == ["motion", "aligned", "standing"]
    assert None in tilt_limits
    assert motion_hardware.HardwareMotionConfig().max_motion_tilt_radians in tilt_limits
    assert len(motion_steps) == 2
    assert standing_steps
    assert all(step == pytest.approx(np.zeros(3)) for step in standing_steps)
    output = capsys.readouterr().out
    assert "Standing locomotion policy resumed." in output
    if motion_exit is KeyboardInterrupt:
        assert "Motion tracking interrupted; transitioning back" in output
        assert "entering damped shutdown" not in output
    else:
        assert "'motion_hardware_summary': 'reference_completed'" in output
