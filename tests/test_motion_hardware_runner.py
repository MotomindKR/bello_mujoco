from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bello_mujoco.hardware import motion as motion_hardware
from bello_mujoco.hardware.runtime import HardwareFrame


def pose_frame(*, age_ms: int = 0) -> motion_hardware.PoseFrame:
    return motion_hardware.PoseFrame(
        position=np.asarray((0.0, 0.0, 0.8)),
        orientation_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
        linear_velocity=np.zeros(3),
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


def test_fused_motion_inference_updates_stack_and_clips_action(monkeypatch) -> None:
    class FakeEnv:
        sampling_bin_count = 1
        observation_size = 2
        action_size = 2
        anchor_body_id = 0
        anchor_body_index = 0

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
    assert result.action == pytest.approx((0.5, -0.05))
    assert result.physical_target == pytest.approx((0.5, -0.05))
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
    preview = motion_hardware.HardwareMotionStep(
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
        coordinate_index={"left": 0, "right": 1},
        nominal_physical_target=np.zeros(2),
        kp=np.ones(2),
        kd=np.ones(2),
        env=SimpleNamespace(action_size=25, motion=SimpleNamespace(duration=1.0)),
        control_timestep=0.02,
        start_time=0.0,
        precompile=lambda: None,
        initialize_alignment=lambda _pose: None,
        step=lambda _frame, _pose, reference_time: preview,
        reset=lambda: None,
    )
    standing_preview = SimpleNamespace(action=np.zeros(2))
    standing_adapter = SimpleNamespace(
        physical_coordinate_names=coordinates,
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
        lambda *_args, **_kwargs: (adapter, SimpleNamespace(task="motion_tracking")),
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


def test_motion_runner_flows_from_standing_to_motion_and_back(
    monkeypatch,
    capsys,
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
        group_states=("RUNNING",),
        group_causes=("START",),
        group_details=("",),
    )
    transitions = []
    standing_steps = []

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

    def motion_step(*_args, **_kwargs):
        raise StopIteration

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

        def read_frame(self):
            return frame

        def send_target(self, _target):
            return None

        def pace(self):
            time.sleep(0.002)

        def reset_deadline(self):
            return None

        def drain(self):
            return None

        def transition_to_nominal(self, adapter, _duration):
            transitions.append(
                "motion" if adapter is motion_adapter else "standing"
            )
            return frame

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
    assert standing_steps
    assert all(step == pytest.approx(np.zeros(3)) for step in standing_steps)
    output = capsys.readouterr().out
    assert "Standing locomotion policy resumed." in output
