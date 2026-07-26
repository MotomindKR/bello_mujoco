from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from enum import Enum
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import jax
import jax.numpy as jnp
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


def test_bounded_beyondmimic_checkpoint_is_hardware_compatible() -> None:
    action_layout = object()
    observation_layout = object()
    metadata = SimpleNamespace(
        task="motion_tracking",
        motion=object(),
        motion_env=SimpleNamespace(formulation="beyondmimic_joint_limits"),
        ppo=SimpleNamespace(
            policy=SimpleNamespace(action_distribution="tanh_gaussian")
        ),
        asset=SimpleNamespace(
            end_effector_sphere_mode="both",
            joint_availability=SimpleNamespace(enabled_joint_names={"waist_yaw_joint"}),
        ),
        action_layout=action_layout,
        actor_observation_layout=observation_layout,
    )
    env = SimpleNamespace(
        action_size=25,
        action_layout=action_layout,
        observation_layout=observation_layout,
    )

    motion_hardware.validate_motion_hardware_checkpoint(env, metadata)

    metadata.ppo.policy.action_distribution = "gaussian"
    with pytest.raises(ValueError, match="tanh_gaussian"):
        motion_hardware.validate_motion_hardware_checkpoint(env, metadata)


def test_fused_motion_inference_updates_stack_and_clips_action(monkeypatch) -> None:
    class FakeEnv:
        sampling_bin_count = 1
        observation_size = 2
        action_size = 2
        anchor_body_id = 0
        anchor_body_index = 0

        def _done_flags(self, _state):
            return {"done/any": jnp.asarray(False)}

        def observation(self, state):
            return state.data.qpos[:2] + state.last_action[:2]

        def _action_to_ctrl(self, state, action):
            return action + state.joint_default_offset[:2]

        def _aligned_reference(self, _time, _alignment_xy, _alignment_yaw):
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
        motion_hardware.MotionHardwareConfig(max_action_abs=0.5),
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


def test_nominal_pose_is_checked_after_ramp() -> None:
    adapter = SimpleNamespace(
        nominal_physical_target=np.zeros(2),
        physical_coordinate_names=("left", "right"),
    )
    frame = SimpleNamespace(positions=np.asarray((0.1, 0.8)))

    with pytest.raises(RuntimeError, match="did not reach.*right"):
        motion_hardware.require_nominal_pose(frame, adapter, 0.75)

    motion_hardware.require_nominal_pose(frame, adapter, 0.8)


def test_armed_runner_requires_async_hardware_binding() -> None:
    with pytest.raises(RuntimeError, match="send_async"):
        motion_hardware.require_async_hardware_binding(SimpleNamespace())

    motion_hardware.require_async_hardware_binding(
        SimpleNamespace(ControlSession=SimpleNamespace(send_async=lambda: None))
    )


def test_motion_targets_are_submitted_to_both_groups_asynchronously(
    monkeypatch,
) -> None:
    submitted = []

    class Receipt:
        def done(self):
            return True

        def result(self):
            return None

    class Control:
        def __init__(self, name):
            self.name = name

        def send_async(self, targets):
            submitted.append((self.name, targets))
            return Receipt()

    config = HardwareLocomotionConfig(groups=("lower", "upper"))
    groups = {name: SimpleNamespace(coordinates=(name,)) for name in config.groups}
    controls = {name: Control(name) for name in config.groups}
    adapter = SimpleNamespace(control_timestep=0.02)
    monkeypatch.setattr(
        motion_hardware,
        "coordinate_targets",
        lambda _binding, coordinates, *_args: {coordinates[0]: coordinates[0]},
    )
    pending = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        motion_hardware.send_motion_targets(
            SimpleNamespace(),
            controls,
            groups,
            np.zeros(2),
            adapter,
            config,
            executor,
            pending,
        )
        motion_hardware.drain_command_receipts(pending)

    assert sorted(submitted) == [
        ("lower", {"lower": "lower"}),
        ("upper", {"upper": "upper"}),
    ]


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
            position=2.0, velocity=0.0, feedback_age_ms=0
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
    output = capsys.readouterr().out
    assert "'nominal_ramp_distance_max': 2.0" in output
    assert "no lease was acquired" in output
