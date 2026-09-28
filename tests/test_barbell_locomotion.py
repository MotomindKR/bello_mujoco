import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from bello_mujoco.evaluation.barbell_locomotion import NativeController


def test_native_feedback_adapter_preserves_tilt_delay_and_held_arm(monkeypatch):
    quaternion = Rotation.from_euler("xyz", [0.1, -0.2, 0.3]).as_quat()
    gravity = Rotation.from_quat(quaternion).inv().apply([0, 0, -1])
    sample = SimpleNamespace(
        orientation_xyzw=quaternion,
        linear_acceleration_xyz=-9.81 * gravity,
        angular_velocity_xyz=(0.1, 0.2, 0.3),
    )
    monkeypatch.setattr(
        "bello_mujoco.evaluation.barbell_locomotion.MujocoImu.bind",
        lambda model: SimpleNamespace(sample=lambda data: sample),
    )
    process = SimpleNamespace(
        stdin=io.StringIO(),
        stdout=io.StringIO((json.dumps({"motor_target": [2.0, -99.0]}) + "\n") * 2),
        wait=lambda timeout: 0,
    )
    monkeypatch.setattr(
        "bello_mujoco.evaluation.barbell_locomotion.subprocess.Popen",
        lambda *args, **kwargs: process,
    )
    config = SimpleNamespace(
        observation_delay_steps=1,
        binary=Path("native"),
        policy=Path("policy"),
        source=Path("source"),
    )
    metadata = dict(
        policy_indices=[0],
        control_timestep=0.02,
        environment={"observation_low_pass_filter": {"enabled": True, "cutoff_hz": 10}},
        motor_limits=[[-1, 1]] * 2,
        joint_limits=[[-1, 1]] * 2,
        joint_from_motor=[[1, 0], [0, 1]],
        position_step_limits=[0.2, 0.2],
    )
    controller = NativeController(
        config, metadata, SimpleNamespace(nu=2), np.array([0, 0.3]), io.StringIO()
    )
    data = SimpleNamespace(
        actuator_length=np.array([0.1, 0.3]), actuator_velocity=np.zeros(2)
    )
    target = controller(data)
    assert target[0] == pytest.approx(0.2, abs=1e-6)
    assert target[1] == pytest.approx(0.3)
    data.actuator_length[0] = 0.4
    controller(data)
    samples = [json.loads(line) for line in process.stdin.getvalue().splitlines()]
    for feedback in samples:
        restored_gravity = (
            Rotation.from_quat(feedback["orientation_xyzw"]).inv().apply([0, 0, -1])
        )
        np.testing.assert_allclose(restored_gravity, gravity, atol=1e-6)
        restored_acc = (
            np.array(feedback["linear_acceleration"]) - 9.80665 * restored_gravity
        )
        np.testing.assert_allclose(restored_acc, -9.81 * gravity, atol=1e-6)
        assert feedback["motor_position"][0] == pytest.approx(0.1)
        assert feedback["command"] == [0, 0, 0]
        assert feedback["arm_target"] == pytest.approx([0.3])
    assert controller.calls == 2
    assert controller.constrained_commands == 2
    controller.close()
