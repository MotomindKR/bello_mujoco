from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bello_mujoco.hardware.policy_log import PolicyStepLogger, policy_phase_schema


def test_policy_phase_schema_preserves_checkpoint_array_order() -> None:
    adapter = SimpleNamespace(
        env=SimpleNamespace(
            observation_layout=SimpleNamespace(
                components=(("gravity", 3), ("joint_position", 2))
            ),
            model=SimpleNamespace(
                actuator_joint_names=("hold", "left", "right"),
                policy_ctrl_indices=np.asarray((2, 1)),
            ),
        ),
        metadata=SimpleNamespace(
            ppo=SimpleNamespace(policy=SimpleNamespace(obs_stack_size=4))
        ),
        physical_coordinate_names=("motor_a", "motor_b", "motor_c"),
    )

    assert policy_phase_schema(adapter) == {
        "observation_layout": [
            {"name": "gravity", "size": 3},
            {"name": "joint_position", "size": 2},
        ],
        "observation_stack_size": 4,
        "action_names": ["right", "left"],
        "physical_target_names": ["motor_a", "motor_b", "motor_c"],
    }


def test_policy_step_logger_writes_complete_jsonl_records(tmp_path: Path) -> None:
    path = tmp_path / "policy.jsonl"
    with PolicyStepLogger(
        path,
        task="motion",
        checkpoints={"motion": Path("checkpoint")},
        phases={"motion": {"observation_stack_size": 2}},
    ) as policy_log:
        policy_log.write_step(
            phase="motion",
            step=7,
            elapsed_seconds=1.25,
            observation=np.asarray((0.1, 0.2), dtype=np.float32),
            policy_observation=np.asarray((0.1, 0.2, 0.1, 0.2), dtype=np.float32),
            raw_action=np.asarray((1.1, -1.1), dtype=np.float32),
            action=np.asarray((1.0, -1.0), dtype=np.float32),
            physical_target=np.asarray((0.4, -0.3), dtype=np.float64),
            policy_target=np.asarray((0.5, -0.2), dtype=np.float64),
            command=np.asarray((0.0, 0.0, 0.0), dtype=np.float32),
            reference_time=2.5,
        )

    header, step = [json.loads(line) for line in path.read_text().splitlines()]
    assert header["type"] == "header"
    assert header["format_version"] == 1
    assert header["task"] == "motion"
    assert header["phases"] == {"motion": {"observation_stack_size": 2}}
    assert step["type"] == "policy_step"
    assert step["phase"] == "motion"
    assert step["step"] == 7
    assert step["elapsed_seconds"] == 1.25
    assert step["observation"] == pytest.approx((0.1, 0.2))
    assert step["policy_observation"] == pytest.approx((0.1, 0.2, 0.1, 0.2))
    assert step["raw_action"] == pytest.approx((1.1, -1.1))
    assert step["action"] == pytest.approx((1.0, -1.0))
    assert step["physical_target"] == pytest.approx((0.4, -0.3))
    assert step["policy_target"] == pytest.approx((0.5, -0.2))
    assert step["command"] == pytest.approx((0.0, 0.0, 0.0))
    assert step["reference_time"] == 2.5
    assert isinstance(step["wall_time_ns"], int)
    assert isinstance(step["monotonic_time_ns"], int)


def test_policy_step_logger_does_not_overwrite_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "policy.jsonl"
    path.write_text("existing\n")

    with pytest.raises(FileExistsError):
        PolicyStepLogger(
            path,
            task="locomotion",
            checkpoints={},
            phases={},
        )

    assert path.read_text() == "existing\n"
