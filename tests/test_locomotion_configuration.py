from dataclasses import replace

import pytest

from bello_mujoco.envs.locomotion_config import (
    BelloEnvConfig,
    ContactConfig,
    DoneConfig,
    ObservationLatencyConfig,
    ObservationLowPassFilterConfig,
    validate_locomotion_config,
)
from bello_mujoco.envs.rewards import CommandConfig, RewardConfig


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"decimation": 0}, "decimation"),
        ({"command_delay_seconds": 0.001}, "whole number"),
        ({"command_delay_seconds": 1.0}, "control_timestep"),
        ({"contact": ContactConfig(foot_contact_force_threshold=-1.0)}, "contact"),
        (
            {"observation_latency": ObservationLatencyConfig(delay_steps=(3, 1))},
            "lower bound",
        ),
        (
            {
                "observation_low_pass_filter": ObservationLowPassFilterConfig(
                    cutoff_hz=0.0
                )
            },
            "cutoff_hz",
        ),
    ],
)
def test_initialization_constraints_need_no_robot_model(changes, message):
    with pytest.raises(ValueError, match=message):
        validate_locomotion_config(
            replace(BelloEnvConfig(), **changes),
            RewardConfig(),
            CommandConfig(),
            DoneConfig(),
            physics_dt=0.0025,
        )
