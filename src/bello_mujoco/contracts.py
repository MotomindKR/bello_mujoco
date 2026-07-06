"""Shared deployment and training contracts for Bello lower-body locomotion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

POLICY_JOINT_ORDER: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
)

DEFAULT_JOINT_POS: dict[str, float] = {
    "left_hip_pitch_joint": -0.325,
    "right_hip_pitch_joint": 0.325,
    "left_hip_roll_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.125,
    "right_hip_yaw_joint": -0.125,
    "left_knee_joint": 0.45,
    "right_knee_joint": 0.45,
    "left_ankle_pitch_joint": 0.30,
    "right_ankle_pitch_joint": 0.30,
    "left_ankle_roll_joint": 0.0,
    "right_ankle_roll_joint": 0.0,
}

ACTION_SCALE: dict[str, float] = {
    "left_hip_pitch_joint": 0.35,
    "right_hip_pitch_joint": 0.35,
    "left_hip_roll_joint": 0.25,
    "right_hip_roll_joint": 0.25,
    "left_hip_yaw_joint": 0.25,
    "right_hip_yaw_joint": 0.25,
    "left_knee_joint": 0.45,
    "right_knee_joint": 0.45,
    "left_ankle_pitch_joint": 0.20,
    "right_ankle_pitch_joint": 0.20,
    "left_ankle_roll_joint": 0.05,
    "right_ankle_roll_joint": 0.05,
}

SOFT_JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "left_hip_pitch_joint": (-1.41372, 1.41372),
    "right_hip_pitch_joint": (-1.41372, 1.41372),
    "left_hip_roll_joint": (-1.41372, 1.41372),
    "right_hip_roll_joint": (-1.41372, 1.41372),
    "left_hip_yaw_joint": (-1.41372, 1.41372),
    "right_hip_yaw_joint": (-1.41372, 1.41372),
    "left_knee_joint": (0.10472, 1.98968),
    "right_knee_joint": (0.10472, 1.98968),
    "left_ankle_pitch_joint": (-0.44506, 0.96866),
    "right_ankle_pitch_joint": (-0.44506, 0.96866),
    "left_ankle_roll_joint": (-0.07857, 0.07857),
    "right_ankle_roll_joint": (-0.07857, 0.07857),
}

ACTUATOR_GAINS: dict[str, tuple[float, float]] = {
    "left_hip_pitch_joint": (180.0, 5.0),
    "right_hip_pitch_joint": (180.0, 5.0),
    "left_hip_roll_joint": (120.0, 5.0),
    "right_hip_roll_joint": (120.0, 5.0),
    "left_hip_yaw_joint": (120.0, 5.0),
    "right_hip_yaw_joint": (120.0, 5.0),
    "left_knee_joint": (200.0, 6.0),
    "right_knee_joint": (200.0, 6.0),
    "left_ankle_pitch_joint": (40.0, 2.0),
    "right_ankle_pitch_joint": (40.0, 2.0),
    "left_ankle_roll_joint": (40.0, 2.0),
    "right_ankle_roll_joint": (40.0, 2.0),
}

FIXED_JOINT_DEFAULT_POS: dict[str, float] = {
    "Revolute 1": 0.0,
    "Revolute 2": 0.0,
    "Revolute 3": 0.0,
    "Revolute 4": 0.0,
    "Revolute 5": 0.0,
    "Revolute 6": 0.0,
    "Revolute 7": 0.0,
    "Revolute 8": 0.0,
    "Revolute 9": 0.0,
    "Revolute 10": 0.0,
    "Revolute 11": 0.0,
    "Revolute 12": 0.0,
    "Revolute 13": 0.0,
    "Revolute 14": 0.0,
    "Revolute 15": 0.0,
    "Revolute 16": 0.0,
    "Revolute 17": 0.0,
    "Revolute 18": 0.0,
    "Revolute 19": 0.0,
    "Slider 21": 0.0,
    "Slider 22": 0.0,
    "Slider 24": 0.0,
    "Slider 25": 0.0,
}

FIXED_JOINT_GAINS: dict[str, tuple[float, float]] = {
    "Revolute 1": (180.0, 12.0),
    "Revolute 2": (240.0, 20.0),
    "Revolute 3": (240.0, 20.0),
    "Revolute 4": (5.5, 0.5),
    "Revolute 5": (5.5, 0.5),
    "Revolute 6": (70.0, 2.0),
    "Revolute 7": (70.0, 2.0),
    "Revolute 8": (20.0, 1.0),
    "Revolute 9": (40.0, 2.0),
    "Revolute 10": (20.0, 1.0),
    "Revolute 11": (10.0, 1.0),
    "Revolute 12": (10.0, 1.0),
    "Revolute 13": (70.0, 2.0),
    "Revolute 14": (70.0, 2.0),
    "Revolute 15": (20.0, 1.0),
    "Revolute 16": (40.0, 2.0),
    "Revolute 17": (20.0, 1.0),
    "Revolute 18": (10.0, 1.0),
    "Revolute 19": (10.0, 1.0),
    "Slider 21": (100.0, 5.0),
    "Slider 22": (100.0, 5.0),
    "Slider 24": (100.0, 5.0),
    "Slider 25": (100.0, 5.0),
}

FIXED_JOINT_ORDER: tuple[str, ...] = tuple(FIXED_JOINT_DEFAULT_POS)
FULL_BODY_ACTION_DIM = len(POLICY_JOINT_ORDER) + len(FIXED_JOINT_ORDER)

OBS_TERM_DIMS: dict[str, int] = {
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "command": 3,
    "actuator_joint_pos": FULL_BODY_ACTION_DIM,
    "actuator_joint_vel": FULL_BODY_ACTION_DIM,
    "last_action": FULL_BODY_ACTION_DIM,
}

FRAME_OBS_DIM = sum(OBS_TERM_DIMS.values())
ACTION_DIM = len(POLICY_JOINT_ORDER)
ACTION_CLIP = 10.0
DEFAULT_ROOT_POS = (0.0, 0.0, 0.91)
DEFAULT_ROOT_QUAT = (1.0, 0.0, 0.0, 0.0)
SIM_TIMESTEP = 0.002
CONTROL_TIMESTEP = 0.02
CONTROL_DECIMATION = int(round(CONTROL_TIMESTEP / SIM_TIMESTEP))
EPISODE_SECONDS = 20.0
MAX_EPISODE_STEPS = int(round(EPISODE_SECONDS / CONTROL_TIMESTEP))


@dataclass(frozen=True)
class LowerBodyContract:
    joint_order: tuple[str, ...] = POLICY_JOINT_ORDER
    action_clip: float = ACTION_CLIP
    sim_timestep: float = SIM_TIMESTEP
    control_timestep: float = CONTROL_TIMESTEP
    control_decimation: int = CONTROL_DECIMATION
    max_episode_steps: int = MAX_EPISODE_STEPS

    @property
    def default_joint_pos(self) -> np.ndarray:
        return ordered_values(DEFAULT_JOINT_POS)

    @property
    def action_scale(self) -> np.ndarray:
        return ordered_values(ACTION_SCALE)

    @property
    def soft_joint_limits(self) -> np.ndarray:
        return np.asarray([SOFT_JOINT_LIMITS[name] for name in self.joint_order], dtype=np.float32)


def ordered_values(values: dict[str, float]) -> np.ndarray:
    return np.asarray([values[name] for name in POLICY_JOINT_ORDER], dtype=np.float32)
