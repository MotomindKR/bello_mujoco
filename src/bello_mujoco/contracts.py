"""Shared deployment and training contracts for Bello lower-body locomotion."""

from __future__ import annotations

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
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "neck_yaw_joint": 0.0,
    "head_pitch_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_pitch_joint": 0.0,
    "right_elbow_yaw_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_pitch_joint": 0.0,
    "left_elbow_yaw_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_finger_right_slide_joint": 0.0,
    "right_finger_left_slide_joint": 0.0,
    "left_finger_right_slide_joint": 0.0,
    "left_finger_left_slide_joint": 0.0,
}

FIXED_JOINT_GAINS: dict[str, tuple[float, float]] = {
    "waist_yaw_joint": (180.0, 12.0),
    "waist_roll_joint": (240.0, 20.0),
    "waist_pitch_joint": (240.0, 20.0),
    "neck_yaw_joint": (5.5, 0.5),
    "head_pitch_joint": (5.5, 0.5),
    "right_shoulder_pitch_joint": (70.0, 2.0),
    "right_shoulder_roll_joint": (70.0, 2.0),
    "right_shoulder_yaw_joint": (20.0, 1.0),
    "right_elbow_pitch_joint": (40.0, 2.0),
    "right_elbow_yaw_joint": (20.0, 1.0),
    "right_wrist_pitch_joint": (10.0, 1.0),
    "right_wrist_yaw_joint": (10.0, 1.0),
    "left_shoulder_pitch_joint": (70.0, 2.0),
    "left_shoulder_roll_joint": (70.0, 2.0),
    "left_shoulder_yaw_joint": (20.0, 1.0),
    "left_elbow_pitch_joint": (40.0, 2.0),
    "left_elbow_yaw_joint": (20.0, 1.0),
    "left_wrist_pitch_joint": (10.0, 1.0),
    "left_wrist_yaw_joint": (10.0, 1.0),
    "right_finger_right_slide_joint": (100.0, 5.0),
    "right_finger_left_slide_joint": (100.0, 5.0),
    "left_finger_right_slide_joint": (100.0, 5.0),
    "left_finger_left_slide_joint": (100.0, 5.0),
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
