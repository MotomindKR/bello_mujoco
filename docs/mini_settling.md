# Mini continuous settling rewards

The previous recovery recipe made standing alignment conditional on both feet
contacting. A synthetic probe of the actual reward function found that an
18 cm stagger earned +0.120 per step with one foot down but -0.242 with both
feet down. A roughly 17.4-degree tilt could disable the alignment cost entirely.
Saved run xetme3xj improved push survival while double support fell to 7%.
These observations motivated structural corrections, not a blanket increase
in standing weights.

Mini now uses bounded `h/(1+h)` alignment costs throughout standing, including
single support. Both alignment weights are -0.25, limiting their combined cost
to 0.01 per step at 50 Hz. Geometry, foot-motion, contact and stillness terms
retain at least 25% strength during recovery. Soft hip/knee/waist and thermal
posture relaxation is otherwise unchanged; hard limits are never relaxed.
Regression tests verify contact-independent alignment costs, preferable
touchdown, affordable recovery steps and a persistent incentive to realign.

The 25% standing share, +/-15 cm torso COM range/curriculum, command ranges,
tracking tolerances, extra stop step, gait cadence, lateral starts, IMU frames,
feedback filtering/latency and actuator effort limits are unchanged. This is
a fresh-training recipe; old checkpoints explicitly retain their old reward
mode when metadata is loaded.

Checkpoint frequency is 1024 updates and video frequency 8192 updates. Each
video event evaluates the same policy in walking, 30-second quiet standing,
and four separate directional push cases. Settling is evaluated independently
of reward and survival, with physical traces and overlays. See
[training configuration](training_configuration.md#evaluation-videos-video_config).
The periodic suite is nominal seed 0, not a complete randomized or emulator
qualification. Multi-seed, transition-phase and native/emulator comparisons
remain required before deployment; no trained-policy success is implied by
unit tests or GPU smoke checks.

Reference review informed the structure, not a wholesale coefficient copy:

- [Booster T1](https://github.com/BoosterRobotics/booster_gym/blob/da396a06d6eed99e2de72d7749c48ee8748950f9/envs/t1.py)
  keeps foot geometry costs independent of double-support gating.
- [Playground T1](https://github.com/google-deepmind/mujoco_playground/blob/4057c147714b6ac09b395377f1a1724bbeacc4d3/mujoco_playground/_src/locomotion/t1/joystick.py)
  freezes zero-command phase while retaining foot-height/pose incentives.
- [Unitree G1](https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py)
  is locomotion-centric; its 2% explicit standing fraction is not our target.
