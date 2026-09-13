# Project guide

- [Commands](commands.md)
- [Configuration](training_configuration.md)
- [Training and evaluation](training_and_evaluation.md)
- [Robot descriptions](robot_assets.md)
- [BeyondMimic tracking](motion_tracking.md)
- [Hardware](hardware.md), [emulator](emulator.md), and [Lima](macos_vcan.md)
- [Vast.ai](vast_ai.md)

## Code map

| Location | Responsibility |
| --- | --- |
| `configs/commands/` | Public Hydra inputs |
| `configs/profiles/train/` | Experiment recipes |
| `deps/bello-stack/` | Pinned robot, protocol, hardware client, and Rust runtime |
| `models/` | Thin canonical URDF-to-MuJoCo compilation |
| `envs/` | Task stepping, shared scalar/batch transitions, observations, gait, geometry, rewards, diagnostics |
| `training/` | PPO optimization/rollouts, explicit environment contract, normalization, metrics, shared host update loop, checkpoints and remote training |
| `evaluation/` | Native CPU rollouts, shared video resources, isolated software workers |
| `hardware/` | Live FastSAM/RTMW retargeting, reference/neutral-command producers, pose diagnostics, and independent hands |
| `motion/` | Reference clip loading and runtime reconstruction |
| `robots/` | Robot task defaults and coordinate mappings |
| `deployment.py` | Self-contained ONNX/reference deployment bundles |
| `rust/bello-policy/` | CPU inference, observations/gait, operator lifecycle, and live-reference following |

Python paths are relative to `src/bello_mujoco/`. Generated models, checkpoints,
videos, and caches are local artifacts, not source. Compatibility is the root
Git commit; there is no parallel asset-manifest or fingerprint versioning system.
Live messages still require shape, frame, freshness, and confidence validation.
Offline GMR/dataset tooling is outside this repository's current scope.
