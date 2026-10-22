# Project guide

- [Commands](commands.md)
- [Configuration](training_configuration.md)
- [Training and evaluation](training_and_evaluation.md)
- [Robot descriptions](robot_assets.md)
- [BeyondMimic tracking](motion_tracking.md)
- [Bello Mini policy console](policy_console.md)
- [SONIC LoRA transfer](sonic_transfer.md)
- [SuperMimic learning and deployment](supermimic.md)
- [SuperMimic port plan for Bello Mini](supermimic_plan.md)
- [Hardware](hardware.md), [emulator](emulator.md), and [Lima](macos_vcan.md)
- [PICO tracking and hand-control startup](pico_startup.md)
- [Hand control API and ownership](hand_control.md)
- [Barbell mode and handoff gates](barbell_mode.md)
- [Robot and tracking systemd workflow](systemd.md)
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
| `supermimic/` | Analytic kinematics, learned dynamics, replay, imagined actor/critic updates, return curriculum, checkpoints, and export |
| `evaluation/` | Native CPU rollouts, shared video resources, isolated software workers |
| `hardware/` | Direct XRoboToolkit body and FastSAM retargeting, reference producers, pose diagnostics, and independent hands |
| `motion/` | Reference clip loading and runtime reconstruction |
| `robots/` | Robot task defaults and coordinate mappings |
| `deployment.py` | Self-contained ONNX/reference deployment bundles |
| `rust/bello-policy/` | CPU inference, observations/gait, operator lifecycle, and live-reference following |

Python paths are relative to `src/bello_mujoco/`. Generated models, checkpoints,
videos, and caches are local artifacts, not source. Compatibility is the root
Git commit; there is no parallel asset-manifest or fingerprint versioning system.
Live messages still require shape, frame, freshness, and confidence validation.
Offline AMASS-to-Mini corpus preparation is documented under
[SONIC transfer](sonic_transfer.md#amass-research-corpus). It uses the pinned
GMR implementation without modifying it; source datasets remain read-only.
