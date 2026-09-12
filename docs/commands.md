# Commands

Commands use Hydra `field=value` overrides. `--cfg job` prints their public
configuration. Paths accept `~`; command execution does not change directory.
The YAML files linked below are the authoritative field/default reference.

| Command | Configuration | Purpose |
| --- | --- | --- |
| `bello-train` | [train.yaml](../configs/commands/train.yaml) | Locomotion or BeyondMimic PPO |
| `bello-eval` | [eval.yaml](../configs/commands/eval.yaml) | Linux CPU evaluation |
| `bello-export-policy` | [export.yaml](../configs/commands/export.yaml) | ONNX/reference deployment bundle |
| `bello-vast-train` | [vast_train.yaml](../configs/commands/vast_train.yaml) | Bounded remote training |
| `bello-run-emulator` | [emulator.yaml](../configs/commands/emulator.yaml) | MuJoCo/vCAN hardware emulation |
| `bello-run-hardware` | [hardware.yaml](../configs/commands/hardware.yaml) | Read-only preflight or explicitly armed policy |
| `bello-run-hands` | [hands.yaml](../configs/commands/hands.yaml) | RTMW-driven Linker hand control |

## Training and evaluation

```bash
uv run bello-train experiment=bello_mini_locomotion seed=1
uv run bello-train experiment=bello_mini_standing
uv run bello-train experiment=motion motion_path=motions/asap_fixed_head/CLIP.npz
uv run bello-train experiment=bello_mini_locomotion resume_path=checkpoints/RUN
uv run bello-eval task=locomotion checkpoint_path=checkpoints/RUN gamepad_enabled=false steps=1500
uv run bello-eval task=motion checkpoint_path=checkpoints/RUN output_path=outputs/motion.mp4
uv run bello-export-policy checkpoint_path=checkpoints/RUN output_path=outputs/policy.zip
```

Select recipes from `configs/profiles/train/`. Edit those recipes for nested
physics/reward changes; the CLI exposes only its declared overrides. Evaluation
and checkpoint resume require the checkpoint's exact clean source commit.
Evaluation rasterization is Mesa llvmpipe software-only on Linux/Lima. Enabled training
video failures are fatal; use `video_enabled=false` to disable videos explicitly.

## Hardware workflow

```bash
uv run bello-run-emulator asset=bello_mini
uv run bello-run-hardware task=locomotion checkpoint_path=checkpoints/RUN \
  remote_endpoint=http://127.0.0.1:50054 upper_body_tracking_enabled=false
```

Start hardwared and the visualizer/remote service separately; see [Lima](macos_vcan.md).
Hardware commands default to read-only preflight. `arm=true` enables the staged
operator-confirmed control workflow. It is not an authorization substitute for
physical safety checks. See [hardware](hardware.md) before arming.

RTMW upper-body following requires the live service and licensed SMPL-X model.
Independent hand control uses classic CAN directly, not hardwared motor targets.
Neither is required when running a fixed-arm locomotion policy.

## Remote jobs

See [Vast.ai](vast_ai.md). Remote completion/failure and the host deadline own
rental teardown. Closing a local monitor does not destroy a running job.
