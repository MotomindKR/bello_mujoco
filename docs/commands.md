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
| `bello-run-hands` | [hands.yaml](../configs/commands/hands.yaml) | RTMW-driven Linker hand control |
| `bello-run-reference` | [reference.yaml](../configs/commands/reference.yaml) | Independent RTMW/GMR joint-reference producer |

## Training and evaluation

Training commands assume the Linux x86 `nix develop .#training` shell.
Use the default `nix develop` shell for CPU evaluation and development.

```bash
uv run --group training bello-train experiment=bello_mini_locomotion seed=1
uv run --group training bello-train experiment=bello_mini_standing
uv run --group training bello-train experiment=motion motion_path=motions/asap_fixed_head/CLIP.npz
uv run --group training bello-train experiment=bello_mini_locomotion resume_path=checkpoints/RUN
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
nix develop .#rust-runtime --command cargo run --locked \
  --manifest-path deps/bello-stack/Cargo.toml -p bello-mujoco -- \
  run --config deps/bello-stack/components/bello-mujoco/config/emulator_mini.yaml
nix develop .#rust-runtime --command cargo run --locked -p bello-policy -- \
  --policy outputs/policy.zip --source . \
  --hardware-config rust/bello-policy/config/hardware.yaml
```

Start hardwared and the visualizer/remote service separately; see [Lima](macos_vcan.md).
The emulator and policy runner are native commands, not Hydra/Python entrypoints.
See [emulator configuration](emulator.md) and the
[policy runtime](../rust/bello-policy/README.md).
Policy control defaults to read-only preflight. `--arm` enables the staged
operator-confirmed workflow for authorized commissioning. Physical deployment
remains gated on trained-policy validation; see [hardware](hardware.md).

RTMW upper-body following requires the live service and licensed SMPL-X model.
Independent hand control uses classic CAN directly, not hardwared motor targets.
Neither is required when running a fixed-arm locomotion policy.

## Remote jobs

See [Vast.ai](vast_ai.md). Remote completion/failure and the host deadline own
rental teardown. Closing a local monitor does not destroy a running job.
