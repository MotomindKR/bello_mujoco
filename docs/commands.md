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
| `bello-run-hands` | [hands.yaml](../configs/commands/hands.yaml) | SMPL-X gRPC driven Linker hand control |
| `bello-run-reference` | [reference.yaml](../configs/commands/reference.yaml) | Direct XRoboToolkit/GMR joint-reference producer |

## Training and evaluation

Training commands assume the Linux x86 `nix develop .#training` shell.
Use the default `nix develop` shell for CPU evaluation and development.

```bash
uv run --group training bello-train experiment=bello_mini_locomotion seed=1
uv run --group training bello-train experiment=bello_mini_standing
uv run --group training bello-train experiment=bello_mini_loaded_standing # requires screened arm bank
uv run --group training bello-train experiment=motion motion_path=motions/asap_fixed_head/CLIP.npz
uv run --group training bello-train experiment=bello_mini_motion motion_path=/path/to/mini/CLIP.npz
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

For a freshly trained torso-conditioned locomotion policy, add
`torso_command=[0.1,-0.15,0.2]` to `bello-eval`. Values are roll/pitch/yaw radians;
the torso target can accompany either fixed velocity commands or gamepad walking.
It is not a motion-playback setting. See [torso command semantics](training_configuration.md#torso-orientation-commands).

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

Live upper-body following requires the updated tracking bridge with typed raw
body poses. Only finger control needs the licensed neutral SMPL-X model.
Start `pico-smplx-bridge` on the tracking PC, then run:

```bash
uv run bello-run-reference asset=bello_mini \
  xrobot.endpoint=192.168.0.216:50051
uv run bello-run-hands \
  smplx.endpoint=192.168.0.216:50051 \
  smplx.neutral_model_path=/path/to/SMPLX_NEUTRAL.npz \
  duration_seconds=10
```

The reference service listens on `127.0.0.1:50056` by default; `listen` changes
that address. The hand command defaults to unarmed preflight and opens no CAN
connection. Armed independent hand control uses classic CAN directly. Neither
command is required for a fixed-arm locomotion policy. See
[live references and hands](hardware.md#live-references-and-hands) for the pose
contract, validity rules, and freshness settings.

## Remote jobs

See [Vast.ai](vast_ai.md). Remote completion/failure and the host deadline own
rental teardown. Closing a local monitor does not destroy a running job.

The barbell recipes (`bello_mini_loaded_standing`, `bello_mini_barbell_motion`)
require the [reference preparation and review gate](barbell_feasibility.md).
Generate artifacts with `uv run python -m bello_mujoco.evaluation.barbell_reference
--output outputs/barbell-training --standing --video`. The current 30 kg hold
fails the effort screen; neither recipe is cleared for long training.
