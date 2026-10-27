# Commands

Commands use Hydra `field=value` overrides. `--cfg job` prints their public
configuration. Paths accept `~`; command execution does not change directory.
The YAML files linked below are the authoritative field/default reference.

| Command | Configuration | Purpose |
| --- | --- | --- |
| `bello-train` | [train.yaml](../configs/commands/train.yaml) | PPO tasks or Mini SuperMimic model-based learning |
| `bello-prepare-lafan` | [prepare_lafan.yaml](../configs/commands/prepare_lafan.yaml) | BVH or clock-matched native SMPL-X sequences through canonical Mini calibration |
| `bello-prepare-bones` | [prepare_bones.yaml](../configs/commands/prepare_bones.yaml) | Family-disjoint BONES SOMA selection, calibrated Mini references, and retained LAFAN splits |
| `bello-audit-reference` | [audit_reference.yaml](../configs/commands/audit_reference.yaml) | Read-only reference torque, contact, and acceleration diagnostics |
| `bello-audit-world-model` | [audit_world_model.yaml](../configs/commands/audit_world_model.yaml) | Contact-stratified autoregressive model errors against native rollouts |
| `bello-eval` | [eval.yaml](../configs/commands/eval.yaml) | Linux CPU evaluation |
| `bello-export-policy` | [export.yaml](../configs/commands/export.yaml) | ONNX/reference deployment bundle |
| `bello-vast-train` | [vast_train.yaml](../configs/commands/vast_train.yaml) | Bounded remote training |
| `bello-run-hands` | [hands.yaml](../configs/commands/hands.yaml) | Leased Linker hand control with application and XRoboToolkit sources |
| `bello-eval-barbell` | [barbell_eval.yaml](../configs/commands/barbell_eval.yaml) | Confirmation-gated barbell cycle in simulation |
| `bello-run-reference` | [reference.yaml](../configs/commands/reference.yaml) | Direct XRoboToolkit/GMR joint-reference producer |

## Training and evaluation

Training commands assume the Linux x86 `nix develop .#training` shell.
Use the default `nix develop` shell for CPU evaluation and development.

```bash
uv run --group training bello-train experiment=bello_mini_locomotion seed=1
uv run --group training bello-train experiment=bello_mini_standing
uv run --group training bello-train experiment=motion motion_path=motions/asap_fixed_head/CLIP.npz
uv run --group training bello-train experiment=bello_mini_motion motion_path=/path/to/mini/CLIP.npz
uv run --group training --group reference-validation bello-train \
  experiment=bello_mini_sonic_transfer motion_dataset=/data/mini-sonic \
  motion_dataset_artifact=ENTITY/PROJECT/mini-sonic:v0 \
  pretrained_checkpoint=/data/sonic_release/last.pt pretrained_revision=COMMIT
uv run --group training bello-train experiment=bello_mini_locomotion resume_path=checkpoints/RUN
uv run --group training bello-train experiment=bello_mini_supermimic_smoke \
  motion_dataset=/data/mini evaluation_clips='[held_out_walk]' wandb_enabled=false
uv run bello-eval task=locomotion checkpoint_path=checkpoints/RUN gamepad_enabled=false steps=1500
uv run bello-eval task=motion checkpoint_path=checkpoints/RUN output_path=outputs/motion.mp4
uv run bello-eval task=sonic_transfer checkpoint_path=checkpoints/RUN \
  motion_dataset=/data/mini-sonic evaluation_clips='[held_out_walk]' \
  output_path=outputs/sonic-evaluation
uv run bello-export-policy checkpoint_path=checkpoints/RUN output_path=outputs/policy.zip
uv run bello-export-policy checkpoint_path=checkpoints/SONIC \
  motion_dataset=/data/mini-sonic reference_clips='[walk]' \
  output_path=outputs/sonic-policy.zip
```

Select recipes from `configs/profiles/train/`. Edit those recipes for nested
physics/reward changes; the CLI exposes only its declared overrides. Evaluation
and checkpoint resume require the checkpoint's exact clean source commit.
Evaluation rasterization is Mesa llvmpipe software-only on Linux/Lima. Enabled training
video failures are fatal; use `video_enabled=false` to disable videos explicitly.
SONIC evaluation accepts only clip names recorded in the checkpoint's held-out
split. SONIC export bundles only `reference_clips`; see
[SONIC transfer](sonic_transfer.md).

SuperMimic selects `experiment=bello_mini_supermimic` or its small
`bello_mini_supermimic_smoke` profile. It accepts `motion_dataset`, optional
immutable `motion_dataset_artifact`, and `evaluation_clips`, with no pretrained
inputs. `motion_path`, payload overrides, pretrained fields, and `video_enabled`
are rejected. Training video is a separate evaluation command. Use
`task=supermimic` with an explicit output directory and held-out clip names for
evaluation; export requires `motion_dataset` and `reference_clips`. See the
[complete commands and input layout](supermimic.md).

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

Start hardwared and the [policy console](policy_console.md) separately; see [Lima](macos_vcan.md).
The emulator and policy runner are native commands, not Hydra/Python entrypoints.
See [emulator configuration](emulator.md) and the
[policy runtime](../rust/bello-policy/README.md).
Policy control defaults to read-only preflight. `--arm` enables the staged
operator-confirmed workflow for authorized commissioning. Physical deployment
remains gated on trained-policy validation; see [hardware](hardware.md).

Live upper-body following requires the updated tracking bridge with typed raw
body and hand poses. Neither consumer needs a neutral SMPL-X model.
Start `pico-smplx-bridge` on the tracking host, then run:

```bash
uv run bello-run-reference asset=bello_mini \
  xrobot.endpoint=127.0.0.1:50052
uv run bello-run-hands \
  xrobot.endpoint=127.0.0.1:50052 \
  duration_seconds=10
```

The reference service emits both the held-joint subset for upper-body following
and named full-body joints plus world-frame root orientation for buffered SONIC
Tracking, from the same retargeted sample. See the runner's
[Tracking configuration](policy_console.md#buffered-tracking-hil).
The reference service listens on `127.0.0.1:50056` by default; `listen` changes
that address. The hand command defaults to unarmed preflight and opens no CAN
connection. Armed independent hand control uses classic CAN directly. Neither
command is required for a fixed-arm locomotion policy. See
[live references and hands](hardware.md#live-references-and-hands) for the pose
contract, validity rules, and freshness settings.

The hand command selects its buses with `left_can_interface` (default `can2`)
and `right_can_interface` (default `can3`). Set both to the same interface for
shared-bus wiring. These fields replace the single `can_interface` setting;
the transport probes both enabled hands before accepting position writes.

With `arm=true calibration_confirmed=true`, the hand control gRPC service listens
at `listen=127.0.0.1:50057`. Both hands start in `HOLD` (no position writes).
Each hand needs an exclusive lease and an explicit source selection. Clients
renew within `lease_timeout_seconds` (default 1 second), including when selecting
tracking. Applications can request calibrated open/close or six normalized
positions, and lower the configured software slew limit per command. See the
[hand control API](hand_control.md) for the contract and client example.

## Remote jobs

See [Vast.ai](vast_ai.md). Remote completion/failure and the host deadline own
rental teardown. Closing a local monitor does not destroy a running job.
`num_envs` optionally overrides the selected profile on the remote host.
The public `offer` tree exposes `gpu_name`, `min_gpu_ram_gb`, `min_cpu_ram_gb`,
`disk_gb` and the existing marketplace filters.
Training supports exactly one GPU per instance. Use `dry_run=true` to inspect
selection without renting.
SuperMimic evaluation accepts `evaluation_split=held_out` (default) or
`evaluation_split=training`, with manifest membership checked in both cases.
Training-motion reports are explicitly labelled as reconstruction rather than
held-out generalization. SuperMimic remote jobs accept
`evaluate_training_split=true` to additionally evaluate all training motions.

SuperMimic remote jobs accept `evaluation_workers` for isolated native CPU
evaluation processes and `evaluation_video_clips` as a unique subset of the full
held-out `evaluation_clips`. The default `null` renders all selected clips;
`[]` scores every clip without videos. Other task routes reject these settings.
