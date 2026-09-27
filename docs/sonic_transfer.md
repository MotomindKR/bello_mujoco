# SONIC LoRA transfer

`bello_mini_sonic_transfer` adapts NVIDIA's released G1 SONIC policy to Bello
Mini. It is a separate task: locomotion and BeyondMimic observations, action
mapping, checkpoints, and runtimes are unchanged.

The implementation follows the pinned Any2Any source at
`deps/gear_sonic` commit `fd8624fe07e4cd85bb82f0a2186d81c230b3963c`.
That dependency is for inspection and numerical validation only; production
rollouts do not import PyTorch or Isaac Lab.

## Inputs

Download the original training checkpoint and matching config from an immutable
Hugging Face revision. The revision inspected when this integration was added
was `6733128a3d8a523b1418b06bca3cdf61c8b0987f` (2026-08-26):

```bash
uv run --group reference-validation hf download nvidia/GEAR-SONIC \
  sonic_release/last.pt sonic_release/config.yaml config.json \
  --revision 6733128a3d8a523b1418b06bca3cdf61c8b0987f \
  --local-dir checkpoints/gear-sonic-source
```

The motion dataset is a directory of already-retargeted Bello Mini `.npz`
clips in the repository's normal motion format. Upload that immutable directory
once as a W&B dataset artifact; do not add it to Git. Training records both the
artifact reference and source-checkpoint SHA-256.

```bash
uv run --group training --group reference-validation bello-train \
  experiment=bello_mini_sonic_transfer \
  motion_dataset=/data/bello-mini-sonic \
  motion_dataset_artifact=ENTITY/PROJECT/bello-mini-sonic:v0 \
  evaluation_clips='[held_out_walk,held_out_dance]' \
  pretrained_checkpoint=checkpoints/gear-sonic-source/sonic_release/last.pt \
  pretrained_revision=6733128a3d8a523b1418b06bca3cdf61c8b0987f
```

Missing checkpoint tensors, an unpinned revision, an absent artifact reference,
an empty corpus, a joint/body layout mismatch, or a non-invertible transmission
fails before rollout compilation. There is no scratch or full-finetuning mode.

Resume repeats those explicit corpus and source-checkpoint inputs and adds
`resume_path=/data/checkpoints/run/update_000100`. It rejects a different
artifact, source hash, clip order, environment, or optimization contract; only
the total training budget and operational checkpoint/W&B locations may change.

## Reference and Mini departures

The actor retains the release layout: ten 50 Hz proprio/action samples, ten
future reference frames spaced at 0.1 seconds, a frozen 640→64 SiLU encoder,
32-level FSQ arranged as two 32-dimensional tokens, and a frozen
994→29 dynamics decoder with rank-64/alpha-128 LoRA on every linear layer. The
pretrained critic uses rank-32/alpha-32 LoRA and a separate 1e-3 learning rate.
The actor uses the Any2Any 5e-5 adaptive rate and Gaussian exploration clamped
to `[0.001, 0.6]` standard deviation.

The release's tokenizer has a historical reshape quirk: all future joint
positions precede all future velocities before the combined buffer is reshaped
into 58-value rows and paired with per-frame root orientations. Training and
Rust playback preserve that ordering because the frozen encoder was trained on
it.

Bello Mini's absent waist-pitch value is filled with the G1 neutral input and
removed before the 28-dimensional Gaussian is sampled, so PPO likelihoods cover
only controllable channels. Elbow aliases are explicit. Other joints are mapped
by canonical name; no Oli hip correction or assumed wrist ordering is used.
The policy's logical target deltas use `0.25 * effort_limit / stiffness`, then
flow through Mini's canonical differential transmission, constraint projection,
and motor step limits. This is intentionally different from BeyondMimic's
bounded-range mapping.

Physics starts at 50 Hz control and 5 ms substeps with no action LPF. Mini's
motion gains, effort limits, feedback filtering, command delay, and modest
dynamics randomization remain the canonical profile values. The pelvis is the
SONIC anchor, matching the source root and Mini's retargeting contract. The source reward
composition includes local five-point tracking, wrist anti-shake, and ankle
acceleration terms with the pinned Any2Any weights.

## Evaluation and deployment

Evaluation requires an explicit subset of the held-out split recorded in the
checkpoint and writes one video and one result entry per clip:

```bash
uv run bello-eval task=sonic_transfer \
  checkpoint_path=/data/checkpoints/run/final \
  motion_dataset=/data/bello-mini-sonic \
  evaluation_clips='[held_out_walk,held_out_dance]' \
  output_path=outputs/sonic-evaluation
```

Export merges LoRA into the frozen decoder and emits one ONNX graph containing
the encoder, exact FSQ operation, decoder, 29-to-28 mapping, SONIC action scale,
and Mini transmission. Only explicitly selected clips are bundled:

```bash
uv run bello-export-policy \
  checkpoint_path=/data/checkpoints/run/final \
  motion_dataset=/data/bello-mini-sonic \
  reference_clips='[walk,dance]' \
  output_path=/data/bello-mini-sonic.zip
```

The Rust runner requires one bundled clip to be selected before it initializes
history. Offline checking and telemetry playback are available; physical arming
remains a later commissioning gate:

```bash
cargo run --locked -p bello-policy -- \
  --policy /data/bello-mini-sonic.zip --source . --clip walk --check
cargo run --locked -p bello-policy -- \
  --policy /data/bello-mini-sonic.zip --source . --clip walk --telemetry
cargo run --locked -p bello-policy -- \
  --policy /data/bello-mini-sonic.zip --source . --clip walk \
  --hardware-config rust/bello-policy/config/hardware.yaml
```

The hardware form is read-only: it validates the IDLE inventory, coordinate
coverage, gains, target limits, IMU frame and fresh telemetry, then assembles
one initialized SONIC observation. `--arm` is rejected for SONIC bundles.

## Validation boundary

The default tests cover clip-boundary clamping, split-aware sampling, the named
waist codec, differential transmissions, FSQ behavior, and zero-LoRA invariance.
PyTorch parity uses the optional `reference-validation` group and the pinned
submodule/checkpoint. The unit suite checks JAX/ONNX parity and the Rust crate is
compiled separately. GPU smoke, Lima HIL, trained-policy HIL, and physical
arming are separate gates. A smoke checkpoint is not a commissioned policy.
