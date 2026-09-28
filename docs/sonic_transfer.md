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
The directory must include `sonic-corpus.json`, which binds clip checksums,
original source-sequence identities and train/evaluation splits. Training
verifies the actual W&B artifact bytes, not only its name. Mutable artifact
aliases and external-reference artifacts are rejected. All evaluation clips
must be listed explicitly and must match the manifest's held-out split.

For an existing set of Mini clips, create a JSON mapping from each filename stem
to `{"split": "train", "source_sequence": "original-sequence-id"}` (or
`"evaluation"`), then prepare and upload:

```bash
uv run python -m bello_mujoco.sonic.corpus prepare /data/bello-mini-sonic \
  --splits /data/splits.json
uv run python -m bello_mujoco.sonic.corpus upload /data/bello-mini-sonic \
  --entity gyungmin --project bello-mujoco --name mini-sonic-corpus
```

Upload requires an existing private W&B project. Keep the printed immutable
`:vN` reference; never use `:latest` for training. Mirrored or segmented versions
of the same source must retain its source-sequence identity. The reference bank
is device-resident; the default uncompressed-array safety limit is 4 GiB and is
checked before environment construction. This limit is not a VRAM estimate.
Loading validates clips on the CPU and transfers concatenated arrays once.
The bank is an explicit JIT argument, not a multi-gigabyte embedded compiler
constant or checkpoint payload; resume obtains it from the recorded artifact.

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
The release includes trainer metadata from TRL, Transformers, and Accelerate.
The importer uses restricted tensor loading with inert replacements for those
known metadata types; those training packages are not imported or required.
Unknown pickle globals still fail explicitly.

Resume repeats those explicit corpus and source-checkpoint inputs and adds
`resume_path=/data/checkpoints/run/update_000100`. It rejects a different
artifact, source hash, clip order, environment, or optimization contract; only
the total training budget and operational checkpoint/W&B locations may change.

## Reference and Mini departures

The actor retains the release layout: ten 50 Hz proprio/action samples, ten
future reference frames spaced at 0.1 seconds, a frozen 640→64 SiLU encoder,
32-level FSQ arranged as two 32-dimensional tokens, and a frozen
994→29 dynamics decoder with rank-64/alpha-128 LoRA on every linear layer. The
pretrained critic uses rank-32/alpha-32 LoRA. Actor and critic use the effective
Any2Any 5e-5 adaptive rate: the reference scheduler overwrites both optimizer
groups, despite its YAML declaring a separate critic rate. The shared PPO
implementation supports separate rates for other explicit configurations.
Gaussian exploration optimizes standard deviation directly and clamps it to
`[0.001, 0.6]`.
The SONIC profile uses 2,048 environments, 24-step rollouts, five epochs and
four trajectory minibatches, sample-standard-deviation advantage normalization,
Adam epsilon `1e-8`, value-loss coefficient 1, and gradient clipping at 0.1.
Following Any2Any, SONIC rejects nonfinite gradients but does not reject finite
raw norms before clipping (`gradient_norm_max: null`). Other tasks retain their
existing raw-norm cutoff. Actor, critic and exploration norms are logged separately.
The analytic old-policy Gaussian KL adapts learning rate after each minibatch.
Actor observations remain unnormalized. The pretrained critic's running
normalizer uses sample variance, initial count 1, `sqrt(var + 1e-5)` and clipping
to +/-5; it updates after normalization during optimization, not during rollout.
These opt-in settings do not change locomotion or BeyondMimic defaults.
Backbone, adapter, and merged-weight matrix products request full float32
precision explicitly. GPU defaults such as TF32 can otherwise change frozen
encoder outputs and FSQ tokens; no process-level precision override is required.

The release's tokenizer has a historical reshape quirk: all future joint
positions precede all future velocities before the combined buffer is reshaped
into 58-value rows and paired with per-frame root orientations. Training and
Rust playback preserve that ordering because the frozen encoder was trained on
it.
Rust also uses the training clock's float32 frame arithmetic and ties-to-even
rounded future-frame offsets, including clips whose frame rate is not a
multiple of 10 Hz. This prevents interpolation drift on long or fast motions.
Training stores the corpus-global clock in frame units, avoiding float32
seconds accumulation drift late in an eight-hour bank. Clip rates must be a
power-of-two multiple of the 50 Hz control rate (for example 25, 50 or 100 Hz),
and the bank must fit the validated exact float32 clock range.

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

### Reward and training-curve alignment

SONIC integrates every weighted reward term over the control timestep, matching
Isaac Lab's `RewardManager`: at 50 Hz the multiplier is 0.02. This applies to
PPO rewards, episode returns and component metrics, not just the plotted values.
BeyondMimic retains its existing reward convention. Earlier SONIC runs before
this correction omitted that multiplier and are not recipe-parity results;
start again from the original NVIDIA checkpoint, not those trained adapters.

The SONIC profile explicitly sets 4-6 second pushes, action clipping at +/-20,
anchor height error 0.25 m, squared full-orientation error 1 rad^2 (including
yaw), and end-effector height error 0.3 m. Clip exhaustion is the timeout:
the reference config's nominal 10-second episode length does not itself add
a time-limit termination term. PPO uses the reference TRL timeout convention,
adding `gamma * V(s_t)` before done-masked GAE. Other profiles keep their
next-state timeout bootstrap. Ankle acceleration uses the velocity difference
over the final physics substep, in both MJX and native CPU evaluation, matching
Isaac Lab's physics-frequency finite differences. It does not use MuJoCo's
solver `qacc` or a difference over the whole control period.
The contact exemptions include ankles, wrists and elbows by canonical role.

W&B SONIC charts use `update` as their default horizontal axis. Compare
`train/objective/rewards` and `train/objective/length` with the README: these
are means over the last 100 completed episodes, with the buffer included in
checkpoints. `train/mean_reward` is still explicitly a per-control-step mean;
`train/mean_episode_return` averages episodes completed in the current rollout.

### Adaptive motion sampling

SONIC uses the pinned Any2Any temporal-bin curriculum, configured by
`sampling_config` in the SONIC profile: 50-frame bins, a 200-frame backward
start window, 10% uniform-bin mixing, a failure-rate clipping threshold of
200 times the active-bin mean, and unit pseudocounts. Each control step adds
`1 / bin_length` exposure to its occupied bin; a terminal failure adds one
failure. Rates are failures divided by exposure, not raw failure counts.
The source's sequence-length-agnostic weighting is `bin_length / bins_in_clip`;
short final bins therefore matter (this is not exactly uniform over clips).
The sampling distribution refreshes once per PPO iteration. Starts are integer
frames in the selected bin, shifted backward by a random 0-199 frames and
clamped at that clip's start. Held-out bins have zero probability.

Counters and the frozen-within-rollout distribution live once in the checkpointed
task state, alongside the completed-episode buffer; they are not replicated
per environment. The shared BeyondMimic EMA sampler is unchanged. SONIC requires
the inherited `env_config.adaptive_sampling` to remain at its inactive defaults;
customization there fails explicitly. Configure `sampling_config` instead.
Old raw-per-clip-EMA SONIC checkpoints cannot resume this recipe and should not
be treated as reference-parity runs. Start fresh from original pretrained weights.

Track `train/sampling/top_clip_probability`, `top_bin_probability`, and
`failure_rate_mean`/`failure_rate_max`, in addition to normalized bin entropy.
Training episode length is curriculum-dependent and can decrease when practice
shifts to difficult segments. It is not a uniform evaluation or sufficient proof
of forgetting. The reference trainer populates `objective/length` from its last
100 completed training episodes; screenshot provenance is not independently verified.

This is recipe alignment, not identical experiments: Mini dynamics, transmission,
hardware filtering/delay, modest dynamics randomization, float32 inference,
and the 4,166-clip AMASS corpus remain deliberate
departures. The README uses a small selected Oli corpus. MuJoCo contact counts
use the final-step contact forces, whereas Isaac Lab uses contact-sensor history.
Identical curves are not promised. Numerical preflight is not evidence of
convergence; inspect fixed held-out motion tracking before extending a rental.

## AMASS research corpus

The current corpus target is **8 hours of accepted training motion plus 0.8
hours of held-out motion**, at 50 Hz, using AMASS only. BONES-SEED is excluded.
This is approximately the 1.5-million-frame scale explored in the
[Any2Any data ablation](https://arxiv.org/html/2605.23733v1), not a claim that
eight hours or a particular individual clip length was optimal for SONIC/Mini.
The reported ablation's best result used more data and a different transfer pair.

AMASS is not commercially unrestricted. Its
[standard license](https://amass.is.tue.mpg.de/license.html) limits use to
non-commercial purposes and restricts redistribution and commercial training.
This preparation was authorized for non-commercial research. SMPL-X and source
collection terms still apply. Do not publish the corpus or assume that a private
W&B project grants additional rights.

The offline preparer consumes the NAS's `source_manifest.yaml`, raw AMASS
SMPL-X `.npz` files, and `body_models/smplx/SMPLX_{MALE,FEMALE,NEUTRAL}.npz`.
It does not consume the old 25-joint Bello retargets or change GMR. Derived data
must be outside the input root and Git:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu \
uv run python -m bello_mujoco.sonic.amass \
  --root /Volumes/home/datasets/amass \
  --manifest /Volumes/home/datasets/amass/source_manifest.yaml \
  --output /Volumes/home/datasets/derived/bello_mini/sonic-amass-8h-20260928 \
  --workers 4
```

Use a fresh output directory for each recipe. `--max-clips 24` creates a small
pilot rather than the full duration budget. An interrupted full preparation can
resume with the same command plus `--resume`; source, body-model, dependency,
Python-source and lockfile fingerprints must match. Completed corpora are
immutable. Progress is durable in `progress.jsonl`; `preparation.json` summarizes
accepted/rejected clips and actual durations. An interrupted, unjournaled clip
is preserved separately before recomputation. A missing source, changed checksum
or insufficient accepted duration fails explicitly, rather than filling the
budget by repeating clips or loosening quality thresholds.

The NAS source directory stays read-only, but the separate `datasets/derived/`
directory holds generated clips, progress and provenance. SMB publication uses
a validated temporary file and exclusive destination reservation followed by
rename; it does not require hard-link support. If migrating an existing run,
stop its workers, verify the copied files before removing local duplicates and
retain the old/new preparation identities in `storage-migration.json`. A storage
migration must not silently waive changes to retargeting or quality settings.
Verified AppleDouble sidecars created by macOS on SMB are removed for newly
published clips and the final corpus manifest; they are filesystem metadata,
not additional motion files. Source data and unknown sidecars are not deleted.

Selection is seeded and round-robin across subdatasets. It deduplicates source
motion fingerprints and holds out subject groups, including known capture-session
aliases and overlapping MPI subject IDs. Names cannot prove identity across all
collections; this is stronger than a random frame split, not a guarantee against
every unknown cross-collection duplicate.
The first recipe accepts complete 2-30 second sequences and excludes GRAB,
PosePrior and obvious terrain/prop-dependent or extreme-motion filenames.
Those duration and filename filters are conservative curation heuristics, not
published optimal hyperparameters or a complete semantic classifier.

Rotations are resampled with shortest-arc SLERP. Shape-dependent SMPL-X joint FK
feeds the existing canonical Mini GMR mapping. After retargeting, a single
clip-wide vertical translation aligns the fifth percentile of the lowest sole
height to the floor; there is no per-frame root snapping. Checks reject canonical
motor/joint-limit violations, speeds over 20 rad/s, penetration over 3 cm,
ground correction over 15 cm, and remaining floor clearance over 10 cm.
Foot labels are geometric proximity within 3 cm, not measured contact forces.
Body poses and velocities are recomputed with Mini FK. These are kinematic
quality checks, not proof that a motion is dynamically trackable.

The finalized `clips/sonic-corpus.json` embeds source provenance and split
metadata, including source hashes, preparation settings and dependency pins.
Review rendered samples and dataset statistics before uploading `clips/` to the
same private `gyungmin/bello-mujoco` project and running full-bank GPU checks.
The preparer does not upload or start training automatically.

## Evaluation and deployment

Evaluation requires an explicit subset of the held-out split recorded in the
checkpoint and writes one video and one result entry per clip.
It also writes `summary.json`, with equal weight per selected clip, deterministic
mean actions, first-frame starts, mean completion fraction, failure fraction and
tracking errors. Compare the same explicit held-out list and step limit across
checkpoints. This fixed evaluation is separate from adaptive training metrics;
it neither updates sampling counts nor consumes training samples.
Nonfinite motion-evaluation state raises an error and closes video resources;
it is not reported as a successful short rollout, even when terminal stopping
is disabled.

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
See the [RTX 5090 validation record](sonic_validation.md) for measured results,
the tested source snapshot, and remaining gates.

Checkpoint restoration is checked bitwise, including optimizer, random keys,
history, and clip-sampling state. Subsequent MJX-Warp rollouts are not expected
to be bitwise deterministic: GPU reductions and contact ordering can vary even
when running twice from the same uninterrupted state. Compare restored actor
outputs before advancing physics, and report continuation drift separately.
