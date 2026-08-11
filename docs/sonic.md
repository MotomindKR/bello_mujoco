# SONIC transfer learning

This port adapts NVIDIA's released [GEAR-SONIC](https://nvlabs.github.io/GEAR-SONIC/)
low-level G1 controller to Bello while deliberately preserving as much of the
learned motion model as the morphology change allows. It is a dedicated task,
not a mode of the legacy motion, TWIST, or TWIST2 environment.

## Transfer boundary

The release controls 29 Unitree G1 joints; Bello exposes 25 policy joints and
uses a different simulator, body topology, actuator model, coordinate basis,
and training stack. The port therefore freezes the semantic core and replaces
only the morphology-dependent edges:

| Block | Initialization | Training status |
| --- | --- | --- |
| Five-layer SMPL encoder, `840 -> 64` | Exact release arrays | Frozen |
| Two 32-value, 32-level FSQ tokens | Exact release operation | Frozen |
| Decoder token projection | Exact release columns | Frozen |
| Decoder proprioceptive projection and first bias | Default-anchored dense G1-from-Bello ridge transport | Trainable |
| Five interior decoder linears | Exact release arrays | Frozen |
| Bello 25-joint output projection and bias | Default-anchored dense Bello-from-G1 ridge transport across all release rows | Trainable |
| Gaussian log standard deviation | Release variance propagated through the dense output map, with a `0.05` floor | Trainable |
| Privileged Bello critic | New initialization | Trainable at its own learning rate |

Frozen parameters are excluded structurally from the optimizer and checkpoint
state, not merely assigned a zero gradient. Their content hashes are checked
before every checkpoint and again after reconstruction for evaluation. There
is no configuration switch that silently unfreezes interior layers.

Independent GMR solves do not produce a reliable joint-by-joint affine map:
different morphology and redundant IK can couple several coordinates even when
joint names match. The builder therefore fits two standardized multivariate
ridge maps on accepted training frames only: `29 x 25` for G1-from-Bello input
state and `25 x 29` for Bello-from-G1 output action. The default ridge strength
is `0.1`; minimum motion variance, mean explained variance, coefficient size,
and RMS residual gates reject degenerate fits. Sufficient covariance moments,
not motion arrays, are retained during this first pass.

Both maps act on positions and actions relative to each robot's configured
default, so zero Bello action maps to zero G1 action and back. Velocities use
the same linear blocks without an intercept. This default anchoring avoids
injecting corpus-pose bias at reset while allowing every Bello joint, including
both elbow-yaw joints, to enter the trainable edge. Released diagonal action
variance is propagated through the complete dense output matrix and then
clamped by the PPO standard-deviation limits. Vector inputs use the explicit
Bello `+Y`-forward to G1 `+X`-forward basis change.

## Why the motion data needs three views

A Bello AMASS retarget is necessary but not sufficient. Every accepted source
frame must retain three timestamp-identical representations:

1. Bello `qpos`, `qvel`, tracked-body poses/velocities, and foot contacts for
   MJX tracking and rewards.
2. The 24-joint SMPL skeleton canonicalized into SONIC's G1-root-local,
   Z-up convention, plus its processed root quaternion and translation, for
   the frozen encoder's online ten-frame command.
3. The 29-joint G1 retarget for the six released wrist inputs and the
   bidirectional morphology-edge calibration.

Using one manifest for both retargeters is what makes the views joinable. The
builder rejects any disagreement in source digest or exact source timestamps;
it does not interpolate one retarget to make a mismatch appear valid. The
existing Bello GMR output can therefore be reused directly, while the SONIC
pipeline adds only the paired G1/SMPL sidecar.

The packed dataset is 50 Hz. The actor uses offsets `[0..9]`; the privileged
critic uses `[0,5,...,45]`, so every sampled start preserves the full future
window. Train/evaluation assignment comes from the licensed manifest. Dense
edge calibration is fitted only on accepted training frames, and duration times
manifest weight determines initial training clip probability. The index binds
the portable manifest, Bello robot, both GMR revisions/configs, calibration,
ordered layouts, clip content, shard capacities, descriptors, and shard hashes.

The builder is deliberately bounded-memory. Its first pass validates one
synchronized clip at a time and accumulates only calibration moments. Its
second pass reloads accepted clips into fixed-capacity shards. The defaults are
65,536 frames and 256 clips per shard; both values are part of the dataset and
checkpoint contracts. Padding gives every shard the same JAX shape without
putting the entire AMASS corpus on the host or accelerator.

## Export the release weights

The training runtime never unpickles the upstream PyTorch checkpoint. Export
it once in a trusted Torch environment after obtaining the official SONIC
`last.pt` and its configuration:

```bash
python tools/export_sonic_release.py \
  --checkpoint /path/to/last.pt \
  --config /path/to/config.yaml \
  --output-dir artifacts/sonic/default_release
```

The exporter refuses to load a checkpoint unless it matches the pinned
official SHA-256
`e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909`.
The release configuration is independently pinned by SHA-256
`f08187795fa16a839a28bc1c18e0555d38d9420e03733744341cdcb56ab629c7`.
It writes only plain `weights.npz`, `golden.npz`, and `manifest.json`; the
loader verifies file, array, architecture, source, golden-vector, and exact
deterministic export hashes.
`artifacts/` is ignored and must not be committed.

## Build the synchronized dataset

Initialize the pinned GMR fork and its environment first. Then generate one
sidecar for every source in the same manifest used by the Bello retarget:

```bash
uv run bello-sonic-data \
  operation=retarget_g1 \
  manifest=/data/amass/manifest.json \
  body_model_root=/data/body_models \
  normalized_root=/data/sonic/normalized \
  output_root=/data/sonic/g1_smpl \
  gmr_python=/path/to/gmr/python \
  jobs=1
```

Each worker loads GMR and SMPL-X data. Keep `jobs=1` when another retargeter is
running or host RAM is constrained. Completed outputs are provenance-checked
and written atomically; reruns skip exact matching outputs. Use `limit=10` for
a deterministic smoke conversion and `keep_normalized=true` only when the
intermediate archives are useful for debugging. `normalized_root` and
`output_root` must be different directories.

Join those sidecars to the already-retargeted Bello clips:

```bash
uv run bello-sonic-data \
  operation=build \
  manifest=/data/amass/manifest.json \
  bello_retargeted_root=/data/amass/bello_retargeted \
  g1_retargeted_root=/data/sonic/g1_smpl \
  output_root=/data/sonic/packed
```

Build output has this layout:

```text
packed/
  index.json
  heldout_manifest.json
  conversion_report.json
  shards/train/00000.npz
  shards/train/...
  shards/evaluation/00000.npz
  shards/evaluation/...
```

The destination must be empty. Quality failures are reported rather than
silently admitted, and both splits must contain accepted clips. Every shard is
pickle-free, float32, fixed-shape, and hash-checked before it is loaded.
Before packing, the builder also requires every sidecar to match the canonical
Bello URDF, the pinned GMR commit, the exact Bello and G1 IK config files, and
the G1 retarget XML in the current checkout. Stale outputs must be regenerated;
they are never silently mixed into a new dataset.
`heldout_manifest.json` preserves the manifest-assigned evaluation boundary,
including entries later rejected by quality control. Licensed AMASS and
body-model content stays outside Git.

## Train and resume

Run the maintained edge-only profile from a clean commit on an `origin/*`
tracking branch:

```bash
uv run bello-train \
  experiment=sonic \
  dataset_path=/data/sonic/packed/index.json \
  sonic_artifact_path=artifacts/sonic/default_release \
  checkpoint_name=sonic_seed1
```

The environment runs at 200 Hz with 50 Hz control, ten-second episodes,
released action clip `20` and action scale `0.5`. The actor receives 1,650 raw
values. Its first 840 values are ten frame-major encoder records containing 24
root-local SMPL joints, root orientation relative to the current robot, and six
G1 wrist joints. The remaining 810 values are term-major ten-frame histories
of gravity, base angular velocity, Bello joint position, Bello joint velocity,
and previous action. Observation corruption is applied once when a history
frame is recorded; the critic retains a separate clean history.

The new critic receives 1,445 raw privileged values: 250 future joint
positions, 250 future joint velocities, a 3-value anchor delta, a 6D anchor
orientation, 42 body-position values, 84 body-orientation values, and 810 clean
history values. The frozen actor stays raw; the new critic uses its own running
mean/std, matching the released critic configuration.

Training loads one fixed-capacity reference shard at a time. A deterministic
sampling-mass-weighted schedule holds a full shard for 32 updates by default,
then resets the vectorized worlds at the update boundary before binding the
next shard. Per-shard failure/attempt bins are cached on the host while the run
is alive; the active shard state is part of checkpoint continuation. Within a
shard, starts use the manifest frame-count-times-weight prior mixed with the
released failure/uniform curriculum and pre-failure offset.

The maintained PPO values are 4,096 worlds, 24 rollout steps, five epochs, four
minibatches, actor-edge learning rate `2e-5`, critic learning rate `1e-3`,
mapped checkpoint exploration standard deviations (clamped to `[0.001, 0.5]`),
desired KL `0.01`, entropy `0.01`, gamma `0.99`, lambda `0.95`, clip ratio
`0.2`, and maximum gradient norm `0.1`.

Resume by adding `resume_path=/path/to/checkpoint`. The typed checkpoint
requires the same dataset, calibration, release artifact, robot/layout,
environment contract, and normally the exact source commit. Because frozen
release arrays are intentionally absent from checkpoint state, the release
artifact is also required for resume and evaluation.

## Simulator and morphology adaptation

The release reward weights and Gaussian widths are retained: anchor pose,
heading-aligned body pose, body velocity, action rate, joint limits, undesired
contacts, wrist/head angular shake, the release-named local `5point` term, and
ankle-joint acceleration. Despite its upstream name, the pinned release
configures only a torso-plus-head-offset proxy and two wrists. Bello uses its
actual head body plus its two wrists. Feet remain separate termination and
acceleration terms.

Termination retains the release overrides: `0.15 m` adaptive root-height and
end-effector-height errors, `0.2` squared root-orientation error, and `0.2 m`
foot position error. Root-height adaptation uses the same `0.1` exponential
moving average. Reset perturbations, ten-frame sensor noise, joint-default
offsets, 4–6 second velocity pushes, friction, mass/COM variation, and actuator
strength are expressed through Bello's MuJoCo model randomizer. These are
semantic simulator adaptations rather than bitwise Isaac Lab event parity:
MuJoCo has one sliding-friction coefficient here, and Bello's mass topology
does not reproduce G1 wrist/torso bodies one-for-one.

## Evaluate the transfer

For a branch-pinned Vast.ai smoke run, short edge baseline, and complete
held-out evaluation in one billed lifecycle, use:

```bash
uv run bello-vast-train \
  experiment=sonic \
  dataset_path=/data/sonic/packed/index.json \
  sonic_artifact_path=artifacts/sonic/default_release \
  total_timesteps=1425408 \
  num_envs=512 \
  reference_shard_updates=1 \
  checkpoint_every_updates=29 \
  branch=feature/sonic-transfer \
  max_price=1.25 \
  duration_hours=6 \
  skip_confirmation=true
```

This budget is one update per training shard for the current 116-shard
production corpus. The launcher verifies every nested dataset and release
artifact file before training. W&B retains checkpoints; the full evaluation,
compact summary, and checksum manifest are retrieved under
`outputs/vast/<run-name>/` before the instance is destroyed.

The same evaluation can be run locally with:

```bash
uv run bello-eval \
  task=sonic \
  checkpoint_path=checkpoints/sonic_seed1/final \
  sonic_artifact_path=artifacts/sonic/default_release \
  output_path=outputs/evaluation/sonic.json
```

Evaluation disables training noise, randomization, adaptive sampling, and
stochastic action sampling. It visits every held-out clip in stable order and
never carries state across a terminal reset. Supply
`video_path=outputs/evaluation/sonic.mp4` for one representative video; `steps`
bounds only that video.

The report applies these initial transfer gates:

| Metric | Acceptance gate |
| --- | ---: |
| Held-out completion rate | at least `0.90` |
| Root-position RMS | at most `0.20 m` |
| Root-orientation RMS | at most `15 deg` |
| Joint-position RMS | at most `0.30 rad` |
| Mean foot slip | at most `0.20 m/s` |

These are validation gates, not a claim of hardware readiness. A policy that
passes still needs actuator-limit review, latency testing, system
identification, and staged safety validation before physical deployment.
