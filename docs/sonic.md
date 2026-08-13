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
| Decoder root-path projection | Zero, preserving the release initialization | Trainable |
| Decoder proprioceptive projection and first bias | Default-anchored dense G1-from-Bello ridge transport | Trainable |
| Five interior decoder linears | Exact release arrays | Configurable frozen core; the final one is trainable in the maintained profile |
| Bello 25-joint output projection and bias | Default-anchored dense Bello-from-G1 ridge transport across all release rows | Trainable |
| Gaussian log standard deviation | Release variance propagated through the dense output map, with a `0.05` floor | Trainable |
| Privileged Bello critic | New initialization | Trainable at its own learning rate |

Frozen parameters are excluded structurally from the optimizer and checkpoint
state, not merely assigned a zero gradient. Their content hashes are checked
before every checkpoint and again after reconstruction for evaluation. The
explicit `sonic_trainable_middle_prefix_layers` and
`sonic_trainable_middle_suffix_layers` policy fields control interior decoder
adaptation. Their ranges cannot overlap. The maintained profile uses prefix
`0`, suffix `1`; the encoder and token projection always remain frozen.

Independent GMR solves do not produce a reliable joint-by-joint affine map:
different morphology and redundant IK can couple several coordinates even when
joint names match. The builder therefore fits two standardized multivariate
ridge maps on accepted training frames only: `29 x 25` for G1-from-Bello input
state and `25 x 29` for Bello-from-G1 output action.

The ridge penalty is anchored to a sparse anatomy prior rather than to an
all-zero matrix. Matching joint axes start with their one-joint regression
slope; Bello elbow pitch maps to G1 elbow, Bello elbow yaw maps to G1 wrist
roll, and Bello wrist pitch maps to G1 wrist pitch. G1 waist roll/pitch and
wrist yaw have no invented Bello counterpart. The dense fit may learn
cross-joint corrections, but the `1.0` penalty pulls spurious corpus-wide
correlations back toward this bootstrap map. In plain matrix notation, the
standardized coefficient solve is:

```text
(source correlation + 1.0 * identity) * fitted map
    = source/target cross-correlation + 1.0 * anatomy prior
```

Minimum motion variance, mean and per-joint explained variance, coefficient
size, and per-joint RMS residual gates reject degenerate fits. The same fitted
maps must pass those gates on the held-out split. Sufficient covariance
moments, not motion arrays, are retained during this first pass.

Both maps act on positions relative to each robot's configured default.
Velocities use the same raw-radian linear blocks without an intercept. Actions
are normalized differently on the two robots, so action-history and decoder
output transport also includes each robot's per-joint action scale. The G1
vector is pinned from NVIDIA's `0.25 * effort_limit / stiffness` definition;
the Bello vector is derived from the selected actuator configuration by the
same rule. Thus zero Bello action maps to zero G1 action and back without
confusing normalized actions with radians. This default anchoring avoids
injecting corpus-pose bias at reset while allowing every Bello joint, including
both elbow-yaw joints, to enter the trainable edge. Released diagonal action
variance is propagated through the complete scale-aware output matrix and then
clamped by the PPO standard-deviation limits. Requested Bello targets are
clipped per joint to the generated model's feasible target interval, while PPO
logs aggregate and per-joint pre-clipping violation fractions. Vector inputs
use the explicit Bello `+Y`-forward to G1 `+X`-forward basis change.

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
ordered layouts, clip content, split descriptors, and split-file hashes.

Before a clip can enter either split, SONIC-specific sanitation labels its root
motion and contact behavior. Name-pattern treadmill clips, stationary-root
gait cycles, and excessive contact-foot slip are quarantined. Calibration
checks mean and per-joint explained variance plus per-joint RMS residuals, then
applies the fitted training maps to the held-out split and records those
metrics in the dataset index. A build fails rather than publishing a dataset
whose morphology map does not generalize.

The builder is deliberately bounded-memory. Its first pass validates one
synchronized clip at a time and accumulates only calibration moments. Its
second pass reloads accepted clips into temporary memory-mapped arrays and
writes one compressed file per split. This keeps the build bounded in process
RAM. Training follows the TWIST implementation: the complete training bank is
loaded and transferred once, allowing the sampler to see the whole corpus
without host-side rotation.
The loader reads the NPZ central directory before allocation and requires the
uncompressed bank size plus 25 percent working headroom and a 2 GiB reserve.
It fails with a memory-size diagnostic instead of allowing the operating system
to kill the process. The maintained Vast offer's 64 GiB host-memory floor
covers this requirement; metadata and split hashes can be audited without
loading either bank.

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
  train.npz
  evaluation.npz
```

The destination must be empty. Quality and semantic sanitation failures are
reported rather than silently admitted, and both splits must contain accepted
clips. Each split file is pickle-free, float32, contiguous by clip, and
hash-checked before it is loaded.
Before packing, the builder also requires every sidecar to match the canonical
Bello URDF, the pinned GMR commit, the exact Bello and G1 IK config files, and
the G1 retarget XML in the current checkout. Stale outputs must be regenerated;
they are never silently mixed into a new dataset.
`heldout_manifest.json` preserves the manifest-assigned evaluation boundary,
including entries later rejected by quality control. Licensed AMASS and
body-model content stays outside Git.

## Train and resume

Run the maintained transfer profile from a clean commit on an `origin/*`
tracking branch:

```bash
uv run bello-train \
  experiment=sonic \
  dataset_path=/data/sonic/packed/index.json \
  sonic_artifact_path=artifacts/sonic/default_release \
  checkpoint_name=sonic_seed1
```

The environment runs at 200 Hz with 50 Hz control and ten-second episodes. It
retains the released action clip `20` but uses the per-joint G1 and Bello scales
described above. The actor receives 1,670 raw values. Its first 840 values are
ten frame-major encoder records containing 24 root-local SMPL joints, root
orientation relative to the current robot, and six G1 wrist joints. The next
20 values are the desired ten-frame XY root path in the current robot frame.
Its projection begins at zero and learns how Bello should realize traveling
motion that the translation-invariant SMPL encoder cannot distinguish from
treadmill motion. The remaining 810 values are term-major ten-frame histories
of gravity, base angular velocity, Bello joint position, Bello joint velocity,
and previous action. Observation corruption is applied once when a history
frame is recorded; the critic retains a separate clean history.

The new critic receives 1,445 raw privileged values: 250 future joint
positions, 250 future joint velocities, a 3-value anchor delta, a 6D anchor
orientation, 42 body-position values, 84 body-orientation values, and 810 clean
history values. The frozen actor stays raw; the new critic uses its own running
mean/std, matching the released critic configuration.

Training transfers the complete training bank to the accelerator once. Starts
use the manifest frame-count-times-weight prior mixed with the released
failure/uniform curriculum and pre-failure offset. Failure/attempt bins cover
the whole corpus and remain in the runner continuation, so PPO updates do not
pause for host data loads and vectorized worlds are not reset at data-boundary
transitions. Sampling uses one uniform draw against the global cumulative
distribution rather than materializing per-environment categorical noise over
every bin.

The maintained PPO values are 4,096 worlds, 24 rollout steps, five epochs, four
minibatches, actor-edge learning rate `2e-5`, critic learning rate `1e-3`,
mapped checkpoint exploration standard deviations (clamped to `[0.001, 0.5]`),
desired KL `0.01`, entropy `0.01`, gamma `0.99`, lambda `0.95`, clip ratio
`0.2`, and maximum gradient norm `0.1`.

Evaluation reports the original clip-weighted acceptance metrics as well as
frame-weighted tracking metrics and strata for stationary/traveling root motion
and gait/non-gait contact behavior.

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
  total_timesteps=71270400 \
  num_envs=4096 \
  checkpoint_every_updates=145 \
  branch=feature/sonic-transfer \
  max_price=1.25 \
  duration_hours=6 \
  skip_confirmation=true
```

This budget is 725 PPO updates at the maintained 4,096-world, 24-step batch.
Before the real job, the launcher runs three updates at the same environment
count and records GPU utilization, peak memory, power, elapsed time, and exit
status. A failed production-scale benchmark prevents the longer run. The
launcher verifies every dataset and release-artifact file before training. W&B
retains checkpoints; `benchmark.json`, the full evaluation, compact summary,
and checksum manifest are retrieved under `outputs/vast/<run-name>/` before
teardown.

For a full from-scratch run, omit `resume_path`. The maintained full budget is
30,001 updates or 2,949,218,304 transitions:

```bash
uv run bello-vast-train \
  experiment=sonic \
  dataset_path=/data/sonic/packed/index.json \
  sonic_artifact_path=artifacts/sonic/default_release \
  total_timesteps=2949218304 \
  num_envs=4096 \
  checkpoint_every_updates=500 \
  branch=feature/sonic-transfer \
  max_price=1.25 \
  duration_hours=36 \
  skip_confirmation=true
```

`total_timesteps` is the final absolute budget, not an additional budget. To
continue an exact 4,096-environment run, add
`resume_path=/path/to/checkpoint`; use `allow_git_commit_mismatch=true` only for
an intentionally reviewed cross-commit resume. The GPU preflight always starts
fresh. The real run restores controller and optimizer state only after the
dataset, frozen release, and resume checkpoint have each passed local and
remote hash verification.

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
never carries state across a terminal reset. The 500-step rollout is compiled
once against the complete evaluation-bank contract; clip selection remains a
dynamic input, and terminal clips stop inside the compiled loop. Supply
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
