# SONIC transfer learning

This port adapts NVIDIA's released [GEAR-SONIC](https://nvlabs.github.io/GEAR-SONIC/)
low-level G1 controller to Bello while deliberately preserving as much of the
learned motion model as the morphology change allows. It is a dedicated task,
not a mode of the legacy motion, TWIST, or TWIST2 environment.

## Transfer boundary

The release controls 29 Unitree G1 joints; Bello exposes 25 policy joints and
uses a different simulator, body topology, actuator model, coordinate basis,
and training stack. The port follows the
[Any2Any](https://arxiv.org/abs/2605.23733) boundary: first align Bello into
the source robot's semantic spaces, then add low-rank residuals only to
dynamics-sensitive modules.

| Block | Initialization | Training status |
| --- | --- | --- |
| Five-layer G1 robot motion encoder, `640 -> 64` | Exact release arrays | Frozen |
| Two 32-value, 32-level FSQ tokens | Exact release operation | Frozen |
| Bello-to-G1 proprioception alignment | Signed semantic scatter plus paired inclined-hip decoupling | Fixed |
| Seven-layer G1 dynamics decoder, `994 -> 29` | Exact release arrays | Frozen base with rank-32 LoRA on proprioception, backbone, and output paths |
| G1-to-Bello action alignment | Explicit inverse semantic transform with action-scale conversion | Fixed |
| Gaussian standard deviation | Exact released 29-value source vector, clamped to `[0.001, 0.5]` | Frozen |
| Privileged G1 critic and observation normalizer | Exact release arrays/statistics | Frozen base with rank-32 LoRA on every critic linear |

Frozen parameters are excluded structurally from the optimizer, not merely
assigned zero gradients. Their content hashes are checked before every
checkpoint and after evaluation reconstruction. `sonic_lora_rank` and
`sonic_lora_alpha` configure the residuals. The paper does not publish an exact
LoRA rank for the external SONIC transfer; `32` for both is the maintained
Bello hypothesis and gives unit LoRA scaling.

The joint interface is a `29 x 25` block-sparse matrix. Same-name scalar axes
scatter directly with the axis sign measured from paired GMR retargets. Each
hip is transported through a common pitch/roll/yaw basis: Bello's mirrored
30-degree pitch/yaw block is composed with G1's `0.1749`-radian roll/yaw
block. These constants come from the pinned robot descriptions, not motion
regression. Bello elbow pitch maps to G1 elbow, Bello elbow yaw maps to G1
wrist roll, and Bello wrist pitch maps to G1 wrist pitch. G1 waist roll/pitch
and wrist yaw are zero-padded. The reverse action map is constructed as the
paired inverse, so Bello-to-G1-to-Bello is identity on all 25 matched joints;
cross-body and opposite-limb correlations cannot enter the controller.

Sampling and PPO log probability stay in the released 29-dimensional G1 action
space. The sampled source action is clamped to `[-20, 20]` and only then passed
through the fixed G1-to-Bello map. The mapped 25-dimensional action is what the
environment executes and stores in its target-side history; it is never used to
evaluate the source Gaussian.

The corpus-wide anchored ridge fit is retained only as a calibration audit and
as axis/basis evidence. It is not a learned policy parameter and is not used as
a dense runtime adapter. Joint positions are transported relative to each
robot's configured default, velocities use the same signed basis, and action
history/output also converts the two robots' per-joint action scales. Vector
inputs use the explicit Bello `+Y`-forward to G1 `+X`-forward basis change.
Requested Bello joint targets are not physically clipped by the environment.
MuJoCo receives the mapped target directly, while the released soft joint-limit
penalty and target-limit diagnostics expose unsafe requests.

## Why the motion data needs three views

A Bello AMASS retarget is necessary but not sufficient. Every accepted source
frame must retain three timestamp-identical representations:

1. Bello `qpos`, `qvel`, tracked-body poses/velocities, and foot contacts for
   MJX tracking and rewards.
2. The 24-joint SMPL skeleton canonicalized into SONIC's G1-root-local, Z-up
   convention, plus its processed root quaternion and translation, retained
   for provenance and future encoder experiments.
3. The paired 29-joint G1 retarget used to fit and audit the bidirectional
   morphology-edge calibration.

Using one manifest for both retargeters is what makes the views joinable. The
builder rejects any disagreement in source digest or exact source timestamps;
it does not interpolate one retarget to make a mismatch appear valid. The
existing Bello GMR output can therefore be reused directly, while the SONIC
pipeline adds only the paired G1/SMPL sidecar.

The packed dataset is 50 Hz. Both the robot motion encoder and privileged
critic use offsets `[0,5,...,45]`; future references clamp to the final source
frame, matching the author implementation. Train/evaluation assignment comes
from the licensed manifest. Dense calibration is fitted only on accepted
training frames. Initial sampling is sequence-length agnostic: manifest weight
sets clip mass and the adaptive sampler applies the released bin-width,
failure-rate, uniform-mixture, and 200-frame pre-failure rules. The index binds
the portable manifest, Bello robot, both GMR revisions/configs, calibration,
separate policy/action and MuJoCo-qpos joint orders, clip content, split
descriptors, and split-file hashes.

Before a clip can enter either split, the pinned GMR universal-profile gate
must pass its tracking, jitter, and collision checks. SONIC-specific sanitation then labels its root
motion, root height, and contact behavior. Name-pattern treadmill/crawl clips,
stationary-root gait cycles, clips that remain near the ground for most of
their duration, and excessive contact-foot slip are quarantined. The shared
quality gate also rejects discontinuities, excessive joint velocity or
acceleration, joint-limit corruption, non-finite values, and floor
penetration. Calibration
checks mean and per-joint explained variance plus per-joint RMS residuals, then
applies the fitted training maps to the held-out split and records those
metrics in the dataset index. The held-out gate permits at most 0.01 negative
explained variance for an individual noncritical joint, while the stricter
training threshold, held-out mean, critical-joint, and RMS gates remain in
force. This avoids rejecting an otherwise stable map for a low-variance wrist
axis. A build fails rather than publishing a dataset whose morphology map does
not generalize.

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
It writes only plain `weights.npz`, `golden.npz`, and `manifest.json`. Schema
version 3 includes the G1 robot motion encoder, released critic weights, and its 1,645-value running
mean/variance in addition to the actor. The loader verifies file, array,
architecture, source, golden-vector, and exact deterministic export hashes.
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

The Bello view uses GMR's pinned `universal` task profile and shared offline
solver directly. Retarget format 10 records its velocity-limit, settling,
smoothing, and complete quality report. A successful conversion is retained
even when the GMR quality gate rejects the motion; the dataset build reports
that clip as quarantined. This keeps data-quality decisions separate from
worker failures while preventing a rejected clip from entering either split.

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

An already sanitized schema-6 bank can be migrated without repacking its large
arrays:

```bash
uv run bello-sonic-data \
  operation=upgrade_v6 \
  input_index=/data/sonic/schema6/index.json \
  output_root=/data/sonic/schema7
```

The command revalidates the self-authenticating index and split hashes, writes
the clip-balanced schema-7 metadata, and hard-links the unchanged NPZ files.

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
described above. The actor receives 1,450 raw values. Its first 640 values are
ten 64-value G1 robot-command rows containing mapped reference joint position,
mapped reference joint velocity, and root orientation relative to the current
robot. For checkpoint parity, all ten qpos rows precede all ten qvel rows before
the released `(10, 58)` reshape; changing that upstream ordering changes every
frozen token. The remaining 810 values are term-major ten-frame histories
of gravity, base angular velocity, Bello joint position, Bello joint velocity,
and previous action. Observation corruption is applied once when a history
frame is recorded; the critic retains a separate clean history.

Training also mirrors the pinned environment randomization: full-strength
observation noise from reset, `[-0.01, 0.01]` joint-default offsets, `0.8` to
`1.5` torso/wrist mass scaling, anisotropic torso COM offsets, sliding friction
from `0.3` to `1.6`, four-to-six-second velocity pushes, randomized initial
episode age, and deterministic 10% motion-level freeze-frame augmentation.
MuJoCo exposes one sliding-friction coefficient rather than PhysX's separate
static/dynamic pair and restitution buckets; those two backend-specific
material degrees of freedom are therefore not fabricated.

The critic receives the released 1,645-value G1-semantic layout: 290 future
joint positions, 290 future joint velocities, a 3-value anchor delta, a 6D
anchor orientation, 42 body-position values, 84 body-orientation values, and
930 clean history values. Bello vectors, joints, and actions are aligned before
this boundary. The exact released running mean/variance then normalizes and
clips the input before the frozen critic plus LoRA residuals.

Training transfers the complete training bank to the accelerator once. Starts
use the sequence-length-agnostic manifest prior mixed with the released
failure/uniform curriculum and pre-failure offset. Failure/attempt bins cover
the whole corpus and remain in the runner continuation, so PPO updates do not
pause for host data loads and vectorized worlds are not reset at data-boundary
transitions. Sampling uses one uniform draw against the global cumulative
distribution rather than materializing per-environment categorical noise over
every bin.

The maintained profile starts PPO directly from zero-output LoRA residuals, as
in Any2Any. The older one-step supervised pose-copy warm-start remains an
explicit opt-in experiment but is disabled by default because it does not
model target dynamics and can move the policy away from the source behavior
before PPO receives survival feedback.

The maintained PPO values are 4,096 worlds, 24 rollout steps, five epochs, four
minibatches, actor-edge learning rate `2e-5`, critic learning rate `1e-3`,
desired KL `0.01`, entropy coefficient `0.01`, gamma `0.99`, lambda `0.95`, clip
ratio `0.2`, and maximum gradient norm `0.1`. The released 29-dimensional
standard deviation remains frozen and is clamped in source space before
sampling. The maintained budget is 30,000 updates
(`2,949,120,000` transitions). Update 8,000 matches the iteration scale used
in Any2Any's compute comparison; the remaining updates provide the longer
Bello convergence tail requested after the earlier learning curves.

Evaluation reports the original clip-weighted acceptance metrics as well as
frame-weighted tracking metrics and strata for stationary/traveling root motion
and gait/non-gait contact behavior. It also evaluates a next-reference-pose
action oracle under the same strict simulator termination rules. The oracle
separates policy-transfer errors from retargeted clips that Bello's actuators
cannot dynamically replay and supplies per-clip evidence for the next dataset
sanitation pass.

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
acceleration terms. Bello adds direct joint-position and foot-contact tracking,
plus a penalty on the requested target's normalized joint-limit excess before
the simulator clips it; this prevents clipping from hiding unsafe commands.

Termination uses the target-transfer limits reported by Any2Any: `0.25 m`
root-height error, `0.8` squared root-orientation error, and `0.30 m` tracked
end-effector-height error; the Bello foot-position limit is also `0.30 m`.
Low-reference motions retain the existing adaptive height allowance.
Root-height adaptation uses the same `0.1` exponential moving average. Reset
perturbations, ten-frame sensor noise, joint-default
offsets, 4–6 second velocity pushes, friction, mass/COM variation, and actuator
strength are expressed through Bello's MuJoCo model randomizer. These are
semantic simulator adaptations rather than bitwise Isaac Lab event parity:
MuJoCo has one sliding-friction coefficient here, and Bello's mass topology
does not reproduce G1 wrist/torso bodies one-for-one. Observation noise ramps
from zero over the first 48,000 world steps, and root pushes remain disabled
for that same warmup so the edge first learns nominal tracking.

## Evaluate the transfer

For a branch-pinned Vast.ai benchmark, 30,000-update run, and complete
held-out evaluation in one billed lifecycle, use:

```bash
uv run bello-vast-train \
  experiment=sonic \
  dataset_path=/data/sonic/packed/index.json \
  sonic_artifact_path=artifacts/sonic/default_release \
  total_timesteps=2949120000 \
  num_envs=4096 \
  checkpoint_every_updates=500 \
  branch=feature/sonic-transfer \
  max_price=1.25 \
  duration_hours=24 \
  skip_confirmation=true
```

This budget is 30,000 PPO updates at the maintained 4,096-world, 24-step batch.
Update 8,000 remains the paper-comparable Any2Any milestone; the longer tail is
intentional because Bello's learning curves were still improving there.
Before the real job, the launcher runs three updates at the same environment
count and records GPU utilization, peak memory, power, elapsed time, and exit
status. A failed production-scale benchmark prevents the longer run. The
launcher verifies every dataset and release-artifact file before training. W&B
retains checkpoints; `benchmark.json`, the full evaluation, compact summary,
and checksum manifest are retrieved under `outputs/vast/<run-name>/` before
teardown.

For a full from-scratch run, omit `resume_path`. `total_timesteps` is the final
absolute budget, not an additional budget. To
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
