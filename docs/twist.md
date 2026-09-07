# TWIST teacher and student

TWIST is an isolated whole-body tracking path. It does not change locomotion,
single-motion training, evaluation, or hardware behavior. The public stages are
deliberately typed:

```bash
uv run bello-train experiment=twist_teacher \
  dataset_path=datasets/twist/index.json

uv run bello-train experiment=twist_student \
  dataset_path=datasets/twist/index.json \
  teacher_checkpoint_path=checkpoints/<teacher>/final

uv run bello-eval task=twist_student \
  checkpoint_path=checkpoints/<student>/final
```

The teacher default is the released 30,002 PPO updates. With 4,096 environments
and 24 rollout steps, that is 2,949,316,608 simulated policy steps. The student
profile remains an extended 200,000-update run so the corrected 60,000-update KL
schedule reaches its endpoint. `total_timesteps` remains an explicit override
for short smoke and capacity runs.

There is no automatic teacher selection. Teacher checkpoints cannot resume a
student run, student checkpoints cannot resume a teacher run, and public TWIST
evaluation rejects teacher and legacy motion checkpoints. TWIST hardware is not
exposed. Vast training stages the explicit packed dataset, and the explicit
teacher checkpoint for a student, from the launch machine over direct SSH.

## Licensed source boundary

The repository never downloads AMASS, OMOMO, SMPL body models, or in-house
motion. Obtain each dataset from its rights holder, accept its terms outside
this repository, and keep raw files, credentials, normalized derivatives,
retargeted clips, packed datasets, and checkpoints untracked.

The three registered source classes are:

- `amass`: SMPL-H or SMPL-X locomotion and general motion. Every entry records
  the AMASS release and originating sub-dataset.
- `omomo`: SMPL-X human trajectories for object interaction. Object metadata is
  retained in the packed index, but objects are not simulated in this version.
- `in_house`: optional SMPL-X or an explicitly named BVH skeleton. `owner` and
  `ownership_basis` metadata are mandatory.

Copy [the example manifest](../configs/twist/source_manifest.example.yaml) and
replace every placeholder path, release, license, identifier, frame rate, and
SHA-256 digest. Absolute paths are used during import but removed from the
portable manifest and dataset fingerprint.

For an aggregated AMASS SMPL-X G tree, generate the reviewable source manifest
directly. The release and license strings are deliberately required instead of
being guessed, and only Stage II motion files are selected (Stage I body-fit
artifacts are excluded):

```bash
uv run bello-twist-data operation=scan_amass \
  source_root=~/amass \
  manifest=~/amass_twist_manifest.yaml \
  dataset_release='<accepted AMASS release identifier>' \
  license_id='<accepted AMASS license identifier>'
```

The scan records each source checksum, sub-dataset, subject/session path,
surface model, frame rate, and a quantized motion fingerprint for duplicate
detection. It can take substantial time because every source file must be read
to produce trustworthy checksums. Malformed and duplicate clips are excluded
and recorded beside the manifest in `source_manifest.scan_report.json`. Review
both files before validation and preserve them with reported runs.

The default versioned curation rules in
`configs/twist/amass_flat_ground.yaml` also exclude generated pipeline outputs
and motions explicitly labeled as stairs, slopes, or ladders. TWIST trains in a
flat-ground environment, so motions requiring unsupported terrain are removed
before split assignment and retargeting. Every exclusion and its reason is
recorded in the scan report. Pass `curation_config=<path>` to use another
reviewed rule set.

Validate before doing any conversion:

```bash
uv run bello-twist-data operation=validate \
  manifest=/operator/manifests/twist.yaml \
  body_model_root=/operator/models/body_models
```

Validation fails for absent body models or source files, checksum drift,
unsupported representations, incomplete provenance, duplicate content,
near-duplicate fingerprints, or inconsistent official splits.

## Splits and leakage

Set `requested_split` to `train` or `evaluation` for an official split. `auto`
uses a fixed-seed 90/10 assignment grouped by source and subject. Splits are
assigned before retargeting, and a source identity, exact content digest, or
declared near-duplicate fingerprint may not occur in both splits. Frames and
windows from one source sequence therefore cannot cross the boundary.

The first packed build writes `heldout_manifest.json`. A later build refuses to
replace it with different content. Archive that file and the dataset
fingerprint with reported results.

## Clean GMR conversion

`deps/GMR` pins upstream GMR at an exact revision. It is a tool-only submodule
and is not imported by Bello training. Initialize it with:

```bash
git submodule update --init --recursive
```

Offline conversion accepts `bello` and `bello_mini`. For Mini, the repository
wrapper prepares its MJCF and injects that generated model together with the
calibrated IK configuration into GMR at runtime. The pinned submodule does not
contain a generated Mini model for its standalone CLI. The maintained TWIST,
TWIST2, and single-motion training profiles still select Bello; Mini support in
this section is an offline retargeting/import boundary.

Prepare the selected robot's motion model (the examples use Bello; substitute
`--asset bello_mini` for Mini conversion):

```bash
uv run python tools/prepare_twist_model.py --asset bello
```

Install GMR's optional dependencies in a separate environment following
`deps/GMR/README.md`. Place licensed SMPL-X models below:

```text
/operator/models/body_models/
  smplx/SMPLX_NEUTRAL.npz
  smplx/SMPLX_FEMALE.npz
  smplx/SMPLX_MALE.npz
```

Initialize GMR's isolated environment once:

```bash
cd deps/GMR
nix develop -c true
cd ../..
```

Retarget a complete manifest with resumable workers. Each clip is normalized
to a compact temporary archive, retargeted into an atomic output, and removed
from the temporary directory after success. Existing output is reused only
when its source, GMR, robot, and configuration fingerprints still match:

```bash
uv run python tools/twist_batch_retarget.py \
  --asset bello \
  --manifest ~/amass/source_manifest.yaml \
  --body-model-root ~/body_models \
  --normalized-root ~/amass/twist_work/normalized \
  --output-root ~/amass/twist_work/retargeted \
  --jobs 1
```

Use `--limit 10` for an initial pipeline smoke run. Failed intermediates remain
available for diagnosis, and `batch_retarget_report.json` records every result.
Avoid high worker counts: each worker loads an SMPL-X model and performs CPU IK.

The underlying single-clip normalizer remains available for diagnosis:

```bash
uv run python tools/twist_normalize_smpl.py \
  --source /operator/data/amass/source.npz \
  --output /operator/work/normalized/source.npz
```

The underlying single-clip retargeter can likewise be run in the GMR
environment. Use stable manifest identifiers in the output filename:

```bash
python tools/twist_gmr_retarget.py \
  --asset bello \
  --source /operator/work/normalized/source.npz \
  --provenance-source /operator/data/amass/source.npz \
  --body-model-root /operator/models/body_models \
  --robot-xml <path-printed-by-prepare-twist-model> \
  --output /operator/work/retargeted/amass__cmu_001_01.npz
```

OMOMO human trajectories use the same SMPL-X retargeter. Object identifiers and
interaction labels stay in the source manifest.

Complete `fastsam3dbody-serve` recordings can be imported as explicitly owned
in-house SMPL-X sources. The importer validates the recorded stream contract,
splits tracking epochs into bounded train-only clips, resamples them to 50 Hz,
and removes monocular horizontal root drift with the same causal support-foot
estimator used by live hardware before GMR. The estimator uses confidence,
speed/height hysteresis, fixed support anchors, and blended support changes; its
version and complete parameters are recorded in every source clip and manifest.
It retains the original recording and writes only local generated source clips
and manifests:

```bash
uv run python tools/twist_import_fastsam_recording.py \
  --recording-root /operator/recordings/<session> \
  --source-root /operator/work/in_house/<session> \
  --source-manifest /operator/manifests/<session>.yaml \
  --base-manifest /operator/manifests/amass.yaml \
  --combined-manifest /operator/manifests/amass-plus-in-house.yaml \
  --owner <owner> \
  --subject-id <subject>
```

The emitted source manifest can be passed to `twist_batch_retarget.py`; the
combined manifest is used only for final packing. Do not commit the recording,
generated source clips, manifests, or retargeted outputs.

Each retargeted clip records source timestamps, root/joint state, body
transforms and velocities, geometry-derived sole heights, contact hints,
actuator/qpos/body ordering, source checksum, GMR revision and configuration,
and the selected robot fingerprint. Conversion uses source foot motion to ground
stance soles against the canonical collision geometry while preserving airborne
phases, aligns initial heading deterministically, and samples at exactly 50 Hz.
The selected motion profile names the pinned GMR IK configuration, source and
target types, task profile, source landmarks, output bodies, feet, and standing
joint policy. GMR's versioned `offline_solver` block is authoritative for native
velocity limiting, per-frame and initial settle passes, smoothing, collision
avoidance, and protected-body tolerances; the repository wrapper does not apply
a second pose conditioner. Knee-pole trajectories retained in the output are
diagnostic metadata for quality review and do not modify GMR targets. Robot
joint limits and the selected GMR mapping own the solved pose.

## Quality control and packing

Thresholds are versioned in [quality.yaml](../configs/twist/quality.yaml) and
are included in the dataset fingerprint. Build with:

```bash
uv run bello-twist-data operation=build \
  manifest=/operator/manifests/twist.yaml \
  body_model_root=/operator/models/body_models \
  retargeted_root=/operator/work/retargeted \
  output_root=datasets/twist
```

The build rejects missing/non-finite state and provenance mismatches. It
quarantines short clips, quaternion discontinuities, timestamp gaps,
persistent joint-limit violations or hip/waist saturation, conversion-scale
velocities, one-frame joint steps, excessive joint acceleration, implausible
root height, or more than 1 cm of sole collision-geometry penetration.
`conversion_report.json` gives accepted, quarantined, and rejected counts and
reasons.

Accepted clips are packed into pickle-free `train.npz` and `evaluation.npz`
files plus `index.json`. Dataset schema 3 stores all motion tensors as float32
and keeps source timestamps as float64. The builder uses temporary memory-mapped
arrays while producing each split file, so it does not retain the decoded
corpus in RAM. Train and evaluation data remain physically separate.

The packed index retains frame-count-weighted probabilities for provenance and
older explicit experiments. Released-parity training instead samples clips from
their manifest `sampling_weight`, independent of clip duration, and then samples
a start time uniformly. Set larger weights explicitly for curated anchor clips;
the released mix used weights of 50 or 100 for 142 mocap anchors and weight 1
for ordinary AMASS/OMOMO clips. An all-AMASS index whose weights are all 1 has
no such curated anchors and cannot reproduce that source mix until the operator
adds them.

Both teacher and student enable the released motion curriculum. Every clip
starts at difficulty 9. Terminations with at most 50% completion increase its
sampling multiplier by 1%, while at least 95% completion decreases it by 1%,
clamped to `[1, 9]`. The released active termination path keeps the root-local
per-key-body threshold fixed at 0.7 m; its dynamic-threshold line is commented
out in the release. Difficulty is updated inside the compiled rollout and
stored in continuation checkpoints. The trainer loads the complete training
split once and continuously samples it for the
entire run. Evaluation visits every held-out clip in stable source/sequence
order. Dataset and runtime sampling contracts are both checkpointed.

## Observation and optimization contracts

Simulation runs at 500 Hz, policy actions at 50 Hz, and episodes for at most 10
seconds or until the selected clip ends. A reference frame contains root height,
root roll/pitch/yaw,
local root linear velocity, yaw rate, 25 ordered joint positions, and nine
local key-body positions. Proprioception contains base angular velocity,
roll/pitch, joint positions and velocities, and the previous action.

The teacher uses the 20 released future offsets
`[1, 5, 10, ..., 95]`. Its temporal encoder is `frame→60`,
`Conv1D(60→40, kernel=6, stride=2)`,
`Conv1D(40→20, kernel=4, stride=2)`, then a 128-dimensional projection.
The student has a separate one-frame reference encoder and receives 11
chronological current-reference/proprioception frames. The newest frame is the
encoder input; no frame contains a future offset or privileged value. Base
linear velocity, root height, current key bodies, foot contacts, and physical
randomization parameters are appended to the teacher and critic observations,
matching the released teacher implementation.

Both stages use `[512, 512, 256, 128]` SiLU actor/critic backbones with
LayerNorm on the final hidden layer. Teacher training is PPO. Student training
uses PPO plus `KL(student || teacher)` with frozen teacher parameters. Its
weight follows the corrected cosine schedule from `0.1` to `0.01` over 60,000
updates. The released teacher and student configs each specify 30,002 updates;
the maintained extended student profile continues beyond 60,000 so the
corrected schedule reaches its configured endpoint.

The maintained profile uses 4,096 environments, 24 rollout steps, five epochs,
four minibatches, learning rate `2e-4`, desired KL `0.008`, entropy `0.005`,
gamma `0.99`, lambda `0.95`, and clip ratio `0.2`. Command-level
`total_timesteps` and `num_envs` overrides remain available for bounded smoke
runs. The full reference bank is passed to the compiled PPO update as dynamic
device buffers, so it is not embedded as an executable constant.

The released Gaussian action is clipped to `[-5, 5]` immediately before the
environment maps it to Bello PD targets. Pose termination compares the maximum
error of any of the nine key bodies after expressing current and reference
positions in their respective root frames. It does not sum errors across bodies
and does not terminate solely because the tracked root translated globally.

The 4,096-environment Warp profile reserves 32 active contacts per environment.
Training reports the peak active-contact capacity fraction and the fraction of
steps that overflow capacity. `mean_physics/contact_capacity_overflow` must
remain zero; otherwise increase `warp.contact_budget_per_env` deliberately.

TWIST owns its reset, reference sampling, observation history, action delay,
PD-target mapping, rewards, disturbances, termination, and domain
randomization. It imports neither the legacy motion-tracking environment nor
the BeyondMimic training pipeline.

## Checkpoints and evaluation

TWIST checkpoint metadata records stage, full configuration, sampling and
curriculum contract, source revision, observation layouts, dataset and robot
fingerprints, and joint/body ordering. Continuation state includes every clip's
current curriculum difficulty.
Student metadata additionally binds the exact teacher metadata fingerprint and
KL schedule. Resume and evaluation reject incompatible stages, datasets,
robots, layouts, or ordering.

Held-out evaluation writes `outputs/evaluation/twist_student.json` by default.
It reports global, per-source, and per-clip tracking RMS, return, reward terms,
foot slip, and termination counts. A tiny generated or repository-owned motion
may be used for tests and CPU smoke only; it is not evidence of model quality.
