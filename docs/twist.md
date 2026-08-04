# TWIST teacher and student

TWIST is an isolated whole-body tracking path. It does not change locomotion,
single-motion training, evaluation, Vast, or hardware behavior. The public
stages are deliberately typed:

```bash
uv run bello-train experiment=twist_teacher \
  dataset_path=datasets/twist/index.json

uv run bello-train experiment=twist_student \
  dataset_path=datasets/twist/index.json \
  teacher_checkpoint_path=checkpoints/<teacher>/final

uv run bello-eval task=twist_student \
  checkpoint_path=checkpoints/<student>/final
```

There is no automatic teacher selection. Teacher checkpoints cannot resume a
student run, student checkpoints cannot resume a teacher run, and public TWIST
evaluation rejects teacher and legacy motion checkpoints. TWIST hardware and
Vast execution are intentionally not exposed.

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
to produce trustworthy checksums. Review the generated manifest before
validation and preserve it with reported runs.

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

Prepare the fixed-waist, fixed-head canonical Bello model:

```bash
uv run python tools/prepare_twist_model.py
```

Install GMR's optional dependencies in a separate environment following
`deps/GMR/README.md`. Place licensed SMPL-X models below:

```text
/operator/models/body_models/
  smplx/SMPLX_NEUTRAL.pkl
  smplx/SMPLX_FEMALE.pkl
  smplx/SMPLX_MALE.pkl
```

Common AMASS SMPL-H archives can be normalized without retaining hand pose:

```bash
uv run python tools/twist_normalize_smpl.py \
  --source /operator/data/amass/source.npz \
  --output /operator/work/normalized/source.npz
```

Run retargeting in the GMR environment. Use the stable manifest identifiers in
the output filename:

```bash
python tools/twist_gmr_retarget.py \
  --source /operator/work/normalized/source.npz \
  --provenance-source /operator/data/amass/source.npz \
  --body-model-root /operator/models/body_models \
  --robot-xml <path-printed-by-prepare-twist-model> \
  --output /operator/work/retargeted/amass__cmu_001_01.npz
```

OMOMO human trajectories use the same SMPL-X retargeter. Object identifiers and
interaction labels stay in the source manifest.

Each retargeted clip records source timestamps, root/joint state, body
transforms and velocities, contact hints, joint/body ordering, source checksum,
GMR revision and configuration, and the Bello fingerprint. Conversion aligns
initial heading and floor deterministically and resamples to 50 Hz.

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
persistent joint-limit violations, conversion-scale velocities, implausible
root height, or floor penetration. `conversion_report.json` gives accepted,
quarantined, and rejected counts and reasons.

Accepted clips are packed into pickle-free NPZ shards plus `index.json`.
Training samples clips in proportion to accepted frame count multiplied by the
explicit manifest weight, then samples a valid start frame without crossing a
clip boundary. Evaluation visits every held-out clip in stable source/sequence
order. The fingerprint covers the portable manifest, accepted hashes, quality
configuration, Bello model, and GMR revision.

## Observation and optimization contracts

Simulation runs at 500 Hz, policy actions at 50 Hz, and episodes for at most 10
seconds. A reference frame contains root height, root orientation in 6D form,
local root linear velocity, yaw rate, 25 ordered joint positions, and nine
local key-body positions. Proprioception contains base angular velocity,
roll/pitch, joint positions and velocities, and the previous action.

The teacher uses the 20 released future offsets
`[1, 5, 10, ..., 95]`. Its temporal encoder is `frame→60`,
`Conv1D(60→40, kernel=6, stride=2)`,
`Conv1D(40→20, kernel=4, stride=2)`, then a 128-dimensional projection.
The student has a separate current-reference encoder and receives only the
current target plus the current and ten historical proprioceptive frames.
Future reference, base linear velocity, root height, current key bodies, foot
contacts, and randomization parameters are critic-only.

Both stages use `[512, 512, 256, 128]` SiLU actor/critic backbones with
LayerNorm on the final hidden layer. Teacher training is PPO. Student training
uses PPO plus `KL(student || teacher)` with frozen teacher parameters. Its
weight follows the corrected cosine schedule from `0.1` to `0.01` over 60,000
updates.

The maintained profile uses 4,096 environments, 24 rollout steps, five epochs,
four minibatches, learning rate `2e-4`, desired KL `0.008`, entropy `0.005`,
gamma `0.99`, lambda `0.95`, and clip ratio `0.2`. Command-level
`total_timesteps` and `num_envs` overrides remain available for bounded smoke
runs.

## Checkpoints and evaluation

TWIST checkpoint metadata records stage, full configuration, source revision,
observation layouts, dataset and robot fingerprints, and joint/body ordering.
Student metadata additionally binds the exact teacher metadata fingerprint and
KL schedule. Resume and evaluation reject incompatible stages, datasets,
robots, layouts, or ordering.

Held-out evaluation writes `outputs/evaluation/twist_student.json` by default.
It reports global, per-source, and per-clip tracking RMS, return, reward terms,
foot slip, and termination counts. A tiny generated or repository-owned motion
may be used for tests and CPU smoke only; it is not evidence of model quality.
