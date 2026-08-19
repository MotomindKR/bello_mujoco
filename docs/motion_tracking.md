# Motion data and tracking

The supported motion schema is the 25-coordinate, fixed-waist/fixed-head,
sphere-hand Bello layout. Motion archives are canonical NPZ files with explicit
format version, frame rate, joint/body names, kinematics, and source identity.
Repository importers additionally emit self-collision diagnostics.

The full offline command surface is documented under
[`bello-motion`](commands.md#bello-motion); training and evaluation fields are
documented under [`bello-train`](commands.md#bello-train) and
[`bello-eval`](commands.md#bello-eval).

## Data lifecycle

| Stage | Behavior |
| --- | --- |
| Import | Convert a trusted local GMR pickle, validate its schema and joint ranges, reconstruct qpos/qvel and tracked-body state, and write one canonical archive. |
| Dataset import | Process a nonrecursive directory of GMR pickles in parallel, reuse or overwrite existing outputs, and write a manifest containing successes and failures. |
| Inspect | Replay a canonical archive through native MuJoCo and report completion, errors, force ratios, contacts, and foot slip. |
| Reference render | Render the archive itself without loading a policy. |
| Training | Bind one archive into the motion environment and learn its tracking policy with MJX-Warp PPO. |
| Policy evaluation | Restore a checkpoint, validate its bound archive, run deterministic native MuJoCo tracking, and emit video plus RMS metrics. |

Legacy pickle input is executable Python serialization and is accepted only as
trusted local data. LAFAN1-derived outputs remain local because of their source
license. Repository-tracked ASAP clips under `motions/asap_fixed_head/` can be
selected by local or Vast training.

Single-file import rejects self-collision by default. Dataset import instead
records collision diagnostics so a complete corpus can be reviewed; the global
single-import rejection setting does not change that dataset behavior.

Inspection emits its report even when replay terminates early. Use
`require_complete=true` when incomplete replay must make the operation fail.

## Tracking formulation

The maintained profile uses `beyondmimic_joint_limits`. A tanh-Gaussian policy
maps each normalized action asymmetrically from the nominal pose toward that
joint's lower or upper soft limit, then applies the physical
differential-ankle transform.

Observations include aligned reference state, anchor/body tracking features,
base and joint state, previous action, and optional noise. Root linear velocity
uses world coordinates in MuJoCo state, while root angular velocity and the
hardware gyro use the root-local free-joint convention. Training can:

- randomize reference start time and initial root/joint state;
- perturb observations;
- delay measured actor feedback by a per-episode randomized number of policy steps;
- apply random base velocity pushes;
- bias future resets toward time bins with higher failure EMA;
- terminate on root height/orientation or end-effector tracking error.

The reward combines Gaussian root, joint, tracked-body pose, and velocity terms
with action-rate, joint-limit, and undesired-contact penalties. Exact fields,
weights, sigmas, and profile values are in the
[training configuration reference](training_configuration.md#motion-selection-and-environment).

The maintained profile samples actor-feedback latency uniformly from zero to
two policy steps, or 0--40 ms at the 50 Hz control rate. The delayed terms are
measured anchor pose, base velocity, and joint position/velocity. Reference
commands and the previous action remain current, and the asymmetric critic
continues to receive current feedback. Native evaluation and physical
deployment do not inject synthetic delay. Existing checkpoints that predate
this field remain usable for evaluation and deployment with latency
randomization disabled. Resuming their saved rollout continuation requires the
original source revision because the older state does not contain feedback
history.

## Checkpoint binding

Motion checkpoints store the archive's:

- content and source hashes;
- format version and frame rate;
- frame count;
- ordered joint and tracked-body schemas;
- selected environment and action formulation.

APIs and runners that expose a motion-path override allow relocation only when
the replacement archive matches that complete contract. The public
`bello-eval task=motion` command exposes no such override and therefore requires
the checkpoint's stored path to remain available. Resume additionally restores
optimizer, environment continuation, observation/value normalization, RNG, and
adaptive failure-bin state.

Motion checkpoints maintain a `final` child below their timestamped run root.
This differs from locomotion, whose final checkpoint is the timestamped root
itself.

## Native evaluation and deployment

Native evaluation uses the same bounded action transformation and observation
normalization stored by training. It reports completion fraction, episode
return, terminal/timeout state, and RMS root, orientation, and joint errors.
The generated motion asset can carry an equal configured payload at each
sphere-hand center. Training records the per-hand mass in checkpoint metadata;
evaluation uses that mass by default and accepts an explicit load-ablation
override. Hardware deployment treats the same field as a declared physical
load and refuses a value that differs from the checkpoint.

Physical motion deployment adds absolute pelvis PoseService input and validates
Pose/IMU agreement, nominal-pose arrival, action bounds, and reference
completion. See [Physical hardware](hardware.md) for that safety boundary.
