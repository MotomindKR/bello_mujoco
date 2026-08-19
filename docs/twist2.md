# One-stage TWIST2

This repository exposes TWIST2 as an isolated, single-stage whole-body PPO
task alongside `twist_teacher` and `twist_student`:

```bash
uv run bello-train \
  experiment=twist2 \
  dataset_path=/home/gmmyung/amass/twist_dataset/index.json \
  checkpoint_name=twist2_seed1
```

There is no teacher, distillation loss, automatic checkpoint selection, or
dependency on the legacy motion/BeyondMimic environment. Training uses the
dedicated `BelloTwist2Env`, `twist2` actor/critic architecture, configuration,
typed checkpoint, and evaluation path.

A completed TWIST2 policy can initialize a new run on a deliberately changed
dataset without disguising the operation as resume:

```bash
uv run bello-train \
  experiment=twist2 \
  dataset_path=/home/gmmyung/amass/twist_dataset/index.json \
  warm_start_path=/path/to/previous/final \
  allow_git_commit_mismatch=true \
  total_timesteps=491520000 \
  learning_rate=0.00005 \
  checkpoint_name=twist2_finetune_seed1
```

Warm start restores actor/critic parameters, learned action distribution, and
normalization snapshots. It resets the optimizer, update counter, learning
rate, RNG, rollout/environment state, and per-clip curriculum. Dataset identity
may change; task, robot, GMR revision, environment and parity contracts,
network architecture, and ordered observation/action layouts may not.

## Reproducibility target

The executable contract is derived from the public
[amazon-far/TWIST2](https://github.com/amazon-far/TWIST2) release pinned at
commit `d5c7108e9ef82d1b8770e5b692f27a1294f3aa8a`, together with the
[TWIST2 paper](https://arxiv.org/abs/2511.02832). The checkpoint records that
revision and refuses a different parity contract.

The maintained run uses 500 Hz simulation, 50 Hz policy control, 500-step
episodes, 4,096 environments, 24 rollout steps, 30,001 PPO updates, five PPO
epochs, four minibatches, learning rate `2e-4`, adaptive desired KL `0.008`,
entropy `0.005`, gamma `0.99`, GAE lambda `0.95`, and clip ratio `0.2`.
Actions come from an unsquashed Gaussian. The release divides its configured
action bound by the `0.5` joint-target scale, so raw actions are clipped to
`[-10, 10]` and produce at most a `[-5, 5]` radian offset before the robot's
joint limits and actuator dynamics. The policy begins with learnable unit
standard deviation. This repository's shared PPO core stores its logarithm,
rather than the release's directly parameterized standard deviation; both
start from the same unit Gaussian, but their optimizer parameterizations are
not identical.

## Observations and networks

For Bello's 25 policy joints, the actor receives 1,252 values and the critic
receives 1,634 values. These sizes and every ordered field are checkpointed.

The actor frame contains the current reference command and proprioception. The
command is local root XY velocity, root height, roll/pitch, local yaw rate, and
25 joint targets. Proprioception is base angular velocity, roll/pitch, joint
position error, joint velocity with the four ankle joint velocities masked to
zero, and previous action. The actor concatenates the current frame, ten
chronological historical frames, and a separate current-command branch. It has
no future reference and no privileged physical state.

The actor uses the released three-part encoder: a current-command MLP, a
60-channel history projection followed by `60->40` kernel-4 stride-2 and
`40->20` kernel-2 stride-1 temporal convolutions, and an offset-zero command
MLP with training-only dropout. Their latents feed a
`[512, 512, 256, 128]` SiLU backbone with final-hidden LayerNorm.

The critic receives 20 reference offsets
`[1, 5, 10, ..., 95]`, current proprioception, base linear velocity, root pose,
current positions of nine key bodies, foot contacts, and the model/domain
randomization parameters. Its privileged temporal encoder uses the released
kernel-6/stride-2 and kernel-4/stride-2 convolutions before the same backbone.

## Data and sampling

The public release's training YAML lists 19,426 clips, each with unit sampling
weight:

| Source directory | Clips |
| --- | ---: |
| `twist1_to_twist2` | 12,788 |
| `OMOMO_g1_GMR` | 5,841 |
| `AMASS_g1_GMR8` | 724 |
| `v1_v2_v3_g1` | 73 |

The current AMASS-only packed dataset is structurally usable, but it is not
the released training mixture. Every run audits total and training-split source
counts and weights, and stores that audit in W&B and the typed checkpoint. Set
`require_released_dataset: true` in `configs/train_twist2/default.yaml` when
exact public training-composition parity is required; the training split must
then contain the 19,426 entries above with unit weights or the run fails before
model initialization.

Clips are initially sampled by unit manifest weight, not duration. The released
per-clip curriculum updates difficulty using gamma `0.01`, clamps it to
`[1, 10]`, and applies the special `0.8` completion factor at near-perfect
completion. Evaluation ignores the curriculum and visits every held-out clip
in stable source/sequence order.

The released launcher trains on all entries in its YAML and does not define
this repository's leakage-safe held-out protocol. A normal local build keeps
train and evaluation clips disjoint, so it is deliberately not exact release
data parity. Exact-composition mode instead requires all released entries in
the training split; any held-out evaluation clips must be an additional,
disjoint corpus.

## Rewards and randomization

The dedicated environment ports the active released tracking rewards for joint
position/velocity, root height/orientation/linear and angular velocity, local
and global key-body position, and alive reward. It also ports foot slip,
contact-force, stumble, joint-limit, torque-limit, velocity, acceleration,
action-rate, air-time, horizontal angular-velocity, and ankle velocity and
acceleration penalties. Reward terms are logged individually.

Per-world randomization covers added base mass and center of mass, friction,
and independent proportional/derivative motor strength. Root XY velocity
pushes occur after simulation every 200 policy steps (four seconds), as in the
release, with magnitude up to 1 m/s. This port assigns every environment an
independent random push phase at reset. The released implementation instead
uses one global phase for all environments, which concentrates every push into
the same PPO rollout and produces periodic value-loss and gradient spikes.
Random phases preserve the per-environment interval and push distribution while
keeping the aggregate rollout distribution stationary.

At each stochastic training reset, active joint positions are initialized at an
independent `0.8` to `1.2` multiple of the sampled reference pose, matching the
release's recovery-state distribution. Deterministic evaluation resets retain
the exact reference joint pose.
The release declares gravity randomization but never calls the corresponding
method, so this implementation does not invent an active gravity perturbation.
Actor observation noise ramps over 1,200,000 policy steps: the configured
50,000 value is multiplied by the 24-step rollout horizon in the executed
source. The effective zero/one-frame action delay starts after that same
1,200,000-step warm-up.

## Intentional source corrections and adaptations

The implementation preserves executed behavior where the public configuration
contains inert fields. In particular, its eight-frame action buffer still
samples only the source-effective zero/one-frame delay, the declared gravity
randomizer stays inactive, and the inactive per-joint
action-standard-deviation list is represented by learnable unit standard
deviation. Actor observation moments are collected during a rollout, applied
on the next rollout, and frozen after 3,000 updates with the release's
`std + 1e-4` denominator. The public DAgger runner accidentally normalizes the
critic through a never-updated teacher normalizer rather than its populated
critic normalizer; this port makes the resulting effectively raw critic input
explicit.

One upstream indexing bug is intentionally corrected: the offset-zero command
encoder receives all 25 Bello joint targets instead of dropping its final
channel as though a disabled mask were present. The correction is explicit in
the checkpoint parity contract. Per-environment root-push phasing is recorded
there as a second intentional training adaptation. The actor's source-specific
Xavier gains, training-only dropout, and final-hidden LayerNorm are preserved. Bello
joint/key-body mappings and reference root deltas are native equivalents of
the G1 release and are covered by layout, robot, dataset, and source
fingerprints.

## Evaluation

Evaluate a completed checkpoint with:

```bash
uv run bello-eval \
  task=twist2 \
  checkpoint_path=checkpoints/twist2_seed1/final
```

This deterministically evaluates every held-out clip, writes global,
per-source, and per-clip tracking metrics to
`outputs/evaluation/twist2.json`, and renders the first held-out clip to
`outputs/evaluation/twist2.mp4`. Use `steps=250` to shorten only the video, or
`video_path=null` to omit it. Evaluation always stops a clip on terminal
failure; this cannot be disabled from the CLI. The report includes episode
return, root position and orientation error, joint-position error, foot slip,
dataset and robot fingerprints, and the pinned upstream revision.

Resume uses the same training command plus
`resume_path=/path/to/checkpoint`. Cross-stage checkpoints, changed observation
contracts, incompatible datasets or robots, and altered parity revisions are
rejected.

TWIST2 hardware deployment of the low-level policy is documented in
[Physical hardware](hardware.md#twist2-mode). High-level vision/language
planning and the paper's diffusion motion generator remain outside this
environment.
