# Training and evaluation

## Train and resume

Use `nix develop .#training` on a Linux x86 CUDA host. The training dependency
group is opt-in; the default shell and Lima do not install JAX's CUDA stack.

```bash
uv run --group training bello-train experiment=bello_mini_locomotion
uv run --group training bello-train experiment=bello_mini_standing
uv run --group training bello-train experiment=motion motion_path=motions/asap_fixed_head/CLIP.npz
uv run --group training bello-train experiment=bello_mini_locomotion resume_path=checkpoints/RUN
```

Training uses MJX-Warp on CUDA. Run `experiment=smoke` before a long job.
Recipes own PPO, commands, rewards, randomization, checkpoints, W&B, and videos.
See [configuration locations](training_configuration.md).

Checkpoint metadata records the root Git commit. Training/resume/evaluation
require that exact clean checkout with pinned dependencies. For an older run,
create a separate worktree at its recorded commit and install that revision's
locked environment. There are no mismatch flags, migrations, asset fingerprints,
or fallback config readers.

Resumable state includes actor/critic, optimizer, normalization moments, PRNG,
rollout continuation, and task state. Writes stage before publishing; file and
directory durability errors fail explicitly. A post-rename fsync failure may
leave a visible checkpoint, but the save is still reported as failed.
Motion checkpoints also contain `motion.npz`, serialized from the reference
loaded by training. Resume and evaluation use that copy; the original source
file need not exist or be tracked in Git. There is no motion-path override.

## Evaluate

On Linux, including Lima:

```bash
uv run bello-eval task=locomotion checkpoint_path=checkpoints/RUN gamepad_enabled=false steps=1500
uv run bello-eval task=motion checkpoint_path=checkpoints/RUN output_path=outputs/motion.mp4
```

Locomotion evaluation streams to Rerun; motion evaluation writes video to
`output_path`.

Commands use the configured robot command frame; Bello and Mini do not share
their forward axis. Evaluation reconstructs saved normalization and feedback
settings rather than silently cleaning observations.
Low-level policy-video calls also require the checkpoint's PPO settings and
normalization explicitly; they do not create replacement statistics.

Policy inference and simulation run on CPU. Rasterization uses Mesa
llvmpipe through EGL only. The Nix shells supply Mesa; on Ubuntu install
`libegl1 libegl-mesa0 libgl1-mesa-dri`. The actual GL renderer is checked before
recording, and a GPU renderer fails initialization. There is no GPU or
native-macOS rasterizer fallback. The Mesa software EGL device is selected
explicitly, even on hosts with physical GPUs. Training videos run in a fresh
software worker per video. Both tasks allow five minutes of startup overhead plus
one wall-clock second per requested simulation step for each video worker. Longer
standing diagnostics therefore receive a proportionally longer timeout.
Enabled video failure, crash, or timeout fails the training run.
Set `video_enabled=false` explicitly when videos are unwanted.

### Standing and push diagnostics

`video_config.standing_suite_enabled` adds a fixed-camera quiet-standing rollout
and four independent world-XY velocity-kick rollouts to each video event. Push
cases are named for world axes, not assumed robot forward/left axes. Every case
uses the same policy and normalization. The suite retains the rendering
environment's sampled physics, observation noise, filtering, and delay; it is
not a clean nominal-physics or emulator qualification. Each case runs in its own
CPU software worker, and an enabled render/trace failure fails training.

The suite records 30 seconds of quiet standing, or a push followed by 15 seconds
of recovery. `standing_push_at_seconds` and `standing_push_speed` configure the
kick. Without the suite, `standing_push_enabled` adds one push rollout and cycles
directions between events. The Mini profile sets video/checkpoint cadence.

W&B receives per-case videos and reward-independent settling metrics; matching
JSONL traces sit next to the videos. Overlays show support, motion, foot alignment,
reward-relaxation scale, and torque saturation. A recovery is successful only
when the robot was settled before the push, settles again afterward, and has no
terminal failure. Training episode timeouts do not truncate the diagnostic
horizon; physical/numerical termination still stops it. The physical criteria
are implemented in `evaluation/settling.py`, separately from reward weights.

## Deployment artifact

```bash
uv run bello-export-policy checkpoint_path=checkpoints/RUN output_path=outputs/policy.zip
```

The ZIP bundle contains `policy.onnx`: the deterministic actor, normalization,
motor-target mapping, and source/inference metadata. `physics.xml` contains the
runtime physics, native joint/tendon transmissions, sensors, and home state,
without visual meshes or external file references. The pinned URDF remains
canonical; this MJCF is a generated deployment artifact, not a second source.
Motion tracking also includes `motion.npz`; locomotion does not.
It is not a resumable checkpoint.
`rust/bello-policy` provides the CPU Rust consumer in this repository, validating
the source commit and live tensor dimensions before inference.
