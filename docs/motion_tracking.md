# Motion data and tracking

The supported motion schema is the 25-joint, fixed-waist/fixed-head,
sphere-hand Bello layout. The canonical tracker is
`beyondmimic_joint_limits`: a tanh-Gaussian policy maps normalized actions
asymmetrically from the nominal pose toward each joint's lower or upper soft
limit, then applies the physical differential-ankle transform.

## Import

Legacy GMR pickle input is trusted local data:

```bash
uv run bello-motion \
  operation=import \
  source=/path/to/source.pkl \
  output=motions/local/sequence.npz
```

Import a directory with a manifest:

```bash
uv run bello-motion \
  operation=import_dataset \
  source=/path/to/gmr \
  output=motions/local/lafan1_fixed_head \
  jobs=8
```

LAFAN1-derived binaries remain local because of their source license. The
tracked ASAP clips under `motions/asap_fixed_head/` may be used directly.

## Inspect and render

```bash
uv run bello-motion operation=inspect source=motions/local/sequence.npz
uv run bello-motion \
  operation=render \
  source=motions/local/sequence.npz \
  output=outputs/reference.mp4
```

Inspection replays the reference through the canonical native MuJoCo model and
reports completion, tracking errors, actuator-force ratios, contacts, and foot
slip.

## Train and evaluate

```bash
uv run bello-train \
  experiment=motion \
  motion_path=motions/local/sequence.npz

uv run bello-eval \
  task=motion \
  checkpoint_path=checkpoints/sequence/final \
  output_path=outputs/sequence_eval.mp4
```

Motion checkpoints bind the motion archive's complete schema and content hash.
Relocation is allowed only when those values match exactly. Adaptive
failure-bin state and observation normalization are restored for resume.
