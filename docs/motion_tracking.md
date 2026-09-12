# Motion tracking

The maintained `motion` experiment trains Bello against one canonical NPZ
reference using BeyondMimic. Offline import/editing, dataset generation, TWIST,
and TWIST2 are removed. Prepare reference data outside this repository.

```bash
uv run --group training bello-train experiment=motion motion_path=motions/asap_fixed_head/CLIP.npz
uv run bello-eval task=motion checkpoint_path=checkpoints/RUN output_path=outputs/motion.mp4
```

Archives contain frame rate, named joints/bodies, qpos/qvel, and tracked-body
kinematics. `motion/clip.py` validates those arrays when loading. Checkpoints
contain the reference clip used by training. Resume, evaluation, and deployment
use the bundled copy, with no external-path or content-hash contract. Source
compatibility is the root Git commit.

BeyondMimic tracks aligned pose/velocity with adaptive failure sampling,
observation corruption, pushes, and reset randomization. The bounded variant
maps actions to motor limits; the effort/stiffness variant scales motor offsets
by configured motor effort and stiffness. Both retain native ankle tendons and
joint-space kinematic/limit rewards. The older reference-residual formulation
no longer exists.

Training and deployment apply actions without an action low-pass filter.
Action/target bounds remain enforced; hardwared still filters sensor feedback.

The [Rust policy runtime](../rust/bello-policy/README.md) supports read-only
motion telemetry inference using the bundled clip and headless MuJoCo kinematics.
Its native standing/motion operator lifecycle is not yet connected; use the
Python hardware runner for live motion control during this migration.

Edit `configs/train_motion/default.yaml` and `configs/profiles/train/motion.yaml`
for recipes. Tracked bodies live in `src/bello_mujoco/robots/*_motion.yaml`.
Mini motion training still requires its own tuned experiment.

Live upper-body retargeting is separate and retains shape/frame and freshness
checks without per-source revision fingerprints. See [hardware](hardware.md).
