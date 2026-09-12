# Motion tracking

The maintained `motion` experiment trains Bello against one canonical NPZ
reference using BeyondMimic. Offline import/editing, dataset generation, TWIST,
and TWIST2 are removed. Prepare reference data outside this repository.

```bash
uv run bello-train experiment=motion motion_path=motions/asap_fixed_head/CLIP.npz
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

Edit `configs/train_motion/default.yaml` and `configs/profiles/train/motion.yaml`
for recipes. Tracked bodies live in `src/bello_mujoco/robots/*_motion.yaml`.
Mini motion training still requires its own tuned experiment.

Live upper-body retargeting is separate and retains shape/frame and freshness
checks without per-source revision fingerprints. See [hardware](hardware.md).
