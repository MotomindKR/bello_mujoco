# Motion tracking

The `motion` experiment trains Bello; `bello_mini_motion` trains Bello Mini against
one canonical NPZ reference using BeyondMimic. Offline import/editing, dataset generation, TWIST,
and TWIST2 are removed. Prepare reference data outside this repository.

```bash
uv run --group training bello-train experiment=motion motion_path=motions/asap_fixed_head/CLIP.npz
uv run --group training bello-train experiment=bello_mini_motion motion_path=/path/to/mini/CLIP.npz
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

For [Vast training](vast_ai.md), `experiment=motion motion_path=/local/clip.npz`
uploads the selected clip over SSH with transfer verification. Motion data does
not need to be tracked in the source commit.

Training and deployment apply actions without an action low-pass filter.
Action/target bounds remain enforced; hardwared still filters sensor feedback.

The [Rust policy runtime](../rust/bello-policy/README.md) supports motion telemetry
and the standing/motion hardware lifecycle using bundled clips and headless
MuJoCo kinematics. Synthetic inference and lifecycle checks pass. The Python
policy runner is removed; physical deployment remains gated on trained-policy
commissioning in the full emulator, as described in [hardware deployment](hardware.md).

Edit `configs/train_motion/default.yaml` and `configs/profiles/train/motion.yaml`
for recipes. Tracked bodies live in `src/bello_mujoco/robots/*_motion.yaml`.
Mini overrides live in [bello_mini_motion.yaml](../configs/profiles/train/bello_mini_motion.yaml).
Supply a Mini-retargeted clip, or `resume_path` to use a checkpoint's bundled clip.
A G1/full-size Bello clip is not interchangeable with Mini motion data. The profile
uses Mini's canonical mass, motor gains/limits, and ankle transmission, without
rescaling the robot to G1. It is a starting recipe, not a commissioned policy.

## BeyondMimic alignment

Comparison reference: upstream [tracking configuration](https://github.com/HybridRobotics/whole_body_tracking/blob/cd65172032893724b445448818c34165846d847d/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py),
[reward functions](https://github.com/HybridRobotics/whole_body_tracking/blob/cd65172032893724b445448818c34165846d847d/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py),
and [G1 setup](https://github.com/HybridRobotics/whole_body_tracking/blob/cd65172032893724b445448818c34165846d847d/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py).

The six anchor/body pose and velocity terms use the same exponential squared-error
formulas and weights. Relative pose targets preserve reference height and align
with the robot's current anchor XY/yaw. Joint-limit shaping sums linear violations
outside a configurable fraction of the logical joint range; hard motor and coupled
joint constraints remain separate. Mini tracks its wrist-roll links rather than
G1's wrist-yaw links. Exact weights, tolerances, and soft-limit factor live in YAML.

This is still an adaptation, not a bit-for-bit upstream reproduction:

- Motor-coordinate bounded actions replace G1's joint-coordinate Gaussian actions.
- Normalized torque costs are retained for Bello hardware. Our tracking and
  regularization terms are per control step; torque costs alone include the timestep,
  unlike Isaac Lab's common reward-manager timestep scaling.
- Contact penalties currently count instantaneous forceful terrain contacts on
  undesired bodies, not upstream's history-based net forces including self-contact.
- Pushes add velocity instead of overwriting it; episode duration follows the clip.

## Sim-to-real path

| Mechanism | Motion behavior |
| --- | --- |
| Actuation | Canonical PD gains, effort bounds, ankle tendons, motor/joint target projection and target-step limits; no action LPF |
| Transport | Previous target is held for the configured physics substeps; actor feedback has sampled latency |
| Feedback | First-order velocity filtering and quaternion orientation filtering; joint positions and reference commands are not smoothed |
| Corruption | Velocity/encoder noise before filtering, optional residual noise afterwards; anchor-error noise remains observation-side |
| Physics variation | Shared per-world mass, inertia, CoM, friction, gains, joint-origin and zero-offset randomization |
| Disturbances | Randomized reset state, default offsets, and independently sampled push intervals |

Mini deliberately uses modest CoM variation, with no locomotion CoM curriculum.
Its exact ranges are in its profile. Gait scheduling, standing rewards, and
locomotion foot-placement goals are not imported into reference-motion tracking.

Motion observes root pose and velocities from the state estimator, not raw IMU
acceleration. The native runner checks the root/IMU orientation relationship and
consumes already-filtered telemetry; it must not apply this training filter again.
The training filter approximates hardwared plus estimator response, not their
full asynchronous implementation. Critic observations and rewards remain clean.

CPU video evaluation uses nominal geometry/dynamics and an unperturbed reference
reset, while retaining command delay and seeded actor noise/filter/latency. It
does not apply training pushes. Training uses randomized MJX worlds. GPU rollout
validation and trained-policy emulator commissioning remain required before deployment.

Live upper-body retargeting is separate and retains shape/frame and freshness
checks without per-source revision fingerprints. See [hardware](hardware.md).
