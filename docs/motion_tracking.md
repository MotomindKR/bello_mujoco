# Motion tracking

The `motion` experiment trains Bello; `bello_mini_motion` trains Bello Mini against
one canonical NPZ reference using BeyondMimic. Offline import/editing, dataset generation, TWIST,
and TWIST2 are removed. Prepare imported human reference data outside this repository. The separate
[barbell reference generator](barbell_feasibility.md) produces a screened rigid
fixed-load lift candidate locally; it does not change GMR.

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

## Observation modes

New motion training defaults to `proprioceptive`, including both `motion` and
`bello_mini_motion`. To opt into estimator-dependent actor inputs, set
`env_config.observation_mode` in the selected training profile, for example
`configs/profiles/train/bello_mini_motion.yaml`:

```yaml
env_config:
  observation_mode: full_state
```

`proprioceptive` follows BeyondMimic's
no-state-estimation actor: reference joint positions/velocities, relative anchor
orientation, root-local angular velocity, measured joints, and previous action.
It excludes anchor-position error and base linear velocity. Critic inputs,
rewards, and simulation height/orientation terminations retain full ground truth
in both modes. Noise, feedback filtering, and latency still apply to actor inputs.

The mode is saved in checkpoints and exported bundles. It requires training a
new policy; changing metadata or zeroing inputs of a full-state policy is not a
conversion. Evaluation/export use the saved mode. Runtime rejects a mode/layout
mismatch instead of adapting the network inputs.

For a proprioceptive bundle, omit `motion.pose` from the hardware YAML. The native
runner uses hardwared's IMU orientation/gyro and measured joints, including the
CAD IMU rotation and actual waist kinematics, with no pose-service connection.
This removes the translational estimator dependency, not IMU orientation sensing.
Absolute torso/foot-height guards are unavailable in this runtime mode;
reference-relative orientation, motor/joint/step, freshness, fault, and control
lifecycle protections remain. Full-state bundles still require `motion.pose`
and retain their height guards. Commission new weights in the emulator before
physical use; removing pose feedback does not guarantee stability.

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
uses Mini's canonical mass, motor limits, and ankle transmission, with rounded,
mass-adjusted G1-reference gains in `asset_config.joint_gains`. Only gains change;
the robot is not rescaled, and the bounded motor-range action mapping is retained.
Locomotion gains remain unchanged. Train fresh weights; existing checkpoints are
not converted. It is a starting recipe, not a commissioned policy.

The native runner blends each motor's KP/KD alongside the target
during standing/motion transitions. Each active policy uses its own trained
gains. An interrupted transition returns from the last applied blend, not from
an assumed endpoint. Preflight validates both policies' gains against hardware
limits; motor-coordinate compatibility and hardware protections remain mandatory.
Commission the standing/motion pair and transitions together before physical use.

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
| Actuation | Configured PD gains, canonical effort bounds, ankle tendons, motor/joint target projection and target-step limits; no action LPF |
| Transport | Previous target is held for the configured physics substeps; actor feedback has sampled latency |
| Feedback | First-order velocity filtering and quaternion orientation filtering; joint positions and reference commands are not smoothed |
| Corruption | Velocity/encoder noise before filtering, optional residual noise afterwards; anchor-error noise remains observation-side |
| Physics variation | Shared per-world mass, inertia, CoM, friction, gains, joint-origin and zero-offset randomization |
| Disturbances | Randomized reset state, default offsets, and independently sampled push intervals |

Mini deliberately uses modest CoM variation, with no locomotion CoM curriculum.
Its exact ranges are in its profile. Gait scheduling, standing rewards, and
locomotion foot-placement goals are not imported into reference-motion tracking.

Full-state motion observes root pose and velocities from the state estimator,
not raw IMU acceleration, and checks root/IMU orientation agreement.
Proprioceptive motion derives orientation and angular velocity directly from
hardwared's IMU feedback. Neither mode filters already-filtered telemetry again.
The training filter approximates feedback response, not the full asynchronous
implementation. Critic observations and rewards remain clean.

CPU video evaluation uses nominal geometry/dynamics and an unperturbed reference
reset, while retaining command delay and seeded actor noise/filter/latency. It
does not apply training pushes. Training uses randomized MJX worlds. GPU rollout
validation and trained-policy emulator commissioning remain required before deployment.

Native evaluation compiles its task math once per rollout while MuJoCo physics
and llvmpipe rasterization remain CPU-only. Dynamic contact buffers use size
buckets to reuse compiled kernels. Video workers have a bounded wall-clock budget
of 300 seconds for setup plus one second per requested simulation step; a long
reference is not subject to the short preflight's fixed five-minute deadline.
Evaluation errors still fail training, and their traceback is logged before W&B
closes so the cause survives host teardown.

Live upper-body retargeting is separate and retains shape/frame and freshness
checks without per-source revision fingerprints. See [hardware](hardware.md).

## Fixed barbell motion

`bello_mini_barbell_motion` uses the shared rigid bar scene and fixed 15 kg
mass/inertia. Its NPZ includes `free_joint_qpos`, `joint_qpos_indices`,
`joint_qvel_indices`, `bar_mass_kg`, and `screen_passed`. The coordinate metadata
must match the loaded scene. Each free-body quaternion is interpolated and
normalized independently. Both bodies undergo the same reset/alignment rigid
transform; scalar joints are indexed by name rather than inferred from array
length. Robot-only reference archives retain their existing representation.

The current candidate is rejected by its nominal-hold torque screen. Review
its video and diagnostics before revising the stance or starting training.

For barbell motion, grip-position and shaft-axis residuals are diagnostics only;
they do not terminate an episode. `done/barbell_contact` reports unintended bar
contact above 1 N, independently of the ordinary tracking/fall conditions. PPO
logs `mean_barbell/*` and `max_barbell/*` for grip errors, constraint forces,
transverse moments, and unintended contact force. These measure numerical ideal
attachments, not finger strength or slip.
