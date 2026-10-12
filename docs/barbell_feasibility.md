# Barbell pose and hold diagnostics

This is an offline pose/load review, not a training task or robot command.
It does not connect to hardware, operate the hands, or change existing policies.
Use it before investing in a loaded clean-and-jerk reference and training task.

```bash
nix develop
uv run python -m bello_mujoco.evaluation.barbell \
  --config configs/analysis/barbell.yaml \
  --output outputs/barbell-review
uv run python -m http.server 8766 --bind 127.0.0.1 \
  --directory outputs/barbell-review
```

Open <http://127.0.0.1:8766>. Drag to orbit; use front/side views to check the
pose. The payload slider changes the static result, not the pose. The browser
loads pinned Three.js modules from jsDelivr; report generation needs no browser,
GPU, or rasterization. Generated GLB/report/HTML files stay under ignored outputs.

## Inputs and method

`configs/analysis/barbell.yaml` owns dimensions, payload samples, provisional grip
frames and the proposed handover position. The current review candidate has a
60-degree forearm reach from downward vertical (approximately +45 degrees wrist
pitch), with the shaft below the hand spheres along hand-local -Z (reviewed
option A), perpendicular to the previous hand-local X offset. This is an
illustrative reduced-lever-arm pose, not a validated grasp. The report gives
the solved angle, palm normals, joint positions and pose residuals.

The checker loads the pinned canonical robot, including its existing hand and
battery masses, and solves only the arms within motor and physical joint limits.
Hand locations/orientations are **sphere-proxy assumptions**, not measured palm
frames. Confirm them before using the load calculation for hardware decisions.
The configured shaft center lies beyond the sphere surface along hand-local -Z,
not the assumed +X palm normal. The report checks this distance against the
actual sphere geometry; the preview arrows show the sphere-to-shaft offset.
The bar axis lies in the palm plane, perpendicular to that normal. Neither this
normal nor the offset establishes a measured grip frame.

End-weight dimensions were supplied by the user. Equal cylindrical shapes and
outer faces flush with the assembly ends remain assumptions. Shaft mass is
neglected. The displayed bar inertia follows these explicit assumptions.

Gravity and each hand's half-payload force are converted to generalized torques
using MuJoCo Jacobians. Physical motor torques use the existing transmission
mapping, including differential ankles. Both flat soles contribute nonnegative
vertical reactions at their collision-geometry support polygons. A linear
program minimizes maximum motor and joint effort-limit utilization while
satisfying the full free-root equilibrium equations. MuJoCo's net joint actuator
limits are checked independently of individual motor limits, including at the
differential ankles. Overloads remain visible rather than being clamped away.
This distinguishes tipping/no equilibrium from effort overload without assuming
a fixed robot root.

Leg loads are one nonunique optimistic ground-force allocation, not a controller
prediction; left/right values may differ even with a symmetric pose. Arm loads
under the assumed equal payload split do not have that ground-force ambiguity.
Rated torque columns are repository references, not indefinite-hold thermal
guarantees. A result below peak is **not deployment approval**.

This static report includes robot self-collisions but does not validate bar-robot
contact, finger strength, grasp closure/slip, electrical current, thermal
capacity, or accelerating loads.

## Free-base hold sweep

```bash
uv run python -m bello_mujoco.evaluation.barbell_hold \
  --config configs/analysis/barbell_hold.yaml \
  --output outputs/barbell-hold-review
uv run python -m http.server 8767 --bind 127.0.0.1 \
  --directory outputs/barbell-hold-review
```

Open <http://127.0.0.1:8767>. Select a payload/controller and play or scrub the
recorded 3D state. Native MuJoCo runs on CPU; the browser displays geometry with
WebGL, not software-rasterized evaluation video. Results and scenes are generated
under ignored outputs, without changing the canonical robot description.

The bar is a free rigid body with explicit mass/inertia and collidable shaft/end
weights. Two **ideal shaft-axis hinges** connect the configured grip sites.
They keep each hand's configured local shaft axis aligned with the shaft and
allow only axial spin. Hold diagnostics default to local +Y; the palms-down
lifting profile uses mirrored local -X/+X axes.
Each hinge uses two point constraints separated by 0.10 m along the shaft,
without adding mass or generalized coordinates. They transmit unlimited force
and transverse torque, but no torque about the free shaft axis:
this is not a finger, grasp-strength or slip model. Only bar contact with the
anonymous hand spheres is excluded; other robot/bar and ground collisions stay
enabled. No root, foot or bar is attached to the world.

The sweep compares native position PD against the same PD with constant static
feedforward inside the existing torque clamps. Both the original individual
motor and combined joint limits stay enabled. Neither controller is a learned
policy or an active whole-body balance controller. Failure of this diagnostic
does not alone establish that a payload is mechanically impossible.

Each case stops at the configured displacement, tilt, grip-gap, joint-range or
collision threshold, or on a solver warning. Reports include motor torques,
time above rated torque, motor saturation, requested versus applied joint effort,
joint clipping, grip forces and body transforms. Initial grip/contact validation
runs against the actual selected model. There is no thermal, current, power-board
or real hand-strength validation.

## Existing locomotion policy

`evaluation.barbell_locomotion` replaces fixed lower-body targets with the existing
Rust consumer's `--telemetry` inference path. The same grip pose is held by arm
PD; there is no added gravity feedforward. A no-bar baseline precedes the loaded
cases. The robot is free to move its body and feet to balance.

Use a local `PolicyHoldConfig` YAML with `policy`, `source` and `binary` paths,
`observation_delay_steps`, `bar_robot_contact`, and a nested `hold` document with
the fields from the hold-sweep config. Here `hold.controllers` must be `[pd]`
for the held arms. All walking/torso commands are zero. Keep machine-specific
paths in ignored outputs. Then run:

```bash
uv run python -m bello_mujoco.evaluation.barbell_locomotion \
  --config outputs/barbell-locomotion-config.yaml \
  --output outputs/barbell-locomotion-review
uv run python -m http.server 8768 --bind 127.0.0.1 \
  --directory outputs/barbell-locomotion-review
```

The native binary requires its matching MuJoCo and ONNX Runtime libraries in the
loader environment. `ORT_DYLIB_PATH` selects ONNX Runtime; on macOS the MuJoCo
library directory may also require `DYLD_LIBRARY_PATH`. The Rust consumer requires
the policy's exact clean source checkout and initialized pinned robot submodule;
this experiment does not rewrite the policy metadata or relax that check.

Feedback uses the canonical IMU site and shared training low-pass filter before
the Rust observation/history/gait assembler. Policy gains, transmission and effort
limits are checked against the simulated model. Recorded settings include the
command/observation delays and both bundled and simulated robot mass. This first
screen uses clean sensors, without CAN/serial transport or state-estimator errors.

Review contacts as well as completion: arm PD sag can bring the shaft against the
pelvis or hips. Contact forces are recorded in world coordinates on the second
geom. `bar_robot_contact: false` is an explicitly counterfactual diagnostic that
allows unphysical overlap to isolate contact assistance; it is not approval of a
collision-free carrying pose. Neither a 30-second balance result nor torque below
peak establishes continuous-duty, grasp-strength or lifting feasibility.

## Next gate

### Rigid-load training preparation

```bash
uv run python -m bello_mujoco.evaluation.barbell_reference \
  --output outputs/barbell-training --standing --video
```

Review `lift.mp4` and `diagnostics.json` before training. A failed screen is
stored as a rejected reference and cannot be loaded for training. If the
endpoint hold fails, shortening the excursion cannot fix it; the generator
reports that blocker and does not produce a standing pose bank. Otherwise it
tries 100%, 75%, 50%, and 25% of the requested excursion, keeping 30 kg.
A zero-excursion candidate is diagnostic only and is not a lift approval.

The current nominal hold fails the 30 kg effort screen (approximately 1.76
of the configured maximum). The 8 cm upward / 3 cm inward candidate reaches
approximately 1.77. Collision and joint-range checks pass, with coincident
ideal grips. These are results for the specified stance, not proof that every
whole-body balancing stance is infeasible. The generated video is kinematic
reference playback, not a successful dynamic rollout. Revise and review the
loaded stance before treating either recipe as training-ready.

`asset_config.barbell` adds one free rigid body after scaling the robot to
49.5 kg. Its explicit mass and inertia are counted once. The same scene builder
is used by training, exported native physics, and the hold diagnostic. Combining
a rigid bar with independent hand payloads is rejected. Shaft-axis hinges permit
relative wrist rotation only around the shaft; bar/body and bar/ground collisions remain enabled,
except for the intended hand-sphere intersections. These grips still have no
finger-strength or slip limit.

`bello_mini_loaded_standing` retains the 14 leg/waist actions and existing gains,
torque limits, feedback filtering, delay/noise, and modest robot randomization.
`env_config.model_randomization.payload_distribution: standing` samples each
world once: 25% zero load, 25% exactly 30 kg, and 50% uniform 5--30 kg. Episode
resets preserve the world's mass. Unloaded worlds disable both grips and park
the positive-mass prop outside the interaction area; zero load never creates a
singular zero-inertia free body.

A screened `env_config.arm_pose_path` contains coherent Cartesian grip
trajectories spanning +/-2 cm forward, +/-3 cm vertically and +/-10 degrees
wrist pitch, plus unloaded rest excursions. Each transition takes four seconds
using a quintic time curve. Loaded worlds select only grip trajectories;
unloaded worlds can also select rest trajectories. All paths start and end at
the common hold to keep changes between selected trajectories continuous.
The bank is checked against joint/motor target bounds before use. Reset joint
noise and joint-origin randomization that would separate the grips are rejected.

Actor observations include measured positions/velocities for all robot joints
and the commanded held-arm joint targets, with the existing observation scales,
filtering and latency. Payload mass is not an actor input. Actions remain the
original leg/waist controls. Rust telemetry requires an explicit `arm_target`
for this layout and overrides the held motor targets within the original target
limits. Walking layouts are unchanged. The hardware controller has no loaded
handover/arm-target producer and rejects such a missing command.

The standing bank is copied into checkpoints as `standing.npz`; its checksum
protects resume and evaluation from a changed bank. Ordinary motion checkpoints
already bundle their reference. The fixed object configuration is saved in the
asset metadata and exported `physics.xml`.

`bello_mini_barbell_motion` layers on Mini BeyondMimic with whole-body actions
and a fixed configured bar mass (currently 15 kg), independent of robot
mass/inertia randomization. References
carry explicit robot joint addresses and free-body quaternion addresses.
Planar alignment and rigid root reset perturbations move both robot and bar;
independent joint perturbations are rejected for this task. Upstream GMR is
unchanged.

The motion profile now selects `outputs/barbell-15kg/handover_cycle.npz`.
Its middle section follows seconds 1.0-7.2 of the supplied demonstration
(`https://www.youtube.com/watch?v=PjY1rH4_MOA`): floor pull, squat clean catch,
stand, dip and drive, split jerk, foot recovery, and overhead finish. Original
speed is retained at the user's request. The earlier 36-second press-and-return
reference changed the demonstrated movement and is no longer selected.

The full cycle lasts 20.2 seconds (4,041 frames at 200 Hz): approved handover
hold (0-2 s), deadlift descent to the clip start (2-6 s), original-speed clip
(6-12.2 s), lowering from overhead to handover (12.2-18.2 s), and handover hold
(18.2-20.2 s). The accepted hinge reference is copied unchanged through the overhead finish
at 12.2 s, including positions and velocities. The return now lowers close to
the front rack with a knee bend, briefly settles, lowers toward the thighs with
a knee/hip bend, and stands into handover by 16.9 s. Shaft-axis hinges remain
enforced throughout; there is no supporting bar/body contact. Start and end match
the approved handover pose exactly, with zero endpoint velocity. Both hinges
remain active throughout; these are loaded handover poses, not a physical hand
release/acquisition controller.

Review `outputs/barbell-15kg/handover_cycle.mp4` for the complete sequence.
Its poses and timing match the approved 30 kg lowering revision exactly.
All saved poses and three interpolants per interval pass grip, axis, collision,
and joint/motor position checks. Maximum off-axis error is 0.000012 degrees
at saved frames and 0.00164 degrees between frames (previously up to 86 degrees).
Maximum interpolated anchor gap is 0.0521 mm. Arms remain mirrored, the bar stays
level, and the intentional split stance is retained. These are sampled checks,
not continuous collision certification or a successful dynamic rollout.

The fixed-15 kg dynamics screen still fails: 161 of 1,011 samples have no accepted
solution (including 3 numerical solver failures); solved samples reach 3.75 times
configured effort limits. The handover hold is below peak limits at 0.948, with
little nominal margin. The dip at cycle 9.30 s is acceleration-limited, while
parts of the split transition have no ground contacts and inconsistent free-flight
dynamics. The overhead lowering also contains statically overloaded poses.
See `phase_audit.json` for static versus dynamic checks by phase. Three
constraint-preserving smoothing trials also failed the dip and airborne checks;
none replaced the approved reference (`smoothing_probe.json`).

Dynamics use the authored float64 positions, velocities and accelerations;
saved-frame and interpolation geometry checks use the actual float32 archive.
A raw float32 derivative-sensitive solve produced a pathological 9.82 s result;
the float64 audit correctly finds no accepted solution there. Its raw report is
retained separately. Screening uses five independent hinge reaction components
per hand: three forces and two transverse moments, with no axial moment.
The archive retains `screen_passed=false`. The 15 kg profile explicitly sets
`env_config.allow_unscreened_reference=true` for exploratory simulation training
and checkpoint evaluation. This relaxes only the exact-reference screen gate:
clip/scene mass matching, physical torque limits, grips, collisions, and reference
speed remain unchanged. Other profiles default to rejecting failed screens, and
the Rust deployment consumer still rejects them. The exploratory setting and
failed screen are preserved in checkpoints; this is not a feasibility approval.
The initial pilot uses 1,024 worlds, 100 million steps, and checkpoints every
100 updates. Success must be assessed from learned rollouts.

The existing RTX 5090 passed a 256-world MJX-Warp synthetic 15 kg hinge test:
axial spin stayed free and maximum anchor separation was 0.108 mm. This verifies
the attachment mechanism, not full-robot motion tracking or PPO throughput.
No GPU was rented. Full-robot dynamic validation remains pending a feasible
reference. The standing distribution remains 0-30 kg and is unchanged.

The earlier point-grip reference and source-comparison video remain under
`outputs/barbell-source-manual/`; its unconstrained hand rotations are superseded
by the hinge reference. The corrected overhead alignment is preserved closely
while solving the stronger grip conditions. The 9 cm hand-local -Z offset,
40 cm hand spacing, fixed object mass, gains, effort limits and collision
exclusions are unchanged. These ideal hinges do not establish a measured palm,
finger grasp, friction capacity or slip limit. Reproduction scripts, optimized
knots, diagnostics and resolved configuration accompany the new reference.

Loaded training rejects unintended bar contacts above 1 N immediately (a
conservative rule that also rejects sustained support). Motion reports this as
`done/barbell_contact`. Grip separation and shaft-axis errors are diagnostics
only and never terminate motion training or policy evaluation. PPO logs means
and maxima of all `barbell/` metrics. Native hold diagnostics
expose `max_grip_axis_error_rad` (default 0.1 rad) and reject misaligned starts.
Diagnostics expose grip errors, resultant constraint forces, transverse grip
torque, shaft-axis error and unintended contact force alongside existing fall, foot-slip, tracking and
torque metrics. Constraint forces remain ideal-attachment forces.

After the references pass and are reviewed:

```bash
uv run --group training bello-train experiment=bello_mini_loaded_standing
uv run --group training bello-train experiment=bello_mini_barbell_motion
```

These commands are not currently cleared for a long run. Paid rentals require
fresh approval, and hardware deployment has a separate commissioning gate.
Do not increase gains, limits, or lower the motion load to bypass a failed screen.

### Validation status and remaining gates

Synthetic tests exercise mass accounting, fixed/randomized loads, grip
activation, coordinate mapping with an object before the robot, coherent resets,
and Python/Rust arm observation parity. Run the GPU grip/throughput test on an
existing CUDA host before any long training:

```bash
JAX_PLATFORMS=cuda uv run --group training pytest -m integration -s \
  tests/test_barbell_training.py::test_warp_hinge_grips_world_loads_and_throughput
```

This synthetic throughput check is followed by a full robot rollout benchmark;
it is not a substitute for one. GPU throughput has not been measured on the
Mac. The native free-base PD hold sweep at 5, 10, 20 and 30 kg reached the 8 cm
bar-displacement stop in approximately 0.21, 0.21, 0.20 and 0.19 seconds. That is
an open-loop baseline failure, not a learned-policy result.

No trained policy exists for the new observation/object contract. Once the
feasibility and reference-review gates pass, evaluate standing at 0/5/10/20/30 kg
across arm poses and motion at its selected fixed load across robot/sensor variations.
Record falls, foot slip, grip gap/force, unintended support, tracking error,
torque exposure and oscillation, then follow with HIL. Those policy evaluations
and HIL remain pending. Randomization does not remove torque, thermal, power or
real grasp limitations.

### Full-bar validation

Confirm the actual grip point and palm orientation, then establish a balanced
stance and loaded hold with credible torque margin. Do not treat a failed hold
as permission to raise effort limits. Only then proceed to a slow loaded
reference and a training task. Keyboard handover and loaded start/stop control
need their own explicit design before a physical trial. This diagnostic does
not implement a deployable clean-and-jerk controller.

Algorithm-only validation:

```bash
uv run pytest -q tests/test_statics.py tests/test_barbell_hold.py tests/test_barbell_locomotion.py
```

These tests use synthetic mechanisms, not snapshots of robot-specific YAML.

## Palms-down lifting experiment

The fixed-15 kg lifting profile now has explicit mirrored shaft and palm axes;
see [training configuration](training_configuration.md#rigid-barbell-profiles).
The receiving pose places both palms over the shaft facing down. The complete
lift is retargeted, rather than adding a wrist offset to an old learned policy.
Geometry checks cover grip coincidence, shaft alignment, body collisions and
joint/motor position limits, including interpolated frames. The inverse-dynamics
screen separately reports effort overload and missing contact equilibrium.

The 40 mm palm-normal and 25 mm finger-direction grip offsets remain provisional
sphere-proxy assumptions. Ideal hinges do not establish physical grip strength.
Training this reference does not update the live robot, its standing policy, or
its handoff targets; those require a matching evaluated policy and integration.
