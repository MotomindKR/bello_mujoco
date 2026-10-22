# SuperMimic port plan for Bello Mini

Build a Mini-native motion tracker with a learned dynamics model and an
imagination-based actor-critic trainer in JAX. Reuse the repository's canonical
robot compilation, retargeted motion data, motor constraints, native evaluation,
and Rust control lifecycle. The first deliverable is a policy that tracks a
small, varied Mini motion set in native MuJoCo. Scale the corpus and commission
hardware only after that loop works.

This is the implementation plan, dated 2026-10-22, for
`~/Developer/bello_mujoco-supermimic` on `feat/supermimic`. The starting upstream
commit is `e3d4b9b`; the pinned stack commit is `29d4d9bc`.

The contracts, analytic kinematics, replay, learned dynamics, imagined actor and
critic updates, collector, checkpoint/resume, native evaluation, ONNX export,
Rust prerecorded telemetry codec, and dataset-driven Vast route are implemented.
The maintained commands and configuration contract are in
[the SuperMimic workflow](supermimic.md). Synthetic CPU and native Mini checks
validate numerical behavior.
A CUDA smoke run, trained-actor JAX/Rust parity, and full checkpoint restoration
have passed on a rented Vast RTX 4090. The first bounded learning pilot and its
native evaluation are recorded in [the workflow](supermimic.md#first-cuda-pilot).
Matched PPO comparisons, corpus-scale learning curves, seed studies, live
tracking, and physical commissioning remain experimental milestones below.

## Method and initial scope

SuperMimic alternates simulator collection, fitting a differentiable world model,
and improving the policy through imagined rollouts. The simulator supplies data;
gradients pass through the learned model. Its critic bootstraps imagined returns.
The [paper, sections 3 and appendices A through D](https://arxiv.org/pdf/2610.28480)
is the method reference.

Use the [unofficial implementation at commit a1508d4](https://github.com/zita-ch/supermimic_vibed/tree/a1508d42c9d1699a52e8233ab16d08c488441ab9)
to inspect concrete implementation choices and construct numerical fixtures.
Its PyTorch/Isaac Lab integration, G1 dimensions, and undocumented choices need
individual review. Record any borrowed code with its MIT attribution; do not
vendor its simulator stack into the training path.

The initial task is offline full-body reference tracking on flat ground, trained
from scratch on `bello_mini`. Add `task=supermimic` and a
`bello_mini_supermimic` training profile. Give it its own immutable configuration,
policy, optimizer state, replay buffer, checkpoint contract, and runtime codec.
The existing PPO loop is structurally different and should remain a separate
trainer. Share small physical and data primitives when their semantics match.

Prerecorded motion playback comes before live PICO following. A velocity-command
encoder, partial-body goals, terrain, object interaction, and barbell handling
are later extensions. Each adds a separate control or dynamics problem and needs
its own experiments.

## Existing foundations and required changes

| Area | Existing implementation | Porting decision |
| --- | --- | --- |
| Robot and inertials | `models/compiler.py`, `sim.py`, pinned Mini URDF | Keep the canonical description and collision model. Validate the compiled model at setup. |
| Joint and motor limits | `envs/actions.py`, `models/target_limits.py` | Share projection and step limiting across collection, imagination, evaluation, and export. |
| Multi-clip data | `sonic/corpus.py`, `sonic/reference.py` | Reuse the checked clip format, provenance, splits, and boundary-safe bank. Extract task-neutral code only where it actually becomes shared. |
| Offline retargeting | `sonic/amass.py`, canonical GMR profile | Start with existing Mini retargets after checking their recorded model and preparation identity. |
| Simulation | `envs/motion_tracking.py`, `envs/sonic_transfer.py` | Reuse stepping and feedback primitives; implement an explicit SuperMimic task state. |
| Differentiable kinematics | Native motion views exist, but no dedicated analytic FK module for imagination was found | Add JAX FK and body-velocity reconstruction from compiled model arrays. |
| Optimization | `training/ppo.py`, `training/runtime.py` | Add a model-based trainer. Reuse logging and checkpoint utilities that do not assume PPO state or fixed rollout length. |
| Evaluation | `evaluation/motion.py`, `evaluation/sonic_transfer.py` | Reuse native simulation and video resources; add the new observation and policy runtime. |
| Export and control | `deployment.py`, `rust/bello-policy` | Add an explicit bundle task and codec; retain the runner's ownership, limits, and failure handling. |
| Remote jobs | `training/vast.py`, `training/vast_remote.py` | Add a dataset-driven task route without SONIC's pretrained-checkpoint requirement. |

The current SONIC task has a G1-shaped pretrained backbone, an FSQ tokenizer,
LoRA adapters, and a historical observation reshape. SuperMimic should operate
directly in Mini joint order. Its model dimensions come from the selected model,
with an initialization check against the recorded contract.

## Robot and controller contract

### Coordinates and state

Start with the full-body Mini motion configuration and the 14 tracked body roles
in `robots/bello_mini_motion.yaml`. Resolve bodies, joints, free-joint addresses,
and actuator coordinates by name. The current Mini layout has 28 controllable
joints; tests should use synthetic layouts rather than hard-code that count.

Define the physical state as root pose, root linear and angular velocity,
logical joint positions, and logical joint velocities. Keep explicit adapters
between MuJoCo free-joint velocity conventions and the world-frame link-origin
twists used by FK and the model. Document WXYZ quaternion order, rotation
direction, metres, radians, seconds, and the pelvis reference frame.

The learning state must also carry the previous accepted motor target and the
controller memory needed by the configured delay and feedback path. Observation
history, reference alignment, episode identity, and reference clock are separate
task state. A physical state alone is insufficient to initialize an imagined
rollout when its action depends on delayed or filtered feedback.

### Actions and timing

Use 50 Hz control with four 5 ms physics substeps as the initial timing contract.
Retain Mini's configured gains, effort limits, differential transmissions, and
joint/motor/step envelopes. Validate this timing against the target runtime.

Define the logical proposal as the neutral joint target plus a scaled action.
The existing `0.25 * effort / stiffness` helper is a useful starting scale;
validate its interpretation for coupled coordinates against the compiled motor
model before fixing the recipe. Transform the proposal to motor coordinates,
project it into the feasible envelope, then limit the step from the previous
accepted target. The delayed actuator path receives that accepted command.

Keep this command path inside the differentiable imagination graph. Its actual
piecewise derivatives should be tested away from boundaries. Log how often
projection or slew limits intervene. An identity gradient through a constrained
command would teach the actor dynamics that the robot cannot execute.

The learned transition should consume the command information actually driving
the simulated control period, including previous/new accepted targets and any
configured substep delay. Store both the proposed policy action and accepted
motor targets in collection records so this correspondence can be audited.

### Randomization and feedback

Expose the actual sampled physical parameters as context to the dynamics model
and privileged critic. Begin with a small, explicit set: friction, calibration
offsets, and torso centre-of-mass variation. Choose ranges from Mini's measured
uncertainty. G1-sized offsets are not Mini calibration evidence.

Every subsequently enabled random variable affecting dynamics must be recorded
or supplied as an input: mass, inertia, actuator strength, gravity, gains, delay,
or external forces. Model samples currently generated per world must remain
associated with that world in replay; do not describe them as per-episode samples
unless their sampling lifecycle is changed deliberately.

Disable the current SONIC profile's unobserved root-velocity pushes during the
initial fit. Test disturbance recovery in native evaluation. A later training
experiment may supply disturbance context or adopt a different model treatment.

Implement observation noise, delay, and low-pass filtering using the same
recurrences as collection. Preserve their memory when starting imagination.
Match the filtering performed by hardwared; adding another runtime filter would
change the trained observation path.

## Actor and reference interface

Use a deterministic motion encoder and action decoder trained directly for Mini.
Freeze the actor's input layout before the first training run:

| Input | Proposed initial representation |
| --- | --- |
| Joint feedback | Logical joint position relative to neutral and velocity |
| IMU feedback | Base angular velocity and projected gravity |
| Previous action | Previous policy action, with proposal/accepted-target semantics documented separately |
| Reference orientation | Six-dimensional current-reference orientation relative to the estimated root orientation |
| Actor memory | Last 10 samples, oldest first, initialized from the first valid sample on entry/reset |
| Motion goal | Ten future frames containing reference-relative root displacement/orientation, root twists, and Mini joint positions/velocities |

The base proprioceptive sample is `3N + 6` values. Including the proposed
reference-orientation term makes it `3N + 12`, or 96 for the current Mini model.
The privileged physical state and context are not actor inputs. The extra
orientation term follows a choice in the
[unofficial MDP implementation](https://github.com/zita-ch/supermimic_vibed/blob/a1508d42c9d1699a52e8233ab16d08c488441ab9/supermimic/mdp.py),
and should have an explicit ablation against the base proprioceptive layout.
Validate heading initialization and drift using the actual hardware orientation
estimate before relying on that feature.

Propose future offsets of 1 through 10 control steps, or 20 through 200 ms. This
matches the consecutive-frame choice in the
[unofficial motion implementation](https://github.com/zita-ch/supermimic_vibed/blob/a1508d42c9d1699a52e8233ab16d08c488441ab9/supermimic/motion.py).
The paper specifies ten future frames but leaves details of their encoding to
the implementation. The existing SONIC assembler instead uses 0.1-second
spacing. Do not inherit that spacing implicitly.

For a concrete starting goal, express root displacement and orientation in the
current reference heading frame, root twists in their declared local frame, and
joint values in Mini order. This gives `15 + 2N` features per future frame, or
710 total for the current model. Persist feature ordering, scaling, offsets,
alignment, and end-of-clip clamping in checkpoint and bundle metadata.

## Implementation milestones

### 1 Define contracts and prove kinematics

Add `supermimic/config.py`, `state.py`, `kinematics.py`, `observations.py`, and
`control.py`, plus synthetic tests. Use frozen configs and registered JAX
dataclasses for array state. Reject unknown fields, incompatible dimensions,
invalid gains, missing links, and unsupported randomization before compilation.

Build FK from the canonical compiled tree, joint axes, fixed transforms, and
effective model parameters. Compute tracked link poses and twists analytically.
Do not differentiate a MuJoCo physics step to obtain imagined kinematics.

Compare against native MuJoCo over valid sampled configurations, velocities,
root rotations, nonzero body offsets, and differential joint configurations.
Check link-origin versus centre-of-mass velocity carefully. Exercise the actually
randomized geometry if geometric randomization is enabled.

**Exit gate:** pose and velocity parity pass within stated float32 tolerances;
directional finite differences agree with JAX derivatives, including near zero
rotation increments. A rigid XY translation/yaw rotation produces the expected
equivariant result. Action projection and command-delay fixtures also agree
with native stepping. Start with position errors below `1e-5 m` and orientation
errors below `1e-4 rad` for ordinary test poses; investigate conditioning before
setting broader tolerances.

### 2 Collect transitions and implement replay

Add `envs/supermimic.py` and `supermimic/replay.py`. Share the existing physics
transition primitives rather than copy the complete SONIC environment. Collect
with the current actor plus explicit exploration noise and record pre-reset
terminal successors.

Use a fixed-capacity device buffer indexed by time and environment. Record state,
successor, action, accepted command, context identity, reference frame, episode
identity, reward, termination, timeout, and compact feedback/controller state.
Store one actor sample per time step and gather its history by index; do not
duplicate a ten-frame stack in every record.

Sample only windows whose valid prefix belongs to one episode, context, and
reference clip. Terminal transitions remain learning examples; post-reset
transitions must be masked out. Separate actual failures from clip exhaustion
and time limits. Define short-clip and buffer-warmup handling without unbounded
rejection loops. Keep reference arrays as explicit JIT arguments.

**Exit gate:** synthetic tests cover wraparound, partially filled buffers,
terminal successors, timeouts, changing context, short clips, padded history,
and deterministic sampling. A small MJX-Warp collection run produces finite
records whose command and successor timing matches native MuJoCo.

### 3 Fit and validate the dynamics model

Add `supermimic/model.py` and the model-update portion of
`training/supermimic.py`. Start with a deterministic residual model with invariant
inputs and local-frame state increments. Use a rotation update on SO(3), with
stable small-angle derivatives and explicit quaternion normalization at storage
boundaries.

Apply dimensional normalization from simulator data and component-specific
variance floors. Keep model-input and target statistics fixed within each
optimization calculation. Predict mechanical state; advance known controller
and feedback recurrences analytically. If a residual actuator model becomes
necessary, introduce it as an explicit experiment with its own validation.

Train autoregressively on logged command windows with episode masks. Freeze
held-out transition sets by source motion and context, rather than randomly
splitting neighbouring replay frames. Compare against persistence and simple
velocity-integration baselines. Include foot landing, lift-off, turns, and limit
intervention in diagnostics.

**Exit gate:** report errors at 1, 8, and 32 control steps, separately by state
component and contact regime. In a fixed-action replay test, use a provisional
32-step p95 height/tilt error budget below 20% of the configured failure margins.
The model must beat the simple baselines and remain finite on the pilot suite.
Fix thresholds before the learning comparison and retain failure-case videos.
Passing prediction checks permits a learning pilot; it does not establish
closed-loop policy performance.

### 4 Implement imagined actor and critic updates

Add `supermimic/imagination.py`. Initialize each imagined rollout from a valid
replay state with its actual controller memory, actor history, alignment, and
context. Advance the goal clock explicitly and keep all reference gathers within
the same clip.

Implement analytic reward and observation functions shared with simulator
evaluation wherever possible. Use a hard failure predicate for simulator
episodes and a separately configured smooth survival function for imagination.
Test both around their boundaries. Mini height and tilt limits must be chosen
from Mini geometry and acceptable tracking error; the reference tilt quantity
is a gravity-component difference, not an angle in radians.

For the initial reward, give body position/orientation and linear/angular
velocity tracking weight 1 each, root position/orientation weight 0.5 each,
action-rate cost weight -0.1, and soft joint-limit cost weight -2, integrated
over the control period. These are starting values from the
[reference MDP code](https://github.com/zita-ch/supermimic_vibed/blob/a1508d42c9d1699a52e8233ab16d08c488441ab9/supermimic/mdp.py).
Retain physical self-collisions. Initially omit SONIC-specific five-point,
anti-shake, contact-force, and ankle-acceleration reward additions; introduce
Mini-specific changes as named experiments with matched baselines. Track
effort, contacts, and saturation as evaluation diagnostics throughout.

Differentiate the actor objective through state, command constraints, dynamics,
reward, survival, and terminal value. Hold dynamics and critic parameters fixed
during that update while preserving their derivatives with respect to inputs.
Stopping the gradient on their outputs would remove useful policy gradients.
Fit the critic to detached imagined-return targets computed once per rollout.

**Exit gate:** tiny deterministic fixtures verify reward indexing, survival
products, bootstrap discounting, and lambda returns. Finite-difference actor
gradients agree on a short smooth system. Actor updates leave model/critic
parameters unchanged; critic updates leave actor/model parameters unchanged.
All gradient and optimizer states remain finite through a small joint loop.

Make replay readiness and model-only warmup explicit. Start actor optimization
after the required valid windows exist and the initial model diagnostics pass;
continue updating the model throughout joint training. Treat a fixed warmup
iteration count, if used, as a recorded implementation choice.

Before the pilot, export an initialized full-size actor for an early timing
measurement on the intended inference device. This can reveal a required
architecture change before a costly run. Milestone 8 still validates the trained
bundle and complete control path.

### 5 Add motion curriculum and complete experiment state

Implement `supermimic/sampling.py` with return estimates for one-second temporal
bins, a uniform component, an emphasis on poor-return bins, and a retained
history of difficulty. Give its estimate update and historical accumulation an
explicit contract. Do not reuse SONIC's failure/exposure counters as if they
measured return.

Test the proposed 50/40/10 reset mixture and treatment of shortened final bins.
Held-out bins have zero training probability. Keep the sampler shared across
environments and include it in checkpoints.

Checkpoint actor, model, critic, all optimizers and normalizers, RNG streams,
actual simulator-step counters, replay contents/cursor, per-world parameters,
environment/controller/history state, and sampler statistics. Dataset bytes stay
in their immutable artifact. Resume checks the exact clean Git commit, dependency
pins, dataset identity, feature layouts, and optimization contract.

Collection may alternate two and three vectorized steps per iteration; derive
environment-step counts from actual collection, not PPO's fixed rollout formula.
Choose checkpoint cadence using measured write/upload cost and tolerated lost
work. A weights-only load cannot be presented as a full resume.

**Exit gate:** interrupted and uninterrupted synthetic runs agree; a same-backend
integration continuation agrees within a declared numerical tolerance. Changed
corpus, action semantics, delay, or layout fails resume validation.

### 6 Demonstrate native MuJoCo learning on a pilot corpus

Create a small immutable corpus of approximately 12-40 reviewed Mini clips:
standing weight shifts, walking, turns, arm motion, crouching, and moderate
dance. Use source-sequence/subject separation for held-out motions. Screen motor
and joint feasibility, foot geometry, velocities, and retarget quality before
training. Introduce ground recovery and high-impact motion only after checking
collision geometry and native contact behaviour for those cases.

Train a single easy clip first to debug the complete loop, then the varied pilot.
Start with the full observation/command path and nominal physical parameters;
add the defined physical randomization after prediction and learning checks
pass. Use explicit smoke settings for reduced model sizes or horizons.

Add `evaluation/supermimic.py` and `supermimic/runtime.py`. Evaluate full clips
from their first frame with deterministic actions in native MuJoCo. Keep a fixed
nominal protocol and a separate fixed robustness protocol. Evaluation must not
update normalizers, replay, or curriculum state.

Report equal-per-clip completion and failure rates, body/root/joint tracking
errors, foot slip and penetration, peak/RMS motor effort, command saturation,
and limit intervention. Log imagined-versus-simulator return disagreement and
the clips with the largest gap.

Report raw return and a normalized return with a fixed denominator: the maximum
positive tracking reward over the entire planned clip duration. With the initial
weights, that bound is `5 * control_dt * planned_steps`. Include penalties in
the numerator and assign no reward to the unplayed portion after a failure.
Do not divide by the shortened episode length, which could make an early
failure look good. Keep the denominator identical for matched comparisons.

**Exit gate:** provisional pilot targets are at least 90% mean completion on the
easy training suite and 80% on its held-out suite, without persistent limiting
or nonfinite episodes, reproduced over three seeds. These are proposed project
gates, not reported Mini results or paper benchmarks. Reassess the corpus and
controller contract before spending a full training budget if they fail.

### 7 Scale data and compare training efficiency

Reuse the existing AMASS preparation and `sonic-corpus.json` provenance path.
The repository's documented research target is 8 hours of accepted training
motion plus 0.8 hours held out; verify the actual available artifact and its
model/preparation pins before choosing a run. A directory name alone does not
establish that the dataset is complete or current.

Increase corpus size in stages: pilot, roughly one hour, then the accepted full
bank. Preserve held-out source identities. Re-run native evaluation after each
stage instead of inferring progress from curriculum-dependent training return.
Existing AMASS/SMPL-X research-use restrictions continue to apply; this port does
not require changing the dataset source or redistributing motion data.

Add a matched scratch PPO comparison using the same actor architecture,
observation, corpus, command path, reward, termination, and evaluation. The
existing pretrained SONIC adapter is a separate practical baseline; its source
training and different formulation prevent a clean optimizer-only comparison.

Compare wall time, GPU-hours, simulator transitions, peak device memory, and
fixed held-out performance. Include collection, dynamics fitting, actor/critic
updates, compilation, and evaluation costs explicitly. State whether setup time
is included. The G1 paper's compute result is motivation, not a Mini runtime
estimate.

**Exit gate:** consistent held-out gains or lower measured compute to a preset
quality target, with no material regression in effort or constraint intervention.
Keep at least three pilot seeds; choose the number of full-bank runs from their
variance and measured cost.

### 8 Export the actor and add Rust playback

Add a `supermimic` bundle task in `deployment.py` and the Rust metadata/parser
dispatch. Export the actor and frozen actor normalization only. Training uses
the world model and critic; control-time inference uses sensor history and a
reference goal.

Bundle selected reference clips, the exact feature/clock layout, Mini coordinate
maps, gains, action scale, transmission, constraints, history initialization,
and source identity. The runtime must reject a SONIC graph or an incompatible
codec rather than attempt an implicit conversion.

Implement the Rust observation/history/goal codec with entry, completion,
interruption, stale-feedback, and reference-loss behaviour. Share the existing
lease and target-limit path. A buffered live reference requires at least the
future window plus transport allowance; its delay and missing-window policy
must be explicit before PICO becomes an input.

**Exit gate:** compare JAX, ONNX Runtime, Python native evaluation, and Rust on
identical saved telemetry/reference traces. Start with actor tolerance
`atol=1e-4, rtol=1e-4`, and tighter command-space checks against hardware encoding
resolution. Test resets, clip ends, non-50-Hz source clips if supported, projection,
and history reseeding. Measure the intended Jetson/host CPU: target p99 actor and
codec time below 10 ms and total nominal cycle below the 20 ms control period.
Do not assume a large training architecture meets that budget.

### 9 Commission through HIL and supported physical trials

Follow [the existing commissioning gate](hardware.md) and
[Lima/vCAN workflow](macos_vcan.md). First run read-only inventory/preflight, then
isolated HIL for stand-to-motion entry, playback, completion, abort, stale data,
disconnect, timing failure, and shutdown. Run nominal and perturbation suites
with recorded telemetry and video.

Commission physical operation separately with calibrated coordinates and IMU,
verified stop behaviour, and the operator's support arrangements. Begin with
supported standing/weight shifts and short low-speed motion before extending
the allowed suite. Retain hardwared protections and the existing ownership
rules. Hardware acceptance requires both tracking quality and acceptable motor
loads/temperature; training success alone does not establish either.

**Exit gate:** reviewed HIL evidence, satisfactory target-device timing, and
operator-reviewed supported physical results for an explicit motion envelope.
High-impact motion remains a separate expansion of that envelope.

## Starting recipe and compute sizing

Use the paper's numerical settings as a reference configuration and label Mini
departures in the experiment record. Initial reference values are `k=8`, `H=32`,
10-frame history, a 32-value latent, dynamics widths `[2048,2048,2048]`,
encoder/decoder widths `[2048,1024,512,256]`, and critic widths `[256,128]`.
Other reference settings are Adam `2e-4`, gamma `0.99`, lambda `0.95`, and 4096
environments. The [paper tables A.4 and A.5](https://arxiv.org/pdf/2610.28480)
specify the remaining optimization schedule.

Represent model, replay, imagination, actor, critic, sampler, and operational
settings as separate frozen config groups. Add public inputs through Hydra's
existing command documents. Reject irrelevant pretrained inputs for the new
scratch task. Update `configuration.py`, `cli.py`, and Vast task handling
together so local and remote execution select the same recipe.

| Stage | Proposed working scale | Purpose |
| --- | --- | --- |
| CPU tests | Tiny synthetic systems and networks | Contracts, masks, derivatives, return arithmetic, and serialization |
| GPU smoke | 64-256 environments; short replay; small batches | Compile the real MJX-Warp and imagination paths and measure memory |
| Learning pilot | 256-1024 environments; full selected network; progressively larger batches | Establish learning and tune Mini-specific state/context handling |
| Full run | Increase toward 4096 environments after profiling | Measure corpus-scale quality and efficiency |

Preflight memory before allocating the bank and replay. The reference buffer
shape is 4,096,000 transitions. A rough core record of 900 bytes already occupies
about 3.4 GiB; recorded actor/filter memory, reference arrays, training activations,
optimizer state, and compiler temporaries add substantially more. A naively
duplicated ten-frame, 96-value history would add about 14.6 GiB by itself.
Report the exact planned allocation from array shapes and dtypes.

A single Linux CUDA GPU with 48-80 GB is a reasonable first full-capacity target;
smaller GPUs can run the smoke/pilot with explicitly reduced batches. This is a
sizing proposal, not a rental specification or a predicted training duration.
Benchmark a short complete loop before setting a run budget. Use `nix develop
.#training` and the existing uv training dependency group. Keep state integration,
rotations, kinematics, normalization, and losses in float32 initially; evaluate
reduced-precision matrix products only as a measured later experiment.

## Validation and implementation order

Land reviewable changes in this order:

1. Config/state/control contracts, analytic FK, and numerical fixtures.
2. Simulator collection, episode-safe replay, and explicit context capture.
3. Dynamics fitting and fixed-action prediction diagnostics.
4. Imagined actor/critic updates and the complete tiny learning loop.
5. Curriculum, strict checkpoints/resume, local/Vast routing, and native evaluation.
6. Pilot and scaling reports with matched baselines.
7. Actor export, Rust codec, trace parity, and HIL integration.
8. Commissioning evidence for the accepted physical motion envelope.

Run focused CPU tests while implementing each module. GPU integration checks
must exercise JIT/vmap, the compiled MJX-Warp collector, replay updates, and BPTT
through the complete imagined command path. Keep default tests CPU-only and
full dynamics opt-in, following `CONTRIBUTION.md`.

Before merging each coherent implementation stage, run the applicable repository
gate: `uv lock --check`, protobuf validation when relevant, Ruff format/check,
and `./scripts/test.sh`. Runtime changes additionally require the Rust build and
synthetic native-control integration suites. A documentation-only plan does not
require compiling robot dynamics.

Update `docs/commands.md`, `docs/training_configuration.md`,
`docs/training_and_evaluation.md`, and a maintained SuperMimic guide as executable
interfaces land. Keep the plan's proposed commands clearly separated from the
currently supported CLI.

## Risks and decisions before the first pilot

| Risk or decision | Required resolution |
| --- | --- |
| Model exploitation | Fixed-action model diagnostics plus real-simulator checkpoint evaluation; inspect growing imagined/native return gaps. |
| Hidden controller state | Include accepted targets, delay, filters, and actor history in replay/imagination initialization. |
| Coupled motor physics | Validate logical-to-motor scaling and per-motor effort on the canonical transmission. |
| Incorrect rotations or velocities | Native FK/twist parity and directional derivative fixtures before training. |
| Reference heading and future availability | Fix goal offsets and heading alignment; test actual IMU-derived orientation and buffered-reference behaviour. |
| Infeasible retargets | Review pilot videos and report explicit rejection reasons; do not silently relax corpus checks to fill a duration budget. |
| Incomplete context | Reject unsupported randomization; record actual values instead of only a seed or bounds. |
| Hardware-incompatible reward | Keep canonical collisions and constraints; track loads and saturation alongside tracking reward. |
| Excessive memory or checkpoint volume | Measure exact allocations and upload cost before a full run. |
| CPU inference misses 50 Hz | Profile the exported actor on the target device early; test a smaller explicit architecture if necessary. |

Resolve these pilot inputs during milestone 1: the immutable available Mini
corpus, physical machine/calibration revision, nominal controller timing and
gains, Mini failure margins, target inference device, and initial GPU memory.
The code and synthetic tests can proceed before all hardware inputs are ready;
those inputs gate the corresponding measured experiments.

For one engineer, a rough planning allowance is four to six weeks for a tested
trainer, pilot experiments, export, and HIL, assuming working data and GPU access.
Physical commissioning and broad motion coverage depend on measured outcomes
and operator availability. After the first CUDA pilot, the next experiments
are longer held-out motions, a larger screened corpus, multiple seeds, and
matched simulator/compute comparisons against the existing PPO tracker.
