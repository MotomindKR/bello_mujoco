# Load-aware contact estimator: default and validation

Status: **sole software default; not approved for physical deployment**.
Both Bello and Mini profiles require the load observer. The kinematic-only runtime
path and its opposite-foot speed settings have been removed. Startup bootstrap
kinematics, confidence smoothing, anchor lifecycle and reconnect behavior remain.
No physical robot, systemd service, policy weights or training settings were changed.

## Implementation and validation

The pinned stack adds a measured-torque momentum observer, whole-body feedback
alignment, and world-frame sole-angle weighting of horizontal anchor information.
Vertical support remains available on tilted toes/heels. It reuses the native
robot model and tendon mapping rather than implementing a second dynamics model.
See the [estimator guide](../deps/bello-stack/components/bello-state-estimatord/README.md)
for required configuration and limitations. The observer, native physics builder,
model configuration and canonical robot inputs all live in bello-stack; the
estimator does not read this training repository or any policy artifact.

The estimator owns one bounded MuJoCo worker. Missing whole-body feedback
withholds poses; it never substitutes commanded torque or silently falls back to
kinematic mode. Transport reconnection preserves odometry and resets observer
history. Missing or null `load_observer` config and obsolete
`contact_world_speed_*` fields are explicit configuration errors.

Initial candidate validation in Linux/aarch64 Lima (before removing the old mode):

- 64 focused unit tests passed, including synthetic dynamics/transmissions,
  torque alignment, momentum residuals, rank-deficient wrench inversion,
  world-frame sole orientation, vertical-only correction, and daemon reconnects.
- Strict Clippy and formatting checks passed.
- Native policy, hardwared, estimator and MuJoCo emulator ran together over
  vCAN and emulated EBIMU UART; emulator truth was recorded separately for scoring.
- Both estimators were exercised with squat, balance, lunge, R&B, and a
  50-second standing/forward/lateral/turning sequence.

## Historical candidate HIL protocol

The final matrix used the existing deployment runner at `df2e4c1`, baseline stack
`f256532`, and candidate estimator `740f414`. Binaries were rebuilt from those
sources. Existing deployment bundles were unchanged. The candidate uses its own
predicted state and measured motor torque, never the truth pose stream.

Common settings: 2.5-ms native physics timestep, 10-ms actuator-command latency,
100-Hz motor feedback, 200-Hz IMU output, 0.005-rad orientation noise and
0.1-m/s2 acceleration noise. Seed 2 was used for the final matrix; exploratory
rounds used seeds 0 and 1. Live scheduling and initial unpowered settling remain
nondeterministic. The four native processes used separate two-core affinity sets
in the final eight-vCPU VM. There was no concurrent compilation during runs.

The sequence was HOLD, nominal ramp, RUN, release torso guide, five seconds of
standing, then MOTION. Successful clips returned to standing before Ctrl-C/IDLE.
The runner reports operator interruption as exit code 1; this is not classified
as a failed clip. Each attempt kept its terminal logs, policy-step log, pose
streams, generated configs and lifecycle events.

The harness used a 0.01-rad startup-feedback allowance because unpowered
simulated wrists can settle beyond MuJoCo's soft stops. This was identical on
both sides and did not widen commanded motor/joint/step limits. All runtime
termination, freshness and daemon safety checks remained enabled. Queue/deadline
and stale-telemetry failures were retained and separated from motion terminations.

Metrics align process-local source clocks to the collector clock using arrival
time and reported age; residual scheduling/transport timing uncertainty remains.
One initial XY origin offset is removed, never a Z offset or time-varying drift.
Joint tracking uses measured motor-to-joint positions versus reference positions,
not the estimator's internal error. Paired metrics use the common surviving
duration, alongside full-run completion/failure outcomes. Locomotion consumes
hardware IMU/encoders rather than estimator pose: its estimator comparison is
passive, not evidence that this change improved the walking controller.

## Historical comparison and deployment gate

The 2026-09-21 comparison retained 33 attempts, including startup/transport
failures. Final-matrix shared-duration RMSE is shown below as baseline ->
candidate. R&B uses the bounded retry pair; locomotion uses the completed
baseline and candidate retry. These are observations, not statistical bounds.

| Task | Shared seconds | Height estimate (cm) | Position estimate (cm) | Joint tracking (rad) | Anchor-height tracking (cm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Squat | 15.13 | 0.10 -> 0.11 | 2.03 -> 1.12 | 0.088 -> 0.087 | 1.35 -> 1.27 |
| Balance | 8.21 | 1.13 -> 1.31 | 12.65 -> 12.01 | 0.107 -> 0.107 | 1.50 -> 1.57 |
| Lunge | 12.04 | 3.58 -> 0.14 | 68.88 -> 2.00 | 0.157 -> 0.105 | 2.30 -> 1.37 |
| R&B, interrupted | 14.66 | 3.95 -> 0.25 | 316.41 -> 11.27 | 0.172 -> 0.162 | 3.03 -> 3.49 |
| Locomotion, passive | 50.01 | 0.36 -> 0.36 | 21.27 -> 9.71 | - | - |

Final outcomes: both squat runs completed; both balance runs stopped on tilt;
baseline lunge stopped on end-effector height at about 12 seconds while the
candidate completed about 32 seconds. Both R&B retries stopped on transport
errors, not successful completion. The candidate completed R&B in the earlier
seed-1 round, but that is not a repeatability pass. Both locomotion variants
completed the 50-second script after the candidate's initial stale-telemetry stop
was retried.

Run variability matters: the earlier seed-1 lunge comparison had *worse* candidate
position RMSE (about 247 cm versus 141 cm over the full clip), despite lower
height error. Final squat was effectively tied in height although earlier
baseline squat attempts terminated. R&B anchor-height tracking also worsened
slightly in the shared retry window despite better state estimation.

The local `bello_mujoco-vcan/outputs/load-aware-hil/` artifact directory retains
`protocol.json` (binary/bundle hashes), every attempt's logs/configs,
`summary.json`, `paired.json`, per-attempt `metrics.npz`, and `comparison.png`.
These generated artifacts and temporary harnesses are not committed.

At the original 69.9-degree cutoff, balance trips the tilt limit. Some lunge/R&B runs show substantial
horizontal odometry drift despite excellent height estimates. Other attempts
hit vCAN transmission or hardware-telemetry deadlines. Neither lower height RMSE
nor completing one clip clears those failures.

Before physical deployment: resolve the balance/horizontal-drift regression, obtain
repeatable full-matrix HIL results, and validate model/torque/encoder calibration
and timing on supported hardware. The physical squat vibration/drop investigation
is a separate next step; these experiments do not establish its root cause.

The later 80-degree motion-only cutoff lets both old and new estimators complete
balance. Two new-estimator runs ended 2.28 m and 0.93 m from their starts, versus
1.72 m and 2.58 m with the old estimator. Truth-pose control ended 2.3 cm away.
Thus completion alone does not validate horizontal estimation. Standing retains
its separate tilt limit. The native runner's motion YAML example documents the
relaxed cutoff; existing deployment configs must be changed explicitly.

## Sole-default validation and fresh benchmark

Stack `8b404d3` makes the observer mandatory, removes the old contact update path
and unused opposite-foot speed settings, and supplies self-contained Bello and
Mini configurations. The cleanup removes 161 lines net from production Rust
estimator logic; expanded synthetic tests cover the now-mandatory native observer
through absent hardware, stream closure, restart and telemetry stalls.

Validation: 62 focused tests (36 estimator, 26 native MuJoCo) passed in release
mode, strict Clippy passed, and both shipped robot profiles passed native
`--check`. Model/hardware/URDF paths were verified to remain inside bello-stack.
The binary was built in an isolated Cargo target to avoid artifacts from other
checkouts. SHA-256:
`5a954ab0dd03de3fedf9a305bff4227e6246000c3212ea3396f2be4482cb3622`.

The fresh Mini matrix used the shipped Mini estimator config and the final
binary, with the same policy bundles, common runtime binaries as the cutoff
follow-up, seed 2, core affinity and noise/latency settings described above.
Motion tilt was 80 degrees; standing remained at 0.7 rad. No other safety limits
were relaxed. Seven attempts were retained. Full-clip results are below, using
the completed retry when an initial attempt stopped on transport.

| Task | Duration (s) | Height estimate RMSE (cm) | Position estimate RMSE (cm) | Joint tracking RMSE (rad) | Anchor-height tracking RMSE (cm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Squat | 15.10 | 0.11 | 3.34 | 0.087 | 1.38 |
| Balance, retry | 17.77 | 0.58 | 70.41 | 0.115 | 1.78 |
| Lunge | 32.47 | 0.26 | 32.79 | 0.109 | 1.49 |
| R&B, retry | 40.99 | 0.24 | 54.68 | 0.161 | 3.17 |
| Locomotion, passive | 50.03 | 0.38 | 10.78 | - | - |

All completed motions resumed standing. Initial balance failed before the clip
on a vCAN queue-full fault; initial R&B stopped at 27.17 seconds on the same
transport fault. Both bounded retries completed. These are not seven clean
passes, nor a controlled repeatability study. The balance retry still ended
1.16 m from its starting XY location: good height estimates and completion do
not establish good horizontal odometry or physical commissioning.

Artifacts are under the existing `outputs/load-aware-hil/` directory in the
Lima checkout, named `default-TASK-sole80-seed2` and, for balance/R&B, `-retry`.
Configs, logs, truth/estimated poses and per-run scores were retained separately
from previous attempts. All harness-owned services were stopped afterward.
Production Jetson services were not changed or restarted.
