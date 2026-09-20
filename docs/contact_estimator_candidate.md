# Load-aware contact estimator candidate

Status: **not approved for physical deployment**. The candidate is opt-in; the
shipped robot profiles still use kinematic contact evidence. No physical robot,
systemd service, policy weights, training settings, or safety limits were changed.

## Implementation and validation

The pinned stack adds a measured-torque momentum observer, whole-body feedback
alignment, and world-frame sole-angle weighting of horizontal anchor information.
Vertical support remains available on tilted toes/heels. It reuses the native
robot model and tendon mapping rather than implementing a second dynamics model.
See the [estimator guide](../deps/bello-stack/components/bello-state-estimatord/README.md)
for explicit opt-in configuration and limitations.

The estimator owns one bounded MuJoCo worker. Missing whole-body feedback
withholds poses; it never substitutes commanded torque or silently falls back to
kinematic mode. Transport reconnection preserves odometry and resets observer
history. Shared probability smoothing and anchor lifecycle code serves both modes.

Validation in Linux/aarch64 Lima:

- 64 focused unit tests passed, including synthetic dynamics/transmissions,
  torque alignment, momentum residuals, rank-deficient wrench inversion,
  world-frame sole orientation, vertical-only correction, and daemon reconnects.
- Strict Clippy and formatting checks passed.
- Native policy, hardwared, estimator and MuJoCo emulator ran together over
  vCAN and emulated EBIMU UART; emulator truth was recorded separately for scoring.
- Both estimators were exercised with squat, balance, lunge, R&B, and a
  50-second standing/forward/lateral/turning sequence.

## HIL protocol

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

## Deployment gate

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

Balance still trips the tilt limit. Some lunge/R&B runs show substantial
horizontal odometry drift despite excellent height estimates. Other attempts
hit vCAN transmission or hardware-telemetry deadlines. Neither lower height RMSE
nor completing one clip clears those failures.

Before physical opt-in: resolve the balance/horizontal-drift regression, obtain
repeatable full-matrix HIL results, and validate model/torque/encoder calibration
and timing on supported hardware. The physical squat vibration/drop investigation
is a separate next step; these experiments do not establish its root cause.
