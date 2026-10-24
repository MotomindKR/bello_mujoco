# Mini motor and world-model audit

The October 25 audit added motor intervention diagnostics, reference dynamics
reports, contact-stratified model validation, and isolated checkpoint checks.
The implementation is in `e57dde1`. Usage and configuration are documented in
[SuperMimic](supermimic.md#reference-and-world-model-audits).

## Trained model measurements

The measured checkpoint is the completed 14,000-update run from clean source
`8cf8cfbbb0e09b7255f61bc9f4129d39f565a442`, using its original gains, self contacts,
actor, normalizers, and dynamics model. The analysis ran from that exact runtime.
The full state checksum was verified; the learner prefix was read directly from
the original Equinox stream without allocating replay. Its actor and observation
and goal normalizers matched `actor.eqx` exactly. Checkpoint files and metadata
were unchanged.

Four training windows were selected before the original actor training. Native
trajectories use deterministic actions, with 400 control steps per window; the
get-up trajectory terminates after 156 steps. The model autoregresses from real
states using recorded executed controls, without recomputing policy actions from
predictions. These are model prediction errors against native trajectories,
not tracking errors against the reference.

| Window | Mean root error, 8 steps | Mean root error, 32 steps | Mean joint RMS error, 32 steps |
| --- | ---: | ---: | ---: |
| Dance | 0.47 cm | 4.97 cm | 2.46 deg |
| Jump | 0.42 cm | 4.03 cm | 2.44 deg |
| Get up | 2.71 cm | 17.60 cm | 15.67 deg |
| Fight/sports | 0.49 cm | 3.80 cm | 2.34 deg |

At 50 Hz, eight steps span 0.16 seconds and 32 steps span 0.64 seconds. Get-up
root error at 32 steps has p95 29.35 cm and maximum 33.45 cm. Its actual native
failure is the height gate at source frame 7656: height error 0.2569 m and
projected-gravity error 0.4767, below the tilt threshold. The dance window also
has an isolated 34.91 cm maximum model root error despite its smaller mean.

This diagnostic uses anchors every four steps: 93 overlapping 32-step windows
for dance, jump, and sports, and 32 for get-up. All predictions are finite.
Every 32-step window contains a ground-contact signature change; none is wholly
airborne or contact-stable. These windows therefore do not isolate contact
transitions as the cause. They show that small short-horizon errors can hide
substantial drift over the actor's imagination horizon. Model exploitation by
the actor remains a hypothesis, not a demonstrated mechanism. The new automatic
checks use their documented bounded sampling defaults, rather than this denser
one-off analysis.

## Reference diagnostics

The reference audit used the current LAFAN training profile at `e57dde1`, including
the adjusted damping and disabled self contacts. It verified all 49 clips in the
central-difference corpus, then sampled 51,409 of the 513,900 frames at stride ten.
No clips or reference values were changed.

The largest literal inverse-torque spikes occur at the waist and hip roll.
Across sampled frames, waist roll exceeds its configured peak in 1.46%, waist
yaw in 1.30%, and the two hip rolls in 0.49% and 0.37%. The maximum motor ratio
is 4.23 in `fallAndGetUp1_subject4`.

However, the median per-clip RMS unexplained root force is 1.11 bodyweights.
The literal kinematic references and configured soft-contact forces do not
supply dynamically consistent support. Consequently these inverse torques are
conditional diagnostics; they cannot certify feasibility, rule out ankle
limitations, or justify removing motions. Loaded contact consistency needs a
separate analysis before interpreting these numbers as actuator requirements.

## Validation and artifacts

Validation passed 986 default CPU tests, two native checkpoint/ONNX/model-audit
integration cases covering both precision settings, formatting, lint, lock, and
generated-protocol checks. CUDA throughput and learning quality of the added
diagnostics still require a fresh training run.

Local reports, the historical analysis driver and its SHA-256, and PNG/SVG
graphs are under
`outputs/supermimic-performance-20261024/implementation-audit/`. The primary files
are `reference-dynamics.json`, `historical-model-errors.json`, and
`world-model-errors.png`. Generated artifacts remain outside Git.
