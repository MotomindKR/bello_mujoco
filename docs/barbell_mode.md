# Barbell mode

The barbell coordinator combines standing v2, arm reaching, explicit finger
control, load transfer, and the trained barbell motion. `bello-eval-barbell`
provides the simulation diagnostic; the Rust runner exposes **Barbell** as a
separate console mode when `barbell` is configured in its hardware YAML.

The native coordinator in `rust/bello-policy/src/barbell/` owns the console's
phase, grip, and confirmation rules. The simulation uses `BarbellMode` in
`src/bello_mujoco/barbell.py` and `BarbellHandSession`, which acquires both hands
using the hand server API, selects application targets, renews leases, and rejects lost ownership or stale
feedback. It never reacquires a lost hand automatically. The evaluation runs
the actual gRPC service on an ephemeral loopback port with simulated hand
devices. It accepts no external hand endpoint and opens no CAN devices.

## Sequence and confirmations

| Phase | Body and hands | Advance condition |
| --- | --- | --- |
| Initial stand | Standing v2, arms down, fingers open | Configured settling interval |
| Reach | Standing v2; smoothly move unloaded arms forward | Reach interval |
| Receive | Maintain forward arms and open fingers | `bar_ready` |
| Close | Adjust fingers with left stick; external holder still supports bar | `grasp_secured` |
| Transfer load | Ramp loaded arm preload and remove holder support together | Transfer interval |
| Enter motion | Evaluate both policies on live state; blend targets and gains | Blend interval |
| Motion | Run complete trained clip; selected grip remains held | Clip completion |
| Return | Live motion-to-standing target and gain blend | Blend interval |
| Await support | Loaded standing with selected grip held | `bar_supported` |
| Transfer to holder | Ramp arm preload down as holder takes load | Transfer interval |
| Open | Request open fingers with the holder supporting the bar | `hands_open` |
| Clear | Maintain forward arms | `bar_clear` |
| Retract and settle | Return unloaded arms down under standing v2 | Configured intervals |

A confirmation identifies both its event and current `phase_id`. Stale or
out-of-order confirmations are rejected. Waiting has no implicit timeout that
releases the object or advances to motion. The evaluation's overall time limit
ends the diagnostic as a failure when a confirmation never arrives.

The simulation supplies confirmations explicitly from its virtual holder,
hand/bar alignment, and synthetic finger feedback. `grasp_secured` authorizes
ideal shaft-axis grip constraints; it is not inferred contact sensing from the
real hand server. `bar_supported` represents a cooperating virtual holder ready
to take the weight through the unloading ramp. The console requires the operator to provide these confirmations. Hand position
agreement alone never establishes grasp security or external support.

The arm targets distinguish the reviewed physical receiving pose, its unloaded
self-gravity compensation, and the loaded standing policy's preload. Applying
the loaded target while the hands are empty caused the earlier excessive arm
raise. The coordinator now supplies separate arm and load fractions. The
simulator computes unloaded compensation from the physical pose and blends to
the policy's loaded target only during confirmed load transfer.

## Interruption behavior

A requested stop cancels the cycle; it never automatically opens fingers:

- Before receiving a bar, retract from the current arm target under standing.
- While receiving an externally held bar, wait for `bar_clear` before retracting.
- After closing or during load transfer, preserve fingers and the current load
  fraction; continue standing and wait for a fresh `bar_supported` confirmation.
- During either blend or motion, return toward standing from the current blend
  weight, retaining the selected grip and the current motion reference time. Then
  wait for support.
- After support confirmation, unload before opening; wait for open/clear
  confirmations before retracting.
- Invalid body feedback or dynamics faults latch `FAULT`, preserving the last
  hand/preload intent. The simulation terminates; it does not claim a physical
  recovery. Hardware fault handling remains the hardware runner's responsibility.

Closing a hand session releases its leases, which suspends future hand writes;
it does not command the fingers open or disable their motors. Unit tests cover
phase-specific cancellation and stale confirmations. Successful normal-cycle
simulation does not establish safe recovery from every physical motion pose.

## Reproduce the diagnostic

Configuration is [barbell_eval.yaml](../configs/commands/barbell_eval.yaml).
Supply exact, clean historical source checkouts and the corresponding assets:

```bash
uv run bello-eval-barbell \
  motion_source=/path/to/motion-checkout \
  motion_checkpoint=/path/to/motion-checkpoint \
  standing_source=/path/to/standing-v2-checkout \
  standing_policy=/path/to/standing-v2.zip \
  native_binary=/path/to/historical-native-policy-binary \
  onnxruntime_library=/path/to/libonnxruntime \
  neutral_pose_path=/path/to/neutral-pose.json \
  output_path=outputs/barbell-mode/review
```

The neutral pose document contains `coordinates` and `motor_positions` arrays
in the bundle's actuator coordinate order. It supplies the reviewed unloaded
arms-down pose; all supplied poses are checked against standing target limits.
Standing metadata is read from the exported ONNX bundle, not a separate config.
The complete motion duration must match the checkpoint clip. No policy is
retrained, and neither checkpoint nor pinned descriptions are modified.

The worker loads motion using its original checkout, while the native standing
actor validates its separate original source. Shared observation/filter/limit
helpers must be byte-identical between these two historical sources. Mismatched
sources or helpers fail explicitly. This worker is a diagnostic for the reviewed
historical pair, not a compatibility route in the production runtime.

`mode.*` controls phase timing. `holder_move_seconds`, `receive_wait_seconds`,
`support_wait_seconds`, and `hand_wait_seconds` control the simulated holder's
confirmation timing. `maximum_seconds` bounds the whole diagnostic;
`worker_timeout_seconds` bounds each worker response. Outputs include the input
manifest, policy provenance, dynamics trace, hand command log, confirmations,
phase history, result, and model snapshot. Existing runs are not overwritten.

Render a saved trace without rerunning policies:

```bash
uv run python tools/barbell/render.py outputs/barbell-mode/review
```

This standalone diagnostic renderer uses the host MuJoCo renderer and records
its renderer name. It is separate from the Linux software-only `bello-eval`
workflow. The video explicitly labels simulated hands and ideal grips. It
writes `cycle.mp4`, `contact-sheet.png`, `last-frame.png`, and `render.json`.

The diagnostic uses full closure with its simulated hands. It does not consume
joystick input or model grip force from the console's selected closure percentage.
The simulator's ideal grip and holder must not be treated as hardware feedback.

## Run from the console

Use [config/barbell.yaml](../rust/bello-policy/config/barbell.yaml) as the runner
configuration. Set the standing-v2 and barbell motion bundle paths and their
exact clean source checkouts. `--policy` remains the ordinary standing/locomotion
bundle; Barbell loads standing v2 independently. All three policies must have the
same control period, robot coordinate mapping, and compatible IMU frame.

Export the motion bundle using its checkpoint's original checkout. The bundled
clip must include its free-body coordinate maps, positive bar mass, and passed
reference screen; Rust checks them against the bundled physics model. Generated
bundles and pose documents stay outside Git. The reviewed receiving-pose document
has `coordinates`, `normal`, `unloaded_forward`, and `loaded_forward` arrays in
full motor order. Normal held coordinates must match the primary policy; loaded
held coordinates must match standing v2's trained preload. All poses are validated
against policy and hardware bounds. Do not substitute the loaded target for the
unloaded receiving pose.

```bash
# Read-only hardware preflight; does not acquire hand or body leases.
bello-policy --policy outputs/policies/locomotion.zip \
  --source outputs/policy-sources/locomotion \
  --hardware-config rust/bello-policy/config/barbell.yaml

# Start operator service after commissioning; normal console arming still applies.
bello-policy --policy outputs/policies/locomotion.zip \
  --source outputs/policy-sources/locomotion \
  --hardware-config rust/bello-policy/config/barbell.yaml --arm
```

Start the [hand service](hand_control.md) with calibrated hands. Its configured
rate ceiling must permit `barbell.hands.max_raw_rate_per_second`. Connect the
matching console and runner (input protocol **4**), enter Stand, then select
**Barbell**. With both hands empty, hold **A for one second** to enter. The runner
acquires both hand leases, opens the fingers, blends into standing v2 with normal
arms, and starts the reach. Failed acquisition releases any partially acquired
lease; it never steals another owner's hand.

For each displayed handoff prompt, release A and then hold it for one second:
bar positioned -> grip secured -> holder supports bar -> hands open -> bar clear.
The runner owns the hold timer and phase generation. Holding A across phases
cannot advance a later prompt. Maintain holder support until grip confirmation;
coordinate taking/transferring load with the displayed preload ramp.

At the grab step, fingers start open. Use **left stick up to tighten** both hands
and **down to loosen** them. Centering the stick holds the selected position;
left/right stick movement and the right stick do not affect grip. The normalized
closure is bounded to 0-100%. `barbell.grip_adjustment_per_second` sets the
full-stick adjustment rate (the example uses 0.25, or 25 percentage points per
second). Smaller deflections adjust more slowly, after the controller deadzone.
The hand server still applies calibration and its configured raw slew limit.

Center the sticks, verify the grasp, then hold A for one second. A zero-closure
setting cannot be confirmed as a grasp. Confirmation locks the selected closure
through load transfer, motion, return, and waiting for support. Stick input in
those phases does not change grip or command walking. After a fresh support
confirmation and unloading, the runner opens the hands. Cancellation or controller
loss also preserves the selected grip until the supported handoff. A new cycle
starts with open hands again.

Status shows hand ownership, phase, requested grip percentage, whether grip is
adjustable, preload fraction, motion blend, and interruption reason. Grip percentage
is a requested finger position, not measured force or evidence of a secure grasp.
The final settling interval blends back to the primary standing policy, then
releases hand leases and returns to Stand.

Hold **Y for one second** to cancel, or press Ctrl-C once in the runner while
Barbell is active. Controller loss also requests a controlled return. Grip/load
intent remains until a fresh support confirmation. A second Ctrl-C, process
termination, body fault, or hand-service fault uses the existing body fault cleanup
and suspends hand writes by releasing leases; it never commands fingers open.
There is no automatic reacquisition or resumed cycle after a fault. Physical load
recovery after a body fault still requires the holder/commissioning procedure.

Hand commands and renewals run in a separate task, so RPCs never block policy
inference. `update_seconds` controls renewal, `rpc_timeout_seconds` bounds each
call, `stale_seconds` bounds the last successful hand state, and
`acquisition_timeout_seconds` bounds startup. The server's lease timeout must
exceed stale + update + two RPC timeouts. Fresh hand feedback is not force or grasp sensing.

When policy logging is enabled, every Barbell tick includes `motor_rails`, using
the same asynchronous power-board subscription as standing and other motion
modes. Each available reading includes voltage, signed current, sample age,
stale status, sequence, and connection generation. Missing readings remain null;
they do not mean zero power consumption.

Validation uses synthetic body devices and the real hand gRPC controller on
loopback, covering the full cycle, held-button rejection, cancellation,
controller loss, hand faults, and ownership conflicts. This does not replace the
existing physical [commissioning requirements](hardware.md).
