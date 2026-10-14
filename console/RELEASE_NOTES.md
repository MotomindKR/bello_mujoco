## Bello Mini Console 0.3.6

- Add an explicit `triggerAxisDirection` setting for controllers whose LT/RT axes decrease when pressed. Set it to `negative` on the Xbox Ally controller to prevent released triggers from starting an operator abort; the default `positive` preserves existing controller behavior.
- Keep the runner input protocol at version 4. No runner update is required for this controller fix.

## Bello Mini Console 0.3.5

- Add the configured Barbell mode: standing v2, arm reach, grab, load transfer, motion playback, supported handoff, and arm retraction.
- Adjust both hands with left stick up/down at the grab step. Center holds the selected grip; confirming the grasp retains it throughout motion and return. Show grip percentage, hand ownership, preload, and motion progress in the console.
- Require fresh one-second A holds for each handoff confirmation. Hold Y for one second to cancel and request handoff; cancellation and controller loss retain grip until support is confirmed.
- Requires the matching runner with input protocol 4 and the leased hand service. Update both with the console. Configure the policy bundles and arm poses before using Barbell mode; the release does not include trained policies.

## Bello Mini Console 0.3.4

- Fix the macOS Settings smoke test so the Linux Switch Pro trigger fix can be published. The console behavior is unchanged from the tagged 0.3.3 build; that build was not published after its macOS test failed.

## Bello Mini Console 0.3.3

- Fix Nintendo Switch Pro controller ZL/ZR input on Linux. SDL now uses the kernel event device, which reports reliable press and release events instead of invalid trigger axis values. This restores held abort and fault recovery without changing the runner protocol.

## Bello Mini Console 0.3.2

- Keep locomotion movement intent active during an unfinished L2 + R2 abort hold. The runner enters Fault and zeros movement only after the full one-second hold. This runner change needs the matching source revision on the robot.
- Show Clear fault rejection and recovery errors outside the scrolling fault description. A click during cleanup reports why it cannot proceed, and a held trigger gets a release hint. Fault recovery still requires neutral input.

## Bello Mini Console 0.3.0

- Stream movement intent continuously while mapping gamepad buttons to semantic operator actions in the console. The runner still validates transitions, neutral input, hold timing, movement bounds, and the 250 ms watchdog.
- Make the on-screen abort and Hold to idle buttons stream a held abort action, timed by the runner. Keep Clear fault as a separate neutral-input request and report rejected recovery immediately.
- Requires runner input protocol 3. Install the matching runner before using this console version.

## Bello Mini Console 0.2.9

### Change

- Keep the red X, green Y, and blue Z labels beside their axis tips while rotating either 3D view. The labels no longer jump when an axis crosses the center of the view.

This patch uses the same runner input protocol as 0.2.8. Install the matching DEB or RPM on Linux, or the DMG on Apple silicon macOS. `SHA256SUMS` contains each installer's checksum.
