## Bello Mini Console 0.3.0

- Stream movement intent continuously while mapping gamepad buttons to semantic operator actions in the console. The runner still validates transitions, neutral input, hold timing, movement bounds, and the 250 ms watchdog.
- Make the on-screen abort and Hold to idle buttons stream a held abort action, timed by the runner. Keep Clear fault as a separate neutral-input request and report rejected recovery immediately.
- Requires runner input protocol 3. Install the matching runner before using this console version.

## Bello Mini Console 0.2.9

### Change

- Keep the red X, green Y, and blue Z labels beside their axis tips while rotating either 3D view. The labels no longer jump when an axis crosses the center of the view.

This patch uses the same runner input protocol as 0.2.8. Install the matching DEB or RPM on Linux, or the DMG on Apple silicon macOS. `SHA256SUMS` contains each installer's checksum.
