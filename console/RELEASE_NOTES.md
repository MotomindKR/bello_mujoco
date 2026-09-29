## Bello Mini Console 0.2.3

### Changes

- After aborting into damping, the stop button becomes **Hold to idle**. Release and hold it, or L1+R1, for one second to disable damped groups and recover to Idle. Re-enter Hold and Stand without restarting the runner.
- Preserve policy history, previous actions, gait state and command smoothing across Stand/Locomotion transitions and disconnect-to-Stand recovery.
- Change assisted lying startup to **Back-down** (face upward), with bent legs, free ankles, balance confirmation and operator-paced straightening.
- Drop Intel macOS builds. macOS installers now target Apple silicon; Linux keeps amd64 and arm64 support.

### Installers

| Platform | Asset suffix |
| --- | --- |
| macOS Apple silicon | `darwin-arm64.dmg` |
| Linux amd64 | `linux-x64.AppImage` |
| Linux arm64 | `linux-arm64.AppImage` |

`SHA256SUMS` contains the checksum for each installer. macOS builds are ad hoc signed, not Developer ID signed or notarized. Allow Local Network access when prompted.

Use the policy runner from this release's source revision for the updated startup and recovery controls. Back-down startup is opt-in through `control.lying_start`; configure and validate the supported pose before physical use. Buffered SONIC Tracking and assisted startup have synthetic/native validation, not a physical stability qualification. The console connects to existing services; these installers do not deploy the runner or policy bundles.

See the repository's console guide and verification record for configuration and commissioning coverage.
