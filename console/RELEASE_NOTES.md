## Bello Mini Console 0.2.7

### Changes

- Release the left or right controller bumper to switch console tabs. Pressing both remains the one-second abort or recovery chord and does not switch tabs.
- Show Stand as the active intermediate state without offering it as a selectable mode. Mark the active mode and selected next mode separately.
- Keep named motions visible in a horizontal list. D-pad up/down or a click selects a motion; a single fresh one-second A hold starts it.
- After fault cleanup is ready, release inputs and hold A for one second to clear the fault. The on-screen Clear fault button remains available.

### Runner update

Update the policy runner alongside the console. This release changes the operator protocol and mode selection behavior. The runner still owns control transitions, hardware recovery and the 250 ms operator-stream lease.

### Installers

| Platform | Asset suffix |
| --- | --- |
| macOS Apple silicon | `darwin-arm64.dmg` |
| Linux amd64 | `linux-x64.AppImage` |
| Linux arm64 | `linux-arm64.AppImage` |

`SHA256SUMS` contains each installer's checksum. macOS builds are ad hoc signed, not Developer ID signed or notarized. Allow Local Network access when prompted.

The console connects to existing services; these installers do not deploy the runner or policy bundles.
