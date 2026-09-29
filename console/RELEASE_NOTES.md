## Bello Mini Console 0.2.2

### Changes

- Fix macOS Local Network access with an app-specific executable UUID, bundle signature and permission description.
- Keep both the status bar and tabs visible while scrolling; restore horizontal inset in the tracking view.
- Match the AppImage desktop and window identity to the Bello face icon.
- Show yellow tracking status while waiting for headset data or when samples are stale, and green for live tracking.
- Treat the tracking server's initial empty stream as Waiting instead of a packet-age error.

### Installers

| Platform | Asset suffix |
| --- | --- |
| macOS Intel | `darwin-x64.dmg` |
| macOS Apple silicon | `darwin-arm64.dmg` |
| Linux amd64 | `linux-x64.AppImage` |
| Linux arm64 | `linux-arm64.AppImage` |

`SHA256SUMS` contains the checksum for each installer. macOS builds are ad hoc signed, not Developer ID signed or notarized. Allow Local Network access when prompted.

Use the policy runner from this release's source revision for the new startup status and controls. Front-down startup is opt-in through `control.lying_start`; configure and validate the supported pose before physical use. Buffered SONIC Tracking and assisted startup have synthetic/native validation, not a physical stability qualification. The console connects to existing services; these installers do not deploy the runner or policy bundles.

See the repository's console guide and verification record for configuration and commissioning coverage.
