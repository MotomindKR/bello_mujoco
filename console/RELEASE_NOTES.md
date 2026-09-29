## Bello Mini Console 0.2.1

### Changes

- Overview fits 16:9 fullscreen displays from 1280 x 720 upward without page scrolling.
- Compact stick indicators on shorter desktop screens keep Abort and fault recovery visible.
- Controller positions remain fixed across startup, motion, tracking and fault states; long fault details scroll within their own panel.
- Electron layout checks cover 1280 x 720, 1366 x 768, 1920 x 1080 and 2560 x 1440.

### Installers

| Platform | Asset suffix |
| --- | --- |
| macOS Intel | `darwin-x64.dmg` |
| macOS Apple silicon | `darwin-arm64.dmg` |
| Linux amd64 | `linux-x64.AppImage` |
| Linux arm64 | `linux-arm64.AppImage` |

`SHA256SUMS` contains the checksum for each installer. macOS builds are unsigned and not notarized.

Use the policy runner from this release's source revision for the new startup status and controls. Front-down startup is opt-in through `control.lying_start`; configure and validate the supported pose before physical use. Buffered SONIC Tracking and assisted startup have synthetic/native validation, not a physical stability qualification. The console connects to existing services; these installers do not deploy the runner or policy bundles.

See the repository's console guide and verification record for configuration and commissioning coverage.
