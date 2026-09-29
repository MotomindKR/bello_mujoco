## Bello Mini Console 0.2.0

### Changes

- Buffered SONIC Tracking mode, with reference readiness and return to Stand on reference loss.
- Crane and optional Front-down startup: bent-leg Hold with free ankles, explicit balance confirmation, stick-paced straightening, and gradual ankle engagement before Stand.
- Startup straightening respects joint speed and motor step limits; releasing the stick pauses movement.
- Compact controller guidance, fixed card geometry, motion selection and upper-body teleop controls.
- Direct hardware, estimator and XRoboToolkit views, power and temperature indicators, persistent settings, and Light/Dark/System appearance.
- Improved native controller, window resizing, fault recovery and software-renderer behavior.

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
