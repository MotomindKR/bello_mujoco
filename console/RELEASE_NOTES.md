## Bello Mini Console 0.2.6

### Changes

- Show **Startup blocked** when the runner is waiting for safe, fresh feedback instead of offering the Hold chord.
- Keep the operator service online and Disarmed while IMU samples are missing, telemetry is stale, or startup feedback violates joint or tilt limits. Retry readiness checks without exhausting systemd restart attempts.
- Require released controller inputs and a fresh Back+Start hold after startup readiness recovers. No motor lease or commands are issued while startup is blocked; joint limits remain enforced.
- Allow the supported torso to tilt during assisted Sit lowering. Upright checks still apply before returning to Stand; hardware faults and explicit aborts retain their damping response.

### Runner update

Update the runner alongside the console to get startup recovery. Select **Back-down** for a supported lying start, or **Crane** for an upright suspended start. The shared bent pose remains hip pitch -0.8 rad / knee 1.2 rad. Policy bundles and hardware calibration are unchanged.

### Installers

| Platform | Asset suffix |
| --- | --- |
| macOS Apple silicon | `darwin-arm64.dmg` |
| Linux amd64 | `linux-x64.AppImage` |
| Linux arm64 | `linux-arm64.AppImage` |

`SHA256SUMS` contains each installer's checksum. macOS builds are ad hoc signed, not Developer ID signed or notarized. Allow Local Network access when prompted.

The console connects to existing services; these installers do not deploy the runner or policy bundles.
