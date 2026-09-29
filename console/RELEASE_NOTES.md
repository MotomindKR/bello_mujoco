## Bello Mini Console 0.2.4

### Changes

- Show **Returning to neutral** while SONIC brings its reference back to the nominal standing pose; hide the Stand prompt during recovery.
- Keep SONIC in control through reference interpolation, preserving its observation history, alignment and gains. Use a separate short handover to the standing policy after settling.
- Use the same neutral return for operator requests and console/Pico reference loss. Bound settling time; keep direct Stand recovery for inference failures and explicit damping for abort/hardware faults.
- Preserve standing-policy history at the end of the handover.

### Runner configuration

Upgrade the runner to this release's source revision and add these required fields under `tracking`:

```yaml
neutral_return_seconds: 1.0
neutral_joint_tolerance_radians: 0.15
neutral_settle_timeout_seconds: 2.0
standing_transition_seconds: 0.2
```

`transition_seconds` remains the entry duration. The new return path has synthetic/native lifecycle coverage; physical tracking quality requires commissioning.

### Installers

| Platform | Asset suffix |
| --- | --- |
| macOS Apple silicon | `darwin-arm64.dmg` |
| Linux amd64 | `linux-x64.AppImage` |
| Linux arm64 | `linux-arm64.AppImage` |

`SHA256SUMS` contains each installer's checksum. macOS builds are ad hoc signed, not Developer ID signed or notarized. Allow Local Network access when prompted.

The console connects to existing services; these installers do not deploy the runner or policy bundles.
