## Bello Mini Console 0.2.5

### Changes

- Add assisted **Sit** from Stand: confirm support, bend with the left stick, confirm lowering onto the back, then confirm floor support before unloading motors and entering Idle.
- Share the bent pose, joint-speed limits and ankle-gain handling with Back-down startup. Push the stick up to reverse before lowering; return to nominal and confirm to restore ankle control and Stand.
- Pause assisted targets, gains and phase on console/network or gamepad loss. Retain motor control without damping or disabling; require neutral input after reconnecting. Explicit aborts and hardware/lease faults retain their damping response.
- Show velocity sticks only during Locomotion and one left-stick widget during assisted leg movement. Highlight up/down glyphs from the selected PlayStation, Xbox or Switch prompt set.
- Keep the controller card and abort control fixed across phases. Show confirmation prompts without competing progress text.

### Runner configuration

Upgrade the runner with the console. A configured `control.lying_start` enables both Back-down startup and Sit. The deployment preset uses:

```yaml
lying_start:
  hip_pitch_radians: -0.8
  knee_radians: 1.2
  straighten_speed_rps: 0.1
  ankle_engage_seconds: 2.0
```

The runner validates these joint angles against the selected policy and hardware limits before acquiring control. The operator supports the torso throughout the assisted sequence and explicitly confirms floor support. The runner does not lower the torso autonomously. Native loopback tests cover completion, reversal, reconnect neutrality, gamepad loss, aborts and hardware faults; physical lowering remains an operator commissioning step.

### Installers

| Platform | Asset suffix |
| --- | --- |
| macOS Apple silicon | `darwin-arm64.dmg` |
| Linux amd64 | `linux-x64.AppImage` |
| Linux arm64 | `linux-arm64.AppImage` |

`SHA256SUMS` contains each installer's checksum. macOS builds are ad hoc signed, not Developer ID signed or notarized. Allow Local Network access when prompted.

The console connects to existing services; these installers do not deploy the runner or policy bundles.
