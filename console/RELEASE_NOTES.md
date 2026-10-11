## Bello Mini Console 0.2.8

### Changes

- Show named motions as a vertical scrolling list that fills the available space above the abort control.
- Keep joystick dials and values fully visible in Sit and assisted standing layouts.
- Show camera-aligned red X, green Y, and blue Z axes in both 3D views.
- Keep Settings actions visible while scrolling connection and update options.
- Install the Linux launcher and icon with DEB and RPM packages.
- Check for console updates in Settings; installed Linux packages can verify and install newer public releases after disarming the robot and granting system authorization.
- Show Stand as the active intermediate state outside the selectable mode row, even when an older runner advertises Stand. Reject clicks that attempt to select Stand.
- Center the tab choices with left and right controller bumper hints immediately beside them.
- Press a bumper to change tabs once; hold both triggers for one second to abort or recover from Fault.
- Require input protocol version 2 in both the console and runner, so mixed versions cannot use the old bumper abort mapping.

### Runner update

Update the runner with this release before operating the console. The runner uses L2 + R2 for abort and excludes Stand from gamepad selection. An older runner appears unavailable and receives only neutral samples.

### Installers

| Platform | Asset suffix |
| --- | --- |
| macOS Apple silicon | `darwin-arm64.dmg` |
| Linux amd64 | `linux-x64.deb`, `linux-x64.rpm` |
| Linux arm64 | `linux-arm64.deb`, `linux-arm64.rpm` |

`SHA256SUMS` contains each installer's checksum. macOS builds are ad hoc signed, not Developer ID signed or notarized. Allow Local Network access when prompted.

The console connects to existing services; these installers do not deploy the runner or policy bundles.
