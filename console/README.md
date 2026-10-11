# Bello Mini console

Electron desktop console with console-owned gamepad bindings, runner-owned control, and direct read-only
hardwared, estimator, and XRoboToolkit tracking streams. Uses bundled Pretendard and Mini GLBs, English
labels, and Light/Dark/System appearance settings (System by default).

See [setup and controller behavior](../docs/policy_console.md) and the
[verification record](../docs/policy_console_verification.md).

```bash
npm ci
npm start
npm test
```

## Code organization

- `main.cjs` owns service connections, controller polling, and the validated IPC boundary.
- `controller.cjs` maps local gamepad bindings into semantic operator actions.
  On Linux, Switch controllers use SDL's event-device driver so ZL/ZR report
  press and release reliably.
- `snapshot.cjs` derives display freshness from immutable copies of received packets.
  Display streams do not gate robot control.
- `src/main.jsx` composes tabs and shared state. `Header`, `Settings`, `RobotView`,
  and `FaultStatus` own their respective UI and resource lifecycles.
- `src/OperatorCard.jsx` presents runner state; all control transitions remain in
  `rust/bello-policy/src/operator.rs` and `control.rs`.

Run `npm run format` before committing and `npm run format:check` to check style.
`npm run test:unit` covers pure controller, settings, tracking, and freshness logic;
`npm test` also builds the renderer and runs the service-backed Electron smoke test.

## Packaging

The Tracking indicator is yellow while its server is connected but body/hand
samples are missing or stale, and green when fresh body or hand samples arrive.
Before the first headset packet, the server's empty stream is shown as Waiting.

Build on the target architecture with `npm run make:mac` (DMG) or
`npm run make:linux` (DEB and RPM). Linux packaging requires `dpkg-deb`,
`fakeroot`, and `rpmbuild`.
Forge rebuilds the SDL controller module for Electron on each native runner.
The Bello face icon source is in [assets](assets/README.md).
Install the DEB or RPM through the distribution package manager to add the
desktop launcher and icon.

In **Settings > Console updates**, check the latest public GitHub release. An
installed Linux DEB or RPM can download its matching newer package, verify it
against the release `SHA256SUMS`, request system authorization through `pkexec`,
install it with `dpkg` or `rpm`, and restart. Disarm the robot before installing.
Updates from an unpacked build, unsupported Linux distributions, and macOS
remain manual through **Open releases**. The repository must be public for
in-app release checks to work.

CI builds macOS for Apple silicon (arm64) and Linux for x64 (amd64) and arm64.
Intel Macs are not supported. It runs for matching
pull requests, manual dispatch, and pushed `console-v*` tags. Set the version
in `package.json` and `package-lock.json`, update `RELEASE_NOTES.md`, commit the
release, then tag it:

```bash
git tag console-v0.3.3
git push origin console-v0.3.3
```

The tag must exactly match `console-v` plus the package version. After all three
packaged smoke tests pass, CI publishes the DMG, DEBs, RPMs and `SHA256SUMS` to a
GitHub Release. The `console-darwin-arm64` / `console-linux-{x64,arm64}`
Actions artifacts remain available as well. macOS bundles use an ad-hoc signature
to bind their bundle identity and privacy description. They are not Developer ID
signed or notarized.
On macOS, allow Bello Mini Console under System Settings > Privacy & Security >
Local Network. If a service reports `EHOSTUNREACH` while terminal connections
to the same host and port succeed, check this permission and relaunch the app.
Mac packaging derives the main executable's UUID from the app archive and stock
Electron UUID before signing. This prevents Local Network permissions from
colliding with stock Electron or other apps using the same runtime. If only
Electron appears in that list, replace the older bundle with a current build
with its own executable UUID and the `kr.motomind.bello-mini-console` signature.
Ad hoc signing is for local builds; Apple recommends an Apple-issued signing
identity for reliable permission tracking across distributed updates. A passing
localhost smoke test does not verify macOS LAN permission: launch from Finder
and verify live hardware telemetry after accepting the Local Network prompt.
Private asset checkouts require `BELLO_STACK_DEPLOY_KEY` and
`PICO_SMPLX_DEPLOY_KEY`. CI tests the packaged Mac app and the contents extracted
from each DEB, including loading the native SDL module, and compares the RPM
launcher and icon with their source files.

Set `BELLO_CONSOLE_CONFIG` to an absolute JSON configuration path to override
`config.json`. The console never acquires a hardware lease or writes motor targets.
