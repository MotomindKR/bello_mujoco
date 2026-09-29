# Bello Mini console

Electron desktop console with runner-owned gamepad control and direct read-only
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

Build on the target architecture with `npm run make:mac` (DMG) or
`npm run make:linux` (AppImage). Linux packaging requires `squashfs-tools`.
Forge rebuilds the SDL controller module for Electron on each native runner.
The Bello face icon source is in [assets](assets/README.md).

CI builds macOS and Linux for both x64 (amd64) and arm64. It runs for matching
pull requests, manual dispatch, and pushed `console-v*` tags. Set the version
in `package.json` and `package-lock.json`, update `RELEASE_NOTES.md`, commit the
release, then tag it:

```bash
git tag console-v0.2.0
git push origin console-v0.2.0
```

The tag must exactly match `console-v` plus the package version. After all four
packaged smoke tests pass, CI publishes the DMGs, AppImages and `SHA256SUMS` to a
GitHub Release. The `console-darwin-{x64,arm64}` / `console-linux-{x64,arm64}`
Actions artifacts remain available as well. These are unsigned builds.
Private asset checkouts require `BELLO_STACK_DEPLOY_KEY` and
`PICO_SMPLX_DEPLOY_KEY`. CI tests the packaged Mac app and the contents extracted
from each AppImage, including loading the native SDL module.

Set `BELLO_CONSOLE_CONFIG` to an absolute JSON configuration path to override
`config.json`. The console never acquires a hardware lease or writes motor targets.
